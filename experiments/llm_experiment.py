# experiments/llm_experiment.py
"""
LLM Baseline: Gemini experiment (with token tracking)
"""
# Import SetUp
import os
import time
import pandas as pd
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json
from tqdm import tqdm

# Google Cloud & AI Libraries
import google.genai as genai
from google.genai import types
from google.genai import errors as genai_errors

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Evaluation Task
from experiments.utils import check_answer, extract_answer
from experiments.utils import extracted_box, len_extract_boxed
from grader import math_equal

# Data class for results
from dataclasses import dataclass, asdict

# MODEL_NAME = "gemini-2.5-flash-lite"
# MODEL_NAME = "gemini-2.5-pro"
MODEL_NAME = "gemini-2.5-flash"
@dataclass
class ProblemResult:
    problem_id: str
    subject: str
    question: str
    ground_truth: str
    prediction: str
    extract_answer: str
    is_correct: bool
    latency_total: float
    reason: str
    input_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0


class GeminiModel:
    def __init__(self, model_name, max_output_tokens):
        self.model_name = model_name
        print(f"Conducting experiments with Gemini Model: {model_name}")
        self.max_output_tokens = max_output_tokens
        load_dotenv()
        print(f"Initializing Gemini Model: {model_name} with max_output_tokens={max_output_tokens}")

        self.SYSTEM_PROMPT = "You are an expert at solving math problems.\
        Solve this problem step by step. \
        Make sure that your final answer should be simplified (no units). \
        Provide your final answer in \\boxed{{answer}} format."
        

        self.ai_client = genai.Client(api_key=GEMINI_API_KEY, vertexai=False)
        print("Gemini API Client Initialized")


        self.CONFIG = types.GenerateContentConfig(
            system_instruction=self.SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=2048)
        )


        # Default agent for completion case without \boxed{}
        self.completion_agent_model = "gemini-2.5-flash-lite"
        self.Completion_CONFIG = types.GenerateContentConfig(
            system_instruction=self.SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )


    async def run(self, prompt: str):
        """Run inference and return (response, input_tokens, output_tokens)"""

        print("\nMax Output Tokens is", self.CONFIG.max_output_tokens)
        t_start = time.time()
        try:
            response = await self.ai_client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.CONFIG
            )
            print("First pass completion finished.")

        except genai_errors.ServerError as e:
            print(f"Server error during generation: {e}")
            err_msg = getattr(e, 'message', 'Unknown server error')
            return "", f"SERVER_ERROR_{err_msg}", 0.0, 0, 0, 0
        t_end = time.time()

        latency = t_end - t_start
        reason = "UNKNOWN_ERROR"
        if response.candidates and response.candidates[0].finish_reason is not None:
            reason = response.candidates[0].finish_reason.name

        if response.text is not None:
            prediction = response.text.strip()
            if reason != "SAFETY":
                 reason = "STOP"
        else:
            # Handle the case where response.text is None (blocked or error)
            prediction = "" # Set prediction to an empty string to avoid errors later

        usage_metadata = response.usage_metadata

        # Get input and output tokens
        input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
        thinking_tokens = getattr(usage_metadata, 'thoughts_token_count', 0)
        output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)

        # Handle None values
        if input_tokens is None:
            input_tokens = 0
        if output_tokens is None:
            output_tokens = 0
        if thinking_tokens is None:
            thinking_tokens = 0


        # Include additional generation if not correct format
        if ("boxed" in prediction and (len_extract_boxed(prediction) >= 1) ) or (prediction == "") :
            print("The completion finished successfully.")
            return prediction, reason, latency, input_tokens, thinking_tokens, output_tokens

        else:
            print("The completion got problems")
            print("Completion:", response)
            modified_question = prompt + prediction + "Provide your final answer in \\boxed{{answer}} format based previous solution."
            self.Completion_CONFIG.max_output_tokens = input_tokens + output_tokens + 1500
            print("The max_output_tokens for second pass is set to", self.Completion_CONFIG.max_output_tokens)
            try:
                response = await self.ai_client.aio.models.generate_content(
                    model=self.completion_agent_model,
                    contents=modified_question,
                    config=self.Completion_CONFIG
                )
            except genai_errors.ServerError as e:
                print(f"Server error during second pass generation: {e}")
                return prediction, f"SERVER_ERROR_{e.status_code}", latency, input_tokens, thinking_tokens, output_tokens
            print("Second pass completion finished.")
            reason = "UNKNOWN_ERROR"
            if response.candidates and response.candidates[0].finish_reason is not None:
                reason = response.candidates[0].finish_reason.name

            if response.text is not None:
                new_prediction = response.text.strip()
                if reason != "SAFETY":
                    reason = "STOP_RUN_2_TIME"
            else:
                # Handle the case where response.text is None (blocked or error)
                new_prediction = "" # Set prediction to an empty string to avoid errors later
            
            # Use the old usage_metadata for token counts
            # Add the new prediction
            prediction = prediction + "Provide your final answer in \\boxed{{answer}} format." + new_prediction


            return prediction, reason, latency, input_tokens, thinking_tokens, output_tokens






async def run_llm_experiment(test_df: pd.DataFrame, 
                             output_file: str, 
                             max_tokens: int, 
                             model_name: str = MODEL_NAME,
                             batch_size: int = 32):
    """Run baseline with Gemini and track tokens"""
    load_dotenv()
    gemini_model = GeminiModel(model_name=model_name, max_output_tokens=max_tokens)
    print(f"Running Gemini on {len(test_df)} problems (max_tokens={max_tokens})")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = json.load(f)
        results = [ProblemResult(**r) for r in data.get("results", [])]
        print(f"✅ Resuming from {len(results)} previously processed problems")
    else:
        results = []

    print(f"Running Gemini on {len(test_df)} problems in batches of {batch_size} (max_tokens={max_tokens})")

    total_input = 0
    total_thinking = 0
    total_output = 0
    total_latency = 0.0
    failed_questions = 0

    # Split into batches
    for start_idx in range(0, len(test_df), batch_size):
        end_idx = min(start_idx + batch_size, len(test_df))
        batch = test_df.iloc[start_idx:end_idx]

        print(f"\n🚀 Processing batch {start_idx // batch_size + 1} ({start_idx}–{end_idx-1})")

        batch_results = []

        for idx, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Batch {start_idx // batch_size + 1}"):

            # Skip already processed problems with no error
            problem_id = str(row.get("problem_id", f"prob_{idx}"))
            existing = next((r for r in results if r.problem_id == problem_id), None)

            if existing and existing.reason in  ["STOP", "STOP_RUN_2_TIME"]:
                print(f"Skipping already processed problem {problem_id} with {existing.reason}")
                total_input += existing.input_tokens
                total_output += existing.output_tokens
                total_thinking += existing.thinking_tokens
                total_latency += existing.latency_total
                print(f"⏩ Skipping already processed problem {problem_id}")
                continue

            if existing and existing.reason not in ["STOP", "STOP_RUN_2_TIME"]:
                print(f"Re-processing problem {problem_id} due to previous reason: {existing.reason}")
                # Remove previous entry
                results = [r for r in results if r.problem_id != problem_id]

            prompt = f"Problem: {row['problem']}"
            prediction, reason, latency = "", "ERROR_UNKNOWN", 0.0
            input_tokens = thinking_tokens = output_tokens = 0
            extracted_answer = ""
            is_correct = False

            # Run model safely
            try:
                prediction, reason, latency, input_tokens, thinking_tokens, output_tokens = await gemini_model.run(prompt)
            except Exception as e:
                print(f"⚠️ Exception in model run for {problem_id}: {e}")
                reason = f"RUNTIME_ERROR_{type(e).__name__}"
            
            extracted_answer = extracted_box(prediction)

            print(f"Reason for completion {problem_id}: {reason}")

            try:
                is_correct = math_equal(extracted_answer, row["answer"], timeout=True)
            except Exception:
                is_correct = False


            status = "✓" if is_correct else "✗"
            print(f"{status} | {latency:.2f}s | {input_tokens}→ {thinking_tokens} + {output_tokens} = {thinking_tokens + output_tokens} tokens")
            
            
            total_input += input_tokens
            total_output += output_tokens
            total_thinking += thinking_tokens
            total_latency += latency

            batch_result = ProblemResult(
                problem_id=problem_id,
                subject=row.get("subject", ""),
                question=row["problem"],
                ground_truth=str(row["answer"]),
                prediction=prediction,
                extract_answer=extracted_answer,
                is_correct=is_correct,
                latency_total=latency,
                reason=reason,
                input_tokens=input_tokens,
                thinking_tokens=thinking_tokens,
                output_tokens=output_tokens,
            )

            batch_results.append(batch_result)
            results.append(batch_result)
            if reason != "STOP":
                print(f"⚠️  Warning: [{problem_id}/{len(test_df)}] failed with reason: {reason}")
                failed_questions += 1

        # 🧾 Write partial batch results to disk after each batch
        n_correct = sum(r.is_correct for r in results)
        summary = {
            "accuracy": n_correct / len(results) if results else 0,
            "correct": n_correct,
            "total": len(results),
            "avg_latency": total_latency / len(results) if results else 0,
            "total_latency": total_latency,
            "avg_input_tokens": total_input / len(results) if results else 0,
            "avg_output_tokens": total_output / len(results) if results else 0,
            "avg_thinking_tokens": total_thinking / len(results) if results else 0,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_thinking_tokens": total_thinking,
            "failed_questions": failed_questions,
        }

        output_data = {
            "summary": summary,
            "results": [asdict(r) for r in results],
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Saved batch {start_idx // batch_size + 1} to {output_file}")

        # Sleep briefly to avoid rate-limiting
        await asyncio.sleep(2)



    print(f"\n{'='*60}")
    print(f"LLM Results: {n_correct}/{len(results)} = {summary['accuracy']:.2%}")
    print(f"Avg Latency: {summary['avg_latency']:.3f}s")
    print(f"Avg Tokens: {summary['avg_input_tokens']:.1f} → {summary['avg_output_tokens']:.1f}")
    print(f"{'='*60}")

    return summary
