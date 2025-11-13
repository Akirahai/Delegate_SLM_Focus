# experiments/router_experiment.py
"""
Router Experiment: Gemini + Qwen tool (with token tracking)
"""
import os
import time
import pandas as pd
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json
from tqdm import tqdm
load_dotenv()

# Data class for results
from dataclasses import dataclass, asdict

# Import Agents Framework
from agents import Runner

# Import agent
from experiments.router_agent import initialize_all_agents


# Evaluation Tasks
from experiments.utils import extracted_box, len_extract_boxed
from grader import math_equal


# Set model names
LLM_MODEL = "gemini/gemini-2.5-flash"
# ORCHESTRATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ORCHESTRATOR_MODEL = "gemini/gemini-2.5-flash-lite"
SLM_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"



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
    hand_off_model: str
    planning_input_tokens: int = 0
    planning_thinking_tokens: int = 0
    planning_output_tokens: int = 0
    input_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0

async def run_agent(agent, question: str):
    latency = 0.0
    result = None

    try:
        time_start = time.time()
        result = await Runner.run(agent, question, max_turns=2)
        time_end = time.time()
        latency = time_end - time_start
        
    except Exception as e:
        time_end = time.time()
        latency = time_end - time_start
        print(f"⚠️ Exception while running agent: {e}")
        # Return a minimal fallback result so callers can handle gracefully
        dict_result = {
            "latency": latency,
            "prediction": "",
            "hand_off_model": "N/A",
            "planning_input_tokens": 0,
            "planning_thinking_tokens": 0,
            "planning_output_tokens": 0,
            "input_tokens": 0,
            "thinking_tokens": 0,
            "output_tokens": 0,
            "reason": f"RUNTIME_ERROR_{type(e).__name__}",
        }
        return dict_result


    
    hand_off_model = result.new_items[-1].agent.name if result.new_items else "N/A"

    planning_response = result.raw_responses[0]
    planning_input_tokens = planning_response.usage.input_tokens
    planning_thinking_tokens = planning_response.usage.output_tokens_details.reasoning_tokens
    planning_output_tokens = planning_response.usage.output_tokens

    prediction = result.final_output

    if hand_off_model == 'LLM Math Experts':

        prediction_response = result.raw_responses[1]
        input_tokens = prediction_response.usage.input_tokens
        thinking_tokens = prediction_response.usage.output_tokens_details.reasoning_tokens
        output_tokens = prediction_response.usage.output_tokens

    elif hand_off_model == 'SLM Math Expert':

        prediction_response = result.raw_responses[1]
        input_tokens = prediction_response.usage.input_tokens 
        thinking_tokens = 0 # SLM does not have separate thinking tokens
        output_tokens = prediction_response.usage.output_tokens

    dict_result = {
        "latency": latency,
        "prediction": prediction,
        "hand_off_model": hand_off_model,
        "planning_input_tokens": planning_input_tokens,
        "planning_thinking_tokens": planning_thinking_tokens,
        "planning_output_tokens": planning_output_tokens,
        "input_tokens": input_tokens,
        "thinking_tokens": thinking_tokens,
        "output_tokens": output_tokens,
        "reason": "STOP",
    }

    return dict_result



async def run_router_experiment(test_df: pd.DataFrame, 
                             output_file: str, 
                             slm_model_name: str = SLM_MODEL,
                             llm_model_name: str = LLM_MODEL,
                             orchestrator_model_name: str = ORCHESTRATOR_MODEL,
                             batch_size: int = 32):
    """Run baseline with Gemini and track tokens"""
    orchestrator_model = initialize_all_agents(slm_model_name, llm_model_name, orchestrator_model_name)
    print(f"Running Gemini on {len(test_df)} problems")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = json.load(f)
        results = [ProblemResult(**r) for r in data.get("results", [])]
        print(f"✅ Resuming from {len(results)} previously processed problems")
    else:
        results = []

    print(f"Running Orchestrator Model on {len(test_df)} problems in batches of {batch_size}")

    total_planning_input = 0
    total_planning_thinking = 0
    total_planning_output = 0
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
            if "problem_id" not in row or pd.isna(row["problem_id"]):
                raise ValueError(f"❌ Missing 'problem_id' at row index {idx}")
            problem_id = str(row["problem_id"])
            
            existing = next((r for r in results if r.problem_id == problem_id), None)

            if existing and existing.reason in  ["STOP"]:
                print(f"Skipping already processed problem {problem_id} with {existing.reason}")

                total_planning_input += existing.planning_input_tokens
                total_planning_thinking += existing.planning_thinking_tokens
                total_planning_output += existing.planning_output_tokens
                total_input += existing.input_tokens
                total_output += existing.output_tokens
                total_thinking += existing.thinking_tokens
                total_latency += existing.latency_total
                print(f"⏩ Skipping already processed problem {problem_id}")
                continue

            if existing and existing.reason not in ["STOP"]:
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
                result = await run_agent(orchestrator_model, prompt)
                latency = result["latency"]
                prediction = result["prediction"]
                reason = result["reason"]
                
                input_tokens = result["input_tokens"]
                thinking_tokens = result["thinking_tokens"]
                output_tokens = result["output_tokens"]

                reason = "STOP"
            except Exception as e:
                print(f"⚠️ Exception in model run for {problem_id}: {e}")
                reason = f"RUNTIME_ERROR_{type(e).__name__}"
                result = {
                    "latency": latency,
                    "prediction": "",
                    "hand_off_model": "N/A",
                    "planning_input_tokens": 0,
                    "planning_thinking_tokens": 0,
                    "planning_output_tokens": 0,
                    "input_tokens": 0,
                    "thinking_tokens": 0,
                    "output_tokens": 0,
                    "reason": reason,
                }


            extracted_answer = extracted_box(prediction)
            if extracted_answer == "":
                extracted_answer = "NO_ANSWER_FOUND"
                reason = "NO_ANSWER_FOUND"


            print(f"Reason for completion {problem_id}: {reason}")

            try:
                is_correct = math_equal(extracted_answer, row["answer"], timeout=True)
            except Exception:
                is_correct = False


            status = "✓" if is_correct else "✗"
            print(f"{status} | {latency:.2f}s | {input_tokens}→ {thinking_tokens} + {output_tokens} = {thinking_tokens + output_tokens} tokens")
            
            
            total_planning_input += result["planning_input_tokens"]
            total_planning_thinking += result["planning_thinking_tokens"]
            total_planning_output += result["planning_output_tokens"]

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
                hand_off_model=result["hand_off_model"],
                planning_input_tokens=result["planning_input_tokens"],
                planning_thinking_tokens=result["planning_thinking_tokens"],
                planning_output_tokens=result["planning_output_tokens"],
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
            "avg_planning_input_tokens": total_planning_input / len(results) if results else 0,
            "avg_planning_output_tokens": total_planning_output / len(results) if results else 0,
            "avg_planning_thinking_tokens": total_planning_thinking / len(results) if results else 0,
            "total_planning_input_tokens": total_planning_input,
            "total_planning_output_tokens": total_planning_output,
            "total_planning_thinking_tokens": total_planning_thinking,
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
