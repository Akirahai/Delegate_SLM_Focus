# experiments/run_experiment.py
"""
SLM Baseline: Qwen 2.5 alone (with token tracking)
"""
import time
import pandas as pd
import asyncio
import torch
import json
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.utils import check_answer, extract_answer
from experiments.utils import extract_boxed, len_extract_boxed
from grader import math_equal

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
    input_tokens: int = 0
    output_tokens: int = 0


class QwenAgent:
    def __init__(self, model_id="Qwen/Qwen2.5-Math-1.5B-Instruct", max_new_tokens=512):
        self.max_new_tokens = max_new_tokens
        print(f"Loading {model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        print("Model ready")

    async def run(self, prompt: str):
        """Run inference and return (response, input_tokens, output_tokens)"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = inputs["input_ids"].shape[1]
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Count only new tokens
        output_tokens = outputs.shape[1] - input_tokens
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Include additional generation if not correct format
        if "boxed" in response and (len_extract_boxed(response) >= 1):
            return response, input_tokens, output_tokens
        else:
            # Modify the question by appending " Therefore, the answer is" at the end
            print("The completion got problems")
            print("Completion:", response)
            modified_question = prompt + response + " Therefore, the answer is \\boxed{"
            new_inputs = self.tokenizer(modified_question, return_tensors="pt").to(self.model.device)
            
            with torch.inference_mode():
                new_outputs = self.model.generate(
                    **new_inputs,
                    max_new_tokens=32,  # Limit the max tokens for the second pass
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            # Add the new output tokens
            
            new_output_tokens = new_outputs.shape[1] - new_inputs["input_ids"].shape[1]
            total_output_tokens = output_tokens + new_output_tokens
            new_response = self.tokenizer.decode(new_outputs[0], skip_special_tokens=True)
            full_response = response + " Therefore, the answer is \\boxed{" + new_response

            return full_response, input_tokens, total_output_tokens



async def run_slm_experiment(test_df: pd.DataFrame, output_file: str, max_tokens: int):
    """Run SLM baseline with token tracking"""
    agent = QwenAgent(max_new_tokens=max_tokens)

    print(f"Running Qwen on {len(test_df)} problems (max_tokens={max_tokens})")

    results = []
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, row in test_df.iterrows():
        print(f"[{idx+1}/{len(test_df)}] Processing...", end=' ')

        # Simple prompt for SLM
        prompt = f"Solve this math problem step by step. Put your final answer in \\boxed{{}}.\n\nProblem: {row['problem']}"

        t_start = time.time()
        prediction, input_tokens, output_tokens = await agent.run(prompt)
        t_end = time.time()

        extracted_answer = extract_answer(prediction)

        is_correct = math_equal(extracted_answer, row["answer"], timeout=True)
        latency = t_end - t_start
        
        total_latency += latency
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        result = ProblemResult(
            problem_id=row.get("problem_id", f"prob_{idx}"),
            subject=row["subject"],
            question=row["problem"],
            ground_truth=str(row["answer"]),
            prediction=prediction,
            extract_answer=extracted_answer,
            is_correct=is_correct,
            latency_total=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        results.append(result)

        status = "✓" if is_correct else "✗"
        print(f"{status} | {latency:.2f}s | {input_tokens}→{output_tokens} tokens")

    # Calculate summary
    n_correct = sum(r.is_correct for r in results)
    n_total = len(results)
    accuracy = n_correct / n_total if n_total else 0

    summary = {
        'accuracy': accuracy,
        'correct': n_correct,
        'total': n_total,
        'avg_latency': total_latency / n_total if n_total else 0,
        'total_latency': total_latency,
        'avg_input_tokens': total_input_tokens / n_total if n_total else 0,
        'avg_output_tokens': total_output_tokens / n_total if n_total else 0,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens
    }

    # Save results
    output = {
        'summary': summary,
        'results': [asdict(r) for r in results]
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SLM Results: {n_correct}/{n_total} = {accuracy:.2%}")
    print(f"Avg Latency: {summary['avg_latency']:.3f}s")
    print(f"Avg Tokens: {summary['avg_input_tokens']:.1f} → {summary['avg_output_tokens']:.1f}")
    print(f"{'='*60}")

    return summary