# experiments/llm_experiment.py
"""
LLM Baseline: GPT-4o alone (with token tracking)
"""
import os
import time
import pandas as pd
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

from experiments.utils import check_answer, extract_answer

# Data class for results
from dataclasses import dataclass, asdict

@dataclass
class ProblemResult:
    problem_id: str
    subject: str
    question: str
    ground_truth: str
    prediction: str
    is_correct: bool
    latency_total: float
    input_tokens: int = 0
    output_tokens: int = 0


async def run_llm_experiment(test_df: pd.DataFrame, output_file: str, max_tokens: int):
    """Run baseline with GPT-4o and track tokens"""
    load_dotenv()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Running GPT-4o on {len(test_df)} problems (max_tokens={max_tokens})")

    results = []
    total_input = 0
    total_output = 0
    total_latency = 0.0

    for idx, row in test_df.iterrows():
        print(f"[{idx+1}/{len(test_df)}] Processing...", end=' ')

        prompt = f"""You are an expert at solving math problems.
Solve this problem step by step.
Provide your final answer in \\boxed{{answer}} format.

Problem: {row['problem']}"""

        t_start = time.time()
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0
        )
        t_end = time.time()

        prediction = resp.choices[0].message.content
        is_correct = check_answer(prediction, row["answer"])
        
        # Track tokens
        input_tokens = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
        latency = t_end - t_start
        
        total_input += input_tokens
        total_output += output_tokens
        total_latency += latency

        result = ProblemResult(
            problem_id=row.get("problem_id", f"prob_{idx}"),
            subject=row["subject"],
            question=row["problem"],
            ground_truth=str(row["answer"]),
            prediction=prediction,
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
    accuracy = n_correct / len(results) if results else 0
    avg_latency = total_latency / len(results) if results else 0
    
    summary = {
        'accuracy': accuracy,
        'correct': n_correct,
        'total': len(results),
        'avg_latency': avg_latency,
        'total_latency': total_latency,
        'avg_input_tokens': total_input / len(results) if results else 0,
        'avg_output_tokens': total_output / len(results) if results else 0,
        'total_input_tokens': total_input,
        'total_output_tokens': total_output
    }

    # Save results
    import json
    output = {
        'summary': summary,
        'results': [asdict(r) for r in results]
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"LLM Results: {n_correct}/{len(results)} = {accuracy:.2%}")
    print(f"Avg Latency: {avg_latency:.3f}s")
    print(f"Avg Tokens: {summary['avg_input_tokens']:.1f} → {summary['avg_output_tokens']:.1f}")
    print(f"{'='*60}")

    return summary