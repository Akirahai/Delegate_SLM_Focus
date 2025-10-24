# experiments/router_experiment.py
"""
Router Experiment: GPT-4o + Qwen tool (with token tracking)
"""
import time
import pandas as pd
import asyncio
import json
from dataclasses import dataclass, asdict
from agents import Runner

from experiments.utils import check_answer, extract_answer

@dataclass
class ProblemResult:
    problem_id: str
    subject: str
    question: str
    ground_truth: str
    prediction: str
    is_correct: bool
    latency_total: float
    latency_llm: float
    latency_slm: float
    tool_calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    slm_input_tokens: int = 0
    slm_output_tokens: int = 0


class RouterTracker:
    """Track metrics for router experiment"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.current_slm_time = 0.0
        self.current_tool_calls = []
        self.current_slm_input_tokens = 0
        self.current_slm_output_tokens = 0
    
    def log_tool_call(self, query, response, latency, input_tokens, output_tokens):
        self.current_tool_calls.append({
            'query': query,
            'response': response[:100],
            'latency': latency,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        })
        self.current_slm_time += latency
        self.current_slm_input_tokens += input_tokens
        self.current_slm_output_tokens += output_tokens

# Global tracker
tracker = RouterTracker()


async def run_router_experiment(test_df: pd.DataFrame, output_file: str, max_tokens: int):
    """Run router experiment with token tracking"""
    
    # Import agent (will be used with tracking)
    from experiments.router_agent import agent
    
    print(f"Running Router on {len(test_df)} problems (max_tokens={max_tokens})")
    
    results = []
    total_latency = 0.0
    total_llm_latency = 0.0
    total_slm_latency = 0.0
    total_tool_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_slm_input_tokens = 0
    total_slm_output_tokens = 0
    
    for idx, row in test_df.iterrows():
        print(f"[{idx+1}/{len(test_df)}] Processing...", end=' ')
        
        # Reset tracker for this problem
        tracker.reset()
        
        try:
            t_start = time.time()
            result = await Runner.run(agent, row["problem"], max_turns=15)
            t_end = time.time()
            
            prediction = result.final_output
            is_correct = check_answer(prediction, row["answer"])
            
            latency_total = t_end - t_start
            latency_slm = tracker.current_slm_time
            latency_llm = latency_total - latency_slm
            
            # Get token counts from result
            # Note: agents library tracks usage in result.usage
            input_tokens = getattr(result, 'input_tokens', 0)
            output_tokens = getattr(result, 'output_tokens', 0)
            
            # If not available, estimate from response
            if input_tokens == 0:
                # Rough estimate: ~4 chars per token
                input_tokens = len(row["problem"]) // 4
                output_tokens = len(prediction) // 4
            
            total_latency += latency_total
            total_llm_latency += latency_llm
            total_slm_latency += latency_slm
            total_tool_calls += len(tracker.current_tool_calls)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_slm_input_tokens += tracker.current_slm_input_tokens
            total_slm_output_tokens += tracker.current_slm_output_tokens
            
            problem_result = ProblemResult(
                problem_id=row.get("problem_id", f"prob_{idx}"),
                subject=row["subject"],
                question=row["problem"],
                ground_truth=str(row["answer"]),
                prediction=prediction,
                is_correct=is_correct,
                latency_total=latency_total,
                latency_llm=latency_llm,
                latency_slm=latency_slm,
                tool_calls=len(tracker.current_tool_calls),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                slm_input_tokens=tracker.current_slm_input_tokens,
                slm_output_tokens=tracker.current_slm_output_tokens
            )
            
            results.append(problem_result)
            
            status = "✓" if is_correct else "✗"
            print(f"{status} | {latency_total:.2f}s | tools={len(tracker.current_tool_calls)} | {input_tokens}→{output_tokens} tokens")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
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
        'avg_llm_latency': total_llm_latency / n_total if n_total else 0,
        'avg_slm_latency': total_slm_latency / n_total if n_total else 0,
        'total_slm_latency': total_slm_latency,
        'avg_tool_calls': total_tool_calls / n_total if n_total else 0,
        'total_tool_calls': total_tool_calls,
        'avg_input_tokens': total_input_tokens / n_total if n_total else 0,
        'avg_output_tokens': total_output_tokens / n_total if n_total else 0,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'avg_slm_input_tokens': total_slm_input_tokens / n_total if n_total else 0,
        'avg_slm_output_tokens': total_slm_output_tokens / n_total if n_total else 0,
        'total_slm_input_tokens': total_slm_input_tokens,
        'total_slm_output_tokens': total_slm_output_tokens
    }
    
    # Save results
    output = {
        'summary': summary,
        'results': [asdict(r) for r in results]
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Router Results: {n_correct}/{n_total} = {accuracy:.2%}")
    print(f"Avg Latency: {summary['avg_latency']:.3f}s (LLM: {summary['avg_llm_latency']:.3f}s, SLM: {summary['avg_slm_latency']:.3f}s)")
    print(f"Avg Tool Calls: {summary['avg_tool_calls']:.2f}")
    print(f"Avg Tokens: {summary['avg_input_tokens']:.1f} → {summary['avg_output_tokens']:.1f}")
    print(f"{'='*60}")
    
    return summary