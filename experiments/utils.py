# experiments/utils.py
"""
Utility functions for MATH500 evaluation experiments
"""
import re
import json
from typing import Dict, List
from dataclasses import dataclass, asdict
import regex

# ---------------------------
# Data Structures
# ---------------------------
@dataclass
class ProblemResult:
    problem_id: str
    subject: str
    difficulty: str
    question: str
    ground_truth: str
    prediction: str
    extract_answer: str
    is_correct: bool
    latency_total: float
    latency_llm: float = 0.0
    latency_slm: float = 0.0
    tool_calls: int = 0
    tool_call_details: List[Dict] = None
    
    def __post_init__(self):
        if self.tool_call_details is None:
            self.tool_call_details = []
    
    def to_dict(self):
        return asdict(self)

# ---------------------------
# Answer Extraction (matches llm_test.py)
# ---------------------------
def extract_answer(text: str) -> str:
    """
    Extract numerical answer from model output.
    Matches the extraction logic from llm_test.py exactly.
    
    Args:
        text: Model output text
        
    Returns:
        Extracted answer as string (only digits)
    """
    # First try "the answer is X" pattern
    m = re.findall(r"(?:the answer is|answer is)\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m[-1]
    
    # Fallback: last number in text
    m = re.findall(r"\d+", text)
    if m:
        return m[-1]
    
    return ""

def extract_ground_truth(answer: str) -> str:
    """
    Extract ground truth answer from answer column.
    Matches llm_test.py logic.
    
    Args:
        answer: Content from 'answer' column
        
    Returns:
        Extracted answer as string (only digits)
    """
    gold_num = re.findall(r"\d+", str(answer))
    return gold_num[-1] if gold_num else ""



# ---------------------------
# New Answer Extraction with boxed format
# ---------------------------
def extracted_box(text: str) -> str:
    """
    Extract the last complete LaTeX expression inside \boxed{...},
    correctly handling nested braces and avoiding regex recursion.
    """
    last_box_start = text.rfind(r"\boxed{")
    if last_box_start == -1:
        return ""

    i = last_box_start + len(r"\boxed{")
    depth = 1
    content = []
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        if depth > 0:
            content.append(text[i])
        i += 1
    
    result = "".join(content).strip()

    # --- optional cleanup: remove redundant outer braces like {{8}} or {8} ---
    while result.startswith("{") and result.endswith("}"):
        inner = result[1:-1].strip()
        # stop if braces inside are unbalanced
        if inner.count("{") != inner.count("}"):
            break
        result = inner

    return result



def len_extract_boxed(text):
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    return len(matches[-1].strip()) if matches else 0

def check_answer(prediction: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth.
    
    Args:
        prediction: Model's full output text
        ground_truth: Content from 'answer' column
        
    Returns:
        True if answers match, False otherwise
    """
    pred = extract_answer(prediction)
    gold = extract_ground_truth(ground_truth)
    return pred == gold

# ---------------------------
# Metrics Tracker
# ---------------------------
class MetricsTracker:
    """Track tool usage metrics during evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracking for new experiment"""
        self.current_tool_calls = []
        self.current_slm_time = 0.0
    
    def start_problem(self):
        """Reset counters for new problem"""
        self.current_tool_calls = []
        self.current_slm_time = 0.0
    
    def log_tool_call(self, question: str, result: str, latency: float):
        """
        Log a single SLM tool call.
        
        Args:
            question: Question sent to SLM
            result: JSON result from SLM
            latency: Time taken by SLM
        """
        self.current_tool_calls.append({
            "question": question,
            "result": result,
            "latency": latency
        })
        self.current_slm_time += latency

# ---------------------------
# Results Saving
# ---------------------------
def save_results(results: List[ProblemResult], filepath: str, summary: Dict = None):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of ProblemResult objects
        filepath: Output file path
        summary: Optional summary statistics dict
    """
    output_data = {
        "detailed_results": [r.to_dict() for r in results]
    }
    
    if summary:
        output_data["summary"] = summary
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {filepath}")

# ---------------------------
# Summary Statistics
# ---------------------------
def calculate_summary(results: List[ProblemResult]) -> Dict:
    """
    Calculate summary statistics from results.
    
    Args:
        results: List of ProblemResult objects
        
    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}
    
    correct_count = sum(1 for r in results if r.is_correct)
    total = len(results)
    
    summary = {
        "total_problems": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total > 0 else 0,
        "avg_latency": sum(r.latency_total for r in results) / total if total > 0 else 0,
    }
    
    # Add router-specific metrics if available
    if any(r.tool_calls > 0 for r in results):
        summary["avg_llm_latency"] = sum(r.latency_llm for r in results) / total if total > 0 else 0
        summary["avg_slm_latency"] = sum(r.latency_slm for r in results) / total if total > 0 else 0
        summary["avg_tool_calls"] = sum(r.tool_calls for r in results) / total if total > 0 else 0
        summary["tool_usage_rate"] = sum(1 for r in results if r.tool_calls > 0) / total if total > 0 else 0
    
    return summary

def print_summary(summary: Dict, experiment_name: str = "Experiment"):
    """
    Print formatted summary statistics.
    
    Args:
        summary: Summary dictionary from calculate_summary()
        experiment_name: Name to display in header
    """
    print("\n" + "="*60)
    print(f"{experiment_name.upper()} RESULTS")
    print("="*60)
    print(f"Total problems: {summary['total_problems']}")
    print(f"Correct: {summary['correct']}")
    print(f"Accuracy: {summary['accuracy']*100:.2f}%")
    print(f"Average latency: {summary['avg_latency']:.2f}s")
    
    if "avg_tool_calls" in summary:
        print(f"Average LLM latency: {summary['avg_llm_latency']:.2f}s")
        print(f"Average SLM latency: {summary['avg_slm_latency']:.2f}s")
        print(f"Tool usage rate: {summary['tool_usage_rate']*100:.1f}%")
        print(f"Average tool calls: {summary['avg_tool_calls']:.2f}")