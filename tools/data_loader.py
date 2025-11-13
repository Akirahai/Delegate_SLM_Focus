import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from datasets import load_dataset

# Get project root but don't change working directory
PROJECT_ROOT = Path(__file__).parent.parent


def remove_calc_annotations(text):
    return re.sub(r"<<.*?>>", "", text)

def extract_gsm8k_answer(answer_text: str) -> str:
    """
    Extract the final numeric answer from GSM8K answer format.
    GSM8K answers end with #### followed by the numeric answer.
    
    Example:
        "She has 5 apples. #### 5" -> "5"
    """
    # GSM8K format: answer is after ####
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
    if match:
        # Remove commas from numbers like "1,000"
        return match.group(1).replace(',', '')
    
    # Fallback: extract last number
    numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
    return numbers[-1] if numbers else ""

def read_gsm8k(n_samples=None, random_seed=42):
    dataset = load_dataset("gsm8k", "main")
    
    test_input = dataset["test"]["question"]
    test_output_raw = dataset["test"]["answer"]
    test_output_clean = [extract_gsm8k_answer(ans) for ans in test_output_raw]
    problem_ids = [f"prob_{i}" for i in range(len(test_input))]
    # Convert to pandas
    df = pd.DataFrame({
        "problem_id": problem_ids,
        "problem": test_input,
        "solution": test_output_raw,
        "answer": test_output_clean
    })
    
    df["subject"] = "Arithmetic"
    
    # Optional sampling
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    
    print(f"Loaded {len(df)} problems from GSM8K (test split)")
    return df


def read_math_500(train = False, n_samples=None, random_seed=42):
    dataset = load_dataset("HuggingFaceH4/MATH-500")

    test_input = dataset.data['test']['problem'].to_pylist()
    test_output_sol = dataset.data['test']['solution'].to_pylist()
    test_output_ans = dataset.data['test']['answer'].to_pylist()
    test_subject  = dataset.data['test']['subject'].to_pylist()
    # Construct problem_ids:
    problem_ids = [f"prob_{i}" for i in range(len(test_input))]
    
    # Convert to pandas
    df = pd.DataFrame({
        "problem_id": problem_ids,
        "problem": test_input,
        "solution": test_output_sol,
        "answer": test_output_ans,
        "subject": test_subject

    })
    
    
    # Optional sampling
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    
    print(f"Loaded {len(df)} problems from GSM8K (test split)")
    return df



def read_data(task, n_samples=None, random_seed=42):
    if task == 'gsm8k':
        return read_gsm8k(n_samples=n_samples, random_seed=random_seed)
    elif task == 'math_500':
        return read_math_500(n_samples=n_samples, random_seed=random_seed)
    