# tools/gsm8k_loader.py
"""
Load and prepare GSM8K dataset for evaluations
"""
import re
from datasets import load_dataset
import pandas as pd

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

def load_gsm8k_as_df(split='test', n_samples=None, random_seed=42):
    """
    Load GSM8K dataset and return as pandas DataFrame.
    
    Args:
        split: 'train' or 'test'
        n_samples: Number of samples to load (None = all)
        random_seed: Seed for random sampling
        
    Returns:
        DataFrame with columns: problem, answer, solution
    """
    # Load dataset from HuggingFace
    ds = load_dataset("openai/gsm8k", "main", split=split)
    
    # Convert to pandas
    df = pd.DataFrame({
        'problem': ds['question'],
        'solution': ds['answer'],  # Full solution text
        'answer': [extract_gsm8k_answer(ans) for ans in ds['answer']]  # Just the number
    })
    
    # Add subject (all GSM8K are arithmetic/algebra)
    df['subject'] = 'Arithmetic'
    
    # Sample if requested
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    
    print(f"Loaded {len(df)} problems from GSM8K ({split} split)")
    return df

def prepare_gsm8k_csv(output_path='gsm8k_test.csv', n_samples=None):
    """
    Prepare GSM8K test set as CSV file compatible with your evaluation scripts.
    Creates a CSV with columns: problem, answer, subject
    """
    df = load_gsm8k_as_df(split='test', n_samples=n_samples)
    
    # Keep only columns needed for evaluation
    df = df[['problem', 'answer', 'subject']]
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} problems to {output_path}")
    return output_path

# CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare GSM8K dataset')
    parser.add_argument('--output', default='gsm8k_test.csv', help='Output CSV path')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples')
    parser.add_argument('--split', default='test', choices=['train', 'test'], 
                       help='Dataset split')
    args = parser.parse_args()
    
    prepare_gsm8k_csv(output_path=args.output, n_samples=args.samples)