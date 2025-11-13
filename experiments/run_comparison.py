# experiments/run_comparison.py
"""
Unified comparison runner for GSM8K experiments
Runs LLM baseline, Router, and SLM on the SAME samples

Usage:
    python run_comparison.py --gpus 0 1 --samples 10 --seed 123
    python run_comparison.py --gpus 0 1 --samples 200 --seed 123
    python run_comparison.py --gpus 0 1 --samples 10 --seed 123 --skip-llm --skip-router --model Qwen/Qwen2.5-Math-1.5B-Instruct --data math_500 --max-tokens 1024
"""

import asyncio
import argparse
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

sys.path.append(str(Path(__file__).parent.parent))

from tools.gsm8k_loader import load_gsm8k_as_df
from tools.data_loader import read_data
import os

async def main():
    parser = argparse.ArgumentParser(description='Run GSM8K comparison experiments')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of gpus to use')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--skip-llm', action='store_true', help='Skip LLM baseline')
    parser.add_argument('--skip-router', action='store_true', help='Skip Router baseline')
    parser.add_argument('--skip-slm', action='store_true', help='Skip SLM baseline')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct", help='SLM Model ID')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens')
    parser.add_argument('--data', type=str, default='gsm8k', help='Dataset to use (gsm8k, math_500)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (optional)')

    args = parser.parse_args()
    
    GPU_list = ','.join(map(str, args.gpus))
    
    
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    # Remove WORLD_SIZE to avoid distributed training issues
    # os.environ["WORLD_SIZE"] = "1"  # This was causing the error
    print(f"Using GPU: {GPU_list}")


    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = args.model.split('/')[-1]
    if args.output_dir is None:
        output_dir = Path(f"results_{model_name}_{args.samples}samples_{args.data}_path")
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"GSM8K COMPARISON EXPERIMENT")
    print(f"Samples: {args.samples} | Seed: {args.seed} | Max Tokens: {args.max_tokens}")
    print("="*80)
    
    # Load the SAME samples for all experiments
    print(f"\n📊 Loading {args.samples} {args.data} samples (seed={args.seed})...")
    # test_df = load_gsm8k_as_df(
    #     split='test',
    #     n_samples=args.samples,
    #     random_seed=args.seed
    # )
    test_df = read_data(args.data, n_samples=args.samples, random_seed=args.seed)


    # Save the sampled dataset for reference
    sample_file = output_dir / "samples.csv"
    test_df.to_csv(sample_file, index=False)
    print(f"✅ Saved samples to {sample_file}")
    
    # Import experiment modules
    from experiments.llm_experiment import run_llm_experiment
    from experiments.router_experiment import run_router_experiment
    from experiments.slm_experiment import run_slm_experiment
    
    results = {}


    # 1. LLM Baseline (GPT-4o)
    if not args.skip_llm:
        print("\n" + "="*80)
        print("EXPERIMENT 1: LLM BASELINE (Gemini alone)")
        print("="*80)
        llm_file = output_dir / "results_llm.json"
        results['llm'] = await run_llm_experiment(
            test_df.copy(), 
            str(llm_file),
            max_tokens=args.max_tokens
        )
    
    # 2. Router (GPT-4o + Qwen tool)
    if not args.skip_router:
        print("\n" + "="*80)
        print("EXPERIMENT 2: ROUTER (GPT-4o + Qwen 2.5 1B Tool)")
        print("="*80)
        router_file = output_dir / "results_router.json"
        results['router'] = await run_router_experiment(
            test_df.copy(),
            str(router_file)
        )
    
    # 3. SLM Baseline (Qwen alone) - Optional
    if not args.skip_slm:
        print("\n" + "="*80)
        print("EXPERIMENT 3: SLM BASELINE (Qwen 2.5 1B alone)")
        print("="*80)
        slm_file = output_dir / "results_slm.json"
        results['slm'] = await run_slm_experiment(
            test_df.copy(),
            str(slm_file),
            max_tokens=args.max_tokens,
            model_id=args.model
        )

    # Generate comparison report
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    comparison = {
        'metadata': {
            'samples': args.samples,
            'seed': args.seed,
            'max_tokens': args.max_tokens,
            'timestamp': timestamp
        },
        'results': {}
    }
    
    # Print comparison table
    print("\nComparison of Results:\n")
    print(f"\n{'Metric':<25}", end='')
    if not args.skip_llm:
        print(f" {'LLM':<15}", end='')
    if not args.skip_router:
        print(f" {'Router':<15}", end='')
    if not args.skip_slm:
        print(f" {'SLM':<15}")
    print("-" * 80)
    
    metrics = [
        ('Accuracy', 'accuracy', '.2%'),
        ('Avg Latency (s)', 'avg_latency', '.3f'),
        ('Total Latency (s)', 'total_latency', '.2f'),
        ('Avg Input Tokens', 'avg_input_tokens', '.1f'),
        ('Avg Output Tokens', 'avg_output_tokens', '.1f'),
        ('Total Input Tokens', 'total_input_tokens', 'd'),
        ('Total Output Tokens', 'total_output_tokens', 'd'),
    ]
    
    for label, key, fmt in metrics:
        print(f"{label:<25}", end='')
        for exp in ['llm', 'router', 'slm']:
            if exp == 'llm' and args.skip_llm:
                continue
            if exp == 'router' and args.skip_router:
                continue
            if exp == 'slm' and args.skip_slm:
                continue
            if exp in results:
                value = results[exp].get(key, 0)
                # Format value first, then apply padding
                if 'Accuracy' in label:
                    formatted = f"{value*100:.2f}%"
                elif 'Latency' in label and 'Total' not in label:
                    formatted = f"{value:.3f}"
                elif 'Latency' in label and 'Total' in label:
                    formatted = f"{value:.2f}"
                elif 'Tokens' in label and 'Total' in label:
                    formatted = f"{int(value):,}"
                else:
                    formatted = f"{value:.1f}"
                print(f"{formatted:<15}", end='')
                comparison['results'].setdefault(exp, {})[key] = value
        print()
    
    # Save comparison
    comparison_file = output_dir / "comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✅ All results saved to: {output_dir}")
    print(f"📊 Comparison summary: {comparison_file}")
    
    return comparison


if __name__ == "__main__":
    asyncio.run(main())