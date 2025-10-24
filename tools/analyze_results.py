# tools/analyze_results.py
"""
Analyze and visualize comparison results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys


def load_comparison(folder: Path):
    """Load comparison results from folder"""
    comparison_file = folder / "comparison.json"
    if not comparison_file.exists():
        print(f"Error: {comparison_file} not found")
        sys.exit(1)
    
    with open(comparison_file) as f:
        data = json.load(f)
    
    # Load individual results
    results = {}
    for exp in ['llm', 'router', 'slm']:
        result_file = folder / f"results_{exp}.json"
        if result_file.exists():
            with open(result_file) as f:
                results[exp] = json.load(f)
    
    return data, results


def create_comparison_table(data):
    """Create formatted comparison table"""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    results = data['results']
    experiments = list(results.keys())
    
    # Create DataFrame
    metrics = {
        'Accuracy (%)': 'accuracy',
        'Avg Latency (s)': 'avg_latency',
        'Total Latency (s)': 'total_latency',
        'Avg Input Tokens': 'avg_input_tokens',
        'Avg Output Tokens': 'avg_output_tokens',
        'Total Tokens': None  # Computed
    }
    
    rows = []
    for metric_name, metric_key in metrics.items():
        row = {'Metric': metric_name}
        for exp in experiments:
            if metric_key:
                val = results[exp].get(metric_key, 0)
                if 'Accuracy' in metric_name:
                    row[exp.upper()] = f"{val*100:.2f}%"
                elif 'Latency' in metric_name:
                    row[exp.upper()] = f"{val:.3f}"
                else:
                    row[exp.upper()] = f"{val:.1f}"
            else:  # Total tokens
                inp = results[exp].get('total_input_tokens', 0)
                out = results[exp].get('total_output_tokens', 0)
                row[exp.upper()] = f"{inp + out:,}"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    
    # Router-specific metrics
    if 'router' in results:
        print(f"\nRouter-Specific Metrics:")
        print(f"  Avg Tool Calls: {results['router'].get('avg_tool_calls', 0):.2f}")
        print(f"  Total Tool Calls: {results['router'].get('total_tool_calls', 0)}")
        print(f"  Avg SLM Latency: {results['router'].get('avg_slm_latency', 0):.3f}s")
    
    return df


def plot_metrics(data, output_folder: Path):
    """Create visualization plots"""
    results = data['results']
    experiments = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GSM8K Comparison Results', fontsize=16, fontweight='bold')
    
    # 1. Accuracy
    ax = axes[0, 0]
    accuracies = [results[exp]['accuracy'] * 100 for exp in experiments]
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(experiments)]
    bars = ax.bar(experiments, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. Latency
    ax = axes[0, 1]
    latencies = [results[exp]['avg_latency'] for exp in experiments]
    bars = ax.bar(experiments, latencies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Average Latency (s)', fontsize=11)
    ax.set_title('Latency Comparison', fontsize=12, fontweight='bold')
    for bar, lat in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{lat:.3f}s', ha='center', va='bottom', fontsize=10)
    
    # 3. Token Usage
    ax = axes[1, 0]
    x = range(len(experiments))
    width = 0.35
    input_tokens = [results[exp]['avg_input_tokens'] for exp in experiments]
    output_tokens = [results[exp]['avg_output_tokens'] for exp in experiments]
    
    ax.bar([i - width/2 for i in x], input_tokens, width, 
           label='Input', color='#3498db', alpha=0.7, edgecolor='black')
    ax.bar([i + width/2 for i in x], output_tokens, width,
           label='Output', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Average Tokens', fontsize=11)
    ax.set_title('Token Usage', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    
    # 4. Efficiency (Accuracy per second)
    ax = axes[1, 1]
    efficiency = [results[exp]['accuracy'] / results[exp]['avg_latency'] * 100 
                  for exp in experiments]
    bars = ax.bar(experiments, efficiency, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy per Second (%/s)', fontsize=11)
    ax.set_title('Efficiency (Accuracy / Latency)', fontsize=12, fontweight='bold')
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{eff:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    plot_file = output_folder / "comparison_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plots saved to: {plot_file}")
    plt.show()


def analyze_problem_difficulty(results_dict, output_folder: Path):
    """Analyze accuracy by problem difficulty"""
    dfs = []
    for exp, data in results_dict.items():
        if 'results' in data:
            df = pd.DataFrame(data['results'])
            df['experiment'] = exp.upper()
            dfs.append(df)
    
    if not dfs:
        return
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Problem-level comparison
    pivot = combined.pivot_table(
        index='problem_id',
        columns='experiment',
        values='is_correct',
        aggfunc='first'
    )
    
    print("\n" + "="*80)
    print("PROBLEM-LEVEL ANALYSIS")
    print("="*80)
    
    # Count agreement
    if len(pivot.columns) >= 2:
        col1, col2 = pivot.columns[0], pivot.columns[1]
        both_correct = ((pivot[col1] == True) & (pivot[col2] == True)).sum()
        only_first = ((pivot[col1] == True) & (pivot[col2] == False)).sum()
        only_second = ((pivot[col1] == False) & (pivot[col2] == True)).sum()
        both_wrong = ((pivot[col1] == False) & (pivot[col2] == False)).sum()
        
        print(f"\nAgreement between {col1} and {col2}:")
        print(f"  Both correct: {both_correct}")
        print(f"  Only {col1}: {only_first}")
        print(f"  Only {col2}: {only_second}")
        print(f"  Both wrong: {both_wrong}")
        print(f"  Agreement rate: {(both_correct + both_wrong) / len(pivot) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Analyze comparison results')
    parser.add_argument('folder', type=str, help='Results folder path')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    args = parser.parse_args()
    
    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder {folder} not found")
        sys.exit(1)
    
    # Load data
    print(f"Loading results from {folder}...")
    data, results = load_comparison(folder)
    
    # Print comparison table
    create_comparison_table(data)
    
    # Problem-level analysis
    analyze_problem_difficulty(results, folder)
    
    # Create plots
    if not args.no_plot:
        try:
            plot_metrics(data, folder)
        except Exception as e:
            print(f"\n⚠️  Could not create plots: {e}")
            print("Install matplotlib and seaborn: pip install matplotlib seaborn")


if __name__ == "__main__":
    main()