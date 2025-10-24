# Delegate 

> A framework that prompts an LLM to reason and delegate appropriate sub-tasks to an SLM to save costs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ☄️ Overview

**Problem Statement:** LLMs are dealt with many routine and easy tasks that do not fully utilize its capabilities, leading to wasteful token usage

**Our Solution:** A system where an LLM can reason and is equipped with the ability to ask (tool-call) an SLM for help on easier tasks

**Models:** 
- LLM: GPT-4o
- SLM: Qwen2.5-Math-1.5B-Instruct (specialized for math)

**Scope:** This implementation focuses specifically on **mathematical problem-solving** (GSM8K dataset). The routing architecture is domain-agnostic and can be adapted to other specialized tasks.

**Key Insight:** Let the big model decide *when* and *what* to delegate, while the small model executes 'easy' reasoning tasks + computations.

---

## 📊 Results Summary
![Results](media/results.png)

| Method           | Accuracy | Avg Latency | Avg Total Tokens |
|------------------|----------|-------------|------------------|
| **GPT-4o Only**  | 86.40%   | 5.24s       | 398.7            |
| **Router (Ours)**| **88.20%** ✓ | 12.56s | **82.7** ✓        |
| **Qwen Only**    | 82.40%   | 8.85s       | 386.1            |

**Key Findings:**
- ✅ **~80% token reduction** (398.7 -> 82.7)
- ✅ 1.8% accuracy improvement over GPT-4o (88.2% vs. 86.4%)
- ⚠️ Trade: +140% slower (12.6s vs. 5.2s)
- Averages 1.16 tool calls per problem
- Averages 7.2s latency for SLM during tool call

---

## 🏗️ Work Flow
![Work Flow](media/workflow.png)

---

## 📺 Demo
[![Watch the demo](https://img.youtube.com/vi/minun1uOmSI/0.jpg)](https://youtu.be/minun1uOmSI)

### Try it yourself! 

```bash
git clone https://github.com/yourusername/delegate
cd delegate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-your-key" > .env
python demo.py
```

The demo shows real-time delegation between LLM and SLM, displaying tool call to SLM and its reasoning process.

---

## ⚡️ Quick Start

### Requirements

**Hardware:**
- GPU recommended (CUDA or Apple MPS) for Qwen2.5-Math-1.5B
- CPU works but 5-10x slower
- 3GB disk space for model weights

**Software:**
- Python 3.8+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/delegate
cd delegate

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Verify setup
python tools/check_setup.py
```

### Run Experiments

```bash
# Quick test (10 samples, ~5 minutes)
python experiments/run_comparison.py --samples 10 --seed 123

# Full benchmark (500 samples, ~2-3 hours)
python experiments/run_comparison.py --samples 500 --seed 123
```

Results are saved to `results/results_comparison_*/` with detailed metrics and visualizations.

---

## 🗄️ Project Structure

```
delegate/
├── demo.py                    # Interactive demo
├── router_agent_demo.py            # Core routing logic (w/out metric tracking)
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env                       # Your API key (create this)
│
├── experiments/               # Benchmarking scripts
│   ├── run_comparison.py      # Compare all 3 approaches
│   ├── llm_experiment.py      # GPT-4o baseline
│   ├── router_experiment.py   # Router system
│   ├── slm_experiment.py      # Qwen baseline
│   ├── router_agent.py        # Core routing logic (w/ metric tracking)
│   └── utils.py               # Answer checking utilities
│
├── tools/                     # Helper utilities
│   ├── check_setup.py         # Verify installation
│   └── analyze_results.py     # Generate charts from results
|   └── gsm8k_loader.py        # GSM8K dataset loader
│
│
└── media/                     # Images and assets
    └── workflow.png
```

**Key files:**
- `demo.py` - Interactive demo showing the system in action
- `experiments/run_comparison.py` - Run benchmarks on GSM8K
- `tools/analyze_results.py` - Visualize experiment results
- `router_agent.py` - Core delegation logic and tool definitions

---

## 🎯 What's Next?
- Expand Routing: Route tasks based on SLM performance strengths.
- Optimize Latency: Improve SLM inference speed
- Expand Domains: Extend evaluation to more 'agentic' tasks (e.g. Deep Search)

---

## 📁 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o API
- **Alibaba Qwen Team** for Qwen2.5-Math-1.5B-Instruct model
- **HuggingFace** for model hosting and transformers library
- **GSM8K** dataset creators (Cobbe et al., 2021)
- **OpenAI Agents** library for tool orchestration framework