# tools/check_setup.py
"""
Setup checker for GSM8K comparison experiments
Verifies all dependencies and files are ready
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  Python 3.8+ recommended")
        return False
    return True


def check_imports():
    """Check required packages"""
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'openai': 'OpenAI',
        'pandas': 'Pandas',
        'datasets': 'HuggingFace Datasets',
        'agents': 'Agents library',
        'dotenv': 'python-dotenv'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True


def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  No CUDA GPU - SLM will run on CPU (slower)")
            return True
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def check_env():
    """Check environment variables"""
    env_file = Path('.env')
    if env_file.exists():
        print("✓ .env file found")
        
        # Check for OpenAI key
        from dotenv import load_dotenv
        load_dotenv()
        
        if os.getenv('OPENAI_API_KEY'):
            print("✓ OPENAI_API_KEY set")
            return True
        else:
            print("✗ OPENAI_API_KEY not found in .env")
            return False
    else:
        print("✗ .env file not found")
        print("  Create .env with: OPENAI_API_KEY=your_key_here")
        return False


def check_files():
    """Check required files exist"""
    required_files = [
        'gsm8k_loader.py',
        'utils.py',
        'run_comparison.py',
        'llm_experiment_v2.py',
        'router_experiment_v2.py',
        'router_agent_v2.py',
        'slm_experiment_v2.py'
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    return True


def check_model_cache():
    """Check if models are cached"""
    cache_dir = Path.home() / '.cache' / 'huggingface'
    if cache_dir.exists():
        print(f"✓ HuggingFace cache: {cache_dir}")
        
        # Check for Qwen model
        model_files = list(cache_dir.rglob('*Qwen*'))
        if model_files:
            print("✓ Qwen model found in cache")
        else:
            print("⚠️  Qwen model not cached - will download on first run (~3GB)")
    else:
        print("⚠️  No HuggingFace cache - models will download on first run")
    
    return True


def main():
    print("="*70)
    print("GSM8K COMPARISON - SETUP CHECK")
    print("="*70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("GPU/CUDA", check_gpu),
        ("Environment Variables", check_env),
        ("Required Files", check_files),
        ("Model Cache", check_model_cache)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 70)
        results[name] = check_func()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to run experiments.")
        print("\nQuick start:")
        print("  python run_comparison.py --samples 10 --seed 123")
    else:
        print("\n⚠️  Some checks failed. Fix issues above before running.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())