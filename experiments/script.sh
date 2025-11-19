#!/bin/bash
# ===== CONFIGURATION =====
# MODELS=(
#   # "Qwen/Qwen2.5-Math-1.5B-Instruct"
#   # "Qwen/Qwen2.5-Math-7B-Instruct"
#   # "arifulFarhad/qwen-2.5-math-7b-4bit-gptq"
#   # "gmonair/Qwen2.5-Math-1.5B-Instruct-8bit-GPTQ"
#   # "shuyuej/mathstral-7B-v0.1-GPTQ"
#   # "TheBloke/Llama-2-7B-Chat-GPTQ"
#   "Gemini/Gemini-2.5-Flash"
# )

GPUS="5"
SAMPLES=500
SEED=0
DATASET="math_500"
# DATASET="gsm8k"
MAX_TOKENS=8000

# Flags
SKIP_LLM=false
SKIP_ROUTER=true
SKIP_SLM=true

# Run on this output_direction
output_dir="results_Gemini-2.5-Flash_New_${SAMPLES}samples_${DATASET}_path"




# ===== MAIN LOOP =====
echo "============================================"
echo "Running comparison for model: $MODEL"
echo "GPUs: $GPUS | Samples: $SAMPLES | Seed: $SEED"
echo "Dataset: $DATASET | Max tokens: $MAX_TOKENS"
echo "============================================"

CMD="python run_comparison.py \
  --gpus $GPUS \
  --samples $SAMPLES \
  --seed $SEED \
  --data $DATASET \
  --max-tokens $MAX_TOKENS \
  --output-dir $output_dir"

if [ "$SKIP_LLM" = true ]; then
  CMD="$CMD --skip-llm"
fi

if [ "$SKIP_ROUTER" = true ]; then
  CMD="$CMD --skip-router"
fi

if [ "$SKIP_SLM" = true ]; then
  CMD="$CMD --skip-slm"
fi

echo "Executing: $CMD"
eval $CMD
echo