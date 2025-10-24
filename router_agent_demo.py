# router_agent_demo.py
import os, re, time, asyncio, torch
from dotenv import load_dotenv

from agents import Agent, function_tool, Runner, ModelSettings
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# ---------------------------
# SLM (Qwen) lazy loader
# ---------------------------
_SLM, _TOK = None, None
_SLM_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"

def _device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32

def _lazy_load_slm():
    global _SLM, _TOK
    if _SLM is None or _TOK is None:
        device, dtype = _device_dtype()
        _SLM = AutoModelForCausalLM.from_pretrained(
            _SLM_ID,
            device_map="auto" if device != "cpu" else None,
            dtype=dtype,
            trust_remote_code=True,
        )
        _TOK = AutoTokenizer.from_pretrained(_SLM_ID)
        _TOK.padding_side = "left"
    return _SLM, _TOK

# ---------------------------
# Tool: call SLM for help
# ---------------------------
@function_tool
def slm_help(question: str) -> str:
    """
    Use specialized math model to solve ANY calculation or equation.
    
    IMPORTANT: You MUST use this tool for ALL arithmetic operations, 
    no matter how simple (addition, subtraction, multiplication, division, etc.)
    
    Args:
        question: The mathematical question or calculation to solve
    
    Returns:
        The definitive answer from the specialized math model
    """
    print(f"[TOOL] slm_help: {question[:60]}...")
    
    try:
        model, tok = _lazy_load_slm()

        sys = "You are a math calculator. Put the final answer in \\boxed{} at the end."
        
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": question},
        ]
        
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([prompt], return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        
        gen = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        latency = time.time() - t0

        print(f"[SLM OUTPUT] Time: {latency:.2f}s")
        print(f"[SLM OUTPUT] Full response:\n{gen}\n")

        # Extract the boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', gen)

        if match:
            answer = match.group(1)
            result = f"CALCULATION COMPLETE: The answer is {answer}"
            print(f"[SLM RESULT] Extracted answer: {answer}")
            return result
        else:
            result = f"CALCULATION COMPLETE: {gen}"
            print(f"[SLM RESULT] No boxed answer found, returning full output")
            return result
        
    except Exception as e:
        return f"CALCULATION ERROR: {str(e)}. Please try a different approach."

# ---------------------------
# Agent with STRONGER routing
# ---------------------------
INSTRUCTIONS = """You are a math problem solver with access to a specialized calculation tool. You reason and think in steps.

You MUST use the `slm_help` tool for EVERY calculation, including:
- Basic arithmetic (addition, subtraction, multiplication, division)
- Percentages and fractions
- Equations and algebra
- Any operation involving numbers

## Your Workflow:

1. Read the problem carefully
2. Identify what calculations are needed
3. For EACH calculation (no matter how simple):
   - Call slm_help(question) with the specific calculation
   - Wait for the result
   - Integrate that result in your reasoning
4. After all calculations are complete, provide the final answer in \\boxed{} format

## Examples:

Problem: "What is 156 + 243?"
→ CORRECT: Call slm_help("156 + 243")
→ WRONG: Answering "399" directly

Problem: "If I buy 3 shirts at $15 each, what's the total?"
→ CORRECT: Call slm_help("3 × 15")
→ WRONG: Saying "3 times 15 is 45"

Problem: "Natalia sold 48 clips in April and half as many in May. Total?"
→ Step 1: Call slm_help("48 ÷ 2") to find May's amount
→ Step 2: Call slm_help("48 + [May's amount]") to find total

## Remember:
- ALWAYS use the tool for calculations
- NEVER compute anything yourself
- Present final answer as \\boxed{answer}
"""

agent = Agent(
    name="Math Expert Agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o",
    model_settings=ModelSettings(
        max_tokens=512,
        parallel_tool_calls=False,
        temperature=0 
    ),
    tools=[slm_help],
)

async def run_agent(question: str):
    result = await Runner.run(agent, question, max_turns=15)
    return result.final_output