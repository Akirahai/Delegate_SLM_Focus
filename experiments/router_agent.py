# experiments/router_agent.py
"""
Router Agent with token tracking
"""
import os, re, time, torch
from dotenv import load_dotenv
from agents import Agent, function_tool, ModelSettings
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# SLM lazy loader
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
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        _TOK = AutoTokenizer.from_pretrained(_SLM_ID)
        _TOK.padding_side = "left"
    return _SLM, _TOK


@function_tool
def slm_help(question: str) -> str:
    """
    Solve a mathematical calculation using specialized math model.
    Returns definitive answer that should be trusted immediately.
    
    Args:
        question: The calculation to perform
    
    Returns:
        Definitive answer in format "CALCULATION COMPLETE: answer"
    """
    print(f"[TOOL] slm_help: {question[:60]}...")
    
    try:
        model, tok = _lazy_load_slm()

        sys = "You are a math calculator. Solve step-by-step. Put final answer in \\boxed{} at the end."
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": question},
        ]
        
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([prompt], return_tensors="pt").to(model.device)
        
        # Count input tokens
        input_tokens = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        latency = time.time() - t0
        
        # Count output tokens (only new tokens)
        output_tokens = out.shape[1] - input_tokens
        
        gen = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        # Extract boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', gen)
        
        # Log to tracker
        try:
            import sys as _sys
            if 'router_experiment' in _sys.modules:
                from router_experiment import tracker
                tracker.log_tool_call(question, gen, latency, input_tokens, output_tokens)
                print(f"[TRACKER] Logged: {latency:.2f}s, {input_tokens}→{output_tokens} tokens")
        except Exception as e:
            print(f"[TRACKER] Failed: {e}")

        if match:
            answer = match.group(1)
            result = f"CALCULATION COMPLETE: The answer is {answer}. Use this directly."
            print(f"[SLM] Answer: {answer} ({latency:.2f}s)")
            return result
        else:
            result = f"CALCULATION COMPLETE: {gen}"
            print(f"[SLM] No boxed answer ({latency:.2f}s)")
            return result
        
    except Exception as e:
        return f"CALCULATION ERROR: {str(e)}. Solve yourself."


# Agent definition
INSTRUCTIONS = (
    "You solve math problems step by step.\n\n"
    
    "WORKFLOW:\n"
    "1. Understand the problem\n"
    "2. For ANY calculation, call slm_help ONCE\n"
    "3. When you receive 'CALCULATION COMPLETE:', use that answer\n"
    "4. NEVER call slm_help again for the same calculation\n"
    "5. Provide final answer in \\boxed{}\n\n"
    
    "CRITICAL:\n"
    "• Call slm_help for ALL arithmetic/algebra\n"
    "• Trust 'CALCULATION COMPLETE' results immediately\n"
    "• Do NOT verify or retry calculations\n"
    "• After final answer, write \\boxed{answer} and STOP"
)

agent = Agent(
    name="Math Router",
    instructions=INSTRUCTIONS,
    model="gpt-4o",
    model_settings=ModelSettings(
        max_tokens=512,
        parallel_tool_calls=False
    ),
    tools=[slm_help],
)