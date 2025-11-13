# experiments/router_agent.py
"""
Router Agent with token tracking
"""
import os, re, time, torch
from dotenv import load_dotenv
from agents import Agent, function_tool, ModelSettings, Runner
from agents.extensions.models.litellm_model import LitellmModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import extracted_box, len_extract_boxed
from grader import math_equal
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SLM_INSTRUCTION = """
You are an expert at solving math problems. Solve this math problem step by step. Put your final answer in \\boxed{{}} format.
"""
LLM_INSTRUCTION = """
You are an expert at solving math problems. Solve this problem step by step. \
Make sure that your final answer should be simplified (no units). \
Provide your final answer in \\boxed{{answer}} format.
"""


Orchestrator_INSTRUCTION = """
You are a math problem orchestrator. You **cannot solve problems yourself**. 
For easy problems, hand off to slm_agent.
For tricky and hard problems, hand off to llm_agent.
Do not attempt any reasoning yourself.
Return exactly the output from the handoff agent.
"""

def initialize_all_agents(SLM_MODEL, LLM_MODEL, ORCHESTRATOR_MODEL):

    slm_agent = Agent(
    name="SLM Math Expert",
    instructions=SLM_INSTRUCTION,
    model=LitellmModel(model=f"hosted_vllm/{SLM_MODEL}", base_url="http://localhost:8000/v1"),
    model_settings=ModelSettings(
        max_tokens=2000,
        include_usage=True,
    ),
    )

    llm_agent = Agent(
    name="LLM Math Experts",
    instructions=LLM_INSTRUCTION,
    model=LitellmModel(model=LLM_MODEL, api_key=GEMINI_API_KEY),
    model_settings=ModelSettings(
        max_tokens=6000,
        parallel_tool_calls=False,
        reasoning={"effort": "medium"},
        include_usage=True,
    )
    )

    orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=Orchestrator_INSTRUCTION,
    model=LitellmModel(model=ORCHESTRATOR_MODEL, api_key=GEMINI_API_KEY),
    # model=LitellmModel(model=f"hosted_vllm/{ORCHESTRATOR_MODEL}", base_url="http://localhost:8000/v1"),
    model_settings=ModelSettings(
        max_tokens=1024,
        reasoning={"effort": "minimal"},
        include_usage=True,
    ),
    handoffs=[slm_agent, llm_agent],
    )


    return orchestrator_agent











