import os
import json
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
client = OpenAI(
    base_url="http://vllm.default:8000/v1",
    api_key="sk-no-key-required"

)

class AgentState(TypedDict):
    """The central state bus passed to every node in the graph."""
    user_query: str
    project_context: Optional[str] 
    current_draft: str              # Code or theoretical response
    critique: Optional[str]         # Bulleted feedback from the critic
    iteration: int                  # Current loop counter
    max_iterations: int             # Hard safety ceiling
    is_approved: bool               # Set to True when critic finds no critical flaws
    final_output: str

from config import MODELS, PATHS
from utils.io import strip_thoughts

def call_local_llm(system_prompt: str, user_prompt: str, model:str = "Qwen/Qwen3.6-35B-A3B-FP8", max_tokens:int=4096, temperature:float=0.2) :
    response = client.chat.completions.create(
         model=model,
         messages=[
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}
         ],
         temperature=temperature,
         max_tokens=max_tokens
     )
    return strip_thoughts(response.choices[0].message.content)

def generator_node(state: AgentState) -> dict:
    """Generates initial code/answer or revises existing draft using critique."""
    iteration = state.get("iteration", 0) + 1
    
    system_prompt = ("")
    with open("prompts/qa/system.md", "r") as f:
        system_prompt += f"{f.read()}\n"

    if iteration == 1:
        user_prompt = f"Task: {state['user_query']}\n."
    else:
        # Revision pass using critic feedback
        user_prompt = (
            f"Task: {state['user_query']}\n\n"
            f"### PREVIOUS DRAFT ###\n{state['current_draft']}\n\n"
            f"### CRITIC FEEDBACK TO RESOLVE ###\n{state['critique']}\n\n"
            f"Rewrite the implementation addressing all points above.")

    draft = call_local_llm(system_prompt, user_prompt)
    
    return {"current_draft": draft,"iteration": iteration}

def critic_node(state: AgentState) -> dict:
    """Audits the draft strictly for correctness, syntax, and project alignment."""
    system_prompt = (
        "You are an Adversarial Code Reviewer. Audit the provided draft.\n"
        "If the draft has bugs, dimension errors, or violates project constraints, "
        "output ONLY a bulleted list of issues.\n"
        "If the draft is production-ready and fully functional, respond with exactly: 'PASSED'.")
    
    user_prompt = (
        f"Requirement: {state['user_query']}\n"
        f"Project Constraints: {state['project_context']}\n"
        f"Draft Implementation:\n{state['current_draft']}")

    feedback = call_local_llm(system_prompt, user_prompt).strip()
    is_approved = ("PASSED" in feedback.upper()) or (state["iteration"] >= state["max_iterations"])

    return {"critique": feedback,"is_approved": is_approved}


def summarizer_node(state: AgentState) -> dict:
    """Updates global project memory files and finalizes delivery."""
    # Persist key deliverables into project_memory.json
    memory_path = "project_memory.json"
    memory_entry = {
        "task": state["user_query"],
        "iterations": state["iteration"],
        "solution": state["current_draft"]
    }
    
    # Simple JSON persistence
    existing_mem = []
    if os.path.exists(memory_path):
        with open(memory_path, "r", encoding="utf-8") as f:
            try:
                existing_mem = json.load(f)
            except Exception:
                existing_mem = []
    existing_mem.append(memory_entry)
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(existing_mem, f, indent=4)

    return {"final_output": state["current_draft"]}

def route_after_critique(state: AgentState) -> str:
    """Determines whether to loop back for another draft or finalize."""
    if state["is_approved"] or state["iteration"] >= state["max_iterations"]:
        return "summarizer"
    return "generator"

def route_after_critique(state: AgentState) -> str:
    """Determines whether to loop back for another draft or finalize."""
    if state["is_approved"] or state["iteration"] >= state["max_iterations"]:
        return "summarizer"
    return "generator"

def build_pipeline():
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("summarizer", summarizer_node)

    # Establish Edges
    workflow.add_edge(START, "generator")
    workflow.add_edge("generator", "critic")
    
    # Conditional Edge: Loop or Exit
    workflow.add_conditional_edges(
        "critic",
        route_after_critique,
        {"generator": "generator","summarizer": "summarizer"})
    workflow.add_edge("summarizer", END)
    return workflow.compile()