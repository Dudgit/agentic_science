# main.py

from agents.base import BaseAgent
from agents.coding import CodingAgent
from backends.vllm_backend import VLLMBackend
from backends.tf_backend import TFBackend

from utils.io import load_json, update_json, update_summary_md
from utils.parser import parse_json_response
from config import PATHS, CURRENT_MODEL
import os
import gc
import torch

# ---------------- BACKEND ----------------
backend = VLLMBackend(base_url="http://localhost:8000/v1",model=CURRENT_MODEL)
#TFBackend(CURRENT_MODEL) 


# ---------------- AGENTS ----------------
agents = {
    "qa": BaseAgent(
        name="qa",
        backend=backend,
        prompt_dir="prompts/qa",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_qa.json")
    ),

    "coding": CodingAgent(
        name="coding",
        backend=backend,
        prompt_dir="prompts/coding",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_coding.json")
    ),

    "writing": BaseAgent(
        name="writing",
        backend=backend,
        prompt_dir="prompts/writing",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_writing.json")
    ),

    "summarizer": BaseAgent(
        name="summarizer",
        backend=backend,
        prompt_dir="prompts/summarizer",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_summarizer.json")
    ),

    "critic": BaseAgent(
        name="critic",
        backend=backend,
        prompt_dir="prompts/critic",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_critic.json")
    ),
}


PROJECT_MEMORY = "project_memory.json"


def summarize_interaction(
        summarizer_agent,
        agent_name,
        user_input,
        agent_response):

    summary_prompt = f"""
You are extracting persistent project knowledge.

Agent used: {agent_name}

User:
{user_input}

Agent response:
{agent_response}

Extract ONLY useful long-term information.

Return JSON with the following schema:

{{
    "important_decisions": [],
    "implemented_features": [],
    "research_ideas": [],
    "todos": [],
    "papers_to_read": []
}}
Only include information worth remembering.
Do not include trivial chat.
"""

    summary = summarizer_agent.run(summary_prompt)

    print("\n--- SUMMARY ---")
    print(summary)

    try:
        summary_json = parse_json_response(summary)

        update_json(
            PROJECT_MEMORY,
            summary_json
        )

    except Exception as e:
        print(f"Could not parse summary: {e}")



# ---------------- CLI LOOP ----------------
def choose_agent():
    print("\nAvailable agents:")
    for k in agents.keys():
        print(f" - {k}")

    while True:
        choice = input("\nSelect agent: ").strip()
        if choice in agents:
            return agents[choice]
        print("Invalid agent. Try again.")


def main():
    print("\nAgent system ready.")

    current_agent = choose_agent()
    print(f"\nUsing agent: {current_agent.name}")

    while True:
        user_input = input("\nInput (or 'switch'): ")

        if user_input == "switch":
            current_agent = choose_agent()
            print(f"\nSwitched to: {current_agent.name}")
            continue

        if user_input in ["exit", "quit"]:
            break

        response = current_agent.run(user_input)

        print("\n--- RESPONSE ---\n")
        print(response)

        summarize_interaction(
            summarizer_agent=agents["summarizer"],
            agent_name=current_agent.name,
            user_input=user_input,
            agent_response=response
        )

        print("\n--- RESPONSE ---\n")
        print(response)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()