import os
import gc
import json
import torch
import chainlit as cl

from agents.base import BaseAgent
from agents.coding import CodingAgent
from backends.vllm_backend import VLLMBackend

from utils.io import load_json, update_json, update_summary_md
from utils.parser import parse_json_response
from config import PATHS, CURRENT_MODEL, MODELS

# ---------------- CONFIGURATION ----------------
DEFAULT_AGENT = "coding"
PROJECT_MEMORY = os.path.join(PATHS["projects_dir"], "project_memory.json")
SUMMARY_MD = os.path.join(PATHS["projects_dir"], "summarized.md")
REPO_PATH = "/home/bdudas/agentFlow"  # Path for the CodingAgent X-Ray vision

# ---------------- AGENTS ----------------
agents = {
    "qa": BaseAgent(
        name="qa",
        backend=VLLMBackend(base_url=MODELS["qa"]["base_url"], model=MODELS["qa"]["model_name"]),
        prompt_dir="prompts/qa",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_qa.json")
    ),

    "coding": CodingAgent(
        name="coding",
        backend=VLLMBackend(
            base_url=MODELS["coding"]["base_url"], 
            model=MODELS["coding"]["model_name"],
            max_tokens=MODELS["coding"].get("max_tokens", 2048),
            temperature=MODELS["coding"].get("temperature", 0.0)
        ),
        prompt_dir="prompts/coding",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_coding.json"),
        repo_path=REPO_PATH  # Required for your custom coding agent
    ),

    "writing": BaseAgent(
        name="writing",
        backend=VLLMBackend(base_url=MODELS["writing"]["base_url"], model=MODELS["writing"]["model_name"]),
        prompt_dir="prompts/writing",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_writing.json")
    ),

    "summarizer": BaseAgent(
        name="summarizer",
        backend=VLLMBackend(base_url=MODELS["summarizer"]["base_url"], model=MODELS["summarizer"]["model_name"]),
        prompt_dir="prompts/summarizer",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_summarizer.json")
    ),

    "critic": BaseAgent(
        name="critic",
        backend=VLLMBackend(base_url=MODELS["critic"]["base_url"], model=MODELS["critic"]["model_name"]),
        prompt_dir="prompts/critic",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_critic.json")
    ),
}

# ---------------- SUMMARIZER LOGIC ----------------
def summarize_interaction(summarizer_agent, agent_name, user_input, agent_response):
    """Runs synchronously in a background thread via cl.make_async"""
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
Only include information worth remembering. Do not include trivial chat.
"""
    # 1. Generate the summary
    summary = summarizer_agent.run(summary_prompt)

    try:
        # 2. Parse and save to JSON archive
        summary_json = parse_json_response(summary)
        update_json(PROJECT_MEMORY, summary_json)
        
        # 3. Create a Markdown version for the 'Clipboard'
        # We format the JSON into a clean markdown string so the Coder can read it easily
        markdown_content = "# Current Project State\n\n"
        for key, items in summary_json.items():
            if items:
                markdown_content += f"### {key.replace('_', ' ').title()}\n"
                for item in items:
                    markdown_content += f"- {item}\n"
                markdown_content += "\n"
                
        # 4. Overwrite the summarized.md file
        update_summary_md(SUMMARY_MD, markdown_content)
        print(f"\n[System] Project Memory & summarized.md updated by {agent_name} interaction.")

    except Exception as e:
        print(f"Could not parse summary: {e}")

# ---------------- CHAINLIT UI ----------------

@cl.on_chat_start
async def on_chat_start():
    # Save active agent for this chat session
    cl.user_session.set("agent_name", DEFAULT_AGENT)

    await cl.Message(
        content=f"""
# 🧪 AgentFlow AI Orchestrator

Welcome to your Medical Physics Research Assistant!

**Current active agent:** `{DEFAULT_AGENT}`

### Available Agents:
- **coding** (Has full repository X-Ray vision)
- **qa** (General questions)
- **writing** (Drafting papers)
- **critic** (Mathematical & logic review)

*To switch agents, type:* `/agent [name]` (e.g., `/agent critic`)
"""
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip()

    # -------------------------------------------
    # Switch agent command
    # -------------------------------------------
    if user_input.startswith("/agent"):
        parts = user_input.split()
        if len(parts) != 2:
            await cl.Message(content="Usage:\n\n`/agent coding`").send()
            return

        new_agent = parts[1].lower()
        if new_agent not in agents:
            await cl.Message(content=f"Unknown agent **{new_agent}**").send()
            return

        cl.user_session.set("agent_name", new_agent)
        await cl.Message(content=f"✅ Successfully switched to **{new_agent}**!").send()
        return

    # -------------------------------------------
    # Process Message
    # -------------------------------------------
    agent_name = cl.user_session.get("agent_name", DEFAULT_AGENT)
    current_agent = agents[agent_name]

    # Display a loading spinner in the UI
    thinking = cl.Message(content=f"🤔 `{agent_name}` is thinking...")
    await thinking.send()

    try:
        # CRITICAL: We wrap the synchronous .run() call in cl.make_async 
        # so it doesn't freeze the Chainlit web server!
        async_run = cl.make_async(current_agent.run)
        response = await async_run(user_input)

        # Update the UI message with the final response
        thinking.content = response
        await thinking.update()

        # ---------------------------------------
        # Background Summarization
        # ---------------------------------------
        # We run the summarizer asynchronously so the user doesn't 
        # have to wait for the JSON parsing to finish before typing again.
        async_summarize = cl.make_async(summarize_interaction)
        await async_summarize(
            summarizer_agent=agents["summarizer"],
            agent_name=agent_name,
            user_input=user_input,
            agent_response=response
        )

    except Exception as e:
        thinking.content = f"❌ **Error running {agent_name}:**\n\n```text\n{e}\n```"
        await thinking.update()

    finally:
        # Note: torch.cuda.empty_cache() only works if the PyTorch model 
        # is loaded in THIS specific Python script. Since your models are 
        # hosted on vLLM API servers, this does nothing, but gc.collect() is good!
        gc.collect()