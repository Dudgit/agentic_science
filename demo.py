import os
import gc
import json
import torch
import chainlit as cl

from agents.base import BaseAgent
from agents.coding import CodingAgent
from backends.vllm_backend import VLLMBackend
from agents.researcher import ResearcherAgent

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
    "researcher": ResearcherAgent(
        name="researcher",
        backend=VLLMBackend(base_url=MODELS["critic"]["base_url"], model=MODELS["critic"]["model_name"]),
        prompt_dir="prompts/researcher",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_researcher.json")
    )
}

# ---------------- INTERFACE CALLBACKS ----------------

@cl.action_callback("switch_agent")
async def on_switch_agent(action: cl.Action):
    # Pull the string out of the payload dictionary
    new_agent = action.payload["agent"]
    
    cl.user_session.set("agent_name", new_agent)
    
    await cl.Message(
        content=f"✅ Switched to **{new_agent.capitalize()}**! Ready for your prompt."
    ).send()




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

    # Dynamically generate a clickable button for every agent
    agent_buttons = [
        # Notice payload={"agent": name} instead of payload=name
        cl.Action(name="switch_agent", payload={"agent": name}, label=name.capitalize())
        for name in agents.keys()
    ]

    await cl.Message(
        content=f"""
# AgentFlow

Welcome!

Current agent: **{DEFAULT_AGENT}**

Click a button below to switch your active agent. 
*(If this menu scrolls out of view, just type `/menu` to bring it back!)*
""",
        actions=agent_buttons
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip()

    # -------------------------------------------
    # Menu trigger
    # -------------------------------------------
    if user_input.lower() == "/menu":
        agent_buttons = [
            # Same here: payload={"agent": k}
            cl.Action(name="switch_agent", payload={"agent": k}, label=k.capitalize())
            for k in agents.keys()
        ]
        await cl.Message(
            content="👇 Click a button to switch your active agent:", 
            actions=agent_buttons
        ).send()
        return

    # -------------------------------------------
    # Current agent processing
    # -------------------------------------------
    agent_name = cl.user_session.get("agent_name", DEFAULT_AGENT)
    current_agent = agents[agent_name]

    # -------------------------------------------
    # Async Inference Execution
    # -------------------------------------------
    thinking = cl.Message(content="🤔 Thinking...")
    await thinking.send()

    try:
        # Run the synchronous backend pipeline inside a non-blocking background thread
        response = await cl.make_async(current_agent.run)(user_input)

        # Update placeholder message with the actual formatted markdown output
        thinking.content = response
        await thinking.update()

        # ---------------------------------------
        # Background Memory Update
        # ---------------------------------------
        # Spin up the summarizer without freezing the UI line
        await cl.make_async(summarize_interaction)(
            summarizer_agent=agents["summarizer"],
            agent_name=agent_name,
            user_input=user_input,
            agent_response=response
        )

    except Exception as e:
        thinking.content = f"❌ Error executing engine:\n\n```text\n{e}\n```"
        await thinking.update()

    finally:
        # VRAM Hygiene loop
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()