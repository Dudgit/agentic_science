import os
import gc
import json
import torch
import chainlit as cl
import re

from agents.base import BaseAgent
from agents.coding import CodingAgent
from backends.vllm_backend import VLLMBackend
from agents.researcher import ResearcherAgent

from utils.io import load_json, update_json, update_summary_md, strip_thoughts, reset_all_memories
from utils.parser import parse_json_response
from config import PATHS, CURRENT_MODEL, MODELS

# ---------------- CONFIGURATION ----------------
DEFAULT_AGENT = "coding"
PROJECT_MEMORY = os.path.join(PATHS["projects_dir"], "project_memory.json")
SUMMARY_MD = os.path.join(PATHS["projects_dir"], "summarized.md")
REPO_PATH = "/home/bdudas/agentFlow/agentic_science/sample_project"  # Path for the CodingAgent X-Ray vision
ENABLE_CIRITC = False
# ---------------- AGENTS ----------------



agents = {
    "qa": BaseAgent(
        name="qa",
        backend=VLLMBackend(**dict(MODELS["qa"])),
        prompt_dir="prompts/qa",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_qa.json")
    ),

    #"coding": CodingAgent(
    #    name="coding",
    #    backend=VLLMBackend(**dict(MODELS["coding"])),
    #    prompt_dir="prompts/coding",
    #    memory_file=os.path.join(PATHS["projects_dir"], "memory_coding.json"),
    #    repo_path=REPO_PATH  # Required for your custom coding agent
    #),

    "writing": BaseAgent(
        name="writing",
        backend=VLLMBackend(**dict(MODELS["writing"])),
        prompt_dir="prompts/writing",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_writing.json")
    ),

    "summarizer": BaseAgent(
        name="summarizer",
        backend=VLLMBackend(**dict(MODELS["summarizer"])),
        prompt_dir="prompts/summarizer",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_summarizer.json")
    ),

    "critic": BaseAgent(
        name="critic",
        backend=VLLMBackend(**dict(MODELS["critic"])),
        prompt_dir="prompts/critic",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_critic.json")
    ),
    "researcher": ResearcherAgent(
        name="researcher",
        backend=VLLMBackend(**dict(MODELS["researcher"])),
        prompt_dir="prompts/researcher",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_researcher.json")
    )
}

chosable_agents = {"qa": agents["qa"], "writing": agents['writing']}
# ---------------- INTERFACE CALLBACKS ----------------

@cl.action_callback("switch_agent")
async def on_switch_agent(action: cl.Action):
    # 1. Instantly set the variable (takes 1 microsecond)
    target_agent = action.payload["agent"]
    cl.user_session.set("agent_name", target_agent)
    
    # 2. DO NOT send a heavy new message. Just remove the button to show it worked.
    await action.remove()




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
        for name in chosable_agents.keys()
    ]

    await cl.Message(
        content=f"""
# ELTE Scientific AI Assistant platform

Welcome!

Current agent: **{DEFAULT_AGENT}**

Please explain what project you are working on and we will try to assist you!
""",
        actions=agent_buttons
    ).send()



@cl.action_callback("reset_memory_action")
async def on_reset_memory(action: cl.Action):
    reset_all_memories()
    await cl.Message(
        content="🧹 **All project memory files, agent logs, and context history have been wiped clean!**"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip()
    if user_input.lower() == "/reset":
        reset_all_memories()
        await cl.Message(content="🧹 **All memory files and agent logs reset.**").send()
        return

    # Handle the menu command override
    if user_input.lower().startswith("/agent"):
        parts = user_input.split()
        if len(parts) > 1:
            target_agent = parts[1].lower()
            if target_agent in chosable_agents:
                cl.user_session.set("agent_name", target_agent)
                await cl.Message(
                    content=f"🤖 Switched active agent to **{target_agent.upper()}**."
                ).send()
                return
            else:
                avail = ", ".join([f"`{k}`" for k in chosable_agents.keys()])
                await cl.Message(
                    content=f"⚠️ Agent `{target_agent}` not found. Available: {avail}"
                ).send()
                return
        else:
            avail = ", ".join([f"`{k}`" for k in chosable_agents.keys()])
            await cl.Message(
                content=f"ℹ️ Usage: `/agent <name>`. Available: {avail}"
            ).send()
            return
    if user_input.lower() == "/reset":
        reset_all_memories()
        await cl.Message(
            content="🧹 **All project memory files, agent logs, and context history have been wiped clean!**"
        ).send()
        return
    # ---------------------------------------------------------
    # IDENTIFY TARGET AGENTS
    # ---------------------------------------------------------
    agent_name = cl.user_session.get("agent_name", DEFAULT_AGENT)
    chosen_agent = chosable_agents[agent_name]
    
    # Fallback to QA agent for planning if an explicit planner agent isn't configured
    planner_agent = agents.get("planner", agents["qa"])
    critic_agent = agents["critic"]

    # Initialize a clean, dynamic status tracker in the UI
    status_msg = cl.Message(content="🤖 Initializing Agentic Workflow Assembly Line...")
    await status_msg.send()

    try:
        # ---------------------------------------------------------
        # STEP 1: Planner
        # ---------------------------------------------------------
        if agent_name == "qa":
            status_msg.content = "🧠 **Thinking and generating theoretical response (QA Agent)...**"
            await status_msg.update()
            
            # QA gets the direct user input, no planner, no execution strings.
            raw_response = await cl.make_async(chosen_agent.run)(user_input)
            clean_response = strip_thoughts(raw_response)

            status_msg.content = f"## Scientific Assistant Response\n\n{clean_response}"
            await status_msg.update()

            # Background summarization
            await cl.make_async(summarize_interaction)(
                agents["summarizer"], agent_name, user_input, clean_response
            )
            return  # <-- EXIT HERE. Do not run the rest of the pipeline!

        # =========================================================
        # PATH B: CODING AGENT (Planner -> Coder Assembly Line)
        # =========================================================
        # ---------------------------------------------------------
        # STEP 1: Planner
        # ---------------------------------------------------------
        status_msg.content = "🏗️ **Phase 1: Architecting System Plan (Planner)...**"
        await status_msg.update()
        
        planner_prompt = (
            f"Break down the user's request into a concrete engineering plan or structural outline.\n"
            f"If you need external documentation, libraries, or recent data to complete this plan accurately, "
            f"you MUST start your response with: SEARCH: [your search query].\n\n"
            f"User Request: {user_input}"
        )
        raw_plan = await cl.make_async(planner_agent.run)(planner_prompt)
        plan_output = strip_thoughts(raw_plan)

        # ---------------------------------------------------------
        # STEP 2: Researcher (Web Search)
        # ---------------------------------------------------------
        if "SEARCH:" in plan_output:
            status_msg.content = "🌐 **Phase 2: Executing Autonomous Live Web Search...**"
            await status_msg.update()
            
            search_query = plan_output.split("SEARCH:")[1].strip()
            from utils.web import search_internet
            web_raw = search_internet(search_query)
            
            status_msg.content = "📊 **Phase 2b: Integrating Web Findings into System Blueprint...**"
            await status_msg.update()
            
            raw_plan = await cl.make_async(planner_agent.run)(
                f"We performed a search for '{search_query}'. Here are the findings:\n\n{web_raw}\n\n"
                f"Rewrite your structural engineering blueprint incorporating these new constraints and facts."
            )
            plan_output = strip_thoughts(raw_plan)

        # ---------------------------------------------------------
        # STEP 3: Execution (Coding Agent)
        # ---------------------------------------------------------
        status_msg.content = f"⚙️ **Phase 3: Generating Code ({agent_name.capitalize()} Agent)...**"
        await status_msg.update()
        
        # Here we enforce the strict "NO THINKING, ONLY FINAL ANSWER" rule, 
        # but it ONLY applies to the Coding agent now!
        execution_prompt = (
            f"### SYSTEM INSTRUCTION ###\n"
            f"You have been provided a pre-calculated architectural blueprint. "
            f"Do NOT use `<think>` blocks. Do NOT re-analyze the problem. "
            f"Immediately begin your response with `### FINAL ANSWER ###` and generate the complete code.\n\n"
            f"### BLUEPRINT TO IMPLEMENT ###\n{plan_output}"
        )
        raw_delivery = await cl.make_async(chosen_agent.run)(execution_prompt)
        initial_delivery = strip_thoughts(raw_delivery)

        # ---------------------------------------------------------
        # STEP 4 & 5: Critic (If Enabled)
        # ---------------------------------------------------------
        if ENABLE_CIRITC:
            status_msg.content = "🔍 **Running Critic Evaluation...**"
            await status_msg.update()
            
            critic_prompt = (
                f"Analyze the implementation against the requirement. Output ONLY bullet points of errors.\n\n"
                f"### REQUIREMENT ###\n{user_input}\n\n### IMPLEMENTATION ###\n{initial_delivery}")
            raw_critique = await cl.make_async(critic_agent.run)(critic_prompt)
            critique = strip_thoughts(raw_critique)

            status_msg.content = f"🛠️ **Executing Final Corrections ({agent_name.capitalize()} Agent)...**"
            await status_msg.update()
            
            correction_prompt = (
                f"Rewrite your previous output to address all identified flaws.\n\n"
                f"### PREVIOUS OUTPUT ###\n{initial_delivery}\n\n### CRITIC FEEDBACK ###\n{critique}\n\n"
                f"Immediately begin your response with `### FINAL ANSWER ###` and generate the corrected code."
            )
            raw_final = await cl.make_async(chosen_agent.run)(correction_prompt)
            final_delivery = strip_thoughts(raw_final)
        else:
            critique = "*- Critic evaluation bypassed -*"
            final_delivery = initial_delivery

        # ---------------------------------------------------------
        # STEP 6: Final Render (Coding Path Only)
        # ---------------------------------------------------------
        status_msg.content = (
            f"## Final System Generation\n"
            f"Executed using the **{agent_name.upper()}** pipeline branch.\n\n"
            f"---\n"
            f"### 📋 System Blueprint\n{plan_output}\n\n"
            f"---\n"
            f"### 🚀 Final Delivery\n{final_delivery}\n\n"
            f"---\n"
            f"### 🔍 Critic Evaluation Matrix\n{critique}"
        )
        await status_msg.update()

        await cl.make_async(summarize_interaction)(
            agents["summarizer"], agent_name, user_input, final_delivery
        )

    except Exception as e:
        status_msg.content = f"❌ **Pipeline Assembly Halted due to Error**\n\n```text\n{e}\n```"
        await status_msg.update()

    finally:
        # Prevent VRAM fragments from polluting future inference steps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()