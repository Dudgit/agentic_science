import os
import gc
import json
import torch
import chainlit as cl
import re
import asyncio
from dotenv import load_dotenv

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data import get_data_layer
from chainlit.types import ThreadDict
from chainlit.input_widget import TextInput

from agents.base import BaseAgent
from agents.coding import CodingAgent
from backends.vllm_backend import VLLMBackend
from agents.researcher import ResearcherAgent

from utils.io import load_json, update_json, update_summary_md, strip_thoughts
from utils.parser import parse_json_response
from config import PATHS, CURRENT_MODEL, MODELS

load_dotenv()

# ---------------- CONFIGURATION ----------------
DEFAULT_AGENT = "coding"
PROJECT_MEMORY = os.path.join(PATHS["projects_dir"], "project_memory.json")
PROJECT_PROFILE = os.path.join(PATHS["projects_dir"], "project_profile.json")
SUMMARY_MD = os.path.join(PATHS["projects_dir"], "summarized.md")
REPO_PATH = "/v/ijziqt/agentic_science/sample_project"  

# ---------------- PERSISTENCE LAYER ----------------
# Run `sqlite3 chainlit_chats.db < schema.sql` once before first launch.
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo="sqlite+aiosqlite:///./chainlit_chats.db"
    )

# ---------------- AUTH ----------------
# Replace this with real credential checking before sharing the app with others.
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username == "researcher" and password == os.environ.get("APP_PASSWORD", "changeme"):
        return cl.User(identifier="researcher")
    return None

# ---------------- AGENTS ----------------

agents = {
    "qa": BaseAgent(
        name="qa",
        backend=VLLMBackend(base_url=MODELS["qa"]["base_url"], model=MODELS["qa"]["model"]),
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
        backend=VLLMBackend(base_url=MODELS["writing"]["base_url"], model=MODELS["writing"]["model"]),
        prompt_dir="prompts/writing",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_writing.json")
    ),

    "summarizer": BaseAgent(
        name="summarizer",
        backend=VLLMBackend(base_url=MODELS["summarizer"]["base_url"], model=MODELS["summarizer"]["model"]),
        prompt_dir="prompts/summarizer",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_summarizer.json")
    ),

    "critic": BaseAgent(
        name="critic",
        backend=VLLMBackend(base_url=MODELS["critic"]["base_url"], model=MODELS["critic"]["model"]),
        prompt_dir="prompts/critic",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_critic.json")
    ),
    "researcher": ResearcherAgent(
        name="researcher",
        backend=VLLMBackend(base_url=MODELS["critic"]["base_url"], model=MODELS["critic"]["model"]),
        prompt_dir="prompts/researcher",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_researcher.json")
    )
}

chosable_agents = {"qa": agents["qa"], "writing": agents['writing']}
# ---------------- INTERFACE CALLBACKS ----------------

@cl.on_shared_thread_view
async def on_shared_thread_view(thread: ThreadDict, viewer: cl.User | None) -> bool:

   #Required for the native 'Share' button (top-right of a thread) to work.
   #Return True to let anyone with the link view a shared thread read-only.
    
    return True


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
    summary = summarizer_agent.run(summary_prompt)

    try:
        summary_json = parse_json_response(summary)
        update_json(PROJECT_MEMORY, summary_json)

        markdown_content = "# Current Project State\n\n"
        for key, items in summary_json.items():
            if items:
                markdown_content += f"### {key.replace('_', ' ').title()}\n"
                for item in items:
                    markdown_content += f"- {item}\n"
                markdown_content += "\n"

        update_summary_md(SUMMARY_MD, markdown_content)
        print(f"\n[System] Project Memory & summarized.md updated by {agent_name} interaction.")

    except Exception as e:
        print(f"Could not parse summary: {e}")

#----------------------- PROJECT PROFILE (Before starting the chat)----------------
def load_project_profile() -> dict:
    """Loads existing project profile if present."""
    if os.path.exists(PROJECT_PROFILE):
        try:
            with open(PROJECT_PROFILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_project_profile(profile_data: dict):
    """Saves profile to disk and session."""
    os.makedirs(os.path.dirname(PROJECT_PROFILE), exist_ok=True)
    with open(PROJECT_PROFILE, "w") as f:
        json.dump(profile_data, f, indent=4)


# ---------------- PROJECT INITIALIZATION MODAL ----------------
async def show_project_setup_modal(profile: dict = None):
    """Renders structured input boxes for each project attribute."""
    profile = profile or {}
    
    settings = await cl.ChatSettings([
        TextInput(
            id="project_name",
            label="Project Name",
            initial=profile.get("project_name", "Autonomous Science Experiment"),
            description="Short title for your project"
        ),
        TextInput(
            id="project_goal",
            label="Project Objective & Core Goal",
            initial=profile.get("project_goal", "Benchmark multi-agent reasoning and verification"),
            description="The core problem or hypothesis to solve"
        ),
        TextInput(
            id="tech_stack",
            label="Target Tech Stack & Libraries",
            initial=profile.get("tech_stack", "Python, PyTorch, vLLM"),
            description="Frameworks, environments, or databases targeted"
        ),
        TextInput(
            id="constraints",
            label="Deliverable Target & Constraints",
            initial=profile.get("constraints", "Modular scripts, PEP8, production-ready"),
            description="Operational constraints or formatting standards"
        )
    ]).send()
    return settings

#----------------------- PROJECT QUESTIONNAIRE ----------------
async def init_project_settings(profile: dict = None):
    """Renders all project questions in a single structured configuration form."""
    profile = profile or {}
    
    settings = await cl.ChatSettings([
        TextInput(
            id="project_name",
            label="Project Name",
            initial=profile.get("project_name", "Autonomous Science Project"),
            description="Short name or title for this workspace"
        ),
        TextInput(
            id="project_goal",
            label="Project Objective & Core Goal",
            initial=profile.get("project_goal", ""),
            description="What problem or feature are we building/researching?"
        ),
        TextInput(
            id="tech_stack",
            label="Target Tech Stack & Libraries",
            initial=profile.get("tech_stack", "Python, PyTorch, vLLM"),
            description="Frameworks, environments, or databases targeted"
        ),
        TextInput(
            id="constraints",
            label="Deliverable Target & Constraints",
            initial=profile.get("constraints", "Modular scripts, production-ready, PEP8"),
            description="Output format, strict conventions, or operational boundaries"
        )
    ]).send()
    
    return settings

#----------------------- User Questionnaire Flow ----------------
@cl.on_settings_update
async def on_settings_update(settings):
    profile = {
        "project_name": settings.get("project_name", "New Project").strip(),
        "project_goal": settings.get("project_goal", "General Workspace").strip(),
        "tech_stack": settings.get("tech_stack", "Standard Python").strip(),
        "constraints": settings.get("constraints", "None").strip(),
    }
    save_project_profile(profile)
    cl.user_session.set("project_profile", profile)

    already_initialized = cl.user_session.get("project_initialized", False)

    if not already_initialized:
        # First-time setup: confirm profile AND show the ready-to-chat screen together
        cl.user_session.set("project_initialized", True)
        agent_buttons = [
            cl.Action(name="switch_agent", payload={"agent": name}, label=name.capitalize())
            for name in agents.keys()
        ]
        await cl.Message(
            content=(
                f"✅ **Project Scope Configured — Ready to chat!**\n\n"
                f"| Field | Value |\n"
                f"|---|---|\n"
                f"| **Project Name** | {profile['project_name']} |\n"
                f"| **Objective** | {profile['project_goal']} |\n"
                f"| **Tech Stack** | `{profile['tech_stack']}` |\n"
                f"| **Constraints** | {profile['constraints']} |\n\n"
                f"**Active Agent:** {DEFAULT_AGENT.capitalize()}\n\n"
                f"Click a button below if you want to switch your active agent:"
            ),
            actions=agent_buttons
        ).send()
    else:
        # Later edits (mid-chat /reset_profile) just confirm the update, no re-welcome
        await cl.Message(
            content=(
                f"✅ **Project Scope Updated**\n\n"
                f"| Field | Value |\n"
                f"|---|---|\n"
                f"| **Project Name** | {profile['project_name']} |\n"
                f"| **Objective** | {profile['project_goal']} |\n"
                f"| **Tech Stack** | `{profile['tech_stack']}` |\n"
                f"| **Constraints** | {profile['constraints']} |\n"
            )
        ).send()

# ---------------- CHAINLIT UI ----------------

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent_name", DEFAULT_AGENT)
    cl.user_session.set("project_initialized", False)

    defaults = {
        "project_name": "Autonomous Science Experiment",
        "project_goal": "Benchmark multi-agent reasoning and verification",
        "tech_stack": "Python, PyTorch, vLLM",
        "constraints": "Modular scripts, PEP8, production-ready",
    }

    # Blocks here until the user presses "Confirm & Start"
    res = await cl.AskElementMessage(
        content="## 🎯 Project Setup\nFill in your project details to begin.",
        element=cl.CustomElement(name="ProjectForm", props=defaults),
        timeout=1800,
    ).send()

    profile = {**defaults, **(res or {})}   # falls back to defaults on timeout
    save_project_profile(profile)
    cl.user_session.set("project_profile", profile)
    cl.user_session.set("project_initialized", True)

    # Keep the ⚙️ panel for later edits, prefilled with what they just entered
    await show_project_setup_modal(profile)

    agent_buttons = [
        cl.Action(name="switch_agent", payload={"agent": n}, label=n.capitalize())
        for n in agents.keys()
    ]
    await cl.Message(
        content=(
            f"✅ **{profile['project_name']}** is ready.\n\n"
            f"- **Goal:** {profile['project_goal']}\n"
            f"- **Stack:** `{profile['tech_stack']}`\n"
            f"- **Constraints:** {profile['constraints']}\n\n"
            f"Active agent: **{DEFAULT_AGENT.capitalize()}**. Switch below or just start typing."
        ),
        actions=agent_buttons,
    ).send()

#-----------------  CHAT RESUME ----------------
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("agent_name", DEFAULT_AGENT)

    # Restore profile if saved on thread, or fall back to general
    thread_meta = thread.get("metadata") or {}
    profile = thread_meta.get("project_profile", {
        "project_name": thread.get("name") or "Resumed Project",
        "project_goal": "Resumed Session",
        "tech_stack": "Python",
        "constraints": "None"
    })
    cl.user_session.set("project_profile", profile)

    agent_buttons = [
        cl.Action(name="switch_agent", payload={"agent": name}, label=name.capitalize())
        for name in agents.keys()
    ]
    await cl.Message(
        content=f"👋 Resumed **{thread['name'] or 'this chat'}**. Active agent: **{DEFAULT_AGENT}**.",
        actions=agent_buttons
    ).send()

# -----------------  RENAME THREAD ----------------
async def rename_thread_if_needed(user_input: str):
   
    if cl.user_session.get("thread_renamed"):
        return
    data_layer = get_data_layer()
    thread_id = cl.context.session.thread_id
    if not data_layer or not thread_id:
        return
    short_title = user_input[:60] + ("..." if len(user_input) > 60 else "")
    try:
        await data_layer.update_thread(thread_id=thread_id, name=short_title)
        cl.user_session.set("thread_renamed", True)
    except Exception as e:
        print(f"[System] Could not rename thread: {e}")


async def tag_thread_with_agent(agent_name: str):
    data_layer = get_data_layer()
    thread_id = cl.context.session.thread_id
    if not data_layer or not thread_id:
        return
    
    profile = cl.user_session.get("project_profile", {})
    try:
        await data_layer.update_thread(
            thread_id=thread_id,
            metadata={"agent": agent_name, "project_profile": profile}
        )
    except Exception as e:
        print(f"[System] Could not tag thread: {e}")

@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip()
    if user_input.lower() == "/reset":
        reset_all_memories()
        await cl.Message(content="🧹 **All memory files and agent logs reset.**").send()
        return

    await rename_thread_if_needed(user_input)

    # Command: View active project scope
    if user_input.lower() == "/profile":
        profile = cl.user_session.get("project_profile") or load_project_profile()
        msg = (
            f"### 📋 Current Project Profile\n"
            f"- **Goal**: {profile.get('project_goal', 'Not set')}\n"
            f"- **Tech Stack**: {profile.get('tech_stack', 'Not set')}\n"
            f"- **Constraints**: {profile.get('constraints', 'Not set')}\n"
        )
        await cl.Message(content=msg).send()
        return

    # Command: Reset profile
    if user_input.lower() == "/reset_profile":
        profile = cl.user_session.get("project_profile", {})
        await show_project_setup_modal(profile)
        return

    if user_input.lower().startswith("/history"):
        parts = user_input.split(maxsplit=1)
        if len(parts) < 2 or parts[1].strip().lower() not in agents:
            avail = ", ".join([f"`{k}`" for k in agents.keys()])
            await cl.Message(content=f"ℹ️ Usage: `/history <agent>`. Available: {avail}").send()
            return

        target_agent = parts[1].strip().lower()
        import aiosqlite
        rows = []
        try:
            async with aiosqlite.connect("chainlit_chats.db") as db:
                cursor = await db.execute(
                    """
                    SELECT id, name, "createdAt" FROM threads
                    WHERE json_extract(metadata, '$.agent') = ?
                    ORDER BY "createdAt" DESC
                    LIMIT 20
                    """,
                    (target_agent,),
                )
                rows = await cursor.fetchall()
        except Exception as e:
            await cl.Message(content=f"❌ Could not read history: {e}").send()
            return

        if not rows:
            await cl.Message(content=f"No past chats found for **{target_agent}** yet.").send()
            return

        lines = [f"### 🗂️ Past **{target_agent.capitalize()}** chats\n"]
        for thread_id, name, created_at in rows:
            lines.append(f"- {name or '(untitled)'}  \n  <sub>{created_at}</sub>")
        await cl.Message(content="\n".join(lines)).send()
        return

    if user_input.lower() == "/menu":
        agent_buttons = [
            cl.Action(name="switch_agent", payload={"agent": k}, label=k.capitalize())
            for k in agents.keys()
        ]
        await cl.Message(
            content="👇 Click a button to switch your active agent:",
            actions=agent_buttons
        ).send()
        return

    agent_name = cl.user_session.get("agent_name", DEFAULT_AGENT)
    chosen_agent = agents[agent_name]
    await tag_thread_with_agent(agent_name)

    planner_agent = agents.get("planner", agents["qa"])
    critic_agent = agents["critic"]

    status_msg = cl.Message(content="🤖 Initializing Agentic Workflow Assembly Line...")
    await status_msg.send()

    # --- INJECT PROJECT CONTEXT HEADER ---
    profile = cl.user_session.get("project_profile") or load_project_profile()
    project_context_header = (
        f"### GLOBAL PROJECT BOUNDARIES & SPECIFICATIONS ###\n"
        f"Project Name: {profile.get('project_name', 'General Workspace')}\n"
        f"Project Goal: {profile.get('project_goal', 'General Workspace')}\n"
        f"Target Tech Stack: {profile.get('tech_stack', 'Standard Python')}\n"
        f"Hard Constraints: {profile.get('constraints', 'None')}\n"
        f"----------------------------------------------------\n\n"
    )
    try:
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
        await asyncio.sleep(0.15)

        # Prepend the context header so the plan adheres to the tech stack and goals
        planner_prompt = (
            f"{project_context_header}"
            f"Break down the user's request into a concrete engineering plan or structural outline conforming to the global project specifications.\n"
            f"If you need external documentation, libraries, or recent data to complete this plan accurately, "
            f"you MUST start your response with: SEARCH: [your search query].\n\n"
            f"User Request: {user_input}"
        )
        raw_plan = await cl.make_async(planner_agent.run)(planner_prompt)
        plan_output = strip_thoughts(raw_plan)

        # STEP 2: Researcher (if needed)
        research_context = ""
        if "SEARCH:" in plan_output:
            status_msg.content = "🌐 **Phase 2: Executing Autonomous Live Web Search...**"
            await status_msg.update()
            await asyncio.sleep(0.15)

            search_query = plan_output.split("SEARCH:")[1].strip()

            from utils.web import search_internet
            web_raw = search_internet(search_query)

            status_msg.content = "📊 **Phase 2b: Integrating Web Findings into System Blueprint...**"
            await status_msg.update()
            await asyncio.sleep(0.15)

            plan_output = await cl.make_async(planner_agent.run)(
                f"{project_context_header}"
                f"We performed a search for '{search_query}'. Here are the findings:\n\n{web_raw}\n\n"
                f"Rewrite your structural engineering blueprint incorporating these new constraints and facts."
            )
            plan_output = strip_thoughts(raw_plan)

        # STEP 3: Chosen agent
        status_msg.content = f"⚙️ **Phase 3: Generating Base Content ({agent_name.capitalize()} Agent)...**"
        await status_msg.update()
        await asyncio.sleep(0.15)

        # Prepend the context header so the generator uses the exact libraries and conventions
        execution_prompt = (
            f"{project_context_header}"
            f"### SYSTEM INSTRUCTION ###\n"
            f"Implement strictly adhering to the target tech stack and project constraints.\n\n"
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
        await asyncio.sleep(0.15)

        await cl.make_async(summarize_interaction)(
            agents["summarizer"], agent_name, user_input, final_delivery
        )

    except Exception as e:
        status_msg.content = f"❌ **Pipeline Assembly Halted due to Error**\n\n```text\n{e}\n```"
        await status_msg.update()
        await asyncio.sleep(0.15)

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()