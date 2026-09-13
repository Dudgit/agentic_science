import os
import gc
import json
import torch
import chainlit as cl
import re
import asyncio

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data import get_data_layer
from chainlit.types import ThreadDict

from agents.base import BaseAgent
from agents.coding import CodingAgent
from backends.vllm_backend import VLLMBackend
from agents.researcher import ResearcherAgent

from utils.io import load_json, update_json, update_summary_md, strip_thoughts
from utils.parser import parse_json_response
from config import PATHS, CURRENT_MODEL, MODELS

# ---------------- CONFIGURATION ----------------
DEFAULT_AGENT = "coding"
PROJECT_MEMORY = os.path.join(PATHS["projects_dir"], "project_memory.json")
SUMMARY_MD = os.path.join(PATHS["projects_dir"], "summarized.md")
REPO_PATH = "/v/ijziqt/agentic_science/sample_project"  # Path for the CodingAgent X-Ray vision

# ---------------- PERSISTENCE LAYER ----------------
# This is what makes Chainlit remember and LIST previous chats in the sidebar.
# Run `sqlite3 chainlit_chats.db < schema.sql` once before first launch.
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo="sqlite+aiosqlite:///./chainlit_chats.db"
    )

# ---------------- AUTH ----------------
# Chainlit only lists/persists chat history per identified user.
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

    "coding": CodingAgent(
        name="coding",
        backend=VLLMBackend(
            base_url=MODELS["coding"]["base_url"],
            model=MODELS["coding"]["model"],
            max_tokens=MODELS["coding"].get("max_tokens", 2048),
            temperature=MODELS["coding"].get("temperature", 0.0)
        ),
        prompt_dir="prompts/coding",
        memory_file=os.path.join(PATHS["projects_dir"], "memory_coding.json"),
        repo_path=REPO_PATH  # Required for your custom coding agent
    ),

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

# ---------------- INTERFACE CALLBACKS ----------------

@cl.on_shared_thread_view
async def on_shared_thread_view(thread: ThreadDict, viewer: cl.User | None) -> bool:
    """
    Required for the native 'Share' button (top-right of a thread) to work.
    Return True to let anyone with the link view a shared thread read-only.
    Tighten this later (e.g. check viewer against an allowlist) if needed.
    """
    return True


@cl.action_callback("switch_agent")
async def on_switch_agent(action: cl.Action):
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

# ---------------- CHAINLIT UI ----------------

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent_name", DEFAULT_AGENT)

    agent_buttons = [
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


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Fires when the user clicks a PREVIOUS chat in the sidebar.
    Without this, Chainlit still lists old chats, but reopening one
    would silently drop back to defaults instead of restoring state.
    """
    cl.user_session.set("agent_name", DEFAULT_AGENT)

    agent_buttons = [
        cl.Action(name="switch_agent", payload={"agent": name}, label=name.capitalize())
        for name in agents.keys()
    ]
    await cl.Message(
        content=f"👋 Resumed **{thread['name'] or 'this chat'}**. Active agent: **{DEFAULT_AGENT}**.",
        actions=agent_buttons
    ).send()


async def rename_thread_if_needed(user_input: str):
    """
    Chainlit names a thread after whichever interaction happens first in the
    session. Since on_chat_start immediately shows agent-switch buttons,
    clicking one becomes 'the first interaction' and the thread gets stuck
    named after the action (e.g. 'switch_agent') forever. This overrides
    that name using the first *real* text message instead, the first time
    it happens in this session.
    """
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
    """
    Implements agent-based grouping: records which agent handled this
    thread, so history can be filtered by agent later (see the /history
    command below).

    NOTE: Chainlit's `tags` column is designed for Postgres's native array
    type. On SQLite it silently fails (Chainlit passes a raw Python list
    straight to the SQLite driver, which can't bind list values at all -
    confirmed directly against this exact code path). `metadata` is JSON
    -encoded before being stored, so it works correctly on SQLite. We use
    metadata instead for that reason.
    """
    data_layer = get_data_layer()
    thread_id = cl.context.session.thread_id
    if not data_layer or not thread_id:
        return
    try:
        await data_layer.update_thread(thread_id=thread_id, metadata={"agent": agent_name})
    except Exception as e:
        print(f"[System] Could not tag thread: {e}")


@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip()

    await rename_thread_if_needed(user_input)

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

    try:
        # STEP 1: Planner
        status_msg.content = " **Phase 1: Architecting System Plan (Planner)...**"
        await status_msg.update()
        await asyncio.sleep(0.15)

        planner_prompt = (
            f"Break down the user's request into a concrete engineering plan or structural outline.\n"
            f"If you need external documentation, libraries, or recent data to complete this plan accurately, "
            f"you MUST start your response with: SEARCH: [your search query].\n\n"
            f"User Request: {user_input}"
        )
        plan_output = await cl.make_async(planner_agent.run)(planner_prompt)

        # STEP 2: Researcher (if needed)
        research_context = ""
        if "SEARCH:" in plan_output:
            status_msg.content = "🌐 **Phase 2: Executing Autonomous Live Web Search (Researcher)...**"
            await status_msg.update()
            await asyncio.sleep(0.15)

            search_query = plan_output.split("SEARCH:")[1].strip()

            from utils.web import search_internet
            web_raw = search_internet(search_query)

            status_msg.content = "📊 **Phase 2b: Integrating Web Findings into System Blueprint...**"
            await status_msg.update()
            await asyncio.sleep(0.15)

            plan_output = await cl.make_async(planner_agent.run)(
                f"We performed a search for '{search_query}'. Here are the findings:\n\n{web_raw}\n\n"
                f"Rewrite your structural engineering blueprint incorporating these new constraints and facts."
            )

        # STEP 3: Chosen agent
        status_msg.content = f"⚙️ **Phase 3: Generating Base Content ({agent_name.capitalize()} Agent)...**"
        await status_msg.update()
        await asyncio.sleep(0.15)

        execution_prompt = (
            f"### SYSTEM INSTRUCTION ###\n"
            f"### BLUEPRINT TO IMPLEMENT ###\n{plan_output}"
        )
        raw_delivery = await cl.make_async(chosen_agent.run)(execution_prompt)
        initial_delivery = strip_thoughts(raw_delivery)

        # STEP 4: Critic
        status_msg.content = "🔍 **Phase 4: Evaluating Output Stability & Quality (Critic)...**"
        await status_msg.update()
        await asyncio.sleep(0.15)

        critic_prompt = (
            f"### TARGET REQUIREMENT ###\n{user_input}\n\n"
            f"### ASSIGNED BLUEPRINT ###\n{plan_output}\n\n"
            f"### RENDERED DELIVERY ###\n{initial_delivery}"
        )
        critique = await cl.make_async(critic_agent.run)(critic_prompt)

        # STEP 5: Final fix
        status_msg.content = f"🛠️ **Phase 5: Executing Final Corrections ({agent_name.capitalize()} Agent)...**"
        await status_msg.update()
        await asyncio.sleep(0.15)

        correction_prompt = (
            f"### SYSTEM INSTRUCTION ###\n"
            f"### PREVIOUS RESPONSE ###\n{initial_delivery}\n\n"
            f"### CRITIC FEEDBACK TO RESOLVE ###\n{critique}"
        )
        final_delivery = await cl.make_async(chosen_agent.run)(correction_prompt)

        # STEP 6: Final render
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
            summarizer_agent=agents["summarizer"],
            agent_name=agent_name,
            user_input=user_input,
            agent_response=final_delivery
        )

    except Exception as e:
        status_msg.content = f"❌ **Pipeline Assembly Halted due to Error**\n\n```text\n{e}\n```"
        await status_msg.update()
        await asyncio.sleep(0.15)

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()