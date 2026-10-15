from main_lg import build_pipeline
import chainlit as cl
from utils.io import reset_all_memories

app = build_pipeline()

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=f"""
# ELTE Scientific AI Assistant platform
Welcome!
Please explain what project you are working on and we will try to assist you!
""").send()

@cl.action_callback("reset_memory_action")
async def on_reset_memory(action: cl.Action):
    reset_all_memories()
    await cl.Message(
        content="🧹 **All project memory files, agent logs, and context history have been wiped clean!**"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    status_msg = cl.Message(content="⚙️ **Running Agent Pipeline...**")
    #past_context = cl.user_session.get("restored_context", "")
    await status_msg.send()
    # Build the initial state dictionary
    initial_state = {
        "user_query": message.content.strip(),
        "project_context": "", # Load your questionnaire JSON
        "current_draft": "",
        "critique": None,
        "iteration": 0,
        "max_iterations": 2,
        "is_approved": False,
        "final_output": ""
    }

    try:
        # Launch the LangGraph pipeline asynchronously
        final_state = await app.ainvoke(initial_state)
        
        # Output directly to Chainlit
        status_msg.content = final_state["final_output"]
        await status_msg.update()

    except Exception as e:
        status_msg.content = f"❌ Pipeline Crashed:\n```\n{e}\n```"
        await status_msg.update()