import chainlit as cl


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    await cl.Message(
        content="Add Arxiv code of the paper to discuss",
    ).send()


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True
