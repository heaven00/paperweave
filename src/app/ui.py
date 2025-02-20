import chainlit as cl


@cl.on_chat_start
async def start():
    # Display a message to the user
    await cl.Message(content="Welcome to the PaperWeave App!").send()

    # Create a text input for arXiv identifier
    arxiv_id = await cl.AskUserMessage(
        content="Please enter an arXiv identifier (e.g., arXiv:1501.00001 or arXiv:0706.0001):",
    ).send()

    if arxiv_id:
        print(arxiv_id)
        # Store the arXiv identifier in the session state
        cl.user_session.set("arxiv_id", arxiv_id["output"])


@cl.on_message
async def handle_message(message):
    arxiv_id = cl.user_session.get("arxiv_id")

    if arxiv_id:
        # Fetch the paper details using the arXiv identifier        # Display the title and summary of the paper
        await cl.Message(content=f"Paper Code: {arxiv_id}").send()
    else:
        await cl.Message(
            content="No arXiv ID found. Please enter a valid arXiv identifier."
        ).send()
