import re
import chainlit as cl

# Regular expression to extract valid Arxiv IDs from URLs or raw inputs
ARXIV_ID_PATTERN = re.compile(
    r"""
    \b
    (?:arXiv:)?          # Optional "arXiv:" prefix
    (\d{4}\.\d{4,5}(?:v\d+)?)  # Core ID (e.g., 1706.03762 or 1706.03762v7)
    \b
    """,
    re.VERBOSE | re.IGNORECASE
)

@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    cl.user_session.set("arxiv_id", None)  # Track validated Arxiv ID
    cl.user_session.set("link_provided", False)  # Track if valid link provided

    await cl.Message(
        content="Please provide the Arxiv link or ID (e.g., '1706.03762v7') of the paper you want to discuss.",
    ).send()

def is_valid_arxiv_id(arxiv_str: str) -> str | None:
    """Extract and validate an Arxiv ID from any input string."""
    match = ARXIV_ID_PATTERN.search(arxiv_str)
    if match:
        return match.group(1)  # Return the clean ID (e.g., '1706.03762v7')
    return None


@cl.on_message
async def handle_message(message: cl.Message):
    arxiv_id = cl.user_session.get("arxiv_id")
    link_provided = cl.user_session.get("link_provided", False)

    if not link_provided:
        # User must provide a valid Arxiv link/ID first
        user_input = message.content.strip()

        # Extract Arxiv ID from input (supports URLs, raw IDs, etc.)
        arxiv_candidate = is_valid_arxiv_id(user_input)
        if not arxiv_candidate:
            await cl.Message(
                content="⚠️ Invalid Arxiv format. Please provide a valid link or ID (e.g., '1706.03762v7').",
                author="PaperWeave",
            ).send()
            return  # Stop processing until valid input

        # Save the valid Arxiv ID and mark as provided
        cl.user_session.set("arxiv_id", arxiv_candidate)
        cl.user_session.set("link_provided", True)

        await cl.Message(
            content=f"✅ Arxiv ID '{arxiv_candidate}' accepted. You can now ask questions about the paper.",
            author="PaperWeave",
        ).send()
    else:
        # Proceed with normal processing (e.g., generate podcast content)
        await process_user_query(message, arxiv_id)

async def process_user_query(message: cl.Message, arxiv_id: str):
    user_question = message.content

    # Example: Fetch paper data and generate podcast content
    # Replace this with your actual logic (e.g., call scripts/run_v1.py)
    await cl.Message(
        content=f"Processing your question about Arxiv paper {arxiv_id}...",
        author="PaperWeave",
    ).send()
