import pytest
from unittest.mock import AsyncMock, patch
from src.app.ui import is_valid_arxiv_id, handle_message
import chainlit as cl

# Test regex validation function
def test_is_valid_arxiv_id():
    assert is_valid_arxiv_id("arXiv:1706.03762v7") == "1706.03762v7"
    assert is_valid_arxiv_id("https://arxiv.org/abs/1810.04805v2") == "1810.04805v2"
    assert is_valid_arxiv_id("invalid") is None
    assert is_valid_arxiv_id("1234.56789v10") == "1234.56789v10"

# Async test for message handling with valid Arxiv ID
@pytest.mark.asyncio
async def test_handle_message_valid_arxiv():
    mock_session = {
        "link_provided": False,
        "arxiv_id": None
    }
    with patch("src.app.ui.cl.user_session", mock_session):
        mock_msg = cl.Message(content="Test 1706.03762v7")
        await handle_message(mock_msg)
        
    assert mock_session["arxiv_id"] == "1706.03762v7"
    assert mock_session["link_provided"] is True

# Async test for message handling with invalid Arxiv ID
@pytest.mark.asyncio
async def test_handle_message_invalid_arxiv():
    mock_session = {"link_provided": False}
    with patch("src.app.ui.cl.user_session", mock_session), \
         patch("src.app.ui.cl.Message.send", new=AsyncMock()) as mock_send:
          
        await handle_message(cl.Message(content="invalid"))
        mock_send.assert_awaited_with(
            "⚠️ Invalid Arxiv format. Please provide a valid link or ID (e.g., '1706.03762v7').",
            author="PaperWeave"
        )

# Async test for message handling when Arxiv ID is already provided
@pytest.mark.asyncio
async def test_handle_message_already_provided():
    mock_session = {"link_provided": True}
    with patch("src.app.ui.cl.user_session", mock_session), \
         patch("src.app.ui.process_user_query", new=AsyncMock()) as mock_process:
         
        await handle_message(cl.Message(content="test"))
        mock_process.assert_awaited_once_with(cl.Message(content="test"), "arxiv_id")
