from app.ui import is_valid_arxiv_id

# Test regex validation function
def test_is_valid_arxiv_id():
    assert is_valid_arxiv_id("arXiv:1706.03762v7") == "1706.03762v7"
    assert is_valid_arxiv_id("https://arxiv.org/abs/1810.04805v2") == "1810.04805v2"
    assert is_valid_arxiv_id("invalid") is None
    assert is_valid_arxiv_id("1234.56789v10") == "1234.56789v10"
