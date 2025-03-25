import pytest
from src.paperweave.graph.utterance_and_interaction import loop_list_condition


def test_loop_list_condition():
    condition = loop_list_condition("index_question", "questions")
    assert condition({"index_question": 0, "questions": ["Q1", "Q2"]}) is True
    assert condition({"index_question": 2, "questions": ["Q1", "Q2"]}) is False
