import pytest
from unittest.mock import patch, MagicMock
from src.paperweave.graph.get_question import HostQuestionAgent
from paperweave.data_type_direct_llm_call import QuestionChoice


def test_host_question_agent_route():
    agent = HostQuestionAgent()
    state = MyState(questions=["Q1", "Q2"])

    with patch('paperweave.flow_elements.flows.get_question_choice') as mock:
        mock.return_value = QuestionChoice.NEW_QUESTION
        result = agent(state)
        assert result.goto == "generate_new_question"
