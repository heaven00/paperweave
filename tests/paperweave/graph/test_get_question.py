from unittest.mock import patch
from paperweave.graph.get_question import HostQuestionAgent
from paperweave.data_type_direct_llm_call import LLMResponseQuestionChoice, QuestionChoice
from paperweave.data_type import MyState, Paper, Podcast


def test_host_question_agent_route():
    agent = HostQuestionAgent()
    state = MyState(podcast=Podcast(paper=Paper()), questions=["Q1", "Q2"])

    with patch("paperweave.graph.get_question.get_question_choice") as mock:
        mock.return_value = LLMResponseQuestionChoice(question_choice=QuestionChoice.NEW_QUESTION)
        result = agent(state)
        assert result.goto == "generate_new_question"
