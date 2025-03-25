import pytest
from unittest.mock import patch, MagicMock
from src.paperweave.graph.extraction_and_intro import GetSectionsAnsQuestions
from src.paperweave.data_type import MyState, Paper, Podcast, Section


def test_get_sections_ans_questions():
    get_sections = GetSectionsAnsQuestions()
    state = MyState(
        podcast=Podcast(paper=Paper(text="mock_text", title="Mock Title"))
    )

    with patch('paperweave.flow_elements.flows.get_sections_questions') as mock:
        mock.return_value = MagicMock(
            sections=[
                Section(section_subject="Section 1", questions=["Q1"])
            ]
        )
        new_state = get_sections(state)

    assert new_state.podcast.sections[0].section_subject == "Section 1"
    assert new_state.sections == ["Section 1"]
