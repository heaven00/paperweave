from unittest.mock import patch, MagicMock
from paperweave.graph.extraction_and_intro import GetSectionsAndQuestions
from paperweave.data_type import MyState, Paper, Podcast, Section


def test_get_sections_and_questions():
    get_sections = GetSectionsAndQuestions()
    state = MyState(podcast=Podcast(paper=Paper(text="mock_text", title="Mock Title")))

    with patch("paperweave.graph.extraction_and_intro.get_sections_questions") as mock:
        mock.return_value = MagicMock(
            sections=[Section(section_subject="Section 1", questions=["Q1"])]
        )
        new_state = get_sections(state)

    assert new_state.podcast.sections[0].section_subject == "Section 1"
    assert new_state.sections == ["Section 1"]
