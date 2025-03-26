from unittest.mock import patch, MagicMock
from paperweave.graph.utterance_and_interaction import (
    ExpertUtterance,
    modify_sections_questions,
    get_utterance_graph,
)
from paperweave.data_type import (
    MyState,
    Section,
    Podcast,
    Persona,
    Paper,
    Utterance,
)


def test_modify_sections_questions():
    # Setup mock state
    initial_sections = [
        Section(section_subject="Section 1", questions=["Q1"]),
        Section(section_subject="Section 2", questions=["Q2"]),
    ]
    podcast = Podcast(
        paper=Paper(text="mock_text", title="Mock Title"),
        sections=initial_sections,
        transcript=[Utterance(speach="Modify this", category="user_directive")],
    )
    state = MyState(
        podcast=podcast,
        index_section=1,
        sections=[s.section_subject for s in initial_sections],
    )

    # Mock LLM response
    with patch(
        "paperweave.graph.utterance_and_interaction.get_modified_sections_questions"
    ) as mock:
        mock.return_value = MagicMock(
            sections=[Section(section_subject="Section 3", questions=["Q3"])]
        )
        new_state = modify_sections_questions(state)

    # Check updated sections
    assert len(new_state.podcast.sections) == 2  # Original + new
    assert new_state.sections == ["Section 1", "Section 3"]


def test_utterance_graph_compiles():
    graph = get_utterance_graph().get_graph()
    assert graph


def test_expert_utterance_empty_transcript():
    podcast = Podcast(
        paper=Paper(text="mock_text", title="Mock Title"),
        transcript=[],
        expert=Persona(name="Expert"),
    )
    state = MyState(podcast=podcast, current_question="Q1")

    expert = ExpertUtterance()
    with patch("paperweave.graph.utterance_and_interaction.create_answer") as mock_create_answer:
        mock_create_answer.return_value = "Mock Answer"
        new_state = expert(state)

    assert new_state.podcast.transcript[-1].speach == "Mock Answer"
