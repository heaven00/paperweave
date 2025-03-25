import pytest
from unittest.mock import patch, MagicMock
from src.paperweave.graph.utterance_and_interaction import (
    modify_sections_questions,
    get_utterance_graph,
)
from src.paperweave.data_type import MyState, Section, Podcast, Persona, Paper


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
    with patch('paperweave.flow_elements.flows.get_modified_sections_questions') as mock:
        mock.return_value = MagicMock(
            sections=[
                Section(section_subject="Section 3", questions=["Q3"])
            ]
        )
        new_state = modify_sections_questions(state)

    # Check updated sections
    assert len(new_state.podcast.sections) == 2  # Original + new
    assert new_state.sections == ["Section 1", "Section 3"]


def test_utterance_graph_transitions():
    graph = get_utterance_graph()

    # Check main path
    assert graph.has_edge("expert", "human")

    # Check conditional edge for "user_directive"
    edges = graph.get_edges("sentence_type")
    assert any(edge.to == "modify_flow" for edge in edges)

    # Check loop condition for "questions"
    loop_edge = graph.get_edge("after_human")
    assert loop_edge.condition is not None
