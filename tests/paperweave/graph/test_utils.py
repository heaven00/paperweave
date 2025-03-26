from paperweave.data_type import MyState, Podcast, Paper
from paperweave.graph.utterance_and_interaction import loop_list_condition


def test_loop_list_condition():
    state = MyState(
        podcast=Podcast(paper=Paper(text="mock_text", title="Mock Title")),
        index_question=0,
        questions=["q1", "q2"]
    )
    condition = loop_list_condition("index_question", "questions")
    assert condition(state) is True

    state2 = MyState(
            podcast=Podcast(paper=Paper(text="mock_text", title="Mock Title")),
            index_question=2,
            questions=["q1", "q2"]
        )
    assert condition(state2) is False
