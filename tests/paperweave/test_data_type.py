import pytest
from paperweave.data_type import MyState, Paper, Podcast


def test_my_state_validation():
    # Missing `podcast` should raise an error
    with pytest.raises(ValueError):
        #ignore
        MyState()

    # Valid state
    valid_state = MyState(
        podcast=Podcast(paper=Paper(text="mock paper")),
        index_question=0,
        index_section=0,
        questions=[],
        current_question="",
        sections=[],
    )
    assert isinstance(valid_state, MyState)
