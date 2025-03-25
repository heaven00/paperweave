import pytest
from src.paperweave.data_type import MyState, Podcast


def test_my_state_validation():
    # Missing `podcast` should raise an error
    with pytest.raises(ValueError):
        MyState()

    # Valid state
    valid_state = MyState(
        podcast=Podcast(),
        index_question=0,
        index_section=0,
        questions=[],
        current_question="",
        sections=[],
    )
    assert isinstance(valid_state, MyState)
