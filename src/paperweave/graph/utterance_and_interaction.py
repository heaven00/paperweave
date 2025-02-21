from paperweave.data_type import MyState, Utterance
from paperweave.flow_elements.flows import create_answer
from paperweave.model import get_chat_model
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal


class HostUttterance:
    def __init__(self, podcast_tech_level: str = "expert"):
        self.model = get_chat_model()
        self.podcast_tech_level = podcast_tech_level

    def __call__(self, state: MyState):
        podcast = state["podcast"]
        question = state["questions"].pop(0)
        state["current_question"] = question
        state["questions_asked"].append(state["current_question"])

        podcast["transcript"].append(
            Utterance(persona=podcast["host"], speach=question, category="question")
        )
        return state


class ExpertUttterance:
    def __init__(self, podcast_tech_level: str = "expert"):
        self.model = get_chat_model()
        self.podcast_tech_level = podcast_tech_level

    def __call__(self, state: MyState):
        podcast = state["podcast"]
        paper_title = podcast["paper"]["title"]
        paper_text = podcast["paper"]["text"]
        question = state["current_question"]

        previous_question = "no previous question"
        previous_answer = "no previous answer"
        if podcast["transcript"]:
            if podcast["transcript"][-1]["persona"] == "expert":
                previous_answer = podcast["transcript"][-1]["speach"]
                previous_question = podcast["transcript"][-2]["speach"]

        answer = create_answer(
            model=self.model,
            paper_title=paper_title,
            podcast_tech_level=self.podcast_tech_level,
            paper=paper_text,
            previous_question=previous_question,
            previous_answer=previous_answer,
            new_question=question,
        )

        podcast["transcript"].append(
            Utterance(persona=podcast["expert"], speach=answer, category="answer")
        )

        state["index_question"] = state["index_question"] + 1
        return state


def human_node(state: MessagesState) -> Command[Literal["host"]]:
    """A node for collecting user input."""
    user_input = interrupt(value="Ready for user input.")
    podcast = state["podcast"]
    if user_input:
        podcast["transcript"].append(
                Utterance(persona="user", speach=user_input, category="user_question")
            )
    return Command(
        update=state,
        goto="host"
    )



def get_utterance_graph(podcast_tech_level: str = "expert"):
    checkpointer = MemorySaver()
    builder = StateGraph(MyState, input=MyState, output=MyState)
    builder.add_node("host", HostUttterance(podcast_tech_level=podcast_tech_level))
    builder.add_node("expert", ExpertUttterance(podcast_tech_level=podcast_tech_level))
    builder.add_node("human", human_node)

    builder.add_edge(START, "host")
    builder.add_edge("host", "expert")
    builder.add_edge("expert", "human")

    graph = builder.compile(checkpointer=checkpointer)
    return graph