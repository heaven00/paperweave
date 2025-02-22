from paperweave.data_type import MyState, Utterance
from paperweave.flow_elements.flows import (
    create_answer,
    get_sentence_type,
    get_modified_sections_questions,
)
from paperweave.model import get_chat_model
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Callable


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


def human_node(state: MyState) -> MyState:
    """A node for collecting user input."""
    user_input = interrupt(value="Ready for user input.")
    podcast = state["podcast"]
    if user_input:
        podcast["transcript"].append(
            Utterance(persona="user", speach=user_input, category="user_question")
        )
        state["current_question"] = user_input
    return state



def sentence_type(state: MyState) -> MyState:
    sentence_type = get_sentence_type(
        model=get_chat_model(),
        sentence=state["podcast"]["transcript"][-1]["speach"],
    )
    state["podcast"]["transcript"][-1]["category"] = sentence_type
    return state


def section_type_condition() -> Callable:
    def should_change_flow(state: MyState) -> bool:
        return state["podcast"]["transcript"][-1]["category"] == "user_directive"

    return should_change_flow


class ModifySectionsQuestions:
    def __init__(
        self,
        nb_section=5,
        nb_question_per_section=2,
        podcast_tech_level="expert",
        sentence="",
    ):
        self.nb_section = nb_section
        self.nb_question_per_section = nb_question_per_section
        self.podcast_tech_level = podcast_tech_level
        self.model = get_chat_model()
        self.sentence = sentence

    def __call__(self, state: MyState) -> MyState:
        podcast = state["podcast"]
        paper = podcast["paper"]
        index_section = state["index_section"]
        previous_sections = state["sections"][: index_section + 1]  # previous+current
        modify_section_and_question = get_modified_sections_questions(
            model=self.model,
            paper_title=paper["title"],
            podcast_tech_level=self.podcast_tech_level,
            paper=paper["text"],
            nb_sections=self.nb_section - (index_section + 1),
            nb_questions_per_section=self.nb_question_per_section,
            previous_sections=previous_sections,
            sentence=self.sentence,
        )
        # update the podcast.sections and sections fields and the index_section
        state["podcast"]["sections"] = state["podcast"]["sections"][: index_section + 1]
        state["podcast"]["sections"] += modify_section_and_question.model_dump()[
            "sections"
        ]
        print("initial sections:", state["sections"])
        state["sections"] = [
            section["section_subject"] for section in state["podcast"]["sections"]
        ]
        print("updated sections", state["sections"])
        state["index_section"] += 1
        return state


def get_utterance_graph(podcast_tech_level: str = "expert"):
    checkpointer = MemorySaver()
    builder = StateGraph(MyState, input=MyState, output=MyState)
    builder.add_node("host", HostUttterance(podcast_tech_level=podcast_tech_level))
    builder.add_node("expert", ExpertUttterance(podcast_tech_level=podcast_tech_level))
    builder.add_node("human", human_node)
    builder.add_node("sentence_type", sentence_type)
    builder.add_node("modify_flow", ModifySectionsQuestions)

    builder.add_edge(START, "host")
    builder.add_edge("host", "expert")
    builder.add_edge("expert", "human")
    builder.add_edge("human","sentence_type")
    builder.add_conditional_edges(
        "sentence_type",
        section_type_condition(),
        {True: "modify_flow", False: "expert"},
    )
    builder.add_edge("modify_flow", "host")

    graph = builder.compile(checkpointer=checkpointer)
    return graph
