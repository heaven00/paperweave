from langgraph.types import Command
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv
from pathlib import Path
import os
from typing import Literal

from paperweave.data_type_direct_llm_call import QuestionChoice
from paperweave.transforms import extract_list, transcript_to_full_text
from paperweave.flow_elements.flows import (
    get_question_choice,
    get_follow_question,
    get_next_question_index,
    reformulate_question,
)
from paperweave.data_type import Utterance


from paperweave.data_type import MyState
from paperweave.model import get_chat_model


class HostQuestionAgent:
    def __init__(self, tech_level: str = "expert"):
        self.model = get_chat_model()
        self.tech_level = tech_level

    def __call__(
        self, state: MyState
    ) -> Command[
        Literal[
            "end_section_questions",
            "take_first_question",
            "generate_new_question",
            "take_any_question",
        ]
    ]:
        transcript = transcript_to_full_text(state["podcast"]["transcript"])
        questions = state["questions"]
        response = get_question_choice(
            model=self.model,
            paper_title="Attention is all you need",
            podcast_tech_level=self.tech_level,
            transcript=transcript,
            questions=questions,
        )

        goto = {
            QuestionChoice.FIRST_QUESTION: "take_first_question",
            QuestionChoice.NEW_QUESTION: "generate_new_question",
            QuestionChoice.OTHER_QUESTION: "take_any_question",
            QuestionChoice.NO_QUESTION: "end_section_questions",
        }
        response = goto.get(response.question_choice, QuestionChoice.FIRST_QUESTION)

        return Command(
            # Specify which agent to call next
            goto=response
        )


# Node 1: Take the first element from list_A and add it to list_B
def take_first_question(state: MyState) -> MyState:
    if state["questions"]:
        state["current_question"] = state["questions"].pop(0)
    return state


class GenerateNewQuestion:
    def __init__(self, tech_level: str = "expert"):
        self.model = get_chat_model()
        self.tech_level = tech_level

    def __call__(self, state: MyState) -> MyState:
        podcast = state["podcast"]
        transcript = podcast["transcript"]
        paper_title = podcast["paper"]["title"]
        question = get_follow_question(
            model=self.model,
            paper_title=paper_title,
            podcast_tech_level=self.tech_level,
            transcript=transcript,
        )
        state["current_question"] = question
        return state


class TakeAnyQuestion:
    def __init__(self, tech_level: str = "expert"):
        self.model = get_chat_model()
        self.tech_level = tech_level

    def __call__(self, state: MyState) -> MyState:
        podcast = state["podcast"]
        transcript = podcast["transcript"]
        paper_title = podcast["paper"]["title"]
        questions = state["questions"]
        question_index = get_next_question_index(
            model=self.model,
            paper_title=paper_title,
            podcast_tech_level=self.tech_level,
            transcript=transcript,
            questions=questions,
        )
        current_question = ""
        if question_index < len(state["questions"]):
            current_question = state["questions"].pop(question_index)
        state["current_question"] = current_question
        return state


class ReformulateQuestion:
    def __init__(self, tech_level: str = "expert"):
        self.model = get_chat_model()
        self.tech_level = tech_level

    def __call__(self, state: MyState) -> MyState:
        question = state["current_question"]
        podcast = state["podcast"]
        transcript = podcast["transcript"]
        paper_title = podcast["paper"]["title"]
        reformulate_question_str = reformulate_question(
            model=self.model,
            paper_title=paper_title,
            podcast_tech_level=self.tech_level,
            transcript=transcript,
            question=question,
        )
        state["current_question"] = reformulate_question_str

        podcast["transcript"].append(
            Utterance(persona=podcast["host"], speach=question, category="question")
        )
        return state


def end_section_questions(state: MyState) -> MyState:
    state["current_question"] = ""
    state["questions"] = []
    return state


def obtain_get_question_graph(podcast_tech_level: str = "expert"):
    # Define the workflow graph
    builder = StateGraph(MyState)
    builder.add_node("take_first_question", take_first_question)
    builder.add_node(
        "generate_new_question", GenerateNewQuestion(tech_level=podcast_tech_level)
    )
    builder.add_node(
        "take_any_question", TakeAnyQuestion(tech_level=podcast_tech_level)
    )
    builder.add_node(
        "reformulate_question", ReformulateQuestion(tech_level=podcast_tech_level)
    )
    builder.add_node("end_section_questions", end_section_questions)
    builder.add_node(
        "host_question_agent", HostQuestionAgent(tech_level=podcast_tech_level)
    )

    # Define the execution flow
    builder.add_edge(START, "host_question_agent")
    builder.add_edge("take_first_question", "reformulate_question")
    builder.add_edge("generate_new_question", "reformulate_question")
    builder.add_edge("take_any_question", "reformulate_question")
    builder.add_edge("end_section_questions", END)
    builder.add_edge("reformulate_question", END)

    graph = builder.compile()
    return graph


if __name__ == "__main__":
    env_file = Path(__file__).parent.parent / ".env"

    # Load the .env file
    load_dotenv(env_file)

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY") or ""

    graph = obtain_get_question_graph()

    graph_as_image = graph.get_graph().draw_mermaid_png()
    with open("image.png", "wb") as f:
        f.write(graph_as_image)

    # Initialize the state
    initial_state = {
        "list_A": ["Entry A1", "Entry A2", "Entry A3"],
        "list_B": [],
        "message_history": [{"role": "user", "content": "Hello"}],
    }

    # Run the graph
    final_state = graph.invoke(initial_state)

    # Output the resulting list_B
    print("Final list B:", final_state["list_B"])
