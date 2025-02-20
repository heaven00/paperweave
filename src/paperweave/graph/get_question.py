from typing import List, Dict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv
from pathlib import Path
import os
from typing import Literal

from paperweave.data_type_direct_llm_call import QuestionChoice
from paperweave.transforms import extract_list, transcript_to_full_text
from paperweave.flow_elements.flows import get_question_choice

env_file = Path(__file__).parent.parent / ".env"

# Load the .env file
load_dotenv(env_file)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY") or ""


from paperweave.data_type import MyState
from paperweave.model import get_chat_model


class Agent:
    def __init__(self):
        self.model = get_chat_model()

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
        questions = state["question"]
        response = get_question_choice(
            model=self.model,
            paper_title="Attention is all you need",
            podcast_tech_level="expert",
            transcript=transcript,
            questions=questions,
        )

        goto = {
            QuestionChoice.FIRST_QUESTION: "take_first_question",
            QuestionChoice.NEW_QUESTION: "generate_new_question",
            QuestionChoice.OTHER_QUESTION: "take_any_question",
            QuestionChoice.NO_QUESTION: "end_section_questions",
        }
        response = QuestionChoice.NO_QUESTION
        return Command(
            # Specify which agent to call next
            goto=goto.get(response, QuestionChoice.FIRST_QUESTION)
        )


# Node 1: Take the first element from list_A and add it to list_B
def take_first_question(state: MyState) -> MyState:
    if state["questions"]:
        state["current_question"] = state["current_question"].pop(0)
    return state


# Node 2: Generate a new entry based on message history and add to list_B
def generate_new_question(state: MyState) -> MyState:
    messages = state["message_history"]
    response = llm.invoke(messages)
    state["list_B"].append(response.content)
    return state


# Node 3: Take a different element from list_A (e.g., the second) and add to list_B
def take_any_question(state: MyState) -> MyState:
    if len(state["list_A"]) > 1:
        state["list_B"].append(state["list_A"][1])
    return state


def reformulate_question(state: MyState) -> MyState:
    return state


def end_section_questions(state: MyState) -> MyState:
    state["current_question"] = ""
    state["questions"] = []
    return state


def obtain_get_question_graph():
    # Define the workflow graph
    builder = StateGraph(MyState)
    builder.add_node("take_first_question", take_first_question)
    builder.add_node("generate_new_question", generate_new_question)
    builder.add_node("take_any_question", take_any_question)
    builder.add_node("reformulate_question", reformulate_question)
    builder.add_node("end_section_questions", end_section_questions)
    builder.add_node("agent", Agent())

    # Define the execution flow
    builder.add_edge(START, "agent")
    builder.add_edge("take_first_question", "reformulate_question")
    builder.add_edge("generate_new_question", "reformulate_question")
    builder.add_edge("take_any_question", "reformulate_question")
    builder.add_edge("end_section_questions", END)
    builder.add_edge("reformulate_question", END)

    graph = builder.compile()
    return graph


if __name__ == "__main__":
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
