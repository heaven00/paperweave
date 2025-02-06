# define the graph
from enum import Enum
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.runnables.graph import MermaidDrawMethod
from docling.document_converter import DocumentConverter


class Speakers(Enum):
    Host = "host"
    Expert = "expert"


class InputState(TypedDict):
    url: str


class PodcastNotes(TypedDict):
    """State object to store every thing required to generate the podcast"""

    content: str
    speech_text: str
    topics: list[str]
    questions: list[str]


class OutputState(TypedDict):
    """The output transcipt (the final output)"""

    transcript: list[dict[Speakers, str]]


def arxiv_url_to_markdown(state: InputState) -> PodcastNotes:
    """Convert arxiv url to markdown text for downstream llms"""
    converter = DocumentConverter()
    return {"content": converter.convert(state["url"])}


def generate_podcast_generation_graph() -> "CompiledStateGraph":  # type: ignore
    builder = StateGraph(PodcastNotes, input=InputState, output=OutputState)

    # add nodes/ python functions to transition to
    builder.add_node(arxiv_url_to_markdown)

    # connect the edges
    builder.add_edge(START, "arxiv_url_to_markdown")

    return builder.compile()


if __name__ == "__main__":
    # generate the graph and print out the image
    graph = generate_podcast_generation_graph()

    with open("podcast_graph.png", "wb") as img_file:
        img_file.write(
            graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
