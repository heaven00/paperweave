import operator
from typing import Annotated, Any

from typing_extensions import TypedDict
import requests
from bs4 import BeautifulSoup

from langgraph.graph import StateGraph, START, END


# data format
class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]


class Paper(TypedDict):
    # The operator.add reducer fn makes this append-only
    text: Annotated[str, "this is the string of the paper"]
    code: Annotated[str, "this is the code of the paper(ex: 2332.2393)"]

class PaperCode(TypedDict):
    # The operator.add reducer fn makes this append-only
    code: Annotated[str, "this is the string of the paper"]


# compute node
class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}


def get_arxiv_text(arxiv_code:str):

    url = f'https://arxiv.org/html/{arxiv_code}v1'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


class GetPaper:

    def __call__(self, paper_code:PaperCode) -> Paper:
        text = get_arxiv_text(paper_code["code"])
        return {"text": text, "code":paper_code["code"]}



builder = StateGraph(input=PaperCode, output=Paper)
builder.add_node("get_paper", GetPaper())
builder.add_edge(START, "get_paper")
builder.add_edge("get_paper", END)

graph = builder.compile()


print(graph.invoke({"code": "2203.11171"}))
print(graph.invoke({"code": "2203.11171"}))




"""
builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_edge(START, "a")

builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)

graph = builder.compile()

print(graph.invoke({"aggregate": ["stuff", 34]}))

"""


a = graph.get_graph().draw_mermaid_png()

with open("image.png", 'wb') as f:
    f.write(a)
