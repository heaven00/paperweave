from typing import Annotated, List, Callable
from typing_extensions import TypedDict


class Paper(TypedDict):
    # The operator.add reducer fn makes this append-only
    text: Annotated[str, "this is the string of the paper"] = ""
    code: Annotated[str, "this is the code of the paper(ex: 2332.2393)"] = ""
    title: Annotated[str, "this is the title of the paper"] = ""


class PaperCode(TypedDict):
    # The operator.add reducer fn makes this append-only
    code: Annotated[str, "this is the string of the paper"] = ""


class Persona(TypedDict):
    name: Annotated[str, "name of the persona"] = ""
    personality: Annotated[str, "describe the personality of the persona"] = ""


class Utterance(TypedDict):
    persona: Annotated[Persona, "person who speak"] = Persona()
    speach: Annotated[str, "what the persona have said"] = ""


# data format
class Podcast(TypedDict):
    # The operator.add reducer fn makes this append-only
    paper: Annotated[Paper, "paper that is discussed in the podcast"] = Paper()
    transcript: Annotated[
        List[Utterance], "list of utterance, i.e. what people said"
    ] = []
    questions: Annotated[List[str], "list of question on a topic"]
    topics: Annotated[List[str], "list of topic of the podcast"] = []


class MyState(TypedDict):
    podcast: Podcast = Podcast()
    index_question: int = 0
    index_topic: int = 0
    questions: List[str] = []
    topics: List[str] = []
