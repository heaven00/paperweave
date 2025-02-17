from typing import Annotated, List, Callable
import json

from langgraph.graph import StateGraph, START, END

from paperweave.flow_elements.prompt_templates import (
    create_questions_template,
    create_intro_template,
)
from paperweave.flow_elements.flows import (
    create_answer,
    create_conclusion,
    get_sections,
    get_sections_questions,
)
from paperweave.transforms import extract_list, transcript_to_full_text
from paperweave.data_type import MyState, Utterance, Persona, Paper, Podcast, Section
from paperweave.get_data import get_arxiv_text, get_paper_title
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from pathlib import Path

env_file = Path(__file__).parent.parent / ".env"

# Load the .env file
load_dotenv(env_file)

# Set your OpenAI API key by default it will set to empty string
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY") or ""


def get_questions(
    model,
    paper: Paper,
    section: str,
    nb_questions: int,
    previous_sections: List[str],
    future_sections: List[str],
    podcast_tech_level: str,
):
    variables = {
        "paper_title": paper["title"],
        "podcast_tech_level": podcast_tech_level,
        "paper": paper["text"],
        "nb_questions": nb_questions,
        "section": section,
        "previous_sections": previous_sections,
        "future_sections": future_sections,
    }

    # Format the prompt with the variables
    prompt = create_questions_template.invoke(variables)

    # Get the model's response
    response = model.invoke(prompt)
    questions = extract_list(response.content)
    return questions


# node
class GetPaper:
    def __call__(self, state: MyState) -> MyState:
        code = state["podcast"]["paper"]["code"]
        text = get_arxiv_text(code)
        title = get_paper_title(code)
        paper = Paper(text=text, code=code, title=title)
        podcast = Podcast(paper=paper)
        state = MyState(podcast=podcast, index_question=0, index_section=0)
        return state


class GetIntro:
    def __init__(self, podcast_level="expert"):
        self.model = get_chat_model()
        self.podcast_tech_level = podcast_level

    def __call__(self, state: MyState) -> MyState:
        podcast = state["podcast"]
        paper = podcast["paper"]
        variables = {
            "paper_title": paper["title"],
            "podcast_tech_level": self.podcast_tech_level,
            "paper": paper["text"],
            "host_name": podcast["host"]["name"],
            "expert_name": podcast["expert"]["name"],
        }

        prompt = create_intro_template.invoke(variables)
        response = self.model.invoke(prompt)

        intro = response.content

        podcast["transcript"].append(
            Utterance(persona=podcast["host"], speach=intro, category="introduction")
        )
        state["podcast"] = podcast

        return state


class GetSectionsAnsQuestions:
    def __init__(
        self, nb_section=5, podcast_tech_level="expert", nb_question_per_section=2
    ):
        self.nb_section = nb_section
        self.podcast_tech_level = podcast_tech_level
        self.model = get_chat_model()
        self.nb_question_per_section = nb_question_per_section

    def __call__(self, state: MyState) -> MyState:
        podcast = state["podcast"]
        paper = podcast["paper"]

        section_and_question = get_sections_questions(
            model=self.model,
            paper_title=paper["title"],
            podcast_tech_level=self.podcast_tech_level,
            paper=paper["text"],
            nb_sections=self.nb_section,
            nb_questions_per_section=self.nb_question_per_section,
        )
        state["podcast"]["sections"] = section_and_question.model_dump()["sections"]
        state["sections"] = [
            section["section_subject"] for section in state["podcast"]["sections"]
        ]

        return state


class GetExpertUtterance:
    def __init__(self, podcast_tech_level: str = "expert"):
        self.model = get_chat_model()
        self.podcast_tech_level = podcast_tech_level

    def __call__(self, state: MyState) -> MyState:
        id_question = state["index_question"]
        podcast = state["podcast"]
        question = state["questions"][id_question]

        previous_question = "no previous question"
        previous_answer = "no previous answer"
        if podcast["transcript"]:
            if podcast["transcript"][-1]["persona"] == "expert":
                previous_answer = podcast["transcript"][-1]["speach"]
                previous_question = podcast["transcript"][-2]["speach"]

        podcast["transcript"].append(
            Utterance(persona=podcast["host"], speach=question, category="question")
        )

        paper_title = podcast["paper"]["title"]
        paper_text = podcast["paper"]["text"]

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

        print(state["index_question"])
        state["index_question"] = state["index_question"] + 1
        return state


def get_chat_model() -> ChatOllama | ChatOpenAI:
    if os.environ["OPENAI_API_KEY"]:
        return ChatOpenAI(model="gpt-4o-mini")
    else:
        return ChatOllama(model="mistral-small:latest")


class GetQuestionsForSection:
    def __init__(self, nb_question_per_section=1, podcast_tech_level="expert"):
        self.model = get_chat_model()
        self.nb_question_per_section = nb_question_per_section
        self.podcast_tech_level = podcast_tech_level

    def __call__(self, state: MyState) -> MyState:
        podcast = state["podcast"]
        index_section = state["index_section"]

        questions = podcast["sections"][index_section]["questions"]
        state["questions"] = questions
        return state


class InitPodcast:
    def __call__(self, state: MyState):
        if "transcript" not in state["podcast"]:
            state["podcast"]["transcript"] = []
        if "sections" not in state["podcast"]:
            state["podcast"]["sections"] = []

        state["podcast"]["host"] = Persona(name="Jimmy")
        state["podcast"]["expert"] = Persona(name="Mike")

        return state


class EndSection:
    def __call__(self, state: MyState) -> MyState:
        state["index_section"] = state["index_section"] + 1
        state["index_question"] = 0
        return state


class Conclusion:
    def __init__(self, podcast_tech_level: str = "expert"):
        self.model = get_chat_model()
        self.podcast_tech_level = podcast_tech_level

    def __call__(self, state: MyState) -> MyState:
        podcast = state["podcast"]
        paper_title = state["podcast"]["paper"]["title"]
        paper_text = state["podcast"]["paper"]["text"]
        transcript = transcript_to_full_text(state["podcast"]["transcript"])
        conclusion_text = create_conclusion(
            model=self.model,
            paper_title=paper_title,
            podcast_tech_level=self.podcast_tech_level,
            paper=paper_text,
            podcast_transcript=transcript,
        )
        state["podcast"]["transcript"].append(
            Utterance(
                persona=podcast["host"], speach=conclusion_text, category="conclusion"
            )
        )


def loop_list_condition(index_name: str, list_name: int) -> Callable:
    def should_continue(state: MyState) -> bool:
        return state[index_name] < len(state[list_name])

    return should_continue


def build_graph(
    nb_section: int = 2,
    begin_nb_question_per_section: int = 2,
    podcast_level: str = "expert",
):
    builder = StateGraph(MyState, input=MyState, output=MyState)
    # define nodes
    builder.add_node("get_paper", GetPaper())
    builder.add_node("init_podcast", InitPodcast())
    builder.add_node("intro_podcast", GetIntro(podcast_level=podcast_level))
    builder.add_node(
        "get_sections",
        GetSectionsAnsQuestions(
            nb_section=nb_section,
            podcast_tech_level=podcast_level,
            nb_question_per_section=begin_nb_question_per_section,
        ),
    )
    builder.add_node(
        "get_question",
        GetQuestionsForSection(),
    )
    builder.add_node(
        "get_utterance", GetExpertUtterance(podcast_tech_level=podcast_level)
    )
    builder.add_node("end_section", EndSection())
    builder.add_node("conclusion", Conclusion(podcast_tech_level=podcast_level))
    # define edges
    builder.add_edge(START, "get_paper")
    builder.add_edge("get_paper", "init_podcast")
    builder.add_edge("init_podcast", "intro_podcast")
    builder.add_edge("intro_podcast", "get_sections")
    builder.add_edge("get_sections", "get_question")
    builder.add_edge("get_question", "get_utterance")
    builder.add_conditional_edges(
        "get_utterance",
        loop_list_condition(index_name="index_question", list_name="questions"),
        {True: "get_utterance", False: "end_section"},
    )
    builder.add_conditional_edges(
        "end_section",
        loop_list_condition(index_name="index_section", list_name="sections"),
        {True: "get_question", False: "conclusion"},
    )
    builder.add_edge("conclusion", END)

    graph = builder.compile()

    return graph


############################################################################################

list_articles = [
    # "2411.17703v1"
    "1706.03762v7"
]  # ,"1810.04805v2","2404.19756v4","2410.10630v1","2411.17703v1"]
nb_section = 3
begin_nb_question_per_section = 2
podcast_level = "expert"
for article in list_articles:
    input = {
        "article_code": article,
        "nb_section": nb_section,
        "begin_nb_question_per_section": begin_nb_question_per_section,
        "podcast_level": podcast_level,
    }
    graph = build_graph(
        nb_section=nb_section,
        begin_nb_question_per_section=begin_nb_question_per_section,
        podcast_level=podcast_level,
    )
    graph_as_image = graph.get_graph().draw_mermaid_png()
    with open("image.png", "wb") as f:
        f.write(graph_as_image)

    result = graph.invoke(
        {"podcast": {"paper": {"code": article}}}, {"recursion_limit": 100}
    )
    result["input"] = input
    print(result)

    json_data = json.dumps(result, indent=4)  # indent for pretty printing, optional

    # Save JSON string to a file
    folder_json = str(Path(__file__).parent.parent / "data" / "pipeline_output")
    json_files = []
    for entry in os.scandir(folder_json):
        if entry.name.startswith("%s" % article):
            json_files.append(entry.name)
    if len(json_files) > 0:
        json_files.sort()
        num = int(json_files[-1].split(".")[-2]) + 1
    else:
        num = 0
    with open("%s/%s.%d.json" % (folder_json, article, num), "w") as json_file:
        json_file.write(json_data)

    # Save txt file with readable trascript
    result = transcript_to_full_text(result["podcast"]["transcript"])
    print(result)
    folder_transcripts = str(Path(__file__).parent.parent / "data" / "transcripts")
    f_name = "%s/transcript_%s.%d.txt" % (folder_transcripts, article, num)
    with open(f_name, "a") as f:
        f.write(result)
