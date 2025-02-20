from paperweave.data_type import MyState, Paper, Persona, Podcast, Utterance
from paperweave.flow_elements.flows import get_sections_questions
from paperweave.get_data import get_arxiv_text, get_paper_title
from paperweave.model import get_chat_model
from paperweave.flow_elements.prompt_templates import create_intro_template
from langgraph.graph import StateGraph, START, END


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


class InitPodcast:
    def __call__(self, state: MyState):
        if "transcript" not in state["podcast"]:
            state["podcast"]["transcript"] = []
        if "sections" not in state["podcast"]:
            state["podcast"]["sections"] = []
        if "questions_asked" not in state:
            state["questions_asked"] = []

        state["podcast"]["host"] = Persona(name="Jimmy")
        state["podcast"]["expert"] = Persona(name="Mike")

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


def init_and_intro_graph(podcast_level: str = "expert"):
    builder = StateGraph(MyState, input=MyState, output=MyState)
    # define nodes
    builder.add_node("get_paper", GetPaper())
    builder.add_node("init_podcast", InitPodcast())
    builder.add_node("intro_podcast", GetIntro(podcast_level=podcast_level))
    builder.add_node(
        "get_sections",
        GetSectionsAnsQuestions(
            podcast_tech_level=podcast_level,
        ),
    )

    # define edges
    builder.add_edge(START, "get_paper")
    builder.add_edge("get_paper", "init_podcast")
    builder.add_edge("init_podcast", "intro_podcast")
    builder.add_edge("intro_podcast", "get_sections")
    builder.add_edge("get_sections", END)

    return builder.compile()
