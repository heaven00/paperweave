from typing import Annotated, List, Callable

from langgraph.graph import StateGraph, START, END

from paperweave.flow_elements.prompt_templates import create_questions_template,create_intro_template
from paperweave.flow_elements.flows import create_answer, create_conclusion
from paperweave.transforms import extract_list, transcript_to_full_text
from paperweave.data_type import MyState, Utterance, Persona, Paper, Podcast
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
    topic: str,
    nb_questions: int,
    previous_topics: List[str],
    future_topics: List[str],
    podcast_tech_level: str,
):
    variables = {
        "paper_title": paper["title"],
        "podcast_tech_level": podcast_tech_level,
        "paper": paper["text"],
        "nb_questions": nb_questions,
        "topic": topic,
        "previous_topics": previous_topics,
        "future_topics": future_topics,
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
        state = MyState(podcast=podcast, index_question=0, index_topic=0)
        return state

class GetIntro:

    def __init__(self):
        self.model = get_chat_model()  
        self.podcast_tech_level = "expert"

    def __call__(self, state:MyState)->MyState:
        podcast = state["podcast"]
        paper = podcast["paper"]
        variables = {"paper_title": paper["title"], 
                    "podcast_tech_level": self.podcast_tech_level, 
                    "paper": paper["text"]}
        
        prompt = create_intro_template.invoke(variables)
        response = self.model.invoke(prompt)

        intro = response.content
        
        podcast["transcript"].append(Utterance(persona=Persona(name="host"), speach=intro))
        state["podcast"] = podcast

        return state

class GetExpertUtterance:
    def __init__(self):
        self.model = get_chat_model()
        self.podcast_tech_level = "expert"

    def __call__(self, state: MyState):
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
            Utterance(persona=Persona(name="host"), speach=question)
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
            Utterance(persona=Persona(name="expert"), speach=answer)
        )

        print(state["index_question"])
        state["index_question"] = state["index_question"] + 1
        return state


def get_chat_model() -> ChatOllama | ChatOpenAI:
    if os.environ["OPENAI_API_KEY"]:
        return ChatOpenAI(model="gpt-4o-mini")
    else:
        return ChatOllama(model="mistral-small:latest")



class GetQuestionsForTopic:
    def __init__(self):
        self.model = get_chat_model()
        self.nb_question_per_topic = 2
        self.podcast_tech_level = "expert"

    def __call__(self, state: MyState) -> MyState:
        podcast = state["podcast"]
        paper = podcast["paper"]
        index_topic = state["index_topic"]
        all_topics = state["topics"]
        previous_topics = all_topics[:index_topic]
        topic = all_topics[index_topic]
        future_topics = all_topics[index_topic + 1 :]
        questions = get_questions(
            model=self.model,
            paper=paper,
            nb_questions=self.nb_question_per_topic,
            topic=topic,
            previous_topics=previous_topics,
            future_topics=future_topics,
            podcast_tech_level="expert",
        )
        state["questions"] = questions
        state["podcast"] = podcast
        return state


class InitPodcast:
    def __call__(self, state: MyState):
        if "transcript" not in state["podcast"]:
            state["podcast"]["transcript"] = []

        state["topics"] = ["soliton", "results", "conclusion"]

        return state


class EndTopic:
    def __call__(self, state: MyState) -> MyState:
        state["index_topic"] = state["index_topic"] + 1
        state["index_question"] = 0
        return state


class Conclusion:
    def __init__(self):
        self.model = get_chat_model()
        self.podcast_tech_level = "expert"

    def __call__(self, state: MyState) -> MyState:
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
            Utterance(persona=Persona(name="host"), speach=conclusion_text)
        )


def loop_list_condition(index_name: str, list_name: int) -> Callable:
    def should_continue(state: MyState) -> bool:
        return state[index_name] < len(state[list_name])

    return should_continue


def build_graph():
    builder = StateGraph(MyState,input=MyState, output=MyState)
    # define nodes
    builder.add_node("get_paper", GetPaper())
    builder.add_node("init_podcast", InitPodcast())
    builder.add_node("intro_podcast", GetIntro())
    builder.add_node("get_question", GetQuestionsForTopic())
    builder.add_node("get_utterance", GetExpertUtterance())
    builder.add_node("end_topic", EndTopic())
    builder.add_node("conclusion", Conclusion())
    # define edges
    builder.add_edge(START, "get_paper")
    builder.add_edge("get_paper", "init_podcast")
    builder.add_edge("init_podcast", "intro_podcast")
    builder.add_edge("intro_podcast", "get_question")
    builder.add_edge("get_question", "get_utterance")
    builder.add_conditional_edges(
        "get_utterance",
        loop_list_condition(index_name="index_question", list_name="questions"),
        {True: "get_utterance", False: "end_topic"},
    )
    builder.add_conditional_edges(
        "end_topic",
        loop_list_condition(index_name="index_topic", list_name="topics"),
        {True: "get_question", False: "conclusion"},
    )
    builder.add_edge("conclusion", END)

    graph = builder.compile()

    return graph

############################################################################################

list_articles = ["1706.03762v7"] #,"1810.04805v2","2404.19756v4","2410.10630v1","2411.17703v1"]
for article in list_articles:
    graph = build_graph()
    graph_as_image = graph.get_graph().draw_mermaid_png()
    with open("image.png", "wb") as f:
        f.write(graph_as_image)

    result = graph.invoke({"podcast": {"paper": {"code": article}}})
    result = transcript_to_full_text(result["podcast"]["transcript"])
    print(result)

    folder_transcripts = 'data/transcripts'
    transcripts_files = []
    for entry in os.scandir(folder_transcripts):
        if entry.name.startswith('transcript_%s'%article):
            transcripts_files.append(entry.name)
    if len(transcripts_files) > 0:
        transcripts_files.sort()
        num = int(transcripts_files[-1].split(".")[-2])+1
    else: num = 0
    f_name = "%s/transcript_%s.%d.txt"%(folder_transcripts,article,num)
    with open(f_name, "a") as f:
        f.write(result)
