import operator
from typing import Annotated, List, Callable

from typing_extensions import TypedDict
import requests
from bs4 import BeautifulSoup



from langgraph.graph import StateGraph, START, END

from paperweave.flow_elements.prompt_templates import create_questions_template
from paperweave.flow_elements.flows import create_answer
from paperweave.transforms.string_extract import extract_list
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
env_file = Path(__file__).parent.parent / ".env"

# Load the .env file
load_dotenv(env_file)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_KEY')




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
    speach : Annotated[str, "what the persona have said"] = ""


# data format
class Podcast(TypedDict):
    # The operator.add reducer fn makes this append-only
    paper: Annotated[Paper, "paper that is discussed in the podcast"] = Paper()
    transcript: Annotated[List[Utterance], "list of utterance, i.e. what people said"] = []
    questions: Annotated[List[str], "list of question on a topic"]
    topics: Annotated[List[str], "list of topic of the podcast"] = []

class MyState(TypedDict):
    podcast: Podcast = Podcast()
    index_question: int = 0
    index_topic:int  =0
    questions: List[str]= []
    topics: List[str] = []



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

    def __call__(self, state:MyState) -> MyState:
        code = state["podcast"]["paper"]["code"]
        text = get_arxiv_text(code)
        paper = Paper(text=text, code=code, title= "stuff")
        podcast = Podcast(paper=paper)
        state = MyState(podcast=podcast, index_question=0, index_topic=0)
        return state




def get_questions(model, paper:Paper, topic:str, nb_questions:int, previous_topics:List[str], future_topics:List[str], podcast_tech_level:str):
    variables = {"paper_title": paper["title"], "podcast_tech_level": podcast_tech_level, "paper": paper["text"], "nb_questions": nb_questions,
                 "topic": topic, "previous_topics":previous_topics, "future_topics":future_topics}

    # Format the prompt with the variables
    prompt = create_questions_template.invoke(variables)

    # Get the model's response
    response = model.invoke(prompt)
    questions = extract_list(response.content)
    return questions




class GetExpertUtterance:

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.podcast_tech_level = "expert"

    def __call__(self, state:MyState):
        id_question = state["index_question"]
        podcast = state["podcast"]
        question = state["questions"][id_question]

        previous_question = "no previous question"
        previous_answer = "no previous answer"
        if podcast["transcript"]:
            if podcast["transcript"][-1]["persona"]=="expert":
                previous_answer = podcast["transcript"][-1]["speach"]
                previous_question = podcast["transcript"][-2]["speach"]

        podcast["transcript"].append(Utterance(persona=Persona(name="host"), speach=question))

        paper_title = podcast["paper"]["title"]
        paper_text = podcast["paper"]["text"]

        answer = create_answer(model=self.model, paper_title=paper_title, podcast_tech_level=self.podcast_tech_level, paper=paper_text, previous_question=previous_question, previous_answer=previous_answer, new_question=question)

        podcast["transcript"].append(Utterance(persona=Persona(name="expert"), speach=answer))


        print(state["index_question"])
        state["index_question"] = state["index_question"] + 1
        return state


def loop_list_condition(index_name:str, list_name:int)->Callable:

    def should_continue(state: MyState) -> bool:
        return state[index_name] < len(state[list_name])
    
    return should_continue




class GetQuestionsForTopic:

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.nb_question_per_topic = 2
        self.podcast_tech_level = "expert"

    def __call__(self, state:MyState)->MyState:
        podcast = state["podcast"]
        paper = podcast["paper"]
        index_topic = state["index_topic"]
        all_topics = state["topics"]
        previous_topics = all_topics[:index_topic]
        topic = all_topics[index_topic]
        future_topics = all_topics[index_topic+1:]
        questions = get_questions(model= self.model, paper = paper, nb_questions = self.nb_question_per_topic,
                                  topic= topic, previous_topics=previous_topics, future_topics=future_topics,
                                  podcast_tech_level="expert")
        state["questions"] = questions
        state["podcast"] = podcast
        return state


class InitPodcast:

    def __call__(self, state:MyState):

        if "transcript" not in state["podcast"]:
            state["podcast"]["transcript"] = []

        state["topics"] = ["soliton", "results", "conclusion"]

        return state


class EndTopic:
    def __call__(self, state:MyState)->MyState:
        state['index_topic'] = state['index_topic']+1
        state['index_question'] = 0
        return state



builder = StateGraph(input= MyState , output=MyState)
builder.add_node("get_paper", GetPaper())
builder.add_node("init_podcast", InitPodcast())
builder.add_node("get_question", GetQuestionsForTopic())
builder.add_node("get_utterance", GetExpertUtterance())
builder.add_node("end_topic", EndTopic())
builder.add_edge(START, "get_paper")
builder.add_edge("get_paper", "init_podcast")
builder.add_edge("init_podcast", "get_question")
builder.add_edge("get_question", "get_utterance")
builder.add_conditional_edges(
    "get_utterance",
    loop_list_condition(index_name = "index_question", list_name= "questions"),
    {True: "get_utterance", False: "end_topic"}
)
builder.add_conditional_edges(
    "end_topic",
    loop_list_condition(index_name = "index_topic", list_name= "topics"),
    {True: "get_question", False: END}
)

graph = builder.compile()

a = graph.get_graph().draw_mermaid_png()

with open("image.png", 'wb') as f:
    f.write(a)


print(graph.invoke({"podcast": {"paper":{"code":"2203.11171"}}}))







