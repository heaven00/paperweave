import operator
from typing import Annotated, Any, List

from typing_extensions import TypedDict
import requests
from bs4 import BeautifulSoup

from langgraph.graph import StateGraph, START, END

from paperweave.flow_elements.prompt_templates import create_questions_template, create_intro_template
from paperweave.transforms.string_extract import extract_list
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
#from pathlib import Path
#env_file = Path(__file__).parent.parent / ".env"

# Load the .env file
load_dotenv()

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_KEY')




class Paper(TypedDict):
    # The operator.add reducer fn makes this append-only
    text: Annotated[str, "this is the string of the paper"] 
    code: Annotated[str, "this is the code of the paper(ex: 2332.2393)"]
    title: Annotated[str, "this is the title of the paper"] 

# class PaperCode(TypedDict):
#     # The operator.add reducer fn makes this append-only
#     code: Annotated[str, "this is the string of the paper"] = ""

class Persona(TypedDict):
    name: Annotated[str, "name of the persona"] 
    personality: Annotated[str, "describe the personality of the persona"]


class Utterance(TypedDict):
    persona: Annotated[Persona, "person who speak"]
    speach : Annotated[str, "what the persona have said"] 


# data format
class Podcast(TypedDict):
    # The operator.add reducer fn makes this append-only
    paper: Annotated[Paper, "paper that is discussed in the podcast"]
    transcript: Annotated[List[Utterance], "list of utterance, i.e. what people said"] 
    questions: Annotated[List[str], "list of question on a topic"]
    topics: Annotated[List[str], "list of topic of the podcast"] 

class MyState(TypedDict):
    podcast: Podcast 
    index_question: int 
    index_topic:int 


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

    def __call__(self, state:MyState):
        id_question = state["index_question"]
        podcast = state["podcast"]
        question = podcast["questions"][id_question]

        podcast["transcript"].append(Utterance(persona=Persona(name="host"), speach=question))


        podcast = state["podcast"]
        if "previous_question" not in podcast:
            podcast["previous_question"] = "no previous question"
        previous_question = podcast["previous_question"]
        if "previous_answer" not in podcast:
            podcast["previous_answer"] = "no previous answer"
        previous_answer = podcast["previous_answer"]


        print(state["index_question"])
        state["index_question"] = state["index_question"] + 1
        return state



def should_continue_index(index_name):

    def should_continue(state: MyState) -> bool:
        return state[index_name] < len(state["podcast"]["questions"])
    
    return should_continue




class GetQuestionsForTopic:

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.nb_question_per_topic = 5
        self.podcast_tech_level = "expert"

    def __call__(self, state:MyState)->MyState:
        podcast = state["podcast"]
        paper = podcast["paper"]
        previous_topics = []
        future_topics = ["results", "conclusion"]
        topic = "soliton"
        questions = get_questions(model = self.model, paper = paper, nb_questions = self.nb_question_per_topic,
                                  topic= topic, previous_topics=previous_topics, future_topics=future_topics,
                                  podcast_tech_level="expert")
        podcast["questions"] = questions
        state["podcast"] = podcast
        return state

class GetIntro:

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.podcast_tech_level = "expert"

    def __call__(self, state:MyState)->MyState:
        podcast = state["podcast"]
        paper = podcast["paper"]
        variables = {"paper_title": paper["title"], 
                    "podcast_tech_level": self.podcast_tech_level, 
                    "paper": paper["text"]}
        # Format the prompt with the variables
        prompt = create_intro_template.invoke(variables)

        # Get the model's response
        response = self.model.invoke(prompt)
        intro = extract_list(response.content)

        podcast["transcript"].append(Utterance(persona=Persona(name="host"), speach=intro))
        state["podcast"] = podcast

        return state


class InitPodcast:

    def __call__(self, state:MyState):

        if "transcript" not in state["podcast"]:
            state["podcast"]["transcript"] = []


        return state



# c= Paper()
# b = ""
# a = MyState(podcast= "")

builder = StateGraph(MyState) #,input= MyState , output=MyState)
builder.add_node("get_paper", GetPaper())
builder.add_node("init_podcast", InitPodcast())
builder.add_node("intro_podcast", GetIntro())
builder.add_node("get_question", GetQuestionsForTopic())
builder.add_node("get_utterance", GetExpertUtterance())
builder.add_edge(START, "get_paper")
builder.add_edge("get_paper", "init_podcast")
builder.add_edge("init_podcast", "intro_podcast")
builder.add_edge("intro_podcast", "get_question")
builder.add_edge("get_question", "get_utterance")
builder.add_conditional_edges(
    "get_utterance",
    should_continue_index("index_question"),
    {True: "get_utterance", False: END}
)


graph = builder.compile()

a = graph.get_graph().draw_mermaid_png()

with open("image.png", 'wb') as f:
    f.write(a)

result = graph.invoke({"podcast": {"paper":{"code":"2411.17703"}}})
print(result) #2411.17703"}}}))
print(result['podcast']['transcript'])






