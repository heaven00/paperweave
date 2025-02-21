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

from paperweave.model import get_chat_model
from paperweave.graph.extraction_and_intro import (
    GetPaper,
    GetIntro,
    GetSectionsAnsQuestions,
    InitPodcast,
)


class HostUttterance:
    def __init__(self, podcast_tech_level: str = "expert"):
        self.model = get_chat_model()
        self.podcast_tech_level = podcast_tech_level

    def __call__(self, state: MyState):
        podcast = state["podcast"]
        question = state["questions"].pop(0)
        state["current_question"] = question
        state["questions_asked"].append(state["current_question"])

        podcast["transcript"].append(
            Utterance(persona=podcast["host"], speach=question, category="question")
        )
        return state


class ExpertUttterance:
    def __init__(self, podcast_tech_level: str = "expert"):
        self.model = get_chat_model()
        self.podcast_tech_level = podcast_tech_level

    def __call__(self, state: MyState):
        podcast = state["podcast"]
        paper_title = podcast["paper"]["title"]
        paper_text = podcast["paper"]["text"]
        question = state["current_question"]

        previous_question = "no previous question"
        previous_answer = "no previous answer"
        if podcast["transcript"]:
            if podcast["transcript"][-1]["persona"] == "expert":
                previous_answer = podcast["transcript"][-1]["speach"]
                previous_question = podcast["transcript"][-2]["speach"]

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


def get_utterance_subgraph(podcast_tech_level: str = "expert"):
    builder = StateGraph(MyState, input=MyState, output=MyState)
    builder.add_node("host", HostUttterance(podcast_tech_level=podcast_tech_level))
    builder.add_node("expert", ExpertUttterance(podcast_tech_level=podcast_level))
    builder.add_edge(START, "host")
    builder.add_edge("host", "expert")
    builder.add_edge("expert", END)
    graph = builder.compile()
    return graph


class GetQuestionsForSection:
    def __init__(self, nb_question_per_section=1, podcast_tech_level="expert"):
        self.model = get_chat_model()
        self.nb_question_per_section = nb_question_per_section
        self.podcast_tech_level = podcast_tech_level

    def __call__(self, state: MyState) -> MyState:
        podcast = state["podcast"]
        index_section = state["index_section"]

        questions = podcast["sections"][index_section]["questions"]
        state["questions"] = [
            question for question in questions
        ]  # simple way to make a copy of the list
        return state
    
class EndSection:
    def __call__(self, state: MyState) -> MyState:
        state["index_section"] = state["index_section"] + 1
        state["index_question"] = 0
        return state


################## MODIFY SECTIONS AND QUESTIONS ##############

class Dummy:
    def __call__(self, state: MyState) -> MyState:
        return state


class ModifySectionsQuestions:
    def __init__(self, nb_section=5, nb_question_per_section=2, podcast_tech_level="expert", sentence="",
    ):
        self.nb_section = nb_section
        self.nb_question_per_section = nb_question_per_section
        self.podcast_tech_level = podcast_tech_level
        self.model = get_chat_model()
        self.sentence = sentence

    def __call__(self, state: MyState) -> MyState:
        podcast = state["podcast"]
        paper = podcast["paper"]
        index_section = state["index_section"]
        previous_sections = state["sections"][:index_section+1] #previous+current
        modify_section_and_question = get_modified_sections_questions(
            model=self.model,
            paper_title=paper["title"],
            podcast_tech_level=self.podcast_tech_level,
            paper=paper["text"],
            nb_sections=self.nb_section-(index_section+1),
            nb_questions_per_section=self.nb_question_per_section,
            previous_sections = previous_sections,
            sentence = self.sentence,
        )
        #update the podcast.sections and sections fields and the index_section
        state["podcast"]["sections"] = state["podcast"]["sections"][:index_section+1]
        state["podcast"]["sections"] += modify_section_and_question.model_dump()["sections"]
        print('initial sections:',state["sections"])
        state["sections"] = [section["section_subject"] for section in state["podcast"]["sections"]]
        print('updated sections',state["sections"])
        ###state["index_section"] +=1 

        return state

from paperweave.data_type_direct_llm_call import (
    SectionQuestionLLMOutput,
    LLMResponseQuestionChoice,
)
def get_modified_sections_questions(
    model,
    paper_title: str,
    podcast_tech_level: str,
    paper: str,
    nb_sections: int,
    nb_questions_per_section: int,
    previous_sections,
    sentence: str,
) -> SectionQuestionLLMOutput:
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "paper": paper,
        "nb_sections": nb_sections,
        "nb_questions_per_section": nb_questions_per_section,
        "previous_sections": previous_sections,
        "sentence": sentence,
    }

    prompt = modify_sections_questions_template.invoke(variables)

    model_with_structure = model.with_structured_output(SectionQuestionLLMOutput)
    response = model_with_structure.invoke(prompt)
    return response

modify_sections_questions_template = """You are the host of a podcast where you discuss the paper titled "{paper_title}".
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
Generate a list of sections of the podcast and questions to be asked to make it an interesting podcast."""

modify_sections_questions_user = """Create a list of sections of the podcast. Each section should contain questions to be asked.  
DO NOT FOLLOW THE STRUCTURE OF THE PAPER. MAKE IT THE STRUCTURE OF AN INTERESTING PODCAST! 

Create a list of {nb_sections} sections (number_of_section) that will follow the sections:
{previous_sections}
and take into account the following directive:
{sentence}
For each newly created section, generate {nb_questions_per_section} questions (number_of_question) to discuss the paper:
{paper}
"""

from langchain_core.prompts import ChatPromptTemplate
modify_sections_questions_template = ChatPromptTemplate.from_messages(
    [("system", modify_sections_questions_template), ("user", modify_sections_questions_user)]
)

#############################
    
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

def loop_index_condition(index_name: str) -> Callable:
    def should_continue(state: MyState) -> bool:
        return state[index_name] == 2# len(state[list_name])

    return should_continue


def loop_non_empty_list(list_name: str) -> Callable:
    def should_continue(state: MyState) -> bool:
        return bool(state[list_name])

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
        "get_utterance", get_utterance_subgraph(podcast_tech_level=podcast_level)
    )
    builder.add_node(
        "modify_flow",
        ModifySectionsQuestions(
            nb_section=nb_section, 
            nb_question_per_section=begin_nb_question_per_section, 
            podcast_tech_level=podcast_level, 
            sentence="Speak about the paper's results", # !! Here the listener's input is hard coded
        ),
    )
    builder.add_node("time_to_modify_flow", Dummy())
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
        loop_non_empty_list(list_name="questions"),
        {True: "get_utterance", False: "time_to_modify_flow"},
    )
    builder.add_conditional_edges(
        "time_to_modify_flow",
        loop_index_condition(index_name="index_section"),
        {True: "modify_flow", False: "end_section"},
    )
    builder.add_conditional_edges(
        "end_section",
        loop_list_condition(index_name="index_section", list_name="sections"),
        {True: "get_question", False: "conclusion"},
    )
    builder.add_edge("modify_flow", "end_section")
    builder.add_edge("conclusion", END)

    graph = builder.compile()

    return graph


############################################################################################

list_articles = [
    # "2411.17703v1"
    "1706.03762v7"
]  # ,"1810.04805v2","2404.19756v4","2410.10630v1","2411.17703v1"]
nb_section = 5
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



############## CLASSIFY INPUT FROM LISTENER ################

from langchain_core.prompts import ChatPromptTemplate
find_sentence_type_system =  """You are the host of a podcast where you discuss a paper".
The person listening to your podcast is talking to you."""

find_sentence_type_user = """The person is saying: "{sentence}". 
Classify what of the following option is true:
1) the person is asking a question about the paper 
2) the person is asking you to change the layout of the podcast or to change the topic to discuss

If the correct option is 1) then print "question", if the correct option is 2) then print "directive", else print "NA".

"""
find_sentence_type_template = ChatPromptTemplate.from_messages(
        [("system",find_sentence_type_system),("user", find_sentence_type_user)]
    )

def get_sentence_type(
    model,
    sentence: str,
):
    variables = {
        "sentence": sentence,
    }
    
    # Format the prompt with the variables
    prompt = find_sentence_type_template.invoke(variables)
    
    # Get the model's response
    response = model.invoke(prompt)
    
    if "question" in response.content:
        sentence_type = 'Q' #question
    elif "directive" in response.content:
        sentence_type = 'D' #directive
    else: sentence_type = 'NA'
    return sentence_type

class SentenceType:
    def __init__(self):
        self.model = get_chat_model()

    def __call__(self,sentence: str):
        sentence_type = get_sentence_type(
            model=self.model,
            sentence = sentence,
        )
        return sentence_type

classifier = SentenceType()

list_requests = ["What is attention?",
                "Can you talk about results and skip the techincal desciption of the architecture?",
                "do not speak about the results of the paper",
                "Who is the president of USA?",
                "What is LSTM?",
                "What is space radiation?",
                "How is the weather today?"
]
for request in list_requests:
    print(request,": ",classifier(request))


