# from typing import Annotated, List, Callable
#
# from langgraph.graph import StateGraph, START, END
#
# from paperweave.flow_elements.prompt_templates import create_questions_template
# from paperweave.flow_elements.flows import create_answer, create_conclusion
# from paperweave.transforms import extract_list, transcript_to_full_text
# from paperweave.data_type import MyState, Utterance, Persona, Paper, Podcast
# from paperweave.get_data import get_arxiv_text, get_paper_title
import os, io
import PyPDF2
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# from pathlib import Path
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import requests



from pypdf import PdfReader

# # env_file = Path(__file__).parent.parent/ ".env"

# # Load the .env file
#load_dotenv(env_file)

# Set your OpenAI API key
#openai.api_key = os.environ.get("OPENAI_API_KEY")

model = ChatOpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    max_tokens=1024,
    temperature=1,
)

OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are an expert in Machine Learning\n"
    + "You can extract a list of sections for podcast from any text\n"
    + "You will select no more than top 10 topics and present them in a list."
)


def pdf2text(webaddress):
    response = requests.get(webaddress)
    pdf_io_bytes = io.BytesIO(response.content)
    text_list = []
    pdf = PyPDF2.PdfReader(pdf_io_bytes)

    num_pages = len(pdf.pages)

    for page in range(num_pages):
        page_text = pdf.pages[page].extract_text()
        text_list.append(page_text)
    text = "\n".join(text_list)

    return text_list


def getTopics(text_list):
    direct_prompt = ChatPromptTemplate.from_messages([
        ("system", OPENAI_SYSTEM_MESSAGE_CHATGPT),
        ("human", """
        Extract topics from {prompt}
        """)
    ])

    chain = direct_prompt | model | (lambda x: x.content)

    topics_list = chain.invoke({
        "prompt": text_list
    })
    return topics_list

text_context = pdf2text("https://arxiv.org/pdf/1706.03762")
list_of_topics = getTopics(text_context)
print(list_of_topics)

