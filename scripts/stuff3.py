import os
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
import requests

from dotenv import load_dotenv
from pathlib import Path

import os

env_file = Path(__file__).parent.parent / ".env"

# Load the .env file
load_dotenv(env_file)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

# Initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-4o-mini")


from paperweave.flow_elements.flows import (
    create_new_question,
    get_arxiv_text,
    loop_question_answer_on_topic,
    get_topics,
)

arxiv_code = "2501.00019"


paper_title = "Nonlinear topological pumping of edge solitons"
paper = get_arxiv_text(arxiv_code)
podcast_tech_level = "eli5"
topic = "non-linarity"
previous_question = "No previous question"
previous_answer = "No previous answer"

nb_question = 2
nb_topics = 7
topics = get_topics(model, paper_title, podcast_tech_level, paper, nb_topics)
print(topics)
topic = "what are soliton?"
# Print the translated text
questions, answers = loop_question_answer_on_topic(
    model,
    paper_title,
    podcast_tech_level,
    paper,
    topic,
    previous_question,
    previous_answer,
    nb_question,
)
for question, answer in zip(questions, answers):
    print(question)
    print()
    print(answer)
    print()
    print()
