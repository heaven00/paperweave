from bs4 import BeautifulSoup
import requests
import re
from paperweave.flow_elements.prompt_templates import (
    new_question_template,
    answer_template,
    find_topics_template,
    host_conclusion_template,
    find_topics_questions_template
)
from paperweave.transforms import extract_list


def get_arxiv_text(arxiv_code: str):
    url = f"https://arxiv.org/html/{arxiv_code}v1"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


def create_new_question(
    model,
    paper_title,
    podcast_tech_level,
    paper,
    topic,
    previous_question,
    previous_answer,
):
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "paper": paper,
        "topic": topic,
        "previous_question": previous_question,
        "previous_answer": previous_answer,
    }

    # Format the prompt with the variables
    prompt = new_question_template.invoke(variables)

    # Get the model's response
    response = model.invoke(prompt)
    return response.content


def create_answer(
    model,
    paper_title,
    podcast_tech_level,
    paper,
    previous_question,
    previous_answer,
    new_question,
):
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "paper": paper,
        "previous_question": previous_question,
        "previous_answer": previous_answer,
        "new_question": new_question,
    }

    # Format the prompt with the variables
    prompt = answer_template.invoke(variables)

    # Get the model's response
    response = model.invoke(prompt)
    return response.content


def create_conclusion(
    model, paper_title, podcast_tech_level, paper, podcast_transcript: str
):
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "paper": paper,
        "podcast_transcript": podcast_transcript,
    }

    # Format the prompt with the variables
    prompt = host_conclusion_template.invoke(variables)

    # Get the model's response
    response = model.invoke(prompt)
    return response.content


def loop_question_answer_on_topic(
    model,
    paper_title,
    podcast_tech_level,
    paper,
    topic,
    previous_question,
    previous_answer,
    nb_question,
):
    questions = []
    answers = []
    for id_question in range(nb_question):
        new_question = create_new_question(
            model,
            paper_title,
            podcast_tech_level,
            paper,
            topic,
            previous_question,
            previous_answer,
        )
        new_answer = create_answer(
            model,
            paper_title,
            podcast_tech_level,
            paper,
            previous_question,
            previous_answer,
            new_question,
        )
        questions.append(new_question)
        answers.append(new_answer)
        previous_question = new_question
        previous_answer = new_answer
    return questions, answers


def get_topics(model, paper_title, podcast_tech_level, paper, nb_topics):
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "paper": paper,
        "nb_topics": nb_topics,
    }

    # Format the prompt with the variables
    prompt = find_topics_template.invoke(variables)

    # Get the model's response
    response = model.invoke(prompt)

    result = extract_list(response.content)
    return result

from pydantic import BaseModel, Field
from typing import Optional
from typing import List

class Section(BaseModel):
    questions : List[str] = Field(description="list of questions of size number_of_question")
    section_subject : str  = Field(description="description of podcast section")
    number_of_question: int =  Field(description="number of questions in the section")

class FullOutput(BaseModel):
    sections: List[Section] = Field(description="list of sections of size number_of_section")
    number_of_section:int =  Field(description="number of sections in the podcast")

def get_sections_questions(model, paper_title, podcast_tech_level, paper, nb_topics, nb_questions_per_topic):
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "paper": paper,
        "nb_topics": nb_topics,
        "nb_questions_per_topic": nb_questions_per_topic
    }

    # Format the prompt with the variables
    prompt = find_topics_questions_template.invoke(variables)

    # # Get the model's response
    # def schema(nb_topics,nb_questions_per_topic):
    #     final_list = {}
    #     for nt in range(nb_topics):
    #         list_questions = ["question to be asked"]*nb_questions_per_topic
    #         final_list["section%d"%(nt+1)] = "Short description of the section"
    #         final_list["questions_for_section%d"%(nt+1)]= "%s"%list_questions
        
    #     print(final_list)
    #     print('type:',type(final_list))


    #     """{'section1': 'Short description of the section', 
    #     'questions_for_section1': "['question to be asked', 'question to be asked']", 
    #     'section2': 'Short description of the section', 
    #     'questions_for_section2': "['question to be asked', 'question to be asked']", 
    #     'section3': 'Short description of the section', 
    #     'questions_for_section3': "['question to be asked', 'question to be asked']"}"""

    # return final_list
    
    model_with_structure = model.with_structured_output(FullOutput) #schema(nb_topics,nb_questions_per_topic))
    response = model_with_structure.invoke(prompt)
    result = response.model_dump() 
    print('HHHHHHHHH',result)
    return result
