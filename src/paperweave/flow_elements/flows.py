from bs4 import BeautifulSoup
import requests
import re
from paperweave.flow_elements.prompt_templates import (
    new_question_template,
    answer_template,
    find_sections_template,
    host_conclusion_template,
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
    section,
    previous_question,
    previous_answer,
):
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "paper": paper,
        "section": section,
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


def loop_question_answer_on_section(
    model,
    paper_title,
    podcast_tech_level,
    paper,
    section,
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
            section,
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


def get_sections(model, paper_title, podcast_tech_level, paper, nb_sections):
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "paper": paper,
        "nb_sections": nb_sections,
    }

    # Format the prompt with the variables
    prompt = find_sections_template.invoke(variables)

    # Get the model's response
    response = model.invoke(prompt)

    result = extract_list(response.content)
    return result
