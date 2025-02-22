from bs4 import BeautifulSoup
import requests
import re
from typing import List
from paperweave.flow_elements.prompt_templates import (
    new_question_template,
    answer_template,
    find_sections_template,
    host_conclusion_template,
    find_sections_questions_template,
    choose_question_choice_template,
    choose_question_index_template,
    reformulate_question_template,
    generate_follow_up_question_template,
)
from paperweave.transforms import extract_list
from paperweave.data_type_direct_llm_call import (
    SectionQuestionLLMOutput,
    LLMResponseQuestionChoice,
    FollowUpQuestion,
    NextQuestionIndex,
    ReformulateQuestion,
)


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


def get_sections_questions(
    model,
    paper_title: str,
    podcast_tech_level: str,
    paper: str,
    nb_sections: int,
    nb_questions_per_section: int,
) -> SectionQuestionLLMOutput:
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "paper": paper,
        "nb_sections": nb_sections,
        "nb_questions_per_section": nb_questions_per_section,
    }

    prompt = find_sections_questions_template.invoke(variables)

    model_with_structure = model.with_structured_output(SectionQuestionLLMOutput)
    response = model_with_structure.invoke(prompt)
    result = response.model_dump()
    return response


def get_question_choice(
    model,
    paper_title: str,
    podcast_tech_level: str,
    transcript: str,
    questions: List[str],
) -> LLMResponseQuestionChoice:
    questions = "\n - ".join(question for question in questions)
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "transcript": transcript,
        "questions": questions,
    }

    prompt = choose_question_choice_template.invoke(variables)

    model_with_structure = model.with_structured_output(LLMResponseQuestionChoice)
    response = model_with_structure.invoke(prompt)
    result = response.model_dump()
    return response


def get_follow_question(
    model,
    paper_title: str,
    podcast_tech_level: str,
    transcript: str,
) -> FollowUpQuestion:
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "transcript": transcript,
    }

    prompt = generate_follow_up_question_template.invoke(variables)

    model_with_structure = model.with_structured_output(FollowUpQuestion)
    response = model_with_structure.invoke(prompt)
    return response.question


def get_next_question_index(
    model,
    paper_title: str,
    podcast_tech_level: str,
    transcript: str,
    questions: List[str],
) -> NextQuestionIndex:
    questions = [f"{index}-{question}" for index, question in enumerate(questions)]
    questions = "\n ".join(question for question in questions)
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "transcript": transcript,
        "questions": questions,
    }
    prompt = choose_question_index_template.invoke(variables)

    model_with_structure = model.with_structured_output(NextQuestionIndex)
    response = model_with_structure.invoke(prompt)
    return response.question_index


def reformulate_question(
    model,
    paper_title: str,
    podcast_tech_level: str,
    transcript: str,
    question: str,
) -> ReformulateQuestion:
    variables = {
        "paper_title": paper_title,
        "podcast_tech_level": podcast_tech_level,
        "transcript": transcript,
        "question": question,
    }

    prompt = reformulate_question_template.invoke(variables)

    model_with_structure = model.with_structured_output(ReformulateQuestion)
    response = model_with_structure.invoke(prompt)
    return response.question
