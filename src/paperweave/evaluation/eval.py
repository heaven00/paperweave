import json
from pathlib import Path
import pandas as pd
from typing import Callable
import pprint


def verify_number_section(data: dict) -> bool:
    input_nb_section = data["input"]["nb_section"]
    output_nb_section = len(data["podcast"]["sections"])
    return input_nb_section == output_nb_section


def verify__number_question(data: dict) -> float:
    input_nb_questions = data["input"]["begin_nb_question_per_section"]
    list_nb_questions = [
        len(section["section_starting_questions"])
        for section in data["podcast"]["sections"]
    ]
    correct_answers = [
        input_nb_questions == pred_value for pred_value in list_nb_questions
    ]
    return sum(correct_answers) / len(correct_answers)


def create_results(folder: Path, previous_annotation: pd.DataFrame) -> pd.DataFrame:
    list_results = []
    user_name = input("What is your username for the data annotations :")
    for json_file in folder.glob("*.json"):
        with json_file.open("r") as file:
            previous_annotation_row = (
                previous_annotation.loc[json_file.stem].to_dict()
                if json_file.stem in previous_annotation.index
                else {}
            )
            data = json.load(file)
            row_data = data["input"]

            row_data = {f"input__{key}": value for key, value in row_data.items()}
            row_data.update(previous_annotation_row)
            row_data["pipeline_file_name"] = json_file.stem
            row_data["correct_nb_sections"] = verify_number_section(data)
            row_data["percentage_correct_number_question"] = verify__number_question(
                data
            )
            add_annotation(
                data=data,
                row_data=row_data,
                annotation_name="completeness_initial_section",
                user_name=user_name,
                annotation_function=verify_completeness_sections,
            )
            add_annotation(
                data=data,
                row_data=row_data,
                annotation_name="logical_sequence_initial_section",
                user_name=user_name,
                annotation_function=verify_logical_sequence_sections,
            )
            add_annotation(
                data=data,
                row_data=row_data,
                annotation_name="relevent_question_for_initial_section",
                user_name=user_name,
                annotation_function=verify_relevent_question_for_section,
            )
            add_annotation(
                data=data,
                row_data=row_data,
                annotation_name="correct_answer_to_question",
                user_name=user_name,
                annotation_function=verify_answer_question,
            )

            list_results.append(row_data)
    results = pd.DataFrame(list_results)
    results = results.set_index("pipeline_file_name")
    return results


def add_annotation(
    data: dict,
    row_data: dict,
    annotation_name: str,
    user_name: str,
    annotation_function: Callable,
):
    annotation_present = (
        True
        if f"{annotation_name}__{user_name}" in row_data
        and not pd.isna(row_data[f"{annotation_name}__{user_name}"])
        else False
    )
    if not annotation_present:
        row_data[f"{annotation_name}__{user_name}"] = annotation_function(data)


def boolean_question(force_answer:bool=False) -> bool | str:
    anwser_choice = ["y", "n"]
    if not force_answer:
        anwser_choice.append("")
    while True:
        user_input = input(
            f"Please choose one of the following: {', '.join(anwser_choice)}: "
        ).lower()
        if user_input in anwser_choice:
            print(f"You chose: {user_input}")
            break  # Exit the loop
        else:
            print("Invalid choice. Please try again.")
    if user_input == "y":
        return True

    if user_input == "n":
        return False
    if user_input == "":
        return ""


def verify_completeness_sections(data: dict) -> bool | str:
    print()
    paper_code = data["podcast"]["paper"]["code"]
    paper_title = data["podcast"]["paper"]["title"]
    question = f"For the paper {paper_code}:{paper_title}, do you think that these sections cover well the paper(i.e.) completeness?"

    pred_result = [section["section_string"] for section in data["podcast"]["sections"]]
    pprint.pp(question)
    pprint.pp(pred_result)
    return boolean_question()


def verify_logical_sequence_sections(data: dict) -> bool | str:
    print()
    paper_code = data["podcast"]["paper"]["code"]
    paper_title = data["podcast"]["paper"]["title"]
    question = f"For the paper {paper_code}:{paper_title}, do you think that these sections are in logical sequence?"

    pred_result = [section["section_string"] for section in data["podcast"]["sections"]]
    pprint.pp(question)
    pprint.pp(pred_result)
    return boolean_question()

def verify_relevent_question_for_section(data:dict)-> float:
    print()
    paper_code = data["podcast"]["paper"]["code"]
    paper_title = data["podcast"]["paper"]["title"]
    context = f"For the paper {paper_code}:{paper_title}, verify that the questions are relevent for the section?"
    pprint.pp(context)

    sections = data["podcast"]["sections"]
    results = []
    for section in sections:
        print()
        section_questions = section["section_starting_questions"]
        section_description = section["section_string"]
        question = f"Do you think that for this \n {section_description}\nDo you think this questions are relevent"
        pprint.pp(question)
        pprint.pp(section_questions)
        results.append(boolean_question(force_answer=True))
    percentage_section_with_relevent_questions = sum(results)/len(results)
    return percentage_section_with_relevent_questions


def verify_answer_question(data:dict)-> float:
    print()
    paper_code = data["podcast"]["paper"]["code"]
    paper_title = data["podcast"]["paper"]["title"]
    context = f"For the paper {paper_code}:{paper_title}, verify that the answer respond correctly to the question?"
    pprint.pp(context)

    utterances = data["podcast"]["transcript"]
    results = []
    question = ""
    answer = ""
    for utterance in utterances:
        print()
        if utterance.get("category", "")=="question":
            question = utterance.get("speach", "")
        if (question) and (utterance.get("category", "")=="answer"):
            answer = utterance.get("speach", "")
            present_question = "Do you think that for this question :"
            pprint.pp(present_question)
            pprint.pp(question)
            present_answer = "that this answer is correct and addapted:"
            pprint.pp(present_answer)
            pprint.pp(answer)
            results.append(boolean_question(force_answer=True))
            answer=""
            question=""

    percentage_section_with_relevent_questions = sum(results) / len(results)
    return percentage_section_with_relevent_questions


