import json
from pathlib import Path
import pandas as pd
from typing import Callable
import pprint

def verify_number_topic(data:dict):
    input_nb_topic = data["input"]["nb_topic"]
    output_nb_topic = len(data["podcast"]["topics"])
    return input_nb_topic == output_nb_topic

def verify__number_question(data:dict):
    input_nb_questions = data["input"]["begin_nb_question_per_topic"]
    list_nb_questions = [len(topic["topic_starting_questions"]) for topic in data["podcast"]["topics"]]
    correct_answers = [input_nb_questions==pred_value for pred_value in list_nb_questions]
    return sum(correct_answers)/len(correct_answers)

def create_results(folder:Path, previous_annotation):
    list_results = []
    user_name = input("What is your username for the data annotations :")
    for json_file in folder.glob('*.json'):
        with json_file.open('r') as file:
            previous_annotation_row = previous_annotation.loc[json_file.stem].to_dict() if json_file.stem in previous_annotation.index else {}
            data = json.load(file)
            row_data = data["input"]

            row_data = {f"input__{key}":value for key, value in row_data.items()}
            row_data.update(previous_annotation_row)
            row_data["pipeline_file_name"] = json_file.stem
            row_data["correct_nb_topics"]=verify_number_topic(data)
            row_data["percentage_correct_number_question"] = verify__number_question(data)
            add_annotation(data=data, row_data=row_data, annotation_name="good_initial_topic", user_name=user_name, annotation_function=verify_good_topics)



            list_results.append(row_data)
    results = pd.DataFrame(list_results)
    results = results.set_index("pipeline_file_name")
    return results


def add_annotation(data:dict,row_data:dict, annotation_name:str,user_name:str,annotation_function:Callable):
    annotation_present = True if f"{annotation_name}__{user_name}" in row_data and not pd.isna(
        row_data[f"{annotation_name}__{user_name}"]) else False
    if not annotation_present:
        row_data[f"{annotation_name}__{user_name}"] = annotation_function(data)


def boolean_question():
    anwser_choice = ["y", "n", ""]
    while True:

        user_input = input(f"Please choose one of the following: {', '.join(anwser_choice)}: ").lower()
        if user_input in anwser_choice:
            print(f"You chose: {user_input}")
            break  # Exit the loop
        else:
            print("Invalid choice. Please try again.")
    if user_input=="y":
        return True

    if user_input=="n":
        return False
    if user_input=="":
        return ""


def verify_good_topics(data):
    question="Do you think that this is a good list of topic generate about this paper is good?"

    pred_result = [topic["topic_string"] for topic in data["podcast"]["topics"]]
    print(question)
    pprint.pp(pred_result)
    return boolean_question()


def read_single_run_file(file_path:Path):
    with file_path.open('r') as file:
        data_dict = json.load(file)

    input_data = data_dict["input"]

    output_data = {"topics":[]}

