import json
from pathlib import Path
import pandas as pd

def verify_number_topic(data:dict):
    input_nb_topic = data["input"]["nb_topic"]
    output_nb_topic = len(data["podcast"]["topics"])
    return input_nb_topic == output_nb_topic

def verify__number_question(data:dict):
    input_nb_questions = data["input"]["begin_nb_question_per_topic"]
    list_nb_questions = [len(topic["topic_starting_questions"]) for topic in data["podcast"]["topics"]]
    correct_answers = [input_nb_questions==pred_value for pred_value in list_nb_questions]
    return sum(correct_answers)/len(correct_answers)

def create_results(folder:Path):
    list_results = []
    for json_file in folder.glob('*.json'):
        with json_file.open('r') as file:
            data = json.load(file)
            df_data = data["input"]
            df_data = {f"input__{key}":value for key, value in df_data.items()}
            df_data["correct_nb_topics"]=verify_number_topic(data)
            df_data["percentage_correct_number_question"] = verify__number_question(data)

            list_results.append(df_data)
    results = pd.DataFrame(list_results)
    return results



def read_single_run_file(file_path:Path):
    with file_path.open('r') as file:
        data_dict = json.load(file)

    input_data = data_dict["input"]

    output_data = {"topics":[]}

