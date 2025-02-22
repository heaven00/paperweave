"""
This a file to put structure data that will be directly call to the LLM as structure output
"""

from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class Section(BaseModel):
    questions: List[str] = Field(
        description="list of questions of size number_of_question"
    )
    section_subject: str = Field(description="description of podcast section")
    number_of_question: int = Field(description="number of questions in the section")


class SectionQuestionLLMOutput(BaseModel):
    sections: List[Section] = Field(
        description="list of sections of size number_of_section"
    )
    number_of_section: int = Field(description="number of sections in the podcast")


class QuestionChoice(Enum):
    FIRST_QUESTION = "use first question to ask next"
    NEW_QUESTION = "generate a follow up question base on previous answer"
    OTHER_QUESTION = "use any question from the list of question"
    NO_QUESTION = "do not ask any more questions"


class LLMResponseQuestionChoice(BaseModel):
    question_choice: QuestionChoice = Field(
        description="Choice for the host of the podcast of what question he should ask"
    )


class FollowUpQuestion(BaseModel):
    question: str = Field(description="Follow up question that the host ask.")
    host_name: str = Field(description="name of the host of the podcast")


class NextQuestionIndex(BaseModel):
    question_index: int = Field(
        description="index of the question the host want to ask."
    )


class ReformulateQuestion(BaseModel):
    question: str = Field(description="Reformulated question that the host will ask.")
    host_name: str = Field(description="name of the host of the podcast")
