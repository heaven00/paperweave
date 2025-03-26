from pydantic import BaseModel, Field
from typing import List

class Paper(BaseModel):
    text: str = Field(description="this is the string of the paper", default="")
    code: str = Field(
        description="this is the code of the paper(ex: 2332.2393)", default=""
    )
    title: str = Field(description="this is the title of the paper", default="")


class PaperCode(BaseModel):
    code: str = Field(description="this is the string of the paper", default="")


class Persona(BaseModel):
    name: str = Field(description="name of the persona", default="")
    personality: str = Field(
        description="describe the personality of the persona", default=""
    )


class Utterance(BaseModel):
    persona: Persona = Field(
        description="person who speak", default_factory=lambda: Persona()
    )
    speach: str = Field(description="what the persona have said", default="")
    category: str = Field(description="what kind of uttenrence(question, answer, ...)")


class Section(BaseModel):
    section_subject: str = Field(description="description of the section")
    questions: List[str] = Field(
        description="list of the question at the begining for the section"
    )


# data format
class Podcast(BaseModel):
    paper: Paper = Field(...)
    transcript: List[Utterance] = Field(default_factory=list)
    host: Persona = Field(
        default_factory=lambda: Persona(name="Default Host")
    )
    expert: Persona = Field(
        default_factory=lambda: Persona(name="Default Expert")
    )
    sections: List[Section] = Field(
        default_factory=list
    )


class MyState(BaseModel):
    podcast: Podcast
    index_question: int = Field(default=0)
    index_section: int = Field(default=0)
    questions_asked: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    current_question: str = Field(default="")
    sections: List[str] = Field(default_factory=list)
