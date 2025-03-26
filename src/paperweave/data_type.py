from pydantic import BaseModel, Field


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
    questions: list[str] = Field(
        description="list of the question at the begining for the section"
    )


# data format
class Podcast(BaseModel):
    paper: Paper = Field(
        description="paper that is discussed in the podcast", default=Paper()
    )
    transcript: list[Utterance] = Field(
        description="list of utterance, i.e. what people said", default_factory=list
    )
    host: Persona = Field(description="host of the podcast")
    expert: Persona = Field(description="expert who speak in the podcast")
    sections: list[Section] = Field(
        description="list of section to talk during the podcast"
    )


class MyState(BaseModel):
    podcast: Podcast
    index_question: int = Field(default=0)
    index_section: int = Field(default=0)
    questions_asked: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    current_question: str = Field(default="")
    sections: list[str] = Field(default_factory=list)
