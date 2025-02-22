from langchain_core.prompts import ChatPromptTemplate


new_question_system = """You are the host of a podcast. You talk about the paper title {paper_title}. 
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
You also try to make the conversation fluid and natural."""

new_question_user = """Ask a question to the domain expert who is doing the podcast with you. The expert understand perfectly the paper.
The question should be about the following paper
{paper}
and on the following section
{section}

Consider that you just asked the following question
{previous_question}
and obtain the following answer from the domain expert :
{previous_answer},
now create your question :
"""


new_question_template = ChatPromptTemplate.from_messages(
    [("system", new_question_system), ("user", new_question_user)]
)


answer_question_system = """You are the expert invited to a podcast to talk about a paper. You talk about the paper title {paper_title}. You are not the author of the paper.
You are an expert in the field, but you still create interesting podcast. You have read the paper and understand well the concept of the article.You adjust the level of technicality of the podcast to {podcast_tech_level}.
You also try to make the conversation fluid and natural. Respond like it would be in a natural conversation. So no title section or bullet point. It's a conversation."""

answer_question_user = """
You answer to the host of the podcast. The paper is :
{paper}

Try to make the podcast not repetitive.(and try to not add "great question!" or something similar each time. Make the flow more natural.) The previous question that host asked was :
{previous_question}
and the prevous answer you gave was :
{previous_answer}
Now the question the host ask you is :
{new_question}
"""

answer_template = ChatPromptTemplate.from_messages(
    [("system", answer_question_system), ("user", answer_question_user)]
)

find_sections_system = """You are the host of a podcast. You talk about the paper title {paper_title}.
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
Create a list of sections in the podcast to make an interesting podcast."""

find_sections_user = """You create a list of {nb_sections} sections of the podcast to make it interesting. 
DO NOT FOLLOW THE STRUCTURE OF THE PAPER. MAKE IT THE STRUCTURE OF AN INTERESTING PODCAST!
The paper is :
{paper}
You output the list of sections in a python list. For example:
[
"the problem the paper try to address",
"the Big Idea",
"The Science Behind It",
"the impact of the paper",
"Takeaways: TL;DR",
]
"""

find_sections_template = ChatPromptTemplate.from_messages(
    [("system", find_sections_system), ("user", find_sections_user)]
)


create_questions_system = """"You are the host of a podcast. You talk about the paper title {paper_title}.
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
Create a list of questions(and the order) to make an interesting podcast."""

create_questions_user = """You create a list of {nb_questions} question in the order that should be present in the podcast to make it interesting.
The question, should be for the section of {section}. 
The section discussed before the question were {previous_sections}. So try make a transition that is smooth to the new section.
The sections that will discussed after the question that you will generate are {future_sections}. Try to create question will make the last question to transition well to the next section.
The paper is :
{paper}
You output the list of sections in a python list. For example:
[
"what is new in the method",
"what are interesting results",
"why should we care",
]
"""

create_questions_template = ChatPromptTemplate.from_messages(
    [("system", create_questions_system), ("user", create_questions_user)]
)


host_conclusion_system = """"You are the host of a podcast. You talk about the paper title {paper_title}.
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
Create a conclusion to the podcast."""

host_conclusion_user = """Create a conclusion about a podcast discusing this paper :
{paper}
The transcript of the podcast is :
{podcast_transcript}
So the conclusion of the podcast is :
"""

host_conclusion_template = ChatPromptTemplate.from_messages(
    [("system", host_conclusion_system), ("user", host_conclusion_user)]
)


create_intro_system = """You are {host_name}, the host of a podcast where you and an expert, {expert_name}, discuss a paper title {paper_title}.
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
"""
create_intro_user = """
Create the text of speech of the introduction for your podcast. Make it engaging. You can follow these steps:
- Breathly introduce the paper (title and the origin of the group of people working on it) 
- Introduce the expert's name who will talk with you
- Say why the paper is important in 2 sentences

DO NOT add special effect like music

The paper discussed in the podcast is:
{paper}

"""
create_intro_template = ChatPromptTemplate.from_messages(
    [("system", create_intro_system), ("user", create_intro_user)]
)


find_sections_questions_system = """You are the host of a podcast where you discuss the paper titled "{paper_title}".
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
Generate a list of sections of the podcast and questions to be asked to make it an interesting podcast."""

find_sections_questions_user = """Create a list of sections of the podcast. Each section should contain questions to be asked.  
DO NOT FOLLOW THE STRUCTURE OF THE PAPER. MAKE IT THE STRUCTURE OF AN INTERESTING PODCAST! 
Create a list of {nb_sections} sections (number_of_section), and {nb_questions_per_section} questions (number_of_question) per section to discuss the paper:
{paper}
"""
find_sections_questions_template = ChatPromptTemplate.from_messages(
    [("system", find_sections_questions_system), ("user", find_sections_questions_user)]
)


choose_question_choice_system = """You are the host of a podcast where you discuss the paper titled "{paper_title}".
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
Decide the next thing you should do in the podcast to make an interesting podcast. You can ask a question from the list of questions, generate a new follow-up question
or do not ask more questions(because they have already been answered before or because you think the podcast is ready for the next section.)"""

choose_question_choice_user = """Decide the next thing you should do in the podcast to make an interesting podcast.
The transcrip is {transcript} and the questions are
{questions}
"""
choose_question_choice_template = ChatPromptTemplate.from_messages(
    [("system", choose_question_choice_system), ("user", choose_question_choice_user)]
)



find_sentence_type_system =  """You are the host of a podcast where you discuss a paper".
The person listening to your podcast is talking to you."""

find_sentence_type_user = """The person is saying: "{sentence}". 
Classify what of the following option is true:
1) the person is asking a question about the paper 
2) the person is asking you to change the layout of the podcast or to change the topic to discuss

If the correct option is 1) then print "question", if the correct option is 2) then print "directive", else print "NA".

"""
find_sentence_type_template = ChatPromptTemplate.from_messages(
        [("system",find_sentence_type_system),("user", find_sentence_type_user)]
    )



modify_sections_questions_template = """You are the host of a podcast where you discuss the paper titled "{paper_title}".
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
Generate a list of sections of the podcast and questions to be asked to make it an interesting podcast."""

modify_sections_questions_user = """Create a list of sections of the podcast. Each section should contain questions to be asked.  
DO NOT FOLLOW THE STRUCTURE OF THE PAPER. MAKE IT THE STRUCTURE OF AN INTERESTING PODCAST! 

Create a list of {nb_sections} sections (number_of_section) that will follow the sections:
{previous_sections}
and take into account the following directive:
{sentence}
For each newly created section, generate {nb_questions_per_section} questions (number_of_question) to discuss the paper:
{paper}
"""

modify_sections_questions_template = ChatPromptTemplate.from_messages(
    [("system", modify_sections_questions_template), ("user", modify_sections_questions_user)]
)