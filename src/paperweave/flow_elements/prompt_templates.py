from langchain_core.prompts import ChatPromptTemplate




new_question_system = """You are the host of a podcast. You talk about the paper title {paper_title}. 
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
You also try to make the conversation fluid and natural."""

new_question_user = """Ask a question to the domain expert who is doing the podcast with you. The expert understand perfectly the paper.
The question should be about the following paper
{paper}
and on the following topic
{topic}

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

The previous question that host asked was :
{previous_question}
and the prevous answer you gave was :
{previous_answer}
Now the question the host ask you is :
{new_question}
"""

answer_template = ChatPromptTemplate.from_messages(
        [("system", answer_question_system), ("user", answer_question_user)]
    )

find_topics_system = """"You are the host of a podcast. You talk about the paper title {paper_title}.
You are an expert in the field, but you still create interesting podcast. You adjust the level of technicality of the podcast to {podcast_tech_level}.
Create a list of topics(and the order) to make an interesting podcast."""

find_topics_user = """You create a list of {nb_topics} topics in the order that should be present in the podcast to make it interesting.
The paper is :
{paper}
You output the list of topics in a python list. For example:
[
context the paper appear,
what is new in the method,
what are interesting results,
why should we care,
]
"""

find_topics_template = ChatPromptTemplate.from_messages(
        [("system", find_topics_system), ("user", find_topics_user)]
    )
