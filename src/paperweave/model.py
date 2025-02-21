import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_chat_model() -> ChatOllama | ChatOpenAI:
    if "OPENAI_KEY" in os.environ and os.environ["OPENAI_KEY"]:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]
        return ChatOpenAI(model="gpt-4o-mini")
    else:
        print("no OPENAI_KEY set, using llama3.1:latest")
        return ChatOllama(model="llama3.1:latest")
