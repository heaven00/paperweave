import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_chat_model() -> ChatOllama | ChatOpenAI:
    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
        return ChatOpenAI(model="gpt-4o-mini")
    else:
        return ChatOllama(model="mistral-small:latest")
