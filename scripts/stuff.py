import openai

from dotenv import load_dotenv
from pathlib import Path
from typing import Annotated
import os

env_file = Path(__file__).parent.parent / ".env"

# Load the .env file
load_dotenv(env_file)

OPENAI_KEY = os.getenv("OPENAI_KEY")

# Set OpenAI API key
openai.api_key = "your-openai-api-key"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key=OPENAI_KEY, model_name="gpt-4o-mini")


print(llm.invoke("what is the meaning of life"))
