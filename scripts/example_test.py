from paperweave.example1 import example_function

print(example_function(3))


from dotenv import load_dotenv
from pathlib import Path
import os
env_file = Path(__file__).parent.parent / ".env"

# Load the .env file
load_dotenv(env_file)

OPENAI_KEY = os.getenv('OPENAI_KEY')

print(OPENAI_KEY)
