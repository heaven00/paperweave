import re
from typing import List

from paperweave.data_type import Utterance


def extract_list(input_str: str) -> List[str]:
    regex = r"\[\s*(.*?)\s*\]"
    matches = re.search(regex, input_str, re.DOTALL)
    if not matches:
        return []
    content = matches.group(1)

    items = re.split(r'"\s*,\s*"', content)
    result = [item.strip().strip('"') for item in items if item.strip().strip('"')]

    return result


def transcript_to_full_text(transcript: List[Utterance]):
    text = ""
    for utterance in transcript:
        name = utterance["persona"]["name"]
        speach = utterance["speach"]
        new_text = f"{name}:\n{speach}\n"
        text += new_text
    return text
