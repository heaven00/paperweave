import re
from typing import List


def extract_list(input_str:str)->List[str]:

    regex = r"\[\s*(.*?)\s*\]"
    matches = re.search(regex, input_str, re.DOTALL)
    if not matches:
        return []
    content = matches.group(1)
    # Split the content into a list by commas, stripping any whitespace
    result = [item.strip().strip('"') for item in content.split(',') if item.strip().strip('"')]

    return result
