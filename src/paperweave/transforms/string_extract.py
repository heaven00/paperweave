import re
from typing import List


def extract_list(input_str:str)->List[str]:

    regex = r"\[\s*(.*?)\s*\]"
    matches = re.search(regex, input_str, re.DOTALL)
    if not matches:
        return []
    content = matches.group(1)

    items = re.split(r'"\s*,\s*"', content)
    result = [item.strip().strip('"') for item in items if item.strip().strip('"')]

    return result
