import html
import re


def clean_str(input):
    if not isinstance(input, str):
        return input
    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
