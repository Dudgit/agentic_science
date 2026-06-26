import json
import re


def parse_json_response(text):

    # Remove markdown fences
    text = text.replace("```json", "")
    text = text.replace("```", "")

    # Remove Qwen thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match is None:
        raise ValueError("No JSON object found")

    json_text = match.group(0)

    return json.loads(json_text)