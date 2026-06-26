import json
from agents.base import BaseAgent
from config import MODELS


class SummarizerAgent(BaseAgent):

    def __init__(self, project_name: str):

        super().__init__(
            name="summarizer",
            model_config=MODELS["summarizer"],
            project_name=project_name
        )

    # override run completely
    def run(self, conversation: str):

        system_prompt = self.load_system_prompt()

        memory = self.load_memory()

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
CURRENT MEMORY:
{json.dumps(memory, indent=2)}

NEW CONVERSATION:
{conversation}

Extract updated memory in JSON.
"""
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=2048
        )

        return response.choices[0].message.content