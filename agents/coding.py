from agents.base import BaseAgent
from config import MODELS
from utils.io import build_codebase_context
import os


class CodingAgent(BaseAgent):
    # 1. Accept all the standard BaseAgent arguments, PLUS repo_path
    def __init__(self, name="coding", backend=None, prompt_dir="prompts/coding", memory_file=None, repo_path=None):
        
        # 2. Pass the standard arguments straight up to the parent class
        super().__init__(
            name=name,
            backend=backend,
            prompt_dir=prompt_dir,
            memory_file=memory_file
        )
        
        # 3. Store the new custom variable for this specific agent
        if repo_path is None:
            repo_path = os.getcwd()
        self.repo_path = repo_path
    def build_messages(self, user_input):
        # 1. Read the live state of your medical physics repo
        live_context = build_codebase_context(self.repo_path)
        
        # 2. Append it to the base system prompt
        dynamic_system_prompt = f"{self.system_prompt}\n\n{live_context}"
        strict_user_input = (
            f"{user_input}\n\n"
            f"Respond directly with the solution. Do not print any planning text or summarize the codebase."
        )
        messages = [
            {"role": "system", "content": dynamic_system_prompt},
            {"role": "user", "content": "Create a simple Python function to add two numbers."},
            {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"},
            {"role": "user", "content": user_input}
        ]
        # 3. Return the standard OpenAI/vLLM message format
        return messages