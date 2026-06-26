# agents/base.py

import json
from pathlib import Path
from utils.io import load_text_files, load_json, save_json


class BaseAgent:
    def __init__(
        self,
        name,
        backend,
        prompt_dir=None,
        memory_file="memory.json"
    ):
        self.name = name
        self.backend = backend

        self.prompt_dir = Path(prompt_dir) if prompt_dir else None
        self.system_prompt = self.load_prompt(prompt_dir)

        self.memory_file = Path(memory_file)
        self.memory = self.load_memory()

        self.history = []

    # ---------------- PROMPT ----------------
    def load_prompt(self, prompt_dir):
        if not prompt_dir:
            return ""

        return load_text_files(prompt_dir)

    # ---------------- MEMORY ----------------
    def load_memory(self):
        if self.memory_file.exists():
            return load_json(self.memory_file)
        return {}

    def save_memory(self):
        save_json(self.memory_file, self.memory)

    # ---------------- CORE ----------------
    def build_messages(self, user_input):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

    def run(self, user_input):
        messages = self.build_messages(user_input)

        response = self.backend.generate(messages)

        self.history.append({
            "input": user_input,
            "output": response
        })

        self.update_memory(user_input, response)
        self.save_memory()

        return response

    def update_memory(self, user_input, response):
        self.memory.setdefault("logs", []).append({
            "input": user_input,
            "output": response
        })