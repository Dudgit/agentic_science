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
        memory_file="memory.json",
        project_memory_file="project_memory.json"
    ):
        self.name = name
        self.backend = backend

        self.prompt_dir = Path(prompt_dir) if prompt_dir else None
        self.system_prompt = self.load_prompt(prompt_dir)

        self.memory_file = Path(memory_file)
        self.memory = self.load_memory()
        self.project_memory = Path(project_memory_file)
        self.project_memory = self.load_project_memory()

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
    def load_project_memory(self):
        if self.project_memory.exists():
            return load_json(self.project_memory)
        return {}


    def save_memory(self):
        save_json(self.memory_file, self.memory)

    # ---------------- CORE ----------------
    def build_messages(self, user_input):
        dynamic_system_prompt = self.system_prompt

        # 1. Inject the Global Project Memory (The "Clipboard")
        if self.project_memory:
            formatted_project_mem = json.dumps(self.project_memory, indent=2)
            dynamic_system_prompt += f"\n\n### GLOBAL PROJECT STATE ###\n```json\n{formatted_project_mem}\n```\n"

        # 2. Inject the Agent's Personal Memory (The "Archive")
        # We ONLY grab the last 3 interactions to prevent token explosion!
        if self.memory and "logs" in self.memory:
            recent_logs = self.memory["logs"][-3:]
            formatted_logs = json.dumps(recent_logs, indent=2)
            dynamic_system_prompt += f"\n\n### YOUR RECENT ACTIONS ###\n```json\n{formatted_logs}\n```\n"

        return [
            {"role": "system", "content": dynamic_system_prompt},
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