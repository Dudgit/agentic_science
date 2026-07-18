# agents/base.py

import json
from pathlib import Path
from utils.io import load_text_files, load_json, save_json
from utils.web import search_internet
import datetime

class BaseAgent:
    def __init__(
        self,
        name,
        backend,
        prompt_dir=None,
        memory_file="memory.json",
        project_memory_file="project_memory.json",
        allow_search=False
    ):
        self.name = name
        self.backend = backend
        self.allow_search = allow_search

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
        if not self.allow_search:
            messages = self.build_messages(user_input)

            response = self.backend.generate(messages)

            self.history.append({
                "input": user_input,
                "output": response
            })

            self.update_memory(user_input, response)
            self.save_memory()

            return response
        current_year = datetime.datetime.now().year
        react_instructions = (
            f"Current Year: {current_year}\n\n"
            f"User request: {user_input}\n\n"
            "INSTRUCTIONS FOR TOOL USE:\n"
            "- If you need to look up current documentation, library source implementations, version changes, or errors, reply with ONLY the exact phrase: SEARCH: [your query here].\n"
            "- If you do not need to search, or if you have gathered enough information from the web results, just provide the final response normally without the SEARCH prefix."
        )

        # Build messages using the sub-class override (which includes codebase context!)
        messages = self.build_messages(react_instructions)
        max_loops = 3

        for _ in range(max_loops):
            response = self.backend.generate(messages)

            # Did the agent decide it needs to look something up?
            if "SEARCH:" in response:
                query = response.split("SEARCH:")[1].strip()
                print(f"\n[{self.name.upper()}] Autonomous Web Search: {query}")

                # Run the DuckDuckGo utility
                web_results = search_internet(query)

                # Append the loop history so the model can read its own tool output
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"OBSERVATION FROM WEB:\n{web_results}\n\nBased on this, either provide the final solution or output another SEARCH: command if you need more details."
                })
                continue # Restart the loop with the new context loaded!
            
            else:
                # No search command found; the agent has provided the final response
                break

        # Save the interaction to memory once the loop concludes
        self.history.append({"input": user_input, "output": response})
        self.update_memory(user_input, response)
        self.save_memory()
        return response

    def update_memory(self, user_input, response):
        self.memory.setdefault("logs", []).append({
            "input": user_input,
            "output": response
        })