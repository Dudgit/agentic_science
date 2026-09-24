from pathlib import Path
import json


def load_text_files(folder: str) -> str:
    """
    Loads and concatenates all .md files in a folder.
    """
    path = Path(folder)

    if not path.exists():
        return ""

    content = []

    for file in sorted(path.glob("*.md")):
        with open(file, "r", encoding="utf-8") as f:
            content.append(f"# {file.name}\n")
            content.append(f.read())

    return "\n\n".join(content)


def load_json(file_path: str) -> dict:
    path = Path(file_path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path: str, data: dict):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def update_json(file_path: str, new_data: dict):

    old = load_json(file_path)

    for key, value in new_data.items():

        if key not in old:
            old[key] = []

        if isinstance(value, list):

            for item in value:
                if item not in old[key]:
                    old[key].append(item)

        else:
            if value not in old[key]:
                old[key].append(value)

    save_json(file_path, old)


import os

def build_codebase_context(root_dir, target_extensions=None, exclude_dirs=None):
    """Scans the directory and returns a formatted string of the codebase."""
    if target_extensions is None:
        target_extensions = ['.py', '.yaml', '.json', '.md']
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'outputs', 'checkpoints', 'data', 'logs']
        
    context_str = "### CURRENT REPOSITORY STATE ###\n"
    
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in target_extensions):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, root_dir)
                
                context_str += f"\n--- File: {relative_path} ---\n"
                context_str += "```python\n" if file.endswith('.py') else "```\n"
                
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        context_str += f.read()
                except Exception as e:
                    context_str += f"# Error reading file: {str(e)}"
                    
                context_str += "\n```\n"
                
    return context_str

def update_summary_md(file_path: str, new_markdown_content: str):
    """Overwrites the global project summary with the latest state."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_markdown_content)
    except Exception as e:
        print(f"Error saving summary markdown: {e}")

import re

def strip_thoughts(text: str) -> str:
    """Removes the <think>...</think> blocks from the model's response."""
    # re.DOTALL ensures the regex catches multi-line thinking blocks
    final_answer_match = re.split(r'#{0,3}\s*\*?\*?FINAL ANSWER\*?\*?:?#{0,3}\s*', text, flags=re.IGNORECASE)
    if len(final_answer_match) > 1:
        return final_answer_match[-1].strip()

    if "</think>" in text.lower():
        # Case-insensitive split on </think>
        text = re.split(r'</think>', text, flags=re.IGNORECASE)[-1]
    
    cleaned_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'### THOUGHTS? ###.*?###', '###', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    return cleaned_text.strip()

from config import PATHS
def reset_all_memories():
    """Wipes project memory, summarized.md, and all agent memory files on disk and in-memory."""
    
    # Safely reference global variables to avoid NameErrors
    global agents
    
    # 1. Build exact paths using your existing PATHS dictionary
    project_memory_path = os.path.join(PATHS["projects_dir"], "project_memory.json")
    summary_md_path = os.path.join(PATHS["projects_dir"], "summarized.md")

    # 2. Reset Project Memory JSON
    empty_project_mem = {
        "important_decisions": [],
        "implemented_features": [],
        "research_ideas": [],
        "todos": [],
        "papers_to_read": []
    }
    
    os.makedirs(os.path.dirname(project_memory_path), exist_ok=True)
    with open(project_memory_path, "w", encoding="utf-8") as f:
        json.dump(empty_project_mem, f, indent=4)

    # 3. Reset summarized.md
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Current Project State\n\n*Memory reset to clean state.*\n")

    # 4. Reset individual agent memory files & active instances
    for name, agent in agents.items():
        # Clear the in-memory Python variables
        agent.history = []
        agent.memory = {"logs": []}
        
        # Clear the physical JSON files on disk
        if hasattr(agent, "memory_file") and agent.memory_file:
            os.makedirs(os.path.dirname(agent.memory_file), exist_ok=True)
            with open(agent.memory_file, "w", encoding="utf-8") as f:
                json.dump({"logs": []}, f, indent=4)

    print("[System] All project and agent memory files reset to a clean state.")