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