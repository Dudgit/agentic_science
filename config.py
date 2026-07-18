
CURRENT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

MODELS = {

    "qa": {
        "model_name": "Qwen/Qwen3.6-35B-A3B-FP8",
        "base_url": "http://vllm.default:8000/v1"
    },

    "coding": {
        "model_name": "Qwen/Qwen3.6-35B-A3B-FP8",
        "base_url": "http://vllm.default:8000/v1",
        "max_tokens": 2048,
        "temperature": 0.7
    },

    "writing": {
        "model_name": "mistral-medium-3.5:128b",
        "base_url": "http://ollama.default:11434/v1"
    },

    "critic": {
        "model_name": "gpt-oss:120b",
        "base_url": "http://ollama.default:11434/v1"
    },

    "summarizer": {
        "model_name": "gpt-oss:20b",
        "base_url": "http://ollama.default:11434/v1"
    },

    "brainstorm": {
        "model_name": "nemotron-3-super:120b",
        "base_url": "http://ollama.default:11434/v1"
    }
}

PATHS = {
    "prompt_dir": "prompts",
    "projects_dir": "/v/hqos8c/agentic_workflow/agentic_science/projects"
}