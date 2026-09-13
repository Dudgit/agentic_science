
CURRENT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

MODELS = {

    "qa": {
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "base_url": "http://vllm.default:8000/v1",
        "max_tokens": 4096,
    },

    "coding": {
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "base_url": "http://vllm.default:8000/v1",
        "max_tokens": 4096,
        "temperature": 0.7
    },

    "writing": {
        "model": "mistral-medium-3.5:128b",
        "base_url": "http://ollama.default:11434/v1"
    },

    "critic": {
        "model": "gpt-oss:120b",
        "base_url": "http://ollama.default:11434/v1",
        "max_tokens": 2048
    },

    "summarizer": {
        "model": "gpt-oss:20b",
        "base_url": "http://ollama.default:11434/v1"
    },

    "researcher": {
        "model": "nemotron-3-super:120b",
        "base_url": "http://ollama.default:11434/v1"
    }
}

PATHS = {
    "prompt_dir": "prompts",
    "projects_dir": "/v/ijziqt/agentic_science/projects"
    
}