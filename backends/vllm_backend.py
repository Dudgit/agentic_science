from openai import OpenAI


class VLLMBackend:
    def __init__(self, base_url="http://localhost:8000/v1", model="Qwen"):
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"   # vLLM doesn't need real key
        )
        self.model = model

    def generate(self, messages, max_tokens=512, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content