from openai import OpenAI


class VLLMBackend:
    def __init__(self, base_url="http://localhost:8000/v1", model="Qwen",max_tokens= 512, temperature=0.7):
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"   # vLLM doesn't need real key
        )
        print(f"{base_url} {model}")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content