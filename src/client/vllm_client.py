
import os

from openai import OpenAI


class VLLMClient:
    def __init__(self):
        self.base_url = os.getenv(
            "VLLM_BASE_URL",
            "http://127.0.0.1:8000/v1",
        )

        self.api_key = os.getenv(
            "VLLM_API_KEY",
            "local-dev-key",
        )

        self.model = os.getenv(
            "VLLM_MODEL",
            "qwen3-8b",
        )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=30.0,
            max_retries=2,
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )

        content = response.choices[0].message.content

        if content is None:
            raise RuntimeError("Model returned no text.")

        return content
