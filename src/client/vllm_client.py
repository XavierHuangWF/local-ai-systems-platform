import os
import time

from openai import OpenAI

BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("VLLM_API_KEY", "local-dev-key")
MODEL = os.getenv("VLLM_MODEL", "qwen3-8b")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


def main():
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a concise AI systems engineer.",
            },
            {
                "role": "user",
                "content": "are you smart enougn to answer my qeustion? ",
            },
        ],
        temperature=0.7,
        max_tokens=256,
    )

    elapsed = time.perf_counter() - start

    print(response.choices[0].message.content)
    print()
    print("Prompt tokens:", response.usage.prompt_tokens)
    print("Completion tokens:", response.usage.completion_tokens)
    print("Total tokens:", response.usage.total_tokens)
    print(f"Latency: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
