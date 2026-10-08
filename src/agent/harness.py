
from src.client.vllm_client import VLLMClient


SYSTEM_PROMPT = (
    "You are a concise AI systems assistant. "
    "Answer technical questions clearly and accurately."
)


class AgentHarness:
    def __init__(self):
        self.model = VLLMClient()

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

    def run(self, user_input: str) -> str:
        user_input = user_input.strip()

        if not user_input:
            raise ValueError("User input cannot be empty.")

        self.messages.append({
            "role": "user",
            "content": user_input,
        })

        try:
            response = self.model.chat(self.messages)
        except Exception:
            self.messages.pop()
            raise

        self.messages.append({
            "role": "assistant",
            "content": response,
        })

        return response

    def reset(self) -> None:
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
