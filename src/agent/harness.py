
from src.client.vllm_client import VLLMClient
from src.agent.token_counter import TokenCounter


SYSTEM_PROMPT = (
    "You are a concise AI systems assistant. "
    "Answer technical questions clearly and accurately."
)

MAX_HISTORY_TURNS = 6


class AgentHarness:
    def __init__(self):
        self.model = VLLMClient()
        self.token_counter = TokenCounter()

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

        # Keep at most five previous exchanges.
        recent_history = self.messages[1:][
            -2 * (MAX_HISTORY_TURNS - 1):
        ]

        # Prepare the next model request.
        pending_messages = [
            self.messages[0],
            *recent_history,
            {
                "role": "user",
                "content": user_input,
            },
        ]

        # Check the actual token budget.
        while not self.token_counter.fits(pending_messages):

            # Only the system message and new user message remain.
            if len(pending_messages) == 2:
                raise ValueError(
                    "Input exceeds the available token budget."
                )

            # Remove the oldest user/assistant exchange.
            del pending_messages[1:3]

        # Call the model only after the input fits.
        response = self.model.chat(pending_messages)

        # Save the conversation only after success.
        self.messages = [
            *pending_messages,
            {
                "role": "assistant",
                "content": response,
            },
        ]

        return response

    def reset(self) -> None:
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
