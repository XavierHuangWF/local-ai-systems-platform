
from transformers import AutoTokenizer


class TokenCounter:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B-AWQ"
        )

        self.max_context_tokens = 8192
        self.reserved_output_tokens = 512

    def count(self, messages: list[dict[str, str]]) -> int:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        token_ids = encoded["input_ids"]

        return len(token_ids)


    def fits(
        self,
        messages: list[dict[str, str]],
        reserved_output_tokens: int | None = None,
    ) -> bool:
        """Check whether input and reserved output fit the context."""

        # Preserve the original 512-token default.
        if reserved_output_tokens is None:
            reserved_output_tokens = self.reserved_output_tokens

        if (
            isinstance(reserved_output_tokens, bool)
            or not isinstance(reserved_output_tokens, int)
            or not 0 < reserved_output_tokens < self.max_context_tokens
        ):
            raise ValueError("Invalid output token reservation.")

        input_tokens = self.count(messages)

        available_tokens = (
            self.max_context_tokens - reserved_output_tokens
        )

        return input_tokens <= available_tokens

