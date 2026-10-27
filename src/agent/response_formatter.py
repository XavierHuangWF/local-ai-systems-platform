
import re


_THINK_BLOCK = re.compile(
    r"\A\s*<think>.*?</think>\s*",
    flags=re.DOTALL,
)

_THINK_TAG = re.compile(r"</?think>")


def clean_model_response(content: str) -> str:
    """
    Remove leading Qwen3 thinking blocks from a model response.

    Reject malformed thinking tags and empty final answers.
    """

    if not isinstance(content, str):
        raise TypeError("Model response must be a string.")

    remaining = content

    # Remove complete thinking blocks at the beginning.
    while True:
        match = _THINK_BLOCK.match(remaining)

        if match is None:
            break

        remaining = remaining[match.end():]

    # Fail closed rather than exposing unexpected reasoning tags.
    if _THINK_TAG.search(remaining):
        raise ValueError(
            "Model response contains malformed or "
            "unexpected thinking tags."
        )

    answer = remaining.strip()

    if not answer:
        raise ValueError(
            "Model returned no final answer."
        )

    return answer
