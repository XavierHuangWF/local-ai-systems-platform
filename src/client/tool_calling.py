
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage


def request_with_tools(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> ChatCompletionMessage:
    """
    Request a model response with available tool definitions.

    This function does not execute tools.
    """

    if not tools:
        raise ValueError(
            "At least one tool definition is required."
        )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    if not response.choices:
        raise RuntimeError(
            "Model returned no completion choices."
        )

    return response.choices[0].message
