
import json
from typing import Any

from openai import OpenAI

from src.client.tool_calling import request_with_tools
from src.tools.executor import ToolExecutor
import logging

MAX_TOOL_CALLS = 3
MAX_TOOL_ROUNDS = 3


def run_tool_turn(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    executor: ToolExecutor,
    max_rounds: int = 1,
) -> str:
    """Execute up to max_rounds model-directed tool-selection rounds.

    A round may contain up to MAX_TOOL_CALLS function calls. After the
    final permitted round, request an answer with tools disabled.
    The original messages list is not modified. Tool side effects are
    not transactional if a later tool fails.
    """
    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or not 1 <= max_rounds <= MAX_TOOL_ROUNDS
    ):
        raise ValueError(
            f"max_rounds must be between 1 and {MAX_TOOL_ROUNDS}."
        )

    tools = executor.get_tool_definitions()
    if not tools:
        raise ValueError("No tools are registered.")

    updated_messages = list(messages)

    for round_number in range(1, max_rounds + 1):
        # Allow Qwen3 to choose new tools using previous tool results.
        assistant = request_with_tools(
            client=client,
            model=model,
            messages=updated_messages,
            tools=tools,
        )
        tool_calls = assistant.tool_calls or []
        
        logger = logging.getLogger("local_ai.tool_execution")

        if tool_calls:
            tool_names = [
                call.function.name
                for call in tool_calls
            ]

            logger.info(
                "Round %d requested: %s",
                round_number,
                ", ".join(tool_names),
            )
        else:
            logger.info(
                "Round %d: Model requested no additional tools",
                round_number,
            )

        # The model may finish without requesting more tools.
        if not tool_calls:
            if not assistant.content:
                raise RuntimeError(
                    "Model returned neither text nor tool calls."
                )
            return assistant.content

        if len(tool_calls) > MAX_TOOL_CALLS:
            raise ValueError("Model requested too many tools.")

        assistant_tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]

        updated_messages.append(
            {
                "role": "assistant",
                "content": assistant.content,
                "tool_calls": assistant_tool_calls,
            }
        )

        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    "Model generated invalid JSON tool arguments."
                ) from exc

            result = executor.execute(
                name=call.function.name,
                arguments=arguments,
            )

            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, allow_nan=False),
                }
            )

    # The maximum number of tool-selection rounds has been reached.
    # Ask for a final answer without offering another tool round.
    final_response = client.chat.completions.create(
        model=model,
        messages=updated_messages,
    )

    if not final_response.choices:
        raise RuntimeError(
            "Model returned no final completion choices."
        )

    final_message = final_response.choices[0].message

    if final_message.tool_calls:
        raise RuntimeError(
            "Unexpected tool calls in final response."
        )

    if not final_message.content:
        raise RuntimeError(
            "Model returned an empty final answer."
        )

    return final_message.content
