
import json
from typing import Any

from openai import OpenAI

from src.client.tool_calling import request_with_tools
from src.tools.executor import ToolExecutor


MAX_TOOL_CALLS = 3


def run_tool_turn(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    executor: ToolExecutor,
) -> str:
    """
    Run one model-to-tool-to-model interaction.

    Tool execution is limited to one round.
    The original messages list is not modified.
    """

    tools = executor.get_tool_definitions()

    if not tools:
        raise ValueError("No tools are registered.")

    # Step 1: Ask the model whether it needs a tool.
    assistant = request_with_tools(
        client=client,
        model=model,
        messages=messages,
        tools=tools,
    )

    tool_calls = assistant.tool_calls or []

    # Step 2: Handle an ordinary response without tools.
    if not tool_calls:
        if not assistant.content:
            raise RuntimeError(
                "Model returned neither text nor tool calls."
            )

        return assistant.content

    # Prevent an unbounded number of tool executions.
    if len(tool_calls) > MAX_TOOL_CALLS:
        raise ValueError(
            "Model requested too many tools."
        )

    # Build a new conversation without modifying the original.
    updated_messages = list(messages)

    assistant_tool_calls = []

    for call in tool_calls:
        assistant_tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        )

    updated_messages.append(
        {
            "role": "assistant",
            "content": assistant.content,
            "tool_calls": assistant_tool_calls,
        }
    )

    # Step 3: Validate and execute the requested tools.
    for call in tool_calls:
        try:
            arguments = json.loads(
                call.function.arguments
            )
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "Model generated invalid JSON tool arguments."
            ) from exc

        # The executor validates arguments before dispatch.
        result = executor.execute(
            name=call.function.name,
            arguments=arguments,
        )

        updated_messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(
                    result,
                    allow_nan=False,
                ),
            }
        )

    # Step 4: Ask the model to interpret tool results.
    # No tools are offered in this second request.
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
