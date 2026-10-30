
import json
from typing import Any

from openai import OpenAI

from src.client.tool_calling import request_with_tools
from src.tools.executor import ToolExecutor

from src.agent.token_counter import TokenCounter
import logging

from src.agent.source_attribution import check_source_attribution

MAX_TOOL_CALLS = 3
MAX_TOOL_ROUNDS = 3


def report_source_attribution(
    answer: str,
    passages: list[dict],
) -> None:
    """Log whether answer references match retrieved evidence."""

    if not passages:
        return

    result = check_source_attribution(
        answer=answer,
        passages=passages,
    )

    logger = logging.getLogger("local_ai.tool_execution")

    if result.matched:
        logger.info(
            "Source references matched retrieved passages: %s",
            ", ".join(result.matched),
        )

    if result.unsupported:
        logger.warning(
            "Unsupported source references: %s",
            ", ".join(result.unsupported),
        )

    if not result.referenced:
        logger.warning(
            "Document search was used, but no Markdown "
            "source reference was found in the answer."
        )


def run_tool_turn(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    executor: ToolExecutor,
    max_rounds: int = 1,
    token_counter: TokenCounter | None = None,
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
    retrieved_passages: list[dict] = []


    for round_number in range(1, max_rounds + 1):


        # Count the complete request, including tool definitions
        # and results from previous execution rounds.
        if token_counter is not None:
            input_tokens = token_counter.count_with_tools(
                updated_messages,
                tools,
            )

            available_tokens = (
                token_counter.max_context_tokens
                - token_counter.reserved_output_tokens
            )

            if input_tokens > available_tokens:
                raise ValueError(
                    "Tool request exceeds the context budget: "
                    f"{input_tokens} tokens used, "
                    f"{available_tokens} available."
                )

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
            report_source_attribution(
                assistant.content,
                retrieved_passages,
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

            # Collect evidence actually returned by document search.
            if (
                call.function.name == "search_project_documents"
                and isinstance(result, dict)
            ):
                passages = result.get("passages", [])

                if isinstance(passages, list):
                    retrieved_passages.extend(
                        passage
                        for passage in passages
                        if isinstance(passage, dict)
                    )


            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, allow_nan=False),
                }
            )

    # The final request does not include tool definitions.
    if token_counter is not None:
        final_input_tokens = token_counter.count(
            updated_messages
        )

        available_tokens = (
            token_counter.max_context_tokens
            - token_counter.reserved_output_tokens
        )

        if final_input_tokens > available_tokens:
            raise ValueError(
                "Final tool response exceeds the context budget: "
                f"{final_input_tokens} tokens used, "
                f"{available_tokens} available."
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

    report_source_attribution(
        final_message.content,
        retrieved_passages,
    )

    return final_message.content
