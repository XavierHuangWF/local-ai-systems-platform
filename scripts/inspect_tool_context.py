
import json

from src.agent.harness import AgentHarness


def main() -> None:
    print("=== Multi-Round Tool Context Inspection ===")

    agent = AgentHarness()
    counter = agent.token_counter

    tools = agent.tool_executor.get_tool_definitions()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical assistant. "
                "Use registered tools when necessary."
            ),
        },
        {
            "role": "user",
            "content": (
                "Find the measurements from the live multi-turn "
                "experiment and calculate their statistics."
            ),
        },
    ]

    def inspect(stage: str) -> None:
        print(f"\n=== {stage} ===")

        # Existing counting method.
        try:
            message_tokens = counter.count(messages)
            print(f"Message tokens: {message_tokens}")
        except Exception as exc:
            print(
                "Message counting failed:",
                type(exc).__name__,
                str(exc),
            )

        # Count the actual tool definitions as well.
        try:
            encoded = counter.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )

            total_tokens = len(encoded["input_ids"])

            print(f"Tokens including tool definitions: {total_tokens}")
            print(
                "Available with 512-token output reservation:",
                counter.max_context_tokens - 512,
            )
            print(
                "Within budget:",
                total_tokens <= counter.max_context_tokens - 512,
            )

        except Exception as exc:
            print(
                "Tool-aware counting failed:",
                type(exc).__name__,
                str(exc),
            )

    inspect("Initial user request")

    # Simulate Qwen3 requesting document search.
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "search_project_documents",
                        "arguments": json.dumps({
                            "query": "live multi-turn experiment measurements",
                            "top_k": 4,
                        }),
                    },
                }
            ],
        }
    )

    # Simulate a document-search result.
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "call_search",
            "content": json.dumps({
                "passages": [
                    {
                        "source": "tool-execution.md",
                        "content": (
                            "The live experiment used measurements "
                            "[13, 25, 61, 92]. "
                        ) * 30,
                    }
                ]
            }),
        }
    )

    inspect("After document search")

    # Simulate the second round.
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_stats",
                    "type": "function",
                    "function": {
                        "name": "summarize_measurements",
                        "arguments": json.dumps({
                            "values": [13, 25, 61, 92],
                        }),
                    },
                }
            ],
        }
    )

    messages.append(
        {
            "role": "tool",
            "tool_call_id": "call_stats",
            "content": json.dumps({
                "count": 4,
                "mean": 47.75,
                "minimum": 13.0,
                "maximum": 92.0,
            }),
        }
    )

    inspect("After second tool round")

    print("\n=== Inspection Completed ===")


if __name__ == "__main__":
    main()
