
import json
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.token_counter import TokenCounter
from src.agent.tool_loop import run_tool_turn
from src.tools.executor import ToolExecutor
from src.tools.definitions import DOCUMENT_SEARCH_DEFINITION


def oversized_search(query: str, top_k: int = 4) -> dict:
    """Return an intentionally oversized simulated search result."""
    return {
        "query": query,
        "passages": [
            {
                "source": "simulated-document.md",
                "chunk_id": "simulated-document.md::0001",
                "content": "Repeated document evidence. " * 6000,
            }
        ],
    }


def main() -> None:
    print("=== Tool Context Overflow Verification ===")

    counter = TokenCounter()

    executor = ToolExecutor()
    executor.register(
        definition=DOCUMENT_SEARCH_DEFINITION,
        handler=oversized_search,
    )

    messages = [
        {
            "role": "system",
            "content": "Use tools to answer questions.",
        },
        {
            "role": "user",
            "content": "Search the documentation for an explanation.",
        },
    ]

    # Simulate one model decision to search documents.
    tool_call = SimpleNamespace(
        id="call_search",
        function=SimpleNamespace(
            name="search_project_documents",
            arguments=json.dumps({
                "query": "technical explanation",
                "top_k": 1,
            }),
        ),
    )

    simulated_response = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
    )

    # No real model request will be made.
    with patch(
        "src.agent.tool_loop.request_with_tools",
        return_value=simulated_response,
    ) as mock_model:

        try:
            run_tool_turn(
                client=None,
                model="qwen3-8b",
                messages=messages,
                executor=executor,
                max_rounds=2,
                token_counter=counter,
            )

        except ValueError as exc:
            print("\nCaught expected error:")
            print(exc)

            if "context budget" not in str(exc):
                raise

        else:
            raise AssertionError(
                "Oversized tool result was not rejected."
            )

        # Round 1 was simulated, but Round 2 must not
        # reach the model after the context overflow.
        if mock_model.call_count != 1:
            raise AssertionError(
                "An additional model request was attempted."
            )

    print("\nPASS: Oversized context was rejected.")
    print("PASS: No second model request was made.")
    print("PASS: Context overflow verification completed.")


if __name__ == "__main__":
    main()
