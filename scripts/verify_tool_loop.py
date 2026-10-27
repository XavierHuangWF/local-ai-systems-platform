
import json
from types import SimpleNamespace
from unittest.mock import Mock

from src.agent.tool_loop import run_tool_turn
from src.tools.definitions import (
    SUMMARIZE_MEASUREMENTS_DEFINITION,
)
from src.tools.executor import ToolExecutor
from src.tools.statistics_tool import summarize_measurements


def completion(message):
    """Build a simulated chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )


def tool_call(arguments):
    """Build a simulated model-generated tool request."""
    return SimpleNamespace(
        id="call_test_001",
        type="function",
        function=SimpleNamespace(
            name="summarize_measurements",
            arguments=arguments,
        ),
    )


def main() -> None:
    print("=== Tool Loop Integration Verification ===")

    executor = ToolExecutor()
    execution_count = 0

    def monitored_handler(values):
        nonlocal execution_count
        execution_count += 1
        return summarize_measurements(values)

    executor.register(
        SUMMARIZE_MEASUREMENTS_DEFINITION,
        monitored_handler,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Calculate the mean of 10, 20, 30, and 40."
            ),
        }
    ]

    client = Mock()

    def simulated_model(**kwargs):
        # First request: the model chooses the statistics tool.
        if "tools" in kwargs:
            return completion(
                SimpleNamespace(
                    content=None,
                    tool_calls=[
                        tool_call(
                            '{"values": [10, 20, 30, 40]}'
                        )
                    ],
                )
            )

        # Second request: the model receives the tool result.
        tool_messages = [
            message
            for message in kwargs["messages"]
            if message["role"] == "tool"
        ]

        assert len(tool_messages) == 1

        tool_message = tool_messages[0]

        assert tool_message["tool_call_id"] == (
            "call_test_001"
        )

        result = json.loads(tool_message["content"])

        assert result == {
            "count": 4,
            "mean": 25.0,
            "minimum": 10.0,
            "maximum": 40.0,
        }

        # Construct the simulated final answer from the
        # actual result produced by the Python function.
        return completion(
            SimpleNamespace(
                content=(
                    f"The mean is {result['mean']} "
                    f"across {result['count']} measurements."
                ),
                tool_calls=None,
            )
        )

    client.chat.completions.create.side_effect = (
        simulated_model
    )

    # Test 1: Complete model-to-tool-to-model workflow.
    answer = run_tool_turn(
        client=client,
        model="qwen3-8b",
        messages=messages,
        executor=executor,
    )

    assert answer == (
        "The mean is 25.0 across 4 measurements."
    )
    print("PASS: Complete tool workflow returned correct result.")

    # Test 2: Python handler executed exactly once.
    assert execution_count == 1
    print("PASS: Registered Python handler executed once.")

    # Test 3: Two model requests were performed.
    assert client.chat.completions.create.call_count == 2

    first_request = (
        client.chat.completions.create.call_args_list[0].kwargs
    )
    second_request = (
        client.chat.completions.create.call_args_list[1].kwargs
    )

    assert len(first_request["tools"]) == 1
    assert "tools" not in second_request

    print("PASS: Tool request and final response separated.")

    # Test 4: Original conversation remains unchanged.
    assert messages == [
        {
            "role": "user",
            "content": (
                "Calculate the mean of 10, 20, 30, and 40."
            ),
        }
    ]
    print("PASS: Original messages were not modified.")

    # Test 5: Invalid arguments never reach the handler.
    invalid_client = Mock()

    invalid_client.chat.completions.create.return_value = (
        completion(
            SimpleNamespace(
                content=None,
                tool_calls=[
                    tool_call('{"values": "10,20,30"}')
                ],
            )
        )
    )

    try:
        run_tool_turn(
            client=invalid_client,
            model="qwen3-8b",
            messages=messages,
            executor=executor,
        )
    except ValueError:
        assert execution_count == 1
        assert (
            invalid_client.chat.completions.create.call_count
            == 1
        )
        print("PASS: Invalid arguments rejected before execution.")
    else:
        raise AssertionError(
            "Invalid arguments were accepted."
        )

    # Test 6: Malformed JSON is rejected.
    malformed_client = Mock()

    malformed_client.chat.completions.create.return_value = (
        completion(
            SimpleNamespace(
                content=None,
                tool_calls=[
                    tool_call('{"values": [10, 20,}')
                ],
            )
        )
    )

    try:
        run_tool_turn(
            client=malformed_client,
            model="qwen3-8b",
            messages=messages,
            executor=executor,
        )
    except ValueError:
        assert execution_count == 1
        print("PASS: Malformed JSON tool arguments rejected.")
    else:
        raise AssertionError(
            "Malformed JSON was accepted."
        )

    # Test 7: Ordinary responses do not execute tools.
    text_client = Mock()

    text_client.chat.completions.create.return_value = (
        completion(
            SimpleNamespace(
                content="No calculation is needed.",
                tool_calls=None,
            )
        )
    )

    text_answer = run_tool_turn(
        client=text_client,
        model="qwen3-8b",
        messages=messages,
        executor=executor,
    )

    assert text_answer == "No calculation is needed."
    assert execution_count == 1

    print("PASS: Ordinary text response preserved.")

    print("\n=== Final Result ===")
    print("PASS: All 7 tool loop integration checks passed.")


if __name__ == "__main__":
    main()
