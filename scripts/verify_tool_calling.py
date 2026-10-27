
import json
from types import SimpleNamespace
from unittest.mock import Mock

from src.client.tool_calling import request_with_tools
from src.tools.definitions import (
    SUMMARIZE_MEASUREMENTS_DEFINITION,
)


def main() -> None:
    print("=== Model Tool Calling Verification ===")

    client = Mock()
    model = "qwen3-8b"

    messages = [
        {
            "role": "user",
            "content": (
                "Calculate the mean of 10, 20, 30, and 40."
            ),
        }
    ]

    tools = [
        SUMMARIZE_MEASUREMENTS_DEFINITION.to_openai_format()
    ]

    # Simulate a structured model tool-call response.
    tool_call = SimpleNamespace(
        id="call_test_001",
        type="function",
        function=SimpleNamespace(
            name="summarize_measurements",
            arguments='{"values": [10, 20, 30, 40]}',
        ),
    )

    tool_message = SimpleNamespace(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )

    client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=tool_message)
            ]
        )
    )

    # Test 1: Request forwarded correctly.
    result = request_with_tools(
        client=client,
        model=model,
        messages=messages,
        tools=tools,
    )

    client.chat.completions.create.assert_called_once_with(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print("PASS: Model request includes tool definitions.")

    # Test 2: Structured tool call preserved.
    assert result is tool_message
    assert len(result.tool_calls) == 1

    returned_call = result.tool_calls[0]

    assert returned_call.id == "call_test_001"
    assert returned_call.function.name == (
        "summarize_measurements"
    )

    arguments = json.loads(
        returned_call.function.arguments
    )

    assert arguments == {
        "values": [10, 20, 30, 40]
    }

    print("PASS: Structured tool call preserved.")
    print(f"Tool name: {returned_call.function.name}")
    print(f"Tool arguments: {arguments}")

    # Test 3: Empty tool definitions rejected.
    client.reset_mock()

    try:
        request_with_tools(
            client=client,
            model=model,
            messages=messages,
            tools=[],
        )
    except ValueError:
        print("PASS: Empty tool definitions rejected.")
    else:
        raise AssertionError(
            "Empty tool definitions accepted."
        )

    client.chat.completions.create.assert_not_called()

    # Test 4: Ordinary text response preserved.
    text_message = SimpleNamespace(
        role="assistant",
        content="The arithmetic mean is 25.",
        tool_calls=None,
    )

    client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=text_message)
            ]
        )
    )

    result = request_with_tools(
        client=client,
        model=model,
        messages=messages,
        tools=tools,
    )

    assert result.content == (
        "The arithmetic mean is 25."
    )
    assert result.tool_calls is None

    print("PASS: Ordinary text response preserved.")

    # Test 5: Empty completion choices rejected.
    client.chat.completions.create.return_value = (
        SimpleNamespace(choices=[])
    )

    try:
        request_with_tools(
            client=client,
            model=model,
            messages=messages,
            tools=tools,
        )
    except RuntimeError:
        print("PASS: Empty completion choices rejected.")
    else:
        raise AssertionError(
            "Empty completion choices accepted."
        )

    print("\n=== Final Result ===")
    print("PASS: All 5 model tool-calling checks passed.")


if __name__ == "__main__":
    main()
