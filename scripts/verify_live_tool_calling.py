
from openai import OpenAI

from src.agent.tool_loop import run_tool_turn
from src.tools.definitions import (
    SUMMARIZE_MEASUREMENTS_DEFINITION,
)
from src.tools.executor import ToolExecutor
from src.tools.statistics_tool import summarize_measurements


def main() -> None:
    print("=== Live Qwen3 Tool-Calling Verification ===")

    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="local-dev-key",
        timeout=120.0,
        max_retries=0,
    )

    executor = ToolExecutor()
    execution_count = 0

    def monitored_handler(values):
        nonlocal execution_count

        execution_count += 1

        print(f"Python tool invoked with: {values}")

        result = summarize_measurements(values)

        print(f"Actual Python result: {result}")

        return result

    executor.register(
        SUMMARIZE_MEASUREMENTS_DEFINITION,
        monitored_handler,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise numerical analysis assistant. "
                "When asked to summarize measurements, you must "
                "call the summarize_measurements tool. "
                "Use its returned result for your final answer. "
                "Do not calculate the statistics manually."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use summarize_measurements to calculate "
                "the count, mean, minimum, and maximum "
                "of [13, 25, 61, 92]. "
                "Report the results."
            ),
        },
    ]

    print("\nSending request to Qwen3...")

    answer = run_tool_turn(
        client=client,
        model="qwen3-8b",
        messages=messages,
        executor=executor,
    )

    print("\n=== Qwen3 Final Answer ===")
    print(answer)

    print("\n=== Verification ===")

    if execution_count != 1:
        raise AssertionError(
            "Expected exactly one actual Python tool execution. "
            f"Observed: {execution_count}"
        )

    print("PASS: Qwen3 requested the Python tool.")

    # Expected mean:
    # (13 + 25 + 61 + 92) / 4 = 47.75
    if "47.75" not in answer:
        raise AssertionError(
            "Qwen3's final answer did not contain "
            "the expected mean of 47.75."
        )

    print("PASS: Final answer contains the expected mean.")

    print("\n=== Final Result ===")
    print("PASS: Live Qwen3 tool-calling verification completed.")


if __name__ == "__main__":
    main()
