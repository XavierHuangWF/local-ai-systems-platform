
from unittest.mock import patch

from src.agent.harness import AgentHarness
from src.tools.statistics_tool import summarize_measurements


def main() -> None:
    print("=== Live Agent Harness Tool Integration ===")

    # Observe actual Python tool execution.
    with patch(
        "src.agent.harness.summarize_measurements",
        wraps=summarize_measurements,
    ) as monitored_tool:

        agent = AgentHarness()

        # Allow additional time for local model inference.
        agent.model.client = agent.model.client.with_options(
            timeout=120.0,
            max_retries=0,
        )

        assert agent.tools_enabled is False
        print("PASS: Tool mode is disabled by default.")

        agent.set_tool_mode(True)
        assert agent.tools_enabled is True

        print("\n=== Turn 1: Tool Execution ===")

        first_answer = agent.run(
            "Use the summarize_measurements tool to "
            "calculate the count, mean, minimum, and "
            "maximum of [13, 25, 61, 92]. "
            "Report all four statistics."
        )

        print(first_answer)

        assert monitored_tool.call_count == 1, (
            "Expected exactly one Python tool execution."
        )

        assert "47.75" in first_answer, (
            "First answer did not contain the correct mean."
        )

        assert len(agent.messages) == 3

        print("PASS: Registered Python tool executed once.")
        print("PASS: First answer contains the correct mean.")
        print("PASS: First exchange saved in conversation history.")

        print("\n=== Turn 2: Conversation Memory ===")

        second_answer = agent.run(
            "What mean did you report in your previous "
            "answer? Reply with only the number."
        )

        print(second_answer)

        assert "47.75" in second_answer, (
            "The model did not recall the previous mean."
        )

        assert len(agent.messages) == 5

        print("PASS: Qwen3 recalled the previous answer.")
        print("PASS: Both exchanges preserved in conversation history.")

        print("\n=== Final Result ===")
        print("PASS: Live multi-turn AgentHarness verification completed.")


if __name__ == "__main__":
    main()
