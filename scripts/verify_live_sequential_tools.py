
import logging

from src.agent.harness import AgentHarness


QUESTION = (
    "Search our project's documentation for the four input "
    "measurements used in the live multi-turn Agent Harness "
    "verification experiment. After retrieving the documented "
    "values, use the statistics tool to calculate their count, "
    "mean, minimum, and maximum. Do not guess the input values "
    "before retrieving them. Cite the document containing "
    "the measurements."
)


class ToolLogCollector(logging.Handler):
    """Capture actual tool execution and round-selection events."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(record.getMessage())


def main() -> None:
    print("=== Live Sequential Tool Execution Verification ===")

    logger = logging.getLogger("local_ai.tool_execution")
    original_level = logger.level

    collector = ToolLogCollector()
    logger.addHandler(collector)
    logger.setLevel(logging.INFO)

    try:
        agent = AgentHarness()
        agent.set_tool_mode(True)

        print("\nQuestion:")
        print(QUESTION)

        answer = agent.run(QUESTION)

    finally:
        logger.removeHandler(collector)
        logger.setLevel(original_level)

    print("\n=== Tool Execution Events ===")
    for event in collector.events:
        print(event)

    print("\n=== Generated Answer ===")
    print(answer)

    expected_sequence = [
        "Round 1 requested: search_project_documents",
        "Executed tool: search_project_documents",
        "Round 2 requested: summarize_measurements",
        "Executed tool: summarize_measurements",
    ]

    # Verify the tools were selected and executed in order.
    cursor = 0

    for expected in expected_sequence:
        while (
            cursor < len(collector.events)
            and collector.events[cursor] != expected
        ):
            cursor += 1

        if cursor == len(collector.events):
            raise AssertionError(
                f"Missing or out-of-order tool event: {expected}"
            )

        cursor += 1

    print("\nPASS: Sequential tool execution confirmed.")

    # Verify the expected numerical result.
    if "47.75" not in answer:
        raise AssertionError(
            "Expected mean 47.75 was not found in the answer."
        )

    print("PASS: Expected mean 47.75 confirmed.")

    # Verify that the answer identifies the document source.
    if "tool-execution.md" not in answer:
        raise AssertionError(
            "Expected document source was not cited."
        )

    print("PASS: Expected document source identified.")

    print(
        "\nPASS: Live sequential tool execution "
        "verification completed."
    )


if __name__ == "__main__":
    main()
