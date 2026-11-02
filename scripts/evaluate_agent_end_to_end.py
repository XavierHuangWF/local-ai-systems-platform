
import logging
import time
from dataclasses import dataclass

from src.agent.harness import AgentHarness


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    question: str
    expected_events: tuple[str, ...]
    expected_answer: tuple[str, ...]


CASES = [
    EvaluationCase(
        name="Direct Python Statistics",
        question=(
            "Use the direct summarize_measurements tool "
            "to calculate the statistics for [4, 8, 12, 16]. "
            "Report count, mean, minimum, and maximum. "
            "Do not use MCP."
        ),
        expected_events=(
            "Executed tool: summarize_measurements",
        ),
        expected_answer=("10.0",),
    ),
    EvaluationCase(
        name="MCP Statistics",
        question=(
            "Use the mcp_summarize_measurements tool "
            "through the MCP server for [15, 25, 35, 45, 60]. "
            "Report count, mean, minimum, and maximum."
        ),
        expected_events=(
            "Executed tool: mcp_summarize_measurements",
        ),
        expected_answer=("36.0",),
    ),
    EvaluationCase(
        name="Multi-Round Retrieval and Calculation",
        question=(
            "Search our project's documentation for the four "
            "input measurements used in the live multi-turn "
            "Agent Harness verification experiment. "
            "After retrieving the values, use the direct "
            "summarize_measurements tool to calculate their "
            "count, mean, minimum, and maximum. "
            "Do not guess the values. Cite the document."
        ),
        expected_events=(
            "Round 1 requested: search_project_documents",
            "Executed tool: search_project_documents",
            "Round 2 requested: summarize_measurements",
            "Executed tool: summarize_measurements",
        ),
        expected_answer=("47.75", "tool-execution.md"),
    ),
]


class ToolEventCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        self.events.append(record.getMessage())


def check_events(actual, expected):
    """Verify that expected tool events occurred in order."""
    position = 0

    for event in expected:
        while position < len(actual) and actual[position] != event:
            position += 1

        if position == len(actual):
            return False, event

        position += 1

    return True, None


def main():
    print("=== End-to-End Agent Evaluation ===")

    logger = logging.getLogger("local_ai.tool_execution")
    original_level = logger.level

    results = []

    try:
        logger.setLevel(logging.INFO)

        for case in CASES:
            print(f"\n=== {case.name} ===")

            collector = ToolEventCollector()
            logger.addHandler(collector)

            started = time.perf_counter()

            try:
                # Fresh agent avoids conversation carryover.
                agent = AgentHarness()
                agent.set_tool_mode(True)

                answer = agent.run(case.question)

                elapsed = time.perf_counter() - started

                sequence_ok, missing_event = check_events(
                    collector.events,
                    case.expected_events,
                )

                answer_ok = all(
                    expected in answer
                    for expected in case.expected_answer
                )

                passed = sequence_ok and answer_ok

                print("\nTool events:")
                for event in collector.events:
                    print(" ", event)

                print("\nAnswer:")
                print(answer)

                print(f"\nElapsed: {elapsed:.2f}s")
                print(f"Tool sequence: {sequence_ok}")
                print(f"Answer checks: {answer_ok}")

                if missing_event:
                    print(f"Missing event: {missing_event}")

                print("PASS" if passed else "FAIL")

            except Exception as exc:
                elapsed = time.perf_counter() - started
                passed = False
                print(
                    f"ERROR: {type(exc).__name__}: {exc}"
                )

            finally:
                logger.removeHandler(collector)

            results.append({
                "name": case.name,
                "passed": passed,
                "elapsed": elapsed,
            })

    finally:
        logger.setLevel(original_level)

    print("\n=== Evaluation Summary ===")

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"

        print(
            f"{status}: {result['name']} "
            f"({result['elapsed']:.2f}s)"
        )

    passed_count = sum(
        result["passed"] for result in results
    )

    total = len(results)

    print(f"\nPassed: {passed_count}/{total}")

    if passed_count != total:
        raise SystemExit(1)

    print("\n=== Evaluation Completed ===")


if __name__ == "__main__":
    main()
