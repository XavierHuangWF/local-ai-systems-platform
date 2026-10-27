
from src.tools.definitions import (
    SUMMARIZE_MEASUREMENTS_DEFINITION,
)
from src.tools.executor import ToolExecutor
from src.tools.statistics_tool import summarize_measurements


def main() -> None:
    print("=== Tool Executor Verification ===")

    executor = ToolExecutor()
    execution_count = 0

    def monitored_handler(values):
        nonlocal execution_count
        execution_count += 1
        return summarize_measurements(values)

    # Test 1: Register an approved tool.
    executor.register(
        SUMMARIZE_MEASUREMENTS_DEFINITION,
        monitored_handler,
    )

    definitions = executor.get_tool_definitions()
    assert len(definitions) == 1
    assert definitions[0]["function"]["name"] == (
        "summarize_measurements"
    )
    print("PASS: Tool registered with its schema.")

    # Test 2: Execute valid arguments.
    result = executor.execute(
        "summarize_measurements",
        {"values": [10, 20, 30, 40]},
    )

    assert result == {
        "count": 4,
        "mean": 25.0,
        "minimum": 10.0,
        "maximum": 40.0,
    }
    assert execution_count == 1
    print("PASS: Valid arguments executed correctly.")

    # Test 3: Invalid arguments must not reach the handler.
    invalid_cases = {
        "Missing required field": {},
        "Incorrect array type": {"values": "10,20,30"},
        "Nonnumeric measurement": {
            "values": [10, "invalid", 30]
        },
        "Unexpected field": {
            "values": [10, 20],
            "extra": True,
        },
        "Empty array": {"values": []},
        "Boolean measurement": {"values": [True, 20]},
        "Oversized array": {"values": [1] * 10001},
    }

    for label, arguments in invalid_cases.items():
        before = execution_count

        try:
            executor.execute(
                "summarize_measurements",
                arguments,
            )
        except ValueError:
            assert execution_count == before
            print(f"PASS: {label} rejected before execution.")
        else:
            raise AssertionError(
                f"Invalid arguments accepted: {label}"
            )

    # Test 4: Unknown tools must be rejected.
    try:
        executor.execute("unknown_tool", {})
    except ValueError:
        print("PASS: Unknown tool rejected.")
    else:
        raise AssertionError("Unknown tool accepted.")

    # Test 5: Duplicate registrations must be rejected.
    try:
        executor.register(
            SUMMARIZE_MEASUREMENTS_DEFINITION,
            monitored_handler,
        )
    except ValueError:
        print("PASS: Duplicate registration rejected.")
    else:
        raise AssertionError("Duplicate tool accepted.")

    assert execution_count == 1
    print("PASS: Only the valid request executed.")

    print("\n=== Final Result ===")
    print("PASS: Tool Executor verification completed.")


if __name__ == "__main__":
    main()
