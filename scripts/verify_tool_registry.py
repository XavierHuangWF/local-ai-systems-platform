
import math

from src.tools.registry import ToolRegistry
from src.tools.statistics_tool import summarize_measurements


def main() -> None:
    print("=== Tool Registry Verification ===")

    registry = ToolRegistry()

    # Test 1: Register an approved function.
    print("\n=== Test 1: Tool Registration ===")

    registry.register(
        name="summarize_measurements",
        handler=summarize_measurements,
    )

    assert registry.list_tools() == [
        "summarize_measurements"
    ]

    print("Available tools:", registry.list_tools())
    print("PASS: Tool registered successfully.")

    # Test 2: Execute an approved function.
    print("\n=== Test 2: Tool Execution ===")

    result = registry.execute(
        name="summarize_measurements",
        arguments={
            "values": [10, 20, 30, 40],
        },
    )

    print("Tool result:", result)

    assert result == {
        "count": 4,
        "mean": 25.0,
        "minimum": 10.0,
        "maximum": 40.0,
    }

    print("PASS: Correct statistics returned.")

    # Test 3: Reject an unknown tool.
    print("\n=== Test 3: Unknown Tool ===")

    try:
        registry.execute(
            name="unknown_tool",
            arguments={},
        )
    except ValueError as exc:
        print("Expected error:", exc)
        print("PASS: Unknown tool rejected.")
    else:
        raise AssertionError("Unknown tool was accepted.")

    # Test 4: Reject duplicate registration.
    print("\n=== Test 4: Duplicate Registration ===")

    try:
        registry.register(
            name="summarize_measurements",
            handler=summarize_measurements,
        )
    except ValueError as exc:
        print("Expected error:", exc)
        print("PASS: Duplicate registration rejected.")
    else:
        raise AssertionError("Duplicate registration accepted.")

    # Test 5: Reject invalid measurements.
    print("\n=== Test 5: Invalid Tool Arguments ===")

    invalid_inputs = [
        [],
        [10, "invalid", 30],
        [10, math.nan],
        [True, 20],
    ]

    for values in invalid_inputs:
        try:
            registry.execute(
                name="summarize_measurements",
                arguments={"values": values},
            )
        except (ValueError, TypeError) as exc:
            print(f"Rejected {values}: {exc}")
        else:
            raise AssertionError(
                f"Invalid input was accepted: {values}"
            )

    print("PASS: Invalid measurements rejected.")

    print("\n=== Final Result ===")
    print("PASS: All five tool registry checks passed.")


if __name__ == "__main__":
    main()
