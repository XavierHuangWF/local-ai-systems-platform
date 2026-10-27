
from typing import Any

from src.tools.definitions import SUMMARIZE_MEASUREMENTS_DEFINITION
from src.tools.validator import ToolArgumentValidator


def expect_rejected(
    validator: ToolArgumentValidator,
    label: str,
    arguments: Any,
    expected_error: type[Exception],
) -> None:
    """Verify that invalid arguments are rejected."""

    try:
        validator.validate(arguments)
    except expected_error as exc:
        print(f"PASS: {label}")

        error_message = str(exc).replace("\n", " ")

        if len(error_message) > 180:
            error_message = error_message[:180] + " ... [truncated]"

        print(f"  Expected error: {error_message}")
    else:
        raise AssertionError(
            f"FAIL: Invalid arguments accepted: {label}"
        )


def main() -> None:
    print("=== Tool Argument Validator Verification ===")

    validator = ToolArgumentValidator(
        SUMMARIZE_MEASUREMENTS_DEFINITION
    )

    # Test 1: Valid arguments
    print("\n=== Test 1: Valid Arguments ===")

    valid_arguments = {
        "values": [10, 20, 30, 40]
    }

    result = validator.validate(valid_arguments)

    assert result == valid_arguments
    print("PASS: Valid arguments accepted.")

    # Test 2: Missing required argument
    print("\n=== Test 2: Missing Required Field ===")

    expect_rejected(
        validator,
        "Missing values",
        {},
        ValueError,
    )

    # Test 3: Incorrect argument type
    print("\n=== Test 3: Incorrect Type ===")

    expect_rejected(
        validator,
        "String instead of array",
        {"values": "10,20,30"},
        ValueError,
    )

    # Test 4: Invalid array element
    print("\n=== Test 4: Invalid Element ===")

    expect_rejected(
        validator,
        "Nonnumeric array element",
        {"values": [10, "invalid", 30]},
        ValueError,
    )

    # Test 5: Unexpected argument
    print("\n=== Test 5: Unexpected Field ===")

    expect_rejected(
        validator,
        "Unknown argument field",
        {"values": [10, 20], "extra": True},
        ValueError,
    )

    # Test 6: Empty array
    print("\n=== Test 6: Empty Array ===")

    expect_rejected(
        validator,
        "Empty measurement array",
        {"values": []},
        ValueError,
    )

    # Test 7: Excessive array length
    print("\n=== Test 7: Array Size Limit ===")

    expect_rejected(
        validator,
        "More than 10000 measurements",
        {"values": [1] * 10001},
        ValueError,
    )

    # Test 8: Invalid argument container
    print("\n=== Test 8: Invalid Container ===")

    expect_rejected(
        validator,
        "List instead of dictionary",
        [10, 20, 30],
        TypeError,
    )

    # Test 9: Boolean is not a JSON Schema number
    print("\n=== Test 9: Boolean Element ===")

    expect_rejected(
        validator,
        "Boolean measurement",
        {"values": [True, 20]},
        ValueError,
    )

    print("\n=== Final Result ===")
    print("PASS: All 9 tool argument validation checks passed.")


if __name__ == "__main__":
    main()
