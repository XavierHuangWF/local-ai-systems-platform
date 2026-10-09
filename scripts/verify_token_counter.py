
from src.agent.token_counter import TokenCounter


def main() -> None:
    counter = TokenCounter()

    input_budget = (
        counter.max_context_tokens
        - counter.reserved_output_tokens
    )

    print("=== Token Counter Verification ===")
    print("Model: Qwen/Qwen3-8B-AWQ")
    print(f"Maximum context: {counter.max_context_tokens}")
    print(f"Reserved output: {counter.reserved_output_tokens}")
    print(f"Input budget: {input_budget}")
    print()

    # Test 1: Normal conversation
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is a transformer?",
        },
    ]

    token_count = counter.count(messages)
    within_budget = counter.fits(messages)

    print("=== Test 1: Normal Conversation ===")
    print(f"Actual input tokens: {token_count}")
    print(f"Within budget: {within_budget}")

    assert token_count > 2, (
        "Token count is suspiciously small."
    )
    assert within_budget, (
        "Short conversation should fit."
    )

    print("PASS: Normal conversation")
    print()

    # Test 2: Oversized conversation
    large_messages = [
        {
            "role": "user",
            "content": "example " * 12000,
        }
    ]

    large_count = counter.count(large_messages)
    large_fits = counter.fits(large_messages)

    print("=== Test 2: Oversized Conversation ===")
    print(f"Oversized input tokens: {large_count}")
    print(f"Within budget: {large_fits}")

    assert large_count > input_budget, (
        "Oversized conversation should exceed the input budget."
    )
    assert not large_fits, (
        "Oversized conversation should be rejected."
    )

    print("PASS: Oversized conversation")
    print()

    print("=== Verification Result ===")
    print("PASS: All token counter checks passed.")


if __name__ == "__main__":
    main()
