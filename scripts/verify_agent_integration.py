
from src.agent.harness import AgentHarness, SYSTEM_PROMPT


def main() -> None:
    print("=== Agent Integration Verification ===")

    agent = AgentHarness()

    # Test 1: Real model inference
    print("\n=== Test 1: Model Response ===")

    response1 = agent.run(
        "My project codename is Cypress. "
        "Acknowledge this briefly. /no_think"
    )

    print("Assistant:", response1)

    assert isinstance(response1, str) and response1.strip()
    print("PASS: Model returned a nonempty response.")

    # Test 2: Conversation history
    print("\n=== Test 2: Conversation History ===")

    response2 = agent.run(
        "What is my project codename? "
        "Answer with one word. /no_think"
    )

    print("Assistant:", response2)

    expected_roles = [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    actual_roles = [
        message["role"]
        for message in agent.messages
    ]

    assert actual_roles == expected_roles
    print("PASS: Conversation history preserved.")

    # Inspect the answer manually for "Cypress".
    # Model accuracy is separate from harness functionality.

    # Test 3: Reset
    print("\n=== Test 3: Reset ===")

    agent.reset()

    assert agent.messages == [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    print("PASS: Conversation reset.")

    print("\n=== Final Result ===")
    print("PASS: Agent integration checks completed.")


if __name__ == "__main__":
    main()
