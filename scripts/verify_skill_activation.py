
from unittest.mock import patch

from src.agent.harness import AgentHarness, SYSTEM_PROMPT


def main() -> None:
    print("=== Skill Activation Verification ===")

    # Simulate the model and token counter.
    # Use the real AgentHarness and SkillLoader.
    with (
        patch("src.agent.harness.VLLMClient") as model_class,
        patch("src.agent.harness.TokenCounter") as counter_class,
    ):
        fake_model = model_class.return_value
        fake_model.chat.return_value = "Simulated response"

        fake_counter = counter_class.return_value
        fake_counter.fits.return_value = True

        agent = AgentHarness()

        # Test 1: Initial state
        print("\n=== Test 1: Initial State ===")

        assert agent.active_skill is None
        assert agent.messages == [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        print("PASS: No skill is active initially.")

        # Test 2: Activate a real skill
        print("\n=== Test 2: Skill Activation ===")

        agent.activate_skill("research_summary")

        assert agent.active_skill is not None
        assert agent.active_skill.name == "research_summary"
        assert "## Instructions" in agent.messages[0]["content"]

        print("Active skill:", agent.active_skill.name)
        print("PASS: Skill instructions loaded.")

        # Test 3: Check the messages sent to the model
        print("\n=== Test 3: Model Request Context ===")

        response = agent.run("Summarize this research.")

        sent_messages = fake_model.chat.call_args.args[0]

        assert "research_summary" in sent_messages[0]["content"]
        assert "## Instructions" in sent_messages[0]["content"]
        assert sent_messages[-1]["content"] == (
            "Summarize this research."
        )
        assert response == "Simulated response"

        print("PASS: Active skill included in model request.")

        # Test 4: Reset preserves the active skill
        print("\n=== Test 4: Conversation Reset ===")

        agent.reset()

        assert agent.active_skill.name == "research_summary"
        assert len(agent.messages) == 1
        assert "research_summary" in agent.messages[0]["content"]

        print("PASS: Reset preserves the active skill.")

        # Test 5: Deactivate the skill
        print("\n=== Test 5: Skill Deactivation ===")

        agent.deactivate_skill()

        assert agent.active_skill is None
        assert agent.messages == [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        print("PASS: Skill deactivated and context restored.")

        # Test 6: Invalid activation preserves existing state
        print("\n=== Test 6: Failed Activation ===")

        agent.activate_skill("research_summary")
        previous_messages = [
            message.copy() for message in agent.messages
        ]

        try:
            agent.activate_skill("nonexistent_skill")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing skill was accepted.")

        assert agent.active_skill.name == "research_summary"
        assert agent.messages == previous_messages

        print("PASS: Failed activation preserved existing state.")

    print("\n=== Final Result ===")
    print("PASS: All six skill activation checks passed.")


if __name__ == "__main__":
    main()
