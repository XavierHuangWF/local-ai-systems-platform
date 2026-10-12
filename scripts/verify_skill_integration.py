
from src.agent.harness import AgentHarness


def main() -> None:
    print("=== Real-Model Skill Integration Verification ===")

    agent = AgentHarness()

    # Activate the research summary skill.
    agent.activate_skill("research_summary")

    assert agent.active_skill is not None
    assert agent.active_skill.name == "research_summary"

    print("PASS: Research summary skill activated.")

    # Use fictional experimental data.
    research_text = """
    Summarize the following fictional research experiment.

    Objective:
    Improve classification accuracy using synthetic training data.

    Method:
    Researchers evaluated two classification models on
    200 synthetic evaluation samples.

    Results:
    Model A achieved a macro F1 score of 0.72.
    Model B achieved a macro F1 score of 0.80.

    Limitations:
    The experiment used synthetic data and a single evaluation split.
    No external validation was performed.

    /no_think
    """

    print("\n=== Model Response ===")

    response = agent.run(research_text)

    print(response)

    assert isinstance(response, str)
    assert response.strip(), "Model returned an empty response."

    print("\nPASS: Real model returned a response.")

    # Check that skill instructions remain in the system message.
    system_message = agent.messages[0]["content"]

    assert "research_summary" in system_message
    assert "## Instructions" in system_message

    print("PASS: Skill instructions preserved in model context.")

    # Check the research summary structure.
    expected_sections = [
        "Research Objective",
        "Methodology",
        "Key Findings",
        "Limitations",
        "Practical Implications",
    ]

    missing_sections = [
        section
        for section in expected_sections
        if section.lower() not in response.lower()
    ]

    if missing_sections:
        print("\nMissing sections:", missing_sections)
        print("FAIL: Model did not follow the complete output format.")
        raise AssertionError("Skill output format incomplete.")

    print("PASS: Required research summary sections present.")

    print("\n=== Final Result ===")
    print("PASS: Real-model skill integration verification completed.")


if __name__ == "__main__":
    main()
