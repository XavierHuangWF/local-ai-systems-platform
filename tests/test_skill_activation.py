
import unittest
from unittest.mock import patch

from src.agent.harness import AgentHarness, SYSTEM_PROMPT


class TestSkillActivation(unittest.TestCase):

    def setUp(self):
        # Replace the model client and token counter
        # with simulated components.
        model_patcher = patch("src.agent.harness.VLLMClient")
        counter_patcher = patch("src.agent.harness.TokenCounter")

        self.addCleanup(model_patcher.stop)
        self.addCleanup(counter_patcher.stop)

        self.fake_model = model_patcher.start().return_value
        self.fake_counter = counter_patcher.start().return_value

        self.fake_counter.fits.return_value = True
        self.fake_model.chat.return_value = "Simulated response"

        # Keep the real AgentHarness and SkillLoader.
        self.agent = AgentHarness()

    def test_initial_state(self):
        self.assertIsNone(self.agent.active_skill)

        self.assertEqual(
            self.agent.messages,
            [{"role": "system", "content": SYSTEM_PROMPT}],
        )

    def test_skill_included_in_model_request(self):
        self.agent.activate_skill("research_summary")

        response = self.agent.run("Summarize this study.")

        sent_messages = self.fake_model.chat.call_args.args[0]

        self.assertEqual(
            self.agent.active_skill.name,
            "research_summary",
        )

        self.assertIn("## Instructions", sent_messages[0]["content"])
        self.assertIn("research_summary", sent_messages[0]["content"])
        self.assertEqual(response, "Simulated response")

    def test_reset_preserves_skill(self):
        self.agent.activate_skill("research_summary")
        self.agent.run("Summarize this study.")

        self.agent.reset()

        self.assertEqual(len(self.agent.messages), 1)
        self.assertEqual(
            self.agent.active_skill.name,
            "research_summary",
        )
        self.assertIn(
            "research_summary",
            self.agent.messages[0]["content"],
        )

    def test_deactivation_restores_general_context(self):
        self.agent.activate_skill("research_summary")
        self.agent.run("Summarize this study.")

        self.agent.deactivate_skill()

        self.assertIsNone(self.agent.active_skill)
        self.assertEqual(
            self.agent.messages,
            [{"role": "system", "content": SYSTEM_PROMPT}],
        )

    def test_failed_activation_preserves_state(self):
        self.agent.activate_skill("research_summary")
        self.agent.run("Summarize this study.")

        previous_messages = [
            message.copy() for message in self.agent.messages
        ]

        with self.assertRaises(FileNotFoundError):
            self.agent.activate_skill("nonexistent_skill")

        self.assertEqual(
            self.agent.active_skill.name,
            "research_summary",
        )
        self.assertEqual(
            self.agent.messages,
            previous_messages,
        )


if __name__ == "__main__":
    unittest.main()
