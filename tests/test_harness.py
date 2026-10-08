
import unittest
from unittest.mock import patch

from src.agent.harness import AgentHarness, SYSTEM_PROMPT


class TestAgentHarness(unittest.TestCase):

    def setUp(self):
        patcher = patch("src.agent.harness.VLLMClient")
        self.addCleanup(patcher.stop)

        self.fake_model = patcher.start().return_value
        self.agent = AgentHarness()

    def test_conversation_history(self):
        snapshots = []

        def fake_chat(messages):
            snapshots.append([msg.copy() for msg in messages])
            return "Test response"

        self.fake_model.chat.side_effect = fake_chat

        self.agent.run("What is Python?")
        self.agent.run("What did I just ask?")

        self.assertEqual(
            [msg["role"] for msg in snapshots[1]],
            ["system", "user", "assistant", "user"],
        )

        self.assertEqual(
            snapshots[1][1]["content"],
            "What is Python?",
        )

    def test_reset(self):
        self.fake_model.chat.return_value = "Test response"

        self.agent.run("Hello")
        self.agent.reset()

        self.assertEqual(
            self.agent.messages,
            [{"role": "system", "content": SYSTEM_PROMPT}],
        )

    def test_model_failure(self):
        self.fake_model.chat.side_effect = ConnectionError(
            "Model server unavailable"
        )

        with self.assertRaises(ConnectionError):
            self.agent.run("Hello")

        self.assertEqual(
            self.agent.messages,
            [{"role": "system", "content": SYSTEM_PROMPT}],
        )

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            self.agent.run("   ")


if __name__ == "__main__":
    unittest.main()
