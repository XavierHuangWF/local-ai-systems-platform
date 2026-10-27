
import unittest
from unittest.mock import patch

from src.agent.harness import AgentHarness


class TestHarnessResponseFormatting(unittest.TestCase):

    def setUp(self):
        model_patcher = patch(
            "src.agent.harness.VLLMClient"
        )
        token_patcher = patch(
            "src.agent.harness.TokenCounter"
        )

        self.mock_model_class = model_patcher.start()
        self.mock_token_class = token_patcher.start()

        self.addCleanup(model_patcher.stop)
        self.addCleanup(token_patcher.stop)

        self.mock_model = self.mock_model_class.return_value
        self.mock_model.model = "qwen3-8b"

        self.mock_counter = self.mock_token_class.return_value
        self.mock_counter.fits.return_value = True

        self.agent = AgentHarness()

    def test_normal_response_preserved(self):
        self.mock_model.chat.return_value = "Hello."

        answer = self.agent.run("Hello")

        self.assertEqual(answer, "Hello.")
        self.assertEqual(
            self.agent.messages[-1]["content"],
            "Hello.",
        )

    def test_normal_chat_removes_thinking(self):
        self.mock_model.chat.return_value = (
            "<think>Internal reasoning.</think>\n"
            "The mean is 47.75."
        )

        answer = self.agent.run("What is the mean?")

        self.assertEqual(answer, "The mean is 47.75.")
        self.assertEqual(
            self.agent.messages[-1]["content"],
            "The mean is 47.75.",
        )

    def test_tool_mode_removes_thinking(self):
        self.agent.set_tool_mode(True)

        with patch(
            "src.agent.harness.run_tool_turn",
            return_value=(
                "<think>Checking the tool result.</think>\n"
                "47.75"
            ),
        ) as mock_tool_turn:
            answer = self.agent.run(
                "Summarize [13, 25, 61, 92]."
            )

        mock_tool_turn.assert_called_once()
        self.assertEqual(answer, "47.75")
        self.assertEqual(
            self.agent.messages[-1]["content"],
            "47.75",
        )

    def test_malformed_thinking_preserves_history(self):
        self.mock_model.chat.return_value = (
            "<think>Unfinished reasoning"
        )

        original_messages = [
            dict(message) for message in self.agent.messages
        ]

        with self.assertRaises(ValueError):
            self.agent.run("Calculate the mean")

        self.assertEqual(
            self.agent.messages,
            original_messages,
        )

    def test_thinking_only_response_rejected(self):
        self.mock_model.chat.return_value = (
            "<think>Reasoning without an answer.</think>"
        )

        with self.assertRaises(ValueError):
            self.agent.run("Hello")

        self.assertEqual(len(self.agent.messages), 1)

    def test_next_turn_uses_clean_history(self):
        self.mock_model.chat.side_effect = [
            "<think>First reasoning.</think>\n47.75",
            "<think>Second reasoning.</think>\n"
            "The previous mean was 47.75.",
        ]

        first_answer = self.agent.run(
            "Calculate the mean."
        )

        second_answer = self.agent.run(
            "What was the previous mean?"
        )

        self.assertEqual(first_answer, "47.75")
        self.assertEqual(
            second_answer,
            "The previous mean was 47.75.",
        )

        second_request = (
            self.mock_model.chat.call_args_list[1].args[0]
        )

        self.assertEqual(
            second_request[-2],
            {
                "role": "assistant",
                "content": "47.75",
            },
        )

        self.assertEqual(len(self.agent.messages), 5)

        for message in self.agent.messages:
            self.assertNotIn(
                "<think>",
                message["content"],
            )


if __name__ == "__main__":
    unittest.main()
