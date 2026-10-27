
import unittest
from unittest.mock import patch

from src.agent.harness import AgentHarness


class TestHarnessToolMode(unittest.TestCase):

    def setUp(self):
        # Replace the model and tokenizer with mocks.
        # No running vLLM server is required.
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
        self.mock_model.chat.return_value = "Normal chat response."

        self.mock_counter = self.mock_token_class.return_value
        self.mock_counter.fits.return_value = True

        self.agent = AgentHarness()

    def test_normal_chat_is_default(self):
        self.assertFalse(self.agent.tools_enabled)

        with patch(
            "src.agent.harness.run_tool_turn"
        ) as mock_tool_turn:
            answer = self.agent.run("Hello")

        self.assertEqual(answer, "Normal chat response.")
        self.mock_model.chat.assert_called_once()
        mock_tool_turn.assert_not_called()

    def test_enabled_tool_mode_uses_tool_loop(self):
        self.agent.set_tool_mode(True)

        with patch(
            "src.agent.harness.run_tool_turn",
            return_value="The mean is 25.0.",
        ) as mock_tool_turn:
            answer = self.agent.run(
                "Find the mean of 10, 20, 30, 40."
            )

        self.assertEqual(answer, "The mean is 25.0.")
        self.mock_model.chat.assert_not_called()

        mock_tool_turn.assert_called_once()

        kwargs = mock_tool_turn.call_args.kwargs

        self.assertIs(
            kwargs["client"],
            self.mock_model.client,
        )
        self.assertEqual(kwargs["model"], "qwen3-8b")
        self.assertIs(
            kwargs["executor"],
            self.agent.tool_executor,
        )

        self.assertEqual(
            kwargs["messages"][-1]["role"],
            "user",
        )

        self.assertEqual(
            self.agent.messages[-1]["content"],
            "The mean is 25.0.",
        )

    def test_disabling_tools_restores_normal_chat(self):
        self.agent.set_tool_mode(True)
        self.agent.set_tool_mode(False)

        with patch(
            "src.agent.harness.run_tool_turn"
        ) as mock_tool_turn:
            answer = self.agent.run("Hello again")

        self.assertEqual(answer, "Normal chat response.")
        mock_tool_turn.assert_not_called()
        self.mock_model.chat.assert_called_once()

    def test_invalid_tool_mode_rejected(self):
        for invalid_value in ["true", 1, None]:
            with self.subTest(value=invalid_value):
                with self.assertRaises(TypeError):
                    self.agent.set_tool_mode(invalid_value)

        self.assertFalse(self.agent.tools_enabled)

    def test_tool_failure_preserves_history(self):
        self.agent.set_tool_mode(True)

        original_messages = [
            dict(message) for message in self.agent.messages
        ]

        with patch(
            "src.agent.harness.run_tool_turn",
            side_effect=RuntimeError("Tool execution failed"),
        ):
            with self.assertRaises(RuntimeError):
                self.agent.run("Calculate statistics")

        self.assertEqual(
            self.agent.messages,
            original_messages,
        )

    def test_token_budget_rejects_before_tool_call(self):
        self.agent.set_tool_mode(True)
        self.mock_counter.fits.return_value = False

        with patch(
            "src.agent.harness.run_tool_turn"
        ) as mock_tool_turn:
            with self.assertRaises(ValueError):
                self.agent.run("Oversized request")

        mock_tool_turn.assert_not_called()
        self.mock_model.chat.assert_not_called()

    def test_reset_preserves_tool_mode(self):
        self.agent.set_tool_mode(True)

        with patch(
            "src.agent.harness.run_tool_turn",
            return_value="Tool response.",
        ):
            self.agent.run("Summarize measurements")

        self.assertEqual(len(self.agent.messages), 3)

        self.agent.reset()

        self.assertTrue(self.agent.tools_enabled)
        self.assertEqual(len(self.agent.messages), 1)
        self.assertEqual(
            self.agent.messages[0]["role"],
            "system",
        )


if __name__ == "__main__":
    unittest.main()
