
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import call, patch

from src.agent.cli import main


class TestAgentCLI(unittest.TestCase):

    def run_cli(self, inputs, run_error=None):
        """Run the CLI with simulated user input."""

        output = StringIO()

        with patch("src.agent.cli.AgentHarness") as mock_class:
            agent = mock_class.return_value
            agent.tools_enabled = False
            agent.run.return_value = "The mean is 47.75."

            if run_error is not None:
                agent.run.side_effect = run_error

            def set_tool_mode(enabled):
                agent.tools_enabled = enabled

            agent.set_tool_mode.side_effect = set_tool_mode

            with patch(
                "builtins.input",
                side_effect=inputs,
            ):
                with redirect_stdout(output):
                    main()

            return agent, output.getvalue()

    def test_default_mode_and_exit(self):
        agent, output = self.run_cli(["/exit"])

        self.assertIn("Tool calling: disabled", output)
        self.assertIn("Session ended.", output)
        agent.run.assert_not_called()

    def test_enable_tool_mode(self):
        agent, output = self.run_cli([
            "/tools on",
            "Calculate the mean of [13, 25, 61, 92].",
            "/exit",
        ])

        agent.set_tool_mode.assert_called_once_with(True)

        agent.run.assert_called_once_with(
            "Calculate the mean of [13, 25, 61, 92]."
        )

        self.assertIn("Tool calling enabled.", output)
        self.assertIn("The mean is 47.75.", output)

    def test_disable_tool_mode(self):
        agent, output = self.run_cli([
            "/tools on",
            "/tools off",
            "/exit",
        ])

        self.assertEqual(
            agent.set_tool_mode.call_args_list,
            [call(True), call(False)],
        )

        self.assertIn("Tool calling disabled.", output)

    def test_tool_mode_status(self):
        agent, output = self.run_cli([
            "/tools status",
            "/tools on",
            "/tools status",
            "/tools off",
            "/tools status",
            "/exit",
        ])

        self.assertEqual(
            output.count("Tool calling is disabled."),
            2,
        )

        self.assertEqual(
            output.count("Tool calling is enabled."),
            1,
        )

    def test_reset_command(self):
        agent, output = self.run_cli([
            "/reset",
            "/exit",
        ])

        agent.reset.assert_called_once_with()

        self.assertIn(
            "Conversation history cleared.",
            output,
        )

    def test_help_command(self):
        agent, output = self.run_cli([
            "/help",
            "/exit",
        ])

        self.assertIn("/tools on", output)
        self.assertIn("/tools off", output)
        self.assertIn("/tools status", output)
        self.assertIn("/reset", output)
        self.assertIn("/exit", output)

        agent.run.assert_not_called()

    def test_unknown_command_rejected(self):
        agent, output = self.run_cli([
            "/unknown",
            "/exit",
        ])

        self.assertIn("Unknown command.", output)
        agent.run.assert_not_called()

    def test_empty_input_ignored(self):
        agent, output = self.run_cli([
            "",
            "   ",
            "/exit",
        ])

        agent.run.assert_not_called()

    def test_model_failure_reported(self):
        agent, output = self.run_cli(
            ["Hello", "/exit"],
            run_error=RuntimeError("Simulated failure"),
        )

        self.assertIn(
            "Request failed: RuntimeError: Simulated failure",
            output,
        )

        self.assertIn("Session ended.", output)


if __name__ == "__main__":
    unittest.main()
    