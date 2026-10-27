
import logging
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from src.agent.cli import main
from src.tools.definitions import SUMMARIZE_MEASUREMENTS_DEFINITION
from src.tools.executor import ToolExecutor
from src.tools.statistics_tool import summarize_measurements


class TestToolExecutionLogging(unittest.TestCase):

    def setUp(self):
        self.executor = ToolExecutor()
        self.executor.register(
            SUMMARIZE_MEASUREMENTS_DEFINITION,
            summarize_measurements,
        )

    def test_successful_execution_logged(self):
        with self.assertLogs(
            "local_ai.tool_execution",
            level="INFO",
        ) as captured:
            result = self.executor.execute(
                "summarize_measurements",
                {"values": [10, 20, 30, 40]},
            )

        self.assertEqual(result["mean"], 25.0)
        self.assertEqual(
            [record.getMessage() for record in captured.records],
            ["Executed tool: summarize_measurements"],
        )

    def test_invalid_arguments_not_logged(self):
        with self.assertNoLogs(
            "local_ai.tool_execution",
            level="INFO",
        ):
            with self.assertRaises(ValueError):
                self.executor.execute(
                    "summarize_measurements",
                    {"values": "invalid"},
                )

    def test_failed_function_not_logged(self):
        executor = ToolExecutor()

        def failing_handler(values):
            raise RuntimeError("Simulated execution failure")

        executor.register(
            SUMMARIZE_MEASUREMENTS_DEFINITION,
            failing_handler,
        )

        with self.assertNoLogs(
            "local_ai.tool_execution",
            level="INFO",
        ):
            with self.assertRaises(RuntimeError):
                executor.execute(
                    "summarize_measurements",
                    {"values": [10, 20]},
                )

    def test_cli_displays_successful_execution(self):
        output = StringIO()
        log_output = StringIO()

        tool_logger = logging.getLogger(
            "local_ai.tool_execution"
        )

        def simulated_agent_response(prompt):
            result = self.executor.execute(
                "summarize_measurements",
                {"values": [10, 20, 30, 40]},
            )
            return f"Mean: {result['mean']}"

        # Isolate CLI logging configuration from other tests.
        with (
            patch.object(tool_logger, "handlers", []),
            patch.object(tool_logger, "level", logging.WARNING),
            patch.object(tool_logger, "propagate", True),
            patch("src.agent.cli.AgentHarness") as mock_harness,
        ):
            agent = mock_harness.return_value
            agent.tools_enabled = False

            def set_tool_mode(enabled):
                agent.tools_enabled = enabled

            agent.set_tool_mode.side_effect = set_tool_mode
            agent.run.side_effect = simulated_agent_response

            with (
                patch(
                    "builtins.input",
                    side_effect=[
                        "/tools on",
                        "Calculate the mean.",
                        "/exit",
                    ],
                ),
                redirect_stdout(output),
                redirect_stderr(log_output),
            ):
                main()

        self.assertIn(
            "[Tool] Executed tool: summarize_measurements",
            log_output.getvalue(),
        )
        self.assertIn(
            "Agent: Mean: 25.0",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
