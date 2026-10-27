
import unittest

from jsonschema.exceptions import SchemaError

from src.tools.definitions import (
    ToolDefinition,
    SUMMARIZE_MEASUREMENTS_DEFINITION,
)
from src.tools.executor import ToolExecutor
from src.tools.statistics_tool import summarize_measurements


class TestToolExecutor(unittest.TestCase):

    def setUp(self):
        self.executor = ToolExecutor()

    def test_registration_exposes_tool_schema(self):
        self.executor.register(
            SUMMARIZE_MEASUREMENTS_DEFINITION,
            summarize_measurements,
        )

        definitions = self.executor.get_tool_definitions()

        self.assertEqual(
            definitions,
            [
                SUMMARIZE_MEASUREMENTS_DEFINITION.to_openai_format()
            ],
        )

    def test_valid_arguments_execute(self):
        self.executor.register(
            SUMMARIZE_MEASUREMENTS_DEFINITION,
            summarize_measurements,
        )

        result = self.executor.execute(
            "summarize_measurements",
            {"values": [10, 20, 30, 40]},
        )

        self.assertEqual(
            result,
            {
                "count": 4,
                "mean": 25.0,
                "minimum": 10.0,
                "maximum": 40.0,
            },
        )

    def test_invalid_arguments_never_reach_handler(self):
        execution_count = 0

        def monitored_handler(values):
            nonlocal execution_count
            execution_count += 1
            return summarize_measurements(values)

        self.executor.register(
            SUMMARIZE_MEASUREMENTS_DEFINITION,
            monitored_handler,
        )

        invalid_cases = {
            "missing_field": {},
            "incorrect_type": {"values": "10,20,30"},
            "nonnumeric_element": {
                "values": [10, "invalid", 30]
            },
            "unexpected_field": {
                "values": [10, 20],
                "extra": True,
            },
            "empty_array": {"values": []},
            "boolean_element": {"values": [True, 20]},
            "oversized_array": {"values": [1] * 10001},
        }

        for label, arguments in invalid_cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    self.executor.execute(
                        "summarize_measurements",
                        arguments,
                    )

                self.assertEqual(execution_count, 0)

    def test_invalid_argument_container(self):
        self.executor.register(
            SUMMARIZE_MEASUREMENTS_DEFINITION,
            summarize_measurements,
        )

        with self.assertRaises(TypeError):
            self.executor.execute(
                "summarize_measurements",
                [10, 20, 30],
            )

    def test_unknown_tool_rejected(self):
        with self.assertRaises(ValueError):
            self.executor.execute(
                "unknown_tool",
                {},
            )

    def test_duplicate_registration_rejected(self):
        self.executor.register(
            SUMMARIZE_MEASUREMENTS_DEFINITION,
            summarize_measurements,
        )

        with self.assertRaises(ValueError):
            self.executor.register(
                SUMMARIZE_MEASUREMENTS_DEFINITION,
                summarize_measurements,
            )

        self.assertEqual(
            len(self.executor.get_tool_definitions()),
            1,
        )

    def test_invalid_definition_rejected(self):
        with self.assertRaises(TypeError):
            self.executor.register(
                "not_a_tool_definition",
                summarize_measurements,
            )

        self.assertEqual(
            self.executor.get_tool_definitions(),
            [],
        )

    def test_invalid_schema_rejected(self):
        definition = ToolDefinition(
            name="invalid_schema",
            description="Invalid schema test.",
            parameters={
                "type": "unsupported_type"
            },
        )

        with self.assertRaises(SchemaError):
            self.executor.register(
                definition,
                summarize_measurements,
            )

        self.assertEqual(
            self.executor.get_tool_definitions(),
            [],
        )

    def test_non_callable_handler_rejected(self):
        with self.assertRaises(TypeError):
            self.executor.register(
                SUMMARIZE_MEASUREMENTS_DEFINITION,
                "not_a_function",
            )

        self.assertEqual(
            self.executor.get_tool_definitions(),
            [],
        )

    def test_handler_failure_propagates(self):
        def failing_handler(values):
            raise RuntimeError("Simulated tool failure")

        self.executor.register(
            SUMMARIZE_MEASUREMENTS_DEFINITION,
            failing_handler,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Simulated tool failure",
        ):
            self.executor.execute(
                "summarize_measurements",
                {"values": [10, 20]},
            )


if __name__ == "__main__":
    unittest.main()
