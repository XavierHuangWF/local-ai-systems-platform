
import math
import unittest

from src.tools.registry import ToolRegistry
from src.tools.statistics_tool import summarize_measurements


class TestToolRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(
            "summarize_measurements",
            summarize_measurements,
        )

    def test_registration(self):
        self.assertEqual(
            self.registry.list_tools(),
            ["summarize_measurements"],
        )

    def test_tool_execution(self):
        result = self.registry.execute(
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

    def test_unknown_tool(self):
        with self.assertRaises(ValueError):
            self.registry.execute(
                "unknown_tool",
                {},
            )

    def test_duplicate_registration(self):
        with self.assertRaises(ValueError):
            self.registry.register(
                "summarize_measurements",
                summarize_measurements,
            )

    def test_invalid_tool_name(self):
        with self.assertRaises(ValueError):
            self.registry.register(
                "../unsafe_tool",
                summarize_measurements,
            )

    def test_non_callable_handler(self):
        with self.assertRaises(TypeError):
            self.registry.register(
                "invalid_handler",
                "not a function",
            )

    def test_invalid_argument_container(self):
        with self.assertRaises(TypeError):
            self.registry.execute(
                "summarize_measurements",
                [10, 20, 30],
            )

    def test_invalid_measurements(self):
        invalid_inputs = [
            [],
            [10, "invalid", 30],
            [10, math.nan],
            [10, math.inf],
            [True, 20],
            [1] * 10001,
        ]

        for values in invalid_inputs:
            with self.subTest(values=values[:5]):
                with self.assertRaises(
                    (ValueError, TypeError)
                ):
                    self.registry.execute(
                        "summarize_measurements",
                        {"values": values},
                    )


if __name__ == "__main__":
    unittest.main()
