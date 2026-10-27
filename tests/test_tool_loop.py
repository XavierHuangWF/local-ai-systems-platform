
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.agent.tool_loop import MAX_TOOL_CALLS, run_tool_turn
from src.tools.definitions import SUMMARIZE_MEASUREMENTS_DEFINITION
from src.tools.executor import ToolExecutor
from src.tools.statistics_tool import summarize_measurements


def completion(message):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )


def assistant_message(content=None, tool_calls=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )


def tool_call(arguments, name="summarize_measurements"):
    return SimpleNamespace(
        id="call_test_001",
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


class TestToolLoop(unittest.TestCase):

    def setUp(self):
        self.client = Mock()
        self.executor = ToolExecutor()
        self.execution_count = 0

        def monitored_handler(values):
            self.execution_count += 1
            return summarize_measurements(values)

        self.executor.register(
            SUMMARIZE_MEASUREMENTS_DEFINITION,
            monitored_handler,
        )

        self.messages = [
            {
                "role": "user",
                "content": "Summarize [13, 25, 61, 92].",
            }
        ]

    def run_turn(self, executor=None):
        return run_tool_turn(
            client=self.client,
            model="qwen3-8b",
            messages=self.messages,
            executor=executor if executor is not None
            else self.executor,
        )

    def test_complete_tool_workflow(self):
        def simulated_model(**kwargs):
            if "tools" in kwargs:
                return completion(
                    assistant_message(
                        tool_calls=[
                            tool_call(
                                '{"values": [13, 25, 61, 92]}'
                            )
                        ]
                    )
                )

            results = [
                message
                for message in kwargs["messages"]
                if message["role"] == "tool"
            ]

            self.assertEqual(len(results), 1)

            result = json.loads(results[0]["content"])

            self.assertEqual(result["mean"], 47.75)
            self.assertEqual(result["count"], 4)
            self.assertEqual(
                results[0]["tool_call_id"],
                "call_test_001",
            )

            return completion(
                assistant_message(
                    content=f"The mean is {result['mean']}."
                )
            )

        self.client.chat.completions.create.side_effect = (
            simulated_model
        )

        original_messages = [
            dict(message) for message in self.messages
        ]

        answer = self.run_turn()

        self.assertEqual(answer, "The mean is 47.75.")
        self.assertEqual(self.execution_count, 1)
        self.assertEqual(
            self.client.chat.completions.create.call_count,
            2,
        )
        self.assertEqual(self.messages, original_messages)

    def test_ordinary_response(self):
        self.client.chat.completions.create.return_value = (
            completion(
                assistant_message(
                    content="No tool is needed."
                )
            )
        )

        answer = self.run_turn()

        self.assertEqual(answer, "No tool is needed.")
        self.assertEqual(self.execution_count, 0)
        self.assertEqual(
            self.client.chat.completions.create.call_count,
            1,
        )

    def test_invalid_arguments_rejected(self):
        self.client.chat.completions.create.return_value = (
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call('{"values": "invalid"}')
                    ]
                )
            )
        )

        with self.assertRaises(ValueError):
            self.run_turn()

        self.assertEqual(self.execution_count, 0)

    def test_malformed_json_rejected(self):
        self.client.chat.completions.create.return_value = (
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call('{"values": [13, 25,}')
                    ]
                )
            )
        )

        with self.assertRaises(ValueError):
            self.run_turn()

        self.assertEqual(self.execution_count, 0)

    def test_unknown_tool_rejected(self):
        self.client.chat.completions.create.return_value = (
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call(
                            "{}",
                            name="unknown_tool",
                        )
                    ]
                )
            )
        )

        with self.assertRaises(ValueError):
            self.run_turn()

        self.assertEqual(self.execution_count, 0)

    def test_tool_call_limit(self):
        calls = [
            tool_call('{"values": [1, 2, 3]}')
            for _ in range(MAX_TOOL_CALLS + 1)
        ]

        self.client.chat.completions.create.return_value = (
            completion(
                assistant_message(tool_calls=calls)
            )
        )

        with self.assertRaises(ValueError):
            self.run_turn()

        self.assertEqual(self.execution_count, 0)

    def test_no_registered_tools(self):
        empty_executor = ToolExecutor()

        with self.assertRaises(ValueError):
            self.run_turn(executor=empty_executor)

        self.client.chat.completions.create.assert_not_called()

    def test_empty_initial_completion(self):
        self.client.chat.completions.create.return_value = (
            SimpleNamespace(choices=[])
        )

        with self.assertRaises(RuntimeError):
            self.run_turn()

        self.assertEqual(self.execution_count, 0)

    def test_empty_final_answer(self):
        self.client.chat.completions.create.side_effect = [
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call('{"values": [10, 20]}')
                    ]
                )
            ),
            completion(
                assistant_message(content="")
            ),
        ]

        with self.assertRaises(RuntimeError):
            self.run_turn()

        self.assertEqual(self.execution_count, 1)

    def test_unexpected_final_tool_call(self):
        self.client.chat.completions.create.side_effect = [
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call('{"values": [10, 20]}')
                    ]
                )
            ),
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call('{"values": [30, 40]}')
                    ]
                )
            ),
        ]

        with self.assertRaises(RuntimeError):
            self.run_turn()

        self.assertEqual(self.execution_count, 1)


if __name__ == "__main__":
    unittest.main()
