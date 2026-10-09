
import unittest
from unittest.mock import patch

from src.agent.harness import AgentHarness, SYSTEM_PROMPT


class TestAgentHarness(unittest.TestCase):


    def setUp(self):
        model_patcher = patch("src.agent.harness.VLLMClient")
        token_patcher = patch("src.agent.harness.TokenCounter")

        self.addCleanup(model_patcher.stop)
        self.addCleanup(token_patcher.stop)

        self.fake_model = model_patcher.start().return_value
        self.fake_counter = token_patcher.start().return_value

        self.fake_counter.fits.return_value = True

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


    def test_history_window(self):
        self.fake_model.chat.return_value = "Test response"

        # Simulate seven conversation exchanges.
        for number in range(1, 8):
            self.agent.run(f"Question {number}")

        # Extract stored user messages.
        user_messages = [
            message["content"]
            for message in self.agent.messages
            if message["role"] == "user"
        ]

        # Question 1 should have been removed.
        self.assertEqual(
            user_messages,
            [
                "Question 2",
                "Question 3",
                "Question 4",
                "Question 5",
                "Question 6",
                "Question 7",
            ],
        )

        # One system message + six user/assistant pairs.
        self.assertEqual(len(self.agent.messages), 13)

        # System instructions must be preserved.
        self.assertEqual(
            self.agent.messages[0]["role"],
            "system",
        )

        def test_token_budget_removes_oldest_exchange(self):
            self.fake_model.chat.return_value = "Test response"

            # Create two completed conversation exchanges.
            self.agent.run("Question 1")
            self.agent.run("Question 2")

            # Simulate a conversation that initially exceeds
            # the token budget but fits after removing history.
            self.fake_counter.fits.reset_mock()
            self.fake_counter.fits.side_effect = [False, True]

            # Submit another question.
            response = self.agent.run("Question 3")

            # Inspect the messages sent to the model.
            sent_messages = self.fake_model.chat.call_args.args[0]

            user_messages = [
                message["content"]
                for message in sent_messages
                if message["role"] == "user"
            ]

            # The oldest question must be removed.
            self.assertEqual(
                user_messages,
                ["Question 2", "Question 3"],
            )

            # Token budget was checked twice.
            self.assertEqual(
                self.fake_counter.fits.call_count,
                2,
            )

            # The system instructions must remain.
            self.assertEqual(
                sent_messages[0]["role"],
                "system",
            )

            self.assertEqual(response, "Test response")



    def test_token_budget_removes_oldest_exchange(self):
        self.fake_model.chat.return_value = "Test response"

        # Create two completed conversation exchanges.
        self.agent.run("Question 1")
        self.agent.run("Question 2")

        # Simulate a conversation that initially exceeds
        # the token budget but fits after removing history.
        self.fake_counter.fits.reset_mock()
        self.fake_counter.fits.side_effect = [False, True]

        # Submit another question.
        response = self.agent.run("Question 3")

        # Inspect the messages sent to the model.
        sent_messages = self.fake_model.chat.call_args.args[0]

        user_messages = [
            message["content"]
            for message in sent_messages
            if message["role"] == "user"
        ]

        # The oldest question must be removed.
        self.assertEqual(
            user_messages,
            ["Question 2", "Question 3"],
        )

        # Token budget was checked twice.
        self.assertEqual(
            self.fake_counter.fits.call_count,
            2,
        )

        # The system instructions must remain.
        self.assertEqual(
            sent_messages[0]["role"],
            "system",
        )

        self.assertEqual(response, "Test response")


    def test_oversized_user_input(self):
        # Simulate an input that exceeds the token budget.
        self.fake_counter.fits.return_value = False

        # The harness should reject the request.
        with self.assertRaisesRegex(
            ValueError,
            "Input exceeds the available token budget.",
        ):
            self.agent.run("An oversized user question")

        # The model must not be called.
        self.fake_model.chat.assert_not_called()

        # Failed input must not change conversation history.
        self.assertEqual(
            self.agent.messages,
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ],
        )

        print("PASS: Oversized input rejected without model call.")



if __name__ == "__main__":
    unittest.main()
