
import unittest

from src.agent.response_formatter import clean_model_response


class TestResponseFormatter(unittest.TestCase):

    def test_plain_response(self):
        result = clean_model_response("47.75")
        self.assertEqual(result, "47.75")

    def test_empty_thinking_block(self):
        result = clean_model_response(
            "<think></think>\n47.75"
        )
        self.assertEqual(result, "47.75")

    def test_thinking_block_with_content(self):
        result = clean_model_response(
            "<think>Let me calculate.</think>\n47.75"
        )
        self.assertEqual(result, "47.75")

    def test_multiline_thinking_block(self):
        content = (
            "<think>\n"
            "First, read the conversation.\n"
            "Then identify the reported mean.\n"
            "</think>\n"
            "The mean is 47.75."
        )

        result = clean_model_response(content)

        self.assertEqual(result, "The mean is 47.75.")

    def test_multiple_leading_thinking_blocks(self):
        content = (
            "<think>First block.</think>\n"
            "<think>Second block.</think>\n"
            "Final answer."
        )

        result = clean_model_response(content)

        self.assertEqual(result, "Final answer.")

    def test_unclosed_thinking_block_rejected(self):
        with self.assertRaises(ValueError):
            clean_model_response(
                "<think>Unfinished reasoning"
            )

    def test_unexpected_thinking_tags_rejected(self):
        with self.assertRaises(ValueError):
            clean_model_response(
                "Answer: 47.75\n"
                "<think>Unexpected content</think>"
            )

    def test_thinking_only_response_rejected(self):
        with self.assertRaises(ValueError):
            clean_model_response(
                "<think>Reasoning only</think>"
            )

    def test_whitespace_only_response_rejected(self):
        with self.assertRaises(ValueError):
            clean_model_response("   \n   ")

    def test_non_string_response_rejected(self):
        for value in [None, 123, ["47.75"]]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    clean_model_response(value)


if __name__ == "__main__":
    unittest.main()
