
import unittest

from jsonschema.exceptions import SchemaError

from src.tools.definitions import (
    ToolDefinition,
    SUMMARIZE_MEASUREMENTS_DEFINITION,
)
from src.tools.validator import ToolArgumentValidator


class TestToolDefinition(unittest.TestCase):

    def test_openai_format(self):
        definition = SUMMARIZE_MEASUREMENTS_DEFINITION
        result = definition.to_openai_format()

        self.assertEqual(result["type"], "function")
        self.assertEqual(
            result["function"]["name"],
            "summarize_measurements",
        )
        self.assertEqual(
            result["function"]["description"],
            definition.description,
        )
        self.assertEqual(
            result["function"]["parameters"],
            definition.parameters,
        )

    def test_schema_constraints(self):
        schema = SUMMARIZE_MEASUREMENTS_DEFINITION.parameters

        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["values"])
        self.assertFalse(schema["additionalProperties"])

        values_schema = schema["properties"]["values"]

        self.assertEqual(values_schema["type"], "array")
        self.assertEqual(values_schema["items"]["type"], "number")
        self.assertEqual(values_schema["minItems"], 1)
        self.assertEqual(values_schema["maxItems"], 10000)


class TestToolArgumentValidator(unittest.TestCase):

    def setUp(self):
        self.validator = ToolArgumentValidator(
            SUMMARIZE_MEASUREMENTS_DEFINITION
        )

    def test_valid_arguments(self):
        arguments = {"values": [10, 20, 30, 40]}

        result = self.validator.validate(arguments)

        self.assertEqual(result, arguments)

    def test_missing_required_field(self):
        with self.assertRaises(ValueError):
            self.validator.validate({})

    def test_incorrect_array_type(self):
        with self.assertRaises(ValueError):
            self.validator.validate(
                {"values": "10,20,30"}
            )

    def test_nonnumeric_element(self):
        with self.assertRaises(ValueError):
            self.validator.validate(
                {"values": [10, "invalid", 30]}
            )

    def test_boolean_element(self):
        with self.assertRaises(ValueError):
            self.validator.validate(
                {"values": [True, 20]}
            )

    def test_unexpected_field(self):
        with self.assertRaises(ValueError):
            self.validator.validate(
                {
                    "values": [10, 20],
                    "extra": True,
                }
            )

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            self.validator.validate(
                {"values": []}
            )

    def test_oversized_array(self):
        with self.assertRaises(ValueError):
            self.validator.validate(
                {"values": [1] * 10001}
            )

    def test_invalid_argument_container(self):
        with self.assertRaises(TypeError):
            self.validator.validate(
                [10, 20, 30]
            )

    def test_invalid_schema(self):
        definition = ToolDefinition(
            name="invalid_schema",
            description="Test invalid schema rejection.",
            parameters={
                "type": "unsupported_type"
            },
        )

        with self.assertRaises(SchemaError):
            ToolArgumentValidator(definition)


if __name__ == "__main__":
    unittest.main()
