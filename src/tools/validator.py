
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from src.tools.definitions import ToolDefinition


class ToolArgumentValidator:
    def __init__(self, definition: ToolDefinition) -> None:
        self.definition = definition

        # Verify that the tool's schema is valid.
        Draft202012Validator.check_schema(
            definition.parameters
        )

        self.validator = Draft202012Validator(
            definition.parameters
        )

    def validate(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        if not isinstance(arguments, dict):
            raise TypeError(
                "Tool arguments must be a dictionary."
            )

        try:
            self.validator.validate(arguments)

        except ValidationError as exc:
            location = ".".join(
                str(part) for part in exc.absolute_path
            ) or "arguments"

            raise ValueError(
                f"Invalid arguments for '{self.definition.name}' "
                f"at '{location}': {exc.message}"
            ) from exc

        return arguments
