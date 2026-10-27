
from collections.abc import Callable
from typing import Any

from src.tools.definitions import ToolDefinition
from src.tools.registry import ToolRegistry
from src.tools.validator import ToolArgumentValidator

import logging

logger = logging.getLogger("local_ai.tool_execution")

class ToolExecutor:
    """Validate and execute explicitly registered tools."""

    def __init__(self) -> None:
        self.registry = ToolRegistry()
        self.definitions: dict[str, ToolDefinition] = {}
        self.validators: dict[str, ToolArgumentValidator] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: Callable[..., Any],
    ) -> None:
        """Register a tool with its argument schema."""

        if not isinstance(definition, ToolDefinition):
            raise TypeError(
                "Tool definition must be a ToolDefinition."
            )

        # Validate the schema before registering the tool.
        validator = ToolArgumentValidator(definition)

        # The registry checks names, duplicates, and handlers.
        self.registry.register(
            name=definition.name,
            handler=handler,
        )

        self.definitions[definition.name] = definition
        self.validators[definition.name] = validator

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Validate arguments before executing a tool."""

        if name not in self.validators:
            raise ValueError(f"Unknown tool: {name}")

        # Reject invalid model-generated arguments.
        validated_arguments = self.validators[name].validate(
            arguments
        )

        # Execute only after validation succeeds.
        result = self.registry.execute(
            name=name,
            arguments=validated_arguments,
        )

        logger.info("Executed tool: %s", name)

        return result

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return registered tools in OpenAI-compatible format."""

        return [
            self.definitions[name].to_openai_format()
            for name in self.registry.list_tools()
        ]
