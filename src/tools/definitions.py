
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        """Convert the definition into an OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


SUMMARIZE_MEASUREMENTS_DEFINITION = ToolDefinition(
    name="summarize_measurements",
    description=(
        "Calculate count, arithmetic mean, minimum, and maximum "
        "for a list of numerical measurements."
    ),
    parameters={
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "description": (
                    "A nonempty list of finite numerical measurements."
                ),
                "items": {
                    "type": "number",
                },
                "minItems": 1,
                "maxItems": 10000,
            }
        },
        "required": ["values"],
        "additionalProperties": False,
    },
)
