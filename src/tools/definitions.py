
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



DOCUMENT_SEARCH_DEFINITION = ToolDefinition(
    name="search_project_documents",
    description=(
        "Search the local AI Systems Platform documentation "
        "for implementation details, architecture, configuration, "
        "troubleshooting, and verified experiment results. "
        "Use this tool when the user asks about the project's "
        "code, design, or documented behavior. "
        "Returns relevant passages with source identifiers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A specific search question or phrase "
                    "describing the information to retrieve."
                ),
                "minLength": 1,
                "maxLength": 500,
            },
            "top_k": {
                "type": "integer",
                "description": (
                    "Maximum number of document passages "
                    "to retrieve. Defaults to 4."
                ),
                "minimum": 1,
                "maximum": 6,
                "default": 4,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)
