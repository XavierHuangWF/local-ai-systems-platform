import re
from collections.abc import Callable
from typing import Any


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
    ) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Invalid tool name.")

        if not callable(handler):
            raise TypeError("Tool handler must be callable.")

        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        self._tools[name] = handler

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a dictionary.")

        handler = self._tools[name]

        return handler(**arguments)

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())
