
from dataclasses import dataclass


class ToolPermissionError(PermissionError):
    """Raised when a tool is not authorized for execution."""


@dataclass(frozen=True)
class ToolPermissionPolicy:
    """Immutable allowlist of tools permitted for execution."""

    allowed_tools: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_tools, frozenset):
            raise TypeError(
                "allowed_tools must be a frozenset of tool names."
            )

        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.allowed_tools
        ):
            raise ValueError(
                "Allowed tool names must be non-empty strings."
            )

    def allows(self, name: str) -> bool:
        """Return whether this policy permits the named tool."""
        return name in self.allowed_tools

    def require_authorized(self, name: str) -> None:
        """Reject execution unless the tool is explicitly allowed."""
        if not self.allows(name):
            raise ToolPermissionError(
                f"Tool execution denied by policy: {name}"
            )
