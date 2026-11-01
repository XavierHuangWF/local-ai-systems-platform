
from mcp.server import MCPServer

from src.tools.statistics_tool import (
    summarize_measurements as compute_statistics,
)


# Create the MCP server.
mcp = MCPServer("local-ai-statistics")


@mcp.tool()
def summarize_measurements(
    values: list[float],
) -> dict[str, int | float]:
    """
    Calculate descriptive statistics for numerical measurements.

    Returns the count, mean, minimum, and maximum.
    The existing Python statistics function performs the
    underlying validation and calculation.
    """
    return compute_statistics(values)


if __name__ == "__main__":
    # Start a local MCP server using standard input/output.
    mcp.run(transport="stdio")
