
import asyncio
import sys

from mcp import Client, StdioServerParameters


class MCPStatisticsTool:
    """Invoke the statistics tool through a separate MCP server."""

    def __call__(self, values: list[float]) -> dict:
        # ToolExecutor uses synchronous handlers.
        # Bridge to the asynchronous MCP client.
        return asyncio.run(self._invoke(values))

    async def _invoke(self, values: list[float]) -> dict:
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_server.server"],
        )

        # The MCP client manages the server subprocess.
        async with Client(server) as client:

            # Discover the remote tool.
            available = await client.list_tools()

            tool_names = {
                tool.name for tool in available.tools
            }

            if "summarize_measurements" not in tool_names:
                raise RuntimeError(
                    "MCP statistics tool was not discovered."
                )

            # Execute through the MCP protocol.
            result = await client.call_tool(
                "summarize_measurements",
                {"values": values},
            )

            if result.is_error:
                raise RuntimeError(
                    f"MCP tool execution failed: {result.content}"
                )

            if not isinstance(result.structured_content, dict):
                raise RuntimeError(
                    "MCP server did not return structured data."
                )

            return result.structured_content
