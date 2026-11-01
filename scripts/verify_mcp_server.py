
import asyncio
import sys

from mcp import Client, StdioServerParameters


async def main() -> None:
    print("=== MCP Server Verification ===")

    # Launch our MCP server in a separate Python process.
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.server"],
    )

    async with Client(server) as client:
        print("\nConnected to MCP server.")

        # Step 1: Discover available tools.
        available = await client.list_tools()

        print("\n=== Discovered Tools ===")
        for tool in available.tools:
            print(f"Name: {tool.name}")
            print(f"Description: {tool.description}")
            print(f"Input schema: {tool.input_schema}")

        tool_names = {
            tool.name for tool in available.tools
        }

        if "summarize_measurements" not in tool_names:
            raise AssertionError(
                "Statistics tool was not discovered."
            )

        print("\nPASS: MCP tool discovery.")

        # Step 2: Invoke the remote MCP tool.
        values = [10, 20, 30, 40]

        result = await client.call_tool(
            "summarize_measurements",
            {"values": values},
        )

        if result.is_error:
            raise RuntimeError(
                f"MCP tool execution failed: {result.content}"
            )

        statistics = result.structured_content

        print("\n=== MCP Tool Result ===")
        print(statistics)

        if not isinstance(statistics, dict):
            raise AssertionError(
                "Expected a structured JSON result."
            )

        # Step 3: Verify the actual calculation.
        expected = {
            "count": 4,
            "mean": 25.0,
            "minimum": 10.0,
            "maximum": 40.0,
        }

        for key, expected_value in expected.items():
            actual = statistics.get(key)

            if actual != expected_value:
                raise AssertionError(
                    f"{key}: expected {expected_value}, "
                    f"received {actual}"
                )

        print("\nPASS: MCP tool execution.")
        print("PASS: Statistics values verified.")

    print("\n=== MCP Verification Completed ===")


if __name__ == "__main__":
    asyncio.run(main())
