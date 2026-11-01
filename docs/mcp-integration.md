
# Model Context Protocol (MCP) Integration

## Objective

Integrate a local Model Context Protocol (MCP) server
with the existing Qwen3 Agent Harness.

Reuse the existing Python statistics implementation
without duplicating its calculation logic.

## Environment

- Model: Qwen3-8B-AWQ
- Model server: vLLM
- MCP Python SDK: 2.3.0
- MCP transport: stdio
- Python environment: Ubuntu under WSL
- MCP server: local Python subprocess

## Implementation

| Component | File | Responsibility |
|---|---|---|
| MCP server | `src/mcp_server/server.py` | Expose statistics through MCP |
| MCP client adapter | `src/tools/mcp_statistics_tool.py` | Invoke the remote MCP function |
| Tool definition | `src/tools/definitions.py` | Declare agent-facing MCP tool |
| Agent integration | `src/agent/harness.py` | Register MCP adapter |
| Verification | `scripts/verify_mcp_server.py` | Test protocol communication |

The MCP server exposes `summarize_measurements`.

The existing statistics implementation performs the
underlying input validation and numerical calculations.

The agent-facing adapter is registered separately as
`mcp_summarize_measurements`, preserving the existing
direct Python statistics tool.

## MCP Architecture

    User
      |
      v
    Qwen3-8B-AWQ
      |
      v
    AgentHarness
      |
      v
    ToolExecutor
      |
      v
    MCPStatisticsTool
      |
      v
    MCP Client
      |
      | stdio
      v
    MCP Server (separate Python process)
      |
      v
    summarize_measurements()
      |
      v
    MCP result -> Qwen3 -> User

## Independent MCP Server Verification

Command:

    python -m scripts.verify_mcp_server

The MCP client launched the server subprocess,
discovered the registered tool, inspected its input
schema, and executed the statistics function.

Input:

    [10, 20, 30, 40]

Observed output:

| Metric | Result |
|---|---:|
| Count | 4 |
| Mean | 25.0 |
| Minimum | 10.0 |
| Maximum | 40.0 |

Verification:

    PASS: MCP tool discovery.
    PASS: MCP tool execution.
    PASS: Statistics values verified.

## Live Qwen3 Integration Verification

Start vLLM:

    bash scripts/serve_vllm.sh

Start the interactive agent:

    python -m src.agent.cli

Enable tools:

    /tools on

Test question:

    Calculate the count, mean, minimum, and maximum
    of [15, 25, 35, 45, 60].
    You must use the registered mcp_summarize_measurements
    tool through the MCP server, not the direct Python
    statistics tool.

Observed execution:

    Round 1 requested: mcp_summarize_measurements
    Executed tool: mcp_summarize_measurements
    Round 2: Model requested no additional tools

Observed results:

| Metric | Result |
|---|---:|
| Count | 5 |
| Mean | 36.0 |
| Minimum | 15.0 |
| Maximum | 60.0 |

The agent returned the correct statistics.

An earlier attempt failed with an APIConnectionError
because the vLLM server was not running. After
restarting vLLM, the MCP request succeeded.

## Engineering Findings

1. A separate Python process successfully served a tool
   using the MCP protocol.
2. The MCP client discovered and invoked that tool.
3. The existing statistics function was reused.
4. The Agent Harness successfully registered the MCP adapter.
5. Qwen3 selected the MCP-specific tool when explicitly requested.
6. The tool result was returned to Qwen3 for answer generation.

## Limitations

- The MCP adapter is registered manually.
- The Agent Harness does not dynamically discover arbitrary
  MCP servers and automatically register their tools.
- A separate MCP server process starts for each invocation.
- Persistent MCP connections are not implemented.
- A local subprocess inherits relevant process permissions
  and is not a security sandbox.
- No generalized MCP permission or authorization policy exists.
- MCP execution was verified for one statistics tool.
- Autonomous MCP selection without an explicit request
  has not been evaluated.
- The successful experiment does not establish broad
  reliability across different servers or tools.

## Conclusion

The initial end-to-end MCP integration is functional.

The verified workflow is:

Qwen3 -> MCP client -> MCP server -> Python function
-> Qwen3 answer.

Further MCP functionality can be developed separately
without blocking completion of this milestone.
