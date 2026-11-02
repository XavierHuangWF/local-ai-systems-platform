
from src.tools.definitions import (
    SUMMARIZE_MEASUREMENTS_DEFINITION,
    MCP_SUMMARIZE_MEASUREMENTS_DEFINITION,
)
from src.tools.executor import ToolExecutor
from src.tools.permissions import (
    ToolPermissionError,
    ToolPermissionPolicy,
)
from src.tools.statistics_tool import summarize_measurements
from src.agent.harness import AgentHarness

def main() -> None:

    # Scenario 4: Verify permissions through the Agent Harness.
    print("\n=== Agent Harness Permission Enforcement ===")

    restricted_agent = AgentHarness(
        permission_policy=ToolPermissionPolicy(
            allowed_tools=frozenset({
                "summarize_measurements",
            })
        )
    )

    # An authorized function should still execute.
    allowed_result = restricted_agent.tool_executor.execute(
        name="summarize_measurements",
        arguments={"values": [10, 20, 30, 40]},
    )

    assert allowed_result["mean"] == 25.0

    print("PASS: Agent Harness authorized tool executed.")

    # Both other registered tools must be denied.
    denied_tools = [
        "search_project_documents",
        "mcp_summarize_measurements",
    ]

    for tool_name in denied_tools:
        try:
            restricted_agent.tool_executor.execute(
                name=tool_name,
                arguments={},
            )
        except ToolPermissionError:
            print(f"PASS: Agent Harness denied {tool_name}.")
        else:
            raise AssertionError(
                f"Unauthorized execution permitted: {tool_name}"
            )

    print("PASS: Agent Harness permission policy enforced.")

    print("=== Tool Permission Verification ===")

    # Explicitly authorize only the direct statistics tool.
    policy = ToolPermissionPolicy(
        allowed_tools=frozenset({
            "summarize_measurements",
        })
    )

    executor = ToolExecutor(permission_policy=policy)

    # Register the allowed tool.
    executor.register(
        definition=SUMMARIZE_MEASUREMENTS_DEFINITION,
        handler=summarize_measurements,
    )

    # Simulate a registered MCP handler.
    # We deliberately do not start an MCP subprocess.
    execution_count = 0

    def restricted_mcp_handler(values: list[float]) -> dict:
        nonlocal execution_count
        execution_count += 1
        return {"count": len(values)}

    executor.register(
        definition=MCP_SUMMARIZE_MEASUREMENTS_DEFINITION,
        handler=restricted_mcp_handler,
    )

    # Scenario 1: Allowed execution.
    print("\n=== Authorized Tool ===")

    result = executor.execute(
        name="summarize_measurements",
        arguments={"values": [10, 20, 30, 40]},
    )

    print("Result:", result)

    assert result["count"] == 4
    assert result["mean"] == 25.0
    assert result["minimum"] == 10.0
    assert result["maximum"] == 40.0

    print("PASS: Authorized tool executed correctly.")

    # Scenario 2: A registered but unauthorized tool.
    print("\n=== Unauthorized Tool ===")

    try:
        executor.execute(
            name="mcp_summarize_measurements",
            arguments={"values": [10, 20, 30, 40]},
        )
    except ToolPermissionError as exc:
        print("Expected denial:", exc)
    else:
        raise AssertionError(
            "Unauthorized MCP tool was allowed to execute."
        )

    assert execution_count == 0

    print("PASS: Unauthorized tool was denied.")
    print("PASS: Unauthorized handler was never executed.")

    # Scenario 3: Authorization occurs before
    # argument-schema validation.
    print("\n=== Authorization Precedes Validation ===")

    try:
        executor.execute(
            name="mcp_summarize_measurements",
            arguments={"values": "invalid-input"},
        )
    except ToolPermissionError:
        print("PASS: Permission check ran before validation.")
    else:
        raise AssertionError(
            "Expected authorization rejection."
        )

    assert execution_count == 0

    print("\n=== Permission Verification Completed ===")


if __name__ == "__main__":
    main()
