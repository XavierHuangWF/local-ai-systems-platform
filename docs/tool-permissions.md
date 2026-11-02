
# Tool Permission Enforcement

## Objective

Introduce explicit authorization controls for registered
Agent Harness tools.

Prevent the Large Language Model (LLM) from executing
a tool solely because it knows the registered tool name.

## Implementation

Files:

- `src/tools/permissions.py`
- `src/tools/executor.py`
- `src/agent/harness.py`
- `scripts/verify_tool_permissions.py`

### Permission Policy

`ToolPermissionPolicy` maintains an immutable allowlist
of authorized tool names.

`ToolPermissionError` is raised when a registered tool
is not included in the active policy.

### Execution Order

1. Check that the tool is registered.
2. Check the configured permission policy.
3. Validate arguments against the tool schema.
4. Execute the authorized Python handler.
5. Log successful tool execution.

Unauthorized tools are rejected before argument validation
and before their Python handlers execute.

### Agent Harness Configuration

The default Agent Harness explicitly permits:

- `summarize_measurements`
- `search_project_documents`
- `mcp_summarize_measurements`

Newly registered tools are denied until explicitly added
to the policy.

A custom permission policy can restrict an Agent Harness
instance to a smaller set of tools.

Standalone `ToolExecutor` instances without an explicit
policy retain backward-compatible permissive behavior
for registered tools.

## Reproducible Verification

Command:

    python -m scripts.verify_tool_permissions

### Observed Results

| Scenario | Result |
|---|---|
| Authorized statistics execution | Passed |
| Mean of [10, 20, 30, 40] | 25.0 |
| Unauthorized MCP execution | Denied |
| Unauthorized handler invocation count | 0 |
| Authorization before validation | Passed |
| Restricted Agent Harness: statistics | Allowed |
| Restricted Agent Harness: document search | Denied |
| Restricted Agent Harness: MCP statistics | Denied |

All verification checks passed.

The Agent Harness scenarios invoked its ToolExecutor
directly without requiring vLLM inference.

## Engineering Findings

1. Registration and authorization are separate concerns.
2. A registered handler can be denied by an explicit policy.
3. Authorization is checked before argument validation.
4. Unauthorized handlers are not invoked.
5. Existing tools remain functional under the default policy.

## Limitations

- Unauthorized tools may still appear in model-visible
  tool definitions.
- Authorization is enforced by ToolExecutor, not by the
  independent MCP server.
- The current policy is per executor instance, not per
  individual user or request.
- No user identity, role-based access control, or approval
  workflow has been implemented.
- A standalone ToolExecutor without a policy remains
  permissive for registered tools.
- This is application-level authorization, not process
  isolation or operating-system sandboxing.
- Model behavior after a denied tool request has not
  been evaluated.

## Conclusion

Explicit tool authorization is implemented and verified
at the Agent Harness execution boundary.

Further permission and sandbox features can be developed
as separate engineering milestones.
