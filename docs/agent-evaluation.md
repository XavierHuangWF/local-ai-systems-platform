
# End-to-End Local AI Agent Evaluation

## Objective

Evaluate the integrated local Qwen3 agent across three
real model-driven workflows:

1. Direct Python statistics execution.
2. Model Context Protocol (MCP) statistics execution.
3. Sequential document retrieval and statistics calculation.

Measure tool-selection behavior, output correctness,
and execution latency.

## Environment

- Model: Qwen3-8B-AWQ
- Inference server: vLLM
- Agent: AgentHarness
- Tool execution: ToolExecutor
- Retrieval: Local hybrid RAG
- MCP transport: stdio
- Model context: 8,192 tokens

## Evaluation Script

`scripts/evaluate_agent_end_to_end.py`

Command:

    python -m scripts.evaluate_agent_end_to_end

Each scenario creates a fresh AgentHarness instance.

The script records tool events, checks expected answer
markers, and measures wall-clock execution time.

No automatic retry is performed within the benchmark.

## Evaluation Scenarios

### Scenario 1: Direct Python Statistics

Input:

    [4, 8, 12, 16]

Required tool:

    summarize_measurements

Expected mean: 10.0

### Scenario 2: MCP Statistics

Input:

    [15, 25, 35, 45, 60]

Required tool:

    mcp_summarize_measurements

Expected mean: 36.0

### Scenario 3: Multi-Round Retrieval and Calculation

The agent must retrieve the four documented measurements
from the live Agent Harness verification experiment,
then calculate their statistics.

Required execution sequence:

1. search_project_documents
2. summarize_measurements

Expected measurements:

    [13, 25, 61, 92]

Expected mean: 47.75

Expected source: tool-execution.md

## First Evaluation Run

| Scenario | Result | Duration |
|---|---|---:|
| Direct Python statistics | PASS | 8.10 s |
| MCP statistics | PASS | 7.35 s |
| Multi-round retrieval and calculation | FAIL | 6.17 s |

Overall: 2/3 passed.

### Failure Analysis

In the multi-round scenario, the model returned ordinary
text containing `<tool_call>` markup rather than structured
tool calls.

Observed event:

    Round 1: Model requested no additional tools

The generated text contained a document-search request
and a statistics request with a placeholder for values
that had not actually been retrieved.

Neither tool executed.

The precise cause within model generation or vLLM tool-call
parsing was not established.

## Safeguard Added

Updated `src/agent/tool_loop.py` to detect unparsed
`<tool_call>` markup in plain assistant text.

When this pattern is detected, the tool loop raises
RuntimeError instead of returning the markup as a
successful answer.

This safeguard does not recover malformed tool calls,
automatically retry requests, or guarantee correct tool
selection.

The second benchmark run did not trigger this rejection
path, so that path has not yet been independently verified.

## Second Evaluation Run

| Scenario | Result | Duration |
|---|---|---:|
| Direct Python statistics | PASS | 13.25 s |
| MCP statistics | PASS | 7.34 s |
| Multi-round retrieval and calculation | PASS | 28.86 s |

Overall: 3/3 passed.

### Successful Multi-Round Execution

Observed execution:

    Round 1: search_project_documents
    Executed tool: search_project_documents
    Round 2: summarize_measurements
    Executed tool: summarize_measurements

Source attribution matched:

    tool-execution.md

Generated statistics:

| Metric | Result |
|---|---:|
| Count | 4 |
| Mean | 47.75 |
| Minimum | 13.0 |
| Maximum | 92.0 |

The required execution sequence and configured answer
checks passed.

## Engineering Findings

1. The direct Python tool worked in both evaluation runs.
2. The MCP statistics tool worked in both evaluation runs.
3. The multi-round workflow failed in the first run and
   succeeded in the second.
4. Qwen3 can emit tool-call markup as ordinary text.
5. Structured tool calls, not generated text, are required
   to authorize actual tool execution.
6. Source attribution successfully recognized the retrieved
   document in the second run.
7. Multi-round execution was slower than the other
   workflows in the second run.
8. The measured times include workload-dependent processing
   and initialization overhead and are not isolated benchmarks
   of the model, MCP, or retrieval components.

## Limitations

- Only two runs per workflow were observed.
- These results do not establish production reliability.
- The benchmark checks expected answer strings, not
  complete structured numerical correctness.
- The malformed tool-call rejection path was not exercised
  during the successful second benchmark.
- The cause of the initial malformed tool-call output
  remains unconfirmed.
- Source matching does not prove factual claim support.
- Latency measurements are not controlled for warm-up,
  caching, or individual component costs.
- No concurrency, load, fault-injection, or security
  penetration testing was performed.

## Decision

Status: Initial end-to-end integration evaluation completed.

The local agent demonstrated all three required workflows.

Retain the observed 2/3 and 3/3 results as separate runs.

Keep the malformed tool-call safeguard.

Do not claim production-grade reliability based on this
small evaluation.

No further functionality is required to close Day 10.
