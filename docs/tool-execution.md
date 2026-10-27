# Controlled Tool Execution

## Objective and current status

Build a local Large Language Model (LLM) agent that can request approved Python functions, validate their arguments, execute them, and incorporate results into a final answer. Keep model inference, tool definitions, validation, execution, conversation management, and presentation as separate components.

**Current implementation:** The interactive `AgentHarness` supports optional tool calling through `/tools on` and `/tools off`. A live `Qwen/Qwen3-8B-AWQ` session successfully requested the registered `summarize_measurements` function; the Python executor logged its successful execution; and Qwen3 produced the correct answer. A subsequent conversational turn also recalled a previously computed mean. Leading Qwen3 `<think>...</think>` blocks are removed before answers are returned or saved to conversation history.

**Automated regression evidence:** The latest *captured, complete* terminal result is **84 tests passed in 0.450 seconds**. Four additional logging tests were subsequently created, bringing the expected discovery count to **88**; their final full-suite output should be confirmed and recorded before claiming 88 passed. Earlier test totals are historical milestones, not additional tests.

The results demonstrate the tested workflows, **not** general tool-choice reliability, authorization, sandbox isolation, or safe execution of arbitrary tools.

## System architecture

```text
Start vLLM: scripts/serve_vllm.sh
    |
    v
Interactive user: python -m src.agent.cli
    |
    v
src/agent/harness.py: AgentHarness.run()
    |-- active skill instructions and conversation history
    |-- input token-budget check and history trimming
    |
    +-- tools disabled --> VLLMClient.chat() --> Qwen3 response ------+
    |                                                               |
    +-- tools enabled --> src/agent/tool_loop.py: run_tool_turn()    |
                            |                                        |
                            +--> request_with_tools() --> Qwen3      |
                            |       ^ tool schema from ToolExecutor |
                            |       v structured tool requests      |
                            +--> parse JSON arguments               |
                            +--> ToolExecutor.execute()              |
                            |      |                                 |
                            |      +--> ToolArgumentValidator       |
                            |      +--> ToolRegistry                |
                            |               |                       |
                            |               v                       |
                            |       summarize_measurements()        |
                            |               |                       |
                            |       successful execution log       |
                            |               |                       |
                            +--> tool result + tool_call_id         |
                            +--> Qwen3 final response --------------+
                                                                    |
                                                                    v
                                         clean_model_response()
                                                                    |
                                                                    v
                                    Save clean answer in history; display
```

`src/tools/definitions.py` supplies the tool description and JSON Schema at registration/request time; it is not a separate executable step after model inference. Qwen3 **requests** functions; Python code in the application validates and executes them. Tool execution happens only when tools are enabled and the model requests one.

## Implementation components

| File | Responsibility |
|---|---|
| `scripts/serve_vllm.sh` | Launches the locally served Qwen3 model with automatic tool calling enabled |
| `src/client/vllm_client.py` | Configures the OpenAI-compatible client and implements ordinary chat |
| `src/client/tool_calling.py` | Sends tool definitions to the model and preserves structured tool requests |
| `src/tools/definitions.py` | Declares the `summarize_measurements` tool and its argument schema |
| `src/tools/validator.py` | Validates arguments using `jsonschema` Draft 2020-12 |
| `src/tools/registry.py` | Stores approved Python functions and dispatches by registered name |
| `src/tools/statistics_tool.py` | Computes count, arithmetic mean, minimum, and maximum |
| `src/tools/executor.py` | Registers schemas and handlers; validates and dispatches tool requests; logs successful execution |
| `src/agent/tool_loop.py` | Coordinates model request, tool execution, tool-result message, and final model response |
| `src/agent/response_formatter.py` | Removes complete leading thinking blocks and rejects malformed or empty final answers |
| `src/agent/harness.py` | Manages history, skills, token-budget checks, optional tool mode, and cleaned responses |
| `src/agent/cli.py` | Provides interactive prompts, tool-mode controls, and execution-log output |

### Tool registration and argument validation

`ToolRegistry` accepts explicitly registered function names and Python callables. It rejects invalid names, duplicate registrations, unknown tools, and non-dictionary argument containers. **It does not independently enforce JSON Schema.** Model-generated requests use `ToolExecutor`, which applies the corresponding `ToolArgumentValidator` before invoking the registry.

`SUMMARIZE_MEASUREMENTS_DEFINITION` requires a top-level `values` array of **1–10,000 numeric elements**, with no unexpected top-level properties. The tool function independently rejects non-lists, empty lists, Boolean/nonnumeric elements, non-finite values (including Not a Number and infinity), and oversized inputs. Schema constraints and function-level checks serve different purposes.

### Tool execution and model round trip

`run_tool_turn()` submits registered tool definitions using `tool_choice="auto"`. If the model returns ordinary text, the tool is not executed. Otherwise, the function parses tool-call arguments, invokes `ToolExecutor`, adds tool-result messages with matching `tool_call_id` values, and requests a final answer from Qwen3.

The current implementation supports **one tool-calling round**, with a maximum of **three tool calls in that round**. It does not implement an indefinite planning loop. The original input message list is not modified by `run_tool_turn()`. When a request fails before the final answer, `AgentHarness` does not commit a new conversation turn. However, multiple requested tools are **not transactional**: a preceding tool may have already executed before a subsequent tool fails.

### Agent Harness and response formatting

`AgentHarness` uses normal chat by default. `set_tool_mode(True)` enables the validated tool loop; `set_tool_mode(False)` restores the original chat path. Its existing skill activation, conversation history (up to six turns), reset behavior, and input token-budget management are retained.

`clean_model_response()` removes complete **leading** `<think>...</think>` blocks, rejects unexpected/malformed thinking tags, and refuses empty final answers. It is applied before storing the assistant's reply. This is **output filtering**, not a guarantee that every form of sensitive or unwanted model output is detected. The input budget currently does not fully account for tool definitions and intermediate tool-result messages.

### Interactive controls and observability

The command-line interface supports:

| Command | Effect |
|---|---|
| `/tools on` | Enable tool calling |
| `/tools off` | Disable tool calling |
| `/tools status` | Show the current tool mode |
| `/reset` | Clear conversation history |
| `/help` | Show available commands |
| `/exit` | End the session |

The dedicated `local_ai.tool_execution` logger emits `[Tool] Executed tool: summarize_measurements` **after** a registered handler returns successfully. It records the function name rather than the user's measurement values. A successful log establishes that the Python function ran; a correct generated answer alone would not establish that fact. The current logger is diagnostic observability, not an audit-grade authorization mechanism.

## Environment and reproduction

Run commands from the project root inside Ubuntu on Windows Subsystem for Linux (WSL):

```bash
source ~/.venv/bin/activate
cd /mnt/c/Users/Xavier/PycharmProjects/local-ai-systems-platform
```

Start vLLM in one terminal:

```bash
bash scripts/serve_vllm.sh
```

The verified model is `Qwen/Qwen3-8B-AWQ`, served as `qwen3-8b` at `127.0.0.1:8000`. The server configuration includes:

```text
--enable-auto-tool-choice
--tool-call-parser hermes
```

The startup log reported `Application startup complete.` Keep the server running when performing live tests. The unit tests and mocked verification scripts do not require it.

Run the interactive agent in another terminal:

```bash
python -m src.agent.cli
```

Then enter `/tools on` before requesting the registered statistics tool.

## Reproducible experiments and observed results

### Registry, validation, and executor

| Script | Observed outcome |
|---|---|
| `python -m scripts.verify_tool_registry` | All **5** checks passed; `[10, 20, 30, 40]` produced count 4, mean 25.0, minimum 10.0, maximum 40.0 |
| `python -m scripts.verify_tool_validator` | All **9** checks passed, including invalid types, fields, and array limits |
| `python -m scripts.verify_tool_executor` | Registered handler executed exactly once for the one valid input; seven invalid requests were rejected before execution; unknown and duplicate tools were rejected |

The validator verification truncates very long exception messages for readability. It does not truncate validation inputs or weaken their checks.

### Mocked model-client and orchestration checks

| Script | Observed outcome |
|---|---|
| `python -m scripts.verify_tool_calling` | All **5** checks passed: tool definitions forwarded, structured response preserved, empty definitions and completions rejected, ordinary text preserved |
| `python -m scripts.verify_tool_loop` | All **7** checks passed: real Python calculation returned through a simulated model round trip; invalid arguments and malformed JSON rejected; original messages preserved |

These tests establish Python orchestration behavior, not real model-generation reliability.

### Live model-to-tool-to-model experiment

Script:

```bash
python -m scripts.verify_live_tool_calling
```

Qwen3 was explicitly instructed to summarize `[13, 25, 61, 92]`. It issued a structured request for `summarize_measurements`. The Python handler executed and returned:

```python
{
    "count": 4,
    "mean": 47.75,
    "minimum": 13.0,
    "maximum": 92.0,
}
```

The final model answer reported the same four statistics. The verification checked **one actual handler invocation** and the presence of mean **47.75** in the final answer; the remaining values were observed in the output but were not individually asserted by that script.

```text
PASS: Qwen3 requested the Python tool.
PASS: Final answer contains the expected mean.
PASS: Live Qwen3 tool-calling verification completed.
```

### Live multi-turn Agent Harness experiment

Script:

```bash
python -m scripts.verify_live_harness_tools
```

With tool mode enabled, the first turn summarized `[13, 25, 61, 92]`; the monitored Python handler executed once and returned mean **47.75**. In the second turn, Qwen3 recalled **47.75** from conversation history. Both exchanges were retained.

An earlier run displayed Qwen3 `<think>` content. After integrating the response formatter, a subsequent live run returned the statistics and then **47.75** without displaying thinking tags. The final recorded outcome was:

```text
PASS: Live multi-turn AgentHarness verification completed.
```

**Observed intermittent failure:** One attempt produced `openai.InternalServerError` (HTTP 500), reporting an `EngineCore` issue. A subsequent run succeeded without an identified root cause. The retry does not establish that the server-side issue has been permanently fixed.

### Interactive execution with visible Python logging

The following was run through `python -m src.agent.cli` after entering `/tools on`:

```text
Use summarize_measurements to calculate the mean of [12, 24, 36, 48].
```

The live terminal showed:

```text
Tool calling enabled.
[Tool] Executed tool: summarize_measurements

Agent: The mean of the measurements [12, 24, 36, 48] is **30.0**.

The `summarize_measurements` function also returns:
- Count: 4
- Minimum: 12.0
- Maximum: 48.0
```

The successful-execution log provides direct evidence of Python dispatch. A separate interactive second-turn test also correctly recalled the earlier mean **47.75**. No `<think>` tags appeared in these final displayed responses.

### Automated regression milestones

Run the test suite with:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

| Recorded milestone | Observed terminal summary |
|---|---|
| Tool Executor | 42 tests in 0.294s — `OK` |
| Tool Loop | 52 tests in 0.270s — individual cases shown as `ok`; final `OK` not visible in the captured image |
| Agent Harness tool mode | 59 tests in 0.359s — `OK` |
| Response formatter | 69 tests in 0.385s — `OK` |
| Harness response formatting | 75 tests in 0.410s — `OK` |
| Interactive CLI | **84 tests in 0.450s — `OK`** |

After that last captured full-suite result, `tests/test_tool_logging.py` was added with four further cases. **Expected current count: 88.** Confirm the actual final `Ran ... tests` and `OK` output and update this line before treating 88 as a verified result; test creation by itself is not a passing result.

## Engineering findings

- **Decision versus execution:** Qwen3 selects a named function; the application parses its request and carries out the approved Python operation.
- **Validation before dispatch:** The executor checks model-generated arguments against the registered schema before invoking the handler. The statistics function applies additional domain-specific validation.
- **Deterministic computation:** The actual arithmetic is performed by Python, not generated as an unverified model calculation.
- **State management:** The Agent Harness commits the response to conversation history only after obtaining a valid, cleaned final answer.
- **Operational observability:** Logging confirms successful tool execution in the interactive interface without exposing the measurement list in the log line.
- **Reproducibility:** Saved scripts, command lines, observed numerical results, and limitations accompany the implementation.

## Known limitations and risks

1. **Reliability:** Only a small number of real numerical requests have been demonstrated. Tool-selection accuracy has not been benchmarked across diverse prompts.
2. **Security:** Registered handlers execute in the current Python process, with its permissions. Registration and schema validation do not constitute a sandbox, caller authorization, or protection against malicious registered code.
3. **Resource control:** No general execution timeouts, process isolation, or per-tool permission policy are implemented. Tool definitions and intermediate results are not fully included in input token-budget checks.
4. **Execution semantics:** At most three calls are allowed in a single tool-calling round. Calls are not atomic; effects of earlier successful calls cannot necessarily be undone if a later call fails.
5. **Response filtering:** Removing leading `<think>` blocks improves presentation but is not a general-purpose content-safety or confidentiality guarantee.
6. **Server stability:** One HTTP 500 / `EngineCore` failure was observed, followed by a successful retry without a confirmed root cause.
7. **Scope:** Only the `summarize_measurements` function has been exercised live. There is no general database, document-retrieval, or file-access tool yet.

## Next engineering direction

The interactive tool-enabled agent is functional for the verified scenario. Subsequent practice should prioritize **new, useful capabilities and real experiments** rather than a separate cluster of unit tests for every small implementation change. A practical next stage is retrieval-augmented generation (RAG) over local documents, using the same separation between model requests and approved application tools.

Before the Day 4 commit, run the current regression suite once and record its actual final status, then stage the intended source, scripts, tests, and documentation while excluding environments, caches, and generated logs.
