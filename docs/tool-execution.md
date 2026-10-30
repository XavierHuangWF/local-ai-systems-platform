# Controlled Tool Execution

## Objective and current status

Build a local Large Language Model (LLM) agent that can request approved Python functions, validate their arguments, execute them, and incorporate results into a final answer. Keep model inference, tool definitions, validation, execution, conversation management, and presentation as separate components.

**Current implementation:** The interactive `AgentHarness` supports optional tool calling through `/tools on` and `/tools off`. Qwen3 can choose between the registered `summarize_measurements` and `search_project_documents` tools. The harness requests up to two tool-selection rounds; the tool loop permits at most three rounds and three calls per round. A live sequential experiment retrieved `[13, 25, 61, 92]` from project documentation in round 1, then executed the statistics tool in round 2, correctly returning mean **47.75**. Context checks now account for registered tool definitions and accumulated tool messages before further model requests. Document search limits serialized evidence to 2,500 tokens, omitting passages that do not fit. Leading Qwen3 `<think>...</think>` blocks are removed before answers are returned or stored in history.

**Automated regression evidence:** The latest observed full-suite terminal result was **88 tests passed in 0.462 seconds** (`OK`) after adding bounded document-search results. Earlier totals below are historical milestones, not additional tests.

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
                            |       search_project_documents()      |
                            |               |                       |
                            |       successful execution log       |
                            |               |                       |
                            +--> tool result + tool_call_id         |
                            +--> check tool-aware context budget    |
                            +--> next decision (up to 2 rounds)     |
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
| `src/tools/definitions.py` | Declares `summarize_measurements` and `search_project_documents` with JSON argument schemas |
| `src/tools/validator.py` | Validates arguments using `jsonschema` Draft 2020-12 |
| `src/tools/registry.py` | Stores approved Python functions and dispatches by registered name |
| `src/tools/statistics_tool.py` | Computes count, arithmetic mean, minimum, and maximum |
| `src/tools/document_search_tool.py` | Uses the hybrid retriever with lazy initialization and a 2,500-token serialized evidence limit |
| `src/tools/executor.py` | Registers schemas and handlers; validates and dispatches tool requests; logs successful execution |
| `src/agent/tool_loop.py` | Coordinates bounded sequential tool rounds, tool-result messages, context checks, and final answer generation |
| `src/agent/response_formatter.py` | Removes complete leading thinking blocks and rejects malformed or empty final answers |
| `src/agent/harness.py` | Manages history, skills, token-budget checks, optional tool mode and automatic RAG mode, and cleaned responses |
| `src/agent/cli.py` | Provides interactive prompts, tool-mode controls, and execution-log output |

### Tool registration and argument validation

`ToolRegistry` accepts explicitly registered function names and Python callables. It rejects invalid names, duplicate registrations, unknown tools, and non-dictionary argument containers. **It does not independently enforce JSON Schema.** Model-generated requests use `ToolExecutor`, which applies the corresponding `ToolArgumentValidator` before invoking the registry.

`SUMMARIZE_MEASUREMENTS_DEFINITION` requires a top-level `values` array of **1–10,000 numeric elements**, with no unexpected top-level properties. The tool function independently rejects non-lists, empty lists, Boolean/nonnumeric elements, non-finite values (including Not a Number and infinity), and oversized inputs. Schema constraints and function-level checks serve different purposes.

### Tool execution and model round trip

`run_tool_turn()` submits registered tool definitions using `tool_choice="auto"`. If the model returns ordinary text, the tool is not executed. Otherwise, the function parses tool-call arguments, invokes `ToolExecutor`, adds tool-result messages with matching `tool_call_id` values, and requests a final answer from Qwen3.

The `run_tool_turn()` function supports **up to three bounded tool-selection rounds**, with at most **three tool calls per round**. Its default is one round for compatibility, while `AgentHarness` currently requests **two rounds**. After each round, Qwen3 may examine prior tool results and request further tools. The original input message list is not modified. When a request fails before the final answer, `AgentHarness` does not commit a new conversation turn. However, tool calls are **not transactional**: a preceding handler may already have executed before a subsequent failure.

### Agent Harness and response formatting

`AgentHarness` uses normal chat by default. `set_tool_mode(True)` enables the validated tool loop; `set_tool_mode(False)` restores the original chat path. Its existing skill activation, conversation history (up to six turns), reset behavior, and input token-budget management are retained.

`clean_model_response()` removes complete **leading** `<think>...</think>` blocks, rejects unexpected/malformed thinking tags, and refuses empty final answers. It is applied before storing the assistant's reply. This is **output filtering**, not a guarantee that every form of sensitive or unwanted model output is detected. Tool-aware input checks include tool definitions and accumulated intermediate messages. Document search independently limits its serialized result to 2,500 tokens by omitting passages that do not fit. Other oversized tool results can still trigger full-context rejection.

### Interactive controls and observability

The command-line interface supports:

| Command | Effect |
|---|---|
| `/tools on` | Enable tool calling |
| `/tools off` | Disable tool calling |
| `/tools status` | Show the current tool mode |
| `/rag on` | Enable automatic document retrieval (separate from tool mode) |
| `/rag off` | Disable automatic retrieval |
| `/rag status` | Show the current RAG mode |
| `/reset` | Clear conversation history |
| `/help` | Show available commands |
| `/exit` | End the session |

The dedicated `local_ai.tool_execution` logger emits messages such as `[Tool] Executed tool: summarize_measurements` and `[Tool] Executed tool: search_project_documents` **after** the corresponding registered handler returns successfully. It records the function name rather than the user's measurement values. A successful log establishes that the Python function ran; a correct generated answer alone would not establish that fact. The current logger is diagnostic observability, not an audit-grade authorization mechanism.

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

The four additional logging cases brought the discovered suite to 88 tests. The latest observed full-suite run after bounded document search reported **88 tests in 0.462s — `OK`**. These are regression checks; live model behavior and oversized-context rejection were verified separately through dedicated experiment scripts.

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
3. **Resource control:** No general execution timeouts, process isolation, or per-tool permission policy are implemented. Document search has a 2,500-token evidence limit, but there is no generic result truncation or summarization mechanism for other registered tools.
4. **Execution semantics:** The harness allows up to two sequential tool-selection rounds with at most three calls per round (the tool loop permits at most three rounds when configured). Calls are not atomic; effects of earlier successful calls cannot necessarily be undone if a later call fails.
5. **Response filtering:** Removing leading `<think>` blocks improves presentation but is not a general-purpose content-safety or confidentiality guarantee.
6. **Server stability:** One HTTP 500 / `EngineCore` failure was observed, followed by a successful retry without a confirmed root cause.
7. **Scope:** Both `summarize_measurements` and `search_project_documents` have been exercised live, including in a sequential two-round workflow. There is no general database or unrestricted file-access tool.

## Next engineering direction

The agent now supports document retrieval as a tool, sequential tool selection, tool-aware context checks, and bounded document-search results. Further work should focus on verifying source attribution, handling other large tool outputs, and measuring tool-selection reliability across varied questions. The verified experiments, limitations, and regression results are recorded below.


## Multi-Round Agent Tool Orchestration

### Objective

Extend the original single-round tool loop to support
sequential model-directed tool execution.

The agent should retrieve project documentation, inspect
the retrieved information, and then decide whether another
tool call is necessary.

### Implementation

**Files:**

- `src/agent/tool_loop.py`
- `src/agent/harness.py`
- `src/tools/document_search_tool.py`
- `src/tools/definitions.py`
- `scripts/verify_live_sequential_tools.py`

The updated `run_tool_turn()` supports:

- Configurable tool-selection rounds.
- Maximum 3 rounds.
- Maximum 3 tool calls per round.
- JSON argument parsing and validation through `ToolExecutor`.
- Tool results returned to Qwen3 between rounds.
- Final-answer generation after the allowed rounds.
- Tool-round execution logging.

The Agent Harness currently specifies `max_rounds=2`.

### Reproducible Live Experiment

**Command:**

```bash
python -m scripts.verify_live_sequential_tools
```

**Question:**

Search the project documentation for the four measurements
used in the live multi-turn Agent Harness verification.
Retrieve those values before calculating their statistics.

**Observed tool sequence:**

```text
Round 1 requested: search_project_documents
Executed tool: search_project_documents
Round 2 requested: summarize_measurements
Executed tool: summarize_measurements
```

**Retrieved measurements:**

`[13, 25, 61, 92]`

**Calculated results:**

| Metric | Result |
|---|---:|
| Count | 4 |
| Mean | 47.75 |
| Minimum | 13.0 |
| Maximum | 92.0 |

**Document cited:**

`tool-execution.md`

Section: Live multi-turn Agent Harness experiment.

**Verification output:**

```text
PASS: Sequential tool execution confirmed.
PASS: Expected mean 47.75 confirmed.
PASS: Expected document source identified.

PASS: Live sequential tool execution verification completed.
```

### Engineering Findings

1. Qwen3 selected document retrieval during the first round.
2. Retrieved information was available for the next model decision.
3. Qwen3 selected the statistics tool in the second round.
4. The Python statistics handler produced the expected values.
5. Round-level logging distinguished sequential tool selection
   from multiple tool requests within a single round.

### Limitations

- One successful live run does not establish general reliability.
- Model-directed tool selection can vary between runs.
- Tool-aware context checks run before model requests. Document search
  omits passages to fit its evidence budget; other oversized tool
  results can still trigger full-context rejection.
- Tool execution is not transactional; earlier tool side effects
  are not automatically rolled back if a later operation fails.
- Document search results can contain irrelevant or outdated
  information.
- Source citation correctness is not automatically verified.
- The current Agent Harness keeps automatic RAG mode and
  tool-calling mode separate. Document search is available
  as a registered tool when tool calling is enabled.

## Multi-Round Tool Context Budget Enforcement

### Objective

Prevent accumulated tool requests, document-search results,
and intermediate messages from exceeding Qwen3's configured
8,192-token context window.

### Implementation

Updated components:

- `src/agent/token_counter.py`
- `src/agent/tool_loop.py`
- `src/agent/harness.py`

Added `TokenCounter.count_with_tools()` to measure formatted
conversation messages together with registered tool definitions.

The tool loop checks context usage before each model request,
including requests after previous tool executions.

The current configuration reserves 512 output tokens, leaving
a maximum of 7,680 input tokens.

### Token-Counting Experiment

Command:

```bash
python -m scripts.inspect_tool_context
```

Observed results:

| Stage | Message-only tokens | Tool-aware tokens |
|---|---:|---:|
| Initial request | 39 | 409 |
| After document search | 732 | 1,102 |
| After statistics | 815 | 1,185 |

Registered tool definitions added approximately 370 tokens
to the formatted requests in this experiment.

### Live Sequential Tool Verification

Command:

```bash
python -m scripts.verify_live_sequential_tools
```

The experiment completed successfully after the context
checks were enabled.

Observed execution order:

1. Round 1: `search_project_documents`
2. Round 2: `summarize_measurements`

The retrieved measurements were `[13, 25, 61, 92]`.

The Python statistics tool returned count 4, mean 47.75,
minimum 13.0, and maximum 92.0.

All recorded live verification checks passed.

### Oversized Context Verification

Command:

```bash
python -m scripts.verify_tool_context_overflow
```

Observed result:

```text
Tool request exceeds the context budget:
30363 tokens used, 7680 available.

PASS: Oversized context was rejected.
PASS: No second model request was made.
PASS: Context overflow verification completed.
```

The test used an intentionally oversized simulated
document-search result and did not require vLLM inference.

### Engineering Findings and Limitations

- Context counting must include registered tool definitions.
- Token usage grows as tool requests and results accumulate.
- Requests exceeding the configured budget are rejected
  before the next model completion.
- Document search bounds its own result by omitting passages.
  Other tools rely on the full-context rejection mechanism.
- Tool execution may have occurred before an overflow is
  detected; tool side effects are not transactional.
- Final-answer length and citation accuracy still require
  separate evaluation.
- The tests establish the observed behavior of these
  scenarios, not comprehensive reliability under all inputs.


## Bounded Document-Search Evidence

### Objective

Prevent large document-search results from consuming excessive
context tokens during multi-round agent execution.

### Implementation

Updated `src/tools/document_search_tool.py` to enforce a
2,500-token limit on serialized document-search results.

The tool:

- Retrieves passages using the existing HybridRetriever.
- Processes passages in retrieval-ranking order.
- Counts serialized JSON evidence with the Qwen3 tokenizer.
- Retains complete passages that fit the budget.
- Omits passages that would exceed the evidence budget.
- Reports the number of omitted passages.

The tool-loop context checks remain responsible for enforcing
the overall 8,192-token model context window.

### Reproducible Evidence-Budget Verification

Command:

    python -m scripts.verify_document_search_budget

Observed real-document retrieval:

| Measurement | Result |
|---|---:|
| Evidence tokens | 1,758 |
| Maximum allowed | 2,500 |
| Returned passages | 6 |
| Omitted passages | 0 |

Observed oversized-passage simulation:

| Measurement | Result |
|---|---:|
| Evidence tokens | 78 |
| Maximum allowed | 2,500 |
| Returned passages | 1 |
| Omitted passages | 1 |

The oversized passage was excluded, while the smaller
passage remained available.

All verification checks passed.

### Live Sequential Tool Verification

Command:

    python -m scripts.verify_live_sequential_tools

After an initial connection-refused error, the vLLM server
was restarted and the experiment was repeated successfully.

Observed sequence:

1. Round 1: search_project_documents
2. Round 2: summarize_measurements

Retrieved measurements: [13, 25, 61, 92]

Results:

- Count: 4
- Mean: 47.75
- Minimum: 13.0
- Maximum: 92.0

The answer cited tool-execution.md.

All live verification checks passed.

### Limitations

- The 2,500-token budget applies only to the document-search
  result, not the complete conversation.
- Oversized passages are omitted, not summarized or split.
- Relevant evidence may be excluded by the budget.
- The full context-budget check may still reject a request
  if conversation history and tool results consume too
  many tokens.
- One successful live run does not establish general
  tool-selection or answer-generation reliability.



## Retrieved Source Attribution Verification

### Objective

Check whether document references in Qwen3-generated answers
correspond to evidence actually returned by document search.

This verifies source provenance, not factual claim support.

### Implementation

Updated components:

- `src/agent/source_attribution.py`
- `src/agent/tool_loop.py`

The attribution checker:

- Extracts Markdown filenames and optional chunk identifiers.
- Matches filenames against retrieved document sources.
- Matches chunk identifiers against retrieved chunk IDs.
- Reports unsupported or fabricated references.
- Logs results after final answer generation.
- Preserves existing multi-round tool execution.

### Reproducible Verification

Command:

    python -m scripts.verify_source_attribution

Observed results:

| Reference | Result |
|---|---|
| `tool-execution.md::0004` | Matched |
| `architecture.md` | Matched |
| `nonexistent.md::9999` | Unsupported |
| `tool-execution.md::9999` | Unsupported |

All three verification checks passed.

### Live Qwen3 Verification

Command:

    python -m scripts.verify_live_sequential_tools

Observed execution:

1. Round 1: `search_project_documents`
2. Round 2: `summarize_measurements`
3. Attribution check matched `tool-execution.md`.

The calculated mean was 47.75.

### Regression Verification

Command:

    python -m unittest discover -s tests -p "test_*.py"

Observed result:

    Ran 88 tests in 0.439s
    OK

### Limitations

- Matching a source identifier does not prove factual support.
- The checker detects recognizable Markdown references, not
  every possible citation format.
- Ordinary filename mentions may also be classified as references.
- Unsupported references are logged, not automatically removed.
- Citation-to-claim semantic verification is not implemented.
