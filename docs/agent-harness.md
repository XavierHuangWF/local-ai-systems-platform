# Agent Harness Architecture

## Objective

Implement a stateful Python agent harness that communicates with a locally hosted Large Language Model (LLM) through an OpenAI-compatible Application Programming Interface (API). Separate user interaction, conversation state, token-budget management, and model communication so that each component can be maintained and verified independently.

## Architecture

```text
User
  |
  v
Command-Line Interface (CLI)
  |  src/agent/cli.py
  v
AgentHarness
  |  src/agent/harness.py
  |-- Conversation history and system instructions
  |-- Input validation and context management
  |
  +--> TokenCounter
  |    src/agent/token_counter.py
  |    Qwen3 tokenizer and chat template
  |
  v
VLLMClient
  |  src/client/vllm_client.py
  v
OpenAI-compatible API
  |
  v
vLLM inference server
  |
  v
Qwen3-8B-AWQ on NVIDIA GPU
```

## Components

### Model client

**File:** `src/client/vllm_client.py`

- Reads model endpoint, authentication key, and model name from environment variables with local-development defaults.
- Sends chat-completion requests through the OpenAI-compatible API.
- Configures a request timeout and retries for eligible failures.
- Returns generated text to the calling application.
- Raises an error if the model returns no text.

### Agent harness

**File:** `src/agent/harness.py`

- Preserves system instructions and conversation state in `self.messages`.
- Accepts user input and rejects empty or whitespace-only requests.
- Limits history to the six most recent completed user–assistant exchanges.
- Uses the token counter to validate the next request before model invocation.
- Removes the oldest completed exchange if the request exceeds the input budget.
- Rejects a request that remains oversized when no old exchanges can be removed.
- Updates stored conversation state only after a successful model response.
- Supports `reset()` to clear conversation history while keeping system instructions.

### Token counter

**File:** `src/agent/token_counter.py`

- Loads the tokenizer associated with `Qwen/Qwen3-8B-AWQ`.
- Applies the model's chat template, including special message-formatting tokens.
- Counts tokens from `encoded["input_ids"]`.
- Checks whether the formatted input fits within the configured input budget.

### Command-line interface

**File:** `src/agent/cli.py`

- Provides an interactive prompt for user questions.
- Displays the assistant response or an error.
- Supports `/reset` and `/exit`.

## Conversation state and context policy

The harness stores an in-memory sequence of messages with `system`, `user`, and `assistant` roles. The model does not retain earlier turns by itself; the harness sends relevant history with each request.

| Configuration | Value |
|---|---:|
| Maximum model context | 8,192 tokens |
| Output-token reservation | 512 tokens |
| Input-token budget | 7,680 tokens |
| Maximum completed exchanges retained | 6 |

The request-handling flow is:

```text
Receive and validate user input
          |
          v
Select recent completed exchanges
          |
          v
Construct pending request with system instructions
          |
          v
Count tokens using the Qwen3 chat template
          |
          v
Fits input budget? -- No --> Remove oldest completed exchange
          |                            |
         Yes                           +--> Recount
          |                            |
          |                       No old exchange left?
          |                            |
          |                            +--> Reject if still oversized
          v
Call model through VLLMClient
          |
          v
Save user message and assistant response on success
```

The harness constructs a pending request without immediately changing `self.messages`. Input-validation failures and failed model calls therefore leave the previously stored history unchanged.

The six-exchange limit is not by itself sufficient: even a single long message may exceed the model's context capacity. Token validation provides a separate check.

## Running the application

Start the vLLM server from the project root:

```bash
source ~/.venv/bin/activate
bash scripts/serve_vllm.sh
```

Keep the server running. In another Ubuntu terminal, activate the same environment, enter the project root, and launch the interactive agent:

```bash
source ~/.venv/bin/activate
python -m src.agent.cli
```

Commands:

```text
/reset  Clear conversation history while preserving system instructions
/exit   Stop the interactive agent
```

## Verification and observed results

### Unit tests: agent behavior

**File:** `tests/test_harness.py`

**Command (from the project root):**

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

**Actual result:** Seven tests passed in **0.006 seconds**.

| Test | Verified behavior |
|---|---|
| `test_conversation_history` | A subsequent request includes previous conversation messages. |
| `test_empty_input` | Empty input is rejected. |
| `test_history_window` | Only the six most recent completed exchanges remain. |
| `test_model_failure` | A failed model request does not modify stored conversation history. |
| `test_oversized_user_input` | An oversized question is rejected without calling the model. |
| `test_reset` | Conversation reset retains the system instructions. |
| `test_token_budget_removes_oldest_exchange` | The oldest completed exchange is removed when the simulated token check fails. |

```text
Ran 7 tests in 0.006s

OK
```

The unit tests replace the model client and token counter with controlled mock objects. This isolates application behavior from inference-server availability and variable model outputs. These tests do not independently establish tokenization accuracy or real-model reliability.

### Token counter: real tokenizer verification

**Files:** `src/agent/token_counter.py`, `scripts/verify_token_counter.py`

**Command:**

```bash
python -m scripts.verify_token_counter
```

**Actual result:**

```text
Maximum context: 8192
Reserved output: 512
Input budget: 7680

Test 1: Normal Conversation
Actual input tokens: 24
Within budget: True
PASS: Normal conversation

Test 2: Oversized Conversation
Oversized input tokens: 12009
Within budget: False
PASS: Oversized conversation

PASS: All token counter checks passed.
```

The oversized case repeats `"example "` 12,000 times. It produced **12,009 input tokens** with the applied chat template. Repetition count and token count are not interchangeable; formatting tokens also contribute to the total.

### Real-model integration verification

**File:** `scripts/verify_agent_integration.py`

Start vLLM, then execute:

```bash
python -m scripts.verify_agent_integration
```

**Observed results:**

```text
Test 1: Model Response
Assistant: Acknowledged, Cypress.
PASS: Model returned a nonempty response.

Test 2: Conversation History
Assistant: Cypress
PASS: Conversation history preserved.

Test 3: Reset
PASS: Conversation reset.

PASS: Agent integration checks completed.
```

The first response demonstrated a successful end-to-end model call. The second demonstrated that Qwen3 could use the previous turn supplied by the harness to recall the synthetic codename `Cypress`. The script checked nonempty responses and the conversation-history structure; codename accuracy was inspected manually rather than asserted automatically.

The response output also contained empty `<think></think>` tags. That formatting can be handled separately from the harness's state-management logic.

## Debugging record: incorrect token count

**Initial symptom:** The first token-counter implementation reported `2` for both a short conversation and an oversized conversation.

**Root cause:** With `return_dict=True`, the tokenizer returned a dictionary. Applying `len()` to the entire dictionary counted its fields, not the token identifiers.

**Correction:** Select the input token identifiers explicitly:

```python
encoded = self.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
)

token_ids = encoded["input_ids"]
return len(token_ids)
```

**Verification:** The corrected version reported **24** tokens for the short conversation and **12,009** tokens for the oversized case; both verification assertions passed.

This defect illustrates why validation must include boundary cases and observed token counts, not only the absence of exceptions.

## Engineering decisions

### Separate model communication from application control

The harness manages state and input preparation; the client manages API requests. This allows either component to be changed or tested independently.

### Preserve complete exchanges

Discarding user and assistant messages together maintains a coherent message sequence. The system message remains in place.

### Validate before execution

Checking the input budget before model invocation prevents known oversized requests from being sent. A single oversized input is rejected if history trimming cannot make it fit.

### Preserve state on failure

A pending request is separate from committed conversation state. The harness updates stored history only after a successful model call.

### Separate unit verification from integration verification

Mock-based tests check deterministic application logic quickly. Real-model integration testing checks that the components communicate successfully. Both are necessary, but they establish different properties.

## Current limitations

- Older exchanges are discarded, not summarized or persisted; their information is unavailable to later turns in that session.
- Conversations and state are kept in one Python process and are not retained across restarts.
- Concurrent requests and multiple independent sessions are not implemented.
- The implementation is intended for text-only chat. Tool outputs and multimodal inputs are not covered by the current token tests.
- The tokenizer verification covers two representative inputs, not all possible inputs or server-side accounting differences.
- The integration checks confirm basic functionality, not long-duration reliability, concurrency safety, or broad model accuracy.
- Automatic trimming has been verified in unit tests, but not yet with an oversized real-model conversation.
- External tool execution, Model Context Protocol (MCP) integration, and dedicated security guardrails remain out of scope for this implementation.
