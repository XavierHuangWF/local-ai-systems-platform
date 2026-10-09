
# Agent Harness Architecture

## Objective

Implement a stateful agent harness that communicates with a locally
hosted Large Language Model (LLM) through an Application Programming
Interface (API).

The harness separates model communication, conversation management,
and user interaction.

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
  v
VLLMClient
  |  src/client/vllm_client.py
  v
OpenAI-compatible API
  |
  v
vLLM Inference Server
  |
  v
Qwen3-8B-AWQ
```

## Components

### Model Client

File: `src/client/vllm_client.py`

Responsibilities:

- Configure the model endpoint and authentication.
- Communicate with the local vLLM server.
- Submit conversation messages.
- Return generated text.
- Configure request timeouts and retries.
- Raise errors when model calls fail.

### Agent Harness

File: `src/agent/harness.py`

Responsibilities:

- Initialize the model client.
- Preserve system instructions.
- Maintain conversation history.
- Append user questions.
- Save assistant responses.
- Reject empty input.
- Roll back the latest user message when a model call fails.
- Reset conversation state.

### Command-Line Interface

File: `src/agent/cli.py`

Responsibilities:

- Accept questions from the terminal.
- Display model responses.
- Keep the conversation running.
- Support `/reset` to clear conversation history.
- Support `/exit` to terminate the application.
- Display errors without terminating the interaction loop.

## Conversation State

Conversation history is stored in memory using:

```python
self.messages
```

The history contains messages with these roles:

- `system`: instructions defining assistant behavior.
- `user`: user questions and requests.
- `assistant`: previously generated responses.

Each model request receives the current message history.

The language model does not permanently remember the conversation.
The harness provides previous messages as context on subsequent requests.

The history exists only while the Python process is running.
It is not persisted to a database or file.

## Run the Application

First start the local vLLM server.

Then activate the Python virtual environment and run the application
from the repository root:

```bash
source ~/.venv/bin/activate
python -m src.agent.cli
```

Supported interactive commands:

```text
/reset  Clear conversation history.
/exit   Exit the application.
```

## Automated Unit Tests

File: `tests/test_harness.py`

The tests use Python's unittest framework and a simulated model
client to verify harness behavior independently of the vLLM server.

Run:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Verified results:

```text
test_conversation_history ... ok
test_empty_input ... ok
test_model_failure ... ok
test_reset ... ok

Ran 4 tests in 0.003s

OK
```

All four tests passed.

## Current Limitations

- Conversation history grows without an automatic size limit.
- The server is configured for a maximum sequence length of 8192 tokens.
- Long conversations may exceed the model's context limit.
- Conversation history is not persisted across application restarts.
- No external tools are available to the model.
- No Model Context Protocol (MCP) integration exists.
- No dedicated security guardrails have been implemented.

## Next Improvements

- Add conversation history limits.
- Implement context management.
- Add structured logging.
- Extend automated testing.
- Introduce tool execution with permission controls.



## Token Budget Verification

### Objective

Verify that the local Qwen3 tokenizer correctly counts input
tokens and identifies conversations that exceed the configured
context budget.

### Implementation

- `src/agent/token_counter.py`: Reusable token-counting component.
- `scripts/verify_token_counter.py`: Executable verification script.

The implementation uses the Qwen3 chat template to account for
conversation formatting and special tokens.

### Configuration

- Model: Qwen/Qwen3-8B-AWQ
- Maximum context: 8192 tokens
- Reserved output: 512 tokens
- Available input budget: 7680 tokens

### Reproduction

From the project root with the Python environment activated:

```bash
python -m scripts.verify_token_counter
```

The verification does not require a running model server.

### Actual Results

```text
Test 1: Normal Conversation
Actual input tokens: 24
Within budget: True
PASS: Normal conversation

Test 2: Oversized Conversation
Oversized input tokens: 12009
Within budget: False
PASS: Oversized conversation

Verification Result:
PASS: All token counter checks passed.
```

### Defect Identified and Corrected

The first implementation incorrectly returned 2 tokens for
both short and oversized conversations.

Root cause:

The tokenizer returned a dictionary containing token identifiers
and associated metadata. The implementation applied `len()` to
the dictionary, counting its fields instead of its token IDs.

Correction:

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

After correction, the short conversation contained 24 tokens,
while the oversized conversation contained 12009 tokens.

### Engineering Significance

Limiting conversation history by message count is not sufficient
to prevent context-window overflow.

A single long message may contain more tokens than several
short conversations combined.

The model-specific tokenizer provides token counts that can be
used for context-budget validation.

### Current Limitations

- Token counting is not yet integrated into AgentHarness.
- Older conversation turns are not yet removed based on token count.
- A single oversized user message still needs explicit handling.
- Only text conversations have been verified.
- The verification does not test actual model inference.
```

Press **Ctrl + S** to save.

## 3. What comes after documentation?

Our next engineering task is to connect the token counter to your harness:

```text
User enters a question
        ↓
AgentHarness collects history
        ↓
TokenCounter counts input tokens
        ↓
Does the input fit within 7,680 tokens?
        ↓
Yes → Send to Qwen3
No  → Remove oldest conversation exchanges
        ↓
Recount and validate before sending
```



## Token-Aware Conversation Management

### Objective

Prevent the Agent Harness from sending conversations that exceed
the configured context window of the local Large Language Model (LLM).

### Implementation

File: `src/agent/harness.py`

The harness enforces two limits:

1. Conversation history: maximum six completed exchanges.
2. Input token budget: maximum 7,680 tokens.

The model's configured maximum context is 8,192 tokens, with
512 tokens reserved for the generated response.

### Execution Flow

```text
Receive user input
        |
        v
Validate nonempty input
        |
        v
Construct conversation history
        |
        v
Count formatted input tokens
        |
        v
Does the input fit?
        |
    +---+---+
    |       |
   Yes      No
    |       |
    |   Remove oldest completed exchange
    |       |
    |   Recheck token budget
    |       |
    |   Reject if no removable history remains
    |
    v
Call the model
    |
    v
Save response and updated history
```

### Key Design Decisions

**1. Preserve system instructions**

The system message is retained when older conversation exchanges
are removed.

**2. Remove complete exchanges**

The harness removes the oldest user message and its corresponding
assistant response together.

This avoids leaving incomplete conversation exchanges.

**3. Validate before model execution**

The harness checks the token budget before calling the model.

A single oversized user message is rejected if removing previous
history cannot make the request fit.

**4. Preserve state on failure**

The stored conversation is updated only after a successful
model response.

Token-budget errors and model failures do not modify the
previously saved conversation state.

### Automated Unit Tests

File: `tests/test_harness.py`

Command:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Actual results:

```text
test_conversation_history ... ok
test_empty_input ... ok
test_history_window ... ok
test_model_failure ... ok
test_oversized_user_input ... ok
test_reset ... ok
test_token_budget_removes_oldest_exchange ... ok

Ran 7 tests in 0.006s

OK
```

All seven unit tests passed.

### Test Coverage

| Test | Verified behavior |
|---|---|
| Conversation history | Previous exchanges are included |
| Empty input | Whitespace-only input is rejected |
| History window | Only six recent exchanges remain |
| Model failure | Failed calls preserve stored state |
| Oversized input | Invalid input is rejected without model execution |
| Reset | History clears while system instructions remain |
| Token-budget trimming | Oldest exchange is removed before model execution |

### Testing Method

The automated tests use simulated model and token-counter objects.

This isolates the Agent Harness logic from external dependencies.

Separately, the real Qwen3 tokenizer was verified through:

```bash
python -m scripts.verify_token_counter
```

Measured results:

```text
Normal conversation: 24 tokens
Within budget: True

Oversized conversation: 12009 tokens
Within budget: False

PASS: All token counter checks passed.
```

### Engineering Significance

Message-count limits alone cannot guarantee that a conversation
fits within a model's context window.

Token-aware validation provides a more precise mechanism for
preventing oversized requests.

Mock-based unit tests verify the application control logic without
requiring model inference.

### Current Limitations

- The context manager discards older exchanges rather than summarizing them.
- Removed exchanges cannot be recovered from the current in-memory history.
- The harness is designed for text-only conversations.
- Concurrent requests and persistent conversation storage are not supported.
- The seven unit tests do not establish end-to-end reliability with the real model server.

### Interview Discussion Points

**Why use both message-count and token-count limits?**

Message-count limits restrict the number of conversation exchanges.
Token-count limits account for the actual size of the formatted input.
Both are useful because individual messages can vary significantly
in length.

**Why remove complete user-assistant exchanges?**

Removing complete exchanges helps preserve coherent conversation
structure and avoids leaving an unmatched assistant response.

**Why use simulated components in unit tests?**

Simulated components make tests deterministic, fast, and independent
of GPU availability or model behavior.

**How is conversation state protected during failures?**

The harness constructs a pending request without immediately changing
stored history. It commits the updated history only after receiving
a successful model response.
```


## Real-Model Integration Verification

### Objective

Verify that the complete Agent Harness operates correctly with
the locally hosted Qwen3 language model.

Unlike unit testing, this verification uses the real model client,
token counter, vLLM inference server, and NVIDIA GPU.

### Components Tested

- AgentHarness: conversation state management
- TokenCounter: input token validation
- VLLMClient: Application Programming Interface (API) communication
- vLLM: Large Language Model (LLM) inference serving
- Qwen3-8B-AWQ: local language model

### Verification Script

File: `scripts/verify_agent_integration.py`

### Reproduction

Start the vLLM server:

```bash
bash scripts/serve_vllm.sh
```

In another Ubuntu terminal, activate the Python environment and
run the integration script from the project root:

```bash
python -m scripts.verify_agent_integration
```

### Actual Results

Test 1: Real model response

```text
Assistant: Acknowledged, Cypress.
PASS: Model returned a nonempty response.
```

Test 2: Multi-turn conversation history

```text
User: What is my project codename?
Assistant: Cypress
PASS: Conversation history preserved.
```

The model correctly recalled the codename from the previous exchange.

Test 3: Conversation reset

```text
PASS: Conversation reset.
```

Final verification:

```text
PASS: Agent integration checks completed.
```

All three integration checks passed.

### Engineering Findings

**1. End-to-end inference**

The application successfully sent requests through the real
model client to the locally hosted vLLM server.

**2. Conversation state**

The harness preserved previous user and assistant messages
between requests.

The model successfully retrieved information from the
conversation history.

**3. State reset**

The reset operation cleared conversation history while
preserving the system instructions.

**4. Model output formatting**

Qwen3 responses contained empty `<think></think>` tags.
The integration checks passed, but output formatting may
require additional processing.

### Testing Limitations

- The integration script asserts nonempty responses and correct
  conversation structure, but codename accuracy was inspected manually.
- The test does not establish model accuracy across different tasks.
- Automatic token-budget trimming was verified through unit tests,
  not through an oversized real-model integration request.
- Concurrent execution, persistent storage, and extended reliability
  testing remain future work.

### Interview Discussion

**How was the Agent Harness validated?**

The implementation was validated at two levels.

First, seven unit tests used simulated model and token-counter
components to verify conversation state, reset behavior,
error handling, and token-budget enforcement.

Second, an integration script exercised the actual local
Qwen3 inference stack. It verified successful model calls,
conversation-history preservation, and state reset.

This separates application logic testing from actual
model-serving integration.

