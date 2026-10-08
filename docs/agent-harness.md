
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
