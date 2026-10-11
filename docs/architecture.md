# System Architecture

## Overview

Local AI Systems Platform is a Python application for experimenting with local large language model (LLM) serving, stateful agent execution, context budgeting, and modular task instructions. The application runs model inference through a separate vLLM server and keeps model-serving concerns distinct from conversation-management logic.

This document distinguishes implemented functionality from components that are still planned.

## Implemented Architecture

```text
User
  |
  v
Command-Line Interface (CLI)
  src/agent/cli.py
  |
  v
AgentHarness
  src/agent/harness.py
  |                    |
  |                    +--> TokenCounter
  |                         src/agent/token_counter.py
  |
  v
VLLMClient
  src/client/vllm_client.py
  |
  v
OpenAI-compatible Application Programming Interface (API)
  |
  v
vLLM inference server (Ubuntu through Windows Subsystem for Linux)
  |
  v
Qwen3-8B-AWQ / PyTorch / CUDA / NVIDIA RTX 3080
```

The skill-loading module currently exists separately from this execution path:

```text
skills/research_summary/SKILL.md
               |
               v
SkillLoader
src/skills/skill_loader.py
               |
               v
Skill(name, description, instructions)
```

Skill instructions are **not yet injected into AgentHarness requests**. Defining and validating a skill does not execute it or grant access to tools.

## Component Responsibilities

| Component | Source | Responsibility |
| --- | --- | --- |
| Command-Line Interface | `src/agent/cli.py` | Read user input, display responses, support `/reset` and `/exit` |
| Agent Harness | `src/agent/harness.py` | Manage system instructions, conversation state, context trimming, and model calls |
| Model client | `src/client/vllm_client.py` | Configure and send chat-completion requests to the local API endpoint |
| Token counter | `src/agent/token_counter.py` | Count messages using the Qwen3 tokenizer and chat template; validate the input budget |
| Inference server | `scripts/serve_vllm.sh` | Start vLLM with the configured model, server address, and context limits |
| Skill definition | `skills/research_summary/SKILL.md` | Store task-specific research-summary instructions and output constraints |
| Skill loader | `src/skills/skill_loader.py` | Locate, validate, parse, and return a structured skill definition |

## Conversation Request Flow

1. The CLI receives a user message and forwards it to `AgentHarness.run()`.
2. The harness validates that the message is not empty.
3. The harness constructs a pending request with its system message, recent completed exchanges, and the new user message.
4. `TokenCounter` counts the formatted input. If it exceeds the budget, the harness removes the oldest complete user-assistant exchange and checks again.
5. If the new message cannot fit even after removable history is exhausted, the harness raises an error without calling the model.
6. If it fits, `VLLMClient` sends the request to the vLLM server, which generates a response using Qwen3.
7. The harness updates stored history only after a successful response, and the CLI displays it.

The current configuration retains at most six completed user-assistant exchanges, with an 8,192-token maximum sequence length and 512 tokens reserved for output. The resulting input budget is 7,680 tokens. This budget applies to formatted text requests, including chat-template tokens.

Conversation history is held in process memory; it is not persisted between application restarts. Removed exchanges are discarded, not summarized.

## Development and Runtime Environments

- **Development:** Windows, PyCharm, Git repository.
- **Runtime:** Ubuntu in Windows Subsystem for Linux (WSL), Python 3.12 virtual environment, PyTorch, CUDA Toolkit 13.2, and vLLM.
- **Hardware used during verification:** NVIDIA GeForce RTX 3080 with 16 GB video memory.
- **Model:** `Qwen/Qwen3-8B-AWQ`, exposed to clients as `qwen3-8b`.
- **Local endpoint:** `http://127.0.0.1:8000/v1`.

The API key in the current development configuration is a local test value, not a production credential. Binding to `127.0.0.1` limits the server listener to the loopback interface of its runtime environment; deployment outside this local setup requires separate networking and authentication review.

## Reproducing the Verified Workflows

Run commands from the project root. Activate the existing Ubuntu Python virtual environment before executing Python scripts.

Start model serving in one terminal:

```bash
source ~/.venv/bin/activate
bash scripts/serve_vllm.sh
```

Execute the interactive agent in another terminal:

```bash
source ~/.venv/bin/activate
python -m src.agent.cli
```

Run the unit tests without a running inference server:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run the tokenizer verification without a running inference server:

```bash
python -m scripts.verify_token_counter
```

Run the real-model integration verification while vLLM is running:

```bash
python -m scripts.verify_agent_integration
```

Run the skill-loader verification without a running inference server:

```bash
python -m scripts.verify_skill_loader
```

## Recorded Verification

| Verification | Observed result | Scope |
| --- | --- | --- |
| Agent harness unit tests | Seven tests passed | Simulated model and token counter; state and control logic |
| Token counter | 24 tokens for a short conversation; 12,009 for an oversized conversation | Real Qwen3 tokenizer and chat template |
| Agent integration | Model generated `Acknowledged, Cypress.`; recalled `Cypress` on follow-up; reset check passed | Real vLLM and Qwen3 inference; recall was manually inspected |
| Skill loader | Three checks passed; research-summary instructions loaded | File parsing, invalid skill name rejection, missing skill handling |

These results demonstrate the specific verified behaviors; they do not establish comprehensive model accuracy, reliability under concurrent load, or resistance to malicious instructions.

## Boundaries and Planned Extensions

Currently implemented: model serving, a reusable model client, stateful agent loop, token-aware conversation trimming, a standalone skill loader, and basic unit and integration verification.

Not yet implemented: skill activation in the harness, external tool execution, Model Context Protocol (MCP) connections, permission enforcement, prompt-injection defenses, persistent state, concurrent-session management, and broader systematic evaluations.

The skill loader is intended to become an instruction source for the harness, but task instructions will need to remain distinct from trusted system rules and tool permissions when that integration is built.

## Related Documentation

- [`local-llm-serving.md`](local-llm-serving.md): inference setup, configuration, and performance observations.
- [`agent-harness.md`](agent-harness.md): state management, token budgets, tests, and integration results.
- [`skills-and-context.md`](skills-and-context.md): skill definition, loader, and verification.
- [`troubleshooting.md`](troubleshooting.md): compiler failures and remedies.
