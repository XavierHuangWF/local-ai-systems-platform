# Local AI Systems Platform

A local, Graphics Processing Unit (GPU)-backed Artificial Intelligence (AI) platform for experimenting with Large Language Model (LLM) inference, stateful agents, context management, and reproducible verification.

The repository contains working Python components, executable verification scripts, and technical records of implementation decisions, test results, and debugging findings. Functionality that has not yet been implemented is identified separately.

## Architecture

### Implemented inference and agent path

```text
User
  |
  v
Command-Line Interface (CLI)
  |  src/agent/cli.py
  v
AgentHarness
  |  Conversation history and reset
  |  Six-exchange history window
  |  Token-budget validation and trimming
  v
VLLMClient
  |  src/client/vllm_client.py
  v
OpenAI-compatible Application Programming Interface (API)
  |
  v
vLLM inference server
  |
  v
Qwen3-8B-AWQ
  |
  v
NVIDIA CUDA / GPU
```

`AgentHarness` uses `TokenCounter` (`src/agent/token_counter.py`) to check the model-specific input token budget before submitting requests. It preserves system instructions and updates saved conversation state only after successful model responses.

### Independently verified skill-loading component

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

The skill loader currently reads and validates local skill definitions. **Skill activation within `AgentHarness` has not yet been implemented.**

## Current Implementation

- **Local inference:** vLLM serving `Qwen/Qwen3-8B-AWQ` through an OpenAI-compatible API.
- **Model client:** Configurable endpoint, model name, timeout, retries, and text response handling.
- **Agent Harness:** Multi-turn conversation state, persistent system instructions within each session, `/reset`, and error handling.
- **Context management:** Maximum of six completed conversation exchanges, model-specific token counting, and removal of older exchanges when input exceeds the budget.
- **Local skill loader:** Markdown skill definitions with validated names, metadata extraction, and explicit handling of missing skills.
- **Verification:** Automated harness unit tests, token-counter checks, skill-loader checks, and an integration script exercising real local model inference.

## Environment

The initial implementation was developed and verified using:

- Windows with Windows Subsystem for Linux 2 (WSL2), Ubuntu
- Python 3.12
- NVIDIA GeForce RTX 3080 with 16 GB video memory
- NVIDIA CUDA Toolkit 13.2
- PyTorch `2.13.0+cu132`
- `Qwen/Qwen3-8B-AWQ` (Activation-aware Weight Quantization)

The Python packages and downloaded model are local environment dependencies; they are not committed to the repository.

## Repository Structure

```text
local-ai-systems-platform/
├── docs/
│   ├── agent-harness.md
│   ├── local-llm-serving.md
│   ├── skills-and-context.md
│   └── troubleshooting.md
├── scripts/
│   ├── serve_vllm.sh
│   ├── verify_agent_integration.py
│   ├── verify_skill_loader.py
│   └── verify_token_counter.py
├── skills/
│   └── research_summary/
│       └── SKILL.md
├── src/
│   ├── agent/
│   │   ├── cli.py
│   │   ├── harness.py
│   │   └── token_counter.py
│   ├── client/
│   │   └── vllm_client.py
│   └── skills/
│       └── skill_loader.py
└── tests/
    └── test_harness.py
```

This tree shows the principal project files; Python package initialization files and repository configuration files are omitted for readability.

## Running the Platform

The following commands are for the existing Ubuntu environment. Start from the repository root and activate the installed Python environment:

```bash
source ~/.venv/bin/activate
cd /mnt/c/Users/Xavier/PycharmProjects/local-ai-systems-platform
```

**Terminal 1 — start the local inference server:**

```bash
bash scripts/serve_vllm.sh
```

Wait for the server's `Application startup complete` message and leave that terminal running.

**Terminal 2 — start the interactive agent:**

```bash
python -m src.agent.cli
```

Use `/reset` to clear conversation history and `/exit` to leave the interactive session.

The startup script is configured for local development (`127.0.0.1:8000`) and uses a development API key. These settings are not a production security configuration.

## Reproducible Verification

Run the following commands from the repository root with the Python environment activated:

**Agent Harness unit tests (no inference server required):**

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

**Qwen3 tokenizer and context-budget verification (no inference server required):**

```bash
python -m scripts.verify_token_counter
```

**Skill-definition parsing and validation (no inference server required):**

```bash
python -m scripts.verify_skill_loader
```

**Real-model integration verification (inference server required):**

```bash
python -m scripts.verify_agent_integration
```

### Recorded Results

| Verification | Observed result |
|---|---|
| Agent Harness unit tests | 7 passed |
| Token-counter check | Short input: 24 tokens; oversized input: 12,009 tokens; both checks passed |
| Skill Loader check | 3 checks passed |
| Real-model integration | Nonempty response, conversation history, and reset checks passed; model recalled the synthetic codename `Cypress` |

These results describe individual verification runs, not continuous testing or a production reliability guarantee. See the linked technical documents for methodology, failure analysis, and limitations.

## Technical Documentation

- [Local LLM Serving](docs/local-llm-serving.md) — model serving, environment configuration, and initial inference verification.
- [Agent Harness](docs/agent-harness.md) — conversation state, token budgets, automated tests, and integration results.
- [Skills and Dynamic Context](docs/skills-and-context.md) — skill file format, loader design, validation, and observed results.
- [Troubleshooting](docs/troubleshooting.md) — compiler and CUDA toolkit setup failures and their resolutions.

## Planned Work

- Integrate explicit skill selection and activation into `AgentHarness`.
- Expand skill coverage and add automated tests for skill behavior.
- Add bounded, permission-controlled tool execution.
- Integrate Model Context Protocol (MCP) components.
- Implement security guardrails, including prompt-injection testing and trust-boundary enforcement.
- Expand evaluation, logging, and operational diagnostics.

The planned features are **not** represented as completed functionality.
