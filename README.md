# Local AI Systems Platform

A local, Graphics Processing Unit (GPU)-backed Artificial Intelligence (AI) platform for exploring Large Language Model (LLM) inference, stateful agent execution, context management, task-specific skills, and reproducible verification.

The repository contains Python components, executable verification scripts, automated tests, and technical documentation. Implemented functionality is distinguished from planned work.

## Architecture

### Inference and agent execution

```text
User
  |
  v
Command-Line Interface (CLI)
  |  src/agent/cli.py
  v
AgentHarness
  |  Conversation history and reset
  |  Maximum six completed exchanges
  |  Token-budget validation and history trimming
  |  Optional active skill instructions
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

`AgentHarness` uses the model-specific `TokenCounter` to check input size before submitting requests. It updates stored conversation history only after successful model responses.

### Skill activation and context injection

```text
skills/research_summary/SKILL.md
                |
                v
       SkillLoader.load()
                |
                v
 Skill(name, description, instructions)
                |
                v
 AgentHarness.activate_skill("research_summary")
                |
                v
 General system prompt + skill instructions
                |
                v
 Token-budget validation -> VLLMClient -> Qwen3
```

Skill activation is currently **explicit through Python code**. Activating or deactivating a skill resets conversation history; resetting a conversation alone preserves the active skill. The existing interactive CLI does **not** yet provide commands for selecting skills. Skills provide instructions; they do not execute code or grant tool permissions.

## Current Implementation

- **Local inference:** vLLM serves `Qwen/Qwen3-8B-AWQ` through an OpenAI-compatible API.
- **Model client:** Configurable endpoint, model name, request timeout, retries, and text response handling.
- **Agent Harness:** Multi-turn conversation state, system instructions, reset, input validation, and failure-safe state updates.
- **Context management:** Up to six completed exchanges, Qwen3 chat-template token counting, and removal of older exchanges when the input exceeds its budget.
- **Skill definitions:** Versioned Markdown instructions, currently including `research_summary`.
- **Skill Loader:** Metadata extraction, skill-name validation, path checks, and missing-file handling.
- **Skill integration:** Explicit activation/deactivation and task-specific instructions in the model's system message.
- **Verification:** Python unit tests, deterministic verification scripts, and real-model integration checks.

## Environment

The initial implementation was developed and verified using:

- Windows with Windows Subsystem for Linux 2 (WSL2) and Ubuntu
- Python 3.12
- NVIDIA GeForce RTX 3080 with 16 GB video memory
- NVIDIA CUDA Toolkit 13.2
- PyTorch `2.13.0+cu132`
- `Qwen/Qwen3-8B-AWQ` (Activation-aware Weight Quantization)

The Python environments, installed dependencies, and downloaded model weights are not stored in the repository.

## Repository Structure

```text
local-ai-systems-platform/
├── docs/
│   ├── architecture.md
│   ├── agent-harness.md
│   ├── local-llm-serving.md
│   ├── skills-and-context.md
│   └── troubleshooting.md
├── scripts/
│   ├── serve_vllm.sh
│   ├── verify_agent_integration.py
│   ├── verify_skill_activation.py
│   ├── verify_skill_integration.py
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
    ├── test_harness.py
    └── test_skill_activation.py
```

This tree omits Python package initialization files and repository configuration files for readability.

## Running the Platform

The following commands apply to the existing Ubuntu environment. Open Ubuntu and run:

```bash
source ~/.venv/bin/activate
cd /mnt/c/Users/Xavier/PycharmProjects/local-ai-systems-platform
```

**Terminal 1 — start the inference server:**

```bash
bash scripts/serve_vllm.sh
```

Wait until `Application startup complete` is reported. Leave the server terminal running.

**Terminal 2 — start the interactive agent:**

```bash
python -m src.agent.cli
```

Use `/reset` to clear conversation history and `/exit` to leave the session. The CLI currently supports general conversation; skill activation is available through `AgentHarness.activate_skill()` in Python and the verification scripts.

The server binds to `127.0.0.1:8000` and uses a development API key. These settings do not constitute a production security configuration.

## Reproducible Verification

Run from the project root with the Python virtual environment activated.

**Automated unit tests (no inference server required):**

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

**Tokenizer and token-budget verification (no inference server required):**

```bash
python -m scripts.verify_token_counter
```

**Skill Loader verification (no inference server required):**

```bash
python -m scripts.verify_skill_loader
```

**Skill Activation verification (no inference server required):**

```bash
python -m scripts.verify_skill_activation
```

**Agent integration verification (running inference server required):**

```bash
python -m scripts.verify_agent_integration
```

**Research Summary Skill integration verification (running inference server required):**

```bash
python -m scripts.verify_skill_integration
```

### Recorded Results

| Verification | Observed result |
|---|---|
| Automated unit tests | 12 passed in 0.120 seconds: 7 harness tests and 5 skill activation tests |
| Token-counter verification | Short input: 24 tokens; oversized input: 12,009 tokens; both checks passed |
| Skill Loader verification | 3 checks passed |
| Skill Activation verification | 6 checks passed using simulated model and token-counter components |
| Real-model agent integration | Nonempty inference response, retained conversation history, and successful reset; Qwen3 recalled the synthetic codename `Cypress` |
| Real-model skill integration | `research_summary` loaded; Qwen3 produced all five required sections for one synthetic study; reported F1 scores of 0.72 and 0.80 were preserved |

These are results from specific verification runs, not continuous tests or evidence of production readiness. Skill-format compliance was automatically checked for the single synthetic study; its numerical content and limitations were reviewed manually.

## Technical Documentation

- [System Architecture](docs/architecture.md) — components, responsibilities, and data flow.
- [Local LLM Serving](docs/local-llm-serving.md) — model serving, environment configuration, and initial inference verification.
- [Agent Harness](docs/agent-harness.md) — conversation state, token budgets, tests, and integration results.
- [Skills and Dynamic Context](docs/skills-and-context.md) — skill definitions, activation, regression tests, and real-model verification.
- [Troubleshooting](docs/troubleshooting.md) — compiler and CUDA toolkit failures and their resolutions.

## Planned Work

- Expose explicit skill selection and deactivation through the interactive CLI.
- Add more skills and evaluate instruction-following across multiple tasks and repeated trials.
- Add bounded, permission-controlled tool execution.
- Integrate Model Context Protocol (MCP) components.
- Implement security guardrails, including prompt-injection testing and trust-boundary enforcement.
- Expand evaluation, structured logging, and operational diagnostics.

Planned features are not represented as completed functionality.
