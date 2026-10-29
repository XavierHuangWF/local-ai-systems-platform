# System Architecture

## Overview

Local AI Systems Platform is a Python application for experimenting with local large language model (LLM) serving, stateful agent execution, token-aware conversation management, reusable task instructions, validated Python tool calls, and retrieval-augmented generation (RAG). Model inference runs in a separate vLLM server. The application keeps model serving, agent orchestration, tool execution, document retrieval, and answer presentation in distinct components.

This document distinguishes **implemented and observed behavior** from limitations and proposed extensions. RAG and tool execution are available as **separate, explicitly selected modes** in the integrated command-line agent; simultaneous execution of both modes is not yet supported.

## Implemented Architecture

```text
User
  |
  v
Interactive Command-Line Interface (CLI)
  src/agent/cli.py
  |   /rag on|off|status     /tools on|off|status     /reset
  v
AgentHarness
  src/agent/harness.py
  |-- System instructions + optional active skill
  |     skills/research_summary/SKILL.md
  |     src/skills/skill_loader.py
  |
  |-- Conversation history + TokenCounter
  |     src/agent/token_counter.py
  |
  |-- Normal chat (default) -----------------------------+
  |                                                       |
  |-- RAG mode (optional)                                 |
  |     HybridRetriever                                   |
  |       src/rag/document_loader.py                      |
  |       src/rag/chunker.py                              |
  |       src/rag/bm25.py + src/rag/embeddings.py          |
  |       src/rag/fusion.py                               |
  |     -> top 6 evidence chunks appended to the          |
  |        temporary model request                        |
  |                                                       |
  |-- Tool mode (optional; mutually exclusive with RAG)   |
  |     src/agent/tool_loop.py                            |
  |     -> ToolExecutor -> ToolArgumentValidator           |
  |     -> ToolRegistry -> registered Python handler      |
  |     -> tool result -> final model response            |
  |                                                       |
  +---------------------> VLLMClient <--------------------+
                           src/client/vllm_client.py
                                   |
                                   v
                      OpenAI-compatible local API
                                   |
                                   v
                      vLLM inference server (WSL Ubuntu)
                                   |
                                   v
                    Qwen3-8B-AWQ / PyTorch / CUDA / RTX 3080
```

A skill is an instruction source, not an executable tool. `SkillLoader` reads and validates a Markdown skill; `AgentHarness.activate_skill(name)` places its instructions in the system-message content and resets conversation history. `deactivate_skill()` returns to the general system instructions. Skill activation exists through the Python API; the current integrated CLI does **not** expose skill activation commands.

The integrated CLI supports `/rag on`, `/rag off`, `/rag status`, `/tools on`, `/tools off`, `/tools status`, `/reset`, `/help`, and `/exit`. Enabling RAG initializes the document retriever lazily, then reuses its in-memory document embeddings for subsequent questions. The separate `src/rag/cli.py` and `src/rag/qa.py` also provide a standalone RAG interface.

## Component Responsibilities

| Component | Source | Responsibility |
| --- | --- | --- |
| Integrated CLI | `src/agent/cli.py` | Accept questions, switch RAG/tool modes, reset history, display generated answers, tool logs, and retrieved-source mappings |
| Agent Harness | `src/agent/harness.py` | Orchestrate modes, system/skill instructions, stateful turns, temporary RAG evidence, token-budget checks, and model requests |
| Model client | `src/client/vllm_client.py` | Issue local chat-completion requests; support a configurable response-token limit (512 by default) |
| Token counter | `src/agent/token_counter.py` | Count Qwen3 chat-template tokens and check input budgets against a configurable output-token reservation |
| Response formatter | `src/agent/response_formatter.py` | Remove complete leading Qwen3 thinking blocks and reject malformed or empty cleaned responses |
| Inference server | `scripts/serve_vllm.sh` | Start the configured local vLLM model and OpenAI-compatible endpoint |
| Skill definition | `skills/research_summary/SKILL.md` | Store task-specific instructions and output constraints |
| Skill loader | `src/skills/skill_loader.py` | Read, validate, and parse skill definitions |
| Tool definitions | `src/tools/definitions.py` | Describe registered tool names and argument schemas |
| Tool validator | `src/tools/validator.py` | Validate model-generated tool arguments with `jsonschema` Draft 2020-12 |
| Tool registry | `src/tools/registry.py` | Manage approved callable names and dispatch registered functions; **does not itself enforce JSON Schema** |
| Tool executor | `src/tools/executor.py` | Validate before dispatch, execute registered handlers, and log successful tool execution |
| Tool loop | `src/agent/tool_loop.py` | Handle model tool requests, validated execution, tool-result messages, and the final model reply |
| Example Python tool | `src/tools/statistics_tool.py` | Calculate count, mean, minimum, and maximum, with additional domain-specific input checks |
| Document loader / chunker | `src/rag/document_loader.py`, `src/rag/chunker.py` | Read project Markdown documents and construct token-bounded, source-labeled chunks |
| Keyword / semantic retrieval | `src/rag/bm25.py`, `src/rag/embeddings.py` | Rank chunks by BM25 keyword matching and normalized local embedding similarity |
| Rank fusion / retriever | `src/rag/fusion.py`, `src/rag/retriever.py` | Fuse keyword and semantic rankings with reciprocal rank fusion (RRF); return top-ranked chunks |
| Standalone RAG | `src/rag/qa.py`, `src/rag/cli.py` | Retrieve passages, ask Qwen3 for a grounded answer, and display corresponding sources |
| Experimental reranker | `src/rag/reranker.py` | Score retrieved candidates using a local cross-encoder for comparisons; **not the current integrated RAG default** |

## Conversation Request Flow

1. The integrated CLI forwards an ordinary user message to `AgentHarness.run()`; commands such as `/rag on` and `/tools on` control its operating mode.
2. The harness rejects empty input, selects recent completed exchanges, and constructs a pending conversation using the original user question. It preserves system instructions and optional activated skill instructions.
3. **When RAG is enabled**, `HybridRetriever` searches the local documentation and selects the top six chunks. The harness adds numbered passages and grounding instructions to a **temporary** model request. Retrieved evidence is not saved as part of conversation history.
4. **When tool mode is enabled instead**, the harness uses the registered-tool execution path. `ToolExecutor` validates model-generated arguments before dispatch through `ToolRegistry`. The Python handler can apply additional validation. Tool execution and RAG are currently mutually exclusive.
5. `TokenCounter` measures the actual formatted request. If it is too long, the harness removes the oldest completed exchange from both the request and pending-history representation and checks again. If the request cannot fit without history, it raises an error before sending the model call.
6. For ordinary chat or RAG, `VLLMClient` sends the request to vLLM; the tool path uses its existing model/tool round trip. Qwen3 produces an answer, which is processed by `clean_model_response()`.
7. Only after successful generation does the harness commit the **original question** and cleaned answer to history. For a successful RAG answer it also retains the latest retrieved chunks separately in `last_rag_sources`, allowing the CLI to display the numbered source mapping. `/reset` clears the conversation and that source mapping while retaining the active skill and current mode settings.

### Context and output budgets

| Mode | Maximum context | Reserved output | Maximum formatted input | Generation setting |
| --- | ---: | ---: | ---: | --- |
| Normal chat | 8,192 | 512 | 7,680 | `max_tokens=512` (default) |
| Integrated RAG | 8,192 | 1,024 | 7,168 | `max_tokens=1024` |
| Tool mode | 8,192 | 512 in the initial input check | 7,680 in that check | Tool-specific round-trip path; see limitation below |

These are context-budget limits, not target output lengths. The existing tool path does **not fully account for tool definitions and intermediate tool-result messages**, so the listed initial token check does not guarantee every tool-call exchange will fit. Standalone RAG also requests 1,024 generation tokens but uses its own request flow rather than the integrated harness's budget-trimming path.

The harness retains at most six completed user–assistant exchanges in process memory. Older exchanges are discarded, not summarized; history is not persisted across restarts. RAG retrieval currently uses the latest question as the search query, rather than rewriting follow-up questions from the full conversation context.

## Development and Runtime Environments

- **Development:** Windows, PyCharm, Git repository.
- **Runtime:** Ubuntu in Windows Subsystem for Linux (WSL), Python 3.12 virtual environment, PyTorch, CUDA Toolkit 13.2, and vLLM.
- **Hardware used during verification:** NVIDIA GeForce RTX 3080 with 16 GB video memory.
- **Model:** `Qwen/Qwen3-8B-AWQ`, exposed to clients as `qwen3-8b`.
- **Local endpoint:** `http://127.0.0.1:8000/v1`.
- **Retrieval:** Local CPU embedding model `BAAI/bge-small-en-v1.5` (384 dimensions), an in-project BM25 implementation, RRF fusion (`k=60`), and an optional `cross-encoder/ms-marco-MiniLM-L6-v2` for evaluation experiments.

The API key in the current development configuration is a local test value, not a production credential. Binding to `127.0.0.1` limits the server listener to the loopback interface of its runtime environment; deployment outside this local setup requires separate networking and authentication review.

## Reproducing the Verified Workflows

Run commands from the project root. Activate the existing Ubuntu Python virtual environment before executing Python scripts.

Start model serving in one terminal:

```bash
source ~/.venv/bin/activate
bash scripts/serve_vllm.sh
```

Execute the integrated interactive agent in another terminal:

```bash
source ~/.venv/bin/activate
python -m src.agent.cli
```

For an integrated RAG question, type `/rag on`, then a question; the answer is followed by numbered retrieved-source mappings. To switch to validated Python tools, type `/rag off`, then `/tools on`. Tool execution can be verified with the request to summarize `[12, 24, 36, 48]`.

Execute the standalone RAG interface (while vLLM is running):

```bash
python -m src.rag.cli
```

Run the recorded live RAG demonstration (while vLLM is running):

```bash
python -m scripts.verify_live_rag
```

Run the retrieval experiments and judged-set evaluation (no vLLM server required):

```bash
python -m scripts.diagnose_retrieval
python -m scripts.evaluate_retrieval
```

Run the existing unit-test suite without the inference server:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run tokenizer, Agent Harness, and skill-loader verification:

```bash
python -m scripts.verify_token_counter
python -m scripts.verify_skill_loader
# Requires running vLLM:
python -m scripts.verify_agent_integration
```

## Recorded Verification

| Verification | Observed result | Scope |
| --- | --- | --- |
| Original Agent Harness tests | Seven tests passed at the early harness milestone | Simulated client/counter; state and control logic |
| Token counter | 24 tokens for a short conversation; 12,009 for an oversized conversation | Real Qwen3 tokenizer and chat template |
| Original Agent Harness integration | Qwen3 returned `Acknowledged, Cypress.`, recalled `Cypress`, and passed the reset check | Real local inference; recall inspected manually |
| Skill loader / activation | Three skill-loader checks and six skill-activation checks passed; real-model skill instructions produced the required structured sections in one experiment | File handling, state transitions, and one real-model example |
| Registered Python tool | `summarize_measurements([12, 24, 36, 48])` executed through the integrated agent; count **4**, mean **30.0**, minimum **12.0**, maximum **48.0** | Live tool-mode test after RAG integration; log confirmed Python handler execution |
| Hybrid retrieval benchmark (before this documentation revision) | Six Markdown documents yielded **101 chunks**; five test questions and **33** manually graded question–passage pairs | Small, nonexhaustive, question-specific judged set |
| Retrieval ranking evaluation (before this revision) | Mean judged-set nDCG@5: **0.838** for hybrid RRF, **0.854** for cross-encoder; mean Recall@5: **0.780** vs **0.893**; mean MRR@5: **1.000** vs **0.900** | Cross-encoder improved some questions but degraded others; differences do not establish a general winner |
| Live standalone RAG | Correctly described the FlashInfer `nvcc` failure and CUDA Toolkit 13.2 resolution, with source citations | One end-to-end question; citations manually checked against the retrieved passages |
| Integrated RAG | `/rag on`, `/reset`, grounded tool-validation answer, and six-source citation display succeeded | One demonstrated end-to-end flow; RAG evidence kept separate from saved conversation turns |
| Mode switching | `/rag off` followed by `/tools on` executed the registered statistics tool correctly, without stale RAG source display | One demonstrated integrated session |

**Reproducibility note:** Updating this document changes text used by the RAG corpus. Chunk boundaries, sequential chunk IDs, embeddings, and ranking results may change. Existing `data/relevance_judgments.json` labels—especially those assigned to outdated `architecture.md` passages—and the 101-chunk benchmark should be treated as a **historical baseline**, not automatically reused as current ground truth. Rebuild the corpus, inspect newly assigned chunk IDs, and refresh relevance labels before claiming new comparable evaluation scores.

The observed workflows do not establish general answer correctness, citation entailment, robustness under concurrent load, or resistance to prompt injection. Tests of prior milestones do not substitute for rerunning checks after changes to the integrated RAG and harness code.

## Boundaries and Planned Extensions

**Implemented:** Local Qwen3 serving; reusable API client; stateful Agent Harness with token-aware history trimming; explicit skill activation through Python; registered, schema-validated Python tool execution; heading-aware Markdown chunking; BM25 and local semantic retrieval; RRF fusion; experimental cross-encoder reranking; standalone RAG; and integrated RAG mode with numbered source display.

**Current limitations:** RAG and tools cannot be enabled simultaneously; only registered Python handlers can execute; the tool path's initial token budget omits some tool-call overhead; RAG search does not rewrite conversational follow-up queries; retrieved citation markers are displayed but not automatically checked for factual support; embedding indexes are in memory and rebuilt on startup; and no general-purpose permission framework, prompt-injection defense, or sandbox isolation is in place. Retrieved text is explicitly treated as untrusted data in the prompt, but instructions alone do not guarantee protection against malicious documents.

**Not yet implemented:** Model Context Protocol (MCP) connections, general external integrations, persistent conversation state and retrieval index, multi-session concurrency management, dynamic retrieval-depth/token-budget optimization, production-grade authorization, or broad systematic end-to-end answer-quality evaluation. The cross-encoder is available for experiments but is not used by default in integrated RAG.

## Related Documentation

- [`local-llm-serving.md`](local-llm-serving.md): inference setup, configuration, and performance observations.
- [`agent-harness.md`](agent-harness.md): state management, token budgets, tests, and integration results.
- [`skills-and-context.md`](skills-and-context.md): skill definition, loader, activation, and verification.
- [`tool-execution.md`](tool-execution.md): tool registration, schema validation, execution, and live observations.
- [`troubleshooting.md`](troubleshooting.md): compiler failures and remedies.
