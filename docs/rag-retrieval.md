# Retrieval-Augmented Generation (RAG) and Retrieval Evaluation

## Objective and implementation status

The Local AI Systems Platform supports local document-grounded question answering using Qwen3-8B-AWQ served by vLLM. A hybrid retriever selects evidence from Markdown documentation; the model receives the question and numbered evidence passages, then generates an answer with inline citations. The same functionality is available through the standalone RAG CLI and an optional RAG mode in the existing Agent Harness.

**Observed status:** A live FlashInfer/CUDA troubleshooting question was answered with appropriate citations to the diagnosis and resolution. The integrated Agent Harness also answered a tool-validation question using retrieved evidence and displayed the numbered source mapping. Switching from `/rag off` to `/tools on` was verified with an actual `summarize_measurements` tool call.

These are specific end-to-end observations, not proof of reliable answer grounding for every possible question.

## Components and responsibilities

| Component | File | Responsibility |
| --- | --- | --- |
| Document loader | `src/rag/document_loader.py` | Read nonempty Markdown files recursively from `docs/`, retain relative source paths, and reject oversized files |
| Chunker | `src/rag/chunker.py` | Split documents using section headings and paragraph/code-block boundaries; maintain source, heading, and chunk identifiers |
| Local embeddings | `src/rag/embeddings.py` | Embed document chunks and user queries with `BAAI/bge-small-en-v1.5` |
| Keyword ranking | `src/rag/bm25.py` | Rank passages by lexical evidence using a locally implemented BM25 scorer |
| Rank fusion | `src/rag/fusion.py` | Merge BM25 and semantic rankings using Reciprocal Rank Fusion (RRF) |
| Optional reranker | `src/rag/reranker.py` | Rescore retrieved question–passage pairs using `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Hybrid retriever | `src/rag/retriever.py` | Load and embed the corpus once, retrieve keyword and semantic candidates, and fuse ranked results |
| Standalone RAG generation | `src/rag/qa.py` | Package numbered evidence, call Qwen3, clean the response, and return the answer plus source chunks |
| Standalone RAG CLI | `src/rag/cli.py` | Ask independent questions while reusing the loaded retrieval index |
| Integrated Agent Harness | `src/agent/harness.py` | Enable RAG optionally, preserve conversation state, enforce a token budget, and record the source mapping of a successful RAG answer |
| Integrated agent CLI | `src/agent/cli.py` | Provide `/rag on`, `/rag off`, `/rag status`, and display retrieved source references |
| Retrieval judgments | `data/relevance_judgments.json` | Record question-specific, manually assigned relevance grades |

### Retrieval and generation flow

```text
User question
    |
    v
HybridRetriever
    |-- BM25 lexical ranking -------------------+
    |                                           |
    +-- BGE embedding cosine similarity -------+
                                                |
                                                v
                                   Reciprocal Rank Fusion (RRF)
                                                |
                                                v
                                     Top-ranked evidence chunks
                                                |
                                                v
                         Qwen3 question + numbered passages + rules
                                                |
                                                v
                          Cleaned answer with [n] citations
                                                |
                                                v
                              Display source-to-citation mapping
```

The default integrated RAG experiment supplies **six** evidence chunks per question. It does **not** place all documents into the model's context. The chunk selection is based on fused keyword and semantic rankings, not solely the six highest embedding similarity scores.

## Chunking and embedding baseline

The initial experiment loaded **six** Markdown documents and produced **101** chunks. Using the Qwen3 tokenizer, the recorded chunk-token statistics were: minimum **31**, median **115**, maximum **348**, with **zero** oversized chunks. The chunker was configured for a maximum of **400 tokens** and an overlap target of **60 tokens**; natural section and paragraph boundaries often yielded smaller chunks.

Document embeddings were generated locally with `BAAI/bge-small-en-v1.5` (384-dimensional vectors) and normalized before cosine-similarity ranking. The initial experiments used CPU inference for the embedder and cross-encoder, leaving the GPU available for vLLM model serving.

**Snapshot warning:** The six-document/101-chunk measurements describe the corpus *before* the architecture document was revised and this documentation was added. The loader reads all Markdown files under `docs/`, so corpus size, chunk IDs, rankings, and relevance judgments may change after documentation edits. Do not represent historical measurements as a fresh measurement of the revised corpus.

## Retrieval strategy and experiments

The initial comparison retrieved the top **20** BM25 candidates and the top **20** embedding candidates per question. RRF combined these lists using the reciprocal-rank parameter `k=60`. The optional cross-encoder then reranked up to 20 fused candidates. The integrated agent uses hybrid RRF without the cross-encoder by default.

Five questions covered conversation context, skill activation, FlashInfer CUDA compilation, tool-argument validation, and Qwen3 thinking-tag removal. Manual assessment assigned relevance grades **0–3**, treating grades **2–3** as relevant for Recall@5 and reciprocal-rank calculations. A total of **33 question–passage judgments** covered the union of the two methods' top-five results for those five questions.

### Recorded judged-set scores

| Question | Hybrid Recall@5 | Cross-encoder Recall@5 | Hybrid nDCG@5 | Cross-encoder nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| Conversation history and context | 0.833 | 0.667 | 0.885 | 0.909 |
| Skill activation | 0.400 | 0.800 | 0.622 | 0.850 |
| FlashInfer CUDA compilation | 1.000 | 1.000 | 0.936 | 0.847 |
| Tool argument validation | 1.000 | 1.000 | 1.000 | 0.994 |
| Qwen3 thinking-tag removal | 0.667 | 1.000 | 0.745 | 0.672 |
| **Macro-average (five questions)** | **0.780** | **0.893** | **0.838** | **0.854** |

Mean Reciprocal Rank at five (MRR@5) averaged **1.000** for hybrid RRF and **0.900** for cross-encoder reranking. Both methods retrieved **2/2** designated essential passages for the FlashInfer diagnosis-and-resolution question at depth five.

**Interpretation:** Cross-encoder reranking improved the average measured Recall@5 and nDCG@5 in this small sample, especially for skill activation. Hybrid RRF had better average MRR@5 and preserved stronger ordering for some questions, including the Qwen3 response-formatting question. The cross-encoder's macro nDCG advantage was approximately **0.017**, too small and sample-dependent to justify treating it as a general winner. The simpler hybrid pipeline was retained as the initial integrated baseline.

**Evaluation limitations:** These are *judged-set* metrics, not exhaustive full-corpus metrics. No unjudged passage was silently assigned grade zero. The questions and judgments were selected for exploratory engineering analysis, not drawn from a statistically representative benchmark; judgments may also be affected by the documentation revisions noted above. No retrieval-latency comparison was recorded.

### Query wording sensitivity

Two nearly equivalent tool-validation questions produced different ranks for the specific implementation chunk `tool-execution.md::0004`:

| Question wording | Target rank | RRF score |
| --- | ---: | ---: |
| “How does **ToolExecutor** validate tool arguments before executing registered Python functions?” | 1 | 0.03178 |
| “How does the **Agent Harness** validate tool arguments before executing Python functions?” | 6 | 0.03037 |

At depth five, the relevant implementation evidence was excluded for the second wording. Increasing the standalone and later integrated evidence depth to six included it. Qwen3 subsequently explained that `ToolArgumentValidator` performs JSON Schema validation, `ToolExecutor` validates before dispatch, and `ToolRegistry` manages approved functions without independently enforcing JSON Schema.

This demonstrates a candidate-cutoff effect and query sensitivity. Six is an experimentally useful value here, **not** a generally optimal retrieval depth.

## Live RAG and integrated agent observations

### Standalone RAG experiment

With vLLM serving `Qwen/Qwen3-8B-AWQ`, the question “What caused the FlashInfer CUDA compilation problem, and how was it resolved?” generated an answer that described FlashInfer just-in-time CUDA kernel compilation, the missing `nvcc` compiler, and installation of CUDA Toolkit **13.2** inside WSL2 Ubuntu. The answer cited retrieved diagnosis, error, engineering explanation, and resolution passages. The model's returned text had no visible leading Qwen3 thinking tags.

### Integrated Agent Harness

The existing interactive agent supports the following commands:

| Command | Behavior |
| --- | --- |
| `/rag on` | Lazily initialize and enable the hybrid retriever |
| `/rag off` | Disable RAG while retaining the initialized retriever for reuse |
| `/rag status` | Show whether RAG is enabled |
| `/tools on`, `/tools off`, `/tools status` | Control the existing validated Python tool mode |
| `/reset` | Clear conversation history and the last RAG source mapping without disabling RAG or removing the active skill |

RAG and tool calling are **mutually exclusive** in the current integrated implementation. The RAG path builds a temporary model request containing numbered excerpts; permanent conversation state stores the original user question and cleaned answer, not the retrieved passages. A successful RAG answer updates `last_rag_sources`, which the CLI displays alongside citation numbers. Normal chat and tool execution do not display stale RAG sources.

For the tested tool-validation question, the integrated agent successfully used evidence at rank six (`tool-execution.md::0004`) and displayed six source entries. After `/rag off` and `/tools on`, a real `summarize_measurements` invocation for `[12, 24, 36, 48]` produced count **4**, mean **30.0**, minimum **12.0**, and maximum **48.0**, with a successful tool-execution log. This verifies the observed mode transition; it does not validate every possible tool or state transition.

### Context budgeting

vLLM was configured for a maximum model context length of **8,192 tokens**. Normal chat reserves **512** output tokens, allowing up to **7,680** formatted input tokens. Integrated RAG reserves **1,024** output tokens, allowing up to **7,168** formatted input tokens. The harness counts the temporary RAG request, including evidence, and trims older completed exchanges before model invocation if needed. The standalone RAG implementation separately requests up to 1,024 generated tokens.

Earlier RAG answers were observed stopping mid-sentence at the 512-token generation limit. After the integrated RAG output allowance increased to 1,024, a tested answer completed. The response API's `finish_reason` was not independently inspected, so the earlier truncation cause remains a plausible diagnosis rather than a conclusively verified one.

## Reproduce the workflows

Run all commands from the project root with the existing Ubuntu virtual environment activated.

### Start local model serving

```bash
source ~/.venv/bin/activate
bash scripts/serve_vllm.sh
```

Keep this server running in one terminal. In a second terminal, run:

```bash
source ~/.venv/bin/activate
python -m src.agent.cli
```

For the integrated agent, enter `/rag on`, then ask a question about the documentation. The answer is followed by `=== Retrieved Sources ===` and numbered source references. To verify the separate tool mode, enter `/rag off`, `/tools on`, and a numerical request for the registered `summarize_measurements` tool.

### Run standalone RAG and retrieval experiments

```bash
# Requires the live vLLM server:
python -m scripts.verify_live_rag
python -m src.rag.cli

# Retrieval and evaluation; do not require vLLM:
python -m scripts.inspect_documents
python -m scripts.inspect_chunks
python -m scripts.compare_hybrid_retrieval
python -m scripts.compare_reranking
python -m scripts.explore_retrieval_questions
python -m scripts.evaluate_retrieval
python -m scripts.diagnose_retrieval
```

`data/relevance_judgments.json` contains manually assigned labels tied to previously observed chunk identifiers. **Before rerunning evaluation against edited documentation, refresh the corpus inspection and relevance judgments**; otherwise a score comparison with the baseline may be invalid.

## Known limitations and next work

1. **Small evaluation set:** Five questions and partially judged candidate pools are insufficient to establish general retrieval and answer quality.
2. **Corpus drift:** Editing or adding Markdown documents changes the retrieval corpus and can invalidate stored chunk identifiers and previous relevance judgments.
3. **Question independence:** Retrieval uses the current question rather than rewriting follow-up questions using conversation context. Prior turns are sent to the model, but follow-up retrieval may miss referents.
4. **Citation validity:** Numbered citation mapping is displayed, but not automatically checked for entailment or support of individual factual claims.
5. **Prompt-injection risk:** Retrieved files are labeled untrusted in instructions, but a prompt is not a comprehensive security boundary.
6. **Context limitations:** Six passages are a fixed test configuration, not a dynamic token-aware selection policy. Separate tool-calling token overhead is not fully accounted for by the simple chat budget.
7. **Mode separation:** Integrated RAG and registered Python tool calling cannot operate simultaneously. The application does not support general agentic tool planning over retrieved documents.
8. **Runtime and persistence:** Retrieval embeddings are rebuilt at process startup. Conversation state is in memory, not persisted between sessions. Concurrent operation, authorization, and production deployment hardening have not been evaluated.

The next engineering priorities are to verify the integrated implementation after documentation changes, refresh or version evaluation judgments, and compare answer correctness and citation grounding on a larger set of questions before changing retrieval models.


## Updated Corpus Inspection — October 28, 2026

After updating `docs/architecture.md` and adding this
RAG documentation, the chunking inspection was rerun.

**Command:**

```bash
python -m scripts.inspect_chunks
```

**Observed results:**

- Documents: 7
- Chunks: 119
- Target chunk size: 400 tokens
- Token overlap: 60
- Minimum chunk size: 31 tokens
- Median chunk size: 140 tokens
- Maximum chunk size: 635 tokens
- Oversized chunks: 3

The oversized chunks were:

| Chunk ID | Tokens |
|---|---:|
| `architecture.md::0004` | 635 |
| `architecture.md::0009` | 513 |
| `rag-retrieval.md::0002` | 427 |

The current chunker permits some sections to exceed
the target token count. These exceptions require
further investigation before enforcing a hard limit.

The earlier retrieval evaluation used 6 documents
and 101 chunks. Its metrics are historical baseline
results, not measurements of this updated corpus.

Document changes may also invalidate previous
chunk-specific relevance judgments.
