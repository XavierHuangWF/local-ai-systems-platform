
# Claim Grounding — NLI Evaluation

## Objective

Evaluate whether a small local Natural Language Inference (NLI)
model can classify claims against technical project documentation.

The three labels are:

- Entailment: the evidence supports the claim.
- Contradiction: the evidence conflicts with the claim.
- Neutral: the evidence does not establish the claim.

This experiment extends source-reference checking toward
claim-level grounding, without modifying production inference.

## Model

Model: `cross-encoder/nli-MiniLM2-L6-H768`

Framework: Sentence Transformers

Execution device: CPU

Label order:

1. contradiction
2. entailment
3. neutral

The model receives evidence as the premise and a claim
as the hypothesis.

## Experiment Files

- `scripts/inspect_claim_grounding.py`
- `scripts/evaluate_claim_grounding.py`
- `data/claim_grounding_cases.json`

## Initial Diagnostic Experiments

The original documentation section was evaluated against
three claims:

- Supported validation behavior
- Contradicted registry validation behavior
- Unsupported Amazon S3 logging behavior

The model correctly classified the explicit contradiction
and unsupported logging claim, but misclassified the original
validation claim.

A focused-evidence comparison changed that prediction
from contradiction to neutral.

Further controlled tests showed:

- Directly supported statement: entailment (0.9888)
- Direct contradiction: contradiction (0.9981)
- Identical evidence: entailment (0.9799)
- Meaning-preserving paraphrase: entailment (0.7226)
- Original technical paraphrase: neutral (0.9544)

These results suggest sensitivity to terminology and
the exact scope of available evidence.

## Labeled Evaluation Dataset

The benchmark contains 12 manually labeled cases across
four documentation excerpts.

Class distribution:

| Label | Examples |
|---|---:|
| Entailment | 4 |
| Contradiction | 4 |
| Neutral | 4 |
| Total | 12 |

The cases cover:

- Tool registration and argument validation
- Measurement input constraints
- Multi-round tool execution
- Document-search evidence budgets

The dataset is a small curated diagnostic set,
not an independently validated production benchmark.

## Reproduction

Activate the project environment and run:

    python -m scripts.evaluate_claim_grounding

The experiment runs locally on CPU without a vLLM server.

## Observed Evaluation Results

Accuracy: 0.833 (10/12 correct)

Macro-F1: 0.833

### Confusion Matrix

Rows are expected labels; columns are predictions.

| Expected / Predicted | Contradiction | Entailment | Neutral |
|---|---:|---:|---:|
| Contradiction | 3 | 0 | 1 |
| Entailment | 0 | 4 | 0 |
| Neutral | 1 | 0 | 3 |

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Contradiction | 0.750 | 0.750 | 0.750 | 4 |
| Entailment | 1.000 | 1.000 | 1.000 | 4 |
| Neutral | 0.750 | 0.750 | 0.750 | 4 |

## Error Analysis

### Case M2 — Missed Contradiction

Claim:

"Boolean elements are accepted as valid measurements."

Expected: contradiction

Predicted: neutral

The evidence explicitly describes Boolean values as invalid.
The model failed to recognize the contradiction.

### Case M3 — Unsupported Claim Classified as Contradiction

Claim:

"The statistics tool stores measurements in SQLite."

Expected: neutral

Predicted: contradiction

The evidence does not establish whether SQLite storage occurs.
The model incorrectly treated absent support as contradiction.

## Engineering Findings

1. The model recognized all four entailment cases.
2. It correctly identified three of four contradictions.
3. It correctly identified three of four neutral cases.
4. The distinction between contradiction and neutral remains unreliable.
5. Technical paraphrase interpretation depends on evidence wording.
6. Softmax probabilities are not calibrated factual guarantees.
7. No false entailments occurred in these 12 cases, but the sample
   is too small to establish an acceptably low false-approval rate.

## Limitations

- Only 12 manually constructed cases were evaluated.
- Evidence excerpts were curated and relatively short.
- The dataset is not representative of production answers.
- Some expected labels may depend on documentation completeness.
- No independent annotation agreement was measured.
- No generalization test across unrelated repositories was performed.
- No calibration, robustness, or threshold-selection study was performed.
- Claim extraction from complete generated answers was not implemented.
- Valid source references do not guarantee supported claims.

## Decision

Status: Experimental — evaluation completed.

Do not use this NLI model to automatically approve, reject,
or suppress production Agent Harness answers.

Keep the deterministic source-attribution checker in the
existing agent workflow.

The NLI model and labeled dataset remain available for
future research and comparisons with stronger approaches.

No further NLI work is required for the current engineering
milestone.
