
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

LABELS = [
    "contradiction",
    "entailment",
    "neutral",
]


def main() -> None:
    print("=== Claim Grounding Inspection ===")

    project_root = Path(__file__).resolve().parents[1]
    document_path = project_root / "docs" / "tool-execution.md"

    document = document_path.read_text(encoding="utf-8")

    # Extract the actual tool-validation documentation.
    evidence = document.split(
        "### Tool registration and argument validation",
        1,
    )[1].split(
        "### Tool execution and model round trip",
        1,
    )[0].strip()

    print("\nEvidence source: docs/tool-execution.md")
    print("\nEvidence:")
    print(evidence)

    claims = [
        (
            "Supported claim",
            "ToolExecutor validates model-generated arguments "
            "before invoking a registered Python function.",
        ),
        (
            "Contradicted claim",
            "ToolRegistry independently enforces JSON Schema "
            "on tool arguments.",
        ),
        (
            "Unsupported claim",
            "The system uploads audit logs to Amazon S3.",
        ),
    ]

    print("\nLoading local NLI cross-encoder...")
    model = CrossEncoder(MODEL_NAME, device="cpu")

    # Order matters: evidence (premise), then claim (hypothesis).

    # Extract the exact implementation statement from
    # the documentation for the supported claim.
    marker = "Model-generated requests use "

    focused_evidence = (
        marker
        + evidence.split(marker, 1)[1].split("\n", 1)[0]
    )

    print("\nFocused evidence for the supported claim:")
    print(focused_evidence)

    # Compare the same claim using full and focused evidence.

    claim_variants = [
        (
            "Original technical paraphrase",
            claims[0][1],
        ),
        (
            "Directly supported statement",
            "ToolExecutor applies ToolArgumentValidator "
            "before invoking the registry.",
        ),
        (
            "Direct contradiction",
            "ToolExecutor does not apply ToolArgumentValidator "
            "before invoking the registry.",
        ),
        (
            "Identical evidence",
            focused_evidence,
        ),

        (
            "Meaning-preserving technical paraphrase",
            "ToolExecutor checks model-generated requests "
            "using ToolArgumentValidator before calling "
            "the registry.",
        ),

    ]

    pairs = [
        (focused_evidence, claim)
        for _, claim in claim_variants
    ]


    labels_for_results = [
        "Full evidence — supported claim",
        "Focused evidence — supported claim",
    ]

    scores = np.asarray(
        model.predict(
            pairs,
            batch_size=3,
            show_progress_bar=False,
        )
    )

    for (name, claim), logits in zip(
            claim_variants,
            scores,
            strict=True,
    ):
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()

        predicted = LABELS[int(np.argmax(probabilities))]

        print(f"\n=== {name} ===")
        print(f"Claim: {claim}")
        print(f"Predicted relationship: {predicted}")

        for label, probability in zip(LABELS, probabilities):
            print(f"  {label}: {probability:.4f}")

    print("\n=== Inspection Completed ===")


if __name__ == "__main__":
    main()
