
import json
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

# Confirmed output-label order for this model.
LABELS = ("contradiction", "entailment", "neutral")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "claim_grounding_cases.json"


def main() -> None:
    print("=== Claim Grounding Evaluation ===")

    # 1. Load the manually labeled dataset.
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    evidence_lookup = {
        item["id"]: item["text"]
        for item in dataset["evidence"]
    }

    cases = dataset["cases"]

    pairs = [
        (evidence_lookup[case["evidence_id"]], case["claim"])
        for case in cases
    ]

    expected = [case["label"] for case in cases]

    print(f"Evaluation examples: {len(cases)}")
    print(f"Evidence passages: {len(evidence_lookup)}")

    # 2. Run local NLI inference.
    print("\nLoading NLI model...")

    model = CrossEncoder(
        MODEL_NAME,
        device="cpu",
    )

    logits = np.asarray(
        model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )
    )

    if logits.shape != (len(cases), len(LABELS)):
        raise ValueError(
            f"Unexpected model output shape: {logits.shape}"
        )

    predicted = [
        LABELS[index]
        for index in np.argmax(logits, axis=1)
    ]

    # 3. Calculate the confusion matrix.
    label_index = {
        label: index
        for index, label in enumerate(LABELS)
    }

    matrix = np.zeros(
        (len(LABELS), len(LABELS)),
        dtype=int,
    )

    for actual, prediction in zip(
        expected,
        predicted,
        strict=True,
    ):
        matrix[
            label_index[actual],
            label_index[prediction],
        ] += 1

    print("\n=== Confusion Matrix ===")
    print("Rows = expected, columns = predicted")
    print(
        f"{'Actual / Predicted':<22}"
        + "".join(f"{label[:8]:>12}" for label in LABELS)
    )

    for label, row in zip(LABELS, matrix):
        print(
            f"{label:<22}"
            + "".join(f"{value:>12}" for value in row)
        )

    # 4. Calculate per-class precision, recall, and F1.
    print("\n=== Classification Metrics ===")

    f1_scores = []

    print(
        f"{'Class':<17}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>10}"
    )

    for index, label in enumerate(LABELS):
        true_positive = int(matrix[index, index])
        actual_count = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())

        precision = (
            true_positive / predicted_count
            if predicted_count else 0.0
        )

        recall = (
            true_positive / actual_count
            if actual_count else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        f1_scores.append(f1)

        print(
            f"{label:<17}"
            f"{precision:>12.3f}"
            f"{recall:>12.3f}"
            f"{f1:>12.3f}"
            f"{actual_count:>10}"
        )

    accuracy = float(np.trace(matrix) / len(cases))
    macro_f1 = float(np.mean(f1_scores))

    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"Macro-F1: {macro_f1:.3f}")

    # 5. Inspect misclassified examples.
    print("\n=== Misclassified Examples ===")

    errors = 0

    for case, actual, prediction in zip(
        cases,
        expected,
        predicted,
        strict=True,
    ):
        if actual != prediction:
            errors += 1

            print(f"\nCase: {case['id']}")
            print(f"Claim: {case['claim']}")
            print(f"Expected: {actual}")
            print(f"Predicted: {prediction}")

    if errors == 0:
        print("No misclassifications in this dataset.")

    print(f"\nTotal errors: {errors}")
    print("\n=== Evaluation Completed ===")


if __name__ == "__main__":
    main()
