
import json
from types import SimpleNamespace

from src.tools.document_search_tool import (
    DocumentSearchTool,
    MAX_EVIDENCE_TOKENS,
)


def count_result_tokens(tokenizer, result):
    serialized = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
    )

    return len(
        tokenizer.encode(
            serialized,
            add_special_tokens=False,
        )
    )


def verify_result(label, tokenizer, result):
    tokens = count_result_tokens(tokenizer, result)

    print(f"\n=== {label} ===")
    print(f"Evidence tokens: {tokens}")
    print(f"Token limit: {MAX_EVIDENCE_TOKENS}")
    print(f"Returned passages: {len(result['passages'])}")
    print(f"Omitted passages: {result['omitted_passages']}")

    for passage in result["passages"]:
        print(f"  {passage['chunk_id']}")

    if tokens > MAX_EVIDENCE_TOKENS:
        raise AssertionError("Evidence budget exceeded.")

    print("PASS: Evidence remains within the token limit.")


class SimulatedRetriever:
    """Return one oversized passage followed by a small passage."""

    def search(self, query, top_k):
        oversized = SimpleNamespace(
            chunk_id="simulated.md::0001",
            source="simulated.md",
            heading="Oversized passage",
            text="Large document evidence. " * 6000,
        )

        small = SimpleNamespace(
            chunk_id="simulated.md::0002",
            source="simulated.md",
            heading="Relevant short passage",
            text="The verified measurements are [13, 25, 61, 92].",
        )

        return [
            (oversized, 0.9),
            (small, 0.8),
        ][:top_k]


def main():
    print("=== Document Search Budget Verification ===")

    # Experiment 1: Actual project documentation.
    tool = DocumentSearchTool()

    real_result = tool(
        query="Agent Harness architecture, tool validation, "
              "and verified experiment results",
        top_k=6,
    )

    verify_result(
        "Real Document Retrieval",
        tool._tokenizer,
        real_result,
    )

    # Experiment 2: Simulated oversized evidence.
    simulated_tool = DocumentSearchTool()
    simulated_tool._retriever = SimulatedRetriever()
    simulated_tool._tokenizer = tool._tokenizer

    simulated_result = simulated_tool(
        query="Find verified measurements",
        top_k=2,
    )

    verify_result(
        "Oversized Evidence Simulation",
        simulated_tool._tokenizer,
        simulated_result,
    )

    returned_ids = {
        passage["chunk_id"]
        for passage in simulated_result["passages"]
    }

    if "simulated.md::0001" in returned_ids:
        raise AssertionError(
            "Oversized passage was not excluded."
        )

    if "simulated.md::0002" not in returned_ids:
        raise AssertionError(
            "Useful smaller passage was lost."
        )

    if simulated_result["omitted_passages"] != 1:
        raise AssertionError(
            "Incorrect number of omitted passages."
        )

    print("\nPASS: Oversized passage excluded.")
    print("PASS: Smaller passage retained.")
    print("PASS: Omission count verified.")
    print("\n=== Verification Completed ===")


if __name__ == "__main__":
    main()
