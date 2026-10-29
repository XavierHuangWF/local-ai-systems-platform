
from src.rag.retriever import HybridRetriever


QUESTIONS = {
    "Evaluation wording": (
        "How does ToolExecutor validate tool arguments "
        "before executing registered Python functions?"
    ),
    "Interactive wording": (
        "How does the Agent Harness validate tool arguments "
        "before executing Python functions?"
    ),
}

TARGET_CHUNK = "tool-execution.md::0004"


def main() -> None:
    print("=== Retrieval Query Sensitivity ===")

    # Build the index once for both questions.
    retriever = HybridRetriever(
        device="cpu",
        candidate_k=20,
    )

    for label, question in QUESTIONS.items():
        print("\n" + "=" * 65)
        print(label)
        print(f"Question: {question}")
        print("=" * 65)

        results = retriever.search(
            query=question,
            top_k=20,
        )

        target_rank = None

        for rank, (chunk, score) in enumerate(
            results,
            start=1,
        ):
            if chunk.chunk_id == TARGET_CHUNK:
                target_rank = rank

            marker = (
                " <-- TARGET"
                if chunk.chunk_id == TARGET_CHUNK
                else ""
            )

            print(
                f"{rank:2}. {chunk.chunk_id} "
                f"| RRF={score:.5f}{marker}"
            )

        print("\nTarget passage:")
        print(TARGET_CHUNK)

        if target_rank is None:
            print("Not found in the top 20.")
        else:
            print(f"Rank: {target_rank}")
            print(f"Included in top five: {target_rank <= 5}")

    print("\n=== Diagnostic Completed ===")


if __name__ == "__main__":
    main()
