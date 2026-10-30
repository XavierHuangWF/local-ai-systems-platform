
from src.agent.source_attribution import check_source_attribution


def main() -> None:
    print("=== Source Attribution Verification ===")

    retrieved_passages = [
        {
            "source": "tool-execution.md",
            "chunk_id": "tool-execution.md::0004",
            "content": "Tool arguments are validated before execution.",
        },
        {
            "source": "architecture.md",
            "chunk_id": "architecture.md::0012",
            "content": "The Agent Harness manages conversation state.",
        },
    ]

    answer = (
        "Tool validation is documented in "
        "tool-execution.md::0004. "
        "The architecture is described in architecture.md. "
        "Additional evidence allegedly appears in "
        "nonexistent.md::9999."
    )

    result = check_source_attribution(
        answer=answer,
        passages=retrieved_passages,
    )

    print("\nReferenced:")
    print(result.referenced)

    print("\nMatched:")
    print(result.matched)

    print("\nUnsupported:")
    print(result.unsupported)

    assert result.matched == [
        "tool-execution.md::0004",
        "architecture.md",
    ]

    assert result.unsupported == [
        "nonexistent.md::9999",
    ]

    print("\nPASS: Retrieved sources matched.")
    print("PASS: Fabricated source detected.")

    # A real document with an unreturned chunk ID
    # must not be considered a valid chunk citation.
    wrong_chunk = check_source_attribution(
        answer="See tool-execution.md::9999.",
        passages=retrieved_passages,
    )

    assert wrong_chunk.unsupported == [
        "tool-execution.md::9999",
    ]

    print("PASS: Unretrieved chunk ID detected.")
    print("\n=== Verification Completed ===")


if __name__ == "__main__":
    main()
