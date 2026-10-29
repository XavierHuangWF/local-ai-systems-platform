
from collections import Counter
from statistics import median

from transformers import AutoTokenizer

from src.rag.document_loader import load_documents
from src.rag.chunker import (
    chunk_documents,
    MAX_CHUNK_TOKENS,
    OVERLAP_TOKENS,
)


def main() -> None:
    print("=== RAG Chunk Inspection ===")

    # Use the tokenizer associated with the local Qwen3 model.
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B-AWQ"
    )

    documents = load_documents()

    chunks = chunk_documents(
        documents=documents,
        tokenizer=tokenizer,
    )

    if not chunks:
        raise RuntimeError("No chunks were generated.")

    token_counts = [chunk.token_count for chunk in chunks]

    print("\n=== Overall Statistics ===")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Target size: {MAX_CHUNK_TOKENS} tokens")
    print(f"Overlap allowance: {OVERLAP_TOKENS} tokens")
    print(f"Minimum tokens: {min(token_counts)}")
    print(f"Median tokens: {median(token_counts)}")
    print(f"Maximum tokens: {max(token_counts)}")

    oversized = [
        chunk for chunk in chunks
        if chunk.over_budget
    ]

    print(f"Oversized chunks: {len(oversized)}")

    print("\n=== Chunks per Document ===")

    counts = Counter(chunk.source for chunk in chunks)

    for source, count in sorted(counts.items()):
        print(f"{source}: {count} chunks")

    print("\n=== Five Largest Chunks ===")

    largest = sorted(
        chunks,
        key=lambda chunk: chunk.token_count,
        reverse=True,
    )[:5]

    for chunk in largest:
        print(f"\nID: {chunk.chunk_id}")
        print(f"Heading: {chunk.heading}")
        print(f"Tokens: {chunk.token_count}")
        print(f"Over budget: {chunk.over_budget}")

    print("\n=== Example Chunk ===")

    example = chunks[0]

    print(f"ID: {example.chunk_id}")
    print(f"Source: {example.source}")
    print(f"Heading: {example.heading}")
    print(f"Tokens: {example.token_count}")
    print("\nContent preview:")
    print(example.text[:1000])

    print("\n=== Oversized Chunk Inspection ===")

    if not oversized:
        print("No chunks exceed the target size.")
    else:
        for chunk in oversized[:10]:
            print(
                f"{chunk.chunk_id}: "
                f"{chunk.token_count} tokens "
                f"| {chunk.heading}"
            )

    print("\n=== Final Result ===")
    print("Document chunk inspection completed.")


if __name__ == "__main__":
    main()
