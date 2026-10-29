
from transformers import AutoTokenizer

from src.rag.document_loader import load_documents
from src.rag.chunker import chunk_documents




TARGET_IDS = {
    # Q2: Skill loading and activation
    "skills-and-context.md::0025",
    "skills-and-context.md::0009",
    "skills-and-context.md::0019",
    "architecture.md::0002",
    "skills-and-context.md::0001",
    "architecture.md::0008",
    "skills-and-context.md::0030",

    # Q3: FlashInfer CUDA compilation
    "troubleshooting.md::0010",
    "troubleshooting.md::0005",

    # Q5: Qwen3 thinking-tag removal
    "agent-harness.md::0011",
    "tool-execution.md::0011",
    "tool-execution.md::0001",
    "skills-and-context.md::0030",
}




def main() -> None:
    print("=== Retrieval Relevance Evidence Review ===")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B-AWQ"
    )

    documents = load_documents()
    chunks = chunk_documents(documents, tokenizer)

    found = set()

    for chunk in chunks:
        if chunk.chunk_id not in TARGET_IDS:
            continue

        found.add(chunk.chunk_id)

        print("\n" + "=" * 70)
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Heading: {chunk.heading}")
        print(f"Tokens: {chunk.token_count}")
        print("=" * 70)

        # Print the complete chunk rather than a short preview.
        print(chunk.text)

    missing = TARGET_IDS - found

    if missing:
        print("\nMissing chunk identifiers:")
        for chunk_id in sorted(missing):
            print(chunk_id)

    print("\n=== Summary ===")
    print(f"Requested chunks: {len(TARGET_IDS)}")
    print(f"Found chunks: {len(found)}")
    print("Relevance evidence inspection completed.")


if __name__ == "__main__":
    main()
