
from src.rag.qa import RAGAssistant


QUESTION = (
    "What caused the FlashInfer CUDA compilation "
    "problem, and how was it resolved?"
)


def main() -> None:
    print("=== Live RAG Experiment ===")
    print(f"\nQuestion:\n{QUESTION}")

    # Initialize the local document retriever and model client.
    assistant = RAGAssistant(top_k=5)

    print("\nRetrieving evidence and generating an answer...")

    # Retrieve document chunks and query Qwen3.
    answer, sources = assistant.answer(QUESTION)

    print("\n=== Qwen3 Generated Answer ===")
    print(answer)

    print("\n=== Retrieved Evidence Sources ===")

    for number, chunk in enumerate(sources, start=1):
        print(f"\n[{number}] {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Heading: {chunk.heading}")
        print(f"Tokens: {chunk.token_count}")

    print("\n=== Experiment Completed ===")
    print("Review the answer and verify its citations manually.")


if __name__ == "__main__":
    main()
