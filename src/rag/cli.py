
from src.rag.qa import RAGAssistant


def main() -> None:
    print("=== Local RAG Assistant ===")
    print("Ask questions about your project documentation.")
    print("Type /exit to quit.")

    # Load documents, chunks, and embeddings once.
    assistant = RAGAssistant(top_k=6)

    print("\nKnowledge base ready.")

    while True:
        try:
            question = input("\nRAG> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if question.lower() in {"/exit", "/quit"}:
            print("Session ended.")
            break

        if not question:
            continue

        try:
            answer, sources = assistant.answer(question)
        except Exception as exc:
            print(f"\nRequest failed: {exc}")
            continue

        print("\n=== Answer ===")
        print(answer)

        print("\n=== Sources ===")

        for number, chunk in enumerate(sources, start=1):
            print(f"[{number}] {chunk.chunk_id}")
            print(f"    {chunk.heading}")


if __name__ == "__main__":
    main()
