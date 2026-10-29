
from src.rag.document_loader import load_documents


def main() -> None:
    print("=== RAG Document Inspection ===")

    documents = load_documents()

    total_characters = 0

    for index, document in enumerate(documents, start=1):
        character_count = len(document.text)
        total_characters += character_count

        headings = [
            line.strip()
            for line in document.text.splitlines()
            if line.startswith("#")
        ]

        print(f"\nDocument {index}: {document.source}")
        print(f"Characters: {character_count}")
        print(f"Headings: {len(headings)}")

        print("First headings:")
        for heading in headings[:5]:
            print(f"  {heading}")

    print("\n=== Summary ===")
    print(f"Documents loaded: {len(documents)}")
    print(f"Total characters: {total_characters}")

    print("\nDocument loading completed.")


if __name__ == "__main__":
    main()
