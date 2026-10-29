
import numpy as np
from transformers import AutoTokenizer

from src.rag.document_loader import load_documents
from src.rag.chunker import chunk_documents
from src.rag.embeddings import LocalEmbedder


QUERY = (
    "How does the Agent Harness validate tool arguments "
    "before executing a Python function?"
)

TOP_K = 5


def main() -> None:
    print("=== Local Embedding and Semantic Search ===")

    # 1. Load and chunk the actual repository documents.
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B-AWQ"
    )

    documents = load_documents()
    chunks = chunk_documents(documents, tokenizer)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")

    # 2. Load the local embedding model on CPU.
    embedder = LocalEmbedder(device="cpu")

    # 3. Generate numerical vectors for every chunk.
    chunk_vectors = embedder.embed_chunks(chunks)

    print("\n=== Embedding Matrix ===")
    print(f"Shape: {chunk_vectors.shape}")
    print(f"Data type: {chunk_vectors.dtype}")

    # 4. Embed the user's question.
    query_vector = embedder.embed_query(QUERY)

    print(f"Query vector shape: {query_vector.shape}")

    # 5. Check vector normalization.
    chunk_norms = np.linalg.norm(
        chunk_vectors,
        axis=1,
    )

    query_norm = np.linalg.norm(query_vector)

    print("\n=== Normalization ===")
    print(f"Minimum chunk norm: {chunk_norms.min():.6f}")
    print(f"Maximum chunk norm: {chunk_norms.max():.6f}")
    print(f"Query norm: {query_norm:.6f}")

    # 6. Calculate cosine similarity.
    # Since vectors are normalized, dot product equals
    # cosine similarity.
    scores = chunk_vectors @ query_vector

    # 7. Sort by similarity, highest first.
    ranked_indices = np.argsort(-scores)[:TOP_K]

    print("\n=== Search Question ===")
    print(QUERY)

    print(f"\n=== Top {TOP_K} Retrieved Chunks ===")

    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[int(index)]
        score = float(scores[index])

        print(f"\nRank: {rank}")
        print(f"Similarity: {score:.4f}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Heading: {chunk.heading}")
        print(f"Tokens: {chunk.token_count}")

        preview = chunk.text[:400].replace("\n", " ")
        print(f"Preview: {preview}")

    print("\n=== Final Result ===")
    print("Local embedding and semantic ranking completed.")


if __name__ == "__main__":
    main()
