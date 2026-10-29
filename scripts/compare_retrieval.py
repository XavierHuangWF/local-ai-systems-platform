
import numpy as np
from transformers import AutoTokenizer

from src.rag.document_loader import load_documents
from src.rag.chunker import chunk_documents
from src.rag.bm25 import BM25Index
from src.rag.embeddings import LocalEmbedder


QUERY = (
    "How does the Agent Harness validate tool arguments "
    "before executing a Python function?"
)

TOP_K = 5


def print_results(label, results):
    """Display ranked chunks with their relevance scores."""

    print(f"\n=== {label} ===")

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nRank {rank} | Score: {score:.4f}")
        print(f"ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Heading: {chunk.heading}")
        print(
            "Preview:",
            chunk.text[:220].replace("\n", " "),
        )


def main():
    print("=== BM25 vs Semantic Retrieval ===")

    # 1. Load and chunk the same documents.
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B-AWQ"
    )

    documents = load_documents()
    chunks = chunk_documents(documents, tokenizer)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Question: {QUERY}")

    # 2. BM25 keyword ranking.
    bm25 = BM25Index(chunks)
    bm25_results = bm25.search(QUERY, top_k=TOP_K)

    print_results("BM25 Keyword Ranking", bm25_results)

    # 3. Embedding-based semantic ranking.
    embedder = LocalEmbedder(device="cpu")

    document_vectors = embedder.embed_chunks(chunks)
    query_vector = embedder.embed_query(QUERY)

    similarity_scores = document_vectors @ query_vector
    top_indices = np.argsort(-similarity_scores)[:TOP_K]

    semantic_results = [
        (chunks[int(index)], float(similarity_scores[index]))
        for index in top_indices
    ]

    print_results(
        "Cosine Similarity Ranking",
        semantic_results,
    )

    # 4. Compare which chunks appear in both rankings.
    bm25_ids = {
        chunk.chunk_id
        for chunk, _ in bm25_results
    }

    semantic_ids = {
        chunk.chunk_id
        for chunk, _ in semantic_results
    }

    shared_ids = bm25_ids & semantic_ids

    print("\n=== Ranking Comparison ===")
    print(f"BM25 results: {len(bm25_results)}")
    print(f"Semantic results: {len(semantic_results)}")
    print(f"Shared top-{TOP_K} chunks: {len(shared_ids)}")

    print("\nShared chunks:")
    for chunk_id in sorted(shared_ids):
        print(f"  {chunk_id}")

    print("\n=== Final Result ===")
    print("BM25 and semantic retrieval comparison completed.")


if __name__ == "__main__":
    main()
