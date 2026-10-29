
import numpy as np
from transformers import AutoTokenizer

from src.rag.document_loader import load_documents
from src.rag.chunker import chunk_documents
from src.rag.bm25 import BM25Index
from src.rag.embeddings import LocalEmbedder
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.reranker import LocalReranker


QUERY = (
    "How does the Agent Harness validate tool arguments "
    "before executing a Python function?"
)

RETRIEVAL_K = 20
RERANK_K = 20
FINAL_K = 5


def display_results(title, results):
    print(f"\n=== {title} ===")

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"Chunk: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Heading: {chunk.heading}")
        print(f"Tokens: {chunk.token_count}")
        print("Preview:", chunk.text[:350].replace("\n", " "))


def main():
    print("=== Hybrid Retrieval vs Cross-Encoder Reranking ===")

    # 1. Load and chunk actual repository documents.
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B-AWQ"
    )

    documents = load_documents()
    chunks = chunk_documents(documents, tokenizer)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Question: {QUERY}")

    # 2. BM25 keyword retrieval.
    bm25 = BM25Index(chunks)
    bm25_results = bm25.search(
        QUERY,
        top_k=RETRIEVAL_K,
    )

    # 3. Semantic embedding retrieval.
    embedder = LocalEmbedder(device="cpu")

    document_vectors = embedder.embed_chunks(chunks)
    query_vector = embedder.embed_query(QUERY)

    similarities = document_vectors @ query_vector

    semantic_indices = np.argsort(
        -similarities
    )[:RETRIEVAL_K]

    semantic_results = [
        (chunks[int(i)], float(similarities[i]))
        for i in semantic_indices
    ]

    # 4. Fuse the ranked lists.
    hybrid_results = reciprocal_rank_fusion(
        rankings=[bm25_results, semantic_results],
        top_k=RERANK_K,
        k=60,
    )

    display_results(
        "Hybrid RRF Top 5",
        hybrid_results[:FINAL_K],
    )

    # 5. Rerank the 20 fused candidates.
    reranker = LocalReranker(device="cpu")

    reranked_results = reranker.rerank(
        query=QUERY,
        candidates=hybrid_results,
        top_k=FINAL_K,
    )

    display_results(
        "Cross-Encoder Reranked Top 5",
        reranked_results,
    )

    # 6. Compare each passage's position before and after.
    original_positions = {
        chunk.chunk_id: rank
        for rank, (chunk, _) in enumerate(
            hybrid_results,
            start=1,
        )
    }

    print("\n=== Ranking Changes ===")

    for rank, (chunk, score) in enumerate(
        reranked_results,
        start=1,
    ):
        previous_rank = original_positions[chunk.chunk_id]

        print(f"\nChunk: {chunk.chunk_id}")
        print(f"RRF rank: {previous_rank}")
        print(f"Reranked position: {rank}")
        print(f"Cross-encoder score: {score:.4f}")

    print("\n=== Final Result ===")
    print("Cross-encoder reranking experiment completed.")


if __name__ == "__main__":
    main()
