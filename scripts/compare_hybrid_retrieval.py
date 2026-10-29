
import numpy as np
from transformers import AutoTokenizer

from src.rag.document_loader import load_documents
from src.rag.chunker import chunk_documents
from src.rag.bm25 import BM25Index
from src.rag.embeddings import LocalEmbedder
from src.rag.fusion import reciprocal_rank_fusion


QUERY = (
    "How does the Agent Harness validate tool arguments "
    "before executing a Python function?"
)

CANDIDATE_K = 20
TOP_K = 5


def display_ranking(title, results):
    print(f"\n=== {title} ===")

    for rank, (chunk, score) in enumerate(results[:TOP_K], 1):
        print(f"\nRank {rank} | Score: {score:.4f}")
        print(f"Chunk: {chunk.chunk_id}")
        print(f"Heading: {chunk.heading}")
        print(
            "Preview:",
            chunk.text[:250].replace("\n", " "),
        )


def main():
    print("=== Hybrid Retrieval Experiment ===")

    # Load the same documents and chunking configuration.
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B-AWQ"
    )

    documents = load_documents()
    chunks = chunk_documents(documents, tokenizer)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Question: {QUERY}")

    # 1. BM25 keyword retrieval.
    bm25 = BM25Index(chunks)

    bm25_results = bm25.search(
        QUERY,
        top_k=CANDIDATE_K,
    )

    # 2. Embedding-based semantic retrieval.
    embedder = LocalEmbedder(device="cpu")

    document_vectors = embedder.embed_chunks(chunks)
    query_vector = embedder.embed_query(QUERY)

    similarity_scores = document_vectors @ query_vector

    semantic_indices = np.argsort(
        -similarity_scores
    )[:CANDIDATE_K]

    semantic_results = [
        (
            chunks[int(index)],
            float(similarity_scores[index]),
        )
        for index in semantic_indices
    ]

    # 3. Combine both rankings using RRF.
    hybrid_results = reciprocal_rank_fusion(
        rankings=[bm25_results, semantic_results],
        top_k=TOP_K,
        k=60,
    )

    display_ranking(
        "BM25 Keyword Ranking",
        bm25_results,
    )

    display_ranking(
        "Semantic Ranking",
        semantic_results,
    )

    display_ranking(
        "Hybrid RRF Ranking",
        hybrid_results,
    )

    # 4. Identify how hybrid results ranked originally.
    bm25_positions = {
        chunk.chunk_id: rank
        for rank, (chunk, _) in enumerate(
            bm25_results, 1
        )
    }

    semantic_positions = {
        chunk.chunk_id: rank
        for rank, (chunk, _) in enumerate(
            semantic_results, 1
        )
    }

    print("\n=== Hybrid Ranking Explanation ===")

    for rank, (chunk, score) in enumerate(
        hybrid_results, 1
    ):
        print(f"\nHybrid rank: {rank}")
        print(f"Chunk: {chunk.chunk_id}")
        print(f"BM25 rank: {bm25_positions.get(chunk.chunk_id, '-')}")
        print(
            "Semantic rank:",
            semantic_positions.get(chunk.chunk_id, "-"),
        )
        print(f"RRF score: {score:.6f}")

    print("\n=== Final Result ===")
    print("Hybrid retrieval comparison completed.")


if __name__ == "__main__":
    main()
