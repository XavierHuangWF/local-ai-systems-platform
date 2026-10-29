
import numpy as np
from transformers import AutoTokenizer

from src.rag.document_loader import load_documents
from src.rag.chunker import chunk_documents
from src.rag.bm25 import BM25Index
from src.rag.embeddings import LocalEmbedder
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.reranker import LocalReranker


QUESTIONS = [
    "How does the Agent Harness manage conversation history "
    "and prevent oversized prompts?",

    "How are skills loaded and activated in the Agent Harness?",

    "What caused the FlashInfer CUDA compilation problem, "
    "and how was it resolved?",

    "How does ToolExecutor validate tool arguments before "
    "executing registered Python functions?",

    "How does the system remove Qwen3 thinking tags "
    "from final responses?",
]

CANDIDATE_K = 20
FINAL_K = 5


def display_results(label, results):
    print(f"\n{label}")

    for rank, (chunk, score) in enumerate(results, 1):
        print(
            f"  {rank}. {chunk.chunk_id} "
            f"| score={score:.4f}"
        )
        print(f"     Heading: {chunk.heading}")

        preview = " ".join(chunk.text.split())
        print(f"     Preview: {preview[:180]}")


def main():
    print("=== Multi-Question Retrieval Exploration ===")

    # Prepare the knowledge base once.
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B-AWQ"
    )

    documents = load_documents()
    chunks = chunk_documents(documents, tokenizer)

    bm25 = BM25Index(chunks)
    embedder = LocalEmbedder(device="cpu")
    reranker = LocalReranker(device="cpu")

    # Compute document embeddings once for all questions.
    document_vectors = embedder.embed_chunks(chunks)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Questions: {len(QUESTIONS)}")

    for question_number, question in enumerate(
        QUESTIONS, 1
    ):
        print("\n" + "=" * 65)
        print(f"QUESTION {question_number}: {question}")
        print("=" * 65)

        # BM25 keyword retrieval.
        keyword_results = bm25.search(
            question,
            top_k=CANDIDATE_K,
        )

        # Semantic embedding retrieval.
        query_vector = embedder.embed_query(question)

        similarities = document_vectors @ query_vector

        indices = np.argsort(
            -similarities
        )[:CANDIDATE_K]

        semantic_results = [
            (chunks[int(i)], float(similarities[i]))
            for i in indices
        ]

        # Hybrid Reciprocal Rank Fusion.
        hybrid_results = reciprocal_rank_fusion(
            rankings=[keyword_results, semantic_results],
            top_k=CANDIDATE_K,
            k=60,
        )

        # Cross-encoder reranking.
        reranked_results = reranker.rerank(
            query=question,
            candidates=hybrid_results,
            top_k=FINAL_K,
        )

        display_results(
            "Hybrid RRF Top 5:",
            hybrid_results[:FINAL_K],
        )

        display_results(
            "Cross-Encoder Top 5:",
            reranked_results,
        )

        # Show which candidate moved to the top.
        rrf_positions = {
            chunk.chunk_id: rank
            for rank, (chunk, _) in enumerate(
                hybrid_results, 1
            )
        }

        best_chunk = reranked_results[0][0]

        print("\nTop reranked passage:")
        print(f"  {best_chunk.chunk_id}")
        print(
            "  Original RRF position:",
            rrf_positions[best_chunk.chunk_id],
        )

    print("\n=== Final Result ===")
    print("Multi-question retrieval exploration completed.")


if __name__ == "__main__":
    main()
