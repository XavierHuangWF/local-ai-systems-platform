
import json
import math
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.rag.document_loader import load_documents
from src.rag.chunker import chunk_documents
from src.rag.bm25 import BM25Index
from src.rag.embeddings import LocalEmbedder
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.reranker import LocalReranker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS_FILE = PROJECT_ROOT / "data" / "relevance_judgments.json"

CANDIDATE_K = 20
TOP_K = 5
RELEVANCE_THRESHOLD = 2


def dcg(grades):
    """Discounted Cumulative Gain: reward highly relevant early hits."""
    return sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def evaluate_ranking(results, judgments):
    """
    Evaluate the actual top-five ranking.

    Return None if any top-five passage is unjudged.
    Never silently treat missing judgments as irrelevant.
    """
    ids = [chunk.chunk_id for chunk, _ in results]

    if any(chunk_id not in judgments for chunk_id in ids):
        return None

    grades = [judgments[chunk_id] for chunk_id in ids]

    relevant_ids = {
        chunk_id
        for chunk_id, grade in judgments.items()
        if grade >= RELEVANCE_THRESHOLD
    }

    # Recall is relative to known relevant judgments.
    hits = sum(chunk_id in relevant_ids for chunk_id in ids)
    recall = hits / len(relevant_ids) if relevant_ids else 0.0

    # Reciprocal rank of the first relevant passage.
    mrr = next(
        (
            1.0 / rank
            for rank, grade in enumerate(grades, start=1)
            if grade >= RELEVANCE_THRESHOLD
        ),
        0.0,
    )

    # Compare the actual ranking against the ideal judged ranking.
    ideal_grades = sorted(
        judgments.values(),
        reverse=True,
    )[:len(ids)]

    ideal_dcg = dcg(ideal_grades)
    ndcg = dcg(grades) / ideal_dcg if ideal_dcg else 0.0

    return {
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def display_evaluation(label, results, judgments, required):
    print(f"\n{label}")

    for rank, (chunk, _) in enumerate(results, start=1):
        grade = judgments.get(chunk.chunk_id)

        grade_text = (
            str(grade) if grade is not None else "UNJUDGED"
        )

        print(
            f"  {rank}. {chunk.chunk_id} "
            f"| relevance={grade_text}"
        )

    metrics = evaluate_ranking(results, judgments)

    if metrics is None:
        print("  Metrics: N/A (top-five contains unjudged passages)")
    else:
        print(f"  Judged Recall@5: {metrics['recall']:.3f}")
        print(f"  MRR@5:           {metrics['mrr']:.3f}")
        print(f"  Judged nDCG@5:   {metrics['ndcg']:.3f}")

    if required:
        retrieved_ids = {
            chunk.chunk_id for chunk, _ in results
        }

        found = sum(
            chunk_id in retrieved_ids
            for chunk_id in required
        )

        print(
            f"  Required evidence: {found}/{len(required)}"
        )


def main():
    print("=== Retrieval Ranking Evaluation ===")

    with JUDGMENTS_FILE.open(encoding="utf-8") as file:
        dataset = json.load(file)

    questions = dataset["questions"]

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B-AWQ"
    )

    documents = load_documents()
    chunks = chunk_documents(documents, tokenizer)

    bm25 = BM25Index(chunks)
    embedder = LocalEmbedder(device="cpu")
    reranker = LocalReranker(device="cpu")

    document_vectors = embedder.embed_chunks(chunks)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Evaluation questions: {len(questions)}")

    for item in questions:
        question = item["question"]
        judgments = item["judgments"]
        required = item.get("required_evidence", [])

        print("\n" + "=" * 65)
        print(f"{item['id'].upper()}: {question}")
        print("=" * 65)

        # 1. Keyword search.
        keyword_results = bm25.search(
            question,
            top_k=CANDIDATE_K,
        )

        # 2. Semantic search.
        query_vector = embedder.embed_query(question)
        similarities = document_vectors @ query_vector

        indices = np.argsort(
            -similarities
        )[:CANDIDATE_K]

        semantic_results = [
            (chunks[int(i)], float(similarities[i]))
            for i in indices
        ]

        # 3. Hybrid ranking using RRF.
        hybrid = reciprocal_rank_fusion(
            rankings=[keyword_results, semantic_results],
            top_k=CANDIDATE_K,
            k=60,
        )

        # 4. Cross-encoder reranking.
        reranked = reranker.rerank(
            query=question,
            candidates=hybrid,
            top_k=TOP_K,
        )

        # 5. Compare actual top-five results.
        display_evaluation(
            "Hybrid RRF:",
            hybrid[:TOP_K],
            judgments,
            required,
        )

        display_evaluation(
            "Cross-Encoder:",
            reranked,
            judgments,
            required,
        )

    print("\n=== Evaluation Completed ===")
    print("Unjudged passages were not assigned relevance grade 0.")


if __name__ == "__main__":
    main()
