
from src.rag.chunker import DocumentChunk


def reciprocal_rank_fusion(
    rankings: list[list[tuple[DocumentChunk, float]]],
    top_k: int = 5,
    k: int = 60,
) -> list[tuple[DocumentChunk, float]]:
    """
    Combine ranked retrieval results using Reciprocal Rank Fusion.

    Input scores are ignored because different retrieval
    algorithms use different scoring scales.
    """

    if not rankings:
        raise ValueError("At least one ranking is required.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if k <= 0:
        raise ValueError("k must be positive.")

    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, DocumentChunk] = {}

    # Process BM25, semantic, or other ranked result lists.
    for ranking in rankings:
        seen = set()

        for rank, (chunk, original_score) in enumerate(
            ranking,
            start=1,
        ):
            chunk_id = chunk.chunk_id

            # Avoid counting duplicates within one ranking.
            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            chunks_by_id[chunk_id] = chunk

            contribution = 1.0 / (k + rank)

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + contribution
            )

    # Rank passages by combined RRF score.
    sorted_results = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [
        (chunks_by_id[chunk_id], score)
        for chunk_id, score in sorted_results[:top_k]
    ]
