
from sentence_transformers import CrossEncoder

from src.rag.chunker import DocumentChunk


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


class LocalReranker:
    """Rerank retrieved chunks using a local cross-encoder."""

    def __init__(self, device: str = "cpu") -> None:
        self.model = CrossEncoder(
            RERANKER_MODEL_NAME,
            device=device,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        candidates: list[tuple[DocumentChunk, float]],
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Score question-passage pairs and reorder candidates."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not candidates:
            raise ValueError("No candidates to rerank.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        # Each candidate is evaluated together with the query.
        pairs = [
            (query, chunk.text)
            for chunk, _ in candidates
        ]

        # Compute relevance scores for all candidate pairs.
        scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        results = [
            (chunk, float(score))
            for (chunk, _), score in zip(
                candidates,
                scores,
                strict=True,
            )
        ]

        # Sort by cross-encoder relevance, highest first.
        results.sort(
            key=lambda item: (
                -item[1],
                item[0].chunk_id,
            )
        )

        return results[:top_k]
