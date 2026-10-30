
import json

from transformers import AutoTokenizer


MAX_EVIDENCE_TOKENS = 2500


class DocumentSearchTool:
    """Retrieve project passages within a bounded evidence budget."""

    def __init__(self) -> None:
        self._retriever = None
        self._tokenizer = None

    def __call__(
        self,
        query: str,
        top_k: int = 4,
    ) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 6
        ):
            raise ValueError("top_k must be between 1 and 6.")

        query = query.strip()

        # Lazily initialize the local retriever.
        if self._retriever is None:
            from src.rag.retriever import HybridRetriever

            self._retriever = HybridRetriever(
                device="cpu",
                candidate_k=20,
            )

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-8B-AWQ"
            )

        results = self._retriever.search(
            query=query,
            top_k=top_k,
        )

        passages = []
        omitted = 0

        # Process passages in relevance-ranking order.
        for chunk, score in results:
            candidate = {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "heading": chunk.heading,
                "content": chunk.text,
            }

            # Count the complete proposed JSON tool result.
            proposed_result = {
                "query": query,
                "passages": passages + [candidate],
                "omitted_passages": omitted,
            }

            serialized = json.dumps(
                proposed_result,
                ensure_ascii=False,
                allow_nan=False,
            )

            token_count = len(
                self._tokenizer.encode(
                    serialized,
                    add_special_tokens=False,
                )
            )

            if token_count <= MAX_EVIDENCE_TOKENS:
                passages.append(candidate)
            else:
                omitted += 1

        return {
            "query": query,
            "passages": passages,
            "omitted_passages": omitted,
        }
