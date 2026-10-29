
class DocumentSearchTool:
    """Expose hybrid document retrieval as a callable agent tool."""

    def __init__(self) -> None:
        # Initialize only when the tool is first called.
        self._retriever = None

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

        # Load the document index once, then reuse it.
        if self._retriever is None:
            from src.rag.retriever import HybridRetriever

            self._retriever = HybridRetriever(
                device="cpu",
                candidate_k=20,
            )

        query = query.strip()

        # Execute hybrid retrieval.
        results = self._retriever.search(
            query=query,
            top_k=top_k,
        )

        # Return JSON-serializable evidence for the agent.
        passages = []

        for chunk, score in results:
            passages.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "heading": chunk.heading,
                    "content": chunk.text,
                }
            )

        return {
            "query": query,
            "passages": passages,
        }
