
import numpy as np
from transformers import AutoTokenizer

from src.rag.document_loader import load_documents
from src.rag.chunker import DocumentChunk, chunk_documents
from src.rag.bm25 import BM25Index
from src.rag.embeddings import LocalEmbedder
from src.rag.fusion import reciprocal_rank_fusion


class HybridRetriever:
    """Retrieve relevant document chunks using BM25, embeddings, and RRF."""

    def __init__(
        self,
        device: str = "cpu",
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")

        self.candidate_k = candidate_k

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B-AWQ"
        )

        documents = load_documents()
        self.chunks = chunk_documents(documents, tokenizer)

        self.bm25 = BM25Index(self.chunks)
        self.embedder = LocalEmbedder(device=device)

        # Calculate document embeddings only once.
        self.document_vectors = self.embedder.embed_chunks(
            self.chunks
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not 1 <= top_k <= self.candidate_k:
            raise ValueError(
                "top_k must be between 1 and candidate_k."
            )

        query = query.strip()

        # Keyword-based retrieval.
        keyword_results = self.bm25.search(
            query,
            top_k=self.candidate_k,
        )

        # Embedding-based semantic retrieval.
        query_vector = self.embedder.embed_query(query)

        similarities = (
            self.document_vectors @ query_vector
        )

        indices = np.argsort(
            -similarities
        )[:self.candidate_k]

        semantic_results = [
            (self.chunks[int(i)], float(similarities[i]))
            for i in indices
        ]

        # Combine both rankings.
        return reciprocal_rank_fusion(
            rankings=[keyword_results, semantic_results],
            top_k=top_k,
            k=60,
        )
