
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.chunker import DocumentChunk


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching "
    "relevant passages: "
)


class LocalEmbedder:
    """Generate normalized local text embeddings."""

    def __init__(self, device: str = "cpu") -> None:
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=device,
        )

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> np.ndarray:
        """Encode document chunks into a vector matrix."""

        if not chunks:
            raise ValueError("No document chunks provided.")

        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a search question into one vector."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        instructed_query = QUERY_INSTRUCTION + query.strip()

        embedding = self.model.encode(
            instructed_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding
