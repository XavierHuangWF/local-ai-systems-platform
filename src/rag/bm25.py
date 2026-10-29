
import math
import re
from collections import Counter

from src.rag.chunker import DocumentChunk


def tokenize(text: str) -> list[str]:
    """Normalize text into searchable terms."""

    # Separate CamelCase identifiers such as ToolExecutor.
    text = re.sub(
        r"([A-Z])([A-Z][a-z])",
        r"\1 \2",
        text,
    )
    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text,
    )

    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Index:
    """Rank document chunks using BM25 keyword relevance."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("Cannot index empty chunks.")

        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters.")

        self.chunks = chunks
        self.k1 = k1
        self.b = b

        # Count term frequency inside each chunk.
        self.term_frequencies = [
            Counter(tokenize(chunk.text))
            for chunk in chunks
        ]

        self.document_lengths = [
            sum(frequencies.values())
            for frequencies in self.term_frequencies
        ]

        self.average_length = (
            sum(self.document_lengths) / len(chunks)
        )

        # Count how many chunks contain each term.
        self.document_frequencies = Counter()

        for frequencies in self.term_frequencies:
            self.document_frequencies.update(
                frequencies.keys()
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return the highest-scoring matching chunks."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_terms = set(tokenize(query))
        document_count = len(self.chunks)

        results = []

        for index, frequencies in enumerate(
            self.term_frequencies
        ):
            score = 0.0
            document_length = self.document_lengths[index]

            for term in query_terms:
                term_frequency = frequencies.get(term, 0)

                if term_frequency == 0:
                    continue

                document_frequency = (
                    self.document_frequencies[term]
                )

                # Inverse document frequency.
                idf = math.log(
                    1 + (
                        document_count
                        - document_frequency
                        + 0.5
                    ) / (document_frequency + 0.5)
                )

                # Length-normalized term frequency.
                denominator = (
                    term_frequency
                    + self.k1 * (
                        1 - self.b
                        + self.b
                        * document_length
                        / self.average_length
                    )
                )

                score += idf * (
                    term_frequency * (self.k1 + 1)
                    / denominator
                )

            if score > 0:
                results.append(
                    (self.chunks[index], score)
                )

        return sorted(
            results,
            key=lambda result: (
                -result[1],
                result[0].chunk_id,
            ),
        )[:top_k]
