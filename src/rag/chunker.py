
import re
from dataclasses import dataclass
from typing import Any

from src.rag.document_loader import SourceDocument


MAX_CHUNK_TOKENS = 400
OVERLAP_TOKENS = 60

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_PATTERN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    heading: str
    text: str
    token_count: int
    over_budget: bool


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown by headings outside fenced code blocks."""

    sections = []
    heading_stack = []
    current_lines = []
    current_heading = "Document introduction"
    fence = None

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))

    for line in text.splitlines():
        marker = FENCE_PATTERN.match(line)

        if marker:
            symbol = marker.group(1)

            if fence is None:
                fence = symbol
            elif (
                symbol[0] == fence[0]
                and len(symbol) >= len(fence)
            ):
                fence = None

        match = HEADING_PATTERN.match(line)

        if match and fence is None and marker is None:
            flush()
            current_lines = []

            level = len(match.group(1))
            title = match.group(2).strip()

            heading_stack = heading_stack[:level - 1]
            heading_stack.append(title)

            current_heading = " > ".join(heading_stack)
        else:
            current_lines.append(line)

    flush()
    return sections


def split_blocks(text: str) -> list[str]:
    """Separate paragraphs while preserving fenced code blocks."""

    blocks = []
    current = []
    fence = None

    def flush():
        block = "\n".join(current).strip()
        if block:
            blocks.append(block)

    for line in text.splitlines():
        marker = FENCE_PATTERN.match(line)

        if marker:
            symbol = marker.group(1)

            if fence is None:
                flush()
                current = []
                fence = symbol
                current.append(line)
                continue

            if (
                symbol[0] == fence[0]
                and len(symbol) >= len(fence)
            ):
                current.append(line)
                flush()
                current = []
                fence = None
                continue

        if not line.strip() and fence is None:
            flush()
            current = []
        else:
            current.append(line)

    flush()
    return blocks


def chunk_documents(
    documents: list[SourceDocument],
    tokenizer: Any,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[DocumentChunk]:
    """Create heading-aware, token-counted Markdown chunks."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("Invalid overlap_tokens.")

    def count(text: str) -> int:
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    chunks = []

    for document in documents:
        chunk_number = 0

        for heading, section_text in split_sections(document.text):
            blocks = split_blocks(section_text)
            current = []

            prefix = f"Section: {heading}\n\n"

            def create_chunk(selected_blocks):
                nonlocal chunk_number

                if not selected_blocks:
                    return

                text = prefix + "\n\n".join(selected_blocks)
                token_count = count(text)
                chunk_number += 1

                chunks.append(
                    DocumentChunk(
                        chunk_id=(
                            f"{document.source}::"
                            f"{chunk_number:04d}"
                        ),
                        source=document.source,
                        heading=heading,
                        text=text,
                        token_count=token_count,
                        over_budget=token_count > max_tokens,
                    )
                )

            for block in blocks:
                candidate = current + [block]

                if current and count(
                    prefix + "\n\n".join(candidate)
                ) > max_tokens:
                    create_chunk(current)

                    # Reuse complete trailing blocks for overlap.
                    overlap = []

                    for previous in reversed(current):
                        proposed = [previous] + overlap

                        if count("\n\n".join(proposed)) > overlap_tokens:
                            break

                        overlap = proposed

                    # Do not let overlap prevent forward progress.
                    if count(
                        prefix + "\n\n".join(overlap + [block])
                    ) <= max_tokens:
                        current = overlap
                    else:
                        current = []

                current.append(block)

            create_chunk(current)

    return chunks
