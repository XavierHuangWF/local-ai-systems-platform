
import re
from dataclasses import dataclass


# Recognize references such as:
# tool-execution.md
# tool-execution.md::0004

SOURCE_PATTERN = re.compile(
    r"(?<![\w./-])"
    r"[A-Za-z0-9_./-]+\.md"
    r"(?:::\d+)?"
    r"(?![\w/:-]|\.[A-Za-z0-9_/-])"
)



@dataclass(frozen=True)
class AttributionResult:
    """Result of comparing source references with retrieved evidence."""

    referenced: list[str]
    matched: list[str]
    unsupported: list[str]


def check_source_attribution(
    answer: str,
    passages: list[dict],
) -> AttributionResult:
    """Check whether source identifiers appear in retrieved passages.

    This checks reference provenance, not factual claim support.
    """

    allowed_files = set()
    allowed_chunks = set()

    for passage in passages:
        source = passage.get("source")
        chunk_id = passage.get("chunk_id")

        if isinstance(source, str) and source:
            allowed_files.add(source)

        if isinstance(chunk_id, str) and chunk_id:
            allowed_chunks.add(chunk_id)

    # Preserve the order in which identifiers appear in the answer.
    referenced = list(
        dict.fromkeys(
            match.group(0)
            for match in SOURCE_PATTERN.finditer(answer)
        )
    )

    matched = []
    unsupported = []

    for reference in referenced:
        if "::" in reference:
            valid = reference in allowed_chunks
        else:
            valid = reference in allowed_files

        if valid:
            matched.append(reference)
        else:
            unsupported.append(reference)

    return AttributionResult(
        referenced=referenced,
        matched=matched,
        unsupported=unsupported,
    )
