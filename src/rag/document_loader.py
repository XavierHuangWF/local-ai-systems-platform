
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "docs"

MAX_DOCUMENT_BYTES = 1_000_000


@dataclass(frozen=True)
class SourceDocument:
    source: str
    text: str


def load_documents(
    directory: Path = DOCUMENTS_DIR,
) -> list[SourceDocument]:
    """Load local Markdown files with source information."""

    directory = Path(directory).resolve()

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Document directory not found: {directory}"
        )

    documents = []

    for path in sorted(directory.rglob("*.md")):
        if not path.is_file() or path.is_symlink():
            continue

        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ValueError(
                f"Document exceeds size limit: {path.name}"
            )

        content = path.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            continue

        documents.append(
            SourceDocument(
                source=path.relative_to(directory).as_posix(),
                text=content,
            )
        )

    if not documents:
        raise ValueError("No Markdown documents were found.")

    return documents
