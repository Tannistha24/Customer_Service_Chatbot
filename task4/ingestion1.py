
from __future__ import annotations

import io
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, BinaryIO, Dict, List, Optional, Union

import markdown as _markdown_lib
import pypdf
from bs4 import BeautifulSoup


class MetadataValidationError(ValueError):
    """Raised when a document's metadata is missing required fields or
    contains invalid values. Carries the full list of problems found
    (rather than stopping at the first one) so all issues can be reported
    at once instead of requiring a fix-and-retry loop.
    """

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class DocumentParsingError(ValueError):
    """Raised when a source document cannot be parsed into text, or when
    an unsupported source format is requested."""


class ChunkingError(ValueError):
    """Raised when chunking configuration is invalid."""


# ---------------------------------------------------------------------------
# Metadata schema and validation
# ---------------------------------------------------------------------------

REQUIRED_METADATA_FIELDS: List[str] = [
    "document_id",
    "version",
    "product",
    "region",
    "access_level",
    "effective_date",
]
# expiry_date is intentionally not in this list: it is required to be
# *present as a field* on every DocumentMetadata, but its value is allowed
# to be None ("no expiry").

ACCESS_LEVELS = frozenset({"public", "internal", "confidential", "admin"})

# SemVer core, with optional pre-release/build metadata, e.g. "1.2.3",
# "1.2.3-beta.1", "1.2.3+build.5".
_SEMVER_RE = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
# A plain monotonic integer version, e.g. "1", "2", "42".
_MONOTONIC_INT_RE = re.compile(r"^\d+$")

SUPPORTED_SOURCE_FORMATS = frozenset({"markdown", "html", "pdf"})


def _is_valid_version(version: str) -> bool:
    return bool(_SEMVER_RE.match(version) or _MONOTONIC_INT_RE.match(version))


def validate_metadata_dict(data: Dict[str, Any]) -> List[str]:
    """
    Validate a plain metadata dict against the required schema.

    Returns a list of human-readable problem descriptions (empty list
    means valid). Never repairs or guesses a value -- only reports what
    is missing or malformed.
    """
    errors: List[str] = []

    for field_name in REQUIRED_METADATA_FIELDS:
        value = data.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"Missing required metadata field: '{field_name}'.")

    access_level = data.get("access_level")
    if access_level is not None and access_level != "" and access_level not in ACCESS_LEVELS:
        errors.append(
            f"Invalid access_level '{access_level}'. Must be one of: {sorted(ACCESS_LEVELS)}."
        )

    version = data.get("version")
    if version is not None and version != "" and not _is_valid_version(str(version)):
        errors.append(
            f"Invalid version '{version}'. Must be SemVer (e.g. '1.2.3') "
            "or a monotonic integer (e.g. '3')."
        )

    effective_date = data.get("effective_date")
    effective_dt: Optional[datetime] = None
    if effective_date is not None and effective_date != "":
        try:
            effective_dt = datetime.fromisoformat(str(effective_date))
        except ValueError:
            errors.append(
                f"Invalid effective_date '{effective_date}'. Must be a valid ISO 8601 date/datetime."
            )

    expiry_date = data.get("expiry_date")
    if expiry_date is not None and expiry_date != "":
        try:
            expiry_dt = datetime.fromisoformat(str(expiry_date))
            if effective_dt is not None and expiry_dt < effective_dt:
                errors.append(
                    f"expiry_date '{expiry_date}' is earlier than effective_date '{effective_date}'."
                )
        except ValueError:
            errors.append(
                f"Invalid expiry_date '{expiry_date}'. Must be a valid ISO 8601 date/datetime, or null."
            )

    return errors


@dataclass(frozen=True)
class DocumentMetadata:
    """
    Structured, governance-relevant metadata for one document version.

    All seven fields are always present on the object; `expiry_date` may
    be `None` to mean "does not expire." Validation runs automatically on
    construction (`__post_init__`), so it is impossible to end up with a
    `DocumentMetadata` instance that doesn't satisfy the schema -- invalid
    input raises `MetadataValidationError` immediately instead of being
    silently accepted or repaired.
    """

    document_id: str
    version: str
    product: str
    region: str
    access_level: str
    effective_date: str  # ISO 8601
    expiry_date: Optional[str] = None  # ISO 8601, or None

    def __post_init__(self) -> None:
        errors = validate_metadata_dict(asdict(self))
        if errors:
            raise MetadataValidationError(errors)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Document:
    """A parsed document: its governance metadata, its source format, and
    the plain-text content extracted from the original source."""

    metadata: DocumentMetadata
    source_format: str
    text_content: str


# ---------------------------------------------------------------------------
# Parsing (Markdown / HTML / PDF -> plain text)
# ---------------------------------------------------------------------------

def parse_markdown_to_text(raw_markdown: str) -> str:
    """Convert Markdown source into plain text (headings, emphasis, links,
    lists, etc. are all resolved down to their readable text)."""
    try:
        html = _markdown_lib.markdown(raw_markdown, extensions=["extra"])
        return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
    except Exception as exc:  # pragma: no cover - defensive
        raise DocumentParsingError(f"Failed to parse Markdown content: {exc}") from exc


def parse_html_to_text(raw_html: str) -> str:
    """Strip HTML markup down to plain, readable text."""
    try:
        return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)
    except Exception as exc:  # pragma: no cover - defensive
        raise DocumentParsingError(f"Failed to parse HTML content: {exc}") from exc


def parse_pdf_to_text(source: Union[str, bytes, BinaryIO]) -> str:
    """
    Extract text from a PDF. `source` may be a file path, raw bytes, or an
    already-open binary file-like object.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            reader = pypdf.PdfReader(io.BytesIO(source))
        else:
            reader = pypdf.PdfReader(source)
        page_texts = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(page_texts).strip()
    except Exception as exc:
        raise DocumentParsingError(f"Failed to parse PDF content: {exc}") from exc


def ingest_document(
    source: Union[str, bytes, BinaryIO],
    source_format: str,
    metadata: DocumentMetadata,
) -> Document:
    """
    Parse `source` (per `source_format`) into plain text and pair it with
    `metadata` to produce a `Document`.

    `metadata` must already be a valid `DocumentMetadata` instance (its
    own constructor validates it) -- this function does not create or
    guess metadata on the caller's behalf.
    """
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise DocumentParsingError(
            f"Unsupported source_format '{source_format}'. "
            f"Supported formats: {sorted(SUPPORTED_SOURCE_FORMATS)}."
        )

    if source_format == "markdown":
        text = parse_markdown_to_text(source)  # type: ignore[arg-type]
    elif source_format == "html":
        text = parse_html_to_text(source)  # type: ignore[arg-type]
    else:  # "pdf"
        text = parse_pdf_to_text(source)

    if not text or not text.strip():
        raise DocumentParsingError(
            f"No textual content could be extracted from the {source_format} "
            f"document '{metadata.document_id}'."
        )

    return Document(metadata=metadata, source_format=source_format, text_content=text)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentChunk:
    """
    One RAG-ready chunk of a document's text, always carrying the full
    lineage/governance metadata of the document version it came from, so
    it can be traced back to its source at any later pipeline step.
    """

    chunk_id: str
    document_id: str
    version: str
    product: str
    region: str
    access_level: str
    effective_date: str
    expiry_date: Optional[str]
    chunk_index: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chunk_document(
    document: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    """
    Split `document.text_content` into overlapping, word-based chunks.

    - `chunk_size`: max number of words per chunk.
    - `chunk_overlap`: number of words repeated between consecutive chunks
      (must be smaller than `chunk_size`).

    Deterministic: the same document content + the same chunk_size/overlap
    always produce the same chunks, in the same order, with the same
    chunk_ids (chunk_id is derived from document_id + version + index --
    no randomness, no wall-clock timestamps).
    """
    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ChunkingError("chunk_overlap must be >= 0.")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("chunk_overlap must be smaller than chunk_size.")

    words = document.text_content.split()
    if not words:
        return []

    metadata = document.metadata
    step = chunk_size - chunk_overlap
    chunks: List[DocumentChunk] = []

    index = 0
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunk_id = f"{metadata.document_id}::v{metadata.version}::chunk{index}"

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=metadata.document_id,
                version=metadata.version,
                product=metadata.product,
                region=metadata.region,
                access_level=metadata.access_level,
                effective_date=metadata.effective_date,
                expiry_date=metadata.expiry_date,
                chunk_index=index,
                text=chunk_text,
            )
        )

        if end == len(words):
            break
        start += step
        index += 1

    return chunks


if __name__ == "__main__":
    meta = DocumentMetadata(
        document_id="DOC-100",
        version="1.2.0",
        product="widgets",
        region="us",
        access_level="internal",
        effective_date="2024-01-01",
        expiry_date=None,
    )
    doc = ingest_document(
        "# Widget Setup\n\nConnect the widget. **Restart** the device.",
        "markdown",
        meta,
    )
    print("Extracted text:", doc.text_content)
    for c in chunk_document(doc, chunk_size=4, chunk_overlap=1):
        print(c.chunk_id, "->", c.text)