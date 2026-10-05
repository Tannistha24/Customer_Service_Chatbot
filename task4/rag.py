
# from dataclasses import dataclass
# from datetime import datetime, date, timezone
# from enum import Enum
# from typing import List, Optional, Union


# # ---------------------------------------------------------------------------
# # Compatibility stub for Step 1's DocumentChunk (see module docstring above).
# # ---------------------------------------------------------------------------
# @dataclass
# class DocumentChunk:
#     chunk_id: str
#     text: str
#     access_level: str
#     effective_date: str
#     expiry_date: Optional[str] = None


# # ---------------------------------------------------------------------------
# # Access levels
# # ---------------------------------------------------------------------------
# class AccessLevel(Enum):
#     PUBLIC = "public"
#     INTERNAL = "internal"
#     CONFIDENTIAL = "confidential"
#     ADMIN = "admin"


# # Hierarchy: higher number = more access.
# _ACCESS_RANK = {
#     AccessLevel.PUBLIC: 0,
#     AccessLevel.INTERNAL: 1,
#     AccessLevel.CONFIDENTIAL: 2,
#     AccessLevel.ADMIN: 3,
# }


# class InvalidAccessLevelError(ValueError):
#     """Raised when a user or document access level string is not recognized."""


# class InvalidDateError(ValueError):
#     """Raised when a date/datetime string or value cannot be parsed cleanly."""


# def _parse_access_level(value: str) -> AccessLevel:
#     if not isinstance(value, str):
#         raise InvalidAccessLevelError(f"Access level must be a string, got {type(value)!r}")
#     try:
#         return AccessLevel(value.strip().lower())
#     except ValueError:
#         raise InvalidAccessLevelError(
#             f"Unknown access level: {value!r}. "
#             f"Valid levels are: {[lvl.value for lvl in AccessLevel]}"
#         )


# def _has_access(user_level: AccessLevel, document_level: AccessLevel) -> bool:
#     """user_access_level >= document_access_level"""
#     return _ACCESS_RANK[user_level] >= _ACCESS_RANK[document_level]


# # ---------------------------------------------------------------------------
# # Date handling
# # ---------------------------------------------------------------------------
# def _parse_date(value: Union[str, date, datetime]) -> date:
#     """
#     Parse an ISO 8601 date/datetime string (or accept an existing
#     date/datetime object) and return a timezone-naive `date` for
#     comparison purposes.

#     Invalid input raises InvalidDateError rather than being silently
#     "repaired" or guessed at.
#     """
#     if isinstance(value, datetime):
#         return value.date()
#     if isinstance(value, date):
#         return value
#     if not isinstance(value, str) or not value.strip():
#         raise InvalidDateError(f"Date must be a non-empty ISO 8601 string, got {value!r}")

#     text = value.strip()
#     # Support "Z" suffix (Zulu/UTC) which Python's fromisoformat rejects
#     # on older versions.
#     normalized = text.replace("Z", "+00:00")

#     try:
#         if "T" in normalized:
#             parsed = datetime.fromisoformat(normalized)
#             if parsed.tzinfo is not None:
#                 parsed = parsed.astimezone(timezone.utc)
#             return parsed.date()
#         return date.fromisoformat(normalized)
#     except ValueError as exc:
#         raise InvalidDateError(f"Invalid ISO 8601 date: {value!r}") from exc


# # ---------------------------------------------------------------------------
# # Rejection reasons
# # ---------------------------------------------------------------------------
# class RejectionReason(Enum):
#     UNAUTHORIZED = "unauthorized"
#     FUTURE = "future"
#     EXPIRED = "expired"
#     NOT_ACTIVE_ON_HISTORICAL_DATE = "not_active_on_historical_date"
#     INVALID_METADATA = "invalid_metadata"


# @dataclass
# class RejectedChunk:
#     chunk: DocumentChunk
#     reason: RejectionReason
#     detail: str = ""


# @dataclass
# class FilterResult:
#     allowed: List[DocumentChunk]
#     rejected: List[RejectedChunk]

#     def allowed_ids(self) -> List[str]:
#         return [c.chunk_id for c in self.allowed]

#     def rejected_ids_with_reasons(self) -> List[tuple]:
#         return [(r.chunk.chunk_id, r.reason.value, r.detail) for r in self.rejected]


# # ---------------------------------------------------------------------------
# # Query specification: current vs. historical
# # ---------------------------------------------------------------------------
# @dataclass
# class TemporalQuery:
#     """
#     Describes *when* the caller is asking about.

#     - current query:    TemporalQuery(current_date="2026-09-15")
#     - historical query: TemporalQuery(current_date="2026-09-15",
#                                        historical_date="2024-01-01")

#     `current_date` must always be supplied explicitly by the caller
#     (never read from the machine clock inside this module), which
#     keeps filtering deterministic and testable.
#     """

#     current_date: Union[str, date, datetime]
#     historical_date: Optional[Union[str, date, datetime]] = None

#     @property
#     def is_historical(self) -> bool:
#         return self.historical_date is not None

#     def reference_date(self) -> date:
#         """The date to filter against: historical date if given, else current date."""
#         raw = self.historical_date if self.is_historical else self.current_date
#         return _parse_date(raw)


# # ---------------------------------------------------------------------------
# # Core filtering logic
# # ---------------------------------------------------------------------------
# def _check_temporal_validity(chunk: DocumentChunk, reference: date, is_historical: bool):
#     """
#     Returns None if the chunk is temporally valid on `reference`,
#     otherwise returns a RejectedChunk describing why not.
#     """
#     try:
#         effective = _parse_date(chunk.effective_date)
#         expiry = _parse_date(chunk.expiry_date) if chunk.expiry_date else None
#     except InvalidDateError as exc:
#         return RejectedChunk(chunk, RejectionReason.INVALID_METADATA, str(exc))

#     if effective > reference:
#         reason = (
#             RejectionReason.NOT_ACTIVE_ON_HISTORICAL_DATE
#             if is_historical
#             else RejectionReason.FUTURE
#         )
#         return RejectedChunk(
#             chunk, reason,
#             f"effective_date {effective.isoformat()} is after reference date {reference.isoformat()}"
#         )

#     if expiry is not None and reference > expiry:
#         reason = (
#             RejectionReason.NOT_ACTIVE_ON_HISTORICAL_DATE
#             if is_historical
#             else RejectionReason.EXPIRED
#         )
#         return RejectedChunk(
#             chunk, reason,
#             f"expiry_date {expiry.isoformat()} is before reference date {reference.isoformat()}"
#         )

#     return None


# def filter_chunks(
#     user_access_level: str,
#     chunks: List[DocumentChunk],
#     temporal_query: TemporalQuery,
# ) -> FilterResult:
#     """
#     The main pre-retrieval filtering API.

#     Parameters
#     ----------
#     user_access_level : str
#         One of "public", "internal", "confidential", "admin".
#     chunks : List[DocumentChunk]
#         Candidate chunks produced by Step 1, BEFORE any retrieval/
#         embedding step has touched them.
#     temporal_query : TemporalQuery
#         Carries the deterministic current_date, and optionally a
#         historical_date for "what was true on date X" questions.

#     Returns
#     -------
#     FilterResult
#         `allowed` contains only chunks the user is authorized to see
#         AND that are temporally valid for the query. `rejected`
#         contains every excluded chunk plus the reason, for debugging.

#     Notes
#     -----
#     - RBAC is checked strictly with `user_rank >= document_rank`.
#     - Unknown access levels raise InvalidAccessLevelError immediately
#       (this is a hard programming/config error, not a per-chunk
#       rejection, since it means the caller itself is misconfigured).
#     - Bad per-chunk metadata (e.g. an invalid document access level or
#       an unparsable date) results in that single chunk being rejected
#       with reason INVALID_METADATA rather than raising, so one bad
#       chunk cannot break filtering for the rest of the corpus.
#     """
#     # Fail fast and loud on a bad *user* access level - this is a caller bug.
#     user_level = _parse_access_level(user_access_level)

#     reference = temporal_query.reference_date()
#     is_historical = temporal_query.is_historical

#     allowed: List[DocumentChunk] = []
#     rejected: List[RejectedChunk] = []

#     for chunk in chunks:
#         # --- RBAC check first: never let unauthorized content leak
#         # further, even into the "why was this rejected" temporal logic. ---
#         try:
#             document_level = _parse_access_level(chunk.access_level)
#         except InvalidAccessLevelError as exc:
#             rejected.append(RejectedChunk(chunk, RejectionReason.INVALID_METADATA, str(exc)))
#             continue

#         if not _has_access(user_level, document_level):
#             rejected.append(
#                 RejectedChunk(
#                     chunk, RejectionReason.UNAUTHORIZED,
#                     f"user level '{user_level.value}' cannot access "
#                     f"document level '{document_level.value}'"
#                 )
#             )
#             continue

#         # --- Temporal check second. ---
#         temporal_rejection = _check_temporal_validity(chunk, reference, is_historical)
#         if temporal_rejection is not None:
#             rejected.append(temporal_rejection)
#             continue

#         allowed.append(chunk)

#     return FilterResult(allowed=allowed, rejected=rejected)



from dataclasses import dataclass
from datetime import datetime, date, timezone
from enum import Enum
from typing import List, Optional, Union
import csv


# ---------------------------------------------------------------------------
# Compatibility stub for Step 1's DocumentChunk (see module docstring above).
# ---------------------------------------------------------------------------
@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    access_level: str
    effective_date: str
    expiry_date: Optional[str] = None


# ---------------------------------------------------------------------------
# Access levels
# ---------------------------------------------------------------------------
class AccessLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    ADMIN = "admin"


# Hierarchy: higher number = more access.
_ACCESS_RANK = {
    AccessLevel.PUBLIC: 0,
    AccessLevel.INTERNAL: 1,
    AccessLevel.CONFIDENTIAL: 2,
    AccessLevel.ADMIN: 3,
}


class InvalidAccessLevelError(ValueError):
    """Raised when a user or document access level string is not recognized."""


class InvalidDateError(ValueError):
    """Raised when a date/datetime string or value cannot be parsed cleanly."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the CSV dataset file cannot be found or read."""


class MissingCsvHeaderError(ValueError):
    """Raised when the CSV dataset file parses but has no header row."""


def _parse_access_level(value: str) -> AccessLevel:
    if not isinstance(value, str):
        raise InvalidAccessLevelError(f"Access level must be a string, got {type(value)!r}")
    try:
        return AccessLevel(value.strip().lower())
    except ValueError:
        raise InvalidAccessLevelError(
            f"Unknown access level: {value!r}. "
            f"Valid levels are: {[lvl.value for lvl in AccessLevel]}"
        )


def _has_access(user_level: AccessLevel, document_level: AccessLevel) -> bool:
    """user_access_level >= document_access_level"""
    return _ACCESS_RANK[user_level] >= _ACCESS_RANK[document_level]


# ---------------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------------
def _parse_date(value: Union[str, date, datetime]) -> date:
    """
    Parse an ISO 8601 date/datetime string (or accept an existing
    date/datetime object) and return a timezone-naive `date` for
    comparison purposes.

    Invalid input raises InvalidDateError rather than being silently
    "repaired" or guessed at.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        raise InvalidDateError(f"Date must be a non-empty ISO 8601 string, got {value!r}")

    text = value.strip()
    # Support "Z" suffix (Zulu/UTC) which Python's fromisoformat rejects
    # on older versions.
    normalized = text.replace("Z", "+00:00")

    try:
        if "T" in normalized:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc)
            return parsed.date()
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise InvalidDateError(f"Invalid ISO 8601 date: {value!r}") from exc


# ---------------------------------------------------------------------------
# Rejection reasons
# ---------------------------------------------------------------------------
class RejectionReason(Enum):
    UNAUTHORIZED = "unauthorized"
    FUTURE = "future"
    EXPIRED = "expired"
    NOT_ACTIVE_ON_HISTORICAL_DATE = "not_active_on_historical_date"
    INVALID_METADATA = "invalid_metadata"


@dataclass
class RejectedChunk:
    chunk: DocumentChunk
    reason: RejectionReason
    detail: str = ""


@dataclass
class FilterResult:
    allowed: List[DocumentChunk]
    rejected: List[RejectedChunk]

    def allowed_ids(self) -> List[str]:
        return [c.chunk_id for c in self.allowed]

    def rejected_ids_with_reasons(self) -> List[tuple]:
        return [(r.chunk.chunk_id, r.reason.value, r.detail) for r in self.rejected]


# ---------------------------------------------------------------------------
# Query specification: current vs. historical
# ---------------------------------------------------------------------------
@dataclass
class TemporalQuery:
    """
    Describes *when* the caller is asking about.

    - current query:    TemporalQuery(current_date="2026-09-15")
    - historical query: TemporalQuery(current_date="2026-09-15",
                                       historical_date="2024-01-01")

    `current_date` must always be supplied explicitly by the caller
    (never read from the machine clock inside this module), which
    keeps filtering deterministic and testable.
    """

    current_date: Union[str, date, datetime]
    historical_date: Optional[Union[str, date, datetime]] = None

    @property
    def is_historical(self) -> bool:
        return self.historical_date is not None

    def reference_date(self) -> date:
        """The date to filter against: historical date if given, else current date."""
        raw = self.historical_date if self.is_historical else self.current_date
        return _parse_date(raw)


# ---------------------------------------------------------------------------
# CSV dataset loading -> DocumentChunk
# ---------------------------------------------------------------------------
@dataclass
class CsvLoadOptions:
    """
    Column-name mapping and defaults for `load_chunks_from_csv`.

    Attributes
    ----------
    prompt_column / response_column:
        CSV columns holding the question and answer text. Defaults
        mirror the reader defaults ("prompt" / "response" with the
        "change column" comments) baked into the loader; override
        them only when your headers differ.
    id_column:
        Optional CSV column whose value becomes chunk_id. If absent
        (or blank in a given row), a row-index-based id is used.
    access_column:
        Optional CSV column holding the per-row access level
        ("public", "internal", ...). Falls back to
        `default_access_level` when the column or cell is missing.
    effective_column / expiry_column:
        Optional CSV columns holding ISO dates. `effective_date` has
        no natural fallback (DocumentChunk requires it), so the
        configurable `default_effective_date` is used when the
        column or cell is missing. `expiry_date` is optional and
        simply stays None when absent.
    default_access_level:
        Applied to every row lacking a usable access-level cell.
    default_effective_date:
        Applied to every row lacking a usable effective-date cell.
        Pass an explicit ISO string (e.g. your ingestion date) so
        filtering stays deterministic.
    """

    prompt_column: str = "prompt"      # change "prompt" to your question column
    response_column: str = "response"  # change "response" to your answer column
    id_column: Optional[str] = None
    access_column: Optional[str] = None
    effective_column: Optional[str] = None
    expiry_column: Optional[str] = None
    default_access_level: str = "public"
    default_effective_date: str = "1970-01-01"


def _cell(row: dict, column: Optional[str]) -> str:
    """
    Safe CSV cell read. csv.DictReader yields None for missing columns
    and for short rows, and "" for empty cells; `row.get` on a None
    cell would hand back None unwrapped, so collapse all of those to "".
    """
    if column is None:
        return ""
    return (row.get(column) or "").strip()


def _build_chunk_text(prompt: str, response: str) -> str:
    """Combine the question and answer into the chunk text."""
    if prompt and response:
        return f"Q: {prompt}\nA: {response}"
    # Keep single-sided rows: only one side was filled in.
    return prompt if prompt else response


def load_chunks_from_csv(
    dataset_path: str,
    options: Optional[CsvLoadOptions] = None,
) -> List[DocumentChunk]:
    """
    Load a prompt/response CSV dataset and turn each row into a
    DocumentChunk suitable for `filter_chunks`.

    Rows where BOTH the prompt and the response are empty are skipped.
    Rows are otherwise handed to the filterer as-is: malformed
    access levels or dates are surfaced later per-chunk as
    RejectionReason.INVALID_METADATA rather than raising here, so one
    bad row cannot prevent the rest of the corpus from loading.

    ("utf-8-sig" transparently strips a BOM if the CSV was exported
    from Excel, so headers like "prompt" match cleanly.)

    Raises
    ------
    DatasetNotFoundError
        If the dataset file does not exist or cannot be read.
    MissingCsvHeaderError
        If the file parses but is empty (no header row).
    """
    opts = options or CsvLoadOptions()
    chunks: List[DocumentChunk] = []
    row_index = 0

    try:
        with open(dataset_path, "r", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise MissingCsvHeaderError(
                    f"CSV at {dataset_path!r} is empty: no header row found."
                )

            for row in reader:
                row_index += 1

                prompt = row.get("prompt", "").strip()      # change "prompt" to your question column
                response = row.get("response", "").strip()  # change "response" to your answer column

                # Optional overrides for non-standard column names via
                # CsvLoadOptions (no-ops when left at their defaults,
                # so the plain prompt/response path above is preserved).
                if opts.prompt_column != "prompt":
                    prompt = _cell(row, opts.prompt_column)
                if opts.response_column != "response":
                    response = _cell(row, opts.response_column)

                if not prompt and not response:
                    continue  # nothing usable in this row

                chunk_id = _cell(row, opts.id_column)
                if not chunk_id:
                    chunk_id = f"csv-row-{row_index}"

                access_level = _cell(row, opts.access_column) or opts.default_access_level

                effective_date = _cell(row, opts.effective_column) or opts.default_effective_date

                expiry_date = _cell(row, opts.expiry_column) or None

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=_build_chunk_text(prompt, response),
                        access_level=access_level,
                        effective_date=effective_date,
                        expiry_date=expiry_date,
                    )
                )
    except FileNotFoundError as exc:
        raise DatasetNotFoundError(f"Dataset file not found: {dataset_path!r}") from exc
    except PermissionError as exc:
        raise DatasetNotFoundError(f"Cannot read dataset file: {dataset_path!r}") from exc

    return chunks


# ---------------------------------------------------------------------------
# Core filtering logic
# ---------------------------------------------------------------------------
def _check_temporal_validity(chunk: DocumentChunk, reference: date, is_historical: bool):
    """
    Returns None if the chunk is temporally valid on `reference`,
    otherwise returns a RejectedChunk describing why not.
    """
    try:
        effective = _parse_date(chunk.effective_date)
        expiry = _parse_date(chunk.expiry_date) if chunk.expiry_date else None
    except InvalidDateError as exc:
        return RejectedChunk(chunk, RejectionReason.INVALID_METADATA, str(exc))

    if effective > reference:
        reason = (
            RejectionReason.NOT_ACTIVE_ON_HISTORICAL_DATE
            if is_historical
            else RejectionReason.FUTURE
        )
        return RejectedChunk(
            chunk, reason,
            f"effective_date {effective.isoformat()} is after reference date {reference.isoformat()}"
        )

    if expiry is not None and reference > expiry:
        reason = (
            RejectionReason.NOT_ACTIVE_ON_HISTORICAL_DATE
            if is_historical
            else RejectionReason.EXPIRED
        )
        return RejectedChunk(
            chunk, reason,
            f"expiry_date {expiry.isoformat()} is before reference date {reference.isoformat()}"
        )

    return None


def filter_chunks(
    user_access_level: str,
    chunks: List[DocumentChunk],
    temporal_query: TemporalQuery,
) -> FilterResult:
    """
    The main pre-retrieval filtering API.

    Parameters
    ----------
    user_access_level : str
        One of "public", "internal", "confidential", "admin".
    chunks : List[DocumentChunk]
        Candidate chunks produced by Step 1 (e.g. via
        `load_chunks_from_csv`), BEFORE any retrieval/embedding step
        has touched them.
    temporal_query : TemporalQuery
        Carries the deterministic current_date, and optionally a
        historical_date for "what was true on date X" questions.

    Returns
    -------
    FilterResult
        `allowed` contains only chunks the user is authorized to see
        AND that are temporally valid for the query. `rejected`
        contains every excluded chunk plus the reason, for debugging.

    Notes
    -----
    - RBAC is checked strictly with `user_rank >= document_rank`.
    - Unknown access levels raise InvalidAccessLevelError immediately
      (this is a hard programming/config error, not a per-chunk
      rejection, since it means the caller itself is misconfigured).
    - Bad per-chunk metadata (e.g. an invalid document access level or
      an unparsable date) results in that single chunk being rejected
      with reason INVALID_METADATA rather than raising, so one bad
      chunk cannot break filtering for the rest of the corpus.
    """
    # Fail fast and loud on a bad *user* access level - this is a caller bug.
    user_level = _parse_access_level(user_access_level)

    reference = temporal_query.reference_date()
    is_historical = temporal_query.is_historical

    allowed: List[DocumentChunk] = []
    rejected: List[RejectedChunk] = []

    for chunk in chunks:
        # --- RBAC check first: never let unauthorized content leak
        # further, even into the "why was this rejected" temporal logic. ---
        try:
            document_level = _parse_access_level(chunk.access_level)
        except InvalidAccessLevelError as exc:
            rejected.append(RejectedChunk(chunk, RejectionReason.INVALID_METADATA, str(exc)))
            continue

        if not _has_access(user_level, document_level):
            rejected.append(
                RejectedChunk(
                    chunk, RejectionReason.UNAUTHORIZED,
                    f"user level '{user_level.value}' cannot access "
                    f"document level '{document_level.value}'"
                )
            )
            continue

        # --- Temporal check second. ---
        temporal_rejection = _check_temporal_validity(chunk, reference, is_historical)
        if temporal_rejection is not None:
            rejected.append(temporal_rejection)
            continue

        allowed.append(chunk)

    return FilterResult(allowed=allowed, rejected=rejected)
