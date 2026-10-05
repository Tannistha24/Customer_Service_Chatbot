import re
from dataclasses import dataclass
from datetime import datetime, date as date_cls
from typing import Any, Dict, List, Optional, Tuple

# Fields we care about for reconciliation. Extend as needed without
# touching Steps 1-4 — this list only drives Step 5's comparison.
KEY_FIELDS = ["order_id", "amount", "date"]

# tokens is treated as "not provided" — never guessed at, never invented.
_MISSING_TOKENS = {
    "", "null", "none", "nil", "unknown", "n/a", "na", "not provided",
    "not available", "not applicable", "-", "--", "tbd", "?",
}


def _is_missing(value: Any) -> bool:
    """True if value represents an absent/unknown field."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_TOKENS
    return False


def _safe_str(value: Any) -> str:
    """str() that can never raise, for defensive formatting of odd types."""
    try:
        return str(value)
    except Exception:
        return ""


# ---------------------------------------------------------------------
# Currency handling
# ---------------------------------------------------------------------
# Symbol -> ISO code. Extend as needed.
_CURRENCY_SYMBOLS = {
    "$": "USD", "₹": "INR", "€": "EUR", "£": "GBP", "¥": "JPY",
}
_CURRENCY_CODES = {"USD", "INR", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY"}

# Set to True to allow "$100" and "100" (no currency stated) to be treated
# as equivalent when their numeric values match. Two *explicit* but
# *different* currencies (e.g. "$100" vs "€100") are always a mismatch —
# this flag only governs the case where currency is absent on one side.
_ALLOW_AMBIGUOUS_CURRENCY_MATCH = True


def _parse_amount(raw: Any) -> Optional[Tuple[Optional[str], float]]:
    """
    Parse an amount into (currency_code_or_None, numeric_value).
    Returns None if the value can't be safely parsed as a number
    (never guesses — caller falls back to a plain string comparison).
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return None, float(raw)
    if not isinstance(raw, str):
        return None  # unexpected type (list/dict/etc.) — don't guess

    s = raw.strip()
    if not s:
        return None

    currency: Optional[str] = None

    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in s:
            currency = code
            s = s.replace(sym, "")
            break

    if currency is None:
        for token in re.findall(r"[A-Za-z]{3,}", s):
            if token.upper() in _CURRENCY_CODES:
                currency = token.upper()
                s = re.sub(re.escape(token), "", s, flags=re.IGNORECASE)
                break

    s = s.replace(",", "").strip()
    if not s:
        return None
    try:
        numeric = float(s)
    except ValueError:
        return None
    return currency, numeric


# ---------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------
_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d",
    "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y",
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%d %B %Y", "%d %b %Y",
]
_ORDINAL_SUFFIX_RE = re.compile(r"(?<=\d)(st|nd|rd|th)\b", re.IGNORECASE)


def _parse_date(raw: Any) -> Optional[str]:
    """
    Parse common date formats into a canonical ISO string (YYYY-MM-DD).
    Returns None if unparseable (never guesses the date).

    Note: numeric formats like "01/02/2026" are inherently ambiguous
    (month/day order). This parser resolves them consistently using the
    order in _DATE_FORMATS (month/day/year first); it does not attempt
    to detect intent beyond that.
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date_cls):
        return raw.isoformat()
    if not isinstance(raw, str):
        return None

    s = re.sub(r"\s+", " ", raw.strip())
    if not s:
        return None
    s = _ORDINAL_SUFFIX_RE.sub("", s)  # "1st" -> "1"

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------
# Order ID handling
# ---------------------------------------------------------------------
def _normalize_order_id(raw: Any) -> Optional[str]:
    """
    Canonicalize an order ID: trim, collapse internal whitespace, and
    upper-case. This is a formatting-only normalization — it never
    changes the identifying characters of the ID.
    """
    if _is_missing(raw):
        return None
    s = raw if isinstance(raw, str) else _safe_str(raw)
    collapsed = re.sub(r"\s+", " ", s.strip())
    return collapsed.upper()


# ---------------------------------------------------------------------
# Legacy / generic normalization (fallback path)
# ---------------------------------------------------------------------
def _legacy_normalize(value: Any) -> Optional[str]:
    """Original generic normalizer, used as a safe fallback when a
    field-specific parse fails (malformed input) or for fields outside
    KEY_FIELDS. Never invents data — only reformats what's present."""
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.2f}"
    s = _safe_str(value).strip()
    cleaned = s
    for sym in _CURRENCY_SYMBOLS:
        cleaned = cleaned.replace(sym, "")
    cleaned = cleaned.replace(",", "").strip()
    try:
        return f"{float(cleaned):.2f}"
    except ValueError:
        return s.lower()


def _normalize(value: Any, field: Optional[str] = None) -> Optional[str]:
    """
    Normalize a value for comparison. Preserved as the module's public
    normalization entry point.

    - field=None (default): original generic behavior (numeric-aware,
      currency-symbol-stripping, case-insensitive fallback). Kept for
      backward compatibility with existing callers.
    - field="order_id" / "amount" / "date": applies the field-aware
      rule for that field (see _normalize_order_id / _parse_amount /
      _parse_date). Falls back to the generic rule if field-aware
      parsing fails on malformed input.

    Never invents data — returns None for missing/blank/unknown values.
    """
    if _is_missing(value):
        return None

    if field == "order_id":
        return _normalize_order_id(value)

    if field == "amount":
        parsed = _parse_amount(value)
        if parsed is None:
            return _legacy_normalize(value)
        currency, numeric = parsed
        return f"{currency or 'UNSPEC'}:{numeric:.2f}"

    if field == "date":
        parsed = _parse_date(value)
        return parsed if parsed is not None else _legacy_normalize(value)

    return _legacy_normalize(value)


# ---------------------------------------------------------------------
# Field comparison
# ---------------------------------------------------------------------
@dataclass
class _FieldResult:
    status: str  # "match" | "mismatch" | "missing"
    missing_in: Optional[str] = None          # "document" | "message" | "both"
    reason: Optional[str] = None              # "value" | "currency" | "malformed_format"


def _compare_generic(doc_raw: Any, msg_raw: Any, malformed: bool = False) -> _FieldResult:
    d = _legacy_normalize(doc_raw)
    m = _legacy_normalize(msg_raw)
    if d is None and m is None:
        return _FieldResult(status="missing", missing_in="both")
    if d is None:
        return _FieldResult(status="missing", missing_in="document")
    if m is None:
        return _FieldResult(status="missing", missing_in="message")
    if d == m:
        return _FieldResult(status="match")
    return _FieldResult(status="mismatch", reason="malformed_format" if malformed else "value")


def _compare_order_id(doc_raw: Any, msg_raw: Any) -> _FieldResult:
    d = _normalize_order_id(doc_raw)
    m = _normalize_order_id(msg_raw)
    if d == m:
        return _FieldResult(status="match")
    return _FieldResult(status="mismatch", reason="value")


def _compare_amount(doc_raw: Any, msg_raw: Any) -> _FieldResult:
    d = _parse_amount(doc_raw)
    m = _parse_amount(msg_raw)
    if d is None or m is None:
        # Unparseable amount — don't guess; fall back to a plain,
        # format-agnostic string comparison so we still catch obvious
        # matches/mismatches without crashing on garbage input.
        return _compare_generic(doc_raw, msg_raw, malformed=True)

    d_currency, d_numeric = d
    m_currency, m_numeric = m

    if abs(d_numeric - m_numeric) > 1e-9:
        return _FieldResult(status="mismatch", reason="value")

    if d_currency and m_currency and d_currency != m_currency:
        return _FieldResult(status="mismatch", reason="currency")

    if (d_currency is None or m_currency is None) and not _ALLOW_AMBIGUOUS_CURRENCY_MATCH:
        return _FieldResult(status="mismatch", reason="currency")

    return _FieldResult(status="match")


def _compare_date(doc_raw: Any, msg_raw: Any) -> _FieldResult:
    d = _parse_date(doc_raw)
    m = _parse_date(msg_raw)
    if d is None or m is None:
        return _compare_generic(doc_raw, msg_raw, malformed=True)
    if d == m:
        return _FieldResult(status="match")
    return _FieldResult(status="mismatch", reason="value")


def _compare_field(field: str, doc_raw: Any, msg_raw: Any) -> _FieldResult:
    doc_missing = _is_missing(doc_raw)
    msg_missing = _is_missing(msg_raw)
    if doc_missing and msg_missing:
        return _FieldResult(status="missing", missing_in="both")
    if doc_missing:
        return _FieldResult(status="missing", missing_in="document")
    if msg_missing:
        return _FieldResult(status="missing", missing_in="message")

    try:
        if field == "amount":
            return _compare_amount(doc_raw, msg_raw)
        if field == "date":
            return _compare_date(doc_raw, msg_raw)
        if field == "order_id":
            return _compare_order_id(doc_raw, msg_raw)
        return _compare_generic(doc_raw, msg_raw)
    except Exception:
        # Defensive last resort: never let a single malformed/unexpected
        # value crash the whole reconciliation flow.
        return _compare_generic(doc_raw, msg_raw, malformed=True)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def cross_verify(document_fields: Optional[Dict[str, Any]],
                  message_fields: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare document_fields vs message_fields for KEY_FIELDS.

    Returns a report dict:
        {
          "matched": [field, ...],
          "mismatched": [
              {"field", "document_value", "message_value", "reason"}, ...
          ],
          "missing": [{"field", "missing_in"}, ...],
          "status": "ok" | "needs_clarification"
        }

    "reason" on a mismatch entry is one of "value", "currency", or
    "malformed_format", and is additive/optional — existing consumers
    that only read "field"/"document_value"/"message_value" are
    unaffected.

    Inputs that are missing, None, or not actually dicts are treated as
    empty (no fields available) rather than raising, so a malformed
    upstream call can never crash Step 5.

    Never fabricates values: absent/unclear fields are reported as
    missing, not guessed.
    """
    doc = document_fields if isinstance(document_fields, dict) else {}
    msg = message_fields if isinstance(message_fields, dict) else {}

    matched: List[str] = []
    mismatched: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []

    for field in KEY_FIELDS:
        doc_raw = doc.get(field)
        msg_raw = msg.get(field)

        result = _compare_field(field, doc_raw, msg_raw)

        if result.status == "match":
            matched.append(field)
        elif result.status == "mismatch":
            entry: Dict[str, Any] = {
                "field": field,
                "document_value": doc_raw,
                "message_value": msg_raw,
            }
            if result.reason:
                entry["reason"] = result.reason
            mismatched.append(entry)
        else:  # missing
            missing.append({"field": field, "missing_in": result.missing_in or "both"})

    status = "ok" if not mismatched and not missing else "needs_clarification"
    return {
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing,
        "status": status,
    }


_REASON_PHRASING = {
    "currency": "the currency doesn't match",
    "malformed_format": "the values don't line up",
}


def format_clarification_request(report: Dict[str, Any]) -> Optional[str]:
    """
    Build a concise, customer-facing message explaining mismatches and
    missing fields, explicitly stating what needs to be confirmed or
    provided. Returns None if nothing needs clarification.
    """
    if report.get("status") == "ok":
        return None

    lines: List[str] = []

    if report.get("mismatched"):
        lines.append("A couple of details don't match between your message and the document:")
        for m in report["mismatched"]:
            label = m["field"].replace("_", " ").title()
            note = _REASON_PHRASING.get(m.get("reason"))
            suffix = f" ({note})" if note else ""
            lines.append(
                f"  - {label}: you mentioned \"{m['message_value']}\", but the document shows "
                f"\"{m['document_value']}\"{suffix}. Please confirm which is correct."
            )

    if report.get("missing"):
        if lines:
            lines.append("")
        lines.append("A few details are missing or unclear:")
        where_map = {
            "document": "the document",
            "message": "your message",
            "both": "both your message and the document",
        }
        for m in report["missing"]:
            label = m["field"].replace("_", " ").title()
            where = where_map.get(m["missing_in"], "the information provided")
            lines.append(f"  - {label} is missing from {where}. Could you please provide it?")

    lines.append("")
    lines.append("Please confirm or correct the above so I can proceed.")
    return "\n".join(lines)


def run_step5(document_fields: Optional[Dict[str, Any]],
              message_fields: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Entry point for the dialogue flow. Call this after Steps 1-4 have
    produced document_fields and message_fields.

    Returns:
        {
          "report": <cross_verify report>,
          "clarification_message": str | None,
          "can_proceed": bool
        }
    """
    report = cross_verify(document_fields, message_fields)
    clarification = format_clarification_request(report)
    return {
        "report": report,
        "clarification_message": clarification,
        "can_proceed": report["status"] == "ok",
    }

