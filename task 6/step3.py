from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only when pydantic is absent
    raise ImportError(
        "Task 6 / Step 3 (entity_extraction.py) requires the 'pydantic' package, "
        "which is not installed in this environment.\n"
        "This module intentionally does NOT modify requirements.txt or any other "
        "project dependency file.\n"
        "To use this component, install it explicitly, e.g.:\n"
        "    pip install pydantic\n"
        "(or add it to the project's own dependency file through the normal "
        "project process)."
    ) from exc


# ---------------------------------------------------------------------------
# 1. Pydantic entity schema
# ---------------------------------------------------------------------------

class ExtractedEntities(BaseModel):
    """Structured result of extracting entities from a single message.

    Uses only Pydantic APIs (BaseModel, Field, Optional[...], default_factory)
    that behave identically on Pydantic 1.x and 2.x, so this schema does not
    depend on which major version happens to be installed.
    """

    customer_name: Optional[str] = None
    order_id: Optional[str] = None
    product_code: Optional[str] = None
    dates: List[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.customer_name or self.order_id or self.product_code or self.dates)


# ---------------------------------------------------------------------------
# 2. Deterministic regex patterns for invariant business identifiers
# ---------------------------------------------------------------------------
#
# These patterns are intentionally strict and deterministic. They are the
# *authoritative* source for identifier values -- nothing downstream is
# permitted to re-derive, re-case, translate, or "correct" the matched
# substring. The exact matched text is what is stored and returned.

# Order IDs: "ORD-" followed by digits (e.g. ORD-98412).
ORDER_ID_PATTERN = re.compile(r"\bORD-\d{3,}\b")

# Product / SKU codes: "SKU-" followed by an alphanumeric code
# (e.g. SKU-B12).
PRODUCT_CODE_PATTERN = re.compile(r"\bSKU-[A-Za-z0-9]{2,}\b")

# ISO date: 2026-09-25
_DATE_ISO = r"\d{4}-\d{2}-\d{2}"
# Slash date: 25/09/2026 or 25-09-2026 (day/month/year assumed)
_DATE_SLASH = r"\d{1,2}[/-]\d{1,2}[/-]\d{4}"
# Long-form date: September 25, 2026 / Sep 25 2026
_MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_DATE_LONG = rf"(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}"

DATE_PATTERN = re.compile(
    rf"\b(?:{_DATE_ISO}|{_DATE_SLASH}|{_DATE_LONG})\b",
    re.IGNORECASE,
)

_MONTH_LOOKUP = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

# Customer name: only triggered by an explicit naming cue, followed by
# one to three Capitalized tokens. This deliberately avoids treating
# every capitalized phrase in a message as a name.
#
# The cue itself is matched case-insensitively via the scoped `(?i:...)`
# group below, but the name token class `[A-Z]...` is deliberately left
# outside that scope so it stays case-SENSITIVE: a genuine name must
# actually start with a capital letter. (Applying IGNORECASE to the
# whole pattern, as before, silently defeated this check and let any
# lowercase word following a cue be captured as a "name".)
_NAME_CUE = (
    r"(?:customer name is|my name is|name is|customer:|name:|"
    r"customer is|this is|i am|i'm)"
)
_NAME_TOKEN = r"[A-Z][a-zA-Z'\-]*"
NAME_PATTERN = re.compile(
    rf"(?i:{_NAME_CUE})\s+({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})"
)

# Words that occasionally follow the naming cues but are not names
# (guards against "I am fine", "this is order ...", etc.).
_NAME_STOPWORDS = {
    "fine", "sure", "ready", "here", "not", "sorry", "good", "okay", "ok",
    "the", "order", "sku", "product", "looking", "trying", "calling",
    "writing", "contacting",
}


def normalize_date(raw: str) -> Optional[str]:
    """Best-effort normalization of a matched date string to ISO (YYYY-MM-DD).

    This is purely additive -- callers are expected to keep the raw string
    as the primary/source-of-truth representation. Returns None if the
    format isn't recognized (should not normally happen for text that
    already matched DATE_PATTERN).
    """
    text = raw.strip()

    iso_match = re.fullmatch(_DATE_ISO, text)
    if iso_match:
        return text

    slash_match = re.fullmatch(_DATE_SLASH, text)
    if slash_match:
        parts = re.split(r"[/-]", text)
        day, month, year = parts[0], parts[1], parts[2]
        return f"{year}-{int(month):02d}-{int(day):02d}"

    long_match = re.fullmatch(
        rf"({_MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})", text, re.IGNORECASE
    )
    if long_match:
        month_name, day, year = long_match.groups()
        month_num = _MONTH_LOOKUP.get(month_name.lower().rstrip("."))
        if month_num:
            return f"{year}-{month_num}-{int(day):02d}"

    return None


# ---------------------------------------------------------------------------
# 3. Extraction function
# ---------------------------------------------------------------------------

def _is_preceded_by_not(text: str, start: int) -> bool:
    """True if the word immediately before position `start` in `text` is
    the literal word "not" (ignoring trailing punctuation), e.g. the
    "not" in "..., not ORD-12345" or "Not September 25, September 30."
    This is the shared building block behind same-message repudiation
    handling for both identifiers and dates.
    """
    preceding_words = text[:start].split()
    if not preceding_words:
        return False
    return preceding_words[-1].strip(",.:;()").lower() == "not"


def _select_active_identifier(pattern: "re.Pattern", text: str) -> Optional[str]:
    """Among all matches of `pattern` in `text`, pick the one that should
    be treated as the "intended" / active value for this single message.

    Handles same-message self-corrections such as:
        "Actually, I meant ORD-54321, not ORD-12345."
    by excluding any match whose immediately preceding word is "not"
    (the classic "X, not Y" repudiation pattern), then taking the last
    remaining match in reading order. If every match is preceded by
    "not" (unlikely), falls back to the last match overall so a value is
    never silently dropped.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    candidates = [m.group(0) for m in matches if not _is_preceded_by_not(text, m.start())]

    if candidates:
        return candidates[-1]
    return matches[-1].group(0)


def extract_entities(text: str) -> ExtractedEntities:
    """Extract all supported entity types from a single message.

    Deterministic and dependency-free apart from Pydantic. Never invokes
    an LLM. Business identifiers (order_id, product_code) are returned
    exactly as they appear in `text` -- no re-casing, translation, or
    reformatting is applied to them anywhere in this function.

    If a slot type appears more than once in the same message (e.g. a
    self-correction within one turn, "ORD-12345, actually ORD-54321"),
    the *last* occurrence in reading order is treated as the intended
    value for that message; callers that need the full set can inspect
    `find_all_order_ids` / `find_all_product_codes` directly.
    """
    if not isinstance(text, str):
        raise TypeError("extract_entities expects a string")

    raw_dates = [
        m.group(0)
        for m in DATE_PATTERN.finditer(text)
        if not _is_preceded_by_not(text, m.start())
    ]

    name_match = None
    for m in NAME_PATTERN.finditer(text):
        candidate = m.group(1).strip()
        first_token = candidate.split()[0].lower()
        if first_token in _NAME_STOPWORDS:
            continue
        name_match = candidate

    return ExtractedEntities(
        customer_name=name_match,
        order_id=_select_active_identifier(ORDER_ID_PATTERN, text),
        product_code=_select_active_identifier(PRODUCT_CODE_PATTERN, text),
        dates=raw_dates,
    )


def find_all_order_ids(text: str) -> List[str]:
    return ORDER_ID_PATTERN.findall(text)


def find_all_product_codes(text: str) -> List[str]:
    return PRODUCT_CODE_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# 4. Dialogue state
# ---------------------------------------------------------------------------

SLOT_NAMES = ("customer_name", "order_id", "product_code", "dates")

# Phrases that signal the message is *correcting* a previously stated
# value rather than introducing a brand-new, additional one.
_CORRECTION_CUES = re.compile(
    r"\bactually\b|\bi meant\b|\bi mean\b|\bcorrection\b|\bsorry,? i meant\b|"
    r"\bnot\b.*\bbut\b|\bchange (?:it|that) to\b|\bupdate (?:it|that) to\b|"
    r"\bno,\s*(?:the\s+)?(?:date|correct)\b|\bthe correct \w+ is\b",
    re.IGNORECASE,
)


def _is_date_correction(text: str) -> bool:
    """True if `text` reads like it is correcting a previously-stated date
    (cue phrases such as "actually", "I meant", "the correct date is",
    "no, the date is") or explicitly repudiates one date in favor of
    another within the same message ("Not September 25, September 30.").

    Used only to decide, at the dialogue-state level, whether a newly
    mentioned date should *replace* the active date(s) or simply be
    *added* alongside any date(s) already active -- a message that
    legitimately mentions multiple dates (e.g. an order date and a
    delivery date) with no such cue is never treated as a correction.
    """
    if _CORRECTION_CUES.search(text):
        return True
    for m in DATE_PATTERN.finditer(text):
        if _is_preceded_by_not(text, m.start()):
            return True
    return False


@dataclass
class DialogueState:
    """Holds the single active value per slot plus per-slot history.

    `values` always reflects the latest confirmed value for each slot.
    `history` retains every distinct value ever seen for that slot, in
    the order it was introduced, so old values are never silently lost.
    """

    values: Dict[str, Optional[object]] = field(
        default_factory=lambda: {name: (list() if name == "dates" else None) for name in SLOT_NAMES}
    )
    history: Dict[str, List[str]] = field(
        default_factory=lambda: {name: [] for name in SLOT_NAMES}
    )

    def get(self, slot: str):
        if slot not in SLOT_NAMES:
            raise KeyError(f"Unknown slot '{slot}'. Valid slots: {SLOT_NAMES}")
        return self.values[slot]

    def as_dict(self) -> Dict[str, object]:
        return deepcopy(self.values)

    def _record_history(self, slot: str, value: str) -> None:
        if value not in self.history[slot]:
            self.history[slot].append(value)

    def set_active(self, slot: str, value: str) -> None:
        """Directly set/correct the active value for a scalar slot."""
        if slot not in SLOT_NAMES:
            raise KeyError(f"Unknown slot '{slot}'. Valid slots: {SLOT_NAMES}")
        if slot == "dates":
            raise ValueError("Use add_date()/set_active_date() for the 'dates' slot")
        self.values[slot] = value
        self._record_history(slot, value)

    def add_date(self, value: str) -> None:
        if value not in self.values["dates"]:
            self.values["dates"].append(value)
        self._record_history("dates", value)

    def replace_dates(self, values: List[str]) -> None:
        """Replace the entire active `dates` list with `values` (used for
        date corrections, e.g. "Actually, I meant September 30, 2026.").
        Every value is still recorded in history, so the previous active
        date is never lost -- it just stops being active.
        """
        for v in values:
            self._record_history("dates", v)
        # Preserve order, drop duplicates.
        self.values["dates"] = list(dict.fromkeys(values))


class DialogueStateManager:
    """Stateful, multi-turn wrapper around `extract_entities` and
    `DialogueState`.

    Typical usage:

        mgr = DialogueStateManager()
        mgr.process_turn("My order is ORD-98412.")
        mgr.get("order_id")            # -> "ORD-98412"
        mgr.process_turn("Actually, I meant ORD-54321, not ORD-98412.")
        mgr.get("order_id")            # -> "ORD-54321"
        mgr.get_history("order_id")    # -> ["ORD-98412", "ORD-54321"]
    """

    def __init__(self) -> None:
        self.state = DialogueState()
        self.turns: List[str] = []

    # -- extraction -------------------------------------------------
    def extract(self, text: str) -> ExtractedEntities:
        return extract_entities(text)

    # -- state update -------------------------------------------------
    def process_turn(self, text: str) -> ExtractedEntities:
        """Extract entities from `text` and merge them into the persistent
        dialogue state, returning what was extracted from this turn.

        A correction is detected via `_CORRECTION_CUES`; in that case the
        newly extracted value simply becomes the active value (the old
        one remains in history). Absent a correction cue, a newly
        mentioned value for a slot still becomes the active value (most
        recent mention wins), consistent with normal multi-turn dialogue
        where the latest statement is authoritative.
        """
        self.turns.append(text)
        entities = self.extract(text)

        if entities.customer_name:
            self.state.set_active("customer_name", entities.customer_name)
        if entities.order_id:
            self.state.set_active("order_id", entities.order_id)
        if entities.product_code:
            self.state.set_active("product_code", entities.product_code)
        if entities.dates:
            if _is_date_correction(text):
                # e.g. "Actually, I meant September 30, 2026." or
                # "Not September 25, September 30." -- the new date(s)
                # become the entire active set; the old value is kept
                # only in history.
                self.state.replace_dates(entities.dates)
            else:
                # No correction language: dates accumulate, since a
                # message can legitimately mention more than one
                # distinct date (e.g. order date and delivery date).
                for d in entities.dates:
                    self.state.add_date(d)

        return entities

    # -- retrieval -------------------------------------------------
    def get(self, slot: str):
        return self.state.get(slot)

    def get_history(self, slot: str) -> List[str]:
        if slot not in SLOT_NAMES:
            raise KeyError(f"Unknown slot '{slot}'. Valid slots: {SLOT_NAMES}")
        return list(self.state.history[slot])

    def snapshot(self) -> Dict[str, object]:
        """Return the current dialogue_state as a plain dict, matching the
        shape described in the Step 3 spec, e.g.:

            {
                "customer_name": "John Smith",
                "order_id": "ORD-98412",
                "product_code": "SKU-B12",
                "dates": ["2026-09-25"],
            }
        """
        return self.state.as_dict()

    # -- corrections -------------------------------------------------
    def apply_correction(self, slot: str, new_value: str) -> None:
        """Explicit, programmatic correction API (as opposed to a
        correction expressed in natural language and picked up by
        `process_turn`). The old value is preserved in history.
        """
        if slot == "dates":
            self.state.add_date(new_value)
        else:
            self.state.set_active(slot, new_value)

    def is_correction_message(self, text: str) -> bool:
        """Heuristic helper exposing whether a message reads like a
        correction. Not required for `process_turn` to behave correctly
        (which already treats "most recent mention wins" as the default),
        but useful for callers/UI that want to flag corrections
        explicitly.
        """
        return bool(_CORRECTION_CUES.search(text))