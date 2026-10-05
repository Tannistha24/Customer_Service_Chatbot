from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# 1. Schema definition
# ---------------------------------------------------------------------------

# Mandatory fields a ticket must have before it can be considered validated.
MANDATORY_FIELDS: List[str] = [
    "customer_name",
    "contact_info",
    "order_id",
    "issue_category",
    "issue_description",
]

# Optional fields we still try to extract, but whose absence does not block
# validation.
OPTIONAL_FIELDS: List[str] = [
    "customer_id",
    "product_id",
    "evidence",
]

ALL_FIELDS: List[str] = MANDATORY_FIELDS + OPTIONAL_FIELDS

# Human-readable prompts used when a mandatory field is missing.
FIELD_PROMPTS: Dict[str, str] = {
    "customer_name": "your full name",
    "contact_info": "a way to reach you (email or phone number)",
    "order_id": "your order ID (or account/reference number)",
    "issue_category": "the type of issue you're experiencing (e.g. billing, technical, shipping)",
    "issue_description": "a short description of the problem you're facing",
}

# Simple, known issue categories used for keyword-based classification.
ISSUE_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "billing": ["charge", "invoice", "refund", "payment", "billed", "billing", "overcharged"],
    "shipping": ["delivery", "shipment", "shipped", "tracking", "package", "courier", "delayed"],
    "technical": ["error", "bug", "crash", "not working", "broken", "failed", "exception", "login"],
    "account": ["password", "account", "login", "locked", "access", "profile"],
    "product": ["defective", "damaged", "quality", "missing part", "wrong item"],
}


@dataclass
class TicketData:
    """Extracted ticket fields. `None` means "not found in the text"."""

    customer_name: Optional[str] = None
    customer_id: Optional[str] = None
    contact_info: Optional[str] = None
    order_id: Optional[str] = None
    product_id: Optional[str] = None
    issue_category: Optional[str] = None
    issue_description: Optional[str] = None
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    ticket_id: Optional[str]
    data: TicketData
    is_valid: bool
    missing_fields: List[str]
    invalid_fields: List[str]
    missing_data_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "data": self.data.to_dict(),
            "is_valid": self.is_valid,
            "missing_fields": self.missing_fields,
            "invalid_fields": self.invalid_fields,
            "missing_data_prompt": self.missing_data_prompt,
        }


# ---------------------------------------------------------------------------
# 2. Entity extraction
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})*")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3}[\s.-]?\d{3,4}")
_ORDER_ID_RE = re.compile(r"\b(?:order|order\s*id|order\s*#)\s*[:#]?\s*([A-Za-z0-9-]{4,})\b", re.IGNORECASE)
_PRODUCT_ID_RE = re.compile(r"\b(?:product|product\s*id|sku)\s*[:#]?\s*([A-Za-z0-9-]{3,})\b", re.IGNORECASE)
_CUSTOMER_ID_RE = re.compile(r"\b(?:customer\s*id|account\s*id|account\s*#|customer\s*#)\s*[:#]?\s*([A-Za-z0-9-]{3,})\b", re.IGNORECASE)
_NAME_RE = re.compile(
    r"\b(?:my name is|i am|i'm|this is)\s+([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,2})",
    re.IGNORECASE,
)
# Matches error/err (optionally followed by "code") plus an alphanumeric code
# that contains at least one digit, e.g. "error code ERR-504", "err 404", "Error: E1234".
_ERROR_CODE_RE = re.compile(r"\b(?:error|err)\w*(?:\s*code)?\s*[:#]?\s*([A-Za-z]{0,6}-?\d{2,6})\b", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
)
_SCREENSHOT_RE = re.compile(r"\b[\w-]+\.(?:png|jpe?g|gif|bmp|pdf)\b", re.IGNORECASE)
_LOG_MENTION_RE = re.compile(r"\b(log(?:s)?(?: file)?|stack trace|traceback)\b", re.IGNORECASE)


def _first_match(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    return m.group(1).strip() if m and m.groups() else (m.group(0).strip() if m else None)


def extract_customer_name(text: str) -> Optional[str]:
    return _first_match(_NAME_RE, text)


def extract_contact_info(text: str) -> Optional[str]:
    email = _first_match(_EMAIL_RE, text)
    if email:
        return email
    phone = _PHONE_RE.search(text)
    if phone and len(re.sub(r"\D", "", phone.group(0))) >= 7:
        return phone.group(0).strip()
    return None


def extract_order_id(text: str) -> Optional[str]:
    return _first_match(_ORDER_ID_RE, text)


def extract_product_id(text: str) -> Optional[str]:
    return _first_match(_PRODUCT_ID_RE, text)


def extract_customer_id(text: str) -> Optional[str]:
    return _first_match(_CUSTOMER_ID_RE, text)


def extract_issue_category(text: str) -> Optional[str]:
    lowered = text.lower()
    for category, keywords in ISSUE_CATEGORY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return category
    return None


def extract_issue_description(text: str) -> Optional[str]:
    """
    Heuristic: take the sentence containing the strongest issue keyword,
    otherwise fall back to the first non-trivial sentence of the message.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return None

    lowered_keywords = [kw for kws in ISSUE_CATEGORY_KEYWORDS.values() for kw in kws]
    for sentence in sentences:
        low = sentence.lower()
        if any(kw in low for kw in lowered_keywords):
            return sentence

    # Fallback: first sentence that isn't just a greeting/name intro.
    for sentence in sentences:
        if not _NAME_RE.search(sentence) and len(sentence.split()) > 3:
            return sentence

    return sentences[0]


def extract_evidence(text: str) -> List[str]:
    evidence: List[str] = []

    for m in _ERROR_CODE_RE.finditer(text):
        evidence.append(f"error_code:{m.group(1)}")

    for m in _TIMESTAMP_RE.finditer(text):
        evidence.append(f"timestamp:{m.group(0)}")

    for m in _SCREENSHOT_RE.finditer(text):
        evidence.append(f"screenshot:{m.group(0)}")

    if _LOG_MENTION_RE.search(text):
        evidence.append("log_reference:mentioned")

    # De-duplicate while preserving order.
    seen = set()
    deduped = []
    for e in evidence:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    return deduped


def extract_ticket_data(text: str) -> TicketData:
    """Run all field extractors over the raw ticket text."""
    return TicketData(
        customer_name=extract_customer_name(text),
        customer_id=extract_customer_id(text),
        contact_info=extract_contact_info(text),
        order_id=extract_order_id(text),
        product_id=extract_product_id(text),
        issue_category=extract_issue_category(text),
        issue_description=extract_issue_description(text),
        evidence=extract_evidence(text),
    )


# ---------------------------------------------------------------------------
# 3. Schema validation
# ---------------------------------------------------------------------------

def validate_ticket(data: TicketData) -> Dict[str, List[str]]:
    """
    Validate extracted data against the schema.

    Returns a dict with:
      - "missing": mandatory fields that are absent (None / empty).
      - "invalid": fields that are present but fail a basic sanity check
        (e.g. malformed contact info).
    """
    missing: List[str] = []
    invalid: List[str] = []

    values = data.to_dict()
    for fname in MANDATORY_FIELDS:
        value = values.get(fname)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(fname)

    # Basic sanity check on contact info, if present.
    contact = values.get("contact_info")
    if contact and not (_EMAIL_RE.fullmatch(contact) or re.sub(r"\D", "", contact).__len__() >= 7):
        invalid.append("contact_info")

    return {"missing": missing, "invalid": invalid}


# ---------------------------------------------------------------------------
# 4. Missing-data prompt generation
# ---------------------------------------------------------------------------

def generate_missing_data_prompt(missing_fields: List[str]) -> Optional[str]:
    """
    Build a customer-facing message requesting the missing mandatory
    information. Returns None if nothing is missing.
    """
    if not missing_fields:
        return None

    asks = [FIELD_PROMPTS.get(f, f.replace("_", " ")) for f in missing_fields]

    if len(asks) == 1:
        body = asks[0]
    else:
        body = ", ".join(asks[:-1]) + f", and {asks[-1]}"

    return (
        "Thanks for reaching out! To help resolve your issue, could you "
        f"please provide {body}? Once we have this, we can proceed with "
        "your request."
    )


# ---------------------------------------------------------------------------
# 5. Orchestration entry point
# ---------------------------------------------------------------------------

def process_ticket(raw_text: str, ticket_id: Optional[str] = None) -> ValidationResult:
    """
    Full pipeline step: extract -> validate -> (maybe) generate prompt.

    A ticket is `is_valid=True` only when there are no missing mandatory
    fields and no invalid fields. No values are ever invented; anything
    not found in `raw_text` stays None/empty.
    """
    data = extract_ticket_data(raw_text)
    report = validate_ticket(data)
    missing = report["missing"]
    invalid = report["invalid"]

    is_valid = not missing and not invalid
    prompt = generate_missing_data_prompt(missing) if missing else None

    return ValidationResult(
        ticket_id=ticket_id,
        data=data,
        is_valid=is_valid,
        missing_fields=missing,
        invalid_fields=invalid,
        missing_data_prompt=prompt,
    )


if __name__ == "__main__":
    sample = (
        "Hi, my name is Jane Doe. I was charged twice for my last purchase. "
        "My email is jane.doe@example.com. Order ID: ORD-88213. "
        "I saw error code ERR-504 in the app on 2024-05-01 14:32. "
        "Attached screenshot.png for reference."
    )
    result = process_ticket(sample, ticket_id="T-1001")
    import json
    print(json.dumps(result.to_dict(), indent=2))