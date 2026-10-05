from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------

PASSWORD_MASK = "[PASSWORD_REDACTED]"
SECRET_MASK = "[REDACTED]"
NATIONAL_ID_MASK = "[NATIONAL_ID_REDACTED]"

# Labeled password/passphrase, e.g. "password: hunter2", "pwd=abc123".
_PASSWORD_RE = re.compile(
    r"\b(?:password|passwd|pwd|pass)\s*[:=]\s*(\S+)", re.IGNORECASE
)

# Labeled generic secrets/credentials, e.g. "api_key: sk-123", "OTP: 445566",
# "token=abcdef", "cvv: 123", "pin: 4321".
_SECRET_RE = re.compile(
    r"\b(?:api[_ ]?key|secret|token|otp|cvv|pin)\s*[:=]\s*(\S+)", re.IGNORECASE
)

# Credit-card-like digit runs (13-19 digits total), allowing spaces/dashes
# as separators, e.g. "4111 1111 1111 1111", "4111-1111-1111-1111",
# "4111111111111111".
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

# US-style SSN format national ID, e.g. "123-45-6789".
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Labeled national ID / passport / equivalent, e.g. "National ID: A1234567",
# "SSN: 123456789", "Passport No: X9988776", "Aadhaar: 123412341234".
_LABELED_NATIONAL_ID_RE = re.compile(
    r"\b(?:national\s*id|nid|ssn|social\s*security(?:\s*number)?|"
    r"passport(?:\s*no\.?|\s*number)?|aadhaar|aadhar|id\s*number)"
    r"\s*[:#]?\s*([A-Za-z0-9-]{5,20})",
    re.IGNORECASE,
)


def _mask_card_digits(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if not (13 <= len(digits) <= 19):
        return match.group(0)  # not actually a plausible card number; leave as-is
    masked = "*" * (len(digits) - 4) + digits[-4:]
    return masked


def _mask_labeled_value(match: re.Match, mask: str) -> str:
    """Replace only the captured value, keeping the label text intact."""
    full = match.group(0)
    value = match.group(1)
    return full[: -len(value)] + mask


def mask_pii(text: Optional[str]) -> str:
    """
    Mask credit card numbers, national IDs, passwords, and other clearly
    labeled secrets/credentials in `text`. Returns an empty string for
    None/empty input. All other text is preserved unchanged.
    """
    if not text:
        return ""

    masked = text

    # Passwords and generic secrets first, so their digit values don't
    # later get mistaken for (and re-masked as) credit card numbers.
    masked = _PASSWORD_RE.sub(lambda m: _mask_labeled_value(m, PASSWORD_MASK), masked)
    masked = _SECRET_RE.sub(lambda m: _mask_labeled_value(m, SECRET_MASK), masked)

    # Credit card numbers (partial mask: only the last 4 digits remain).
    masked = _CREDIT_CARD_RE.sub(_mask_card_digits, masked)

    # National IDs: SSN-shaped, then any other labeled ID.
    masked = _SSN_RE.sub(NATIONAL_ID_MASK, masked)
    masked = _LABELED_NATIONAL_ID_RE.sub(lambda m: _mask_labeled_value(m, NATIONAL_ID_MASK), masked)

    return masked


def mask_list(items: Optional[List[str]]) -> List[str]:
    """Apply `mask_pii` to every item in a list; missing input -> empty list."""
    if not items:
        return []
    return [mask_pii(item) for item in items]


# ---------------------------------------------------------------------------
# Agent handoff summary
# ---------------------------------------------------------------------------

NOT_PROVIDED = "Not provided."


@dataclass(frozen=True)
class HandoffSummary:
    core_issue: str
    steps_tried: List[str] = field(default_factory=list)
    customer_mood: str = NOT_PROVIDED
    order_details: str = NOT_PROVIDED

    def render(self) -> str:
        steps_block = (
            "\n".join(f"  - {step}" for step in self.steps_tried)
            if self.steps_tried
            else "  - No steps recorded yet."
        )
        return (
            "=== Agent Handoff Summary ===\n"
            f"Core Issue: {self.core_issue}\n"
            f"Steps Already Tried:\n{steps_block}\n"
            f"Customer Mood: {self.customer_mood}\n"
            f"Order Details: {self.order_details}\n"
        )


def generate_handoff_summary(
    core_issue: Optional[str] = None,
    steps_tried: Optional[List[str]] = None,
    customer_mood: Optional[str] = None,
    order_details: Optional[str] = None,
) -> HandoffSummary:
    """
    Build a `HandoffSummary` using only masked/safe content. Missing
    fields are shown as "Not provided." rather than guessed at.
    """
    masked_issue = mask_pii(core_issue) or NOT_PROVIDED
    masked_steps = mask_list(steps_tried)
    masked_mood = mask_pii(customer_mood) or NOT_PROVIDED
    masked_order = mask_pii(order_details) or NOT_PROVIDED

    return HandoffSummary(
        core_issue=masked_issue,
        steps_tried=masked_steps,
        customer_mood=masked_mood,
        order_details=masked_order,
    )


if __name__ == "__main__":
    raw_notes = (
        "Customer is frustrated. Card 4111 1111 1111 1111 was charged twice. "
        "SSN 123-45-6789 on file. password: hunter2 was reset."
    )
    print("Masked notes:\n", mask_pii(raw_notes))

    summary = generate_handoff_summary(
        core_issue="Customer was double-charged on their last order.",
        steps_tried=["Verified charge in billing system", "Issued refund request"],
        customer_mood="Frustrated but cooperative",
        order_details="Order #ORD-5521, card ending in 1111",
    )
    print("\n" + summary.render())