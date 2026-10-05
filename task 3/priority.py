from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


# Weights for the five 0-100 sub-scores. Must sum to 1.0.
WEIGHT_SEVERITY = 0.30
WEIGHT_IMPACT = 0.25
WEIGHT_SENTIMENT = 0.15
WEIGHT_URGENCY = 0.10
WEIGHT_WAITING = 0.20

# Flat bonus added to the final score for VIP/premium/enterprise customers.
VIP_BONUS = 10.0

# Neutral fallback values used when a signal cannot be detected from the
# ticket at all. Chosen to be "moderate-low" so that missing data never
# pushes a ticket into P1/P2 by accident, but also doesn't silently hide
# a real ticket by scoring it at zero.
DEFAULT_SEVERITY_SCORE = 40.0
DEFAULT_IMPACT_SCORE = 40.0
DEFAULT_SENTIMENT_SCORE = 20.0
DEFAULT_URGENCY_SCORE = 30.0
DEFAULT_WAITING_SCORE = 30.0

# Final score -> priority thresholds (inclusive lower bound).
PRIORITY_THRESHOLDS = (
    ("P1 Critical", 80.0),
    ("P2 High", 60.0),
    ("P3 Medium", 35.0),
    ("P4 Low", 0.0),
)

# Waiting time buckets: (upper_bound_minutes_exclusive, score)
# Evaluated in order; first bucket the elapsed minutes falls under wins.
WAITING_TIME_BUCKETS: Tuple[Tuple[float, float], ...] = (
    (15, 10.0),      # < 15 min
    (60, 30.0),      # < 1 hour
    (240, 55.0),     # < 4 hours
    (1440, 75.0),    # < 24 hours
)
WAITING_TIME_MAX_SCORE = 95.0  # >= 24 hours

# Keyword -> severity score. Matched by substring against lower-cased text.
SEVERITY_KEYWORDS: Dict[str, float] = {
    "data loss": 100.0,
    "security breach": 100.0,
    "account blocked": 95.0,
    "account suspended": 95.0,
    "payment failed": 85.0,
    "system crash": 85.0,
    "cannot log in": 80.0,
    "cannot login": 80.0,
    "can't log in": 80.0,
    "crash": 75.0,
    "not working": 55.0,
    "error": 45.0,
    "ui glitch": 15.0,
    "minor ui": 15.0,
    "minor issue": 15.0,
    "cosmetic": 10.0,
    "typo": 5.0,
}

# Keyword -> customer impact score.
IMPACT_KEYWORDS: Dict[str, float] = {
    "production down": 100.0,
    "all users": 95.0,
    "entire team": 90.0,
    "company-wide": 90.0,
    "whole organization": 90.0,
    "multiple customers": 85.0,
    "several users": 70.0,
    "one user": 15.0,
    "single user": 15.0,
    "just me": 10.0,
    "only affects me": 10.0,
}

# Keyword -> frustration/sentiment base score.
SENTIMENT_KEYWORDS: Dict[str, float] = {
    "furious": 90.0,
    "unacceptable": 85.0,
    "angry": 80.0,
    "worst": 75.0,
    "frustrated": 70.0,
    "ridiculous": 65.0,
    "terrible": 60.0,
    "annoyed": 55.0,
    "disappointed": 50.0,
}
# Extra bonus per exclamation mark in the customer's text, capped.
EXCLAMATION_BONUS_PER_MARK = 5.0
EXCLAMATION_BONUS_CAP = 20.0

# Keyword -> stated urgency score.
URGENCY_KEYWORDS: Dict[str, float] = {
    "emergency": 95.0,
    "asap": 90.0,
    "immediately": 90.0,
    "right away": 85.0,
    "urgent": 85.0,
    "critical": 80.0,
    "low priority": 10.0,
    "no rush": 5.0,
    "whenever": 10.0,
}

# Customer tier strings treated as VIP.
VIP_TIERS = {"vip", "premium", "enterprise"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SignalBreakdown:
    """Transparent record of every signal used in the score, and whether
    each one was actually detected in the ticket or fell back to a default.
    """
    severity_score: float
    severity_matched_keyword: Optional[str]
    severity_was_defaulted: bool

    impact_score: float
    impact_matched_keyword: Optional[str]
    impact_was_defaulted: bool

    sentiment_score: float
    sentiment_matched_keyword: Optional[str]
    sentiment_exclamation_bonus: float
    sentiment_was_defaulted: bool

    urgency_score: float
    urgency_matched_keyword: Optional[str]
    urgency_was_defaulted: bool

    waiting_score: float
    waiting_minutes_used: Optional[float]
    waiting_was_defaulted: bool

    is_vip: bool
    vip_was_defaulted: bool
    vip_bonus_applied: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PriorityResult:
    score: float
    priority: str
    signals: SignalBreakdown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "priority": self.priority,
            "signals": self.signals.to_dict(),
        }


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def _extract_text(ticket: Dict[str, Any]) -> str:
    """Pull whatever free-text is available on the ticket and normalize it."""
    for key in ("text", "message", "description", "body", "comment"):
        value = ticket.get(key)
        if isinstance(value, str) and value.strip():
            return value.lower()
    return ""


def _match_keywords(text: str, keyword_scores: Dict[str, float]) -> Tuple[Optional[str], Optional[float]]:
    """Return the (keyword, score) of the highest-scoring keyword found in
    text, or (None, None) if nothing matched. Deterministic: ties are broken
    by keyword insertion order in the dict (dict iteration order is stable).
    """
    best_keyword: Optional[str] = None
    best_score: Optional[float] = None
    for keyword, score in keyword_scores.items():
        if keyword in text:
            if best_score is None or score > best_score:
                best_keyword, best_score = keyword, score
    return best_keyword, best_score


def _score_sentiment(text: str) -> Tuple[float, Optional[str], float, bool]:
    """Returns (score, matched_keyword, exclamation_bonus, was_defaulted)."""
    matched_keyword, base_score = _match_keywords(text, SENTIMENT_KEYWORDS)
    exclamation_bonus = min(text.count("!") * EXCLAMATION_BONUS_PER_MARK, EXCLAMATION_BONUS_CAP)

    if base_score is None and exclamation_bonus == 0.0:
        return DEFAULT_SENTIMENT_SCORE, None, 0.0, True

    score = min(100.0, (base_score or 0.0) + exclamation_bonus)
    return score, matched_keyword, exclamation_bonus, False


def _extract_waiting_minutes(ticket: Dict[str, Any]) -> Optional[float]:
    """Accepts a few common field spellings for elapsed waiting time."""
    if "waiting_minutes" in ticket and ticket["waiting_minutes"] is not None:
        return max(0.0, float(ticket["waiting_minutes"]))
    if "wait_minutes" in ticket and ticket["wait_minutes"] is not None:
        return max(0.0, float(ticket["wait_minutes"]))
    if "waiting_hours" in ticket and ticket["waiting_hours"] is not None:
        return max(0.0, float(ticket["waiting_hours"]) * 60.0)
    return None


def _score_waiting_time(minutes: Optional[float]) -> Tuple[float, bool]:
    if minutes is None:
        return DEFAULT_WAITING_SCORE, True
    for upper_bound, score in WAITING_TIME_BUCKETS:
        if minutes < upper_bound:
            return score, False
    return WAITING_TIME_MAX_SCORE, False


def _extract_vip(ticket: Dict[str, Any]) -> Tuple[bool, bool]:
    """Returns (is_vip, was_defaulted)."""
    tier = ticket.get("customer_tier")
    if isinstance(tier, str) and tier.strip():
        return tier.strip().lower() in VIP_TIERS, False
    # Also accept an explicit boolean flag if the caller already knows.
    if isinstance(ticket.get("is_vip"), bool):
        return ticket["is_vip"], False
    return False, True


def extract_signals(ticket: Dict[str, Any]) -> SignalBreakdown:
    """Extract all signals needed for scoring from a raw ticket dict.

    `ticket` is intentionally a plain dict so this function can be used
    with tickets coming from any upstream source (webhook payload, DB row,
    etc). Recognized optional keys:

      text / message / description / body / comment : str
      waiting_minutes / wait_minutes / waiting_hours : number
      customer_tier : str (e.g. "vip", "standard")
      is_vip : bool (fallback if customer_tier absent)

    Missing/unknown fields are never guessed at with invented specifics;
    they fall back to documented neutral constants and are flagged as
    defaulted in the returned breakdown.
    """
    text = _extract_text(ticket)

    severity_kw, severity_val = _match_keywords(text, SEVERITY_KEYWORDS)
    severity_score = severity_val if severity_val is not None else DEFAULT_SEVERITY_SCORE
    severity_defaulted = severity_val is None

    impact_kw, impact_val = _match_keywords(text, IMPACT_KEYWORDS)
    impact_score = impact_val if impact_val is not None else DEFAULT_IMPACT_SCORE
    impact_defaulted = impact_val is None

    sentiment_score, sentiment_kw, excl_bonus, sentiment_defaulted = _score_sentiment(text)

    urgency_kw, urgency_val = _match_keywords(text, URGENCY_KEYWORDS)
    urgency_score = urgency_val if urgency_val is not None else DEFAULT_URGENCY_SCORE
    urgency_defaulted = urgency_val is None

    waiting_minutes = _extract_waiting_minutes(ticket)
    waiting_score, waiting_defaulted = _score_waiting_time(waiting_minutes)

    is_vip, vip_defaulted = _extract_vip(ticket)
    vip_bonus = VIP_BONUS if is_vip else 0.0

    return SignalBreakdown(
        severity_score=severity_score,
        severity_matched_keyword=severity_kw,
        severity_was_defaulted=severity_defaulted,
        impact_score=impact_score,
        impact_matched_keyword=impact_kw,
        impact_was_defaulted=impact_defaulted,
        sentiment_score=sentiment_score,
        sentiment_matched_keyword=sentiment_kw,
        sentiment_exclamation_bonus=excl_bonus,
        sentiment_was_defaulted=sentiment_defaulted,
        urgency_score=urgency_score,
        urgency_matched_keyword=urgency_kw,
        urgency_was_defaulted=urgency_defaulted,
        waiting_score=waiting_score,
        waiting_minutes_used=waiting_minutes,
        waiting_was_defaulted=waiting_defaulted,
        is_vip=is_vip,
        vip_was_defaulted=vip_defaulted,
        vip_bonus_applied=vip_bonus,
    )


# ---------------------------------------------------------------------------
# Scoring / priority mapping
# ---------------------------------------------------------------------------

def compute_score(signals: SignalBreakdown) -> float:
    """Deterministic weighted sum of the five sub-scores plus the VIP bonus,
    clamped to the [0, 100] range.
    """
    weighted = (
        signals.severity_score * WEIGHT_SEVERITY
        + signals.impact_score * WEIGHT_IMPACT
        + signals.sentiment_score * WEIGHT_SENTIMENT
        + signals.urgency_score * WEIGHT_URGENCY
        + signals.waiting_score * WEIGHT_WAITING
    )
    total = weighted + signals.vip_bonus_applied
    return max(0.0, min(100.0, total))


def score_to_priority(score: float) -> str:
    for label, lower_bound in PRIORITY_THRESHOLDS:
        if score >= lower_bound:
            return label
    return PRIORITY_THRESHOLDS[-1][0]  # unreachable given 0.0 floor, kept for safety


def score_ticket(ticket: Dict[str, Any]) -> PriorityResult:
    """Full pipeline: extract signals -> compute score -> map to priority.

    Same `ticket` dict always yields the same `PriorityResult` (pure
    function, no randomness, no I/O, no external calls).
    """
    signals = extract_signals(ticket)
    score = compute_score(signals)
    priority = score_to_priority(score)
    return PriorityResult(score=score, priority=priority, signals=signals)


if __name__ == "__main__":
    example = {
        "text": "Our account is blocked and this affects the entire team! "
                "This is unacceptable, please help immediately!!!",
        "customer_tier": "vip",
        "waiting_minutes": 200,
    }
    result = score_ticket(example)
    import json
    print(json.dumps(result.to_dict(), indent=2))