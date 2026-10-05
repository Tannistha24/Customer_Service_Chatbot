
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

# ---------------------------------------------------------------------------
# 0. Configured intent taxonomy
# ---------------------------------------------------------------------------
#
# Documented, closed set of customer-support intents this component can
# recognize. Anything that does not match any of these is "unknown".

UNKNOWN_INTENT = "unknown"

SUPPORTED_INTENTS: Tuple[str, ...] = (
    "cancel_order",
    "keep_order",          # the *opposite* of cancel_order; exists so a
                            # message can be detected as self-conflicting
                            # (e.g. "cancel my order, actually keep it")
    "track_order",
    "refund_status",
    "return_item",
    "billing_issue",
    "update_address",
    "product_inquiry",
    "complaint",
)

INTENT_DESCRIPTIONS: Dict[str, str] = {
    "cancel_order": "cancel your order",
    "keep_order": "keep your order as it is",
    "track_order": "track / check the status of your order",
    "refund_status": "check the status of your refund",
    "return_item": "return an item",
    "billing_issue": "resolve a billing issue",
    "update_address": "update your shipping address",
    "product_inquiry": "ask a question about a product",
    "complaint": "file a complaint",
    UNKNOWN_INTENT: "something I could not confidently identify",
}

# Pairs of intents that are directly at odds with one another. Order
# within each tuple does not matter -- both directions are checked.
CONFLICTING_INTENT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("cancel_order", "keep_order"),
)

# ---------------------------------------------------------------------------
# 1. Deterministic scoring patterns
# ---------------------------------------------------------------------------
#
# weight tiers:
#   0.90 -> "strong"   : an explicit, unambiguous phrase for the intent
#   0.75 -> "boundary" : a clear but slightly less explicit phrase
#                        (chosen so a dedicated test can sit exactly at
#                        the default confidence threshold of 0.75)
#   0.60 -> "medium"   : a single generic keyword, on its own, below the
#                        default threshold

STRONG = 0.90
BOUNDARY = 0.75
MEDIUM = 0.60

# Each entry: intent -> list of (compiled_pattern, weight)
_INTENT_PATTERNS: Dict[str, List[Tuple[Pattern, float]]] = {
    "cancel_order": [
        (re.compile(r"\bcancel\s+(?:my|the|this)\s+order\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bi\s+want\s+to\s+cancel\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bcancel\s+it\b", re.IGNORECASE), BOUNDARY),
        (re.compile(r"\bcancel\b", re.IGNORECASE), MEDIUM),
    ],
    "keep_order": [
        (re.compile(r"\bkeep\s+(?:my|the|this)\s+order\b", re.IGNORECASE), STRONG),
        (re.compile(r"\b(?:don'?t|do\s+not)\s+cancel\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bkeep\s+it\b", re.IGNORECASE), BOUNDARY),
    ],
    "track_order": [
        (re.compile(r"\bwhere\s+is\s+my\s+order\b", re.IGNORECASE), STRONG),
        (re.compile(r"\btrack\s+my\s+order\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bstatus\s+of\s+my\s+order\b", re.IGNORECASE), BOUNDARY),
        (re.compile(r"\btrack\b", re.IGNORECASE), MEDIUM),
    ],
    "refund_status": [
        (re.compile(r"\bwhen\s+will\s+my\s+refund\s+arrive\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bwhen.*\brefund\b.*\barrive\b", re.IGNORECASE), STRONG),
        (re.compile(r"\brefund\s+status\b", re.IGNORECASE), BOUNDARY),
        (re.compile(r"\brefund\b", re.IGNORECASE), MEDIUM),
    ],
    "return_item": [
        (re.compile(r"\breturn\s+(?:this|my|the)\s+item\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bi\s+want\s+to\s+return\b", re.IGNORECASE), STRONG),
        (re.compile(r"\breturn\s+my\s+order\b", re.IGNORECASE), BOUNDARY),
        (re.compile(r"\breturn\b", re.IGNORECASE), MEDIUM),
    ],
    "billing_issue": [
        (re.compile(r"\bcharged\s+(?:incorrectly|twice|the\s+wrong\s+amount)\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bbilling\s+issue\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bwrong\s+charge\b", re.IGNORECASE), BOUNDARY),
        (re.compile(r"\bbilling\b", re.IGNORECASE), MEDIUM),
    ],
    "update_address": [
        (re.compile(r"\b(?:change|update)\s+my\s+shipping\s+address\b", re.IGNORECASE), STRONG),
        (re.compile(r"\b(?:change|update)\s+my\s+address\b", re.IGNORECASE), BOUNDARY),
        (re.compile(r"\baddress\b", re.IGNORECASE), MEDIUM),
    ],
    "product_inquiry": [
        (re.compile(r"\bis\s+this\s+product\s+available\b", re.IGNORECASE), STRONG),
        (re.compile(r"\btell\s+me\s+about\s+this\s+product\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bproduct\s+question\b", re.IGNORECASE), BOUNDARY),
        (re.compile(r"\bproduct\b", re.IGNORECASE), MEDIUM),
    ],
    "complaint": [
        (re.compile(r"\bfile\s+a\s+complaint\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bthis\s+is\s+unacceptable\b", re.IGNORECASE), STRONG),
        (re.compile(r"\bvery\s+disappointed\b", re.IGNORECASE), BOUNDARY),
        (re.compile(r"\bcomplaint\b", re.IGNORECASE), MEDIUM),
    ],
}

# Any intent whose score is at least this high counts as an actual
# "detected request" in the message (as opposed to background noise).
DETECTION_THRESHOLD = 0.5

# If the top two candidate intents' scores differ by no more than this,
# the message is treated as genuinely ambiguous between them.
AMBIGUITY_EPSILON = 0.05

DEFAULT_CONFIDENCE_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# 2. Structured result
# ---------------------------------------------------------------------------

@dataclass
class IntentClassificationResult:
    original_message: object
    intent: str
    confidence: float
    clarification_required: bool
    clarification_reasons: List[str] = field(default_factory=list)
    detected_requests: List[str] = field(default_factory=list)
    clarification_message: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "original_message": self.original_message,
            "intent": self.intent,
            "confidence": self.confidence,
            "clarification_required": self.clarification_required,
            "clarification_reasons": list(self.clarification_reasons),
            "detected_requests": list(self.detected_requests),
            "clarification_message": self.clarification_message,
        }


# ---------------------------------------------------------------------------
# 3. Scoring
# ---------------------------------------------------------------------------

def _score_intents(text: str) -> Dict[str, float]:
    """Return {intent: score} for every configured intent that matched at
    least one pattern. An intent's score is the highest-weight pattern
    that matched (not a sum), so scores always stay within a fixed,
    documented set of tiers and never exceed 1.0.
    """
    scores: Dict[str, float] = {}
    for intent, patterns in _INTENT_PATTERNS.items():
        best = 0.0
        for pattern, weight in patterns:
            if weight > best and pattern.search(text):
                best = weight
        if best > 0.0:
            scores[intent] = best
    return scores


def _detected_requests(scores: Dict[str, float]) -> List[str]:
    """Intents that cleared the detection threshold, ordered by score
    (descending) and then by name for a fully deterministic order.
    """
    detected = [intent for intent, score in scores.items() if score >= DETECTION_THRESHOLD]
    detected.sort(key=lambda i: (-scores[i], i))
    return detected


def _has_conflicting_pair(detected: List[str]) -> bool:
    detected_set = set(detected)
    for a, b in CONFLICTING_INTENT_PAIRS:
        if a in detected_set and b in detected_set:
            return True
    return False


def _is_ambiguous(scores: Dict[str, float]) -> bool:
    if len(scores) < 2:
        return False
    ordered = sorted(scores.values(), reverse=True)
    return (ordered[0] - ordered[1]) <= AMBIGUITY_EPSILON


# ---------------------------------------------------------------------------
# 4. Clarification message construction
# ---------------------------------------------------------------------------

def _describe(intent: str) -> str:
    return INTENT_DESCRIPTIONS.get(intent, intent)


def _build_clarification_message(
    intent: str,
    detected_requests: List[str],
    reasons: List[str],
) -> str:
    """Builds a clarification message that always:
    1. states what the system understood,
    2. names the ambiguity / conflict,
    3. explicitly asks the customer to choose or specify, and
    4. makes clear no action will be taken yet.

    Never invents information: only ever references intents/requests
    that were actually detected in the message.
    """
    no_action_clause = "I won't take any action until you let me know."

    if "unknown_intent" in reasons and not detected_requests:
        return (
            "I'm not fully sure what you'd like help with. Could you tell "
            "me more specifically what you need (for example: cancelling "
            "an order, checking a refund, tracking an order, returning an "
            "item, a billing issue, or a product question)? "
            + no_action_clause
        )

    if len(detected_requests) >= 2:
        options = [_describe(i) for i in detected_requests]
        understood = " and also to ".join(options)
        choices = " or to ".join(options)
        if "conflicting_intents" in reasons:
            lead = (
                f"I understand that you may want to {understood}, and those "
                "seem to conflict with each other."
            )
        else:
            lead = f"I understand that you may want to {understood}."
        return (
            f"{lead} Which would you like help with first: {choices}? "
            + no_action_clause
        )

    if "ambiguous_request" in reasons and len(detected_requests) == 1:
        return (
            f"It sounds like you might be asking me to {_describe(detected_requests[0])}, "
            "but your message could also be read another way, so I'm not "
            "fully certain. Could you confirm exactly what you'd like help "
            "with? " + no_action_clause
        )

    if detected_requests:
        return (
            f"It sounds like you might be asking me to {_describe(detected_requests[0])}, "
            "but I'm not fully confident about that. Could you confirm "
            "that's what you'd like, or let me know what you actually "
            "need? " + no_action_clause
        )

    return (
        "I'm not fully sure I understood your request. Could you clarify "
        "what you'd like help with? " + no_action_clause
    )


# ---------------------------------------------------------------------------
# 5. Public classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Deterministic intent classifier with configurable confidence
    threshold. Stateless / side-effect free -- safe to reuse across
    many messages, and safe to construct fresh per message.
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        if not isinstance(confidence_threshold, (int, float)) or isinstance(confidence_threshold, bool):
            raise TypeError("confidence_threshold must be a number")
        if not (0.0 <= float(confidence_threshold) <= 1.0):
            raise ValueError("confidence_threshold must be within 0.0-1.0")
        self.confidence_threshold = float(confidence_threshold)

    def classify(self, message: object) -> IntentClassificationResult:
        """Classify a single customer message. Never raises on empty,
        whitespace-only, or non-string input -- such input is simply
        treated as unclassifiable (intent="unknown", confidence=0.0),
        with `original_message` still preserving exactly what was
        passed in.
        """
        text = message if isinstance(message, str) else ""
        text_is_usable = isinstance(message, str) and text.strip() != ""

        scores: Dict[str, float] = _score_intents(text) if text_is_usable else {}
        detected = _detected_requests(scores)

        if detected:
            top_intent = detected[0]
            confidence = scores[top_intent]
        else:
            top_intent = UNKNOWN_INTENT
            confidence = 0.0

        reasons: List[str] = []
        if confidence < self.confidence_threshold:
            reasons.append("low_confidence")
        if top_intent == UNKNOWN_INTENT:
            reasons.append("unknown_intent")
        if len(detected) > 1:
            reasons.append("multiple_requests")
        if _has_conflicting_pair(detected):
            reasons.append("conflicting_intents")
        if _is_ambiguous(scores):
            reasons.append("ambiguous_request")

        clarification_required = len(reasons) > 0
        clarification_message = (
            _build_clarification_message(top_intent, detected, reasons)
            if clarification_required
            else None
        )

        return IntentClassificationResult(
            original_message=message,
            intent=top_intent,
            confidence=confidence,
            clarification_required=clarification_required,
            clarification_reasons=reasons,
            detected_requests=detected,
            clarification_message=clarification_message,
        )


# Module-level convenience classifier using the default threshold, plus
# a thin function wrapper for callers who don't need a custom threshold.
_default_classifier = IntentClassifier()


def classify_message(
    message: object, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> IntentClassificationResult:
    """Convenience function: classify_message(text, threshold=0.75)."""
    if confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD:
        return _default_classifier.classify(message)
    return IntentClassifier(confidence_threshold=confidence_threshold).classify(message)