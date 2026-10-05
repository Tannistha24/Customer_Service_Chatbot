import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Result model (unchanged shape from v1/v2, so downstream steps don't need
# to change) plus a `source` field so callers can see which engine answered.
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    message: str
    sentiment: str                 # positive / neutral / negative
    sentiment_scores: dict         # positive/neutral/negative/frustrated/urgent/sarcastic -> 0-1
    intent: str
    intent_confidence: float
    high_risk: bool
    high_risk_category: Optional[str]
    high_risk_confidence: float
    calm_high_risk: bool
    language: Optional[str]        # best-guess language code/name
    source: str                    # "llm" or "fallback_rules"
    classification_status: str     # "ok" or "unavailable"
    notes: list = field(default_factory=list)


class LLMResponseError(Exception):
    """Raised when the model's reply can't be parsed into a valid result."""


# ---------------------------------------------------------------------------
# Pluggable chat-model client
# ---------------------------------------------------------------------------

@runtime_checkable
class ChatModelClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw text reply to one turn."""
        ...


class GeminiClient:
    """Production client: calls Gemini to do the actual multilingual
    classification using structured output mode (JSON Schema enforcement).
    This is the 'automatic, chatbot-style' engine - it understands any
    language it was trained on natively, with nothing hand-coded per language.

    Requires: `pip install google-genai` and a GEMINI_API_KEY environment
    variable (or pass api_key explicitly).
    """

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None, timeout: int = 10):
        try:
            import google.genai as genai
        except ImportError:
            raise ImportError(
                "google-genai is required. Install it with: pip install google-genai"
            )
        self._genai = genai
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set the environment variable or pass api_key parameter. "
                "Get a key from https://aistudio.google.com/apikey"
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call Gemini with structured output (JSON Schema).
        Returns the raw JSON string response.
        """
        # Define the expected JSON schema for the response
        response_schema = {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "Language code or name (e.g., 'en', 'es', 'fr', 'hi')"
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative"],
                    "description": "Primary sentiment classification"
                },
                "scores": {
                    "type": "object",
                    "properties": {
                        "positive": {"type": "number", "description": "Score 0.0-1.0"},
                        "neutral": {"type": "number", "description": "Score 0.0-1.0"},
                        "negative": {"type": "number", "description": "Score 0.0-1.0"},
                        "frustrated": {"type": "number", "description": "Score 0.0-1.0"},
                        "urgent": {"type": "number", "description": "Score 0.0-1.0"},
                        "sarcastic": {"type": "number", "description": "Score 0.0-1.0"}
                    },
                    "required": ["positive", "neutral", "negative", "frustrated", "urgent", "sarcastic"]
                },
                "intent": {
                    "type": "string",
                    "description": "Intent label (e.g., 'refund_request', 'account_security', 'complaint')"
                },
                "intent_confidence": {
                    "type": "number",
                    "description": "Confidence 0.0-1.0"
                },
                "high_risk": {
                    "type": "boolean",
                    "description": "Is this message high risk?"
                },
                "high_risk_category": {
                    "type": ["string", "null"],
                    "enum": ["account_compromise", "legal_threat", "fraud", None],
                    "description": "Category of high risk, if applicable"
                },
                "high_risk_confidence": {
                    "type": "number",
                    "description": "High risk confidence 0.0-1.0"
                }
            },
            "required": [
                "language", "sentiment", "scores", "intent", "intent_confidence",
                "high_risk", "high_risk_category", "high_risk_confidence"
            ]
        }

        try:
            response = self._client.interactions.create(
                model=self._model,
                input=user_prompt,
                system=[system_prompt],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": response_schema
                },
                config={
                    "temperature": 0,  # minimize variance
                    "timeout": self._timeout
                }
            )
            # Extract text from the response
            if hasattr(response, 'output_text'):
                return response.output_text
            elif hasattr(response, 'text'):
                return response.text
            else:
                raise LLMResponseError(f"Unexpected response structure: {response}")
        except Exception as e:
            raise LLMResponseError(f"Gemini API call failed: {e}") from e


class FakeChatClient:
    """Test-only stand-in for self-checks below, so the classification
    logic (prompt building, JSON parsing, validation, fallback) can be
    exercised deterministically with no network access. Not for production
    use - it does not understand language, it just replays canned answers."""

    def __init__(self, canned_by_keyword: dict, default: Optional[str] = None):
        self._canned = canned_by_keyword
        self._default = default

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        for keyword, reply in self._canned.items():
            if keyword in user_prompt:
                return reply
        if self._default is not None:
            return self._default
        raise LLMResponseError("FakeChatClient: no canned reply matched this input")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a message-classification engine for a customer support pipeline.
You will be given a customer message, and optionally prior conversation history, in ANY language.
Read and reason about the message in its ORIGINAL language - do not translate it first.

Return ONLY a single JSON object (conforming to the provided schema) with exactly these fields:

{
  "language": "<your best guess at the language, e.g. 'en', 'es', 'fr', 'hi'>",
  "sentiment": "<one of: positive, neutral, negative>",
  "scores": {
    "positive": <0.0-1.0>,
    "neutral": <0.0-1.0>,
    "negative": <0.0-1.0>,
    "frustrated": <0.0-1.0>,
    "urgent": <0.0-1.0>,
    "sarcastic": <0.0-1.0>
  },
  "intent": "<short snake_case label, e.g. refund_request, billing_inquiry, account_security,
              legal_threat, technical_support, cancellation_request, complaint, compliment,
              greeting, general_inquiry, other>",
  "intent_confidence": <0.0-1.0>,
  "high_risk": <true/false>,
  "high_risk_category": "<one of: account_compromise, legal_threat, fraud, null>",
  "high_risk_confidence": <0.0-1.0>
}

Guidance:
- Sarcasm: judge it from the SITUATION, not just positive-sounding words. Positive words next to
  an unresolved problem (e.g. "great, broken again", "third time this happened, fantastic") or
  contradicting a problem mentioned earlier in the history are signs of sarcasm. If sarcastic is
  0.5 or higher, set "sentiment" to "negative" even though the wording looks positive.
- High risk (account compromise, legal threat, fraud) must be judged INDEPENDENTLY of tone. A
  calm, politely worded message can still be high risk (e.g. "Could you look into an unauthorized
  charge on my account when you get a chance?" is calm AND high risk).
- Never invent facts about the customer's account or situation beyond what the message states.
- All numeric scores must be between 0.0 and 1.0, inclusive.
"""


def _build_user_prompt(message: str, history: Optional[list]) -> str:
    parts = []
    if history:
        parts.append("Conversation history (oldest to newest):")
        for i, turn in enumerate(history, 1):
            parts.append(f"{i}. {turn}")
        parts.append("")
    parts.append("Current message to classify:")
    parts.append(message)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing / validation - never invent missing fields, just fail
# clearly so the caller can fall back.
# ---------------------------------------------------------------------------

REQUIRED_SCORE_KEYS = ("positive", "neutral", "negative", "frustrated", "urgent", "sarcastic")
VALID_SENTIMENTS = ("positive", "neutral", "negative")
VALID_HIGH_RISK_CATEGORIES = ("account_compromise", "legal_threat", "fraud", None)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text


def _clamp01(x) -> float:
    """Clamp a value to [0.0, 1.0] range. Raise if invalid."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise LLMResponseError(f"expected a number between 0 and 1, got {x!r}")
    if not (0.0 <= x <= 1.0):
        raise LLMResponseError(f"score {x} is outside the required 0.0-1.0 range")
    return round(x, 2)


def _parse_and_validate(raw_text: str) -> dict:
    """Parse JSON and validate all required fields and ranges.
    Raises LLMResponseError if anything is invalid.
    Never invents missing data.
    """
    cleaned = _strip_code_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"model reply was not valid JSON: {e}")

    # Check all required top-level fields
    for key in ("language", "sentiment", "scores", "intent", "intent_confidence",
                "high_risk", "high_risk_category", "high_risk_confidence"):
        if key not in parsed:
            raise LLMResponseError(f"model reply is missing required field '{key}'")

    # Validate sentiment
    if parsed["sentiment"] not in VALID_SENTIMENTS:
        raise LLMResponseError(f"invalid sentiment '{parsed['sentiment']}'")

    # Validate and clamp scores
    scores = parsed["scores"]
    if not isinstance(scores, dict):
        raise LLMResponseError(f"scores field must be a dict, got {type(scores)}")
    for key in REQUIRED_SCORE_KEYS:
        if key not in scores:
            raise LLMResponseError(f"scores object is missing '{key}'")
        scores[key] = _clamp01(scores[key])

    # Validate intent confidence
    parsed["intent_confidence"] = _clamp01(parsed["intent_confidence"])

    # Validate high_risk flag
    if not isinstance(parsed["high_risk"], bool):
        raise LLMResponseError("'high_risk' must be a boolean")

    # Validate high_risk_category
    if parsed["high_risk_category"] not in VALID_HIGH_RISK_CATEGORIES:
        raise LLMResponseError(f"invalid high_risk_category '{parsed['high_risk_category']}'")

    # Validate high_risk_confidence
    parsed["high_risk_confidence"] = _clamp01(parsed["high_risk_confidence"])

    return parsed


def _result_from_llm_json(message: str, parsed: dict) -> DetectionResult:
    """Convert validated JSON into a DetectionResult."""
    scores = parsed["scores"]
    calm_high_risk = (
        parsed["high_risk"]
        and parsed["sentiment"] != "negative"
        and scores["frustrated"] < 0.3
        and scores["urgent"] < 0.3
    )
    return DetectionResult(
        message=message,
        sentiment=parsed["sentiment"],
        sentiment_scores=scores,
        intent=parsed["intent"],
        intent_confidence=parsed["intent_confidence"],
        high_risk=parsed["high_risk"],
        high_risk_category=parsed["high_risk_category"],
        high_risk_confidence=parsed["high_risk_confidence"],
        calm_high_risk=calm_high_risk,
        language=parsed.get("language"),
        source="llm",
        classification_status="ok",
        notes=["classified automatically by the language model"],
    )


# ---------------------------------------------------------------------------
# Deterministic fallback (used only if no client is given, or the model
# call/parse/validation fails). Small, readable, English-centric - it exists
# purely as a safety net, not as the primary multilingual engine anymore.
# ---------------------------------------------------------------------------

_FALLBACK_POSITIVE = ["thank you", "thanks", "great", "amazing", "love", "happy", "excellent"]
_FALLBACK_NEGATIVE = ["bad", "terrible", "awful", "hate", "worst", "broken", "frustrated"]
_FALLBACK_URGENT = ["urgent", "immediately", "asap", "right now", "emergency"]
_FALLBACK_RISK = {
    "account_compromise": ["hacked", "compromised", "unauthorized access", "stolen password"],
    "legal_threat": ["sue", "lawsuit", "legal action", "my lawyer"],
    "fraud": ["fraud", "unauthorized transaction", "fraudulent charge", "identity theft"],
}


def _contains_word(text: str, phrase: str) -> bool:
    """Check if phrase appears as whole word(s) in text."""
    if " " in phrase:
        return phrase in text
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def _fallback_analyze(message: str, history: Optional[list]) -> DetectionResult:
    """Rule-based fallback classifier. English-centric, low fidelity."""
    text = message.lower()

    pos_hits = [p for p in _FALLBACK_POSITIVE if _contains_word(text, p)]
    neg_hits = [p for p in _FALLBACK_NEGATIVE if _contains_word(text, p)]
    urgent_hits = [p for p in _FALLBACK_URGENT if _contains_word(text, p)]

    positive_score = min(1.0, len(pos_hits) / 2) if pos_hits else 0.0
    negative_score = min(1.0, len(neg_hits) / 2) if neg_hits else 0.0
    urgent_score = min(1.0, len(urgent_hits) / 1) if urgent_hits else 0.0
    frustrated_score = negative_score if "again" in text or "still" in text else 0.0

    if negative_score > positive_score and negative_score > 0:
        sentiment = "negative"
    elif positive_score > negative_score and positive_score > 0:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    scores = {
        "positive": round(positive_score, 2),
        "neutral": round(max(0.0, 1.0 - max(positive_score, negative_score)), 2),
        "negative": round(negative_score, 2),
        "frustrated": round(frustrated_score, 2),
        "urgent": round(urgent_score, 2),
        "sarcastic": 0.0,  # fallback does not attempt sarcasm detection
    }

    high_risk_category, high_risk_confidence = None, 0.0
    for category, phrases in _FALLBACK_RISK.items():
        hits = [p for p in phrases if _contains_word(text, p)]
        if hits:
            high_risk_category, high_risk_confidence = category, 1.0
            break

    high_risk = high_risk_category is not None
    calm_high_risk = high_risk and sentiment != "negative" and frustrated_score < 0.3 and urgent_score < 0.3

    intent = "other"
    if "?" in message:
        intent = "general_inquiry"
    if high_risk_category == "account_compromise":
        intent = "account_security"
    elif high_risk_category == "legal_threat":
        intent = "legal_threat"

    return DetectionResult(
        message=message,
        sentiment=sentiment,
        sentiment_scores=scores,
        intent=intent,
        intent_confidence=0.5 if intent != "other" else 0.2,
        high_risk=high_risk,
        high_risk_category=high_risk_category,
        high_risk_confidence=round(high_risk_confidence, 2),
        calm_high_risk=calm_high_risk,
        language="en (assumed; fallback does not detect language)",
        source="fallback_rules",
        classification_status="unavailable",
        notes=["LLM path unavailable/invalid - used the deterministic English-only fallback"],
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_message(
    message: str,
    history: Optional[list] = None,
    client: Optional[ChatModelClient] = None,
    use_fallback_on_error: bool = True,
) -> DetectionResult:
    """Main entry point.

    Args:
        message: the current customer message, in any language.
        history: optional list of prior message strings (oldest to newest).
        client: a ChatModelClient (e.g. GeminiClient()) that does the
            actual automatic, multilingual classification. If omitted, the
            deterministic fallback is used directly.
        use_fallback_on_error: if the client raises, times out, or returns
            something that fails validation, fall back to the deterministic
            rule-based scorer instead of propagating the error. Set to
            False if you'd rather see the raw error.

    Returns:
        DetectionResult. `result.source` tells you whether "llm" or
        "fallback_rules" produced it. `result.classification_status` is
        "ok" or "unavailable".
    """
    if client is None:
        return _fallback_analyze(message, history)

    try:
        user_prompt = _build_user_prompt(message, history)
        raw_reply = client.complete(SYSTEM_PROMPT, user_prompt)
        parsed = _parse_and_validate(raw_reply)
        return _result_from_llm_json(message, parsed)
    except Exception as exc:
        if not use_fallback_on_error:
            raise
        result = _fallback_analyze(message, history)
        result.notes.append(f"LLM path failed ({type(exc).__name__}: {exc}); used fallback instead")
        return result


# ---------------------------------------------------------------------------
# Self-checks / runnable example
# ---------------------------------------------------------------------------

def _canned(language, sentiment, scores, intent, intent_confidence,
            high_risk, high_risk_category, high_risk_confidence) -> str:
    return json.dumps({
        "language": language, "sentiment": sentiment, "scores": scores,
        "intent": intent, "intent_confidence": intent_confidence,
        "high_risk": high_risk, "high_risk_category": high_risk_category,
        "high_risk_confidence": high_risk_confidence,
    })


def _in_range(x):
    return 0.0 <= x <= 1.0


def _run_self_checks():
    print("Running Task 5 Step 1 (v3, Gemini-based) self-checks...\n")

    fake = FakeChatClient({
        "amazing": _canned("en", "positive",
                            {"positive": 1.0, "neutral": 0.0, "negative": 0.0,
                             "frustrated": 0.0, "urgent": 0.0, "sarcastic": 0.0},
                            "compliment", 0.9, False, None, 0.0),
        "third time": _canned("en", "negative",
                               {"positive": 0.0, "neutral": 0.0, "negative": 0.7,
                                "frustrated": 0.8, "urgent": 0.0, "sarcastic": 0.0},
                               "technical_support", 0.8, False, None, 0.0),
        "emergency": _canned("en", "neutral",
                              {"positive": 0.0, "neutral": 0.2, "negative": 0.1,
                               "frustrated": 0.2, "urgent": 1.0, "sarcastic": 0.0},
                              "technical_support", 0.7, False, None, 0.0),
        "just wonderful": _canned("en", "negative",
                                   {"positive": 0.3, "neutral": 0.0, "negative": 0.6,
                                    "frustrated": 0.5, "urgent": 0.0, "sarcastic": 0.9},
                                   "technical_support", 0.6, False, None, 0.0),
        "unauthorized transaction": _canned("en", "neutral",
                                             {"positive": 0.1, "neutral": 0.8, "negative": 0.0,
                                              "frustrated": 0.0, "urgent": 0.0, "sarcastic": 0.0},
                                             "account_security", 0.9, True, "fraud", 0.95),
        "hackeada": _canned("es", "neutral",
                             {"positive": 0.0, "neutral": 0.3, "negative": 0.2,
                              "frustrated": 0.1, "urgent": 1.0, "sarcastic": 0.0},
                             "account_security", 0.9, True, "account_compromise", 0.9),
        "excellent": _canned("fr", "positive",
                              {"positive": 1.0, "neutral": 0.0, "negative": 0.0,
                               "frustrated": 0.0, "urgent": 0.0, "sarcastic": 0.0},
                              "compliment", 0.85, False, None, 0.0),
        "GARBLED": "not valid json at all {{{",
    })

    # --- positive / negative / frustrated, via the automatic (LLM) path ---
    r = analyze_message("Thank you so much, your team is amazing!", client=fake)
    assert r.source == "llm" and r.sentiment == "positive" and r.classification_status == "ok"

    r = analyze_message("This is the third time my order hasn't arrived, I'm so frustrated!", client=fake)
    assert r.source == "llm" and r.sentiment == "negative" and r.sentiment_scores["frustrated"] > 0

    # --- urgent ---
    r = analyze_message("I need this fixed immediately, it's an emergency!", client=fake)
    assert r.source == "llm" and r.sentiment_scores["urgent"] >= 0.5

    # --- sarcasm ---
    r = analyze_message("Oh great, my package is broken again, just wonderful.", client=fake)
    assert r.source == "llm" and r.sentiment_scores["sarcastic"] >= 0.5 and r.sentiment == "negative"

    # --- calm high-risk ---
    r = analyze_message(
        "Hello, could you look into an unauthorized transaction on my account when you get a chance?",
        client=fake,
    )
    assert r.source == "llm" and r.high_risk is True and r.calm_high_risk is True

    # --- multilingual (Spanish, French) - model reads native language directly ---
    r_es = analyze_message("Necesito ayuda urgente, mi cuenta fue hackeada!", client=fake)
    assert r_es.source == "llm" and r_es.high_risk_category == "account_compromise"
    assert r_es.language == "es"

    r_fr = analyze_message("Merci beaucoup, c'était excellent.", client=fake)
    assert r_fr.source == "llm" and r_fr.sentiment == "positive" and r_fr.language == "fr"

    # --- confidence scores stay valid probabilities ---
    for result in (r, r_es, r_fr):
        for v in result.sentiment_scores.values():
            assert _in_range(v), f"sentiment score out of range: {v}"
        assert _in_range(result.intent_confidence), f"intent_confidence out of range: {result.intent_confidence}"
        assert _in_range(result.high_risk_confidence), f"high_risk_confidence out of range: {result.high_risk_confidence}"

    # --- graceful fallback when the model reply is malformed ---
    r = analyze_message("GARBLED input that breaks the fake model", client=fake)
    assert r.source == "fallback_rules" and r.classification_status == "unavailable"
    assert any("failed" in note.lower() for note in r.notes)

    # --- fallback also works with no client at all (offline / dev mode) ---
    r_no_client = analyze_message("This is terrible, still broken again!")
    assert r_no_client.source == "fallback_rules" and r_no_client.classification_status == "unavailable"

    # --- deterministic output: same fake-client input twice -> identical result ---
    a = analyze_message("Thank you so much, your team is amazing!", client=fake)
    b = analyze_message("Thank you so much, your team is amazing!", client=fake)
    assert a.sentiment_scores == b.sentiment_scores and a.intent == b.intent

    # --- API failure gracefully falls back ---
    def failing_client(sp, up):
        raise TimeoutError("API timeout")
    
    r_timeout = analyze_message("test message", client=type('C', (), {'complete': failing_client})())
    assert r_timeout.source == "fallback_rules" and r_timeout.classification_status == "unavailable"

    print("✓ All Task 5 Step 1 (v3, Gemini-based) self-checks passed.\n")

    print("Demo: automatic classification via (fake) Gemini client")
    demo = analyze_message(
        "Hello, could you look into an unauthorized transaction on my account when you get a chance?",
        client=fake,
    )
    print(f"  source={demo.source} status={demo.classification_status} language={demo.language}")
    print(f"  sentiment={demo.sentiment} scores={demo.sentiment_scores}")
    print(f"  high_risk={demo.high_risk} category={demo.high_risk_category} calm_high_risk={demo.calm_high_risk}")
    print(f"  intent={demo.intent} (confidence={demo.intent_confidence})")

    print("\nTo use the real model in production:")
    print("  from task5_step1_sentiment_intent_detection import analyze_message, GeminiClient")
    print("  result = analyze_message(message, history, client=GeminiClient())")
    print("\nSetup:")
    print("  1. Install: pip install google-genai")
    print("  2. Get API key: https://aistudio.google.com/apikey")
    print("  3. Set environment: export GEMINI_API_KEY='your-key-here'")


if __name__ == "__main__":
    _run_self_checks()
