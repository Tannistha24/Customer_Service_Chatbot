        
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

GENESIS_HASH = "0" * 64
MAX_STR = 300
MAX_LIST = 20
MAX_DEPTH = 5

# ---------------------------------------------------------------- sanitizing

_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|passw|secret|token|authoriz|credential|e-?mail|phone|ssn|"
    r"address|customer_name|user_name|full_name|transcript|raw_|message_text|messages)",
    re.IGNORECASE,
)
_VALUE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(password|passwd|pwd|api[_-]?key|token|secret)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"\b(?:sk|pk|AIza|ghp|xox[bp])[-_A-Za-z0-9]{10,}"), "[REDACTED_KEY]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[REDACTED_EMAIL]"),
    (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "[REDACTED_NUMBER]"),
]


def _scrub_text(text: str) -> str:
    for pattern, repl in _VALUE_PATTERNS:
        text = pattern.sub(repl, text)
    return text if len(text) <= MAX_STR else text[:MAX_STR] + "...[truncated]"


def sanitize(value: Any, depth: int = 0) -> Any:
    """Return a JSON-safe copy with secrets/PII removed and size capped."""
    if depth > MAX_DEPTH:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, (bytes, bytearray)):
        return "[BYTES]"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k in sorted(value, key=str)[:MAX_LIST * 2]:
            key = str(k)
            out[key] = "[REDACTED]" if _SENSITIVE_KEY.search(key) else sanitize(value[k], depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else list(value)
        return [sanitize(v, depth + 1) for v in items[:MAX_LIST]]
    return _scrub_text(str(value))


# ------------------------------------------------------------------- helpers

def _as_dict(x: Any) -> Tuple[Dict[str, Any], bool]:
    return (x, True) if isinstance(x, dict) else ({}, False)


def _pick(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "y"):
            return True
        if s in ("false", "no", "0", "n"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def _parse_ts(v: Any, strict: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """Normalize a datetime / ISO string to UTC ISO-8601.

    Returns (timestamp_or_None, note_or_None). A naive (timezone-less) timestamp
    is NOT silently trusted: by default it is interpreted as UTC and a note is
    returned (compatibility); with strict=True it is rejected.
    """
    if v is None:
        return None, None
    try:
        if isinstance(v, datetime):
            dt = v
        elif isinstance(v, str) and v.strip():
            dt = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        else:
            return None, "unusable timestamp ignored"
        note = None
        if dt.tzinfo is None:
            if strict:
                return None, "naive timestamp rejected (strict mode)"
            note = "naive timestamp (no timezone) interpreted as UTC - unverified"
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(), note
    except (ValueError, TypeError):
        return None, "unparseable timestamp ignored"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Boolean flags (from any step) that count as "activated conditions" only when
# Step 3 did not already list its own conditions explicitly.
# Sarcasm is deliberately NOT listed: it is context, not an escalation trigger.
_CONDITION_FLAGS = (
    "threat_detected", "high_risk", "timer_expired",
    "unresolved_negative", "urgent", "after_hours",
)
_NO_ROUTE = {"", "NONE", "N/A", "NULL"}


# -------------------------------------------------------------------- record

@dataclass(frozen=True)
class AuditRecord:
    audit_id: str
    sequence: int
    timestamp: Optional[str]
    timestamp_source: str            # "caller" | "step_data" | "system_clock" | "unavailable"
    conversation_ref: Optional[str]  # pseudonymous hash, never the raw id
    decision: str                    # ESCALATE | ROUTE | NO_ESCALATION | UNKNOWN
    escalated: Optional[bool]        # Step 3's explicit escalation flag (independent of route)
    route: Optional[str]
    reason: str
    conditions: List[str]
    conversation_summary: str
    step_data: Dict[str, Any]
    data_warnings: List[str]
    prev_hash: str
    record_hash: str = field(default="")

    def body(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("record_hash", None)
        return d

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# -------------------------------------------------------------------- logger

class AuditLogger:
    """Append-only, hash-chained audit log (in memory, optional JSONL file)."""

    def __init__(self, path: Optional[str] = None,
                 clock: Optional[Callable[[], datetime]] = None,
                 strict_timestamps: bool = False):
        self._strict_ts = strict_timestamps  # True: reject naive timestamps
        self._path = path
        self._clock = clock  # only used if no timestamp is supplied anywhere
        self._records: List[AuditRecord] = []

    @property
    def records(self) -> List[AuditRecord]:
        return list(self._records)

    def log_decision(self, step1: Any, step2: Any, step3: Any, step4: Any,
                     conversation_summary: Optional[str] = None,
                     conversation_id: Optional[str] = None,
                     timestamp: Any = None) -> AuditRecord:
        warnings: List[str] = []
        steps: Dict[str, Dict[str, Any]] = {}
        for name, raw in (("step1", step1), ("step2", step2), ("step3", step3), ("step4", step4)):
            d, ok = _as_dict(raw)
            if not ok:
                warnings.append(f"{name}: missing or invalid input (expected dict)")
            steps[name] = d
        s1, s2, s3, s4 = steps["step1"], steps["step2"], steps["step3"], steps["step4"]

        # --- decision & route (read from Step 3, never recomputed)
        esc = _as_bool(_pick(s3, "escalate", "should_escalate", "escalation_required", "escalated"))
        route_raw = _pick(s3, "route", "routing", "destination")
        route = str(route_raw).strip().upper() if route_raw is not None else None
        if route in _NO_ROUTE:
            route = None
        # Escalation is only what Step 3 explicitly says; routing alone is not escalation.
        if esc is True:
            decision = "ESCALATE"
        elif route:
            decision = "ROUTE"
        elif esc is False:
            decision = "NO_ESCALATION"
        else:
            decision = "UNKNOWN"
            warnings.append("step3: no escalation flag or route found")

        reason = _pick(s3, "reason", "escalation_reason", "explanation")
        if not reason:
            warnings.append("step3: no reason provided")
        reason = _scrub_text(str(reason)) if reason else "not provided"

        # --- activated conditions
        conds = _pick(s3, "conditions", "activated_conditions", "triggers")
        if isinstance(conds, str):
            conds = [conds]
        if isinstance(conds, (list, tuple, set)) and conds:
            conditions = sorted({_scrub_text(str(c)) for c in conds})
        else:
            conditions = sorted({
                flag for s in (s1, s2, s3, s4) for flag in _CONDITION_FLAGS
                if _as_bool(s.get(flag)) is True
            })
            if not conditions and decision in ("ESCALATE", "ROUTE"):
                warnings.append("no activated conditions reported")

        # --- summary (never raw messages)
        summary = conversation_summary or _pick(s3, "conversation_summary", "summary") \
            or _pick(s1, "conversation_summary", "summary")
        if not summary:
            warnings.append("no conversation summary provided")
        summary = _scrub_text(str(summary)) if summary else "not provided"

        # --- timestamp (never invented silently)
        ts, ts_src = None, "unavailable"
        for src, cand in (("caller", timestamp),
                          ("step_data", _pick(s4, "timestamp", "evaluated_at", "now")),
                          ("step_data", _pick(s3, "timestamp", "evaluated_at"))):
            ts, note = _parse_ts(cand, self._strict_ts)
            if note:
                warnings.append(f"timestamp ({src}): {note}")
            if ts:
                ts_src = src
                break
        if ts is None and self._clock is not None:
            ts, note = _parse_ts(self._clock(), self._strict_ts)
            if note:
                warnings.append(f"timestamp (system_clock): {note}")
            if ts:
                ts_src = "system_clock"
        if ts is None:
            warnings.append("no valid timestamp available")

        conv_ref = ("conv_" + _sha(str(conversation_id))[:12]) if conversation_id else None

        prev = self._records[-1].record_hash if self._records else GENESIS_HASH
        seq = len(self._records) + 1
        rec = AuditRecord(
            audit_id="", sequence=seq, timestamp=ts, timestamp_source=ts_src,
            conversation_ref=conv_ref, decision=decision, escalated=esc, route=route, reason=reason,
            conditions=conditions, conversation_summary=summary,
            step_data={k: sanitize(v) for k, v in steps.items()},
            data_warnings=warnings, prev_hash=prev,
        )
        rec_hash = _sha(prev + _canonical({**rec.body(), "audit_id": ""}))
        object.__setattr__(rec, "record_hash", rec_hash)
        object.__setattr__(rec, "audit_id", f"aud-{seq:06d}-{rec_hash[:8]}")
        self._records.append(rec)
        self._write(rec)
        return rec

    def _write(self, rec: AuditRecord) -> None:
        if not self._path:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(_canonical(rec.to_dict()) + "\n")
        except OSError as exc:
            # Keep the in-memory record; surface the problem without raising.
            self._records[-1].data_warnings.append(f"file write failed: {type(exc).__name__}")

    # ---------------------------------------------------------------- audit
    def verify_chain(self) -> bool:
        """True if no record was altered, removed or reordered."""
        prev = GENESIS_HASH
        for i, r in enumerate(self._records, start=1):
            expected_id = f"aud-{i:06d}-{r.record_hash[:8]}"
            recomputed = _sha(prev + _canonical({**r.body(), "audit_id": ""}))
            if r.sequence != i or r.prev_hash != prev or r.record_hash != recomputed \
                    or r.audit_id != expected_id:
                return False
            prev = r.record_hash
        return True

    @staticmethod
    def explain(rec: AuditRecord) -> str:
        """Human-readable what / why / which condition / when."""
        when = rec.timestamp or "unknown time"
        lines = [
            f"[{rec.audit_id}] {when} (time source: {rec.timestamp_source})",
            f"  What:      decision={rec.decision}, escalated={rec.escalated}"
            + (f", route={rec.route}" if rec.route else ""),
            f"  Why:       {rec.reason}",
            f"  Triggered: {', '.join(rec.conditions) if rec.conditions else 'none reported'}",
            f"  Summary:   {rec.conversation_summary}",
        ]
        if rec.data_warnings:
            lines.append("  Warnings:  " + "; ".join(rec.data_warnings))
        return "\n".join(lines)

    def export(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records]


# ---------------------------------------------------------- evaluation harness
# FIXTURE MODE (default): each scenario's "fixture" is a SAMPLE of Step 1-4
# outputs (hand-written data, NOT produced by the real steps). It verifies that the
# audit layer records/explains such outputs correctly.
# END-TO-END MODE: pass pipeline_fn(scenario_input) -> (step1, step2, step3, step4)
# to run the real Steps 1-4 on each scenario's "input" text instead.
# "expect" describes the required outcome; escalation and routing are asserted
# independently ("escalated" vs "route").

_TS = "2026-03-10T22:30:00+00:00"

SCENARIOS: List[Dict[str, Any]] = [
    {"name": "sarcasm", "input": "Wow, what a fantastic job you did. Truly brilliant work.",
     "fixture": ({"sentiment": "negative", "sarcasm_detected": True},
                 {"risk_level": "low", "response_tone": "professional_deescalating"},
                 {"escalate": False, "route": None,
                  "reason": "Sarcastic compliment read as negative context; handled with a professional "
                            "de-escalating tone, no escalation required",
                  "conditions": []},
                 {"timestamp": _TS, "elapsed_minutes": 2}),
     "expect": {"decision": "NO_ESCALATION", "escalated": False, "route": None,
                "no_condition": "sarcas", "sarcasm": True, "tone": ("professional", "escalat")}},
    {"name": "calm_high_risk_threat", "input": "I will calmly explain what I plan to do to your staff.",
     "fixture": ({"sentiment": "neutral", "tone": "calm"},
                 {"risk_level": "high", "threat_detected": True},
                 {"escalate": True, "route": "LIVE_AGENT", "reason": "Calm tone but threat detected",
                  "conditions": ["threat_detected", "high_risk"]},
                 {"timestamp": _TS, "elapsed_minutes": 1}),
     "expect": {"decision": "ESCALATE", "escalated": True, "condition": "threat"}},
    {"name": "urgent_after_hours_on_call", "input": "Production is down, need help now.",
     "fixture": ({"sentiment": "negative"}, {"risk_level": "high"},
                 {"escalate": True, "route": "ON_CALL", "reason": "Urgent issue outside business hours",
                  "conditions": ["urgent", "after_hours"]},
                 {"timestamp": _TS, "after_hours": True}),
     "expect": {"decision": "ESCALATE", "escalated": True, "route": "ON_CALL", "condition": "urgent"}},
    {"name": "routine_after_hours_next_day", "input": "Question about my billing invoice format.",
     "fixture": ({"sentiment": "neutral"}, {"risk_level": "low"},
                 {"escalate": False, "route": "NEXT_WORKING_DAY",
                  "reason": "Routine billing question outside business hours",
                  "conditions": ["after_hours"]},
                 {"timestamp": _TS, "after_hours": True}),
     "expect": {"decision": "ROUTE", "escalated": False, "route": "NEXT_WORKING_DAY",
                "condition": "after_hours"}},
    {"name": "unresolved_negative_16_min", "input": "Still not fixed, this is unacceptable.",
     "fixture": ({"sentiment": "negative"}, {"risk_level": "medium"},
                 {"escalate": True, "route": "LIVE_AGENT", "reason": "Negative sentiment unresolved past 15 minutes",
                  "conditions": ["unresolved_negative_timer"]},
                 {"timestamp": "2026-03-10T10:16:00+00:00", "elapsed_minutes": 16, "timer_expired": True}),
     "expect": {"decision": "ESCALATE", "escalated": True, "condition": "unresolved_negative", "elapsed": 16}},
]


def _run_scenario(sc: Dict[str, Any], logger: AuditLogger,
                  pipeline_fn: Optional[Callable[[Any], Tuple[Any, Any, Any, Any]]]) -> AuditRecord:
    s1, s2, s3, s4 = pipeline_fn(sc["input"]) if pipeline_fn else sc["fixture"]
    return logger.log_decision(s1, s2, s3, s4, conversation_summary=f"Scenario: {sc['name']}",
                               conversation_id=sc["name"])


def run_self_checks(pipeline_fn: Optional[Callable[[Any], Tuple[Any, Any, Any, Any]]] = None,
                    verbose: bool = True) -> bool:
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    logger = AuditLogger()
    for sc in SCENARIOS:
        rec = _run_scenario(sc, logger, pipeline_fn)
        exp = sc["expect"]
        ok = rec.decision == exp["decision"]
        ok = ok and rec.escalated is exp["escalated"]              # escalation checked on its own
        ok = ok and (("route" not in exp) or rec.route == exp["route"])  # routing checked on its own
        if "condition" in exp:
            ok = ok and any(exp["condition"] in c for c in rec.conditions)
        if "no_condition" in exp:                                   # e.g. sarcasm is not a trigger
            ok = ok and not any(exp["no_condition"] in c.lower() for c in rec.conditions)
        if exp.get("sarcasm"):                                      # sarcasm + negative context recorded
            s1 = rec.step_data["step1"]
            ok = ok and s1.get("sarcasm_detected") is True and str(s1.get("sentiment", "")).lower() == "negative"
        if "tone" in exp:                                           # Step 2 tone is professional/de-escalating
            s2 = rec.step_data["step2"]
            tone = str(s2.get("response_tone") or s2.get("tone") or "").lower()
            ok = ok and any(t in tone for t in exp["tone"])
        ok = ok and rec.timestamp is not None and rec.reason != "not provided"
        if "elapsed" in exp:
            ok = ok and rec.step_data["step4"].get("elapsed_minutes") == exp["elapsed"]
        check(sc["name"], ok, f"decision={rec.decision} escalated={rec.escalated} route={rec.route} "
                              f"conditions={rec.conditions}")

    check("chain_verifies", logger.verify_chain())

    # Determinism: same inputs, fresh loggers -> identical hashes.
    a, b = AuditLogger(), AuditLogger()
    for sc in SCENARIOS:
        _run_scenario(sc, a, pipeline_fn)
        _run_scenario(sc, b, pipeline_fn)
    check("deterministic_repeat", [r.record_hash for r in a.records] == [r.record_hash for r in b.records])

    # Tamper detection.
    t = AuditLogger()
    _run_scenario(SCENARIOS[0], t, pipeline_fn)
    object.__setattr__(t._records[0], "reason", "tampered")
    check("tamper_detected", not t.verify_chain())

    # Safety: secrets / PII removed, bad input handled without raising.
    s = AuditLogger()
    rec = s.log_decision(
        {"sentiment": "negative", "api_key": "sk-abcdefghijklmnop", "customer_email": "a@b.com",
         "note": "call me on +1 415 555 0100, password: hunter2"},
        "not-a-dict", None, {"timestamp": "garbage"}, conversation_summary="Contact x@y.org")
    blob = _canonical(rec.to_dict())
    leaked = any(x in blob for x in ("sk-abcdefghij", "a@b.com", "hunter2", "415 555", "x@y.org"))
    check("no_secrets_or_pii", not leaked)
    check("bad_input_safe", rec.decision == "UNKNOWN" and len(rec.data_warnings) >= 3)

    # Timestamp safety: naive timestamps warn (default) or are rejected (strict).
    naive = {"timestamp": "2026-03-10T10:00:00"}
    r_default = AuditLogger().log_decision({}, {}, {"escalate": False, "reason": "x"}, naive)
    r_strict = AuditLogger(strict_timestamps=True).log_decision({}, {}, {"escalate": False, "reason": "x"}, naive)
    check("naive_timestamp_warned", r_default.timestamp is not None
          and any("naive" in w for w in r_default.data_warnings))
    check("naive_timestamp_strict_rejected", r_strict.timestamp is None
          and any("naive" in w for w in r_strict.data_warnings))

    passed = all(ok for _, ok, _ in results)
    if verbose:
        print("Mode: " + ("END-TO-END via pipeline_fn" if pipeline_fn
                          else "FIXTURE (sample Step 1-4 outputs, not real pipeline results)") + "\n")
        for name, ok, detail in results:
            print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
        print("\nSample audit explanation:\n" + AuditLogger.explain(logger.records[2]))
        print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run_self_checks() else 1)