
from __future__ import annotations

import os
import time
import uuid
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "idle_timeout_seconds": 1800,          # 30 min
    "summary_retention_seconds": 86400,    # 24 hr
    # Base language ("en") + 3 additional configurable languages by default.
    "supported_languages": ["en", "es", "fr", "de"],
    "language_confidence_threshold": 0.60,
    "intent_confidence_threshold": 0.75,
    "min_retained_messages": 10,
    "max_history_length": 50,
}

_REQUIRED_TYPES = {
    "idle_timeout_seconds": (int, float),
    "summary_retention_seconds": (int, float),
    "supported_languages": list,
    "language_confidence_threshold": (int, float),
    "intent_confidence_threshold": (int, float),
    "min_retained_messages": int,
    "max_history_length": int,
}


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    """
    Tiny YAML-subset parser covering exactly what this config needs: flat
    `key: value` pairs and simple block lists:

        key: value
        list_key:
          - item1
          - item2

    Used only as a fallback when PyYAML is not installed, so this module
    carries no hard third-party dependency.
    """
    data: Dict[str, Any] = {}
    current_list_key: Optional[str] = None

    def _coerce(raw: str) -> Any:
        raw = raw.strip()
        if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
            return raw[1:-1]
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        try:
            return float(raw) if "." in raw else int(raw)
        except ValueError:
            return raw

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- ") and current_list_key is not None:
            data.setdefault(current_list_key, [])
            data[current_list_key].append(_coerce(stripped[2:]))
            continue

        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key, value = key.strip(), value.strip()
            if value == "":
                current_list_key = key
                data.setdefault(key, [])
            elif value.startswith("[") and value.endswith("]"):
                inner = value[1:-1].strip()
                data[key] = [] if not inner else [_coerce(v) for v in inner.split(",")]
                current_list_key = None
            else:
                data[key] = _coerce(value)
                current_list_key = None
        # Unsupported syntax is silently ignored -> falls through to safe defaults.
    return data


def _load_raw_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        return _minimal_yaml_parse(text)


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from `path`, validate every field, and merge with
    safe defaults. Never raises: a missing file, parse error, or invalid
    value silently falls back to the corresponding default so the system
    always ends up with a usable, safe configuration. Warnings (if any)
    are collected under config["_warnings"].

    Returns a fresh dict (never a shared mutable reference).
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    warnings: List[str] = []

    if not path or not os.path.isfile(path):
        return config

    try:
        raw = _load_raw_yaml(path)
    except Exception as exc:  # noqa: BLE001 - any load failure -> safe defaults
        warnings.append(f"Failed to read/parse '{path}': {exc}. Using defaults.")
        config["_warnings"] = warnings
        return config

    for key, default_value in DEFAULT_CONFIG.items():
        if key not in raw:
            continue
        value = raw[key]
        expected_types = _REQUIRED_TYPES[key]
        if not isinstance(value, expected_types) or isinstance(value, bool):
            warnings.append(
                f"Invalid type for '{key}' ({value!r}); keeping default {default_value!r}."
            )
            continue
        config[key] = value

    _validate_and_fix(config, warnings)
    if warnings:
        config["_warnings"] = warnings
    return config


def _validate_and_fix(config: Dict[str, Any], warnings: List[str]) -> None:
    """In-place range/semantic validation with safe-default fallback."""
    if config["idle_timeout_seconds"] <= 0:
        warnings.append("idle_timeout_seconds must be > 0; reverting to default.")
        config["idle_timeout_seconds"] = DEFAULT_CONFIG["idle_timeout_seconds"]

    if config["summary_retention_seconds"] <= config["idle_timeout_seconds"]:
        warnings.append(
            "summary_retention_seconds must exceed idle_timeout_seconds; reverting to default."
        )
        config["summary_retention_seconds"] = DEFAULT_CONFIG["summary_retention_seconds"]

    for key in ("language_confidence_threshold", "intent_confidence_threshold"):
        if not (0.0 <= config[key] <= 1.0):
            warnings.append(f"{key} must be in [0, 1]; reverting to default.")
            config[key] = DEFAULT_CONFIG[key]

    cleaned = [str(l).strip() for l in config["supported_languages"] if str(l).strip()]
    deduped = list(dict.fromkeys(cleaned))
    if not deduped:
        warnings.append("supported_languages empty/invalid; reverting to default.")
        deduped = list(DEFAULT_CONFIG["supported_languages"])
    config["supported_languages"] = deduped

    if config["min_retained_messages"] < 1:
        warnings.append("min_retained_messages must be >= 1; reverting to default.")
        config["min_retained_messages"] = DEFAULT_CONFIG["min_retained_messages"]

    if config["max_history_length"] < config["min_retained_messages"]:
        warnings.append(
            "max_history_length must be >= min_retained_messages; reverting to default."
        )
        config["max_history_length"] = max(
            DEFAULT_CONFIG["max_history_length"], config["min_retained_messages"]
        )


# ---------------------------------------------------------------------------
# 2. SESSION DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str
    text: str
    timestamp: float


@dataclass
class Session:
    session_id: str
    customer_id: str
    created_timestamp: float
    last_active_timestamp: float
    message_history: List[Message] = field(default_factory=list)
    conversation_summary: str = ""
    dialogue_state: Dict[str, Any] = field(default_factory=dict)
    lifecycle_status: str = "new"  # new | active | restored_from_summary | expired_fresh


# ---------------------------------------------------------------------------
# 3. SESSION MANAGER
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Tracks one active session per customer_id and applies deterministic
    lifecycle rules on access. Sessions are fully isolated: each
    customer_id maps to its own independent Session object.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 clock: Optional[Callable[[], float]] = None):
        self.config = config if config is not None else load_config()
        self.clock = clock or time.time
        self._sessions: Dict[str, Session] = {}

    # -- internal helpers -----------------------------------------------

    def _new_session_id(self, customer_id: str, at_time: float) -> str:
        # Deterministic given (customer_id, at_time) for reproducible
        # tests, while remaining unique across customers/times.
        seed = f"{customer_id}:{at_time}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))

    def _create_fresh_session(self, customer_id: str, at_time: float,
                               status: str = "new") -> Session:
        session = Session(
            session_id=self._new_session_id(customer_id, at_time),
            customer_id=customer_id,
            created_timestamp=at_time,
            last_active_timestamp=at_time,
            message_history=[],
            conversation_summary="",
            dialogue_state={},
            lifecycle_status=status,
        )
        self._sessions[customer_id] = session
        return session

    def _generate_summary(self, session: Session) -> str:
        """
        Deterministic, non-NLU placeholder summary generator. Real
        summarization is explicitly out of scope for Step 1 - this only
        proves the architecture (summary slot, restore path) works.
        """
        count = len(session.message_history)
        if count == 0:
            return session.conversation_summary or "(empty conversation)"
        last = session.message_history[-1]
        return f"[{count} messages] last: {last.role}: {last.text[:60]}"

    def _apply_lifecycle(self, session: Session, at_time: float) -> Session:
        idle = at_time - session.last_active_timestamp
        idle_timeout = self.config["idle_timeout_seconds"]
        retention = self.config["summary_retention_seconds"]

        if idle <= idle_timeout:
            session.lifecycle_status = "active"
            return session

        if idle <= retention:
            if not session.conversation_summary:
                session.conversation_summary = self._generate_summary(session)
            session.message_history = []
            session.lifecycle_status = "restored_from_summary"
            # NOTE: do NOT update last_active_timestamp here; it must always
            # represent the customer's actual last activity, not a lookup/restore time.
            return session

        # idle > retention -> discard everything, fresh session
        return self._create_fresh_session(session.customer_id, at_time,
                                           status="expired_fresh")

    # -- public API -------------------------------------------------------

    def get_session(self, customer_id: str, at_time: Optional[float] = None) -> Session:
        """Fetch (creating if needed) the session for customer_id, applying
        lifecycle rules relative to `at_time` (injected for determinism;
        defaults to self.clock())."""
        now = self.clock() if at_time is None else at_time
        existing = self._sessions.get(customer_id)
        if existing is None:
            return self._create_fresh_session(customer_id, now)
        return self._apply_lifecycle(existing, now)

    def add_message(self, customer_id: str, role: str, text: str,
                     at_time: Optional[float] = None) -> Session:
        now = self.clock() if at_time is None else at_time
        session = self.get_session(customer_id, at_time=now)
        session.message_history.append(Message(role=role, text=text, timestamp=now))
        max_len = self.config["max_history_length"]
        if len(session.message_history) > max_len:
            session.message_history = session.message_history[-max_len:]
        session.last_active_timestamp = now
        session.lifecycle_status = "active"
        return session

    def get_history(self, customer_id: str) -> List[Message]:
        session = self._sessions.get(customer_id)
        return list(session.message_history) if session else []

    def reset(self) -> None:
        self._sessions.clear()


# ---------------------------------------------------------------------------
# 4. SELF-CHECKS + UNIT TESTS
# ---------------------------------------------------------------------------

def _run_self_checks() -> None:
    failures = []

    def check(name, condition):
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")
        if not condition:
            failures.append(name)

    fake_time = {"t": 1_000_000.0}
    clock = lambda: fake_time["t"]

    cfg = load_config("does_not_exist.yaml")  # -> defaults
    mgr = SessionManager(config=cfg, clock=clock)

    # --- <=30 min retains history ---
    for i in range(5):
        mgr.add_message("cust_A", "user", f"msg {i}", at_time=fake_time["t"])
        fake_time["t"] += 60
    fake_time["t"] += 20 * 60  # idle ~20 min, < 30 min timeout
    session_a = mgr.get_session("cust_A", at_time=fake_time["t"])
    check("<=30 min idle retains full history", len(session_a.message_history) == 5)
    check("<=30 min idle status is 'active'", session_a.lifecycle_status == "active")

    # --- 10+ message history supported ---
    fake_time["t"] += 1
    for i in range(12):
        mgr.add_message("cust_A", "user", f"extra {i}", at_time=fake_time["t"])
        fake_time["t"] += 1
    session_a = mgr.get_session("cust_A", at_time=fake_time["t"])
    check("10+ messages retained in history", len(session_a.message_history) >= 10)

    # --- 31 min idle -> summary context, raw history cleared ---
    pre_len = len(session_a.message_history)
    fake_time["t"] += 31 * 60
    session_a_31 = mgr.get_session("cust_A", at_time=fake_time["t"])
    check("31 min idle -> 'restored_from_summary' status",
          session_a_31.lifecycle_status == "restored_from_summary")
    check("31 min idle produces non-empty summary", bool(session_a_31.conversation_summary))
    check("31 min idle clears raw detailed history",
          len(session_a_31.message_history) == 0 and pre_len > 0)
    check("31 min idle keeps same session_id (no reset)",
          session_a_31.session_id == session_a.session_id)

    # --- Timestamp correctness: last_active_timestamp NOT updated on restore ---
    last_active_before = session_a_31.last_active_timestamp
    fake_time["t"] += 10 * 60  # Advance 10 more minutes (still within 24hr window)
    session_a_check = mgr.get_session("cust_A", at_time=fake_time["t"])
    check("restore does NOT update last_active_timestamp",
          session_a_check.last_active_timestamp == last_active_before)
    check("lookup at T+10min still in restored_from_summary (not expired)",
          session_a_check.lifecycle_status == "restored_from_summary")

    # --- 25 hr idle -> brand-new session ---
    old_session_id = session_a_31.session_id
    fake_time["t"] += 23 * 3600  # Total idle now ~24h 41min, exceeds 24h retention
    session_a_25h = mgr.get_session("cust_A", at_time=fake_time["t"])
    check("25 hr idle creates a fresh session (new id)",
          session_a_25h.session_id != old_session_id)
    check("25 hr idle fresh session has empty history",
          len(session_a_25h.message_history) == 0)
    check("25 hr idle fresh session has empty summary",
          session_a_25h.conversation_summary == "")
    check("25 hr idle status is 'expired_fresh'",
          session_a_25h.lifecycle_status == "expired_fresh")

    # --- customer isolation ---
    fake_time["t"] += 1
    mgr.add_message("cust_A", "user", "hello from A", at_time=fake_time["t"])
    mgr.add_message("cust_B", "user", "hello from B", at_time=fake_time["t"])
    hist_a, hist_b = mgr.get_history("cust_A"), mgr.get_history("cust_B")
    check("isolation: B does not see A's messages", all("from A" not in m.text for m in hist_b))
    check("isolation: A does not see B's messages", all("from B" not in m.text for m in hist_a))
    check("isolation: distinct session ids per customer",
          mgr.get_session("cust_A", fake_time["t"]).session_id !=
          mgr.get_session("cust_B", fake_time["t"]).session_id)

    # --- configurable thresholds/timings ---
    custom_cfg = copy.deepcopy(DEFAULT_CONFIG)
    custom_cfg["idle_timeout_seconds"] = 60
    custom_cfg["summary_retention_seconds"] = 300
    custom_cfg["intent_confidence_threshold"] = 0.9
    fake_time2 = {"t": 5_000_000.0}
    mgr2 = SessionManager(config=custom_cfg, clock=lambda: fake_time2["t"])
    mgr2.add_message("cust_C", "user", "hi", at_time=fake_time2["t"])
    fake_time2["t"] += 90  # > custom 60s timeout, <= 300s retention
    s_c = mgr2.get_session("cust_C", at_time=fake_time2["t"])
    check("custom config: shorter idle_timeout honored",
          s_c.lifecycle_status == "restored_from_summary")
    check("custom config: intent_confidence_threshold honored",
          mgr2.config["intent_confidence_threshold"] == 0.9)

    # --- 30-minute boundary (exact threshold) ---
    fake_time3 = {"t": 2_000_000.0}
    mgr3 = SessionManager(config=cfg, clock=lambda: fake_time3["t"])
    mgr3.add_message("cust_D", "user", "boundary test", at_time=fake_time3["t"])
    last_msg_time = fake_time3["t"]
    fake_time3["t"] += 1800  # exactly 30 min (idle_timeout_seconds)
    s_d_exact = mgr3.get_session("cust_D", at_time=fake_time3["t"])
    check("30-minute exact boundary stays 'active'", s_d_exact.lifecycle_status == "active")
    fake_time3["t"] += 1  # one more second -> > 30 min
    s_d_over = mgr3.get_session("cust_D", at_time=fake_time3["t"])
    check("30-minute + 1sec boundary transitions to 'restored_from_summary'",
          s_d_over.lifecycle_status == "restored_from_summary")

    # --- determinism ---
    def scenario():
        ft = {"t": 42.0}
        m = SessionManager(config=copy.deepcopy(DEFAULT_CONFIG), clock=lambda: ft["t"])
        m.add_message("det_cust", "user", "ping", at_time=ft["t"])
        ft["t"] += 10 * 60
        s = m.get_session("det_cust", at_time=ft["t"])
        return s.session_id, s.lifecycle_status, len(s.message_history)

    check("deterministic: identical inputs -> identical outputs", scenario() == scenario())

    # --- invalid config falls back to safe defaults ---
    bad_path = "_bad_test_config.yaml"
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write("idle_timeout_seconds: -5\n"
                 "intent_confidence_threshold: 5\n"
                 "supported_languages: []\n")
    bad_cfg = load_config(bad_path)
    os.remove(bad_path)
    check("invalid idle_timeout_seconds -> default",
          bad_cfg["idle_timeout_seconds"] == DEFAULT_CONFIG["idle_timeout_seconds"])
    check("invalid intent_confidence_threshold -> default",
          bad_cfg["intent_confidence_threshold"] == DEFAULT_CONFIG["intent_confidence_threshold"])
    check("empty supported_languages -> default",
          bad_cfg["supported_languages"] == DEFAULT_CONFIG["supported_languages"])

    print()
    print(f"{len(failures)} CHECK(S) FAILED: {failures}" if failures else "ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    _run_self_checks()
