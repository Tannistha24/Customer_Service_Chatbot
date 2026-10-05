
from __future__ import annotations

import sys
import traceback
from datetime import datetime, timedelta
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Make the existing standalone Task 6 components importable.
#
# They are not packaged as part of a larger installed project in this
# environment; the real files live in /mnt/user-data/outputs (where Steps
# 3 and 4 were delivered). We only ever *import* them below -- nothing in
# this file redefines, monkey-patches, or duplicates their logic.
# ---------------------------------------------------------------------------

_COMPONENT_DIR = "/mnt/user-data/outputs"
if _COMPONENT_DIR not in sys.path:
    sys.path.insert(0, _COMPONENT_DIR)

_entity_extraction_import_error: Optional[BaseException] = None
_intent_classification_import_error: Optional[BaseException] = None

try:
    entity_extraction = __import__("entity_extraction")  # Task 6 - Step 3 (real component)
except ImportError as exc:  # pragma: no cover - depends on environment
    entity_extraction = None  # type: ignore[assignment]
    _entity_extraction_import_error = exc

try:
    import intent_classification  # type: ignore[import-not-found]  # Task 6 - Step 4 (real component)
except ImportError as exc:  # pragma: no cover - depends on environment
    intent_classification = None  # type: ignore[assignment]
    _intent_classification_import_error = exc

# Task 6 - Step 1 (session/timestamp architecture) is required for
# scenarios 1-3 below. It was never created in this environment. We
# probe a handful of plausible module names so this script would work
# unmodified if such a component is later added to the project, but we
# never fabricate one ourselves.
session_module = None
_session_candidate_names = (
    "session_manager",
    "session_architecture",
    "task6_step1_session",
    "task6_step1",
    "session",
)
for _name in _session_candidate_names:
    try:
        session_module = __import__(_name)
        break
    except ImportError:
        continue


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

Result = Tuple[str, str, str]  # (status, label, detail)


def _limitation(label: str, reason: str) -> Result:
    return ("LIMITATION", label, reason)


def _run_guarded(label: str, fn) -> Result:
    """Run `fn`, turning AssertionError into FAIL and any other
    unexpected exception into a reported (not silently swallowed) FAIL,
    so one scenario erroring out never stops the others from reporting.
    """
    try:
        return fn()
    except AssertionError as exc:
        return ("FAIL", label, str(exc))
    except Exception as exc:  # unexpected error while evaluating
        return ("FAIL", label, f"unexpected error during evaluation: {exc!r}\n{traceback.format_exc(limit=3)}")


# ---------------------------------------------------------------------------
# 1. Session expiry - 31 minutes
# ---------------------------------------------------------------------------

def eval_session_expiry_31_minutes() -> Result:
    label = "31-minute session expiry"
    if session_module is None:
        return _limitation(
            label,
            "Task 6 Step 1 (session/timestamp architecture) is not present in this "
            "environment as an importable component -- only Step 3 (entity_extraction.py) "
            "and Step 4 (intent_classification.py) exist here, and neither one models a "
            "session, a last-activity timestamp, or a 'summarized vs. full recent dialogue' "
            "context mode. This scenario requires exercising that real component; per the "
            "evaluation rules, it is not reimplemented or mocked here, so the required "
            "behavior cannot be observed through any current API.",
        )

    def _run() -> Result:
        # This branch only executes if a real Step 1 session module is
        # ever added to the environment -- written defensively against a
        # couple of plausible API shapes, but never invoked when no such
        # module exists.
        create_session = getattr(session_module, "create_session", None)
        assert create_session is not None, "session module has no create_session()"
        session = create_session(customer_id="eval_customer_31m")
        add_turn = getattr(session, "add_turn", None) or getattr(session_module, "add_turn", None)
        assert add_turn is not None, "session module has no way to add a conversation turn"
        add_turn(session, "My order is ORD-11111.")
        t0 = getattr(session, "last_activity", None) or datetime.now()
        simulated_now = t0 + timedelta(minutes=31)
        get_context_mode = getattr(session, "get_context_mode", None) or getattr(
            session_module, "get_context_mode", None
        )
        assert get_context_mode is not None, "session module exposes no observable context mode"
        mode = get_context_mode(session, now=simulated_now)
        assert mode in ("summarized", "background"), f"expected summarized/background mode, got {mode!r}"
        return ("PASS", label, f"context mode at +31min = {mode!r}")

    return _run_guarded(label, _run)


# ---------------------------------------------------------------------------
# 2. Session expiry - 25 hours
# ---------------------------------------------------------------------------

def eval_session_expiry_25_hours() -> Result:
    label = "25-hour clean slate"
    if session_module is None:
        return _limitation(
            label,
            "Same underlying gap as scenario 1: Task 6 Step 1 (session/timestamp "
            "architecture) does not exist as an importable component in this environment, "
            "so there is no real session object whose 'discard previous context after 24h' "
            "behavior can be exercised. Not reimplemented or mocked, per the evaluation rules.",
        )

    def _run() -> Result:
        create_session = getattr(session_module, "create_session", None)
        assert create_session is not None, "session module has no create_session()"
        session = create_session(customer_id="eval_customer_25h")
        add_turn = getattr(session, "add_turn", None) or getattr(session_module, "add_turn", None)
        assert add_turn is not None, "session module has no way to add a conversation turn"
        add_turn(session, "My order is ORD-22222.")
        t0 = getattr(session, "last_activity", None) or datetime.now()
        simulated_now = t0 + timedelta(hours=25)
        get_active_context = getattr(session, "get_active_context", None) or getattr(
            session_module, "get_active_context", None
        )
        assert get_active_context is not None, "session module exposes no observable active context"
        context = get_active_context(session, now=simulated_now)
        assert "ORD-22222" not in str(context), f"stale order id leaked into fresh session: {context!r}"
        return ("PASS", label, "old context discarded after 25h gap")

    return _run_guarded(label, _run)


# ---------------------------------------------------------------------------
# 3. Simultaneous session isolation
# ---------------------------------------------------------------------------

def eval_session_isolation() -> Result:
    label = "Session isolation"
    if session_module is None:
        return _limitation(
            label,
            "Task 6 Step 1 (session/timestamp architecture) -- the component that would own "
            "per-customer session objects -- does not exist as an importable module in this "
            "environment. Step 3's DialogueStateManager (the only stateful component that "
            "does exist) is a plain, unshared Python object: two separately constructed "
            "instances are trivially isolated by ordinary object identity, but that only "
            "demonstrates that two objects in memory don't share attributes -- it says "
            "nothing about the real system's actual customer-session routing/isolation, "
            "which lives in the missing Step 1 component. Reporting this as a limitation "
            "rather than presenting an unrelated toy demonstration as if it validated the "
            "real system.",
        )

    def _run() -> Result:
        create_session = getattr(session_module, "create_session", None)
        assert create_session is not None, "session module has no create_session()"
        session_a = create_session(customer_id="customer_A")
        session_b = create_session(customer_id="customer_B")
        add_turn = getattr(session_module, "add_turn", None)
        add_turn(session_a, "My order is ORD-11111.")
        add_turn(session_b, "My order is ORD-22222.")
        get_active_context = getattr(session_module, "get_active_context")
        ctx_a = str(get_active_context(session_a))
        ctx_b = str(get_active_context(session_b))
        assert "ORD-22222" not in ctx_a, "Session A leaked Session B's order id"
        assert "ORD-11111" not in ctx_b, "Session B leaked Session A's order id"
        return ("PASS", label, "no cross-session leakage observed")

    return _run_guarded(label, _run)


# ---------------------------------------------------------------------------
# 4. Mid-sentence language switching
# ---------------------------------------------------------------------------

def eval_mid_sentence_language_switching() -> Result:
    label = "Mid-sentence language switching"
    if entity_extraction is None:
        return _limitation(
            label,
            f"entity_extraction.py (Task 6 - Step 3) could not be imported: "
            f"{_entity_extraction_import_error!r}",
        )

    def _run() -> Result:
        # The naming cue ("My name is") is kept in English because Step 3
        # only supports English naming cues -- Step 2 (multilingual NLU)
        # does not exist in this environment, so this evaluates Step 3's
        # actual, documented multilingual guarantee: identifiers/dates
        # are extracted unchanged regardless of surrounding language,
        # not full multilingual name-cue recognition (out of scope for
        # Step 3, and explicitly not something this file may add).
        text = (
            "My name is John Smith, please check my order ORD-98412 -- "
            "pero necesito saber cu\u00e1ndo llegar\u00e1 el producto SKU-B12, "
            "la fecha es 2026-09-25."
        )
        result = entity_extraction.extract_entities(text)
        assert result.customer_name == "John Smith", f"customer_name corrupted: {result.customer_name!r}"
        assert result.order_id == "ORD-98412", f"order_id corrupted: {result.order_id!r}"
        assert result.product_code == "SKU-B12", f"product_code corrupted: {result.product_code!r}"
        assert "2026-09-25" in result.dates, f"date not preserved: {result.dates!r}"
        return (
            "PASS",
            label,
            f"customer_name={result.customer_name!r}, order_id={result.order_id!r}, "
            f"product_code={result.product_code!r}, dates={result.dates!r}",
        )

    return _run_guarded(label, _run)


# ---------------------------------------------------------------------------
# 5. Corrected details (multi-turn corrections)
# ---------------------------------------------------------------------------

def eval_corrected_details() -> Result:
    label = "Corrected order ID and date"
    if entity_extraction is None:
        return _limitation(
            label,
            f"entity_extraction.py (Task 6 - Step 3) could not be imported: "
            f"{_entity_extraction_import_error!r}",
        )

    def _run() -> Result:
        # Required example: order_id correction.
        mgr = entity_extraction.DialogueStateManager()
        mgr.process_turn("My order is ORD-12345.")
        mgr.process_turn("Actually, I meant ORD-54321.")
        active_order_id = mgr.get("order_id")
        assert active_order_id == "ORD-54321", (
            f"active order_id is {active_order_id!r}, expected 'ORD-54321' "
            "(the correction did not take effect)"
        )
        history = mgr.get_history("order_id")
        assert "ORD-12345" in history, "superseded value unexpectedly missing from history"

        # Additional required example: another entity type (date).
        mgr2 = entity_extraction.DialogueStateManager()
        mgr2.process_turn("My delivery date is September 25, 2026.")
        mgr2.process_turn("Actually, I meant September 30, 2026.")
        active_dates = mgr2.get("dates")
        assert active_dates == ["September 30, 2026"], (
            f"active dates={active_dates!r}, expected only the corrected date to be active"
        )

        return (
            "PASS",
            label,
            f"order_id -> {active_order_id!r} (history={history!r}); "
            f"dates -> {active_dates!r}",
        )

    return _run_guarded(label, _run)


# ---------------------------------------------------------------------------
# 6. Ambiguous follow-up
# ---------------------------------------------------------------------------

def eval_ambiguous_followup() -> Result:
    label = "Ambiguous follow-up"
    detail_parts = []

    if entity_extraction is not None:
        mgr = entity_extraction.DialogueStateManager()
        mgr.process_turn("The replacement should be for order ORD-12345.")
        followup_entities = mgr.extract("Can you do that to the other one instead?")
        detail_parts.append(
            f"Step 3 state after turn 1: order_id={mgr.get('order_id')!r}. "
            f"Step 3 extraction of the follow-up itself: order_id="
            f"{followup_entities.order_id!r} (the follow-up contains no identifier "
            "at all for Step 3 to act on)."
        )
    else:
        detail_parts.append(f"entity_extraction.py could not be imported: {_entity_extraction_import_error!r}")

    if intent_classification is not None:
        ic_result = intent_classification.classify_message("Can you do that to the other one instead?")
        detail_parts.append(
            f"Step 4 classification of the follow-up: intent={ic_result.intent!r}, "
            f"confidence={ic_result.confidence}, "
            f"clarification_required={ic_result.clarification_required}."
        )
    else:
        detail_parts.append(
            f"intent_classification.py could not be imported: {_intent_classification_import_error!r}"
        )

    reason = (
        "The follow-up ('the other one') is an anaphoric reference to some other, "
        "never-explicitly-named order, distinct from ORD-12345. Resolving that reference "
        "requires coreference resolution across turns -- deciding what 'the other one' "
        "refers to -- which is not implemented by either existing component: Step 3 only "
        "extracts literal ORD-/SKU- style tokens (it finds none in the follow-up), and "
        "Step 4 classifies the follow-up's intent in isolation with no mechanism to know a "
        "second, different order was ever implied. Task 6 Step 2 (multilingual NLU) and any "
        "dedicated coreference-resolution logic are also out of scope for this file. There is "
        "therefore no existing API surface that reports whether this ambiguity would "
        "correctly trigger clarification versus silently being resolved against ORD-12345. "
        + " ".join(detail_parts)
    )
    return _limitation(label, reason)


# ---------------------------------------------------------------------------
# 7. Corrupted order IDs
# ---------------------------------------------------------------------------

def eval_corrupted_order_ids() -> Result:
    label = "Corrupted order ID handling"
    if entity_extraction is None:
        return _limitation(
            label,
            f"entity_extraction.py (Task 6 - Step 3) could not be imported: "
            f"{_entity_extraction_import_error!r}",
        )

    def _run() -> Result:
        cases = [
            "ORD-98A412",   # letter injected into the digit run
            "ORD-9841",     # shorter digit run than the canonical example
            "ORD98412",     # missing hyphen
            "ord-98412",    # lowercase prefix
            "ORD--98412",   # doubled hyphen
            "ORD-98412X",   # trailing stray character glued onto the id
        ]
        observations = []
        for case in cases:
            text = f"Please check {case} for me regarding my order."
            extracted = entity_extraction.extract_entities(text).order_id
            if extracted is not None:
                # Whatever *is* extracted must be an exact, unmodified
                # substring of the original text -- never a
                # transformed/invented different valid identifier.
                assert extracted in text, (
                    f"input {case!r} produced order_id {extracted!r}, which is not an exact "
                    "substring of the input -- the malformed id would have been silently "
                    "transformed into something else"
                )
            observations.append(f"{case!r} -> {extracted!r}")
        return ("PASS", label, "; ".join(observations))

    return _run_guarded(label, _run)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    scenarios = [
        eval_session_expiry_31_minutes,
        eval_session_expiry_25_hours,
        eval_session_isolation,
        eval_mid_sentence_language_switching,
        eval_corrected_details,
        eval_ambiguous_followup,
        eval_corrupted_order_ids,
    ]

    print("=== Task 6 Step 5 Evaluation ===\n")

    results: list[Result] = []
    for scenario in scenarios:
        status, label, detail = scenario()
        results.append((status, label, detail))

    for status, label, _detail in results:
        print(f"{status} \u2014 {label}")

    print("\n--- Details ---")
    for status, label, detail in results:
        print(f"\n[{status}] {label}\n{detail}")

    # Deterministic, non-zero exit only if something genuinely FAILed
    # (LIMITATION and PASS both exit 0 -- a limitation is an honest
    # report, not a failure of this evaluation script).
    return 1 if any(status == "FAIL" for status, _, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())