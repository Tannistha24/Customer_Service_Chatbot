
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time as dtime, timezone
from typing import Any, Dict, List, Optional, Set, Union

try:
    from zoneinfo import ZoneInfo  # stdlib, Python 3.9+
except ImportError:  # pragma: no cover - fallback for very old interpreters
    ZoneInfo = None  # type: ignore


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

QUEUE_ON_CALL = "ON_CALL"
QUEUE_PRIORITY = "PRIORITY_QUEUE"
QUEUE_STANDARD = "STANDARD_QUEUE"
QUEUE_NEXT_WORKING_DAY = "NEXT_WORKING_DAY"

NEGATIVE_UNRESOLVED_THRESHOLD_MINUTES = 15


# --------------------------------------------------------------------------
# Business hours configuration
# --------------------------------------------------------------------------

@dataclass
class BusinessHoursConfig:
    """Configurable business hours window. Defaults: Mon-Fri, 09:00-18:00
    UTC. start is inclusive, end is exclusive (09:00 counts as business
    hours, 18:00 exactly does not)."""
    days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # Mon=0 ... Sun=6
    start: dtime = dtime(9, 0)
    end: dtime = dtime(18, 0)
    tz_name: str = "UTC"

    def tzinfo(self):
        if self.tz_name == "UTC" or ZoneInfo is None:
            return timezone.utc
        try:
            return ZoneInfo(self.tz_name)
        except Exception:
            return timezone.utc


def is_business_hours(current_time: datetime, config: BusinessHoursConfig) -> bool:
    """Safely evaluates business hours against a timezone-aware timestamp,
    normalized into the configured business-hours timezone."""
    local_dt = current_time.astimezone(config.tzinfo())
    if local_dt.weekday() not in config.days:
        return False
    local_t = local_dt.time()
    return config.start <= local_t < config.end


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _ensure_aware(dt: Optional[datetime], field_name: str) -> Optional[datetime]:
    if dt is None:
        return None
    if not isinstance(dt, datetime):
        raise TypeError(f"{field_name} must be a datetime instance, got {type(dt)!r}")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(
            f"{field_name} must be timezone-aware (naive datetimes are unsafe "
            f"for cross-timezone comparisons)."
        )
    return dt


def _get_field(step3_result: Any, name: str, default: Any = None) -> Any:
    """Duck-types over the Step 3 result, which may be a dataclass/object
    with attributes (e.g. EscalationDecision) or a plain dict."""
    if isinstance(step3_result, dict):
        return step3_result.get(name, default)
    return getattr(step3_result, name, default)


# --------------------------------------------------------------------------
# Result object
# --------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    routing_queue: str
    escalation_required: bool
    routing_reason: str
    urgency: str  # "urgent" | "standard"
    business_hours: bool
    negative_start_time: Optional[str]
    current_time: str
    elapsed_negative_minutes: Optional[float]
    triggered_conditions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# 15-minute unresolved-negative timer
# --------------------------------------------------------------------------

def evaluate_negative_timer(
    current_time: datetime,
    negative_start_time: Optional[datetime],
    negative_resolved: bool,
    threshold_minutes: float = NEGATIVE_UNRESOLVED_THRESHOLD_MINUTES,
) -> Dict[str, Any]:
    """Computes elapsed unresolved-negative minutes and whether the
    escalation threshold has been strictly exceeded. A resolved
    conversation, or one with no negative_start_time, never escalates
    here (elapsed is still reported when computable, for inspectability)."""
    if negative_start_time is None:
        return {"elapsed_negative_minutes": None, "time_escalated": False}

    elapsed_seconds = (current_time - negative_start_time).total_seconds()
    elapsed_minutes = round(elapsed_seconds / 60.0, 4)

    if negative_resolved:
        return {"elapsed_negative_minutes": elapsed_minutes, "time_escalated": False}

    # Strictly greater than the threshold; exactly 15.0 minutes does NOT trigger.
    time_escalated = elapsed_minutes > threshold_minutes
    return {"elapsed_negative_minutes": elapsed_minutes, "time_escalated": time_escalated}


# --------------------------------------------------------------------------
# Main routing function
# --------------------------------------------------------------------------

def route_conversation(
    step3_result: Any,
    current_time: datetime,
    negative_start_time: Optional[datetime] = None,
    negative_resolved: bool = False,
    business_hours: Optional[BusinessHoursConfig] = None,
) -> RoutingDecision:
    """Deterministic routing decision. `step3_result` is consumed as-is
    (its `escalation_required` field drives urgency) -- Step 3's risk
    rules are never redefined or re-derived here."""
    current_time = _ensure_aware(current_time, "current_time")
    negative_start_time = _ensure_aware(negative_start_time, "negative_start_time")
    config = business_hours or BusinessHoursConfig()

    # --- Pull Step 3 result as data only (no re-evaluation of risk) ---
    step3_escalation_required = bool(_get_field(step3_result, "escalation_required", False))
    step3_triggered = list(_get_field(step3_result, "triggered_conditions", []) or [])

    # --- Business hours ---
    in_business_hours = is_business_hours(current_time, config)

    # --- 15-minute unresolved-negative timer ---
    timer_result = evaluate_negative_timer(current_time, negative_start_time, negative_resolved)
    elapsed_negative_minutes = timer_result["elapsed_negative_minutes"]
    time_escalated = timer_result["time_escalated"]

    triggered_conditions: List[str] = list(step3_triggered)
    if time_escalated:
        triggered_conditions.append("unresolved_negative_over_15min")
    triggered_conditions = sorted(set(triggered_conditions))

    # --- Final urgency: Step 3's verdict OR the 15-minute timer ---
    urgent = step3_escalation_required or time_escalated
    urgency_label = "urgent" if urgent else "standard"
    escalation_required = step3_escalation_required or time_escalated

    # --- Deterministic queue assignment ---
    if in_business_hours:
        if urgent:
            queue = QUEUE_PRIORITY
            reason = "Urgent/high-risk conversation during business hours -> priority queue."
        else:
            queue = QUEUE_STANDARD
            reason = "Standard conversation during business hours -> standard queue."
    else:
        if urgent:
            queue = QUEUE_ON_CALL
            reason = "Urgent/high-risk conversation outside business hours -> on-call."
        else:
            queue = QUEUE_NEXT_WORKING_DAY
            reason = "Standard conversation outside business hours -> next working day queue."

    if time_escalated and not step3_escalation_required:
        reason += " (Escalated further: unresolved negative state exceeded 15 minutes.)"

    return RoutingDecision(
        routing_queue=queue,
        escalation_required=escalation_required,
        routing_reason=reason,
        urgency=urgency_label,
        business_hours=in_business_hours,
        negative_start_time=negative_start_time.isoformat() if negative_start_time else None,
        current_time=current_time.isoformat(),
        elapsed_negative_minutes=elapsed_negative_minutes,
        triggered_conditions=triggered_conditions,
    )


# ==========================================================================
# Self-checks (fully deterministic, simulated timestamps only)
# ==========================================================================

def _run_self_checks() -> None:
    results = []

    def check(name, condition):
        results.append((name, bool(condition)))
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")

    UTC = timezone.utc
    default_hours = BusinessHoursConfig()  # Mon-Fri 09:00-18:00 UTC

    urgent_step3 = {"escalation_required": True, "triggered_conditions": ["account_compromise"]}
    standard_step3 = {"escalation_required": False, "triggered_conditions": []}

    # 1. Urgent after-hours -> ON_CALL
    # Tuesday 20:00 UTC (after hours)
    t = datetime(2025, 6, 10, 20, 0, tzinfo=UTC)
    d = route_conversation(urgent_step3, t, business_hours=default_hours)
    check("urgent after-hours routes to ON_CALL",
          d.routing_queue == QUEUE_ON_CALL and not d.business_hours)

    # 2. Standard after-hours -> NEXT_WORKING_DAY
    d = route_conversation(standard_step3, t, business_hours=default_hours)
    check("standard after-hours routes to NEXT_WORKING_DAY",
          d.routing_queue == QUEUE_NEXT_WORKING_DAY)

    # 3. Normal business-hours case -> regular queue
    # Tuesday 11:00 UTC (business hours)
    t_bh = datetime(2025, 6, 10, 11, 0, tzinfo=UTC)
    d = route_conversation(standard_step3, t_bh, business_hours=default_hours)
    check("standard business-hours case routes to STANDARD_QUEUE",
          d.routing_queue == QUEUE_STANDARD and d.business_hours)

    d_urgent_bh = route_conversation(urgent_step3, t_bh, business_hours=default_hours)
    check("urgent business-hours case routes to PRIORITY_QUEUE",
          d_urgent_bh.routing_queue == QUEUE_PRIORITY)

    # 4. Negative unresolved at 14 minutes -> not time-escalated
    neg_start = t_bh - timedelta(minutes=14)
    d = route_conversation(standard_step3, t_bh, negative_start_time=neg_start,
                            negative_resolved=False, business_hours=default_hours)
    check("14 minutes unresolved negative does not time-escalate",
          "unresolved_negative_over_15min" not in d.triggered_conditions
          and d.elapsed_negative_minutes == 14.0)

    # 5. Exactly 15 minutes -> not yet escalated
    neg_start_15 = t_bh - timedelta(minutes=15)
    d = route_conversation(standard_step3, t_bh, negative_start_time=neg_start_15,
                            negative_resolved=False, business_hours=default_hours)
    check("exactly 15 minutes is not > 15, so not time-escalated",
          "unresolved_negative_over_15min" not in d.triggered_conditions
          and d.elapsed_negative_minutes == 15.0)

    # 6. 16 minutes unresolved negative -> escalated
    neg_start_16 = t_bh - timedelta(minutes=16)
    d = route_conversation(standard_step3, t_bh, negative_start_time=neg_start_16,
                            negative_resolved=False, business_hours=default_hours)
    check("16 minutes unresolved negative time-escalates",
          "unresolved_negative_over_15min" in d.triggered_conditions
          and d.escalation_required
          and d.routing_queue == QUEUE_PRIORITY)

    # 7. Resolved negative conversation -> no 15-minute escalation even if elapsed > 15
    d = route_conversation(standard_step3, t_bh, negative_start_time=neg_start_16,
                            negative_resolved=True, business_hours=default_hours)
    check("resolved negative conversation does not time-escalate",
          "unresolved_negative_over_15min" not in d.triggered_conditions
          and not d.escalation_required)

    # 8. Explicit simulated timestamp -> deterministic results
    d1 = route_conversation(urgent_step3, t, negative_start_time=neg_start_16,
                             negative_resolved=False, business_hours=default_hours)
    d2 = route_conversation(urgent_step3, t, negative_start_time=neg_start_16,
                             negative_resolved=False, business_hours=default_hours)
    check("identical inputs produce identical results", d1.to_dict() == d2.to_dict())

    # 9. Timezone-aware handling: same instant, different input tz representation
    # 11:00 UTC Tuesday == 07:00 EST-ish offset Tuesday (fixed -04:00 for test simplicity)
    t_utc = datetime(2025, 6, 10, 11, 0, tzinfo=UTC)
    t_offset = t_utc.astimezone(timezone(timedelta(hours=-4)))
    d_utc = route_conversation(standard_step3, t_utc, business_hours=default_hours)
    d_offset = route_conversation(standard_step3, t_offset, business_hours=default_hours)
    check("equivalent instants in different tz representations agree",
          d_utc.routing_queue == d_offset.routing_queue
          and d_utc.business_hours == d_offset.business_hours)

    if ZoneInfo is not None:
        ny_hours = BusinessHoursConfig(tz_name="America/New_York")
        # 14:00 UTC on a Tuesday in June = 10:00 EDT -> within 09:00-18:00 local business hours
        t_ny = datetime(2025, 6, 10, 14, 0, tzinfo=UTC)
        d_ny = route_conversation(standard_step3, t_ny, business_hours=ny_hours)
        check("named-timezone business hours config evaluated correctly",
              d_ny.business_hours is True)

    # 10. Business-hours boundary cases
    start_boundary = datetime(2025, 6, 10, 9, 0, tzinfo=UTC)   # inclusive start
    just_before = datetime(2025, 6, 10, 8, 59, tzinfo=UTC)      # not yet open
    end_boundary = datetime(2025, 6, 10, 18, 0, tzinfo=UTC)     # exclusive end
    just_before_end = datetime(2025, 6, 10, 17, 59, tzinfo=UTC)
    weekend = datetime(2025, 6, 14, 11, 0, tzinfo=UTC)          # Saturday

    check("09:00 boundary counts as business hours",
          route_conversation(standard_step3, start_boundary, business_hours=default_hours).business_hours)
    check("08:59 does not count as business hours",
          not route_conversation(standard_step3, just_before, business_hours=default_hours).business_hours)
    check("18:00 boundary does NOT count as business hours (exclusive end)",
          not route_conversation(standard_step3, end_boundary, business_hours=default_hours).business_hours)
    check("17:59 counts as business hours",
          route_conversation(standard_step3, just_before_end, business_hours=default_hours).business_hours)
    check("weekend is never business hours",
          not route_conversation(standard_step3, weekend, business_hours=default_hours).business_hours)

    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} self-checks passed.")


if __name__ == "__main__":
    _run_self_checks()
