
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Configuration objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BusinessCalendar:
    """
    Configurable definition of "business time".

    - timezone: IANA timezone name (e.g. "UTC", "America/New_York"). All
      internal date/time math is performed in this timezone.
    - business_days: weekdays that count as working days.
      Monday=0 ... Sunday=6. (This is how weekends are configured/excluded:
      simply omit them from this set.)
    - start_time / end_time: the daily business window, applied to every
      business day. start_time must be strictly before end_time.
    - holidays: specific calendar dates (in `timezone`) that are excluded
      even if they fall on a business day.
    """

    timezone: str
    business_days: FrozenSet[int]
    start_time: time
    end_time: time
    holidays: FrozenSet[date] = field(default_factory=frozenset)

    def __post_init__(self):
        if not self.timezone:
            raise ValueError("BusinessCalendar.timezone is required (e.g. 'UTC').")
        try:
            ZoneInfo(self.timezone)
        except Exception as exc:
            raise ValueError(f"Invalid timezone '{self.timezone}': {exc}") from exc

        if not self.business_days:
            raise ValueError("BusinessCalendar.business_days must not be empty.")
        if not self.business_days.issubset(set(range(7))):
            raise ValueError("BusinessCalendar.business_days must contain values 0-6 (Mon-Sun).")

        if self.start_time >= self.end_time:
            raise ValueError("BusinessCalendar.start_time must be before end_time.")

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def is_business_day(self, day: date) -> bool:
        return day.weekday() in self.business_days and day not in self.holidays

    def day_window(self, day: date) -> Tuple[datetime, datetime]:
        """Return (start, end) tz-aware datetimes for the business window on `day`."""
        tz = self.tzinfo
        return (
            datetime.combine(day, self.start_time, tzinfo=tz),
            datetime.combine(day, self.end_time, tzinfo=tz),
        )

    def daily_business_seconds(self) -> float:
        return (
            datetime.combine(date.min, self.end_time)
            - datetime.combine(date.min, self.start_time)
        ).total_seconds()


@dataclass(frozen=True)
class SLAPolicy:
    """
    Maps ticket priority -> SLA duration (business time). Durations are
    counted purely in business hours per `BusinessCalendar`.
    """

    name: str
    priority_durations: Dict[str, timedelta]

    def __post_init__(self):
        if not self.name:
            raise ValueError("SLAPolicy.name is required.")
        if not self.priority_durations:
            raise ValueError("SLAPolicy.priority_durations must not be empty.")
        for priority, duration in self.priority_durations.items():
            if duration <= timedelta(0):
                raise ValueError(
                    f"SLAPolicy '{self.name}': duration for priority '{priority}' "
                    "must be a positive timedelta."
                )

    def duration_for(self, priority: str) -> timedelta:
        if priority not in self.priority_durations:
            raise KeyError(
                f"SLAPolicy '{self.name}' has no configured duration for priority "
                f"'{priority}'. Configured priorities: "
                f"{sorted(self.priority_durations)}."
            )
        return self.priority_durations[priority]


# ---------------------------------------------------------------------------
# Core pure functions
# ---------------------------------------------------------------------------

def _ensure_aware(dt: datetime, label: str) -> None:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware datetime.")


def _advance_to_business_start(current: datetime, calendar: BusinessCalendar) -> datetime:
    """
    Move `current` forward (never backward) to the next point in time that
    falls inside a business window, without consuming any SLA duration.
    If `current` is already inside a business window, it is returned as-is.
    """
    tz = calendar.tzinfo
    current = current.astimezone(tz)
    # Safety bound: business_days is non-empty (validated), so within 7
    # days there is guaranteed to be at least one business day, unless
    # every one of those 7 days happens to be a configured holiday.
    for _ in range(3660):  # ~10 years of daily steps as a hard safety cap
        day = current.date()
        if calendar.is_business_day(day):
            start_dt, end_dt = calendar.day_window(day)
            if current < start_dt:
                return start_dt
            if current < end_dt:
                return current
            # current >= end_dt -> fall through to next day
        current = datetime.combine(day + timedelta(days=1), time.min, tzinfo=tz)
    raise RuntimeError(
        "Could not find a business day within a 10-year horizon; "
        "check BusinessCalendar configuration (business_days/holidays)."
    )


def calculate_deadline(
    created_at: datetime,
    duration: timedelta,
    calendar: BusinessCalendar,
) -> datetime:
    """
    Deterministically compute the SLA deadline by counting only business
    hours (per `calendar`) starting at `created_at`.

    - If `created_at` falls outside business hours (evenings, weekends,
      holidays), counting starts at the next business-hour opening.
    - Returns a timezone-aware datetime in `calendar`'s timezone.
    """
    _ensure_aware(created_at, "created_at")
    if duration <= timedelta(0):
        raise ValueError("duration must be a positive timedelta.")

    tz = calendar.tzinfo
    current = _advance_to_business_start(created_at.astimezone(tz), calendar)
    remaining = duration

    for _ in range(3660):  # safety cap, mirrors _advance_to_business_start
        _, end_dt = calendar.day_window(current.date())
        available = end_dt - current
        if remaining <= available:
            return current + remaining
        remaining -= available
        next_midnight = datetime.combine(current.date() + timedelta(days=1), time.min, tzinfo=tz)
        current = _advance_to_business_start(next_midnight, calendar)

    raise RuntimeError("Deadline calculation did not converge; check calendar configuration.")


def business_time_elapsed(
    start: datetime,
    end: datetime,
    calendar: BusinessCalendar,
) -> timedelta:
    """
    Compute how much business time (per `calendar`) has elapsed between
    `start` and `end`. Returns timedelta(0) if `end` <= `start`.
    """
    _ensure_aware(start, "start")
    _ensure_aware(end, "end")

    tz = calendar.tzinfo
    s = start.astimezone(tz)
    e = end.astimezone(tz)
    if e <= s:
        return timedelta(0)

    total = timedelta(0)
    current = _advance_to_business_start(s, calendar)
    if current >= e:
        return timedelta(0)

    for _ in range(3660):  # safety cap
        _, end_dt = calendar.day_window(current.date())
        day_end = min(end_dt, e)
        if day_end > current:
            total += day_end - current
        if e <= end_dt:
            break
        next_midnight = datetime.combine(current.date() + timedelta(days=1), time.min, tzinfo=tz)
        current = _advance_to_business_start(next_midnight, calendar)
        if current >= e:
            break

    return total


# ---------------------------------------------------------------------------
# Timer management
# ---------------------------------------------------------------------------

class SLAState(str, Enum):
    ACTIVE = "active"
    WARNING = "warning"
    BREACHED = "breached"


WARNING_THRESHOLD = 0.75  # 75% of SLA consumed


@dataclass
class TimerStatus:
    ticket_id: str
    state: SLAState
    deadline: datetime
    elapsed_business_time: timedelta
    remaining_business_time: timedelta
    percent_consumed: float  # 0-100+, may exceed 100 if breached


class TicketSLATimer:
    """
    Tracks a single ticket's SLA deadline and timer state.

    The deadline is always derived fresh from (created_at, priority,
    policy, calendar) -- it is never adjusted incrementally -- so calling
    `update_priority` or `update_policy` and then re-checking status can
    never rely on a stale deadline.
    """

    def __init__(
        self,
        ticket_id: str,
        created_at: datetime,
        priority: str,
        policy: SLAPolicy,
        calendar: BusinessCalendar,
    ):
        _ensure_aware(created_at, "created_at")
        self.ticket_id = ticket_id
        self.created_at = created_at
        self.priority = priority
        self.policy = policy
        self.calendar = calendar
        self._deadline: Optional[datetime] = None
        self._duration: Optional[timedelta] = None
        self.recalculate()

    def recalculate(self) -> None:
        """Recompute duration and deadline from scratch (created_at is fixed)."""
        self._duration = self.policy.duration_for(self.priority)
        self._deadline = calculate_deadline(self.created_at, self._duration, self.calendar)

    def update_priority(self, new_priority: str) -> None:
        """Change ticket priority and recalculate the deadline from scratch."""
        self.priority = new_priority
        self.recalculate()

    def update_policy(self, new_policy: SLAPolicy) -> None:
        """Change the SLA policy in effect and recalculate the deadline from scratch."""
        self.policy = new_policy
        self.recalculate()

    @property
    def deadline(self) -> datetime:
        assert self._deadline is not None  # set in __init__/recalculate
        return self._deadline

    @property
    def duration(self) -> timedelta:
        assert self._duration is not None
        return self._duration

    def get_status(self, now: datetime) -> TimerStatus:
        """
        Compute the current timer status as of `now`.

        - state = BREACHED if `now` is at/after the deadline.
        - state = WARNING if business time consumed >= 75% of the SLA
          duration (and not yet breached).
        - state = ACTIVE otherwise.
        """
        _ensure_aware(now, "now")

        elapsed = business_time_elapsed(self.created_at, now, self.calendar)
        duration_seconds = self.duration.total_seconds()
        percent_consumed = (elapsed.total_seconds() / duration_seconds) * 100.0

        now_tz = now.astimezone(self.calendar.tzinfo)
        if now_tz >= self.deadline:
            state = SLAState.BREACHED
            remaining = timedelta(0)
        else:
            remaining = self.duration - elapsed
            if remaining < timedelta(0):
                remaining = timedelta(0)
            state = SLAState.WARNING if percent_consumed >= WARNING_THRESHOLD * 100.0 else SLAState.ACTIVE

        return TimerStatus(
            ticket_id=self.ticket_id,
            state=state,
            deadline=self.deadline,
            elapsed_business_time=elapsed,
            remaining_business_time=remaining,
            percent_consumed=percent_consumed,
        )


if __name__ == "__main__":
    calendar = BusinessCalendar(
        timezone="UTC",
        business_days=frozenset({0, 1, 2, 3, 4}),  # Mon-Fri
        start_time=time(9, 0),
        end_time=time(17, 0),
        holidays=frozenset({date(2024, 12, 25)}),
    )
    policy = SLAPolicy(
        name="default",
        priority_durations={
            "low": timedelta(hours=24),
            "high": timedelta(hours=4),
        },
    )

    created = datetime(2024, 5, 1, 10, 0, tzinfo=ZoneInfo("UTC"))  # a Wednesday
    timer = TicketSLATimer("T-1", created, "high", policy, calendar)
    print("Deadline:", timer.deadline)
    print("Status @ +2h:", timer.get_status(created + timedelta(hours=2)))