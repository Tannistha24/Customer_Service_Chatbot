from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Callable, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("step4_scheduler")


# ---------------------------------------------------------------------------
# 1. MAINTENANCE WINDOW CONFIG
# ---------------------------------------------------------------------------

@dataclass
class MaintenanceWindowConfig:
    """
    Configurable start/end time of the daily maintenance window.

    Using `datetime.time` (just hour/minute, no date) because a
    maintenance window repeats every day at the same clock time.

    Supports windows that cross midnight, e.g. start=22:00, end=02:00.
    """
    start: time = time(hour=2, minute=0)   # default 02:00
    end: time = time(hour=4, minute=0)     # default 04:00

    def is_in_window(self, current_time: datetime) -> bool:
        """
        Returns True if current_time's clock time falls inside
        [start, end).

        Two cases:
          - Normal window (start < end), e.g. 02:00-04:00:
                in-window if start <= now_time < end
          - Overnight window (start > end), e.g. 22:00-02:00:
                in-window if now_time >= start OR now_time < end
        """
        now_t = current_time.time()

        if self.start <= self.end:
            return self.start <= now_t < self.end
        else:
            # window wraps past midnight
            return now_t >= self.start or now_t < self.end


# Free function version, since the task asks for a standalone
# is_in_maintenance_window(current_time) function too.
_default_config = MaintenanceWindowConfig()


def is_in_maintenance_window(current_time: datetime,
                              config: MaintenanceWindowConfig = _default_config) -> bool:
    """Standalone helper. Uses `config` (defaults to a module-level default)."""
    return config.is_in_window(current_time)


# ---------------------------------------------------------------------------
# 2. THE QUEUED UPDATE
# ---------------------------------------------------------------------------

@dataclass
class PendingUpdate:
    """
    Represents one approved KB update waiting to be applied.

    - update_fn: the function to call to actually perform the update
      (e.g. "swap active pointer to new KB version" from Step 2).
      It should raise an exception if it fails.
    - kb_version: just a label for logging.
    - attempts: how many times we've already tried and failed.
    """
    kb_version: str
    update_fn: Callable[[], None]
    attempts: int = 0
    submitted_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# 3. THE SCHEDULER
# ---------------------------------------------------------------------------

class MaintenanceScheduler:
    """
    Wraps an APScheduler BackgroundScheduler to:
      - periodically check whether we're in the maintenance window
      - run queued approved updates when the window is open
      - retry failed updates after 15 / 30 / 60 minutes
    """

    # Retry backoff. Expressed as timedeltas so you could use seconds
    # in a demo/test and minutes in production.
    DEFAULT_RETRY_DELAYS = [timedelta(minutes=15), timedelta(minutes=30), timedelta(minutes=60)]

    def __init__(
        self,
        config: MaintenanceWindowConfig,
        check_interval_seconds: int = 60,
        retry_delays: Optional[List[timedelta]] = None,
        clock: Callable[[], datetime] = datetime.now,
    ):
        self.config = config
        self.check_interval_seconds = check_interval_seconds
        self.retry_delays = retry_delays or self.DEFAULT_RETRY_DELAYS
        self.clock = clock  # overridable "now" function, handy for tests/demo

        self.queue: List[PendingUpdate] = []
        self.scheduler = BackgroundScheduler()

    # -- public API ---------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler and its periodic queue check."""
        self.scheduler.add_job(
            self._check_and_run_queue,
            "interval",
            seconds=self.check_interval_seconds,
            id="maintenance_queue_check",
            replace_existing=True,
        )
        self.scheduler.start()
        log.info("Scheduler started. Checking queue every %ss.", self.check_interval_seconds)

    def shutdown(self) -> None:
        self.scheduler.shutdown(wait=False)
        log.info("Scheduler shut down.")

    def is_in_maintenance_window(self, current_time: Optional[datetime] = None) -> bool:
        """Public convenience method, uses this scheduler's own config/clock."""
        return self.config.is_in_window(current_time or self.clock())

    def submit_approved_update(self, kb_version: str, update_fn: Callable[[], None]) -> None:
        """
        Call this once Step 3 has approved a KB update.

        - If we're currently inside the maintenance window: run it now.
        - If not: put it in the queue. It will run automatically the
          next time the periodic check finds us inside the window.
        """
        item = PendingUpdate(kb_version=kb_version, update_fn=update_fn)

        if self.is_in_maintenance_window():
            log.info("[%s] Inside maintenance window right now -> running immediately.", kb_version)
            self._run_update(item)
        else:
            self.queue.append(item)
            log.info(
                "[%s] Outside maintenance window (%s-%s) -> queued. Queue size=%d",
                kb_version, self.config.start, self.config.end, len(self.queue),
            )

    # -- internal machinery --------------------------------------------

    def _check_and_run_queue(self) -> None:
        """
        Runs periodically (every check_interval_seconds) via APScheduler.
        If we're in the maintenance window and updates are waiting,
        run them now.
        """
        if not self.queue:
            return

        if not self.is_in_maintenance_window():
            log.debug("Queue check: still outside maintenance window, waiting.")
            return

        log.info("Maintenance window is open. Draining queue (%d item(s)).", len(self.queue))
        # Take a snapshot and clear the queue; failed items get
        # re-queued or scheduled for retry inside _run_update.
        pending = self.queue[:]
        self.queue.clear()
        for item in pending:
            self._run_update(item)

    def _run_update(self, item: PendingUpdate) -> None:
        """Actually calls the update function and handles success/failure."""
        try:
            item.update_fn()
            log.info("[%s] Update applied successfully.", item.kb_version)
        except Exception as exc:  # noqa: BLE001 - we want to catch anything and retry
            log.warning("[%s] Update failed (attempt %d): %s", item.kb_version, item.attempts + 1, exc)
            self._schedule_retry(item)

    def _schedule_retry(self, item: PendingUpdate) -> None:
        """
        Schedules a one-off retry after the next delay in
        self.retry_delays (15 min, then 30 min, then 60 min).
        If all retries are exhausted, gives up and logs an error.
        """
        item.attempts += 1
        attempt_index = item.attempts - 1  # 0-based index into retry_delays

        if attempt_index >= len(self.retry_delays):
            log.error("[%s] All retries exhausted. Giving up.", item.kb_version)
            return

        delay = self.retry_delays[attempt_index]
        run_at = self.clock() + delay

        log.info(
            "[%s] Retry #%d scheduled for %s (delay=%s).",
            item.kb_version, item.attempts, run_at, delay,
        )

        self.scheduler.add_job(
            self._retry_wrapper,
            "date",
            run_date=run_at,
            args=[item],
            id=f"retry_{item.kb_version}_{item.attempts}",
            replace_existing=True,
        )

    def _retry_wrapper(self, item: PendingUpdate) -> None:
        """
        Called by APScheduler when a retry's delay has elapsed.
        Still respects the maintenance window: if the window has
        closed again by the time the retry fires, we re-queue instead
        of forcing the update through.
        """
        if self.is_in_maintenance_window():
            self._run_update(item)
        else:
            log.info("[%s] Retry fired but window is closed again -> re-queuing.", item.kb_version)
