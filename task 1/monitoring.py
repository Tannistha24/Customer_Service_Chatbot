from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_TEST_QUERIES = (
    "What are your business hours?",
    "How can I reset my password?",
    "How do I contact customer support?",
)


@dataclass
class HealthCheckResult:
    """Measurements and decision from one health check."""

    kb_version: str
    passed: bool
    total_requests: int
    successful_requests: int
    error_count: int
    error_rate: float
    average_latency_seconds: float
    maximum_latency_seconds: float
    max_error_rate: float
    max_latency_seconds: float
    duration_seconds: float
    rollback_attempted: bool = False
    rollback_succeeded: Optional[bool] = None
    rollback_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""
        return asdict(self)

    def print_summary(self) -> None:
        """Print a clear health-check result."""

        status = "PASSED" if self.passed else "FAILED"

        print(f"HEALTH CHECK {status}: KB version {self.kb_version}")
        print(
            f"  requests={self.total_requests}, "
            f"errors={self.error_count} ({self.error_rate:.2%}), "
            f"average_latency={self.average_latency_seconds:.3f}s, "
            f"maximum_latency={self.maximum_latency_seconds:.3f}s"
        )
        print(
            f"  limits: error_rate<={self.max_error_rate:.2%}, "
            f"latency<={self.max_latency_seconds:.3f}s"
        )

        if self.rollback_attempted:
            if self.rollback_succeeded:
                print("  rollback: completed")
            else:
                print(f"  rollback: failed ({self.rollback_error})")


def _validate_settings(
    duration_seconds: float,
    max_error_rate: float,
    max_latency_seconds: float,
    interval_seconds: float,
) -> None:
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be 0 or greater")

    if not 0 <= max_error_rate <= 1:
        raise ValueError("max_error_rate must be between 0 and 1")

    if max_latency_seconds < 0:
        raise ValueError("max_latency_seconds must be 0 or greater")

    if interval_seconds < 0:
        raise ValueError("interval_seconds must be 0 or greater")


def _ask_chatbot(chatbot: Any, query: str, kb_version: str) -> Any:
    """Call a mock chatbot or chatbot object."""

    if callable(chatbot):
        return chatbot(query, kb_version)

    for method_name in ("ask", "query", "answer"):
        method = getattr(chatbot, method_name, None)

        if callable(method):
            return method(query, kb_version)

    raise TypeError(
        "chatbot must be callable or provide ask(), query(), or answer()"
    )


def _rollback(
    version_manager: Any,
    previous_valid_version: Optional[str],
) -> None:
    """Use the existing Step 2 manager to restore the previous KB."""

    if previous_valid_version is None:
        raise ValueError(
            "previous_valid_version is required for rollback"
        )

    # Step 2 KBVersionManager uses rollback_to(version_name).
    rollback_to = getattr(version_manager, "rollback_to", None)

    if callable(rollback_to):
        rollback_to(previous_valid_version)
        return

    raise AttributeError(
        "version_manager must provide rollback_to(version_name)"
    )


def run_health_check(
    chatbot: Any,
    kb_version: str,
    *,
    duration_seconds: float = 300,
    test_queries: Optional[Iterable[str]] = None,
    max_error_rate: float = 0.05,
    max_latency_seconds: float = 2.0,
    interval_seconds: float = 1.0,
) -> HealthCheckResult:
    """Run automated test queries for five minutes by default.

    The health check passes only when:

    1. Error rate is less than or equal to max_error_rate.
    2. Maximum response latency is less than or equal to
       max_latency_seconds.
    """

    _validate_settings(
        duration_seconds,
        max_error_rate,
        max_latency_seconds,
        interval_seconds,
    )

    queries = tuple(test_queries or DEFAULT_TEST_QUERIES)

    if not queries:
        raise ValueError("test_queries must contain at least one query")

    started_at = time.monotonic()
    deadline = started_at + duration_seconds

    total_requests = 0
    error_count = 0
    latencies: list[float] = []
    query_index = 0

    # Always run at least one query, including in short tests.
    while total_requests == 0 or time.monotonic() < deadline:
        query = queries[query_index % len(queries)]
        query_index += 1

        request_started_at = time.monotonic()

        try:
            _ask_chatbot(chatbot, query, kb_version)
        except Exception:
            # Any chatbot exception counts as a failed request.
            error_count += 1
        finally:
            latency = time.monotonic() - request_started_at
            latencies.append(latency)
            total_requests += 1

        remaining_seconds = deadline - time.monotonic()

        if interval_seconds > 0 and remaining_seconds > 0:
            time.sleep(min(interval_seconds, remaining_seconds))

    elapsed_seconds = time.monotonic() - started_at

    error_rate = error_count / total_requests
    average_latency = sum(latencies) / len(latencies)
    maximum_latency = max(latencies)

    passed = (
        error_rate <= max_error_rate
        and maximum_latency <= max_latency_seconds
    )

    return HealthCheckResult(
        kb_version=kb_version,
        passed=passed,
        total_requests=total_requests,
        successful_requests=total_requests - error_count,
        error_count=error_count,
        error_rate=error_rate,
        average_latency_seconds=average_latency,
        maximum_latency_seconds=maximum_latency,
        max_error_rate=max_error_rate,
        max_latency_seconds=max_latency_seconds,
        duration_seconds=elapsed_seconds,
    )


def monitor_activated_version(
    version_manager: Any,
    chatbot: Any,
    activated_version: str,
    *,
    previous_valid_version: Optional[str] = None,
    result_path: Optional[str | Path] = None,
    **health_check_settings: Any,
) -> HealthCheckResult:
    """Monitor an activated KB and roll back if it is unhealthy.

    The new KB remains active when the health check passes.

    If the health check fails, the existing Step 2 manager rolls back
    using rollback_to(previous_valid_version).
    """

    result = run_health_check(
        chatbot,
        activated_version,
        **health_check_settings,
    )

    if not result.passed:
        result.rollback_attempted = True

        try:
            _rollback(version_manager, previous_valid_version)
            result.rollback_succeeded = True
        except Exception as error:
            result.rollback_succeeded = False
            result.rollback_error = str(error)

    if result_path is not None:
        path = Path(result_path)

        path.write_text(
            json.dumps(result.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    result.print_summary()

    return result

