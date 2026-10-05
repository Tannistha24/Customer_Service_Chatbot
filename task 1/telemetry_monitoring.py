"""Step 7: file-based telemetry and monitoring.

This module adds observation around Steps 1-6 without changing their core
logic. It uses wrappers: the wrapped function keeps its original arguments,
return value, and exceptions.

Typical integration:

    telemetry = TelemetryMonitor(
        metrics_path="step7_telemetry.jsonl",
        summary_path="step7_summary.json",
        confidence_threshold=0.70,
    )

    # Around the Step 1 ingestion function:
    telemetry.run_ingestion(step1_ingest, document)

    # Around retrieval and Step 6 generation:
    documents = telemetry.run_retrieval(step2_retrieve, query)
    answer = telemetry.run_generation(step6_generate, query, documents)

For score-aware fallback handling, use ``monitor_query``. It runs retrieval,
records similarity/confidence scores, and uses the fallback/human-escalation
path when the score is below the configured threshold.

Run the built-in demo with:

    python telemetry_monitoring.py
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Optional


class TelemetryWriteError(RuntimeError):
    """Raised when strict telemetry file writing cannot complete."""


@dataclass
class QueryResult:
    """Result returned by ``monitor_query``."""

    answer: Any
    retrieval_result: Any
    used_fallback: bool
    human_escalated: bool
    confidence_score: Optional[float]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_latency(latency_seconds: float) -> float:
    latency = float(latency_seconds)
    if not math.isfinite(latency) or latency < 0:
        raise ValueError("latency_seconds must be a finite number >= 0")
    return latency


def _validate_score(
    score: Optional[float],
    score_name: str,
) -> Optional[float]:
    if score is None:
        return None

    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"{score_name} must be a finite number")
    return value


def _score_statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "average": None,
            "minimum": None,
            "maximum": None,
        }

    return {
        "count": len(values),
        "average": sum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
    }


class TelemetryMonitor:
    """Collect Step 7 metrics in memory and in JSON files."""

    def __init__(
        self,
        metrics_path: str | Path = "step7_telemetry.jsonl",
        summary_path: str | Path = "step7_summary.json",
        confidence_threshold: float = 0.70,
        *,
        strict_file_writes: bool = False,
    ) -> None:
        threshold = float(confidence_threshold)
        if not 0 <= threshold <= 1:
            raise ValueError(
                "confidence_threshold must be between 0 and 1"
            )

        self.metrics_path = Path(metrics_path)
        self.summary_path = Path(summary_path)
        self.confidence_threshold = threshold
        self.strict_file_writes = strict_file_writes
        self.telemetry_write_errors = 0
        self.last_write_error: Optional[str] = None

        self._ingestion_attempts = 0
        self._ingestion_failures = 0
        self._retrieval_attempts = 0
        self._retrieval_failures = 0
        self._generation_attempts = 0
        self._generation_failures = 0

        self._retrieval_latencies: list[float] = []
        self._generation_latencies: list[float] = []
        self._similarity_scores: list[float] = []
        self._confidence_scores: list[float] = []

        self._low_confidence_retrievals = 0
        self._fallback_events = 0
        self._human_escalation_events = 0

    def _write_event(self, event: dict[str, Any]) -> None:
        """Append one event to the JSONL file.

        Telemetry should not break the production pipeline by default.
        Set strict_file_writes=True when a file-write problem should raise.
        """

        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event) + "\n")
        except (OSError, TypeError, ValueError) as error:
            self.telemetry_write_errors += 1
            self.last_write_error = str(error)

            if self.strict_file_writes:
                raise TelemetryWriteError(str(error)) from error

    def _record_event(
        self,
        event_type: str,
        *,
        success: Optional[bool] = None,
        latency_seconds: Optional[float] = None,
        similarity_score: Optional[float] = None,
        confidence_score: Optional[float] = None,
        error: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        event: dict[str, Any] = {
            "timestamp": _utc_timestamp(),
            "event_type": event_type,
        }

        if success is not None:
            event["success"] = success
        if latency_seconds is not None:
            event["latency_seconds"] = latency_seconds
        if similarity_score is not None:
            event["similarity_score"] = similarity_score
        if confidence_score is not None:
            event["confidence_score"] = confidence_score
        if error is not None:
            event["error"] = error
        if reason is not None:
            event["reason"] = reason

        self._write_event(event)

    def record_ingestion(
        self,
        success: bool,
        *,
        error: Optional[str] = None,
    ) -> None:
        """Record one Step 1 ingestion attempt."""

        self._ingestion_attempts += 1
        if not success:
            self._ingestion_failures += 1

        self._record_event(
            "ingestion",
            success=success,
            error=error,
        )

    def record_retrieval(
        self,
        success: bool,
        latency_seconds: float,
        *,
        similarity_score: Optional[float] = None,
        confidence_score: Optional[float] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Record retrieval metrics and return whether confidence is low.

        Scores are recorded as supplied. For the low-confidence decision,
        confidence_score is preferred; similarity_score is used when no
        confidence score is available.
        """

        latency = _validate_latency(latency_seconds)
        similarity = _validate_score(similarity_score, "similarity_score")
        confidence = _validate_score(confidence_score, "confidence_score")

        self._retrieval_attempts += 1
        self._retrieval_latencies.append(latency)

        if not success:
            self._retrieval_failures += 1

        if similarity is not None:
            self._similarity_scores.append(similarity)
        if confidence is not None:
            self._confidence_scores.append(confidence)

        score_for_fallback = (
            confidence if confidence is not None else similarity
        )
        low_confidence = (
            score_for_fallback is not None
            and score_for_fallback < self.confidence_threshold
        )

        if low_confidence:
            self._low_confidence_retrievals += 1

        self._record_event(
            "retrieval",
            success=success,
            latency_seconds=latency,
            similarity_score=similarity,
            confidence_score=confidence,
            error=error,
        )

        return low_confidence

    def record_generation(
        self,
        success: bool,
        latency_seconds: float,
        *,
        error: Optional[str] = None,
    ) -> None:
        """Record one Step 6 LLM/mock generation attempt."""

        latency = _validate_latency(latency_seconds)
        self._generation_attempts += 1
        self._generation_latencies.append(latency)

        if not success:
            self._generation_failures += 1

        self._record_event(
            "llm_generation",
            success=success,
            latency_seconds=latency,
            error=error,
        )

    def record_fallback(
        self,
        *,
        confidence_score: Optional[float],
        reason: str = "confidence below threshold",
    ) -> None:
        """Record a fallback event caused by low confidence."""

        confidence = _validate_score(confidence_score, "confidence_score")
        self._fallback_events += 1

        self._record_event(
            "fallback",
            confidence_score=confidence,
            reason=reason,
        )

    def record_human_escalation(
        self,
        *,
        confidence_score: Optional[float],
        reason: str = "confidence below threshold",
    ) -> None:
        """Record a human-escalation event caused by low confidence."""

        confidence = _validate_score(confidence_score, "confidence_score")
        self._human_escalation_events += 1

        self._record_event(
            "human_escalation",
            confidence_score=confidence,
            reason=reason,
        )

    def run_ingestion(
        self,
        ingestion_function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run an existing ingestion function and record success or failure."""

        try:
            result = ingestion_function(*args, **kwargs)
        except Exception as error:
            self.record_ingestion(False, error=str(error))
            raise

        self.record_ingestion(True)
        return result

    def run_retrieval(
        self,
        retrieval_function: Callable[..., Any],
        *args: Any,
        score_getter: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run an existing retrieval function and measure its latency.

        ``score_getter`` is optional. It may return either:

        - ``{"similarity_score": 0.9, "confidence_score": 0.85}``
        - ``(similarity_score, confidence_score)``

        Without a getter, a dictionary result is checked for those keys.
        """

        started_at = time.monotonic()

        try:
            result = retrieval_function(*args, **kwargs)
        except Exception as error:
            latency = time.monotonic() - started_at
            self.record_retrieval(
                False,
                latency,
                error=str(error),
            )
            raise

        latency = time.monotonic() - started_at
        similarity, confidence = self._get_scores(result, score_getter)
        self.record_retrieval(
            True,
            latency,
            similarity_score=similarity,
            confidence_score=confidence,
        )
        return result

    def run_generation(
        self,
        generation_function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run an existing Step 6 generation function and measure latency."""

        started_at = time.monotonic()

        try:
            result = generation_function(*args, **kwargs)
        except Exception as error:
            latency = time.monotonic() - started_at
            self.record_generation(False, latency, error=str(error))
            raise

        latency = time.monotonic() - started_at
        self.record_generation(True, latency)
        return result

    @staticmethod
    def _get_scores(
        result: Any,
        score_getter: Optional[Callable[[Any], Any]],
    ) -> tuple[Optional[float], Optional[float]]:
        if score_getter is not None:
            scores = score_getter(result)
        elif isinstance(result, dict):
            scores = result
        else:
            scores = {}

        if isinstance(scores, (tuple, list)):
            similarity = scores[0] if len(scores) > 0 else None
            confidence = scores[1] if len(scores) > 1 else None
            return similarity, confidence

        if not isinstance(scores, dict):
            return None, None

        similarity = scores.get("similarity_score")
        if similarity is None:
            similarity = scores.get("similarity")

        confidence = scores.get("confidence_score")
        if confidence is None:
            confidence = scores.get("confidence")

        return similarity, confidence

    def monitor_query(
        self,
        query: Any,
        retrieval_function: Callable[..., Any],
        generation_function: Callable[..., Any],
        *,
        fallback_function: Optional[Callable[[Any, Any], Any]] = None,
        human_escalation_function: Optional[Callable[[Any, Any], Any]] = None,
        score_getter: Optional[Callable[[Any], Any]] = None,
        retrieval_args: tuple[Any, ...] = (),
        retrieval_kwargs: Optional[dict[str, Any]] = None,
    ) -> QueryResult:
        """Monitor one retrieval-plus-generation query.

        When confidence is below the configured threshold, generation is not
        called. A fallback and human-escalation event are recorded instead.
        Optional callbacks let the existing application handle those paths.
        """

        kwargs = retrieval_kwargs or {}
        retrieval_result = self.run_retrieval(
            retrieval_function,
            query,
            *retrieval_args,
            score_getter=score_getter,
            **kwargs,
        )

        similarity, confidence = self._get_scores(
            retrieval_result,
            score_getter,
        )
        score_for_fallback = (
            confidence if confidence is not None else similarity
        )
        low_confidence = (
            score_for_fallback is not None
            and score_for_fallback < self.confidence_threshold
        )

        if low_confidence:
            self.record_fallback(
                confidence_score=confidence,
            )
            self.record_human_escalation(
                confidence_score=confidence,
            )

            answer = None
            if fallback_function is not None:
                answer = fallback_function(query, retrieval_result)

            if human_escalation_function is not None:
                human_escalation_function(query, retrieval_result)

            return QueryResult(
                answer=answer,
                retrieval_result=retrieval_result,
                used_fallback=True,
                human_escalated=True,
                confidence_score=confidence,
            )

        answer = self.run_generation(
            generation_function,
            query,
            retrieval_result,
        )

        return QueryResult(
            answer=answer,
            retrieval_result=retrieval_result,
            used_fallback=False,
            human_escalated=False,
            confidence_score=confidence,
        )

    def summary(self) -> dict[str, Any]:
        """Return clear metrics for the current monitor instance."""

        ingestion_failure_rate = (
            self._ingestion_failures / self._ingestion_attempts
            if self._ingestion_attempts
            else 0.0
        )
        retrieval_failure_rate = (
            self._retrieval_failures / self._retrieval_attempts
            if self._retrieval_attempts
            else 0.0
        )
        generation_failure_rate = (
            self._generation_failures / self._generation_attempts
            if self._generation_attempts
            else 0.0
        )

        return {
            "confidence_threshold": self.confidence_threshold,
            "ingestion": {
                "attempts": self._ingestion_attempts,
                "failures": self._ingestion_failures,
                "failure_rate": ingestion_failure_rate,
            },
            "retrieval": {
                "attempts": self._retrieval_attempts,
                "failures": self._retrieval_failures,
                "failure_rate": retrieval_failure_rate,
                "latency_seconds": _score_statistics(
                    self._retrieval_latencies
                ),
                "similarity_scores": _score_statistics(
                    self._similarity_scores
                ),
                "confidence_scores": _score_statistics(
                    self._confidence_scores
                ),
                "low_confidence_retrievals": (
                    self._low_confidence_retrievals
                ),
            },
            "llm_generation": {
                "attempts": self._generation_attempts,
                "failures": self._generation_failures,
                "failure_rate": generation_failure_rate,
                "latency_seconds": _score_statistics(
                    self._generation_latencies
                ),
            },
            "fallback_events": self._fallback_events,
            "human_escalation_events": self._human_escalation_events,
            "telemetry_write_errors": self.telemetry_write_errors,
        }

    def save_summary(self) -> dict[str, Any]:
        """Save the current summary to a JSON file and return it."""

        result = self.summary()

        try:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text(
                json.dumps(result, indent=2) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as error:
            self.telemetry_write_errors += 1
            self.last_write_error = str(error)

            if self.strict_file_writes:
                raise TelemetryWriteError(str(error)) from error

        return result

    def print_summary(self) -> None:
        """Print the current summary."""

        print(json.dumps(self.summary(), indent=2))


# ---------------------------------------------------------------------------
# Small demo/test
# ---------------------------------------------------------------------------


def _demo_ingest(document: str) -> str:
    return f"ingested: {document}"


def _demo_ingest_failure(document: str) -> str:
    raise RuntimeError("demo ingestion failure")


def _demo_retrieve_high_confidence(query: str) -> dict[str, Any]:
    return {
        "documents": ["Password reset instructions"],
        "similarity_score": 0.92,
        "confidence_score": 0.88,
    }


def _demo_retrieve_low_confidence(query: str) -> dict[str, Any]:
    return {
        "documents": [],
        "similarity_score": 0.30,
        "confidence_score": 0.25,
    }


def _demo_retrieve_failure(query: str) -> dict[str, Any]:
    raise RuntimeError("demo retrieval failure")


def _demo_generate(query: str, retrieval_result: dict[str, Any]) -> str:
    time.sleep(0.001)
    return "Here are the password reset instructions."


def _demo_fallback(
    query: str,
    retrieval_result: dict[str, Any],
) -> str:
    return "I am sending this question to a support representative."


def run_demo() -> None:
    """Exercise success, failure, latency, score, and escalation metrics."""

    with TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        telemetry = TelemetryMonitor(
            metrics_path=directory / "telemetry.jsonl",
            summary_path=directory / "summary.json",
            confidence_threshold=0.70,
        )

        # Ingestion success and failure.
        telemetry.run_ingestion(_demo_ingest, "help.txt")

        try:
            telemetry.run_ingestion(
                _demo_ingest_failure,
                "broken-help.txt",
            )
        except RuntimeError:
            pass

        # High confidence: generation runs and the query succeeds.
        high_confidence_result = telemetry.monitor_query(
            "How do I reset my password?",
            _demo_retrieve_high_confidence,
            _demo_generate,
        )

        assert high_confidence_result.used_fallback is False
        assert high_confidence_result.human_escalated is False
        assert telemetry._generation_attempts == 1

        # Low confidence: fallback and human escalation are recorded.
        low_confidence_result = telemetry.monitor_query(
            "I have a question not covered by the KB.",
            _demo_retrieve_low_confidence,
            _demo_generate,
            fallback_function=_demo_fallback,
        )

        assert low_confidence_result.used_fallback is True
        assert low_confidence_result.human_escalated is True
        assert telemetry._fallback_events == 1
        assert telemetry._human_escalation_events == 1
        assert telemetry._generation_attempts == 1

        # Retrieval failure is counted and the original exception is kept.
        try:
            telemetry.run_retrieval(
                _demo_retrieve_failure,
                "A retrieval failure test",
            )
        except RuntimeError:
            pass

        result = telemetry.save_summary()
        telemetry.print_summary()

        assert result["ingestion"]["failures"] == 1
        assert result["retrieval"]["failures"] == 1
        assert result["fallback_events"] == 1
        assert result["human_escalation_events"] == 1
        assert result["retrieval"]["latency_seconds"]["count"] == 3
        assert result["llm_generation"]["latency_seconds"]["count"] == 1
        assert (directory / "telemetry.jsonl").exists()
        assert (directory / "summary.json").exists()

    print("Step 7 demo assertions passed.")


if __name__ == "__main__":
    run_demo()