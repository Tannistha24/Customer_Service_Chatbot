from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import logging
import re
import sys
import tempfile
from datetime import time as clock_time
from difflib import SequenceMatcher
from pathlib import Path
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

from telemetry_monitoring import TelemetryMonitor
from schedular import MaintenanceScheduler, MaintenanceWindowConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
ATTACHED_ROOT = PROJECT_ROOT / "attached_assets"


def _load_existing_module(
    module_name: str,
    normal_filename: str,
    attached_filename: str,
) -> ModuleType:
    """Load an existing step module without editing it."""

    candidates = (
        PROJECT_ROOT / normal_filename,
        ATTACHED_ROOT / attached_filename,
    )

    for path in candidates:
        if not path.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {module_name}. Searched: {searched}"
    )


step1_hash = _load_existing_module(
    "step1_hash",
    "hash.py",
    "hash_1788356957516.py",
)

step2_manager = _load_existing_module(
    "step2_manager",
    "kb_manager.py",
    "kb_manager_1788356957520.py",
)

step3_quality = _load_existing_module(
    "step3_quality",
    "quality.py",
    "quality_1788356957522.py",
)

step5_security = _load_existing_module(
    "step5_security",
    "security.py",
    "security_1788356957522.py",
)

step6_monitoring = _load_existing_module(
    "step6_monitoring",
    # Prefer the project-local Step 6 source when available. The attached
    # assets name remains a fallback for previously shipped bundles.
    "monitoring.py",
    "monitoring_1788356957521.py",
)


class MockHealthChatbot:
    """Offline chatbot used by Step 6 post-activation health checks."""

    def __call__(self, query: str, kb_version: str) -> str:
        return f"{kb_version} health-check response"


def default_generation(
    query: str,
    retrieval_result: dict[str, Any],
) -> str:
    """Simple offline generation function for local testing."""

    retrieved_chunk = retrieval_result.get("retrieved_chunk", "")

    if retrieved_chunk:
        return retrieved_chunk

    return "I don't know."


def default_fallback(
    query: str,
    retrieval_result: dict[str, Any],
) -> str:
    return "I am forwarding this question to a human support representative."


def default_human_escalation(
    query: str,
    retrieval_result: dict[str, Any],
) -> None:
    print(f"[human-escalation] Query forwarded: {query}")


class RAGRuntime:
    """Connect existing pipeline steps and add Step 7 telemetry."""

    def __init__(
        self,
        root: str | Path,
        *,
        generation_function: Optional[Callable[..., Any]] = None,
        health_chatbot: Optional[Callable[[str, str], Any]] = None,
        confidence_threshold: float = 0.70,
        health_check_duration_seconds: float = 300,
        health_check_interval_seconds: float = 1.0,
        maintenance_window_start: clock_time = clock_time(hour=2, minute=0),
        maintenance_window_end: clock_time = clock_time(hour=4, minute=0),
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        self.ready_dir = self.root / "ready"
        self.storage_root = self.root / "kb_storage"
        self.eval_results_dir = self.root / "eval_results"
        self.eval_dataset_path = self.root / "eval_dataset.json"
        self.eval_config_path = self.root / "eval_config.json"

        self.telemetry = TelemetryMonitor(
            metrics_path=self.root / "step7_telemetry.jsonl",
            summary_path=self.root / "step7_summary.json",
            confidence_threshold=confidence_threshold,
        )

        self.generation_function = (
            generation_function or default_generation
        )
        self.health_chatbot = health_chatbot or MockHealthChatbot()
        self.health_check_duration_seconds = health_check_duration_seconds
        self.health_check_interval_seconds = health_check_interval_seconds

        self.version_manager = step2_manager.KBVersionManager(
            storage_root=str(self.storage_root),
        )

        # Step 4: maintenance window configuration for when updates may be applied.
        self.maintenance_config = MaintenanceWindowConfig(
            start=maintenance_window_start,
            end=maintenance_window_end,
        )

    def _run_step1_with_telemetry(self) -> dict[str, int]:
        """Run Step 1 and record its aggregate result as one event."""

        try:
            counts = step1_hash.run_step1(self.root)
        except Exception as error:
            self.telemetry.record_ingestion(
                False,
                error=str(error),
            )
            raise

        if not isinstance(counts, dict):
            self.telemetry.record_ingestion(
                False,
                error="Step 1 returned an unexpected result",
            )
            raise TypeError("Step 1 must return its counts dictionary")

        failures = int(counts.get("errors", 0))
        quarantined = int(counts.get("quarantine", 0))
        ingestion_ok = failures == 0 and quarantined == 0

        error_message = None
        if not ingestion_ok:
            error_message = (
                f"Step 1 reported errors={failures}, "
                f"quarantine={quarantined}"
            )

        self.telemetry.record_ingestion(
            ingestion_ok,
            error=error_message,
        )

        return counts

    def _retrieve(
        self,
        query: str,
        kb_data_path: Path,
    ) -> dict[str, Any]:
        """Use the existing Step 3 retrieval helpers."""

        chunks = step3_quality.load_kb_chunks(
            str(kb_data_path),
        )
        retrieved_chunk, retrieval_score = (
            step3_quality.retrieve_best_chunk(
                query,
                chunks,
            )
        )

        return {
            "retrieved_chunk": retrieved_chunk or "",
            "retrieval_score": retrieval_score,
            "similarity_score": retrieval_score,
            "confidence_score": retrieval_score,
            "num_chunks": len(chunks),
        }
    def _lookup_dataset(self, query: str, min_score: float = 0.75):
        """Return the closest company dataset answer, or None."""
        path = Path(__file__).resolve().parent.parent / "dataset" / "dataset.csv"
        try:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                text = path.read_text(encoding="cp1252")
            rows = list(csv.DictReader(io.StringIO(text, newline="")))
        except Exception as e:
            logger.warning(f"Dataset lookup unavailable: {e}")
            return None

        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()

        q = norm(query)
        best, best_score = None, 0.0
        for row in rows:
            prompt = (row.get("prompt") or "").strip()
            response = (row.get("response") or "").strip()
            if not prompt or not response:
                continue
            score = SequenceMatcher(None, q, norm(prompt)).ratio()
            if score > best_score:
                best, best_score = response, score

        if best and best_score >= min_score:
            return {"response": best, "score": round(best_score, 3)}
        return None

    def answer_query(
        self,
        query: str,
        *,
        access_token: Optional[str] = None,
        version_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run access control, security, retrieval, generation, and telemetry."""

        # security_result = step5_security.security_check(
        #     query,
        #     access_token=access_token,
        # )
        security_result = step5_security.security_check(
            query,
        )

        if not security_result["allowed"]:
            return security_result
        match = self._lookup_dataset(security_result["sanitized_input"])
        if match:
            return {
                **security_result,
                "llm_response": match["response"],
                "used_fallback": False,
                "human_escalated": False,
                "confidence_score": match["score"],
            }

        active_version = (
            version_name
            or self.version_manager.get_active_version()
        )

        if active_version is None:
            raise RuntimeError("No active KB version is available")

        kb_data_path = (
            self.storage_root
            / "kb_versions"
            / active_version
            / "kb_data"
        )

        def retrieve(query_text: str) -> dict[str, Any]:
            return self._retrieve(query_text, kb_data_path)

        result = self.telemetry.monitor_query(
            query=security_result["sanitized_input"],
            retrieval_function=retrieve,
            generation_function=self.generation_function,
            fallback_function=default_fallback,
            human_escalation_function=default_human_escalation,
        )

        return {
            **security_result,
            "llm_response": result.answer,
            "used_fallback": result.used_fallback,
            "human_escalated": result.human_escalated,
            "confidence_score": result.confidence_score,
        }

    def run_update(self) -> dict[str, Any]:
        """Run ingestion, versioning, quality, activation, and health check."""

        try:
            ingestion_counts = self._run_step1_with_telemetry()

            ready_files = [
                path
                for path in self.ready_dir.iterdir()
                if path.is_file()
            ]

            if not ready_files:
                raise RuntimeError(
                    "Step 1 completed but no files are available in the ready folder"
                )

            previous_valid_version = (
                self.version_manager.get_active_version()
            )

            new_version = self.version_manager.create_version(
                str(self.ready_dir),
                notes="Created by runtime.py",
            )

            kb_data_path = (
                self.storage_root
                / "kb_versions"
                / new_version
                / "kb_data"
            )

            quality_result = step3_quality.run_quality_gate(
                version_name=new_version,
                kb_data_path=str(kb_data_path),
                eval_dataset_path=str(self.eval_dataset_path),
                eval_results_dir=str(self.eval_results_dir),
                eval_config_path=str(self.eval_config_path),
            )

            if not quality_result["passed"]:
                return {
                    "status": "rejected_by_quality_gate",
                    "ingestion": ingestion_counts,
                    "version": new_version,
                    "quality": quality_result,
                }

            # Step 4: apply approved updates only in the maintenance window.
            scheduler = MaintenanceScheduler(
                config=self.maintenance_config,
                # A long-lived service would call scheduler.start() and let the
                # periodic check drain the queue. This script returns a queued
                # status when outside the window.
                check_interval_seconds=60,
            )

            def apply_update() -> None:
                # Existing Step 2 activation.
                self.version_manager.set_active_version(new_version)

                # Existing Step 6 health check. It calls the actual
                # Step 2 rollback_to(previous_valid_version) on failure.
                health_result = step6_monitoring.monitor_activated_version(
                    version_manager=self.version_manager,
                    chatbot=self.health_chatbot,
                    activated_version=new_version,
                    previous_valid_version=previous_valid_version,
                    duration_seconds=self.health_check_duration_seconds,
                    interval_seconds=self.health_check_interval_seconds,
                )

                # A failed health check means the update should be treated as
                # unsuccessful (it was rolled back or could not be validated).
                if not health_result.passed:
                    raise RuntimeError("Post-activation health check failed")

            if not scheduler.is_in_maintenance_window():
                scheduler.submit_approved_update(new_version, apply_update)
                return {
                    "status": "approved_pending_maintenance_window",
                    "ingestion": ingestion_counts,
                    "version": new_version,
                    "quality": quality_result,
                    "active_version": self.version_manager.get_active_version(),
                    "maintenance_window": {
                        "start": str(self.maintenance_config.start),
                        "end": str(self.maintenance_config.end),
                    },
                }

            # In-window: apply immediately (submit will run synchronously).
            scheduler.submit_approved_update(new_version, apply_update)
            return {
                "status": "active",
                "ingestion": ingestion_counts,
                "version": new_version,
                "quality": quality_result,
                "active_version": self.version_manager.get_active_version(),
            }
        finally:
            # Keep the summary available even when an integration step raises.
            self.telemetry.save_summary()


def create_demo_data(root: Path) -> None:
    """Create one valid document and a small Step 3 evaluation dataset."""

    incoming = root / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)

    (incoming / "support.txt").write_text(
        "How do I reset my password? "
        "Customers can reset their password from account settings. "
        "Contact customer support for additional help.",
        encoding="utf-8",
    )

    dataset = [
        {
            "question": "How do I reset my password?",
            "expected_answer": (
                "Customers can reset their password from account settings."
            ),
        }
    ]

    (root / "eval_dataset.json").write_text(
        json.dumps(dataset, indent=2),
        encoding="utf-8",
    )


def run_demo() -> None:
    """Run the complete runtime without LangChain or external APIs."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        create_demo_data(root)

        runtime = RAGRuntime(
            root,
            # The demo's simple word-overlap retriever produces scores
            # below 0.70 for a longer paragraph. Production remains 0.70.
            confidence_threshold=0.30,
            health_check_duration_seconds=0.01,
            health_check_interval_seconds=0.001,
            # Keep the demo deterministic: always allow updates to apply.
            maintenance_window_start=clock_time(hour=0, minute=0),
            maintenance_window_end=clock_time(hour=23, minute=59),
        )

        pipeline_result = runtime.run_update()

        assert pipeline_result["status"] == "active"
        assert pipeline_result["active_version"] is not None

        answer_result = runtime.answer_query(
            "How do I reset my password?",
        )

        assert answer_result["allowed"] is True
        assert answer_result["used_fallback"] is False
        assert answer_result["human_escalated"] is False

        # Verify the actual Step 5 dictionary fields and sanitized-input path.
        pii_check = step5_security.security_check(
            "How do I reset my password? Email demo@example.com",
        )
        assert pii_check["allowed"] is True
        assert "demo@example.com" not in pii_check["sanitized_input"]
        assert "[EMAIL_REDACTED]" in pii_check["sanitized_input"]

        # Low confidence must skip generation and record fallback/escalation.
        low_confidence_result = runtime.answer_query(
            "What is the refund policy?",
        )
        assert low_confidence_result["allowed"] is True
        assert low_confidence_result["used_fallback"] is True
        assert low_confidence_result["human_escalated"] is True

        summary = runtime.telemetry.save_summary()

        assert summary["ingestion"]["attempts"] == 1
        assert summary["retrieval"]["attempts"] == 2
        assert summary["llm_generation"]["attempts"] == 1
        assert summary["retrieval"]["low_confidence_retrievals"] == 1
        assert summary["fallback_events"] == 1
        assert summary["human_escalation_events"] == 1
        assert (root / "step7_telemetry.jsonl").exists()
        assert (root / "step7_summary.json").exists()

        print(json.dumps(pipeline_result, indent=2))
        print(json.dumps(summary, indent=2))
        print("Runtime demo assertions passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the existing RAG pipeline with Step 7 telemetry",
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Pipeline data directory. If omitted, run an isolated demo."
        ),
    )
    args = parser.parse_args()

    if args.root is None:
        run_demo()
        return

    runtime = RAGRuntime(args.root)
    result = runtime.run_update()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()