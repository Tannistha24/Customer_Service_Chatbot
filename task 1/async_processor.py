from __future__ import annotations

import concurrent.futures
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hash import QueueCallback, compute_sha256, utc_now_iso

PROCESSING_TIMEOUT_SECONDS = 30.0


@dataclass
class BackgroundJob:
    job_id: str
    file_id: str
    sha256: str
    storage_path: str
    mime_type: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    customer_acknowledged: bool = False


@dataclass
class ProcessingOutcome:
    status: str
    result: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    customer_message: Optional[str] = None
    error_message: Optional[str] = None


class BackgroundWorker:
    def __init__(self, max_workers: int = 4):
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._jobs: Dict[str, BackgroundJob] = {}
        self._futures: Dict[str, concurrent.futures.Future] = {}
        self._processing_files: set[str] = set()
        self._lock = __import__('threading').RLock()
        self._max_workers = max_workers

    def start(self) -> None:
        if self._executor is None or self._executor._shutdown:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="bg-worker-"
            )

    def shutdown(self, wait: bool = True) -> None:
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def is_file_processing(self, file_id: str) -> bool:
        """Check if a file is already being processed."""
        with self._lock:
            return file_id in self._processing_files

    def submit(
        self,
        processor: Callable[[BackgroundJob], Dict[str, Any]],
        file_id: str,
        sha256: str,
        storage_path: str,
        mime_type: str,
        on_acknowledge: Optional[Callable[[BackgroundJob], None]] = None,
    ) -> Optional[BackgroundJob]:
        """Submit a job for background processing.
        
        Returns None if the file is already being processed to prevent duplicates.
        """
        self.start()

        with self._lock:
            if file_id in self._processing_files:
                return None

            self._processing_files.add(file_id)

        job = BackgroundJob(
            job_id=f"bg_{uuid.uuid4().hex[:12]}",
            file_id=file_id,
            sha256=sha256,
            storage_path=storage_path,
            mime_type=mime_type,
            status="queued",
            created_at=utc_now_iso(),
        )

        with self._lock:
            self._jobs[job.job_id] = job

        def _execute() -> Dict[str, Any]:
            try:
                with self._lock:
                    job.status = "processing"
                    job.started_at = utc_now_iso()
                
                result = processor(job)
                
                with self._lock:
                    job.status = "completed"
                    job.completed_at = utc_now_iso()
                    job.result = result
                return result
            except Exception as e:
                with self._lock:
                    job.status = "failed"
                    job.completed_at = utc_now_iso()
                    job.error_message = str(e)
                raise
            finally:
                with self._lock:
                    self._processing_files.discard(file_id)

        future = self._executor.submit(_execute)

        with self._lock:
            self._futures[job.job_id] = future

        if on_acknowledge:
            on_acknowledge(job)
            with self._lock:
                job.customer_acknowledged = True

        return job

    def get_job(self, job_id: str) -> Optional[BackgroundJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Optional[BackgroundJob]:
        future = None
        with self._lock:
            future = self._futures.get(job_id)

        if future and not future.done():
            try:
                future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                pass
            except Exception:
                pass

        with self._lock:
            if job_id in self._futures and self._futures[job_id].done():
                del self._futures[job_id]

        return self.get_job(job_id)


class FileProcessingTimeout:
    def __init__(
        self,
        timeout_seconds: float = PROCESSING_TIMEOUT_SECONDS,
        background_worker: Optional[BackgroundWorker] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self._background = background_worker or BackgroundWorker()
        self._acknowledgment_callbacks: list[Callable[[str, str], None]] = []

    def on_customer_acknowledgment(self, callback: Callable[[str, str], None]) -> None:
        self._acknowledgment_callbacks.append(callback)

    def _send_acknowledgment(self, job: BackgroundJob) -> None:
        message = (
            "We are analyzing your document in the background "
            f"and will update you shortly. Job ID: {job.job_id}"
        )
        for callback in self._acknowledgment_callbacks:
            try:
                callback(job.job_id, message)
            except Exception:
                pass

    def process(
        self,
        file_path: Path,
        processor: Callable[[Path], Dict[str, Any]],
        file_id: Optional[str] = None,
        sha256: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> ProcessingOutcome:
        """Process a file with a 30-second timeout.
        
        If processing completes within the timeout, returns the result immediately.
        If timeout occurs, returns immediately with job_id and sends async acknowledgment.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(processor, file_path)

        try:
            result = future.result(timeout=self.timeout_seconds)
            executor.shutdown(wait=False)
            return ProcessingOutcome(status="completed", result=result)

        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False)

        except Exception as e:
            executor.shutdown(wait=False)
            return ProcessingOutcome(status="error", error_message=str(e))

        if not sha256:
            try:
                sha256, _, _, _ = compute_sha256(file_path)
            except Exception:
                sha256 = "unknown"
        file_id = file_id or sha256 or "unknown"

        job = self._background.submit(
            processor=lambda job: processor(Path(job.storage_path)),
            file_id=file_id,
            sha256=sha256,
            storage_path=str(file_path),
            mime_type=mime_type or "application/octet-stream",
            on_acknowledge=self._send_acknowledgment,
        )

        if job is None:
            return ProcessingOutcome(
                status="duplicate",
                customer_message=(
                    f"This file (ID: {file_id}) is already being processed. "
                    "Please wait for it to complete."
                ),
            )

        return ProcessingOutcome(
            status="timeout",
            job_id=job.job_id,
            customer_message=(
                "We are analyzing your document in the background "
                f"and will update you shortly. Job ID: {job.job_id}"
            ),
        )

    def shutdown(self) -> None:
        self._background.shutdown(wait=True)


def create_step2_queue_callback(
    timeout_seconds: float = PROCESSING_TIMEOUT_SECONDS,
    customer_notifier: Optional[Callable[[str, str], None]] = None,
    file_processor: Optional[Callable[[Path], Dict[str, Any]]] = None,
) -> QueueCallback:
    timeout_handler = FileProcessingTimeout(timeout_seconds=timeout_seconds)

    if customer_notifier:
        timeout_handler.on_customer_acknowledgment(customer_notifier)

    def _queue_callback(payload: Dict[str, Any]) -> Dict[str, Any]:
        storage_path = payload.get("storage_path", "")

        def _process(path: Path) -> Dict[str, Any]:
            if file_processor:
                return file_processor(path)
            return {"status": "processed", "path": str(path)}

        outcome = timeout_handler.process(
            file_path=Path(storage_path),
            processor=_process,
            file_id=payload.get("file_id"),
            sha256=payload.get("sha256"),
            mime_type=payload.get("mime_type", "application/octet-stream"),
        )

        result = {
            "job_id": outcome.job_id or payload.get("job_id"),
            "file_id": payload.get("file_id"),
            "sha256": payload.get("sha256"),
            "storage_path": storage_path,
            "mime_type": payload.get("mime_type"),
            "created_at": payload.get("created_at", utc_now_iso()),
        }

        if outcome.status == "completed":
            result.update({
                "status": "completed",
                "completed_at": utc_now_iso(),
                "result": outcome.result,
            })
        elif outcome.status == "duplicate":
            result.update({
                "status": "duplicate",
                "acknowledgment": outcome.customer_message,
            })
        else:
            result.update({
                "status": "background_processing",
                "acknowledgment": outcome.customer_message,
                "background_job_id": outcome.job_id,
            })

        return result

    _queue_callback._timeout_handler = timeout_handler  # type: ignore
    return _queue_callback


def shutdown_step2_callback(callback: QueueCallback) -> None:
    if hasattr(callback, '_timeout_handler'):
        callback._timeout_handler.shutdown()  # type: ignore
