from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


MANIFEST_NAME = "manifest.json"
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_RETENTION_HOURS = 24 * 7
DEFAULT_CLEANUP_INTERVAL_SECONDS = 60 * 60
DEFAULT_PDF_SECURITY_SCAN_BYTES = 2 * 1024 * 1024

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".html",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
}

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".html"}
OOXML_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
EXPECTED_MIME_TYPES: Dict[str, set[str]] = {
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/plain"},
    ".csv": {"text/csv", "text/plain"},
    ".json": {"application/json", "text/json", "text/plain"},
    ".html": {"text/html", "application/xhtml+xml", "text/plain"},
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
    },
    ".pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/zip",
    },
    ".xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/zip",
    },
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
}

UNSAFE_PDF_MARKERS = (
    b"/JavaScript",
    b"/JS",
    b"/Launch",
    b"/EmbeddedFile",
    b"/RichMedia",
    b"/OpenAction",
    b"/AA",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def parse_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class IngestionConfig:
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    retention_hours: int = DEFAULT_RETENTION_HOURS

    @classmethod
    def from_env(cls) -> "IngestionConfig":
        max_size = int(os.getenv("INGEST_MAX_FILE_SIZE_BYTES", DEFAULT_MAX_FILE_SIZE_BYTES))
        retention_hours = int(os.getenv("INGEST_RETENTION_HOURS", DEFAULT_RETENTION_HOURS))
        return cls(max_file_size_bytes=max_size, retention_hours=retention_hours)

    @property
    def retention_delta(self) -> timedelta:
        return timedelta(hours=self.retention_hours)


@dataclass
class ValidationResult:
    ok: bool
    file_extension: str
    detected_extension: Optional[str]
    client_mime_type: Optional[str]
    detected_mime_type: Optional[str]
    sha256: Optional[str]
    size_bytes: Optional[int]
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class FileResult:
    status: str  # "ready" | "duplicate" | "quarantine" | "queue_failed"
    reason: str
    file_id: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    source_path: Optional[str] = None
    dest_path: Optional[str] = None
    job_id: Optional[str] = None


QueueCallback = Callable[[Dict[str, Any]], Dict[str, Any]]


def default_manifest() -> Dict[str, Any]:
    return {
        "version": 2,
        "generated_at": utc_now_iso(),
        "documents": {},
        "duplicates": [],
        "quarantine": [],
        "jobs": [],
        "job_failures": [],
        "deleted": [],
    }


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return default_manifest()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        defaults = default_manifest()
        for key, value in defaults.items():
            data.setdefault(key, value if not isinstance(value, list) else [])
        return data
    except Exception:
        backup = path.with_suffix(".corrupted.backup.json")
        try:
            shutil.copy2(path, backup)
        except Exception:
            pass
        manifest = default_manifest()
        manifest["quarantine"].append(
            {
                "file_name": path.name,
                "source_path": str(path),
                "dest_path": None,
                "reason": "Manifest was unreadable or corrupted; started a fresh manifest.",
                "timestamp": utc_now_iso(),
            }
        )
        return manifest


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    manifest["generated_at"] = utc_now_iso()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def unique_destination(dest_dir: Path, file_name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / file_name
    if not candidate.exists():
        return candidate

    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    for i in range(1, 10_000):
        candidate = dest_dir / f"{stem}__{stamp}__{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique destination name for {file_name}")


def relpath_str(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def compute_sha256(path: Path) -> Tuple[str, int, bytes, bytes]:
    h = hashlib.sha256()
    size = 0
    first_chunk = b""
    last_chunk = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            if not first_chunk:
                first_chunk = chunk[:4096]
            last_chunk = chunk[-4096:]
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size, first_chunk, last_chunk


def sniff_binary_file_type(first_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    if first_bytes.startswith(b"%PDF-"):
        return ".pdf", "application/pdf"
    if first_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if first_bytes.startswith(b"\xFF\xD8\xFF"):
        return ".jpg", "image/jpeg"
    if first_bytes.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return ".zip", "application/zip"
    return None, None


def inspect_ooxml_type(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return None, None, "Corrupted OOXML/ZIP file"
    except OSError:
        return None, None, "File could not be opened for OOXML inspection"

    if "[Content_Types].xml" not in names:
        return None, None, "OOXML file missing [Content_Types].xml"
    if any(name.startswith("word/") for name in names):
        return ".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", None
    if any(name.startswith("ppt/") for name in names):
        return ".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", None
    if any(name.startswith("xl/") for name in names):
        return ".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", None
    return None, None, "Unsupported OOXML container"


def guess_client_mime_type(path: Path) -> Optional[str]:
    mime, _ = mimetypes.guess_type(path.name)
    return mime


def validate_binary_integrity(
    *,
    path: Path,
    file_extension: str,
    detected_extension: Optional[str],
    first_bytes: bytes,
    last_bytes: bytes,
) -> Optional[str]:
    if file_extension == ".pdf":
        if detected_extension != ".pdf":
            return "File header does not match PDF content"
        if b"%%EOF" not in last_bytes:
            return "Corrupted PDF file"
    elif file_extension == ".png":
        if detected_extension != ".png":
            return "File header does not match PNG content"
        if b"IEND" not in last_bytes:
            return "Corrupted PNG file"
    elif file_extension in {".jpg", ".jpeg"}:
        if detected_extension != ".jpg":
            return "File header does not match JPEG content"
        if not last_bytes.endswith(b"\xFF\xD9"):
            return "Corrupted JPEG file"
    elif file_extension in OOXML_EXTENSIONS:
        if detected_extension != ".zip":
            return "File header does not match OOXML content"
        _, _, ooxml_error = inspect_ooxml_type(path)
        if ooxml_error:
            return ooxml_error
    return None


def validate_text_file(path: Path, file_extension: str, size_bytes: int) -> Optional[str]:
    try:
        with path.open("rb") as f:
            sample = f.read(min(size_bytes, 256 * 1024))
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return "Text file is not valid UTF-8"
    except OSError:
        return "Text file could not be read"

    if file_extension == ".json" and size_bytes <= 5 * 1024 * 1024:
        try:
            with path.open("r", encoding="utf-8") as f:
                json.load(f)
        except Exception:
            return "Corrupted JSON file"
    return None


def security_screen(path: Path, file_extension: str) -> Optional[str]:
    if file_extension != ".pdf":
        return None
    try:
        scan_limit = int(os.getenv("INGEST_PDF_SECURITY_SCAN_BYTES", DEFAULT_PDF_SECURITY_SCAN_BYTES))
        scan_limit = max(0, scan_limit)
        with path.open("rb") as f:
            sample = f.read(scan_limit)
    except OSError:
        return "Security screening failed"

    for marker in UNSAFE_PDF_MARKERS:
        if marker in sample:
            return "Unsafe PDF features detected"
    return None


def validate_and_hash(
    path: Path,
    *,
    config: Optional[IngestionConfig] = None,
) -> ValidationResult:
    config = config or IngestionConfig.from_env()

    if not path.exists():
        return ValidationResult(
            ok=False,
            file_extension="",
            detected_extension=None,
            client_mime_type=None,
            detected_mime_type=None,
            sha256=None,
            size_bytes=None,
            error_code="missing_file",
            error_message="File does not exist",
        )
    if not path.is_file():
        return ValidationResult(
            ok=False,
            file_extension=path.suffix.lower(),
            detected_extension=None,
            client_mime_type=None,
            detected_mime_type=None,
            sha256=None,
            size_bytes=None,
            error_code="invalid_file",
            error_message="Not a regular file",
        )

    file_extension = path.suffix.lower()
    client_mime_type = guess_client_mime_type(path)
    if file_extension not in SUPPORTED_EXTENSIONS:
        return ValidationResult(
            ok=False,
            file_extension=file_extension,
            detected_extension=None,
            client_mime_type=client_mime_type,
            detected_mime_type=None,
            sha256=None,
            size_bytes=None,
            error_code="unsupported_type",
            error_message=f"Unsupported file type: {file_extension or '(no extension)'}",
        )

    try:
        sha256, size_bytes, first_bytes, last_bytes = compute_sha256(path)
    except PermissionError:
        return ValidationResult(
            ok=False,
            file_extension=file_extension,
            detected_extension=None,
            client_mime_type=client_mime_type,
            detected_mime_type=None,
            sha256=None,
            size_bytes=None,
            error_code="read_error",
            error_message="File is not readable",
        )
    except OSError:
        return ValidationResult(
            ok=False,
            file_extension=file_extension,
            detected_extension=None,
            client_mime_type=client_mime_type,
            detected_mime_type=None,
            sha256=None,
            size_bytes=None,
            error_code="read_error",
            error_message="File read failed",
        )

    if size_bytes == 0:
        return ValidationResult(
            ok=False,
            file_extension=file_extension,
            detected_extension=None,
            client_mime_type=client_mime_type,
            detected_mime_type=None,
            sha256=None,
            size_bytes=0,
            error_code="corrupted_file",
            error_message="Empty file",
        )

    if size_bytes > config.max_file_size_bytes:
        return ValidationResult(
            ok=False,
            file_extension=file_extension,
            detected_extension=None,
            client_mime_type=client_mime_type,
            detected_mime_type=None,
            sha256=None,
            size_bytes=size_bytes,
            error_code="oversized_file",
            error_message="File exceeds the configured maximum size",
        )

    detected_extension = None
    detected_mime_type = None

    if file_extension in TEXT_EXTENSIONS:
        text_error = validate_text_file(path, file_extension, size_bytes)
        if text_error:
            return ValidationResult(
                ok=False,
                file_extension=file_extension,
                detected_extension=file_extension,
                client_mime_type=client_mime_type,
                detected_mime_type=client_mime_type or "text/plain",
                sha256=None,
                size_bytes=size_bytes,
                error_code="corrupted_file",
                error_message=text_error,
            )
        detected_extension = file_extension
        detected_mime_type = client_mime_type or "text/plain"
    else:
        detected_extension, detected_mime_type = sniff_binary_file_type(first_bytes)
        integrity_error = validate_binary_integrity(
            path=path,
            file_extension=file_extension,
            detected_extension=detected_extension,
            first_bytes=first_bytes,
            last_bytes=last_bytes,
        )
        if integrity_error:
            mismatch = "does not match" in integrity_error
            return ValidationResult(
                ok=False,
                file_extension=file_extension,
                detected_extension=detected_extension,
                client_mime_type=client_mime_type,
                detected_mime_type=detected_mime_type,
                sha256=None,
                size_bytes=size_bytes,
                error_code="mime_header_mismatch" if mismatch else "corrupted_file",
                error_message=integrity_error,
            )
        if file_extension in OOXML_EXTENSIONS:
            detected_extension, detected_mime_type, _ = inspect_ooxml_type(path)

    if detected_mime_type is None:
        return ValidationResult(
            ok=False,
            file_extension=file_extension,
            detected_extension=detected_extension,
            client_mime_type=client_mime_type,
            detected_mime_type=None,
            sha256=None,
            size_bytes=size_bytes,
            error_code="unsupported_type",
            error_message="Unsupported or unrecognized file content",
        )

    if detected_extension not in {file_extension, ".jpg"} or (
        file_extension == ".jpeg" and detected_extension != ".jpg"
    ):
        return ValidationResult(
            ok=False,
            file_extension=file_extension,
            detected_extension=detected_extension,
            client_mime_type=client_mime_type,
            detected_mime_type=detected_mime_type,
            sha256=None,
            size_bytes=size_bytes,
            error_code="mime_header_mismatch",
            error_message="File content does not match the file extension",
        )

    expected_mimes = EXPECTED_MIME_TYPES.get(file_extension, set())
    if expected_mimes and detected_mime_type not in expected_mimes:
        return ValidationResult(
            ok=False,
            file_extension=file_extension,
            detected_extension=detected_extension,
            client_mime_type=client_mime_type,
            detected_mime_type=detected_mime_type,
            sha256=None,
            size_bytes=size_bytes,
            error_code="mime_header_mismatch",
            error_message="File content does not match the expected MIME type",
        )

    unsafe_reason = security_screen(path, file_extension)
    if unsafe_reason:
        return ValidationResult(
            ok=False,
            file_extension=file_extension,
            detected_extension=detected_extension,
            client_mime_type=client_mime_type,
            detected_mime_type=detected_mime_type,
            sha256=None,
            size_bytes=size_bytes,
            error_code="unsafe_file",
            error_message=unsafe_reason,
        )

    return ValidationResult(
        ok=True,
        file_extension=file_extension,
        detected_extension=detected_extension,
        client_mime_type=client_mime_type,
        detected_mime_type=detected_mime_type,
        sha256=sha256,
        size_bytes=size_bytes,
    )


def move_file(src: Path, dest_dir: Path) -> Path:
    dest = unique_destination(dest_dir, src.name)
    shutil.move(str(src), str(dest))
    return dest


def ensure_document_record_defaults(record: Dict[str, Any]) -> Dict[str, Any]:
    record.setdefault("status", "stored")
    record.setdefault("deleted_at", None)
    record.setdefault("job_id", None)
    return record


def enqueue_processing_job(
    *,
    manifest: Dict[str, Any],
    document_record: Dict[str, Any],
    queue_job: Optional[QueueCallback] = None,
) -> Dict[str, Any]:
    payload = {
        "job_id": f"job_{document_record['file_id']}",
        "file_id": document_record["file_id"],
        "sha256": document_record["sha256"],
        "storage_path": document_record["ready_path"],
        "mime_type": document_record["mime_type"],
        "status": "queued",
        "created_at": utc_now_iso(),
    }

    if queue_job is None:
        manifest["jobs"].append(payload)
        return payload

    queued_job = queue_job(payload)
    queued_job = dict(queued_job)
    queued_job.setdefault("job_id", payload["job_id"])
    queued_job.setdefault("status", "queued")
    queued_job.setdefault("created_at", payload["created_at"])
    manifest["jobs"].append(queued_job)
    return queued_job


def process_one_file(
    *,
    root: Path,
    file_path: Path,
    manifest: Dict[str, Any],
    ready_dir: Path,
    quarantine_dir: Path,
    duplicates_dir: Path,
    config: Optional[IngestionConfig] = None,
    queue_job: Optional[QueueCallback] = None,
) -> FileResult:
    config = config or IngestionConfig.from_env()
    validation = validate_and_hash(file_path, config=config)

    if not validation.ok:
        dest = move_file(file_path, quarantine_dir)
        manifest["quarantine"].append(
            {
                "file_name": dest.name,
                "source_path": relpath_str(root, file_path),
                "dest_path": relpath_str(root, dest),
                "reason": validation.error_message,
                "error_code": validation.error_code,
                "timestamp": utc_now_iso(),
            }
        )
        return FileResult(
            status="quarantine",
            reason=validation.error_message or "File validation failed",
            size_bytes=validation.size_bytes,
            source_path=relpath_str(root, file_path),
            dest_path=relpath_str(root, dest),
        )

    assert validation.sha256 is not None
    assert validation.size_bytes is not None
    assert validation.detected_mime_type is not None

    if validation.sha256 in manifest["documents"]:
        existing = ensure_document_record_defaults(manifest["documents"][validation.sha256])
        existing["last_seen"] = utc_now_iso()
        dest = move_file(file_path, duplicates_dir)
        manifest["duplicates"].append(
            {
                "file_name": dest.name,
                "source_path": relpath_str(root, file_path),
                "dest_path": relpath_str(root, dest),
                "hash": validation.sha256,
                "reason": "Duplicate or unchanged file",
                "timestamp": utc_now_iso(),
            }
        )
        return FileResult(
            status="duplicate",
            reason="Duplicate or unchanged file",
            file_id=existing.get("file_id"),
            sha256=validation.sha256,
            size_bytes=validation.size_bytes,
            source_path=relpath_str(root, file_path),
            dest_path=relpath_str(root, dest),
            job_id=existing.get("job_id"),
        )

    dest = move_file(file_path, ready_dir)
    first_seen = utc_now()
    expires_at = first_seen + config.retention_delta
    document_record = {
        "file_id": validation.sha256,
        "sha256": validation.sha256,
        "original_name": file_path.name,
        "ready_path": relpath_str(root, dest),
        "size_bytes": validation.size_bytes,
        "mime_type": validation.detected_mime_type,
        "extension": validation.file_extension,
        "first_seen": first_seen.isoformat(),
        "last_seen": first_seen.isoformat(),
        "expires_at": expires_at.isoformat(),
        "status": "stored",
        "deleted_at": None,
        "job_id": None,
    }
    manifest["documents"][validation.sha256] = document_record

    try:
        job = enqueue_processing_job(
            manifest=manifest,
            document_record=document_record,
            queue_job=queue_job,
        )
        document_record["status"] = "queued"
        document_record["job_id"] = job["job_id"]
        return FileResult(
            status="ready",
            reason="Valid file stored and queued for processing",
            file_id=document_record["file_id"],
            sha256=validation.sha256,
            size_bytes=validation.size_bytes,
            source_path=relpath_str(root, file_path),
            dest_path=relpath_str(root, dest),
            job_id=job["job_id"],
        )
    except Exception:
        document_record["status"] = "queue_failed"
        manifest["job_failures"].append(
            {
                "file_id": document_record["file_id"],
                "sha256": document_record["sha256"],
                "storage_path": document_record["ready_path"],
                "timestamp": utc_now_iso(),
                "reason": "Processing queue unavailable",
            }
        )
        return FileResult(
            status="queue_failed",
            reason="File stored but processing queue submission failed",
            file_id=document_record["file_id"],
            sha256=validation.sha256,
            size_bytes=validation.size_bytes,
            source_path=relpath_str(root, file_path),
            dest_path=relpath_str(root, dest),
        )


def cleanup_expired_files(
    root: Path,
    manifest: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> int:
    now = now or utc_now()
    deleted_count = 0

    for sha256, record in list(manifest["documents"].items()):
        ensure_document_record_defaults(record)
        if record.get("deleted_at"):
            continue
        expires_at_raw = record.get("expires_at")
        if not expires_at_raw:
            continue
        if parse_utc_datetime(expires_at_raw) > now:
            continue

        storage_path = root / record["ready_path"]
        try:
            if storage_path.exists():
                storage_path.unlink()
        except OSError:
            continue

        deleted_at = now.isoformat()
        record["status"] = "deleted"
        record["deleted_at"] = deleted_at
        manifest["deleted"].append(
            {
                "file_id": record["file_id"],
                "sha256": sha256,
                "storage_path": record["ready_path"],
                "deleted_at": deleted_at,
            }
        )
        deleted_count += 1

    return deleted_count


def run_cleanup_once(root: Path, *, now: Optional[datetime] = None) -> int:
    """
    Convenience wrapper to load + cleanup + save.
    Intended to be called by a scheduler/cron or a background loop.
    """
    manifest_path = root / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    deleted = cleanup_expired_files(root, manifest, now=now)
    if deleted:
        save_manifest(manifest_path, manifest)
    return deleted


def start_cleanup_scheduler(
    *,
    root: Path,
    interval_seconds: int,
    stop_event: threading.Event,
) -> threading.Thread:
    """
    Minimal background scheduler using only the standard library.
    It periodically runs `run_cleanup_once()` until `stop_event` is set.
    """

    def _loop() -> None:
        while not stop_event.is_set():
            try:
                run_cleanup_once(root)
            except Exception:
                # Do not crash the host process because of a cleanup issue.
                pass
            stop_event.wait(interval_seconds)

    t = threading.Thread(target=_loop, name="ingest-cleanup-scheduler", daemon=True)
    t.start()
    return t


def iter_incoming_files(incoming_dir: Path) -> Iterable[Path]:
    for file_path in sorted(incoming_dir.iterdir()):
        if file_path.is_file():
            yield file_path


def run_step1(
    root: Path,
    *,
    config: Optional[IngestionConfig] = None,
    queue_job: Optional[QueueCallback] = None,
) -> Dict[str, int]:
    config = config or IngestionConfig.from_env()
    incoming_dir = root / "incoming"
    ready_dir = root / "ready"
    quarantine_dir = root / "quarantine"
    duplicates_dir = root / "duplicates"
    manifest_path = root / MANIFEST_NAME

    incoming_dir.mkdir(parents=True, exist_ok=True)
    ready_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    duplicates_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    deleted_count = cleanup_expired_files(root, manifest)
    counts = {
        "ready": 0,
        "duplicate": 0,
        "quarantine": 0,
        "queue_failed": 0,
        "errors": 0,
        "scanned": 0,
        "deleted": deleted_count,
    }

    for file_path in iter_incoming_files(incoming_dir):
        counts["scanned"] += 1
        try:
            result = process_one_file(
                root=root,
                file_path=file_path,
                manifest=manifest,
                ready_dir=ready_dir,
                quarantine_dir=quarantine_dir,
                duplicates_dir=duplicates_dir,
                config=config,
                queue_job=queue_job,
            )
            counts[result.status] += 1
        except Exception:
            counts["errors"] += 1
            try:
                dest = move_file(file_path, quarantine_dir)
                manifest["quarantine"].append(
                    {
                        "file_name": dest.name,
                        "source_path": relpath_str(root, file_path),
                        "dest_path": relpath_str(root, dest),
                        "reason": "Unexpected ingestion error",
                        "error_code": "internal_error",
                        "timestamp": utc_now_iso(),
                    }
                )
            except Exception:
                manifest["quarantine"].append(
                    {
                        "file_name": file_path.name,
                        "source_path": relpath_str(root, file_path),
                        "dest_path": None,
                        "reason": "Unexpected ingestion error and quarantine move failed",
                        "error_code": "internal_error",
                        "timestamp": utc_now_iso(),
                    }
                )

    save_manifest(manifest_path, manifest)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="STEP 1: Document ingestion, validation, queueing, and cleanup")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Folder containing incoming/, ready/, quarantine/, duplicates/, and manifest.json",
    )
    parser.add_argument(
        "--max-size-bytes",
        type=int,
        default=None,
        help="Maximum allowed file size in bytes. Defaults to INGEST_MAX_FILE_SIZE_BYTES or 10MB.",
    )
    parser.add_argument(
        "--retention-hours",
        type=int,
        default=None,
        help="Retention period before stored files expire. Defaults to INGEST_RETENTION_HOURS or 168 hours.",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run expired-file cleanup once (no ingestion), then exit.",
    )
    parser.add_argument(
        "--cleanup-loop",
        action="store_true",
        help="Run expired-file cleanup periodically in the foreground until interrupted.",
    )
    parser.add_argument(
        "--cleanup-interval-seconds",
        type=int,
        default=None,
        help="Cleanup loop interval. Defaults to INGEST_CLEANUP_INTERVAL_SECONDS or 3600 seconds.",
    )
    args = parser.parse_args()

    env_config = IngestionConfig.from_env()
    config = IngestionConfig(
        max_file_size_bytes=args.max_size_bytes or env_config.max_file_size_bytes,
        retention_hours=args.retention_hours or env_config.retention_hours,
    )

    root = Path(args.root).resolve()
    if args.cleanup_only:
        deleted = run_cleanup_once(root)
        print(json.dumps({"deleted": deleted}, indent=2))
        return

    if args.cleanup_loop:
        interval = args.cleanup_interval_seconds or int(
            os.getenv("INGEST_CLEANUP_INTERVAL_SECONDS", DEFAULT_CLEANUP_INTERVAL_SECONDS)
        )
        interval = max(1, interval)
        stop = threading.Event()
        start_cleanup_scheduler(root=root, interval_seconds=interval, stop_event=stop)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            stop.set()
        return

    counts = run_step1(root, config=config)
    print(json.dumps(counts, indent=2))
    print(f"Manifest updated: {root / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
