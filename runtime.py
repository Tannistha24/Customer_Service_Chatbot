"""
Unified customer-service runtime with automatic dataset integration.

CRITICAL FIX:
  - Intent clarification is NOW COMPLETELY BYPASSED for general questions
  - Unknown or low-confidence intents go DIRECTLY to RAG retrieval
  - Only genuinely structured intent-based actions ask for clarification
  - This allows knowledge-base questions to be answered properly

Dataset integration:
    dataset/dataset.csv (user-maintained)
        ->
    runtime_data/kb_metadata.json (change tracking)
        ->
    existing Task 1 ingestion/KB pipeline
        ->
    RAG retrieval and LLM generation
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Existing task directories
# ---------------------------------------------------------------------------

for _d in [
    BASE_DIR / "task 1",
    BASE_DIR / "task 3",
    BASE_DIR / "task 5",
    BASE_DIR / "task 6",
    BASE_DIR / "task2",
    BASE_DIR / "task4",
]:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))


# ---------------------------------------------------------------------------
# Dynamic module loader
# ---------------------------------------------------------------------------

def _load_file_module(name: str, path: Path):
    """Load an existing task module from a file path."""
    if not path.exists():
        raise ImportError(f"Required module file not found: {path}")

    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------------
# Existing components
# ---------------------------------------------------------------------------

security = _load_file_module(
    "task1_security_unified",
    BASE_DIR / "task 1" / "security.py",
)

session_mod = _load_file_module(
    "task6_step1_unified",
    BASE_DIR / "task 6" / "step1.py",
)

nlu_mod = _load_file_module(
    "task6_step2_unified",
    BASE_DIR / "task 6" / "step2.py",
)

entity_mod = _load_file_module(
    "task6_step3_unified",
    BASE_DIR / "task 6" / "step3.py",
)

intent_mod = _load_file_module(
    "task6_step4_unified",
    BASE_DIR / "task 6" / "step4.py",
)

ticket_mod = _load_file_module(
    "task3_ticket_unified",
    BASE_DIR / "task 3" / "ticket.py",
)

priority_mod = _load_file_module(
    "task3_priority_unified",
    BASE_DIR / "task 3" / "priority.py",
)

detection_mod = _load_file_module(
    "task5_detection_unified",
    BASE_DIR / "task 5" / "detection.py",
)

routing_mod = _load_file_module(
    "task5_routing_unified",
    BASE_DIR / "task 5" / "step.py",
)


# ---------------------------------------------------------------------------
# Dataset metadata and change detection
# ---------------------------------------------------------------------------

class DatasetMetadata:
    """Track dataset state to avoid redundant ingestion."""

    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        """Load existing metadata or create new."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return {}
        return {}

    def _save(self) -> None:
        """Persist metadata to disk."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def get_dataset_hash(self) -> Optional[str]:
        """Retrieve stored hash of the dataset."""
        return self.data.get("dataset_hash")

    def set_dataset_hash(self, hash_value: str) -> None:
        """Store hash of the dataset."""
        self.data["dataset_hash"] = hash_value
        self._save()

    def get_kb_version(self) -> Optional[str]:
        """Retrieve the knowledge base version created from dataset."""
        return self.data.get("kb_version")

    def set_kb_version(self, version: str) -> None:
        """Store the knowledge base version."""
        self.data["kb_version"] = version
        self._save()

    def is_dataset_changed(self, current_hash: str) -> bool:
        """Check if dataset has changed since last ingestion."""
        stored_hash = self.get_dataset_hash()
        return stored_hash != current_hash


# ---------------------------------------------------------------------------
# Dataset operations
# ---------------------------------------------------------------------------

class DatasetManager:
    """Manage dataset loading, hashing, and change detection."""

    def __init__(self, dataset_path: Path, metadata: DatasetMetadata):
        self.dataset_path = dataset_path
        self.metadata = metadata

    def read_csv(self) -> list[dict[str, str]]:
        """
        Read dataset CSV with 'prompt' and 'response' columns.

        Returns:
            List of dicts with 'prompt' and 'response' keys.

        Raises:
            FileNotFoundError: If dataset does not exist.
            ValueError: If CSV is malformed.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        try:
            try:
                text = self.dataset_path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                text = self.dataset_path.read_text(encoding="cp1252")

            reader = csv.DictReader(io.StringIO(text, newline=""))

            if reader.fieldnames is None:
                raise ValueError("CSV has no headers")

            required = {"prompt", "response"}
            if not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"CSV must have 'prompt' and 'response' columns. "
                    f"Found: {reader.fieldnames}"
                )

            rows = []
            for row_num, row in enumerate(reader, start=2):
                prompt = (row.get("prompt") or "").strip()
                response = (row.get("response") or "").strip()

                if not prompt:
                    logger.warning(f"Row {row_num}: empty prompt, skipping")
                    continue
                if not response:
                    logger.warning(f"Row {row_num}: empty response, skipping")
                    continue

                rows.append({"prompt": prompt, "response": response})

            if not rows:
                raise ValueError(
                    "Dataset contains no valid prompt-response pairs"
                )

            logger.info(
                f"Loaded {len(rows)} prompt-response pairs from dataset"
            )
            return rows

        except (IOError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to read CSV: {e}")

    def compute_hash(self) -> str:
        """
        Compute SHA256 hash of the dataset file.

        Returns:
            Hex digest of the dataset.
        """
        if not self.dataset_path.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(self.dataset_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def has_changed(self) -> bool:
        """Check if dataset has changed since last ingestion."""
        current_hash = self.compute_hash()
        if not current_hash:
            return False
        return self.metadata.is_dataset_changed(current_hash)

    def mark_ingested(self) -> None:
        """Record that dataset has been successfully ingested."""
        current_hash = self.compute_hash()
        if current_hash:
            self.metadata.set_dataset_hash(current_hash)


# ---------------------------------------------------------------------------
# RAG ingestion controller
# ---------------------------------------------------------------------------

class RAGIngestionController:
    """
    Orchestrate dataset ingestion into the Task 1 RAG pipeline.

    This controller bridges the dataset manager with the existing Task 1
    ingestion machinery without rewriting it.
    """

    def __init__(self, rag_runtime: Any, dataset_manager: DatasetManager):
        self.rag = rag_runtime
        self.dataset_manager = dataset_manager

    def ingest_dataset(self) -> Optional[str]:
        """
        Ingest the dataset into the RAG pipeline.

        Returns:
            Version ID of the created knowledge base, or None if failed.
        """
        try:
            # 1. Read and validate dataset
            logger.info("Reading dataset...")
            rows = self.dataset_manager.read_csv()

            # 2. Convert to documents format expected by Task 1
            documents = self._rows_to_documents(rows)
            logger.info(
                f"Converted {len(documents)} documents for ingestion"
            )

            # 3. Trigger Task 1 ingestion
            logger.info("Starting RAG ingestion...")
            version_id = self._ingest_documents(documents)
            if not version_id:
                version_id = self._ingest_with_kb_manager(documents)

            if not version_id:
                logger.error("Ingestion returned no version ID")
                return None

            logger.info(
                f"Ingestion complete. Knowledge base version: {version_id}"
            )

            # 4. Activate the new version
            logger.info(
                f"Activating knowledge base version {version_id}..."
            )
            if not self._activate_version(version_id):
                logger.warning(
                    f"Failed to activate version {version_id}, "
                    "but ingestion was successful"
                )

            # 5. Mark dataset as ingested
            self.dataset_manager.mark_ingested()
            self.dataset_manager.metadata.set_kb_version(version_id)

            logger.info("Dataset ingestion and activation complete")
            return version_id

        except Exception as e:
            logger.error(
                f"Dataset ingestion failed: {e}",
                exc_info=True
            )
            return None

    def _ingest_with_kb_manager(
        self,
        documents: list[dict[str, str]],
    ) -> Optional[str]:
        """Save documents as a new KB version using Task 1's version manager."""
        try:
            import tempfile

            vm = self.rag.version_manager
            with tempfile.TemporaryDirectory() as tmp:
                with open(Path(tmp) / "documents.json", "w", encoding="utf-8") as f:
                    json.dump(documents, f, ensure_ascii=False, indent=2)
                with open(
                    Path(tmp) / "dataset_documents.jsonl", "w", encoding="utf-8"
                ) as f:
                    for doc in documents:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                version = vm.create_version(
                    tmp, notes="Company dataset ingestion"
                )
            vm.set_active_version(version)
            logger.info(f"Created and activated KB version: {version}")
            return version
        except Exception as e:
            logger.error(f"KB manager ingestion failed: {e}", exc_info=True)
            return None

    def _rows_to_documents(
        self,
        rows: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Convert CSV rows into document format for Task 1 ingestion.

        Each row becomes a document with combined prompt+response content.
        Metadata includes individual prompt and response for retrieval.
        """
        documents = []
        for i, row in enumerate(rows, start=1):
            doc = {
                "id": f"doc_{i}",
                "content": f"{row['prompt']}\n\n{row['response']}",
                "metadata": {
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "source": "dataset.csv",
                    "row_index": i,
                },
            }
            documents.append(doc)
        return documents

    def _ingest_documents(self, documents: list[dict[str, str]]) -> Optional[str]:
        """
        Call the existing Task 1 ingestion pipeline.

        This method adapts the documents to whatever interface Task 1 expects.
        Supports multiple ingestion patterns:
          1. Direct method: rag.ingest_documents(documents)
          2. Manager pattern: rag.ingestion_manager.ingest(documents)
          3. File-based: write to incoming/ and trigger ingestion
        """
        try:
            # Pattern 1: Direct ingestion method
            if hasattr(self.rag, "ingest_documents"):
                logger.info("Using direct ingestion method...")
                version = self.rag.ingest_documents(documents)
                if version:
                    return version

            # Pattern 2: Ingestion manager pattern
            if hasattr(self.rag, "ingestion_manager"):
                logger.info("Using ingestion manager pattern...")
                version = self.rag.ingestion_manager.ingest(documents)
                if version:
                    return version

            # Pattern 3: File-based ingestion (most common)
            if hasattr(self.rag, "data_root"):
                logger.info("Using file-based ingestion...")
                incoming_dir = self.rag.data_root / "incoming"
                incoming_dir.mkdir(parents=True, exist_ok=True)

                # Write documents as JSONL
                doc_file = incoming_dir / "dataset_documents.jsonl"
                logger.info(f"Writing documents to {doc_file}...")
                with open(doc_file, "w") as f:
                    for doc in documents:
                        f.write(json.dumps(doc) + "\n")

                # Attempt to trigger ingestion
                if hasattr(self.rag, "run_ingestion"):
                    logger.info("Triggering ingestion pipeline...")
                    version = self.rag.run_ingestion()
                    if version:
                        return version

                if hasattr(self.rag, "ingest"):
                    logger.info("Triggering ingest method...")
                    version = self.rag.ingest()
                    if version:
                        return version

                # Check for version manager (ingestion might be async)
                if hasattr(self.rag, "version_manager"):
                    active = self.rag.version_manager.get_active_version()
                    if active:
                        return active

                logger.warning(
                    "Documents written to incoming/ but no automatic "
                    "ingestion trigger found. You may need to call "
                    "RAGRuntime.run_ingestion() or similar."
                )
                return None

            logger.error(
                "Task 1 RAG runtime has no recognized ingestion method. "
                "Please check Task 1 implementation."
            )
            return None

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}", exc_info=True)
            return None

    def _activate_version(self, version_id: str) -> bool:
        """Activate a knowledge base version in Task 1."""
        try:
            if hasattr(self.rag, "activate_version"):
                logger.info("Using activate_version method...")
                self.rag.activate_version(version_id)
                return True

            if hasattr(self.rag, "version_manager"):
                logger.info("Using version_manager pattern...")
                self.rag.version_manager.set_active_version(version_id)
                return True

            logger.warning(
                "Could not find version activation method in Task 1 RAG runtime"
            )
            return False

        except Exception as e:
            logger.error(f"Version activation failed: {e}", exc_info=True)
            return False


# ---------------------------------------------------------------------------
# Main runtime
# ---------------------------------------------------------------------------

class CustomerServiceRuntime:
    """Single application-facing customer-service pipeline."""

    def __init__(self, data_root: Optional[str | Path] = None):

        # Main runtime data directory
        self.data_root = Path(
            data_root or (BASE_DIR / "runtime_data")
        ).resolve()

        self.data_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Runtime data directory: {self.data_root}")

        # -------------------------------------------------------------------
        # Dataset and metadata
        # -------------------------------------------------------------------

        self.dataset_path = BASE_DIR / "dataset" / "dataset.csv"
        self.metadata_path = self.data_root / "kb_metadata.json"
        self.metadata = DatasetMetadata(self.metadata_path)
        self.dataset_manager = DatasetManager(
            self.dataset_path,
            self.metadata,
        )

        # -------------------------------------------------------------------
        # Session management
        # -------------------------------------------------------------------

        self.sessions = session_mod.SessionManager()

        # -------------------------------------------------------------------
        # Persistent entity state
        # -------------------------------------------------------------------

        self._entity_states: dict[str, Any] = {}

        # -------------------------------------------------------------------
        # RAG runtime and ingestion
        # -------------------------------------------------------------------

        self._rag_runtime = None
        self._ingestion_controller = None
        self._initialize_rag()

    # -----------------------------------------------------------------------
    # RAG initialization and ingestion
    # -----------------------------------------------------------------------

    def _initialize_rag(self) -> None:
        """
        Initialize the RAG runtime and handle dataset ingestion.

        This runs once on application startup and detects dataset changes.
        """
        try:
            logger.info("=" * 70)
            logger.info("Initializing RAG runtime and dataset integration...")
            logger.info("=" * 70)

            self._rag_runtime = self._get_rag()

            if not self._rag_runtime:
                logger.warning(
                    "RAG runtime initialization failed. "
                    "Knowledge base will not be available."
                )
                return

            self._ingestion_controller = RAGIngestionController(
                self._rag_runtime,
                self.dataset_manager,
            )

            # Check if dataset exists
            if not self.dataset_path.exists():
                logger.warning(
                    f"Dataset not found at {self.dataset_path}\n"
                    "RAG will not have knowledge base data.\n"
                    "Please create dataset/dataset.csv with columns: "
                    "'prompt' and 'response'"
                )
                return

            logger.info(f"Dataset found at: {self.dataset_path}")

            # Check if we need to ingest
            has_active_kb = (
                self._rag_runtime.version_manager.get_active_version()
                is not None
            )
            dataset_changed = self.dataset_manager.has_changed()

            logger.info(f"Active knowledge base: {has_active_kb}")
            logger.info(f"Dataset changed: {dataset_changed}")

            if not has_active_kb:
                logger.info(
                    "No active knowledge base found. "
                    "Triggering ingestion..."
                )
                self._ingestion_controller.ingest_dataset()

            elif dataset_changed:
                logger.info(
                    "Dataset has changed. "
                    "Triggering re-ingestion..."
                )
                self._ingestion_controller.ingest_dataset()

            else:
                logger.info(
                    "Dataset unchanged and KB is active. "
                    "Skipping ingestion."
                )

            logger.info("=" * 70)
            logger.info("RAG initialization complete")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(
                f"RAG initialization failed: {e}",
                exc_info=True
            )

    # -----------------------------------------------------------------------
    # Session helpers
    # -----------------------------------------------------------------------

    def _session(self, session_id: str):
        """Return the existing session."""
        return self.sessions.get_session(session_id)

    # -----------------------------------------------------------------------
    # Entity helpers
    # -----------------------------------------------------------------------

    def _entity_state(self, session_id: str):
        """Return the persistent entity state for a session."""
        if session_id not in self._entity_states:
            self._entity_states[session_id] = (
                entity_mod.DialogueStateManager()
            )
        return self._entity_states[session_id]

    # -----------------------------------------------------------------------
    # Serialization helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _to_dict(value: Any) -> Any:
        """Convert task results into JSON-friendly structures."""
        if value is None:
            return None

        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except TypeError:
                pass

        if is_dataclass(value):
            return asdict(value)

        if isinstance(value, dict):
            return value

        return getattr(value, "__dict__", str(value))

    # -----------------------------------------------------------------------
    # RAG
    # -----------------------------------------------------------------------

    def _get_rag(self):
        """Lazily load the existing Task 1 RAG runtime."""
        if self._rag_runtime is not None:
            return self._rag_runtime

        try:
            logger.info("Loading Task 1 RAG runtime...")
            runtime_mod = _load_file_module(
                "task1_runtime_unified",
                BASE_DIR / "task 1" / "runtime.py",
            )

            self._rag_runtime = runtime_mod.RAGRuntime(
                self.data_root
            )
            logger.info("Task 1 RAG runtime loaded successfully")

        except Exception as e:
            logger.error(
                f"Failed to load RAG runtime: {e}",
                exc_info=True
            )
            self._rag_runtime = False

        return self._rag_runtime if self._rag_runtime else None

    # -----------------------------------------------------------------------
    # RAG query
    # -----------------------------------------------------------------------

    def _retrieve_answer(
        self,
        message: str,
    ) -> Optional[dict[str, Any]]:
        """
        Use the existing Task 1 RAG pipeline when an active KB exists.

        This queries the ingested dataset for relevant answers.
        """
        rag = self._rag_runtime

        if not rag:
            logger.debug("RAG runtime not available")
            return None

        try:
            # RAG requires an active knowledge-base version
            if rag.version_manager.get_active_version() is None:
                logger.debug("No active knowledge base version")
                return None

            logger.info(f"Querying RAG with: {message}")
            result = rag.answer_query(message)

            if (
                result.get("allowed")
                and result.get("llm_response")
            ):
                logger.info(f"RAG returned answer")
                return result

            logger.debug(f"RAG did not return a valid answer: {result}")

        except Exception as e:
            logger.error(f"RAG query failed: {e}", exc_info=True)
            return None

        return None

    # -----------------------------------------------------------------------
    # CRITICAL FIX: Intent clarification bypass for general questions
    # -----------------------------------------------------------------------

    def _should_ask_clarification(self, intent_obj: Any) -> bool:
        """
        CRITICAL LOGIC FIX:
        
        Clarification is ONLY for structured intent-based actions
        (e.g., "cancel my order", "track my package").
        
        General questions (even with unknown/low-confidence intents)
        should go DIRECTLY to RAG, not ask for clarification.
        
        This prevents blocking legitimate knowledge-base questions.
        """
        
        # Get the intent classification
        intent_name = getattr(intent_obj, "intent", None)
        confidence = getattr(intent_obj, "confidence", 0.0)
        clarification_required = getattr(
            intent_obj,
            "clarification_required",
            False
        )

        logger.info(
            f"Intent analysis: "
            f"intent={intent_name}, "
            f"confidence={confidence:.2f}, "
            f"clarification_required={clarification_required}"
        )

        # ===================================================================
        # DECISION LOGIC
        # ===================================================================

        # If intent is recognized and has decent confidence,
        # DO NOT ask for clarification (proceed to RAG or action)
        if intent_name and intent_name.lower() != "unknown":
            if confidence >= 0.7:
                logger.info(
                    f"Intent '{intent_name}' recognized with "
                    f"confidence {confidence:.2f} - proceeding to RAG"
                )
                return False

        # If intent is unknown/ambiguous,
        # we should TRY RAG instead of asking for clarification
        # RAG can often answer questions even if intent is unclear
        if not intent_name or intent_name.lower() == "unknown":
            logger.info(
                "Intent is unknown - will attempt RAG retrieval "
                "instead of asking for clarification"
            )
            return False

        # If we reach here, ONLY ask clarification if:
        # - It's explicitly marked as requiring clarification
        # - AND we have no confidence at all
        # This is a true ambiguity that RAG cannot help with
        if clarification_required and confidence < 0.1:
            logger.info(
                "True ambiguity detected - asking for clarification"
            )
            return True

        # Default: proceed to RAG, do NOT ask for clarification
        logger.info("Defaulting to RAG retrieval (no clarification needed)")
        return False

    # -----------------------------------------------------------------------
    # Main application pipeline
    # -----------------------------------------------------------------------

    def process_message(
        self,
        message: str,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Run one customer message through the unified pipeline.

        CRITICAL CHANGE:
          Step 10 (Clarification) now BYPASSES for general questions
          and lets them proceed to RAG retrieval.

        Pipeline:
            1. Security check
            2. Session management
            3. Multilingual NLU
            4. Entity extraction
            5. Intent classification
            6. Detection (sentiment/risk)
            7. Ticket processing
            8. Priority scoring
            9. Routing/escalation
            10. Clarification (ONLY for true ambiguity)
            11. RAG retrieval (for all other questions)
            12. Response generation
        """

        message = (message or "").strip()

        if not message:
            return {
                "ok": False,
                "session_id": session_id,
                "response": (
                    "Please enter a message so I can help you."
                ),
            }

        # -------------------------------------------------------------------
        # 1. Security
        # -------------------------------------------------------------------

        sec = security.security_check(message)

        if not sec.get("allowed", False):
            return {
                "ok": False,
                "session_id": session_id,
                "response": sec.get(
                    "reason",
                    "I can't process that request.",
                ),
                "security": sec,
            }

        sanitized = sec.get("sanitized_input", message)

        # -------------------------------------------------------------------
        # 2. Session + conversation history
        # -------------------------------------------------------------------

        self._session(session_id)

        self.sessions.add_message(
            session_id,
            "user",
            sanitized,
        )

        # -------------------------------------------------------------------
        # 3. Multilingual NLU
        # -------------------------------------------------------------------

        try:
            nlu = nlu_mod.analyze_message(sanitized)
            nlu_data = self._to_dict(nlu)
        except Exception as exc:
            nlu_data = {"error": str(exc)}

        # -------------------------------------------------------------------
        # 4. Persistent entities / corrections
        # -------------------------------------------------------------------

        entity_state = self._entity_state(session_id)
        entity_state.process_turn(sanitized)

        # -------------------------------------------------------------------
        # 5. Intent / confidence
        # -------------------------------------------------------------------

        intent = intent_mod.classify_message(sanitized)
        intent_data = self._to_dict(intent)

        # -------------------------------------------------------------------
        # 6. Task 5 detection
        # -------------------------------------------------------------------

        history = [
            getattr(m, "text", "")
            for m in self.sessions.get_history(session_id)
        ]

        try:
            detection = detection_mod.analyze_message(
                sanitized,
                history=history,
            )
            detection_data = self._to_dict(detection)
        except Exception as exc:
            detection_data = {"error": str(exc)}

        # -------------------------------------------------------------------
        # 7. Ticket processing
        # -------------------------------------------------------------------

        try:
            ticket_result = ticket_mod.process_ticket(sanitized)
            ticket_data = self._to_dict(ticket_result)
        except Exception as exc:
            ticket_data = {"error": str(exc)}

        # -------------------------------------------------------------------
        # 8. Priority
        # -------------------------------------------------------------------

        try:
            priority = priority_mod.score_ticket(
                {
                    "text": sanitized,
                    "message": sanitized,
                }
            )
            priority_data = self._to_dict(priority)
        except Exception as exc:
            priority_data = {"error": str(exc)}

        # -------------------------------------------------------------------
        # 9. Escalation / routing
        # -------------------------------------------------------------------

        routing_input = {
            "escalation_required": bool(
                detection_data.get("high_risk", False)
                if isinstance(detection_data, dict)
                else False
            ),
            "triggered_conditions": [],
        }

        try:
            from datetime import datetime, timezone

            routing = routing_mod.route_conversation(
                routing_input,
                datetime.now(timezone.utc),
            )
            routing_data = self._to_dict(routing)
        except Exception as exc:
            routing_data = {"error": str(exc)}

        # -------------------------------------------------------------------
        # 10. CRITICAL FIX: Clarification only for true ambiguity
        # -------------------------------------------------------------------

        if self._should_ask_clarification(intent):
            clarification = getattr(
                intent,
                "clarification_message",
                None,
            )
            response = (
                clarification
                or "Could you clarify what you'd like me to help with?"
            )

            self.sessions.add_message(
                session_id,
                "assistant",
                response,
            )

            logger.info("Returning clarification response")

            return {
                "ok": True,
                "session_id": session_id,
                "response": response,
                "intent": intent_data,
                "entities": entity_state.snapshot(),
                "nlu": nlu_data,
                "detection": detection_data,
                "routing": routing_data,
                "used_rag": False,
            }

        # -------------------------------------------------------------------
        # 11. RAG / Knowledge Base (normal flow for ALL questions)
        # -------------------------------------------------------------------

        logger.info("Attempting RAG retrieval...")
        rag_result = self._retrieve_answer(sanitized)

        if rag_result:
            response = (
                rag_result.get("llm_response")
                or "I couldn't find an answer."
            )
            used_rag = True
            logger.info("RAG retrieval successful")
        else:
            response = (
                "I've received your request, but I don't have information "
                "about that in my knowledge base. Please try rephrasing your "
                "question or ask to speak with a support representative."
            )
            used_rag = False
            logger.info("RAG retrieval failed - returning fallback response")

        # -------------------------------------------------------------------
        # 12. Save assistant response
        # -------------------------------------------------------------------

        self.sessions.add_message(
            session_id,
            "assistant",
            response,
        )

        # -------------------------------------------------------------------
        # 13. Unified result
        # -------------------------------------------------------------------

        return {
            "ok": True,
            "session_id": session_id,
            "response": response,
            "intent": intent_data,
            "entities": entity_state.snapshot(),
            "nlu": nlu_data,
            "detection": detection_data,
            "ticket": ticket_data,
            "priority": priority_data,
            "routing": routing_data,
            "rag": rag_result,
            "used_rag": used_rag,
        }


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_runtime(
    data_root: Optional[str | Path] = None,
) -> CustomerServiceRuntime:
    """
    Create the unified customer-service runtime.

    This is the function imported by app.py.

    Args:
        data_root: Optional override for runtime data directory.
                   Defaults to ./runtime_data

    Returns:
        Initialized CustomerServiceRuntime instance.
    """
    return CustomerServiceRuntime(data_root=data_root)