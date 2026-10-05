from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol


@dataclass
class OCRTextResult:
    """Result from OCR text extraction."""
    text: str
    confidence: float
    page_count: int = 1

    def __post_init__(self) -> None:
        """Validate and clamp OCR result values."""
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.page_count = max(1, int(self.page_count))


class OCRBackend(Protocol):
    """Protocol for OCR extraction backends."""
    def extract_text(self, file_path: Path) -> OCRTextResult:
        """Extract text from image or document file."""
        ...


@dataclass
class ItemDetail:
    """Extracted product/item from invoice or document."""
    name: str
    quantity: Optional[float] = None
    unit_price: Optional[str] = None
    line_total: Optional[str] = None


@dataclass
class ExtractedFields:
    """All structured fields extracted from document."""
    order_ids: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    amounts: list[str] = field(default_factory=list)
    item_details: list[ItemDetail] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)


@dataclass
class CustomerMessageComparison:
    """Comparison between customer-provided data and extracted document fields."""
    provided_order_id: Optional[str] = None
    provided_date: Optional[str] = None
    provided_amount: Optional[str] = None
    provided_product: Optional[str] = None
    provided_error_code: Optional[str] = None
    conflicts: list[str] = field(default_factory=list)
    requires_clarification: bool = False
    clarification_message: str = ""


@dataclass
class AnalysisResult:
    """Complete analysis result for a document."""
    file_path: str
    file_type: str
    page_count: int
    fields: ExtractedFields
    readability_score: float
    ocr_confidence_score: float
    needs_clearer_file: bool
    rejected_reason: Optional[str] = None
    mismatches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    customer_comparison: Optional[CustomerMessageComparison] = None

    def __post_init__(self) -> None:
        """Validate result fields."""
        self.readability_score = max(0.0, min(1.0, float(self.readability_score)))
        self.ocr_confidence_score = max(0.0, min(1.0, float(self.ocr_confidence_score)))
        self.page_count = max(0, int(self.page_count))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
