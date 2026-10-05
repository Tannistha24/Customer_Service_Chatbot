from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Optional

from .config import AnalyzerConfig
from .types import (
    AnalysisResult,
    CustomerMessageComparison,
    ExtractedFields,
    ItemDetail,
    OCRBackend,
    OCRTextResult,
)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
PDF_MAGIC = b"%PDF-"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"tool\s+call", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+safety", re.IGNORECASE),
    re.compile(r"override\s+settings", re.IGNORECASE),
    re.compile(r"execute\s+command", re.IGNORECASE),
    re.compile(r"bypass\s+validation", re.IGNORECASE),
    re.compile(r"disable\s+safety", re.IGNORECASE),
]

# Item field keywords that should not be part of product name
ITEM_KEYWORDS = {"QTY", "QUANTITY", "UNIT", "PRICE", "TOTAL", "AMOUNT", "COST", "UNIT PRICE"}

# Patterns that indicate a grand/order total line (not a line-item total)
GRAND_TOTAL_PATTERN = re.compile(
    r"(?:GRAND\s+TOTAL|ORDER\s+TOTAL|TOTAL\s+DUE|TOTAL\s+AMOUNT|INVOICE\s+TOTAL|"
    r"AMOUNT\s+DUE|BALANCE\s+DUE|NET\s+TOTAL|SUBTOTAL)"
    r"\s*[:\s]*(?:USD|EUR|GBP|[$])?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
    re.IGNORECASE,
)

# Month name mapping for date normalization
_MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _read_file_header(path: Path, num_bytes: int = 8) -> bytes:
    """Read the first *num_bytes* of a file safely."""
    with path.open("rb") as fh:
        return fh.read(num_bytes)


class UnsafeOrUnsupportedFileError(ValueError):
    """Raised when file is unsafe or unsupported."""
    pass


class SidecarOCRBackend:
    """Deterministic OCR backend using sidecar JSON files.

    If `<file>.ocr.json` exists beside an image, its contents are used as OCR output:
    {"text": "...", "confidence": 0.98, "page_count": 1}
    """

    def extract_text(self, file_path: Path) -> OCRTextResult:
        """Extract text from sidecar OCR file if it exists."""
        sidecar = file_path.with_suffix(file_path.suffix + ".ocr.json")
        if not sidecar.exists():
            return OCRTextResult(text="", confidence=0.0, page_count=1)

        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return OCRTextResult(text="", confidence=0.0, page_count=1)

        text_value = payload.get("text", "")
        if not isinstance(text_value, str):
            text_value = ""

        confidence_value = payload.get("confidence", 0.0)
        try:
            confidence_value = float(confidence_value)
        except (TypeError, ValueError):
            confidence_value = 0.0

        page_count_value = payload.get("page_count", 1)
        try:
            page_count_value = int(page_count_value)
        except (TypeError, ValueError):
            page_count_value = 1

        return OCRTextResult(
            text=str(text_value),
            confidence=confidence_value,
            page_count=page_count_value,
        )


class TesseractOCRBackend:
    """Optional local OCR backend for images when `tesseract` is installed.

    Only processes image files (PNG, JPG, JPEG), not PDFs.
    """

    def __init__(self, executable: str = "tesseract", timeout_seconds: int = 15) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def extract_text(self, file_path: Path) -> OCRTextResult:
        """Extract text from image using Tesseract."""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return OCRTextResult(text="", confidence=0.0, page_count=1)

        if shutil.which(self.executable) is None:
            return OCRTextResult(text="", confidence=0.0, page_count=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_base = Path(temp_dir) / "ocr_output"
            command = [
                self.executable,
                str(file_path),
                str(output_base),
                "--psm",
                "6",
                "quiet",
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=self.timeout_seconds,
                    shell=False,
                )
            except (OSError, subprocess.SubprocessError):
                return OCRTextResult(text="", confidence=0.0, page_count=1)

            if completed.returncode != 0:
                return OCRTextResult(text="", confidence=0.0, page_count=1)

            text_path = output_base.with_suffix(".txt")
            if not text_path.exists():
                return OCRTextResult(text="", confidence=0.0, page_count=1)

            try:
                text = text_path.read_text(encoding="utf-8", errors="ignore")
                confidence = _clamp_score(_simple_text_confidence(text))
                return OCRTextResult(text=text, confidence=confidence, page_count=1)
            except OSError:
                return OCRTextResult(text="", confidence=0.0, page_count=1)


class DocumentAnalyzer:
    """Main document analyzer combining OCR, image quality, and field extraction."""

    def __init__(
        self,
        ocr_backend: Optional[OCRBackend] = None,
        config: Optional[AnalyzerConfig] = None,
    ) -> None:
        self.ocr_backend = ocr_backend or TesseractThenSidecarBackend()
        self.config = config or AnalyzerConfig()

    def analyze(
        self,
        file_path: str | Path,
        customer_message: Optional[str] = None,
    ) -> AnalysisResult:
        """Analyze a document file with optional customer message comparison."""
        path = Path(file_path)
        ext = path.suffix.lower()
        logs: list[str] = []

        try:
            self._validate_file(path)
        except UnsafeOrUnsupportedFileError as exc:
            return AnalysisResult(
                file_path=str(path),
                file_type=ext.lstrip(".") or "unknown",
                page_count=0,
                fields=ExtractedFields(),
                readability_score=0.0,
                ocr_confidence_score=0.0,
                needs_clearer_file=False,
                rejected_reason=str(exc),
                logs=[self._mask_log(f"Rejected file {path.name}: {exc}")],
            )

        page_count = 1
        ocr_result = None

        if ext == ".pdf":
            raw_text, page_count, is_scanned = self._extract_pdf_text(path)
            if is_scanned:
                # Tesseract cannot directly OCR PDFs.  Try the sidecar fallback
                # which may have pre-computed OCR text for this file.
                ocr_backend_result = self.ocr_backend.extract_text(path)
                if ocr_backend_result.text.strip():
                    raw_text = ocr_backend_result.text
                    logs.append(
                        self._mask_log("PDF identified as scanned, OCR backend applied")
                    )
                else:
                    logs.append(
                        self._mask_log(
                            "PDF identified as scanned; no OCR text available "
                            "(Tesseract cannot directly process PDFs and no sidecar found)"
                        )
                    )
            ocr_result = OCRTextResult(
                text=raw_text,
                confidence=self._estimate_text_confidence(raw_text),
                page_count=page_count,
            )
            readability = self._estimate_readability(ocr_result.text)
            blur_score = None  # blur detection not applicable to PDFs
        else:
            ocr_result = self.ocr_backend.extract_text(path)
            readability = self._estimate_readability(ocr_result.text)
            # Blur detection for images – malformed/truncated PNGs must be
            # rejected rather than silently ignored.
            try:
                blur_score = self._image_readability_score(path)
            except UnsafeOrUnsupportedFileError as exc:
                return AnalysisResult(
                    file_path=str(path),
                    file_type=ext.lstrip(".") or "unknown",
                    page_count=0,
                    fields=ExtractedFields(),
                    readability_score=0.0,
                    ocr_confidence_score=0.0,
                    needs_clearer_file=False,
                    rejected_reason=str(exc),
                    logs=[self._mask_log(f"Rejected file {path.name}: {exc}")],
                )

        sanitized_text, warnings = self._sanitize_untrusted_text(ocr_result.text)
        logs.append(
            self._mask_log(
                f"Processed {path.name} pages={ocr_result.page_count} confidence={ocr_result.confidence:.2f}"
            )
        )

        fields, mismatches = self._extract_fields(sanitized_text)

        # OCR confidence score: pure OCR confidence, not blended with other signals
        confidence = _clamp_score(ocr_result.confidence)

        # For images, incorporate blur detection into readability only
        if ext != ".pdf" and blur_score is not None:
            readability = _clamp_score((readability * 0.5) + (blur_score * 0.5))

        needs_clearer_file = (
            ext != ".pdf"
            and blur_score is not None
            and (
                blur_score < self.config.low_readability_threshold
                or confidence < self.config.low_confidence_threshold
                or readability < self.config.low_readability_threshold
            )
        )

        customer_comparison = None
        if customer_message:
            customer_comparison = self._compare_customer_message(customer_message, fields)
            if customer_comparison.requires_clarification:
                warnings.append("customer_message_requires_clarification")

        return AnalysisResult(
            file_path=str(path),
            file_type=ext.lstrip(".") or "unknown",
            page_count=page_count,
            fields=fields,
            readability_score=round(readability, 3),
            ocr_confidence_score=round(confidence, 3),
            needs_clearer_file=needs_clearer_file,
            mismatches=mismatches,
            warnings=warnings,
            logs=logs,
            customer_comparison=customer_comparison,
        )

    def _validate_file(self, path: Path) -> None:
        """Validate file exists, has correct type, and passes safety checks."""
        try:
            if not path.exists() or not path.is_file():
                raise UnsafeOrUnsupportedFileError("file_not_found")
        except OSError:
            raise UnsafeOrUnsupportedFileError("file_access_error")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsafeOrUnsupportedFileError("unsupported_file_type")

        try:
            file_size = path.stat().st_size
        except OSError:
            raise UnsafeOrUnsupportedFileError("file_access_error")

        if file_size > self.config.max_file_size_bytes:
            raise UnsafeOrUnsupportedFileError("file_too_large")

        if file_size == 0:
            raise UnsafeOrUnsupportedFileError("file_empty")

        try:
            header = _read_file_header(path, min(8, file_size))
        except OSError:
            raise UnsafeOrUnsupportedFileError("file_read_error")

        if ext == ".pdf":
            if not header.startswith(PDF_MAGIC):
                raise UnsafeOrUnsupportedFileError("unsafe_or_malformed_pdf")
        elif ext == ".png":
            if not header.startswith(PNG_MAGIC):
                raise UnsafeOrUnsupportedFileError("unsafe_or_malformed_image")
        elif ext in {".jpg", ".jpeg"}:
            if not header.startswith(JPEG_MAGIC):
                raise UnsafeOrUnsupportedFileError("unsafe_or_malformed_image")

        self._check_injection_in_filename(path.name)

    def _check_injection_in_filename(self, filename: str) -> None:
        """Detect prompt injection attempts in filename."""
        normalized = filename.lower()
        for pattern in INJECTION_PATTERNS:
            if pattern.search(normalized):
                raise UnsafeOrUnsupportedFileError("unsafe_filename_detected")

    def _sanitize_untrusted_text(self, text: str) -> tuple[str, list[str]]:
        """Remove prompt-injection patterns and log detection."""
        warnings: list[str] = []
        kept_lines: list[str] = []

        for line in text.splitlines():
            normalized_line = line.lower()
            found_injection = False
            for pattern in INJECTION_PATTERNS:
                if pattern.search(normalized_line):
                    found_injection = True
                    break

            if found_injection:
                warnings.append("prompt_injection_ignored")
            else:
                kept_lines.append(line)

        return "\n".join(kept_lines), warnings

    def _compare_customer_message(
        self, customer_message: str, fields: ExtractedFields
    ) -> CustomerMessageComparison:
        """Compare customer-provided message with extracted fields."""
        comparison = CustomerMessageComparison()

        extracted_order_id = fields.order_ids[0] if fields.order_ids else None
        extracted_date = fields.dates[0] if fields.dates else None
        extracted_amount = fields.amounts[0] if fields.amounts else None
        extracted_product = fields.item_details[0].name if fields.item_details else None
        extracted_error_code = fields.error_codes[0] if fields.error_codes else None

        # Parse customer-provided order ID
        order_match = re.search(
            r"order\s*(?:id|no|number)?[:\s#-]*([A-Z0-9-]{4,})",
            customer_message,
            re.IGNORECASE,
        )
        if order_match:
            comparison.provided_order_id = order_match.group(1).strip()
            if extracted_order_id:
                if not _normalize_id(comparison.provided_order_id) == _normalize_id(extracted_order_id):
                    comparison.conflicts.append(
                        f"Order ID mismatch: customer provided '{comparison.provided_order_id}', "
                        f"document shows '{extracted_order_id}'"
                    )

        # Parse customer-provided date
        date_match = re.search(
            r"(?:date|on|from)[:\s]*(\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4}|[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4})",
            customer_message,
            re.IGNORECASE,
        )
        if date_match:
            comparison.provided_date = date_match.group(1).strip()
            if extracted_date:
                if not _dates_equivalent(comparison.provided_date, extracted_date):
                    comparison.conflicts.append(
                        f"Date mismatch: customer provided '{comparison.provided_date}', "
                        f"document shows '{extracted_date}'"
                    )

        # Parse customer-provided amount
        amount_match = re.search(
            r"(?:amount|total|price)[:\s]*(?:of\s+)?(?:USD|EUR|GBP|[$])?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
            customer_message,
            re.IGNORECASE,
        )
        if amount_match:
            comparison.provided_amount = amount_match.group(1).strip()
            if extracted_amount:
                provided_numeric = _parse_amount(comparison.provided_amount)
                extracted_numeric = _parse_amount(extracted_amount)
                if provided_numeric is not None and extracted_numeric is not None:
                    if abs(provided_numeric - extracted_numeric) > 0.009:
                        comparison.conflicts.append(
                            f"Amount mismatch: customer provided '{comparison.provided_amount}', "
                            f"document shows '{extracted_amount}'"
                        )

        # Parse customer-provided product
        product_match = re.search(
            r"(?:product|item|name)[:\s]*([A-Za-z0-9 _./()-]{2,}?)(?=\s+(?:order|quantity|qty|price|amount|date|error|$)|$)",
            customer_message,
            re.IGNORECASE,
        )
        if product_match:
            comparison.provided_product = product_match.group(1).strip()
            if extracted_product:
                if not _normalize_product_name(comparison.provided_product) == _normalize_product_name(extracted_product):
                    comparison.conflicts.append(
                        f"Product mismatch: customer provided '{comparison.provided_product}', "
                        f"document shows '{extracted_product}'"
                    )

        # Parse customer-provided error code
        error_match = re.search(
            r"(?:error|code)[:\s]*([A-Z]?-?\d{3,6})",
            customer_message,
            re.IGNORECASE,
        )
        if error_match:
            comparison.provided_error_code = error_match.group(1).strip()
            if extracted_error_code:
                if not _normalize_id(comparison.provided_error_code) == _normalize_id(extracted_error_code):
                    comparison.conflicts.append(
                        f"Error code mismatch: customer provided '{comparison.provided_error_code}', "
                        f"document shows '{extracted_error_code}'"
                    )

        if comparison.conflicts:
            comparison.requires_clarification = True
            comparison.clarification_message = (
                "The document does not match the information you provided. Please clarify: "
                + "; ".join(comparison.conflicts)
            )

        return comparison

    def _extract_fields(self, text: str) -> tuple[ExtractedFields, list[str]]:
        """Extract structured fields from document text."""
        order_ids = _extract_order_ids(text)
        dates = _extract_dates(text)
        amounts = _extract_amounts(text)
        error_codes = _extract_error_codes(text)
        item_details = self._extract_items(text)
        mismatches = self._detect_mismatches(text, order_ids, amounts, item_details)

        return (
            ExtractedFields(
                order_ids=order_ids,
                dates=dates,
                amounts=amounts,
                item_details=item_details,
                error_codes=error_codes,
            ),
            mismatches,
        )

    def _extract_items(self, text: str) -> list[ItemDetail]:
        """Extract product/item details with improved format handling."""
        items: list[ItemDetail] = []
        seen_names = set()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or len(line) < 3:
                continue

            # Match ITEM: or PRODUCT: or similar keyword at start
            match = re.match(
                r"(?:ITEM|PRODUCT|SKU|ARTICLE)\s*[:\s#-]?\s*(.+?)(?:\s+(?:QTY|QUANTITY|UNIT|PRICE|TOTAL|AMOUNT|COST|UNIT\s+PRICE))",
                line,
                re.IGNORECASE,
            )

            if not match:
                continue

            name = match.group(1).strip()
            if not name:
                continue

            # Verify name doesn't consist only of keywords
            if name.upper() in ITEM_KEYWORDS:
                continue

            normalized_name = _normalize_product_name(name)
            if normalized_name in seen_names:
                continue

            seen_names.add(normalized_name)

            # Extract quantity
            qty_match = re.search(r"(?:QTY|QUANTITY)[:\s=]*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
            qty = float(qty_match.group(1)) if qty_match else None

            # Extract unit price
            price_match = re.search(
                r"(?:UNIT\s+PRICE|UNIT|PRICE)[:\s=]*(?:[$])?\s*(\d+(?:\.\d{2})?)",
                line,
                re.IGNORECASE,
            )
            unit_price = price_match.group(1) if price_match else None

            # Extract total
            total_match = re.search(
                r"(?:TOTAL|AMOUNT|COST)[:\s=]*(?:[$])?\s*(\d+(?:\.\d{2})?)",
                line,
                re.IGNORECASE,
            )
            line_total = total_match.group(1) if total_match else None

            items.append(
                ItemDetail(
                    name=name,
                    quantity=qty,
                    unit_price=unit_price,
                    line_total=line_total,
                )
            )

        return items

    def _detect_mismatches(
        self,
        text: str,
        order_ids: list[str],
        amounts: list[str],
        item_details: list[ItemDetail],
    ) -> list[str]:
        """Detect inconsistencies in extracted data."""
        mismatches: list[str] = []

        if len(order_ids) > 1:
            mismatches.append("multiple_order_ids_detected")

        line_totals = []
        for item in item_details:
            if item.line_total:
                numeric = _parse_amount(item.line_total)
                if numeric is not None:
                    line_totals.append(numeric)

        # Only compare against an explicitly labelled grand/order total,
        # not every extracted amount (which may include unit prices).
        if line_totals:
            grand_total_match = GRAND_TOTAL_PATTERN.search(text)
            if grand_total_match:
                grand_total_numeric = _parse_amount(grand_total_match.group(1))
                if grand_total_numeric is not None:
                    expected_total = round(sum(line_totals), 2)
                    if abs(grand_total_numeric - expected_total) > 0.009:
                        mismatches.append("line_items_total_mismatch")

        return mismatches

    def _estimate_text_confidence(self, text: str) -> float:
        """Estimate OCR confidence based on text characteristics."""
        if not text.strip():
            return 0.0
        useful_chars = sum(ch.isalnum() or ch in " .,:/-$#\n" for ch in text)
        return _clamp_score(useful_chars / max(len(text), 1))

    def _estimate_readability(self, text: str) -> float:
        """Estimate document readability from text."""
        stripped = text.strip()
        if not stripped:
            return 0.0
        printable = sum(ch.isprintable() and not ch.isspace() for ch in stripped)
        noise = sum(ch in "@^~`|" for ch in stripped)
        line_count = max(1, len([line for line in text.splitlines() if line.strip()]))
        density = printable / max(len(stripped), 1)
        penalty = min(0.5, noise / max(len(stripped), 1))
        balance = min(1.0, line_count / 6.0)
        return _clamp_score((density * 0.7) + (balance * 0.3) - penalty)

    def _image_readability_score(self, path: Path) -> Optional[float]:
        """Assess image quality for PNG, JPG, JPEG via blur detection.

        Raises UnsafeOrUnsupportedFileError for malformed/truncated/unsafe
        images so that the caller can reject the file properly.
        Returns None only for genuinely unsupported-but-safe cases (e.g. JPEG
        where we lack a native decoder but the file is otherwise valid).
        """
        try:
            image = _load_grayscale_image(path, self.config)
        except _ImageUnsupportedError:
            # JPEG blur detection not available – this is not a safety issue
            return None
        except ValueError as exc:
            # Malformed, truncated, unsafe PNG – propagate as rejection
            raise UnsafeOrUnsupportedFileError(str(exc)) from exc

        if not image:
            return None

        variance = _laplacian_variance(image)
        return _clamp_score(variance / self.config.blur_variance_threshold)

    def _extract_pdf_text(self, path: Path) -> tuple[str, int, bool]:
        """Extract text from PDF.

        Returns: (text, page_count, is_scanned)
        is_scanned=True indicates the PDF appears to be image-based (scanned).
        """
        try:
            data = path.read_bytes()
        except OSError:
            return "", 1, True

        if len(data) < 20:
            return "", 1, True

        page_count = max(1, len(re.findall(rb"/Type\s*/Page\b", data)))
        page_text: list[str] = []
        has_selectable_text = False

        for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL):
            stream = match.group(1)
            decoded = _try_decode_pdf_stream(stream, self.config)
            if not decoded:
                continue
            segments = _extract_text_segments(decoded)
            if segments:
                has_selectable_text = True
            page_text.extend(segments)

        extracted = ("\n".join(segment for segment in page_text if segment)).strip()
        is_scanned = not has_selectable_text or len(extracted) < 20

        return extracted, page_count, is_scanned

    def _mask_log(self, message: str) -> str:
        """Mask personal and payment information in logs."""
        masked = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", "[REDACTED_EMAIL]", message)
        masked = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[REDACTED_CARD]", masked)
        masked = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", masked)
        masked = re.sub(r"(?:CVV|CVC)[:\s=]*\d{3,4}", "[REDACTED_CVV]", masked)
        masked = re.sub(r"\b(?:\d[ -]?){13,19}\b", "[REDACTED_PAYMENT]", masked)
        return masked


def _clamp_score(value: float) -> float:
    """Clamp a score to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def _simple_text_confidence(text: str) -> float:
    """Simple text quality estimation."""
    if not text.strip():
        return 0.0
    useful_chars = sum(ch.isalnum() or ch in " .,:/-$#\n" for ch in text)
    return _clamp_score(useful_chars / max(len(text), 1))


def _parse_amount(value: Optional[str]) -> Optional[float]:
    """Parse currency amount string to float.

    Safely handles:
    - Currency symbols ($, €, £)
    - Currency codes (USD, EUR, GBP)
    - Thousand separators (commas)
    - Decimal points
    - Zero amounts

    Returns None for invalid/malformed values.
    """
    if not value:
        return None

    value_str = str(value).strip()
    if not value_str:
        return None

    # Remove currency codes
    cleaned = re.sub(r"\b(?:USD|EUR|GBP|CAD|AUD|CHF)\b", "", value_str, flags=re.IGNORECASE).strip()

    # Remove currency symbols
    cleaned = re.sub(r"[$€£]", "", cleaned)

    # Remove spaces and keep only digit, comma, decimal
    cleaned = re.sub(r"[^\d.,]", "", cleaned)

    if not cleaned:
        return None

    # Validate format: prevent multiple decimal points or invalid patterns
    decimal_count = cleaned.count(".")
    comma_count = cleaned.count(",")

    # Must have at most one decimal point
    if decimal_count > 1:
        return None

    # If we have both comma and decimal, decimal should come last
    if comma_count > 0 and decimal_count > 0:
        last_comma = cleaned.rfind(",")
        last_decimal = cleaned.rfind(".")
        if last_comma > last_decimal:
            # Invalid: comma after decimal
            return None

    # Remove commas used as thousand separators
    cleaned = cleaned.replace(",", "")

    if not cleaned or cleaned == ".":
        return None

    try:
        numeric = float(cleaned)
        return numeric
    except ValueError:
        return None


def _normalize_id(value: str) -> str:
    """Normalize order ID or error code for comparison."""
    return re.sub(r"[-\s]", "", value).upper()


def _normalize_product_name(name: str) -> str:
    """Normalize product name for comparison."""
    return re.sub(r"\s+", " ", name).strip().lower()


def _parse_date_to_tuple(date_str: str) -> Optional[tuple[int, int, int]]:
    """Parse a date string into a (year, month, day) tuple.

    Supported formats:
      - YYYY-MM-DD
      - MM/DD/YYYY
      - DD-MM-YYYY
      - Month D, YYYY  (e.g. "January 5, 2024")
      - Mon D, YYYY    (e.g. "Jan 5, 2024")
    """
    s = date_str.strip()

    # YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # MM/DD/YYYY
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", s)
    if m:
        return (int(m.group(3)), int(m.group(1)), int(m.group(2)))

    # DD-MM-YYYY
    m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", s)
    if m:
        return (int(m.group(3)), int(m.group(2)), int(m.group(1)))

    # Month D, YYYY  or  Mon D, YYYY
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", s)
    if m:
        month_name = m.group(1).lower()
        month_num = _MONTH_NAMES.get(month_name)
        if month_num is not None:
            return (int(m.group(3)), month_num, int(m.group(2)))

    return None


def _dates_equivalent(date1: str, date2: str) -> bool:
    """Check if two date strings represent the same date.

    Handles formats: YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY, Month D, YYYY,
    Mon D, YYYY.
    """
    norm1 = date1.strip()
    norm2 = date2.strip()

    if norm1 == norm2:
        return True

    t1 = _parse_date_to_tuple(norm1)
    t2 = _parse_date_to_tuple(norm2)

    if t1 is not None and t2 is not None:
        return t1 == t2

    return False


def _extract_order_ids(text: str) -> list[str]:
    """Extract unique order IDs."""
    matches = re.findall(
        r"\b(?:ORDER|ORD|PO)[-:\s#]*([A-Z0-9-]{4,})\b",
        text,
        re.IGNORECASE,
    )
    return list(dict.fromkeys(match.strip() for match in matches if match.strip()))


def _extract_dates(text: str) -> list[str]:
    """Extract unique dates."""
    matches = re.findall(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4}|[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4})\b",
        text,
        re.IGNORECASE,
    )
    return list(dict.fromkeys(match.strip() for match in matches if match.strip()))


def _extract_amounts(text: str) -> list[str]:
    """Extract unique currency amounts."""
    matches = re.findall(
        r"(?:USD|EUR|GBP)\s*\d+(?:\.\d{2})?|\$\s?\d+(?:,\d{3})*(?:\.\d{2})?|€\s?\d+(?:,\d{3})*(?:\.\d{2})?|£\s?\d+(?:,\d{3})*(?:\.\d{2})?",
        text,
        re.IGNORECASE,
    )
    return list(dict.fromkeys(match.strip() for match in matches if match.strip()))


def _extract_error_codes(text: str) -> list[str]:
    """Extract unique error codes."""
    matches = re.findall(
        r"\b(?:ERR|ERROR|E)[-_]?\d{3,6}\b",
        text,
        re.IGNORECASE,
    )
    return list(dict.fromkeys(match.strip() for match in matches if match.strip()))


class TesseractThenSidecarBackend:
    """Composite OCR backend: tries Tesseract, then falls back to sidecar JSON."""

    def __init__(
        self,
        primary: Optional[OCRBackend] = None,
        fallback: Optional[OCRBackend] = None,
    ) -> None:
        self.primary = primary or TesseractOCRBackend()
        self.fallback = fallback or SidecarOCRBackend()

    def extract_text(self, file_path: Path) -> OCRTextResult:
        """Extract text using primary backend, fall back if needed."""
        try:
            primary_result = self.primary.extract_text(file_path)
        except Exception:
            primary_result = OCRTextResult(text="", confidence=0.0, page_count=1)

        if primary_result.text.strip():
            return primary_result

        try:
            fallback_result = self.fallback.extract_text(file_path)
            return fallback_result
        except Exception:
            return OCRTextResult(text="", confidence=0.0, page_count=1)


def _try_decode_pdf_stream(stream: bytes, config: AnalyzerConfig) -> str:
    """Attempt to decode PDF stream, trying common encodings/compressions."""
    candidates = [stream]
    try:
        decompressed = zlib.decompress(stream)
        if len(decompressed) <= config.max_decompressed_size:
            candidates.append(decompressed)
    except zlib.error:
        pass

    for candidate in candidates:
        for encoding in ("utf-8", "latin-1", "ascii"):
            try:
                return candidate.decode(encoding, errors="ignore")
            except UnicodeDecodeError:
                continue

    return ""


def _extract_text_segments(decoded_stream: str) -> list[str]:
    """Extract text segments from decoded PDF stream."""
    segments: list[str] = []

    # Extract text from (text) Tj operators
    for match in re.finditer(r"\((.*?)\)\s*Tj", decoded_stream, re.DOTALL):
        segments.append(_decode_pdf_literal(match.group(1)))

    # Extract text from [(text...)] TJ operators
    for match in re.finditer(r"\[(.*?)\]\s*TJ", decoded_stream, re.DOTALL):
        for literal_match in re.finditer(r"\((.*?)\)", match.group(1), re.DOTALL):
            segments.append(_decode_pdf_literal(literal_match.group(1)))

    return [s for s in segments if s.strip()]


def _decode_pdf_literal(value: str) -> str:
    """Decode PDF string literal with escape sequences."""
    result = io.StringIO()
    idx = 0
    while idx < len(value):
        char = value[idx]
        if char == "\\" and idx + 1 < len(value):
            nxt = value[idx + 1]
            escapes = {
                "n": "\n",
                "r": "\r",
                "t": "\t",
                "b": "\b",
                "f": "\f",
                "\\": "\\",
                "(": "(",
                ")": ")",
            }
            if nxt in escapes:
                result.write(escapes[nxt])
                idx += 2
                continue
            if nxt.isdigit():
                octal_match = re.match(r"(\d{1,3})", value[idx + 1 :])
                if octal_match:
                    octal_str = octal_match.group(1)[:3]
                    try:
                        result.write(chr(int(octal_str, 8)))
                        idx += 1 + len(octal_str)
                        continue
                    except ValueError:
                        pass
        result.write(char)
        idx += 1
    return result.getvalue()


class _ImageUnsupportedError(Exception):
    """Raised when image format is valid but blur detection is not available."""
    pass


def _load_grayscale_image(path: Path, config: AnalyzerConfig) -> list[list[int]]:
    """Load image as grayscale matrix for blur detection.

    Supports PNG natively. For JPEG/JPG, raises _ImageUnsupportedError.
    For malformed/unsafe PNGs, raises ValueError to trigger rejection.
    """
    try:
        header = _read_file_header(path, 8)
    except OSError:
        raise ValueError("image_read_error")

    if header.startswith(PNG_MAGIC):
        return _load_png_grayscale(path, config)

    if header.startswith(JPEG_MAGIC):
        raise _ImageUnsupportedError("jpeg_blur_detection_unavailable")

    raise ValueError("unsupported_image_format")


def _load_png_grayscale(path: Path, config: AnalyzerConfig) -> list[list[int]]:
    """Parse PNG file and return grayscale pixel matrix.

    Raises ValueError for any structural, safety, or data integrity issue
    so that the caller can reject the file.
    """
    try:
        data = path.read_bytes()
    except OSError:
        raise ValueError("png_read_error")

    if not data.startswith(PNG_MAGIC):
        raise ValueError("not_a_png")

    if len(data) < 16:
        raise ValueError("png_truncated")

    pos = len(PNG_MAGIC)
    width = 0
    height = 0
    bit_depth = 8
    color_type = 0
    compressed = bytearray()
    interlace_method = 0
    found_ihdr = False
    found_idat = False
    found_iend = False

    while pos < len(data):
        if pos + 8 > len(data):
            break

        length = int.from_bytes(data[pos : pos + 4], "big")
        chunk_type = data[pos + 4 : pos + 8]
        pos += 8

        if length < 0 or length > config.max_png_chunk_size:
            raise ValueError("png_chunk_invalid")

        if pos + length > len(data):
            raise ValueError("png_chunk_truncated")

        chunk_data = data[pos : pos + length]
        pos += length + 4  # skip data + CRC

        if chunk_type == b"IHDR":
            if length != 13:
                raise ValueError("png_ihdr_invalid")
            found_ihdr = True
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
            interlace_method = chunk_data[11] if len(chunk_data) > 11 else 0

            if width <= 0 or height <= 0 or width > config.max_image_dimension or height > config.max_image_dimension:
                raise ValueError("png_dimensions_invalid")
            if bit_depth != 8:
                raise ValueError("png_bitdepth_unsupported")
            if color_type not in {0, 2, 6}:
                raise ValueError("png_colortype_unsupported")
            if interlace_method != 0:
                raise ValueError("png_interlace_unsupported")

        elif chunk_type == b"IDAT":
            if not found_ihdr:
                raise ValueError("png_idat_before_ihdr")
            found_idat = True
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            found_iend = True
            break

    # Validate required PNG structure
    if not found_ihdr:
        raise ValueError("png_no_ihdr")
    if not found_idat:
        raise ValueError("png_no_idat")
    if not found_iend:
        raise ValueError("png_no_iend")

    if width == 0 or height == 0:
        raise ValueError("png_no_ihdr")

    # Validate expected decompressed size before decompression to prevent
    # excessive memory allocation.
    channels = {0: 1, 2: 3, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError("png_colortype_unsupported")
    stride = width * channels
    expected_size = height * (1 + stride)  # 1 filter byte per row

    if expected_size > config.max_decompressed_size:
        raise ValueError("png_decompressed_too_large")

    try:
        raw = zlib.decompress(bytes(compressed))
    except zlib.error:
        raise ValueError("png_decompress_failed")

    # Verify actual decompressed size against safety limit
    if len(raw) > config.max_decompressed_size:
        raise ValueError("png_decompressed_too_large")

    if len(raw) < expected_size:
        raise ValueError("png_scanlines_truncated")

    rows: list[list[int]] = []
    index = 0
    previous = [0] * stride

    for row_idx in range(height):
        if index >= len(raw):
            raise ValueError("png_scanline_out_of_bounds")

        filter_type = raw[index]
        index += 1

        if index + stride > len(raw):
            raise ValueError("png_scanline_truncated")

        scanline = bytearray(raw[index : index + stride])
        index += stride

        recon = _reconstruct_png_scanline(filter_type, scanline, previous, channels)
        previous = list(recon)

        if color_type == 0:
            rows.append(list(recon))
        else:
            gray_row = []
            for offset in range(0, len(recon), channels):
                if offset + 2 >= len(recon):
                    break
                r = recon[offset]
                g = recon[offset + 1]
                b = recon[offset + 2]
                gray_row.append(int(round((0.299 * r) + (0.587 * g) + (0.114 * b))))
            rows.append(gray_row)

    return rows


def _reconstruct_png_scanline(
    filter_type: int,
    scanline: bytearray,
    previous: list[int],
    bytes_per_pixel: int,
) -> bytearray:
    """Reconstruct PNG scanline from filtered bytes."""
    recon = bytearray(len(scanline))

    for index, value in enumerate(scanline):
        left = recon[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        up = previous[index] if index < len(previous) else 0
        up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel and index < len(previous) else 0

        if filter_type == 0:
            recon[index] = value
        elif filter_type == 1:
            recon[index] = (value + left) & 0xFF
        elif filter_type == 2:
            recon[index] = (value + up) & 0xFF
        elif filter_type == 3:
            recon[index] = (value + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            recon[index] = (value + _paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"unsupported_png_filter_{filter_type}")

    return recon


def _paeth_predictor(a: int, b: int, c: int) -> int:
    """Paeth predictor function for PNG decompression."""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _laplacian_variance(image: list[list[int]]) -> float:
    """Compute Laplacian variance for blur detection."""
    if len(image) < 3 or len(image[0]) < 3:
        return 0.0

    responses: list[float] = []
    height = len(image)
    width = len(image[0])

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = image[y][x]
            response = (
                (4 * center)
                - image[y - 1][x]
                - image[y + 1][x]
                - image[y][x - 1]
                - image[y][x + 1]
            )
            responses.append(float(response))

    if not responses:
        return 0.0

    mean = sum(responses) / len(responses)
    variance = sum((value - mean) ** 2 for value in responses) / len(responses)
    return variance
