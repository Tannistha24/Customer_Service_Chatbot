from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyzerConfig:
    """Configuration for document analyzer."""
    max_file_size_bytes: int = 10 * 1024 * 1024
    blur_variance_threshold: float = 60.0
    low_confidence_threshold: float = 0.45
    low_readability_threshold: float = 0.45
    max_image_dimension: int = 16384
    max_png_chunk_size: int = 100 * 1024 * 1024
    max_decompressed_size: int = 500 * 1024 * 1024
