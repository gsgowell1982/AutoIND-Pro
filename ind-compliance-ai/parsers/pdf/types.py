from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PDF_PARSER_HINT = "pdf-ast-v5"


@dataclass(slots=True)
class PdfPipelineCounters:
    semantic_merge_count: int = 0
    image_text_recovered_count: int = 0
    rejected_table_candidates: int = 0
    table_fragment_merge_count: int = 0
    duplicate_image_blocks_removed: int = 0
    header_footer_filtered_count: int = 0
    cross_page_table_links: int = 0


@dataclass(slots=True)
class PdfPipelineState:
    page_payloads: list[dict[str, Any]] = field(default_factory=list)
    table_asts: list[dict[str, Any]] = field(default_factory=list)
    figure_nodes: list[dict[str, Any]] = field(default_factory=list)
    page_heights: dict[int, float] = field(default_factory=dict)
    counters: PdfPipelineCounters = field(default_factory=PdfPipelineCounters)

