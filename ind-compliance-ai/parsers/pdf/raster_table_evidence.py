"""Source-isolated raster table evidence contracts.

This module is the boundary between page-image/table-region evidence and the
shared table semantic pipeline.  It deliberately stops at evidence contracts and
observe-only region candidates; it does not build table ASTs or presentation
grids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .region_ownership import RegionCandidate, RegionType

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore[assignment]


BBox = tuple[float, float, float, float]


class RasterTableEvidenceProvider(Protocol):
    provider_name: str

    def detect(self, image_path: Path | str, *, page: int = 1) -> list["RasterTableRegionCandidate"]:
        """Return source-isolated raster table region candidates."""


@dataclass(slots=True)
class RasterEvidenceSource:
    source_type: str
    confidence: float
    bbox: BBox
    signals: dict[str, Any] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "confidence": round(float(self.confidence), 3),
            "bbox": [float(value) for value in self.bbox],
            "signals": dict(self.signals),
            "provenance": list(self.provenance),
        }


@dataclass(slots=True)
class RasterTableRegionCandidate:
    page: int
    bbox: BBox
    sources: list[RasterEvidenceSource]
    confidence: float | None = None
    role: str = "table_region"
    warnings: list[str] = field(default_factory=list)
    candidate_id: str = ""
    signals: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is None:
            self.confidence = _confidence_from_sources(self.sources)
        if not self.candidate_id:
            self.candidate_id = _candidate_id(self.page, self.bbox, self.sources)

    @property
    def source_types(self) -> list[str]:
        return sorted({source.source_type for source in self.sources})

    @property
    def provenance(self) -> list[str]:
        result: list[str] = []
        for source in self.sources:
            for item in source.provenance:
                if item not in result:
                    result.append(item)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "page": int(self.page),
            "bbox": [float(value) for value in self.bbox],
            "source": _public_source_name(self.source_types),
            "source_type": "raster_image",
            "confidence": round(float(self.confidence or 0.0), 3),
            "role": self.role,
            "sources": [source.to_dict() for source in self.sources],
            "signals": {
                "source_types": self.source_types,
                **dict(self.signals),
            },
            "provenance": self.provenance,
            "warnings": list(self.warnings),
            "observe_only": True,
        }


@dataclass(slots=True)
class TableEvidence:
    table_id: str
    page: int
    bbox: BBox
    source_type: str
    region_confidence: float
    evidence_sources: list[dict[str, Any]] = field(default_factory=list)
    text_tokens: list[dict[str, Any]] = field(default_factory=list)
    visual_lines: list[dict[str, Any]] = field(default_factory=list)
    cell_candidates: list[dict[str, Any]] = field(default_factory=list)
    caption_candidates: list[dict[str, Any]] = field(default_factory=list)
    note_candidates: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page": int(self.page),
            "bbox": [float(value) for value in self.bbox],
            "source_type": self.source_type,
            "region_confidence": round(float(self.region_confidence), 3),
            "evidence_sources": [dict(item) for item in self.evidence_sources],
            "text_tokens": [dict(item) for item in self.text_tokens],
            "visual_lines": [dict(item) for item in self.visual_lines],
            "cell_candidates": [dict(item) for item in self.cell_candidates],
            "caption_candidates": [dict(item) for item in self.caption_candidates],
            "note_candidates": [dict(item) for item in self.note_candidates],
            "provenance": list(self.provenance),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class VisualRasterTableEvidenceProvider:
    """Cheap visual fallback provider for raster table region diagnostics."""

    provider_name: str = "visual_raster_table_evidence"

    def detect(self, image_path: Path | str, *, page: int = 1) -> list[RasterTableRegionCandidate]:
        return _detect_visual_raster_table_region_candidates(Path(image_path), page=page)


@dataclass(slots=True)
class CompositeRasterTableEvidenceProvider:
    """Run independent raster providers and fuse their region evidence."""

    providers: list[RasterTableEvidenceProvider]
    provider_name: str = "composite_raster_table_evidence"
    last_warnings: list[str] = field(default_factory=list, init=False)

    def detect(self, image_path: Path | str, *, page: int = 1) -> list[RasterTableRegionCandidate]:
        self.last_warnings = []
        candidates: list[RasterTableRegionCandidate] = []
        successful_provider_count = 0
        for provider in self.providers:
            provider_name = str(getattr(provider, "provider_name", provider.__class__.__name__))
            try:
                provider_candidates = provider.detect(image_path, page=page)
            except Exception as exc:
                self.last_warnings.append(f"provider_failed:{provider_name}:{type(exc).__name__}")
                continue
            successful_provider_count += 1
            for candidate in provider_candidates:
                candidate.signals.setdefault("provider_name", provider_name)
                candidates.append(candidate)
        fused = fuse_raster_table_region_candidates(candidates)
        for candidate in fused:
            candidate.signals["provider_count"] = successful_provider_count
            for warning in self.last_warnings:
                if warning not in candidate.warnings:
                    candidate.warnings.append(warning)
        return fused


def fuse_raster_table_region_candidates(
    candidates: list[RasterTableRegionCandidate],
) -> list[RasterTableRegionCandidate]:
    """Fuse overlapping candidates by source agreement while preserving evidence."""

    fused: list[RasterTableRegionCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-float(item.confidence or 0.0), item.candidate_id)):
        overlapping_index = next(
            (
                index
                for index, existing in enumerate(fused)
                if existing.page == candidate.page
                and (
                    bbox_iou(existing.bbox, candidate.bbox) >= 0.25
                    or bbox_intersection_over_min(existing.bbox, candidate.bbox) >= 0.65
                )
            ),
            None,
        )
        if overlapping_index is None:
            fused.append(candidate)
            continue
        existing = fused[overlapping_index]
        sources = [*existing.sources]
        for source in candidate.sources:
            if not _same_source_seen(source, sources):
                sources.append(source)
        bbox = union_bboxes([existing.bbox, candidate.bbox])
        source_types = sorted({source.source_type for source in sources})
        signals = {
            **dict(existing.signals),
            **dict(candidate.signals),
            "source_agreement": len(source_types),
            "source_types": source_types,
        }
        fused[overlapping_index] = RasterTableRegionCandidate(
            page=existing.page,
            bbox=bbox,
            sources=sources,
            confidence=min(1.0, max(float(existing.confidence or 0.0), float(candidate.confidence or 0.0)) + 0.04 * max(1, len(source_types) - 1)),
            role=existing.role,
            warnings=[*existing.warnings, *[item for item in candidate.warnings if item not in existing.warnings]],
            candidate_id=existing.candidate_id,
            signals=signals,
        )
    return sorted(fused, key=lambda item: (-float(item.confidence or 0.0), item.bbox[1], item.bbox[0]))


def raster_table_candidate_to_region_candidate(candidate: RasterTableRegionCandidate) -> RegionCandidate:
    return RegionCandidate(
        candidate_id=candidate.candidate_id,
        page=candidate.page,
        region_type=RegionType.TABLE,
        bbox=candidate.bbox,
        source="raster_table_evidence",
        evidence_refs=candidate.provenance,
        text="",
        confidence=float(candidate.confidence or 0.0),
        signals={
            "source_types": candidate.source_types,
            **dict(candidate.signals),
        },
        metadata={
            "source_type": "raster_image",
            "observe_only": True,
            "role": candidate.role,
            "evidence_sources": [source.to_dict() for source in candidate.sources],
            "warnings": list(candidate.warnings),
        },
    )


def detect_raster_table_regions(image_path: Path, *, page: int = 1) -> list[dict[str, Any]]:
    """Return observe-only raster table region candidates as serializable dicts.

    The default provider is a cheap visual fallback used for diagnostics.  The
    contract is intentionally compatible with stronger OCR/layout providers that
    can add `layout_model_table_region` or `ocr_text_matrix` sources later.
    """

    candidates = detect_raster_table_region_candidates(image_path, page=page)
    return [candidate.to_dict() for candidate in candidates]


def detect_raster_table_region_candidates(image_path: Path, *, page: int = 1) -> list[RasterTableRegionCandidate]:
    return VisualRasterTableEvidenceProvider().detect(image_path, page=page)


def _detect_visual_raster_table_region_candidates(image_path: Path, *, page: int = 1) -> list[RasterTableRegionCandidate]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for raster table region diagnostics.")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image for raster table diagnostics: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    line_candidates = _detect_line_grid_table_regions(gray, page=page, image_path=image_path)
    text_candidates = _merge_aligned_text_matrix_regions(
        _detect_text_matrix_table_regions(gray, page=page, image_path=image_path),
        page_height=gray.shape[0],
    )
    return fuse_raster_table_region_candidates(line_candidates + text_candidates)


def merge_aligned_text_matrix_regions(regions: list[dict[str, Any]], *, page_height: float) -> list[dict[str, Any]]:
    candidates = [
        _candidate_from_region_dict(region, page=1)
        for region in regions
        if str(region.get("source") or "") in {"raster_text_matrix", "ocr_text_matrix"}
    ]
    merged = _merge_aligned_text_matrix_regions(candidates, page_height=page_height)
    passthrough = [
        region
        for region in regions
        if str(region.get("source") or "") not in {"raster_text_matrix", "ocr_text_matrix"}
    ]
    return passthrough + [candidate.to_dict() for candidate in merged]


def _detect_line_grid_table_regions(gray: Any, *, page: int, image_path: Path) -> list[RasterTableRegionCandidate]:
    height, width = gray.shape[:2]
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, width // 50), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, height // 70)))
    horizontal = cv2.dilate(cv2.erode(binary, horizontal_kernel, iterations=1), horizontal_kernel, iterations=1)
    vertical = cv2.dilate(cv2.erode(binary, vertical_kernel, iterations=1), vertical_kernel, iterations=1)
    grid = cv2.bitwise_or(horizontal, vertical)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, width // 300), max(5, height // 300)))
    grid = cv2.dilate(grid, close_kernel, iterations=2)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[RasterTableRegionCandidate] = []
    page_area = max(1.0, float(width * height))
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = float(box_width * box_height)
        area_ratio = area / page_area
        if area < page_area * 0.002:
            continue
        if box_width < width * 0.12 or box_height < height * 0.035:
            continue
        if box_width > width * 0.98 and box_height > height * 0.90:
            continue
        bbox = (float(x), float(y), float(x + box_width), float(y + box_height))
        horizontal_line_count = _count_line_segments_in_bbox(horizontal, bbox, orientation="horizontal")
        vertical_line_count = _count_line_segments_in_bbox(vertical, bbox, orientation="vertical")
        if area_ratio > 0.55 and horizontal_line_count <= 3 and vertical_line_count <= 3:
            continue
        source = RasterEvidenceSource(
            source_type="visual_line_grid",
            confidence=0.82,
            bbox=bbox,
            signals={
                "page_width": width,
                "page_height": height,
                "area_ratio": round(area_ratio, 6),
                "horizontal_line_count": horizontal_line_count,
                "vertical_line_count": vertical_line_count,
            },
            provenance=[f"image:{image_path.name}", "visual:line_grid"],
        )
        regions.append(RasterTableRegionCandidate(page=page, bbox=bbox, sources=[source]))
    return fuse_raster_table_region_candidates(regions)


def _count_line_segments_in_bbox(mask: Any, bbox: BBox, *, orientation: str) -> int:
    height, width = mask.shape[:2]
    x0 = max(0, int(bbox[0]))
    y0 = max(0, int(bbox[1]))
    x1 = min(width, int(bbox[2]))
    y1 = min(height, int(bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return 0
    crop = mask[y0:y1, x0:x1]
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in contours:
        _, _, box_width, box_height = cv2.boundingRect(contour)
        if orientation == "horizontal":
            if box_width >= max(18, (x1 - x0) * 0.20) and box_width >= box_height * 6:
                count += 1
        else:
            if box_height >= max(18, (y1 - y0) * 0.20) and box_height >= box_width * 6:
                count += 1
    return count


def _detect_text_matrix_table_regions(gray: Any, *, page: int, image_path: Path) -> list[RasterTableRegionCandidate]:
    height, width = gray.shape[:2]
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 18)
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, width // 90), max(2, height // 900)))
    dilated = cv2.dilate(binary, text_kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BBox] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 8 or box_height < 5:
            continue
        if box_width > width * 0.95 or box_height > height * 0.12:
            continue
        if box_width * box_height < 30:
            continue
        boxes.append((float(x), float(y), float(x + box_width), float(y + box_height)))
    rows = _group_boxes_into_visual_rows(boxes)
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if _row_has_matrix_shape(row, page_width=width):
            if current and float(row["cy"]) - float(current[-1]["cy"]) > max(45.0, height * 0.035):
                if len(current) >= 2:
                    groups.append(current)
                current = []
            current.append(row)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = []
    if len(current) >= 2:
        groups.append(current)

    regions: list[RasterTableRegionCandidate] = []
    for group in groups:
        bbox = union_bboxes([tuple(row["bbox"]) for row in group])
        if (bbox[2] - bbox[0]) < width * 0.18 or (bbox[3] - bbox[1]) < height * 0.03:
            continue
        matrix_score = _text_matrix_alignment_score(group, page_width=width)
        if matrix_score["stable_column_count"] < 2:
            continue
        if matrix_score["aligned_row_ratio"] < 0.55:
            continue
        source = RasterEvidenceSource(
            source_type="visual_text_matrix",
            confidence=min(0.80, 0.44 + len(group) * 0.05 + matrix_score["aligned_row_ratio"] * 0.16),
            bbox=bbox,
            signals={
                "visual_row_count": len(group),
                "page_width": width,
                "page_height": height,
                **matrix_score,
            },
            provenance=[f"image:{image_path.name}", "visual:text_matrix"],
        )
        regions.append(RasterTableRegionCandidate(page=page, bbox=bbox, sources=[source]))
    return regions


def _text_matrix_alignment_score(group: list[dict[str, Any]], *, page_width: float) -> dict[str, Any]:
    tolerance = max(16.0, page_width * 0.018)
    clusters: list[dict[str, Any]] = []
    aligned_rows: set[int] = set()
    for row_index, row in enumerate(group):
        row_boxes = sorted(row.get("boxes") or [], key=lambda item: item[0])
        row_cluster_hits: set[int] = set()
        for bbox in row_boxes:
            x0 = float(bbox[0])
            best_index: int | None = None
            best_distance = 999999.0
            for cluster_index, cluster in enumerate(clusters):
                distance = abs(x0 - float(cluster["x"]))
                if distance <= tolerance and distance < best_distance:
                    best_index = cluster_index
                    best_distance = distance
            if best_index is None:
                clusters.append({"x": x0, "rows": {row_index}, "count": 1})
                row_cluster_hits.add(len(clusters) - 1)
            else:
                cluster = clusters[best_index]
                rows = cluster["rows"]
                rows.add(row_index)
                cluster["count"] = int(cluster["count"]) + 1
                cluster["x"] = (float(cluster["x"]) * (int(cluster["count"]) - 1) + x0) / int(cluster["count"])
                row_cluster_hits.add(best_index)
        if len(row_cluster_hits) >= 2:
            aligned_rows.add(row_index)
    min_rows = max(2, min(len(group), int(round(len(group) * 0.45))))
    stable_columns = [cluster for cluster in clusters if len(cluster["rows"]) >= min_rows]
    aligned_row_count = sum(
        1
        for row_index, row in enumerate(group)
        if sum(1 for cluster in stable_columns if row_index in cluster["rows"]) >= 2
    )
    return {
        "stable_column_count": len(stable_columns),
        "aligned_row_ratio": round(aligned_row_count / max(1, len(group)), 6),
    }


def _merge_aligned_text_matrix_regions(
    regions: list[RasterTableRegionCandidate],
    *,
    page_height: float,
) -> list[RasterTableRegionCandidate]:
    text_regions = [
        region
        for region in regions
        if any(source.source_type in {"visual_text_matrix", "ocr_text_matrix"} for source in region.sources)
    ]
    other_regions = [region for region in regions if region not in text_regions]
    merged: list[RasterTableRegionCandidate] = []
    consumed: set[int] = set()
    ordered = sorted(enumerate(text_regions), key=lambda item: (item[1].bbox[1], item[1].bbox[0]))
    for original_index, region in ordered:
        if original_index in consumed:
            continue
        group = [region]
        consumed.add(original_index)
        changed = True
        while changed:
            changed = False
            group_bbox = union_bboxes([item.bbox for item in group])
            for candidate_index, candidate in ordered:
                if candidate_index in consumed:
                    continue
                candidate_bbox = candidate.bbox
                gap = candidate_bbox[1] - group_bbox[3]
                if gap < 0:
                    gap = group_bbox[1] - candidate_bbox[3]
                if gap < 0 or gap > max(90.0, page_height * 0.055):
                    continue
                if horizontal_overlap_over_min(group_bbox, candidate_bbox) < 0.72:
                    continue
                left_delta = abs(group_bbox[0] - candidate_bbox[0])
                right_delta = abs(group_bbox[2] - candidate_bbox[2])
                reference_width = max(1.0, min(group_bbox[2] - group_bbox[0], candidate_bbox[2] - candidate_bbox[0]))
                if max(left_delta, right_delta) > max(90.0, reference_width * 0.18):
                    continue
                group.append(candidate)
                consumed.add(candidate_index)
                changed = True
        if len(group) == 1:
            merged.append(group[0])
            continue
        bbox = union_bboxes([item.bbox for item in group])
        visual_row_count = 0
        sources: list[RasterEvidenceSource] = []
        for item in group:
            for source in item.sources:
                source_signals = source.signals if isinstance(source.signals, dict) else {}
                visual_row_count += int(source_signals.get("visual_row_count", 0) or 0)
                sources.append(source)
        merged.append(
            RasterTableRegionCandidate(
                page=group[0].page,
                bbox=bbox,
                sources=sources,
                confidence=min(0.84, max(float(item.confidence or 0.0) for item in group) + 0.04),
                signals={
                    "visual_row_count": visual_row_count,
                    "merged_region_count": len(group),
                    "merge_strategy": "aligned_text_matrix_vertical_sections",
                },
            )
        )
    return sorted(other_regions + merged, key=lambda item: (item.bbox[1], item.bbox[0]))


def horizontal_overlap_over_min(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    return overlap / max(1.0, min(left[2] - left[0], right[2] - right[0]))


def _group_boxes_into_visual_rows(boxes: list[BBox]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bbox in sorted(boxes, key=lambda item: (item[1], item[0])):
        cy = (bbox[1] + bbox[3]) / 2.0
        box_height = max(1.0, bbox[3] - bbox[1])
        best_row: dict[str, Any] | None = None
        best_distance = 999999.0
        for row in rows:
            tolerance = max(6.0, min(18.0, (float(row["height"]) + box_height) * 0.7))
            distance = abs(cy - float(row["cy"]))
            if distance <= tolerance and distance < best_distance:
                best_row = row
                best_distance = distance
        if best_row is None:
            rows.append({"boxes": [bbox], "cy": cy, "height": box_height, "bbox": bbox})
            continue
        best_row["boxes"].append(bbox)
        best_row["cy"] = sum((item[1] + item[3]) / 2.0 for item in best_row["boxes"]) / len(best_row["boxes"])
        best_row["height"] = max(float(best_row["height"]), box_height)
        best_row["bbox"] = union_bboxes([tuple(best_row["bbox"]), bbox])
    return sorted(rows, key=lambda row: float(row["cy"]))


def _row_has_matrix_shape(row: dict[str, Any], *, page_width: float) -> bool:
    boxes = sorted(row.get("boxes") or [], key=lambda item: item[0])
    if len(boxes) < 2:
        return False
    gaps = [boxes[index + 1][0] - boxes[index][2] for index in range(len(boxes) - 1)]
    separated_cells = 1 + sum(1 for gap in gaps if gap > max(12.0, page_width * 0.01))
    bbox = tuple(float(value) for value in row["bbox"])
    row_width = bbox[2] - bbox[0]
    return (separated_cells >= 2 and row_width >= page_width * 0.22) or (
        len(boxes) >= 3 and row_width >= page_width * 0.18
    )


def bbox_iou(left: BBox, right: BBox) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    denominator = left_area + right_area - intersection
    return intersection / denominator if denominator > 0.0 else 0.0


def bbox_intersection_over_min(left: BBox, right: BBox) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1.0, min(left_area, right_area))


def union_bboxes(bboxes: list[BBox]) -> BBox:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _candidate_from_region_dict(region: dict[str, Any], *, page: int) -> RasterTableRegionCandidate:
    bbox = tuple(float(value) for value in region.get("bbox", (0, 0, 0, 0)))
    source_name = str(region.get("source") or "raster_text_matrix")
    source_type = "visual_text_matrix" if source_name == "raster_text_matrix" else source_name
    source = RasterEvidenceSource(
        source_type=source_type,
        confidence=float(region.get("confidence", 0.0) or 0.0),
        bbox=bbox,
        signals=dict(region.get("signals") or {}),
        provenance=[str(item) for item in region.get("provenance") or []],
    )
    return RasterTableRegionCandidate(
        page=page,
        bbox=bbox,
        sources=[source],
        confidence=float(region.get("confidence", 0.0) or 0.0),
    )


def _same_source_seen(source: RasterEvidenceSource, sources: list[RasterEvidenceSource]) -> bool:
    return any(
        item.source_type == source.source_type
        and bbox_iou(item.bbox, source.bbox) > 0.98
        and item.provenance == source.provenance
        for item in sources
    )


def _confidence_from_sources(sources: list[RasterEvidenceSource]) -> float:
    if not sources:
        return 0.0
    source_types = {source.source_type for source in sources}
    return min(1.0, max(float(source.confidence) for source in sources) + 0.04 * max(0, len(source_types) - 1))


def _public_source_name(source_types: list[str]) -> str:
    if len(source_types) > 1:
        return "raster_fused_table_region"
    if not source_types:
        return "raster_table_region"
    source_type = source_types[0]
    if source_type == "visual_line_grid":
        return "raster_line_grid"
    if source_type in {"visual_text_matrix", "ocr_text_matrix"}:
        return "raster_text_matrix"
    if source_type == "layout_model_table_region":
        return "raster_layout_table_region"
    return f"raster_{source_type}"


def _candidate_id(page: int, bbox: BBox, sources: list[RasterEvidenceSource]) -> str:
    source = _public_source_name(sorted({item.source_type for item in sources}))
    x0, y0, x1, y1 = (int(round(value)) for value in bbox)
    return f"raster-table-p{int(page)}-{source}-{x0}-{y0}-{x1}-{y1}"
