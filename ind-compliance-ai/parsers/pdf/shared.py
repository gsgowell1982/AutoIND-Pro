from __future__ import annotations

from dataclasses import dataclass
import re


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\x00", " ").split())


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _compact_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", _clean_text(text).lower())


def _text_contains_text(source: str, target: str) -> bool:
    source_compact = _compact_text(source)
    target_compact = _compact_text(target)
    if not source_compact or not target_compact:
        return False
    return target_compact in source_compact


def _bbox_to_list(bbox: tuple[float, float, float, float]) -> list[float]:
    return [round(float(item), 2) for item in bbox]


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_union(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(item[0] for item in bboxes)
    y0 = min(item[1] for item in bboxes)
    x1 = max(item[2] for item in bboxes)
    y1 = max(item[3] for item in bboxes)
    return (x0, y0, x1, y1)


def _bbox_intersection_ratio(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
) -> float:
    x0 = max(a_bbox[0], b_bbox[0])
    y0 = max(a_bbox[1], b_bbox[1])
    x1 = min(a_bbox[2], b_bbox[2])
    y1 = min(a_bbox[3], b_bbox[3])
    intersection = _bbox_area((x0, y0, x1, y1))
    min_area = max(1.0, min(_bbox_area(a_bbox), _bbox_area(b_bbox)))
    return intersection / min_area


def _horizontal_overlap_ratio(
    a_bbox: tuple[float, float, float, float],
    b_bbox: tuple[float, float, float, float],
) -> float:
    left = max(a_bbox[0], b_bbox[0])
    right = min(a_bbox[2], b_bbox[2])
    overlap = max(0.0, right - left)
    min_width = max(1.0, min(a_bbox[2] - a_bbox[0], b_bbox[2] - b_bbox[0]))
    return overlap / min_width


@dataclass(slots=True)
class _Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)
