from __future__ import annotations

from typing import Any


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        bbox = tuple(float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _bbox_union_loose(bboxes: list[Any]) -> tuple[float, float, float, float]:
    valid = [_coerce_bbox(bbox) for bbox in bboxes]
    valid = [bbox for bbox in valid if bbox is not None]
    if not valid:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    )


def _bbox_union(bboxes: list[list[float] | tuple[float, float, float, float]]) -> list[float]:
    valid = [bbox for bbox in bboxes if len(bbox) >= 4]
    if not valid:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(float(bbox[0]) for bbox in valid),
        min(float(bbox[1]) for bbox in valid),
        max(float(bbox[2]) for bbox in valid),
        max(float(bbox[3]) for bbox in valid),
    ]


def _bbox_center_y(value: Any) -> float:
    bbox = _coerce_bbox(value)
    if bbox is None:
        return 0.0
    return (bbox[1] + bbox[3]) / 2.0


def _bbox_height(bbox: list[float] | tuple[float, float, float, float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_center_y_value(block: dict[str, Any]) -> float:
    bbox = list(block.get("bbox", []) or [])
    if len(bbox) < 4:
        return 0.0
    try:
        return (float(bbox[1]) + float(bbox[3])) / 2.0
    except (TypeError, ValueError):
        return 0.0


def _bbox_center_inside(value: Any, container: tuple[float, float, float, float], *, tolerance: float = 0.0) -> bool:
    bbox = _coerce_bbox(value)
    if bbox is None:
        return False
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    return (
        container[0] - tolerance <= center_x <= container[2] + tolerance
        and container[1] - tolerance <= center_y <= container[3] + tolerance
    )


def _block_vertical_gap(
    upper_block: dict[str, Any] | None,
    lower_block: dict[str, Any],
) -> float | None:
    upper_bbox = _coerce_bbox((upper_block or {}).get("bbox"))
    lower_bbox = _coerce_bbox(lower_block.get("bbox"))
    if upper_bbox is None or lower_bbox is None:
        return None
    return float(lower_bbox[1]) - float(upper_bbox[3])


def _node_physical_order_key(node: dict[str, Any]) -> tuple[float, float]:
    bbox = node.get("bbox", []) or []
    if len(bbox) < 4:
        return (0.0, 0.0)
    return (float(bbox[1]), float(bbox[0]))


def _valid_block_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if not bbox or len(bbox) != 4:
        return None
    try:
        normalized = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        return None
    return normalized


def _bbox_overlaps_any(
    bbox: tuple[float, float, float, float] | None,
    occupied_bboxes: list[tuple[float, float, float, float]],
    *,
    threshold: float,
) -> bool:
    if bbox is None:
        return False
    return any(_bbox_overlap_ratio(bbox, occupied_bbox) >= threshold for occupied_bbox in occupied_bboxes)


def _bbox_overlap_ratio(
    bbox: tuple[float, float, float, float],
    other_bbox: tuple[float, float, float, float],
) -> float:
    x_overlap = max(0.0, min(float(bbox[2]), float(other_bbox[2])) - max(float(bbox[0]), float(other_bbox[0])))
    y_overlap = max(0.0, min(float(bbox[3]), float(other_bbox[3])) - max(float(bbox[1]), float(other_bbox[1])))
    if x_overlap <= 0.0 or y_overlap <= 0.0:
        return 0.0
    bbox_area = max(1.0, (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))
    return (x_overlap * y_overlap) / bbox_area
