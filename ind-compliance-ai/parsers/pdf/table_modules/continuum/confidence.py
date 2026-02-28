"""Phase 6: Confidence, Risk & Review Policy - 置信度评估

Architecture:
    All Phases → Phase 6 → Confidence Assessment

负责综合评估表格置信度：
- 综合所有阶段结果
- 风险标记
- 人工审核判定
- 表格有效性验证
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..assembly import TableInstance
    from .identity import IdentityResolution
    from .grid import GridStabilization
    from .continuity import ContinuityResult
    from .nested import NestedStructure


# Patterns (整合自 validation.py)
TOC_LINE_PATTERN = re.compile(r"[\.·…]{4,}\s*\d{1,3}\s*$")
REFERENCE_ROW_PATTERN = re.compile(r"^\s*(?:\d+\.\s+|[（(]?\d+[)）])")


@dataclass
class ConfidenceAssessment:
    """Phase 6 结果：置信度评估"""
    overall_confidence: float
    risk_flags: list[str] = field(default_factory=list)
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    is_valid_table: bool = True  # 整合自 validation.py
    toc_row_ratio: float = 0.0
    reference_like_ratio: float = 0.0


def assess_confidence(
    instance: "TableInstance",
    identity: "IdentityResolution",
    grid: "GridStabilization",
    continuity: "ContinuityResult",
    nested: "NestedStructure",
    context: dict[str, Any] | None = None,
) -> ConfidenceAssessment:
    """Phase 6: 评估置信度

    整合自原 validation.py 的验证逻辑。

    综合所有阶段结果，给出：
    - 总体置信度
    - 风险标记
    - 是否需要人工审核
    - 表格有效性判定

    Args:
        instance: 当前表格实例
        identity: Phase 1 结果
        grid: Phase 2 结果
        continuity: Phase 3 结果
        nested: Phase 5 结果
        context: 上下文信息（title_block, section_hint, toc_context, grid_line_score）

    Returns:
        ConfidenceAssessment 包含完整评估结果
    """
    risk_flags = []
    review_reasons = []
    confidence_factors = []

    # 获取上下文
    title_block = context.get("title_block") if context else None
    section_hint = context.get("section_hint") if context else None
    toc_context = context.get("toc_context", False) if context else False
    grid_line_score = context.get("grid_line_score", 0.0) if context else 0.0

    # Factor 1: 身份置信度
    confidence_factors.append(identity.confidence)
    if identity.confidence < 0.6:
        risk_flags.append("low_identity_confidence")

    # Factor 2: 网格稳定性
    confidence_factors.append(grid.structure_score)
    if not grid.stable:
        risk_flags.append("unstable_grid")
        review_reasons.append("表格结构不稳定")

    # Factor 3: 连续性
    if continuity.is_continuation:
        if continuity.similarity_score < 0.5:
            risk_flags.append("low_continuation_similarity")
            review_reasons.append("续表相似度低")
        else:
            confidence_factors.append(continuity.similarity_score)

    # Factor 4: 嵌套结构
    if nested.has_nested:
        confidence_factors.append(0.8)  # 嵌套结构增加复杂性

    # Factor 5: 单元格覆盖率
    non_null_ratio = sum(1 for c in instance.cells if c.get("text")) / max(1, len(instance.cells))
    confidence_factors.append(non_null_ratio)

    if non_null_ratio < 0.3:
        risk_flags.append("sparse_content")
        review_reasons.append("单元格内容稀疏")

    # === 整合自 validation.py ===

    # Factor 6: TOC 行比例
    toc_row_ratio = _calculate_toc_row_ratio(instance)

    # Factor 7: 引用行比例
    reference_like_ratio = _calculate_reference_like_ratio(instance)

    # 验证逻辑 (整合自 is_valid_table_candidate)
    is_valid = _validate_table_candidate(
        instance=instance,
        identity=identity,
        grid=grid,
        continuity=continuity,
        title_block=title_block,
        section_hint=section_hint,
        toc_context=toc_context,
        grid_line_score=grid_line_score,
        toc_row_ratio=toc_row_ratio,
        reference_like_ratio=reference_like_ratio,
    )

    # 计算总置信度
    overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    # TOC 过滤
    if toc_context or toc_row_ratio >= 0.38:
        is_valid = False
        risk_flags.append("toc_like_table")

    # 判定是否需要审核
    review_required = (
        len(risk_flags) >= 2 or
        overall_confidence < 0.5 or
        any(f in risk_flags for f in ["unstable_grid", "low_continuation_similarity"]) or
        not is_valid
    )

    return ConfidenceAssessment(
        overall_confidence=min(1.0, overall_confidence),
        risk_flags=risk_flags,
        review_required=review_required,
        review_reasons=review_reasons,
        is_valid_table=is_valid,
        toc_row_ratio=toc_row_ratio,
        reference_like_ratio=reference_like_ratio,
    )


def _validate_table_candidate(
    instance: "TableInstance",
    identity: "IdentityResolution",
    grid: "GridStabilization",
    continuity: "ContinuityResult",
    title_block: dict[str, Any] | None,
    section_hint: dict[str, Any] | None,
    toc_context: bool,
    grid_line_score: float,
    toc_row_ratio: float,
    reference_like_ratio: float,
) -> bool:
    """验证表格是否为有效候选

    整合自原 validation.py 的 is_valid_table_candidate()。
    """
    from .identity import TableIdentity

    score = grid.structure_score
    row_count = instance.row_count
    col_count = instance.col_count

    # Reject TOC-like tables
    if toc_context or toc_row_ratio >= 0.38:
        return False

    # Reject reference-like tables without strong signals
    if title_block is None and continuity.is_continuation is False and reference_like_ratio >= 0.75:
        return False

    if (
        title_block is None
        and continuity.is_continuation is False
        and section_hint is None
        and grid_line_score < 0.3
        and reference_like_ratio >= 0.42
    ):
        return False

    if (
        title_block is None
        and continuity.is_continuation is False
        and grid_line_score < 0.2
        and col_count >= 4
        and row_count <= 8
        and reference_like_ratio >= 0.25
    ):
        return False

    # Accept with title
    if title_block is not None:
        if col_count == 1:
            return row_count >= 4 and score >= 0.15
        return score >= 0.25 and row_count >= 2 and col_count >= 2

    # Accept continuation tables
    if continuity.is_continuation:
        if row_count >= 2 and col_count >= 2 and score >= 0.15:
            return True
        if row_count >= 3 and col_count >= 2:
            return score >= 0.12
        return row_count >= 4 and col_count >= 1 and score >= 0.15

    # Accept with section hint
    if section_hint is not None:
        section_text = str(section_hint.get("text", "")).strip().lower()
        if "术语表" in section_text or "glossary" in section_text:
            return row_count >= 4 and col_count >= 2 and score >= 0.25
        return row_count >= 4 and col_count >= 2 and score >= 0.55

    # Accept with strong grid lines
    if grid_line_score >= 0.45:
        return row_count >= 4 and col_count >= 2 and score >= 0.56

    # Accept tables marked as continuation
    if instance.is_continuation and row_count >= 2 and col_count >= 2:
        return score >= 0.15

    # Without caption/hint/grid, keep only very strong grids
    strong_multi_col = row_count >= 5 and col_count >= 3 and score >= 0.85 and toc_row_ratio <= 0.15
    strong_two_col = row_count >= 6 and col_count >= 2 and score >= 0.85 and toc_row_ratio <= 0.1
    decent_table = row_count >= 4 and col_count >= 2 and score >= 0.7 and toc_row_ratio <= 0.2

    return strong_multi_col or strong_two_col or decent_table


def _calculate_toc_row_ratio(instance: "TableInstance") -> float:
    """计算 TOC 行比例

    整合自原 validation.py 的 table_toc_row_ratio()。
    """
    if not instance.row_texts:
        return 0.0

    toc_rows = 0.0
    for row_text in instance.row_texts:
        cleaned = row_text.strip() if row_text else ""
        if not cleaned:
            continue

        # 点号引导
        if TOC_LINE_PATTERN.search(cleaned):
            toc_rows += 1.0
            continue

        # 编号格式
        tokens = cleaned.split()
        if len(tokens) >= 3:
            if re.fullmatch(r"\d+(?:\.\d+){1,4}", tokens[0]) and re.fullmatch(r"\d{1,3}", tokens[-1]):
                middle = " ".join(tokens[1:-1])
                has_textual_middle = bool(re.search(r"[A-Za-z\u4e00-\u9fff]", middle))
                has_dot_leader = bool(re.search(r"[\.·…]{2,}", middle))
                if has_textual_middle and (has_dot_leader or len(middle) >= 3):
                    toc_rows += 0.75
                    continue

        if re.search(r"[\.·…]{2,}", cleaned) and re.search(r"\d{1,3}\s*$", cleaned):
            toc_rows += 0.65

    return min(1.0, toc_rows / max(1, len(instance.row_texts)))


def _calculate_reference_like_ratio(instance: "TableInstance") -> float:
    """计算引用行比例

    整合自原 validation.py 的 table_reference_like_ratio()。
    """
    row_texts = [r for r in instance.row_texts if r and r.strip()]
    if not row_texts:
        return 0.0

    score = 0.0
    for row_text in row_texts:
        cleaned = row_text.strip()
        if not cleaned:
            continue

        row_score = 0.0
        lowered = cleaned.lower()

        # 书名引用
        if "——《" in cleaned or ("《" in cleaned and "》" in cleaned):
            row_score += 0.9

        # 编号引用
        if REFERENCE_ROW_PATTERN.match(cleaned):
            row_score += 0.75

        # 版本号
        if re.search(r"\bV\d+(?:\.\d+){1,3}\b", cleaned, re.IGNORECASE):
            row_score += 0.45

        # ICH 引用
        if "ich" in lowered:
            row_score += 0.35

        # 技术文档引用
        if re.search(r"(specification|document|technical|change request)", lowered):
            row_score += 0.25

        score += min(1.0, row_score)

    return min(1.0, score / max(1, len(row_texts)))


__all__ = [
    "TOC_LINE_PATTERN",
    "REFERENCE_ROW_PATTERN",
    "ConfidenceAssessment",
    "assess_confidence",
]
