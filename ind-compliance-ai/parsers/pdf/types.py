# Version: v1.3.1
# Optimization Summary:
# - Add continuation evidence data structures for multi-dimensional evidence collection.
# - Add StructureValidation for recording column structure inheritance and correction.
# - Support "evidence collection → evidence assessment" decision framework.
# - Add current-table local context evidence so early continuation assessment can
#   reject local new-table titles and narrative barriers before inheritance.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


PDF_PARSER_HINT = "pdf-ast-v5"


# ============================================================================
# Continuation Decision Types
# ============================================================================

class DecisionType(Enum):
    """续表判定类型
    
    明确续表关系的主要证据来源。
    """
    LAYOUT_CONTINUITY = "layout_continuity"  # 基于版式连续性
    STRUCTURE_CONSISTENCY = "structure_consistency"  # 基于结构一致性
    CONTEXT_SEMANTIC = "context_semantic"  # 基于上下文语义
    MULTI_EVIDENCE = "multi_evidence"  # 多种证据组合
    EXPLICIT_HINT = "explicit_hint"  # 明确的续表提示
    FALLBACK = "fallback"  # 降级判断


# ============================================================================
# Evidence Structures for Step1 (Evidence Collection)
# ============================================================================

@dataclass
class LayoutEvidence:
    """版式证据
    
    记录候选父表与当前表格之间的版式关系证据。
    """
    # 页序连续性
    page_sequential: bool = False  # prev_page == curr_page - 1
    page_gap: int = 0  # 页码间隔
    
    # 版式位置关系
    prev_near_bottom: bool = False  # 父表在页面底部区域
    curr_near_top: bool = False  # 当前表在页面顶部区域
    prev_bottom_ratio: float = 0.0  # 父表底部在页面中的位置比例
    curr_top_ratio: float = 0.0  # 当前表顶部在页面中的位置比例
    
    # 水平覆盖情况
    horizontal_overlap_ratio: float = 0.0  # 水平重叠比例
    left_boundary_diff: float = 0.0  # 左边界差异（点）
    right_boundary_diff: float = 0.0  # 右边界差异（点）
    width_ratio: float = 0.0  # 宽度比例（当前表宽度/父表宽度）


@dataclass
class StructureEvidence:
    """结构证据
    
    记录候选父表与当前表格之间的结构关系证据。
    """
    # 列结构
    parent_col_count: int = 0  # 父表列数
    current_physical_col_count: int = 0  # 当前表物理列数
    current_max_content_cols: int = 0  # 当前表最大内容列数
    
    # 列分布模式
    parent_column_signature: list[float] = field(default_factory=list)
    parent_column_boundaries: list[float] = field(default_factory=list)  # 列边界 x 坐标
    
    # 行模式
    parent_row_count: int = 0  # 父表行数
    current_row_count: int = 0  # 当前表行数
    
    # 表头信息
    parent_header: list[dict[str, Any]] = field(default_factory=list)
    parent_header_texts: list[str] = field(default_factory=list)  # 表头文本列表
    current_header_texts: list[str] = field(default_factory=list)
    current_has_header_row: bool = False
    header_similarity_score: float = 0.0


@dataclass
class ContextEvidence:
    """上下文证据
    
    记录候选父表与当前表格之间的上下文关系证据。
    """
    # 标题信息
    parent_title: str = ""
    current_title: str = ""
    title_available: bool = False  # 父表是否有标题
    titles_compatible: bool = False
    title_conflict: bool = False
    
    # 章节信息
    parent_section_hint: str = ""
    section_continuity: bool = False  # 章节是否连续
    
    # 续表提示
    has_continuation_hint: bool = False  # 是否有明确的续表提示词
    continuation_hint_text: str = ""  # 续表提示文本
    current_has_continuation_hint: bool = False
    current_continuation_hint_text: str = ""

    # 当前表本地上下文
    current_preceding_text: str = ""
    current_local_signal: str = "none"
    current_has_new_table_title: bool = False
    current_has_narrative_barrier: bool = False
    current_has_section_heading: bool = False


@dataclass
class EvidenceBundle:
    """证据集合
    
    Step1 输出的证据集合，包含候选父表与当前表格之间的所有维度证据。
    Step2 负责对这些证据进行整合评估。
    """
    # 候选父表基本信息
    parent_table_id: str = ""
    parent_page: int = 0
    parent_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    
    # 多维度证据
    layout: LayoutEvidence = field(default_factory=LayoutEvidence)
    structure: StructureEvidence = field(default_factory=StructureEvidence)
    context: ContextEvidence = field(default_factory=ContextEvidence)
    
    # 当前表格基本信息（用于证据评估）
    current_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    current_page: int = 0
    current_page_height: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典，便于序列化和审计"""
        return {
            "parent_table_id": self.parent_table_id,
            "parent_page": self.parent_page,
            "parent_bbox": list(self.parent_bbox),
            "layout": {
                "page_sequential": self.layout.page_sequential,
                "prev_near_bottom": self.layout.prev_near_bottom,
                "curr_near_top": self.layout.curr_near_top,
                "horizontal_overlap_ratio": round(self.layout.horizontal_overlap_ratio, 3),
                "left_boundary_diff": round(self.layout.left_boundary_diff, 2),
                "right_boundary_diff": round(self.layout.right_boundary_diff, 2),
            },
            "structure": {
                "parent_col_count": self.structure.parent_col_count,
                "current_physical_col_count": self.structure.current_physical_col_count,
                "current_max_content_cols": self.structure.current_max_content_cols,
                "current_has_header_row": self.structure.current_has_header_row,
                "header_similarity_score": round(self.structure.header_similarity_score, 3),
                "current_header_texts": self.structure.current_header_texts,
            },
            "context": {
                "parent_title": self.context.parent_title,
                "has_continuation_hint": self.context.has_continuation_hint,
                "current_title": self.context.current_title,
                "current_local_signal": self.context.current_local_signal,
                "titles_compatible": self.context.titles_compatible,
                "title_conflict": self.context.title_conflict,
            },
        }


# ============================================================================
# Assessment Structure for Step2 (Evidence Assessment)
# ============================================================================

@dataclass
class ContinuationAssessment:
    """续表评估结果
    
    Step2 输出的评估结果，包含综合评分、决策依据和判定类型。
    """
    # 是否确认为续表
    is_continuation: bool = False
    
    # 判定类型
    decision_type: DecisionType = DecisionType.FALLBACK
    
    # 综合评分
    overall_confidence: float = 0.0  # 综合置信度 0.0-1.0
    
    # 分项评分
    layout_score: float = 0.0  # 版式延续性评分
    structure_score: float = 0.0  # 结构连续性评分
    context_score: float = 0.0  # 上下文一致性评分
    
    # 决策依据
    decision_factors: list[str] = field(default_factory=list)  # 决策因素列表
    primary_evidence: str = ""  # 主要证据描述
    
    # 选中的父表信息
    selected_parent_id: str = ""
    selected_parent_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    selected_parent_col_count: int = 0
    selected_parent_header: list[dict[str, Any]] = field(default_factory=list)
    selected_parent_column_boundaries: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "is_continuation": self.is_continuation,
            "decision_type": self.decision_type.value,
            "overall_confidence": round(self.overall_confidence, 3),
            "layout_score": round(self.layout_score, 3),
            "structure_score": round(self.structure_score, 3),
            "context_score": round(self.context_score, 3),
            "decision_factors": self.decision_factors,
            "primary_evidence": self.primary_evidence,
            "selected_parent_id": self.selected_parent_id,
        }


# ============================================================================
# Structure Validation for Step3 (Structure Inheritance)
# ============================================================================

@dataclass
class StructureValidation:
    """结构校验结果
    
    Step3 规范化阶段的结构继承和修正校验结果。
    用于记录列结构继承的详细信息，便于后续阶段利用和审计。
    """
    # 列数校验
    col_count_inherited: bool = False  # 是否继承了父表列数
    col_count_match: bool = False  # 列数是否一致
    parent_col_count: int = 0  # 父表列数
    inferred_col_count: int = 0  # 推断的列数
    final_col_count: int = 0  # 最终使用的列数
    
    # 列映射校验
    column_mapping_method: str = "default"  # 列映射方法：parent_boundaries / content_clustering / default
    column_mapping_confidence: float = 0.0  # 列映射置信度 0.0-1.0
    empty_columns: list[int] = field(default_factory=list)  # 空列索引列表
    
    # 结构调整
    structure_adjusted: bool = False  # 是否进行了结构调整
    adjustment_reason: str = ""  # 调整原因
    original_physical_col_count: int = 0  # 原始物理列数
    
    # bbox 修正
    bbox_corrected: bool = False  # bbox 是否被修正
    original_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    corrected_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    
    # 校验结果
    validation_passed: bool = False  # 结构校验是否通过
    validation_warnings: list[str] = field(default_factory=list)  # 校验警告
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "col_count_inherited": self.col_count_inherited,
            "col_count_match": self.col_count_match,
            "parent_col_count": self.parent_col_count,
            "final_col_count": self.final_col_count,
            "column_mapping_method": self.column_mapping_method,
            "column_mapping_confidence": round(self.column_mapping_confidence, 3),
            "empty_columns": self.empty_columns,
            "structure_adjusted": self.structure_adjusted,
            "bbox_corrected": self.bbox_corrected,
            "validation_passed": self.validation_passed,
            "validation_warnings": self.validation_warnings,
        }


# ============================================================================
# Pipeline State
# ============================================================================

@dataclass(slots=True)
class PdfPipelineCounters:
    raw_table_candidates: int = 0
    accepted_table_candidates: int = 0
    toc_block_count: int = 0
    semantic_merge_count: int = 0
    image_text_recovered_count: int = 0
    full_page_ocr_recovered_count: int = 0
    image_text_ocr_skipped_count: int = 0
    embedded_image_table_ocr_skipped_count: int = 0
    rejected_table_candidates: int = 0
    table_fragment_merge_count: int = 0
    duplicate_image_blocks_removed: int = 0
    header_footer_filtered_count: int = 0
    table_text_suppressed_count: int = 0
    cross_page_table_links: int = 0
    region_candidate_count: int = 0
    region_ownership_decision_count: int = 0
    region_node_count: int = 0


@dataclass(slots=True)
class PdfPipelineState:
    page_payloads: list[dict[str, Any]] = field(default_factory=list)
    table_asts: list[dict[str, Any]] = field(default_factory=list)
    toc_nodes: list[dict[str, Any]] = field(default_factory=list)
    figure_nodes: list[dict[str, Any]] = field(default_factory=list)
    external_region_candidates: list[Any] = field(default_factory=list)
    region_candidates: list[dict[str, Any]] = field(default_factory=list)
    ownership_decisions: list[dict[str, Any]] = field(default_factory=list)
    region_nodes: list[dict[str, Any]] = field(default_factory=list)
    page_heights: dict[int, float] = field(default_factory=dict)
    embedded_outline_count: int = 0
    embedded_outline_depth: int = 0
    embedded_file_count: int = 0
    embedded_file_names: list[str] = field(default_factory=list)
    link_annotation_count: int = 0
    navigational_link_count: int = 0
    internal_link_count: int = 0
    external_file_link_count: int = 0
    external_uri_link_count: int = 0
    external_file_link_targets: list[str] = field(default_factory=list)
    link_annotation_xref_page_map: dict[int, int] = field(default_factory=dict)
    non_link_annotation_count: int = 0
    non_link_annotation_types: list[str] = field(default_factory=list)
    counters: PdfPipelineCounters = field(default_factory=PdfPipelineCounters)
    stage_timings: dict[str, float] = field(default_factory=dict)

