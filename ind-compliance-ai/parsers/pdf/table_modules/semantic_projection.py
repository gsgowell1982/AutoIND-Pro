from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


_GENERIC_HEADER_RE = re.compile(r"^(?:column|col)\s*\d+$", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"^[\s\d.,%+\-/*()]+$")
_SECTION_LABEL_RE = re.compile(r"[:：]\s*$")
_ARROW_RE = re.compile(r"(?:→|->|=>|➔|⇢|arrow)", re.IGNORECASE)
_MEASUREMENT_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"\d+(?:\.\d+)?"
    r"\s*(?:"
    r"(?:[μ渭碌u?]\s*)?"
    r"(?:l|g|mol|iu)"
    r"|mg|kg|ng|pg|ml|µl|μl|ul|µL|μL|uL"
    r")"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_RASTER_OR_VECTOR_SOURCES = {
    "embedded_image_ocr",
    "vector_ocr",
    "raster_table_evidence",
    "raster_line_grid",
    "raster_text_matrix",
    "raster_fused_table_region",
}

_HORIZONTAL_RULE_SOURCES = {
    "caption_anchored_horizontal_rules",
}

_TEXT_ALIGNED_SOURCES = {
    "text_aligned_borderless_grid",
}

_WORD_CLUSTER_SOURCES = {
    "word_clustering",
}

_VISUAL_STRUCTURE_SOURCES = {
    "visual_structure_grid",
    "structured_text_region",
}


def ensure_table_structure_profile(table: dict[str, Any]) -> dict[str, Any]:
    """Attach the source/border/evidence profile used for table-family routing.

    The profile is intentionally separate from `table_family`: `table_family`
    describes semantic table shape, while this profile describes the evidence
    and border model. This mirrors mature document-AI contracts that keep table
    structure evidence, merged-cell/header/footer semantics, and final rendering
    as distinct layers.
    """

    grid = _primary_grid(table)
    col_count = _col_count(table, grid)
    row_count = len(grid)
    source = str(table.get("detection_source") or table.get("detection_method") or table.get("source") or "").strip()
    evidence_summary = table.get("raw_evidence_summary") if isinstance(table.get("raw_evidence_summary"), dict) else {}
    horizontal_line_count = _safe_int(
        evidence_summary.get("horizontal_line_count", table.get("horizontal_line_count", table.get("raw_horizontal_line_count", 0)))
    )
    vertical_line_count = _safe_int(
        evidence_summary.get("vertical_line_count", table.get("vertical_line_count", table.get("raw_vertical_line_count", 0)))
    )
    drawing_count = _safe_int(
        evidence_summary.get("drawing_count", table.get("drawing_count", table.get("raw_drawing_count", 0)))
    )
    if horizontal_line_count <= 0 and vertical_line_count <= 0 and drawing_count <= 0:
        grid_line_score = float(table.get("grid_line_score", 0.0) or 0.0)
        if grid_line_score >= 0.45:
            horizontal_line_count = 2

    border_model = _classify_border_model(
        source=source,
        horizontal_line_count=horizontal_line_count,
        vertical_line_count=vertical_line_count,
        drawing_count=drawing_count,
    )
    structure_family, evidence_model = _structure_family_for_border_model(source, border_model)
    profile = {
        "version": 1,
        "source": "table_structure_profile",
        "structure_family": structure_family,
        "border_model": border_model,
        "evidence_model": evidence_model,
        "source_provider": source or "unknown",
        "row_count": row_count,
        "col_count": col_count,
        "horizontal_line_count": horizontal_line_count,
        "vertical_line_count": vertical_line_count,
        "drawing_count": drawing_count,
        "has_table_title": bool(table.get("title") or table.get("title_block") or table.get("caption_text")),
        "has_table_footer": bool(table.get("note_blocks") or table.get("footer_blocks")),
        "has_column_header": bool(table.get("header") or (grid and any(_cell_text(cell) for cell in grid[0]))),
        "has_merged_cell_evidence": bool(table.get("header_column_groups") or table.get("header_row_groups") or table.get("row_groups")),
        "has_note_references": bool(table.get("header_note_refs") or table.get("cell_note_refs")),
        "is_continuation": bool(table.get("is_continuation") or table.get("continued_from")),
    }
    table["table_structure_profile"] = profile
    return profile


def _classify_border_model(
    *,
    source: str,
    horizontal_line_count: int,
    vertical_line_count: int,
    drawing_count: int,
) -> str:
    normalized_source = source.lower()
    if normalized_source in _RASTER_OR_VECTOR_SOURCES:
        return "raster_or_vector_table"
    if normalized_source in _TEXT_ALIGNED_SOURCES:
        return "borderless_aligned"
    if normalized_source in _WORD_CLUSTER_SOURCES:
        return "borderless_overview"
    if normalized_source in _HORIZONTAL_RULE_SOURCES:
        return "horizontal_rules"
    if vertical_line_count >= 2 and horizontal_line_count >= 2:
        return "full_grid"
    if horizontal_line_count >= 2 and vertical_line_count <= 1:
        return "horizontal_rules"
    if drawing_count > 0 or horizontal_line_count > 0 or vertical_line_count > 0:
        return "sparse_rules"
    if normalized_source in _VISUAL_STRUCTURE_SOURCES:
        return "sparse_or_visual"
    return "borderless_aligned"


def _structure_family_for_border_model(source: str, border_model: str) -> tuple[str, str]:
    normalized_source = source.lower()
    if border_model == "full_grid":
        return "ruled_grid", "cell_grid_geometry"
    if border_model == "horizontal_rules":
        return "horizontal_rule_table", "row_rule_with_text_columns"
    if border_model == "sparse_rules":
        return "sparse_rule_table", "line_text_hybrid_grid"
    if border_model == "borderless_overview":
        return "clustered_text_table", "word_cluster_grid"
    if border_model == "raster_or_vector_table":
        return "image_or_vector_ocr_table", "ocr_or_vector_region"
    if border_model == "sparse_or_visual":
        return "visual_structure_table", "visual_text_region_grid"
    if normalized_source in _TEXT_ALIGNED_SOURCES:
        return "aligned_text_grid", "text_anchor_grid"
    return "aligned_text_grid", "text_anchor_grid"


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def apply_table_semantic_projection_v2(table: dict[str, Any]) -> bool:
    """Add an audit-preserving semantic projection for common structured tables.

    This layer deliberately keeps the parser's observed/raw grid intact. It
    records table-family and logical-cell semantics that customer exports and
    benchmark adapters can consume without pretending weak visual evidence is a
    stronger raw extraction.
    """
    grid = _primary_grid(table)
    if not grid:
        return False
    grid = _rectangular_grid(grid, _col_count(table, grid))
    structure_profile = ensure_table_structure_profile(table)
    family = _classify_table_family(table, grid)
    if not family:
        return False

    semantic_grid = deepcopy(grid)
    diagnostics: dict[str, Any] = {}
    if family == "two_column_inventory":
        semantic_grid, removed = _remove_adjacent_duplicate_rows(semantic_grid)
        diagnostics["duplicate_boundary_rows_removed"] = removed
        if removed:
            _rewrite_lossless_duplicate_public_grids(table)
        projected_pairs_grid, pair_projection = _project_two_column_label_after_value_pairs(
            table,
            semantic_grid,
        )
        if projected_pairs_grid:
            semantic_grid = projected_pairs_grid
            diagnostics["label_after_value_pair_projection"] = pair_projection
        compacted_parallel_grid, parallel_compaction = _compact_parallel_inventory_lists(
            table,
            semantic_grid,
        )
        if compacted_parallel_grid:
            semantic_grid = compacted_parallel_grid
            diagnostics["parallel_inventory_list_compaction"] = parallel_compaction
    elif family == "projected_stub_matrix":
        compacted_grid, compaction = _build_projected_stub_semantic_grid(table, semantic_grid)
        if compacted_grid:
            semantic_grid = compacted_grid
            diagnostics["projected_stub_compaction"] = compaction
    elif family == "keyed_long_list":
        projected_grid, list_projection = _project_single_column_keyed_long_list(table, semantic_grid)
        if projected_grid:
            semantic_grid = projected_grid
            diagnostics["single_column_keyed_list_projection"] = list_projection
    elif family == "two_column_spanning_header_table":
        projected_grid, spanning_header_projection = _project_two_column_spanning_header_grid(table, semantic_grid)
        if projected_grid:
            semantic_grid = projected_grid
            diagnostics["two_column_spanning_header_projection"] = spanning_header_projection
    elif family == "flowchart_matrix":
        projected_grid, connector_projection = _project_flowchart_connector_semantic_grid(table, semantic_grid)
        if projected_grid:
            semantic_grid = projected_grid
            diagnostics["flowchart_connector_projection"] = connector_projection
        compacted_flowchart_grid, flowchart_compaction = _compact_flowchart_matrix_wrapped_rows(table, semantic_grid, family)
        if compacted_flowchart_grid:
            semantic_grid = compacted_flowchart_grid
            diagnostics["flowchart_wrapped_row_compaction"] = flowchart_compaction
    elif family == "compressed_image_measurement_matrix":
        projected_grid, measurement_projection = _project_compressed_image_measurement_matrix(table, semantic_grid)
        if projected_grid:
            semantic_grid = projected_grid
            diagnostics["compressed_measurement_matrix_projection"] = measurement_projection
    elif family == "numeric_schedule_table":
        expanded_grid, stacked_projection = _expand_stacked_numeric_schedule_rows(table, semantic_grid)
        if expanded_grid:
            semantic_grid = expanded_grid
            diagnostics["stacked_numeric_atom_row_expansion"] = stacked_projection
    elif family == "rowspan_grouped_table":
        projected_hierarchy_grid, hierarchy_projection = _project_two_column_hierarchy_child_column_grid(
            table,
            semantic_grid,
        )
        if projected_hierarchy_grid:
            semantic_grid = projected_hierarchy_grid
            diagnostics["two_column_hierarchy_child_column_projection"] = hierarchy_projection
    header_override: list[str] | None = None
    boundary_projected_grid, boundary_header, boundary_projection = _project_leading_boundary_to_data_header(
        table,
        semantic_grid,
        family,
    )
    if boundary_projected_grid:
        semantic_grid = boundary_projected_grid
        header_override = boundary_header
        diagnostics["leading_boundary_header_projection"] = boundary_projection
    compact_header_grid, compact_header, compact_header_projection = _project_compact_single_cell_header_to_data_columns(
        table,
        semantic_grid,
        family,
    )
    if compact_header_grid:
        semantic_grid = compact_header_grid
        header_override = compact_header
        diagnostics["compact_single_cell_header_projection"] = compact_header_projection
    projected_header_grid, projected_header, header_projection = _project_multiline_schema_header_semantic_grid(
        table,
        semantic_grid,
        family,
    )
    if projected_header_grid:
        semantic_grid = projected_header_grid
        header_override = projected_header
        diagnostics["multiline_schema_header_projection"] = header_projection
    anchor_projected_grid, anchor_projected_header, anchor_projection = _project_sparse_body_anchor_columns_semantic_grid(
        table,
        semantic_grid,
        family,
    )
    if anchor_projected_grid:
        semantic_grid = anchor_projected_grid
        header_override = anchor_projected_header
        diagnostics["sparse_body_anchor_column_projection"] = anchor_projection
    early_compressed_header_groups, early_compressed_header_projection = _project_compressed_multilevel_metric_header_groups(
        table,
        semantic_grid,
        family,
        structure_profile,
    )
    if early_compressed_header_projection:
        diagnostics["compressed_multilevel_metric_header_projection"] = early_compressed_header_projection
    if not (
        "multiline_schema_header_projection" in diagnostics
        or "sparse_body_anchor_column_projection" in diagnostics
        or "leading_boundary_header_projection" in diagnostics
        or "compressed_multilevel_metric_header_projection" in diagnostics
    ):
        header_prepended_grid, header_prepended, external_header_projection = _project_external_header_body_grid(
            table,
            semantic_grid,
            family,
        )
        if header_prepended_grid:
            semantic_grid = header_prepended_grid
            header_override = header_prepended
            diagnostics["external_header_grid_projection"] = external_header_projection
    packed_marker_grid, packed_marker_header, packed_marker_projection = _project_packed_leading_stub_marker_column(
        table,
        semantic_grid,
        family,
    )
    if packed_marker_grid:
        semantic_grid = packed_marker_grid
        header_override = packed_marker_header
        diagnostics["packed_leading_stub_marker_projection"] = packed_marker_projection
    merged_stub_grid, stub_merge = (None, None)
    if "packed_leading_stub_marker_projection" not in diagnostics:
        merged_stub_grid, stub_merge = _merge_split_leading_stub_label_column(table, semantic_grid, family)
    if merged_stub_grid:
        semantic_grid = merged_stub_grid
        diagnostics["split_leading_stub_label_column_merge"] = stub_merge
    if "split_leading_stub_label_column_merge" not in diagnostics:
        missing_stub_grid, missing_stub_header, missing_stub_projection = _project_missing_leading_stub_header_cell(
            table,
            semantic_grid,
            family,
        )
        if missing_stub_grid:
            semantic_grid = missing_stub_grid
            header_override = missing_stub_header
            diagnostics["missing_leading_stub_header_projection"] = missing_stub_projection
    sparse_numeric_grid, sparse_numeric_header, sparse_numeric_projection = _project_sparse_numeric_anchor_columns(
        table,
        semantic_grid,
        family,
    )
    if sparse_numeric_grid:
        semantic_grid = sparse_numeric_grid
        header_override = sparse_numeric_header
        diagnostics["sparse_numeric_anchor_column_projection"] = sparse_numeric_projection
    compact_body_grid, compact_body_projection = _project_compact_single_cell_body_rows(
        table,
        semantic_grid,
        family,
    )
    if compact_body_grid:
        semantic_grid = compact_body_grid
        diagnostics["compact_single_cell_body_row_projection"] = compact_body_projection
    rowspan_compacted_grid, rowspan_compaction = _compact_rowspan_repeated_key_description_rows(
        table,
        semantic_grid,
        family,
    )
    if rowspan_compacted_grid:
        semantic_grid = rowspan_compacted_grid
        diagnostics["rowspan_key_repetition_compaction"] = rowspan_compaction
    compacted_wrapped_grid, wrapped_record_projection = _compact_multicolumn_wrapped_record_rows(
        table,
        semantic_grid,
        family,
        projected_header_rows=1 if "multiline_schema_header_projection" in diagnostics else None,
    )
    if compacted_wrapped_grid:
        semantic_grid = compacted_wrapped_grid
        diagnostics["multicolumn_wrapped_record_compaction"] = wrapped_record_projection
    split_numeric_grid, split_numeric_projection = _project_merged_adjacent_numeric_value_columns(
        table,
        semantic_grid,
        family,
    )
    if split_numeric_grid:
        semantic_grid = split_numeric_grid
        diagnostics["merged_adjacent_numeric_value_projection"] = split_numeric_projection
    compacted_long_description_grid, long_description_projection = _compact_keyed_long_description_rows(
        table,
        semantic_grid,
        family,
    )
    if compacted_long_description_grid:
        semantic_grid = compacted_long_description_grid
        diagnostics["keyed_long_description_row_compaction"] = long_description_projection
    projected_blank_grid, blank_projection = _project_blank_form_comparison_semantic_grid(table, semantic_grid, family)
    if projected_blank_grid:
        semantic_grid = projected_blank_grid
        diagnostics["blank_stub_column_projection"] = blank_projection
    refined_grid, boundary_refinement = _refine_semantic_row_run_boundary(table, semantic_grid, family)
    if refined_grid:
        semantic_grid = refined_grid
        diagnostics["semantic_row_run_boundary_refinement"] = boundary_refinement

    inferred_header_column_groups, inferred_header_row_groups, sparse_header_projection = _infer_sparse_header_span_groups(
        semantic_grid,
        family,
    )
    caption_filled_grid, caption_fill = _fill_sparse_multilevel_blank_header_leaf_from_context(
        table,
        semantic_grid,
        family,
        inferred_header_column_groups,
        inferred_header_row_groups,
        sparse_header_projection,
    )
    if caption_filled_grid:
        semantic_grid = caption_filled_grid
        diagnostics["caption_context_blank_header_leaf_fill"] = caption_fill
        inferred_header_column_groups, inferred_header_row_groups, sparse_header_projection = _infer_sparse_header_span_groups(
            semantic_grid,
            family,
        )
    if sparse_header_projection:
        diagnostics["sparse_header_span_projection"] = sparse_header_projection
    (
        grouped_header_override,
        grouped_header_column_groups,
        grouped_header_row_groups,
        grouped_row_groups,
        grouped_projection,
    ) = _project_grouped_multilevel_borderless_table(
        table,
        semantic_grid,
        family,
        structure_profile,
    )
    if grouped_projection:
        word_projected_grid, word_projected_groups, word_projected_rows, word_projection = (
            _project_grouped_multilevel_word_logical_grid(table, grouped_header_override, grouped_projection)
        )
        if word_projected_grid:
            semantic_grid = word_projected_grid
            grouped_header_override = [_cell_text(cell) for cell in word_projected_grid[0]]
            if word_projected_groups:
                grouped_header_column_groups = word_projected_groups
            if word_projected_rows:
                grouped_row_groups = word_projected_rows
            grouped_projection = {
                **grouped_projection,
                "projected_col_count": len(grouped_header_override),
                "word_logical_grid_projection": word_projection,
                "header_column_groups": grouped_header_column_groups,
                "row_groups": grouped_row_groups,
            }
        diagnostics["grouped_multilevel_borderless_projection"] = grouped_projection
        header_override = grouped_header_override or header_override
        if grouped_header_column_groups:
            inferred_header_column_groups = [
                *inferred_header_column_groups,
                *grouped_header_column_groups,
            ]
        if grouped_header_row_groups:
            inferred_header_row_groups = [
                *inferred_header_row_groups,
                *grouped_header_row_groups,
            ]

    compressed_header_groups = early_compressed_header_groups
    if not compressed_header_groups:
        compressed_header_groups, compressed_header_projection = _project_compressed_multilevel_metric_header_groups(
            table,
            semantic_grid,
            family,
            structure_profile,
        )
    else:
        compressed_header_projection = early_compressed_header_projection
    if compressed_header_projection:
        diagnostics["compressed_multilevel_metric_header_projection"] = compressed_header_projection
        inferred_header_column_groups = [
            *inferred_header_column_groups,
            *compressed_header_groups,
        ]

    semantic_header = _build_semantic_header(table, semantic_grid, family, header_override=header_override)
    logical_cells = _build_logical_cells(
        table,
        semantic_grid,
        include_row_groups=(
            "parallel_inventory_list_compaction" not in diagnostics
            and family not in {"flowchart_matrix"}
            and "two_column_hierarchy_child_column_projection" not in diagnostics
        ),
        extra_row_groups=(
            [
                *(
                    diagnostics.get("two_column_hierarchy_child_column_projection", {}).get("row_groups", [])
                    if isinstance(diagnostics.get("two_column_hierarchy_child_column_projection"), dict)
                    else []
                ),
                *grouped_row_groups,
            ]
            if grouped_row_groups or isinstance(diagnostics.get("two_column_hierarchy_child_column_projection"), dict)
            else None
        ),
        extra_header_column_groups=inferred_header_column_groups,
        extra_header_row_groups=inferred_header_row_groups,
    )
    if family == "two_column_spanning_header_table":
        logical_cells = _logical_cells_with_two_column_spanning_header(logical_cells, semantic_grid)
    projection = {
        "version": 2,
        "table_family": family,
        "source": "table_semantic_projection_v2",
        "structure_profile": deepcopy(structure_profile),
        **diagnostics,
    }
    if table.get("row_groups"):
        projection["row_group_count"] = len([item for item in table.get("row_groups", []) or [] if isinstance(item, dict)])
    combined_header_column_groups = _dedupe_semantic_groups(
        [
            *[dict(item) for item in table.get("header_column_groups", []) or [] if isinstance(item, dict)],
            *[dict(item) for item in inferred_header_column_groups if isinstance(item, dict)],
        ]
    )
    combined_header_row_groups = _dedupe_semantic_groups(
        [
            *[dict(item) for item in table.get("header_row_groups", []) or [] if isinstance(item, dict)],
            *[dict(item) for item in inferred_header_row_groups if isinstance(item, dict)],
        ]
    )
    if combined_header_column_groups:
        table["header_column_groups"] = combined_header_column_groups
    if combined_header_row_groups:
        table["header_row_groups"] = combined_header_row_groups

    if table.get("header_column_groups"):
        projection["header_column_group_count"] = len(
            [item for item in table.get("header_column_groups", []) or [] if isinstance(item, dict)]
        )
    if table.get("header_row_groups"):
        projection["header_row_group_count"] = len(
            [item for item in table.get("header_row_groups", []) or [] if isinstance(item, dict)]
        )

    changed = (
        table.get("table_family") != family
        or table.get("semantic_grid") != semantic_grid
        or table.get("semantic_header") != semantic_header
        or table.get("logical_cells") != logical_cells
        or table.get("semantic_projection_v2") != projection
    )
    table["table_family"] = family
    table["semantic_grid"] = semantic_grid
    table["semantic_header"] = semantic_header
    table["logical_cells"] = logical_cells
    table["semantic_projection_v2"] = projection
    if table.get("header_column_groups"):
        table["semantic_header_column_groups"] = deepcopy(table.get("header_column_groups") or [])
    if table.get("header_row_groups"):
        table["semantic_header_row_groups"] = deepcopy(table.get("header_row_groups") or [])
    return changed


def merge_semantic_table_fragments_v2(table_asts: list[dict[str, Any]]) -> int:
    """Merge adjacent fragments that share one logical table owner.

    The merge stays ownership-driven: two-column inventory fragments are joined
    through schema compatibility, and single-column tail fragments are attached
    to matrix/flowchart owners only when their text and geometry continue an
    existing owner cell.
    """
    if len(table_asts) < 2:
        return 0
    merged = 0
    index = 0
    while index < len(table_asts) - 1:
        left = table_asts[index]
        right = table_asts[index + 1]
        if _should_merge_same_row_wide_table_fragments(left, right):
            _merge_same_row_wide_table_fragment_into_left(left, right)
            del table_asts[index + 1]
            apply_table_semantic_projection_v2(left)
            merged += 1
            continue
        if _should_merge_stacked_header_body_table_fragments(left, right):
            _merge_stacked_header_body_table_fragment_into_left(left, right)
            del table_asts[index + 1]
            apply_table_semantic_projection_v2(left)
            merged += 1
            continue
        if _should_drop_header_only_fragment_before_body_table(left, right):
            _record_semantic_fragment_merge(right, left, "header_only_fragment_drop")
            del table_asts[index]
            apply_table_semantic_projection_v2(right)
            merged += 1
            continue
        if _should_merge_matrix_right_edge_tail_fragment(left, right):
            _merge_matrix_right_edge_tail_fragment_into_left(left, right)
            del table_asts[index + 1]
            apply_table_semantic_projection_v2(left)
            merged += 1
            continue
        if _should_merge_flowchart_tail_fragment(left, right):
            _merge_flowchart_tail_fragment_into_left(left, right)
            del table_asts[index + 1]
            apply_table_semantic_projection_v2(left)
            merged += 1
            continue
        if not _should_merge_two_column_inventory_fragments(left, right):
            index += 1
            continue
        _merge_table_fragment_into_left(left, right)
        del table_asts[index + 1]
        apply_table_semantic_projection_v2(left)
        merged += 1
    return merged


def _primary_grid(table: dict[str, Any]) -> list[list[Any]]:
    for key in ("display_grid", "raw_grid", "grid"):
        value = table.get(key)
        if isinstance(value, list) and value:
            return _clone_grid(value)
    data_grid = table.get("data_grid")
    if isinstance(data_grid, list) and data_grid:
        header_row = _header_texts(table)
        return [header_row, *_clone_grid(data_grid)] if header_row else _clone_grid(data_grid)
    return []


def _clone_grid(value: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in value or []:
        if isinstance(row, (list, tuple)):
            rows.append(list(row))
    return rows


def _rectangular_grid(grid: list[list[Any]], col_count: int) -> list[list[Any]]:
    if col_count <= 0:
        col_count = max((len(row) for row in grid), default=0)
    result: list[list[Any]] = []
    for row in grid:
        next_row = list(row[:col_count])
        while len(next_row) < col_count:
            next_row.append(None)
        result.append(next_row)
    return result


def _col_count(table: dict[str, Any], grid: list[list[Any]]) -> int:
    value = table.get("col_count")
    if isinstance(value, int) and value > 0:
        return value
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except Exception:
        pass
    return max((len(row) for row in grid), default=0)


def _classify_table_family(table: dict[str, Any], grid: list[list[Any]]) -> str | None:
    col_count = _col_count(table, grid)
    if col_count == 1 and _looks_like_single_column_keyed_long_list(table, grid):
        return "keyed_long_list"
    if col_count == 2 and _looks_like_compressed_image_measurement_matrix(table, grid):
        return "compressed_image_measurement_matrix"
    if col_count == 2 and _looks_like_two_column_label_after_value_pairs(table, grid):
        return "two_column_inventory"
    if col_count == 2 and _has_body_rowspan_groups(table) and _looks_like_two_column_hierarchy_rowspan_table(table, grid):
        return "rowspan_grouped_table"
    if col_count == 2 and _looks_like_blank_form_comparison_matrix(table, grid):
        return "blank_form_comparison_matrix"
    if col_count == 2 and _looks_like_two_column_spanning_header_table(table, grid):
        return "two_column_spanning_header_table"
    if col_count == 2 and _looks_like_two_column_inventory(table, grid):
        return "two_column_inventory"
    if _has_body_rowspan_groups(table) and _looks_like_attribute_rowspan_table(table, grid):
        return "rowspan_grouped_table"
    if col_count >= 3 and _looks_like_projected_stub_matrix(table, grid):
        return "projected_stub_matrix"
    if col_count >= 3 and _looks_like_flowchart_matrix(table, grid):
        return "flowchart_matrix"
    if col_count >= 3 and _looks_like_spreadsheet_matrix(table, grid):
        return "spreadsheet_matrix"
    if col_count >= 3 and _looks_like_numeric_schedule_table(table, grid):
        return "numeric_schedule_table"
    if col_count >= 3 and _looks_like_compressed_multilevel_metric_header_table(table, grid):
        return "comparison_matrix"
    if col_count >= 3 and _looks_like_compact_single_cell_header_table(table, grid):
        return "comparison_matrix"
    if col_count >= 3 and _looks_like_sparse_numeric_anchor_table(table, grid):
        return "comparison_matrix"
    if col_count >= 3 and _looks_like_dense_record_matrix(table, grid):
        return "comparison_matrix"
    if col_count >= 3 and _looks_like_comparison_matrix(table, grid):
        return "comparison_matrix"
    if _has_body_rowspan_groups(table):
        return "rowspan_grouped_table"
    if col_count >= 3 and _looks_like_clustered_listing_table(table, grid):
        return "clustered_listing_table"
    return None


def _has_body_rowspan_groups(table: dict[str, Any]) -> bool:
    for group in table.get("row_groups", []) or []:
        if not isinstance(group, dict):
            continue
        try:
            if int(group.get("rowspan", 1) or 1) > 1:
                return True
        except Exception:
            continue
    return False


def _looks_like_clustered_listing_table(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    source = str(table.get("detection_source") or table.get("detection_method") or "").strip().lower()
    if source != "word_clustering":
        return False
    col_count = _col_count(table, grid)
    if col_count < 4 or len(grid) < 8:
        return False
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0][:col_count]]
    header_tokens = {
        token
        for text in header
        for token in re.findall(r"[a-z]+", _cell_text(text).lower())
    }
    schema_terms = {
        "date",
        "content",
        "file",
        "folder",
        "document",
        "submission",
        "sequence",
        "item",
        "description",
        "destination",
        "section",
        "module",
        "link",
        "location",
        "version",
        "status",
    }
    if len(header_tokens & schema_terms) < 2:
        return False
    body_rows = _body_rows(table, grid)
    if len(body_rows) < 6:
        return False
    anchor_rows = 0
    continuation_rows = 0
    structured_values = 0
    for row in body_rows:
        non_empty_cols = [idx for idx, cell in enumerate(row[:col_count]) if _cell_text(cell)]
        if not non_empty_cols:
            continue
        if 0 in non_empty_cols:
            anchor_rows += 1
        elif non_empty_cols[0] >= 1:
            continuation_rows += 1
        for cell in row[:col_count]:
            value = _cell_text(cell)
            if _looks_like_listing_structured_value(value):
                structured_values += 1
    return anchor_rows >= 2 and continuation_rows >= 3 and structured_values >= 4


def _looks_like_compressed_multilevel_metric_header_table(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    col_count = _col_count(table, grid)
    if col_count < 5 or len(grid) < 6:
        return False
    rows = _rectangular_grid(grid, col_count)
    if not _has_compressed_header_stub_rows(rows):
        return False
    value_cols = _supported_numeric_value_columns(rows)
    if len(value_cols) < 4:
        return False
    runs = _contiguous_index_runs(value_cols)
    return any(len(run) >= 4 for run in runs)


def _looks_like_listing_structured_value(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    return bool(
        _looks_numeric(cleaned)
        or re.search(r"\b\d+(?:\.\d+)+\b", cleaned)
        or re.search(r"\b[\w.-]+\.(?:pdf|xml|xsd|dtd|docx?|xlsx?|txt|zip)\b", cleaned, re.IGNORECASE)
        or "\\" in cleaned
        or "/" in cleaned
    )


def _looks_like_single_column_keyed_long_list(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    if _col_count(table, grid) != 1 or len(grid) < 4:
        return False
    if _cell_text(table.get("title")) or _cell_text(table.get("caption_text")):
        return False
    header = _header_texts(table) or [_cell_text(grid[0][0] if grid and grid[0] else "")]
    key = _cell_text(header[0] if header else "")
    if not _looks_like_keyed_long_list_field_label(key):
        return False
    rows = _body_rows(table, grid)
    values = [_cell_text(row[0] if row else None) for row in rows]
    values = [value for value in values if value]
    if len(values) < 3:
        return False
    if any(_looks_like_table_caption_like_text(value) for value in values[:2]):
        return False
    if sum(1 for value in values if _looks_like_list_value_atom(value)) < max(3, len(values) - 1):
        return False
    if sum(1 for value in values if _looks_numeric(value)) > len(values) // 2:
        return False
    return True


def _looks_like_keyed_long_list_field_label(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned or _is_generic_header(cleaned) or _looks_numeric(cleaned):
        return False
    if _looks_like_table_caption_like_text(cleaned):
        return False
    if _row_text_looks_like_prose_sentence(cleaned):
        return False
    return len(cleaned) <= 80 and len(cleaned.split()) <= 8


def _looks_like_list_value_atom(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _row_text_looks_like_prose_sentence(cleaned):
        return False
    if len(cleaned) <= 96 and len(cleaned.split()) <= 6:
        return bool(re.search(r"[A-Za-z\u4e00-\u9fff0-9]", cleaned))
    return bool(re.fullmatch(r"[\w:./+\-]+", cleaned, re.UNICODE))


def _looks_like_two_column_inventory(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    if not _looks_like_two_column_inventory_compatible(table, grid):
        return False
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0]]
    if len(header) >= 2 and _is_generic_header(header[0]) and _is_generic_header(header[1]):
        return False
    rows = _body_rows(table, grid)
    section_labels = sum(1 for row in rows if any(_SECTION_LABEL_RE.search(_cell_text(cell)) for cell in row[:2]))
    sparse_rows = sum(1 for row in rows if _non_empty_count(row[:2]) == 1)
    text_rows = sum(1 for row in rows if _non_empty_count(row[:2]) >= 1)
    return text_rows >= 3 and (section_labels >= 1 or sparse_rows >= 1)


def _looks_like_two_column_label_after_value_pairs(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    if not _looks_like_two_column_inventory_compatible(table, grid):
        return False
    header = _semantic_two_column_inventory_header(table, grid)
    if len(header) < 2:
        return False
    rows = _body_rows(table, grid)
    if len(rows) < 4:
        return False
    observed_pairs = _two_column_label_after_value_pair_specs(rows)
    if len(observed_pairs) < 2:
        return False
    label_only_rows = 0
    value_only_rows = 0
    for row in rows:
        left = _cell_text(row[0] if len(row) > 0 else None)
        right = _cell_text(row[1] if len(row) > 1 else None)
        if left and not right and _looks_like_two_column_inventory_label(left):
            label_only_rows += 1
        elif right and not left and _looks_like_two_column_inventory_value(right):
            value_only_rows += 1
    return label_only_rows >= len(observed_pairs) and value_only_rows >= len(observed_pairs)


def _looks_like_two_column_hierarchy_rowspan_table(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    if not _looks_like_two_column_inventory_compatible(table, grid):
        return False
    body = _body_rows(table, grid)
    if len(body) < 3:
        return False
    outline_rows = 0
    parent_child_rows = 0
    for row in body:
        left = _cell_text(row[0] if len(row) > 0 else None)
        right = _cell_text(row[1] if len(row) > 1 else None)
        if left and _leading_outline_prefix(left):
            outline_rows += 1
        if left and right:
            left_prefix = _leading_outline_prefix(left)
            right_prefix = _leading_outline_prefix(right)
            if left_prefix and right_prefix and right_prefix.startswith(left_prefix + "."):
                parent_child_rows += 1
        if left and not right:
            parent_part, child_part = _split_parent_and_child_marker(left)
            if parent_part and child_part:
                parent_child_rows += 1
    if parent_child_rows >= 1 and outline_rows >= 2:
        return True
    row_group_texts = [
        _cell_text(group.get("text"))
        for group in table.get("row_groups", []) or []
        if isinstance(group, dict)
    ]
    numbered_group_count = sum(1 for text in row_group_texts if _leading_outline_prefix(text))
    return numbered_group_count >= 2 and outline_rows >= 2


def _looks_like_compressed_image_measurement_matrix(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    if _col_count(table, grid) != 2 or len(grid) < 5:
        return False
    source = str(table.get("detection_source") or table.get("detection_method") or table.get("source") or "").lower()
    if not any(token in source for token in ("ocr", "image", "visual")):
        return False
    rows = _clone_grid(grid)
    data_rows = _compressed_measurement_data_rows(rows)
    if len(data_rows) < 2:
        return False
    header_rows = rows[: data_rows[0][0]]
    if len(header_rows) < 2:
        return False
    max_values = max(len(_measurement_value_tokens(_cell_text(row[1]))) for _, row in data_rows)
    if max_values < 3:
        return False
    header_text = " ".join(_cell_text(cell) for row in header_rows for cell in row)
    if len(_compressed_header_tokens(header_text)) < max(4, max_values + 1):
        return False
    return True


def _looks_like_two_column_inventory_compatible(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    if _col_count(table, grid) != 2 or len(grid) < 2:
        return False
    rows = _body_rows(table, grid)
    if not rows:
        return False
    meaningful_rows = [row for row in rows if _non_empty_count(row[:2]) >= 1]
    if len(meaningful_rows) < 2:
        return False
    numeric_rows = sum(1 for row in meaningful_rows if all(_looks_numeric(_cell_text(cell)) for cell in row if _cell_text(cell)))
    return numeric_rows < len(meaningful_rows)


def _looks_like_blank_form_comparison_matrix(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    if _col_count(table, grid) != 2 or len(grid) < 5:
        return False
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0][:2]]
    if len(header) < 2 or any(_is_generic_header(text) for text in header[:2]):
        return False
    if not all(_looks_like_comparison_axis_header(text) for text in header[:2]):
        return False
    rows = _body_rows(table, grid)
    if len(rows) < 4:
        return False
    stub_rows = 0
    for row in rows[: min(len(rows), 12)]:
        left = _cell_text(row[0] if len(row) > 0 else None)
        right = _cell_text(row[1] if len(row) > 1 else None)
        if not left:
            continue
        if right and _row_text_looks_like_prose_sentence(f"{left} {right}"):
            break
        if _semantic_row_is_probable_narrative_tail([left, right or None, None], 3):
            break
        if right:
            continue
        if not right and _looks_like_blank_form_stub_fragment(left):
            stub_rows += 1
    return stub_rows >= 3


def _looks_like_comparison_axis_header(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned or len(cleaned) > 140:
        return False
    if _looks_numeric(cleaned) or cleaned.endswith(":"):
        return False
    if re.search(r"[.;!?。；！？]\s*$", cleaned):
        return False
    if re.match(r"^\s*(?:\d+|[A-Za-z])[\.)]\s+\S+", cleaned):
        return False
    return len(cleaned.split()) <= 16 and bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned))


def _looks_like_blank_form_stub_fragment(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if re.match(r"^\s*(?:\d+|[A-Za-z])[\.)]\s+\S+", cleaned):
        return False
    if _looks_numeric(cleaned) or _row_text_looks_like_prose_sentence(cleaned):
        return False
    return len(cleaned) <= 80 and len(cleaned.split()) <= 8


def _looks_like_two_column_spanning_header_table(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    if _col_count(table, grid) != 2 or len(grid) < 4:
        return False
    header = _header_texts(table)
    if len(header) >= 2 and _two_column_table_has_explicit_column_header_evidence(grid):
        return False
    rows = _rectangular_grid(_clone_grid(grid), 2)
    first = [_cell_text(cell) for cell in rows[0][:2]]
    if not all(first):
        return False
    combined = _join_schema_header_parts(first)
    if not _looks_like_comparison_axis_header(combined):
        return False
    if any(_looks_numeric(text) or _row_text_looks_like_prose_sentence(text) for text in first):
        return False
    body_rows = rows[1:]
    dense_body_rows = 0
    for row in body_rows[:8]:
        left = _cell_text(row[0])
        right = _cell_text(row[1])
        if not left or not right:
            continue
        if _semantic_row_is_probable_narrative_tail([left, right], 2):
            return False
        if _row_text_looks_like_prose_sentence(f"{left} {right}"):
            return False
        dense_body_rows += 1
    return dense_body_rows >= 3


def _two_column_table_has_explicit_column_header_evidence(grid: list[list[Any]]) -> bool:
    rows = _rectangular_grid(_clone_grid(grid), 2)
    if not rows:
        return False
    first = [_cell_text(cell) for cell in rows[0][:2]]
    if not all(first):
        return False
    if not all(_looks_like_comparison_axis_header(text) for text in first):
        return False
    body_rows = rows[1:]
    if any(
        bool(_cell_text(row[1])) and not _cell_text(row[0]) and _looks_like_unit_or_header_marker(_cell_text(row[1]))
        for row in body_rows[:2]
    ):
        return True
    dense_pairs = 0
    numeric_value_pairs = 0
    for row in body_rows[:8]:
        left = _cell_text(row[0])
        right = _cell_text(row[1])
        if not left or not right:
            continue
        dense_pairs += 1
        if _looks_numeric(right):
            numeric_value_pairs += 1
    return dense_pairs >= 3 and numeric_value_pairs >= max(2, dense_pairs - 1)


def _looks_like_flowchart_matrix(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    text = _grid_text(grid).lower()
    header_text = " ".join(_header_texts(table) or [_cell_text(cell) for cell in grid[0]]).lower()
    source = str(table.get("detection_source") or table.get("source") or "").lower()
    has_arrow = bool(_ARROW_RE.search(text)) or any(token in text for token in ("→", "↔", "¡ú"))
    has_flow_terms = any(
        token in header_text or token in text
        for token in (
            "gene",
            "genes",
            "protein",
            "characteristic",
            "characteristics",
            "process",
            "step",
            "result",
            "outcome",
            "阶段",
            "步骤",
            "结果",
            "过程",
        )
    )
    sparse_or_visual = "visual" in source or sum(1 for row in grid if 0 < _non_empty_count(row) < len(row)) >= 2
    return has_arrow and has_flow_terms and sparse_or_visual


def _looks_like_spreadsheet_matrix(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 4:
        return False
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    first_row = [_cell_text(cell) for cell in rows[0]]
    first_values = [text for text in first_row if text]
    if len(first_values) < max(2, col_count - 2):
        return False
    if not all(_looks_like_spreadsheet_column_label(text) for text in first_values):
        return False
    if first_values[0].upper() != "A":
        return False
    if not _looks_like_spreadsheet_row_number(_cell_text(rows[1][0]), expected=1):
        return False
    schema_values = [_cell_text(cell) for cell in rows[1][1:col_count] if _cell_text(cell)]
    if len(schema_values) < 2:
        return False
    if sum(1 for text in schema_values if not _looks_numeric(text)) < 2:
        return False
    data_rows = rows[2 : min(len(rows), 7)]
    indexed_rows = 0
    value_rows = 0
    for offset, row in enumerate(data_rows, start=2):
        if _looks_like_spreadsheet_row_number(_cell_text(row[0]), expected=offset):
            indexed_rows += 1
        if any(
            _cell_text(cell)
            and (_looks_numeric(_cell_text(cell)) or _looks_like_compact_table_value(_cell_text(cell)))
            for cell in row[1:]
        ):
            value_rows += 1
    return indexed_rows >= 2 and value_rows >= 2


def _looks_like_spreadsheet_column_label(text: str) -> bool:
    cleaned = _cell_text(text).upper()
    if not re.fullmatch(r"[A-Z]{1,3}", cleaned):
        return False
    value = 0
    for char in cleaned:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value >= 1


def _looks_like_spreadsheet_row_number(text: str, *, expected: int | None = None) -> bool:
    cleaned = _cell_text(text)
    if not re.fullmatch(r"\d{1,5}", cleaned):
        return False
    value = int(cleaned)
    if expected is not None:
        return value == expected
    return value >= 1


def _looks_like_comparison_matrix(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0]]
    if len(header) < 3:
        return False
    non_generic_headers = [text for text in header[1:] if text and not _is_generic_header(text)]
    if len(non_generic_headers) < 2:
        return False
    rows = _body_rows(table, grid)
    if len(rows) < 2:
        return False
    stub_values = [_cell_text(row[0]) for row in rows if row and _cell_text(row[0])]
    if len(stub_values) < 2:
        return False
    if sum(1 for value in stub_values if _looks_numeric(value)) > len(stub_values) // 2:
        return False
    comparable_rows = 0
    for row in rows:
        if len(row) >= 3 and _cell_text(row[0]) and _non_empty_count(row[1:]) >= 1:
            comparable_rows += 1
    if comparable_rows >= 2:
        return True
    blank_form_stub_rows = 0
    for row in rows:
        if len(row) < 2 or not _cell_text(row[0]):
            continue
        if _looks_numeric(_cell_text(row[0])):
            continue
        if _non_empty_count(row[1:]) == 0:
            blank_form_stub_rows += 1
    if blank_form_stub_rows >= 3 and len(non_generic_headers) >= 2:
        return True
    return comparable_rows >= 2


def _looks_like_numeric_schedule_table(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 4:
        return False
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0]]
    if len(header) < col_count:
        return False
    if sum(1 for text in header[:col_count] if text and not _is_generic_header(text)) < max(2, col_count - 1):
        return False
    body_rows = _body_rows(table, grid)
    if len(body_rows) < 3:
        return False
    keyed_rows = 0
    compact_rows = 0
    stacked_rows = 0
    for row in body_rows:
        cells = [_cell_text(cell) for cell in row[:col_count]]
        if not any(cells):
            continue
        first_atoms = _split_stacked_numeric_atom_text(cells[0])
        if first_atoms and all(_looks_like_compact_table_value(atom) or _looks_numeric(atom) for atom in first_atoms):
            keyed_rows += 1
        non_empty = [text for text in cells[1:] if text]
        if non_empty and all(_all_stacked_atoms_are_compact_values(text) for text in non_empty):
            compact_rows += 1
        if len(first_atoms) > 1 and any(len(_split_stacked_numeric_atom_text(text)) > 1 for text in cells[1:]):
            stacked_rows += 1
    return keyed_rows >= 3 and compact_rows >= 3 and stacked_rows >= 1


def _looks_like_compact_single_cell_header_table(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 4:
        return False
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    header_index, header_labels = _find_compact_single_cell_header_index(rows, col_count)
    if header_index is None or not header_labels:
        return False
    leading_rows = rows[:header_index]
    if leading_rows and not all(_row_is_boundary_before_compact_single_cell_header(row, col_count) for row in leading_rows):
        return False
    return _rows_support_compact_single_cell_header_data(rows[header_index + 1 :], col_count)


def _looks_like_dense_record_matrix(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 3:
        return False
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0][:col_count]]
    if len(header) < col_count:
        return False
    header_cells = [text for text in header[:col_count] if text]
    if len(header_cells) < max(2, col_count - 1):
        return False
    if any(_is_generic_header(text) or _looks_numeric(text) for text in header_cells):
        return False
    if any(_row_text_looks_like_prose_sentence(text) for text in header_cells):
        return False

    rows = _body_rows(table, grid)
    meaningful = [row[:col_count] for row in rows if _non_empty_count(row[:col_count]) > 0]
    if len(meaningful) < 2:
        return False
    dense_rows = [row for row in meaningful if _non_empty_count(row) >= max(2, col_count - 1)]
    if len(dense_rows) < 2:
        return False
    compact_or_value_rows = 0
    text_value_rows = 0
    for row in dense_rows[:8]:
        values = [_cell_text(cell) for cell in row[:col_count] if _cell_text(cell)]
        if not values:
            continue
        compact_values = sum(1 for value in values if _looks_like_compact_body_value_atom(value))
        if compact_values >= max(2, len(values) // 2):
            compact_or_value_rows += 1
        if any(
            value
            and not _looks_numeric(value)
            and not _MEASUREMENT_VALUE_RE.fullmatch(value)
            and not _row_text_looks_like_prose_sentence(value)
            for value in values[1:]
        ):
            text_value_rows += 1
    return compact_or_value_rows >= 2 and text_value_rows >= 1


def _expand_stacked_numeric_schedule_rows(
    table: dict[str, Any],
    grid: list[list[Any]],
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    col_count = _col_count(table, grid)
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    if len(rows) < 2:
        return None, None
    expanded: list[list[str | None]] = []
    changed = False
    for row_index, row in enumerate(rows):
        texts = [_cell_text(cell) for cell in row[:col_count]]
        if row_index == 0:
            expanded.append([text or None for text in texts])
            continue
        first_atoms = _split_stacked_numeric_atom_text(texts[0])
        if len(first_atoms) <= 1:
            expanded.append([text or None for text in texts])
            continue
        column_atoms = [_split_stacked_numeric_atom_text(text) for text in texts]
        if not any(len(atoms) > 1 for atoms in column_atoms[1:]):
            expanded.append([text or None for text in texts])
            continue
        if any(atoms and not all(_looks_like_compact_table_value(atom) or _looks_numeric(atom) for atom in atoms) for atoms in column_atoms):
            expanded.append([text or None for text in texts])
            continue
        span = max(len(atoms) for atoms in column_atoms)
        if span != len(first_atoms):
            expanded.append([text or None for text in texts])
            continue
        for atom_index in range(span):
            next_row: list[str | None] = []
            for atoms in column_atoms:
                if not atoms:
                    next_row.append(None)
                elif len(atoms) == 1:
                    next_row.append(atoms[0] if atom_index == 0 else None)
                else:
                    next_row.append(atoms[atom_index] if atom_index < len(atoms) else None)
            expanded.append(next_row)
        changed = True
    if not changed:
        return None, None
    return expanded, {
        "source": "stacked_numeric_atom_row_expansion",
        "source_row_count": len(rows),
        "expanded_row_count": len(expanded),
    }


def _all_stacked_atoms_are_compact_values(text: str) -> bool:
    atoms = _split_stacked_numeric_atom_text(text)
    return bool(atoms) and all(_looks_like_compact_table_value(atom) or _looks_numeric(atom) for atom in atoms)


def _split_stacked_numeric_atom_text(text: str) -> list[str]:
    cleaned = _cell_text(text)
    if not cleaned:
        return []
    if re.search(r"[A-Za-z]", cleaned.replace("Year", ""), re.IGNORECASE):
        return []
    atoms = re.findall(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\.\d+%?", cleaned)
    if not atoms:
        return []
    if _cell_text(" ".join(atoms)) != cleaned:
        return []
    return atoms


def _looks_like_projected_stub_matrix(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    source = str(table.get("detection_source") or table.get("detection_method") or table.get("source") or "").lower()
    if "visual_structure_grid" not in source:
        return False
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0]]
    if len(header) < 3:
        return False
    if any(not text or _is_generic_header(text) for text in header):
        return False
    rows = _body_rows(table, grid)
    if len(rows) < 4:
        return False
    if sum(1 for row in rows if _non_empty_count(row) >= 2) < 3:
        return False
    anchors = _projected_stub_label_anchors(table, grid)
    if not anchors:
        return False
    if not any(str(anchor.get("source") or "") != "cell_tail_label" for anchor in anchors):
        return False
    if len(rows) <= len(anchors) + 1:
        return False
    return True


def _looks_like_attribute_rowspan_table(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    header = [_normalize_text(text) for text in (_header_texts(table) or [_cell_text(cell) for cell in grid[0]])]
    if len(header) < 3:
        return False
    source = str(table.get("detection_source") or table.get("source") or "").lower()
    if "visual_structure_grid" in source:
        return False
    attribute_headers = {
        "function",
        "function name",
        "explanation",
        "description",
        "expected benefit",
        "benefit",
        "stage",
        "service stage",
        "item",
        "name",
        "type",
        "category",
        "备注",
        "说明",
        "描述",
        "名称",
        "类型",
        "阶段",
        "功能",
        "预期收益",
    }
    matched = 0
    for text in header:
        if text in attribute_headers or any(token in text for token in ("description", "explanation", "benefit")):
            matched += 1
    return matched >= max(2, len(header) // 2)


def _looks_like_sparse_numeric_anchor_table(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    header = _header_texts(table)
    col_count = _col_count(table, grid)
    if len(header) != col_count or col_count < 5:
        return False
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    data_rows = [row for row in rows if _row_looks_like_sparse_numeric_data_row(row, col_count)]
    if len(data_rows) < 3:
        return False
    anchor_cols = _sparse_numeric_anchor_columns(data_rows, col_count)
    return len(anchor_cols) >= 3 and len(anchor_cols) < col_count and 0 in anchor_cols


def _project_multiline_schema_header_semantic_grid(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, list[str] | None, dict[str, Any] | None]:
    """Collapse visual multi-row schema headers into one logical header row.

    Raw/display grids remain untouched. This semantic view is for consumers that
    need the logical table shape: wrapped header labels are joined by column and
    the body starts at the first row that behaves like data or a continuation
    row rather than another header fragment.
    """
    if family not in {"comparison_matrix", "rowspan_grouped_table", "projected_stub_matrix"}:
        return None, None, None
    existing_header = _header_texts(table)
    if existing_header and len(grid) >= 2 and _row_looks_like_data_record(grid[1], _col_count(table, grid)):
        return None, None, None
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 4:
        return None, None, None

    max_header_rows = min(6, len(grid) - 1)
    header_rows: list[list[Any]] = []
    for row_idx in range(max_header_rows):
        row = list(grid[row_idx][:col_count])
        if _row_looks_like_schema_header_fragment(row, col_count, row_idx=row_idx):
            header_rows.append(row)
            continue
        if _row_starts_schema_body(row, col_count):
            break
        break

    if len(header_rows) < 2:
        return None, None, None

    header_texts = _merge_schema_header_rows(header_rows, col_count)
    if sum(1 for text in header_texts if text) < max(2, col_count - 1):
        return None, None, None

    body_start = len(header_rows)
    if body_start >= len(grid):
        return None, None, None
    body_rows = _clone_grid(grid[body_start:])
    if len(body_rows) < 1:
        return None, None, None
    if not _schema_header_projection_has_body_evidence(body_rows, col_count):
        return None, None, None

    if existing_header:
        merged_existing: list[str] = []
        for col_idx in range(col_count):
            existing_text = existing_header[col_idx] if col_idx < len(existing_header) else ""
            projected_text = header_texts[col_idx]
            if existing_text and projected_text:
                merged_existing.append(_join_schema_header_parts([existing_text, projected_text]))
            else:
                merged_existing.append(projected_text or existing_text)
        header_texts = [
            merged if len(_normalize_text(merged)) >= len(_normalize_text(header_texts[idx])) else header_texts[idx]
            for idx, merged in enumerate(merged_existing)
        ]

    semantic_grid: list[list[str | None]] = [[text or None for text in header_texts]]
    semantic_grid.extend(_clone_grid(body_rows))
    return semantic_grid, header_texts, {
        "header_row_count": len(header_rows),
        "body_start_row": body_start,
        "source": "multiline_schema_header_semantic_projection",
    }


def _project_single_column_keyed_long_list(
    table: dict[str, Any],
    grid: list[list[Any]],
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    if _col_count(table, grid) != 1:
        return None, None
    header = _header_texts(table) or [_cell_text(grid[0][0] if grid and grid[0] else "")]
    key = _cell_text(header[0] if header else "")
    if not _looks_like_keyed_long_list_field_label(key):
        return None, None
    rows = _body_rows(table, grid)
    values = [_cell_text(row[0] if row else None) for row in rows]
    values = [value for value in values if value]
    if len(values) < 3:
        return None, None
    if sum(1 for value in values if _looks_like_list_value_atom(value)) < max(3, len(values) - 1):
        return None, None
    return [
        ["Field", "Value"],
        [key, _join_wrapped_description_text("", " ".join(values))],
    ], {
        "source": "single_column_keyed_long_list_projection",
        "source_row_count": len(grid),
        "value_count": len(values),
    }


def _project_sparse_body_anchor_columns_semantic_grid(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, list[str] | None, dict[str, Any] | None]:
    """Collapse false physical columns created by wrapped schema headers.

    Some ruled and borderless tables expose body values only under a subset of
    the extracted columns. The intervening columns are header wrap fragments,
    not independent data fields. The semantic grid therefore follows the body
    anchor columns and carries the header text from the whole visual band into
    those anchors.
    """
    if family not in {"comparison_matrix", "rowspan_grouped_table", "projected_stub_matrix"}:
        return None, None, None
    col_count = _col_count(table, grid)
    if col_count < 4 or len(grid) < 4:
        return None, None, None
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    header_row_count = _sparse_anchor_header_row_count(rows, col_count)
    if header_row_count < 1 or header_row_count >= len(rows) - 1:
        return None, None, None
    if _sparse_anchor_header_rows_are_really_body_data(table, rows[:header_row_count], col_count):
        return None, None, None
    body_rows = rows[header_row_count:]
    anchor_cols = _sparse_body_anchor_columns(body_rows, col_count)
    if len(anchor_cols) < 3 or len(anchor_cols) >= col_count:
        return None, None, None
    if 0 not in anchor_cols:
        return None, None, None
    if anchor_cols[-1] != col_count - 1:
        return None, None, None
    if not _sparse_anchor_columns_have_compact_body_values(body_rows, anchor_cols):
        return None, None, None

    header_rows = rows[:header_row_count]
    groups: list[tuple[int, int]] = []
    previous = -1
    for anchor in anchor_cols:
        groups.append((previous + 1, anchor))
        previous = anchor

    header: list[str] = []
    for start, end in groups:
        parts: list[str] = []
        for header_row in header_rows:
            for col_idx in range(start, end + 1):
                text = _cell_text(header_row[col_idx] if col_idx < len(header_row) else None)
                if text:
                    parts.append(text)
        header.append(_join_schema_header_parts(parts))
    if sum(1 for text in header if text) < len(anchor_cols):
        return None, None, None

    projected: list[list[str | None]] = [[text or None for text in header]]
    changed = False
    for body_row in body_rows:
        projected_row: list[str | None] = []
        for start, end in groups:
            anchor_text = _cell_text(body_row[end] if end < len(body_row) else None)
            group_texts = [
                _cell_text(body_row[col_idx] if col_idx < len(body_row) else None)
                for col_idx in range(start, end + 1)
            ]
            group_texts = [text for text in group_texts if text]
            value = anchor_text or _join_schema_header_parts(group_texts)
            if len(group_texts) > 1 or (group_texts and not anchor_text):
                changed = True
            projected_row.append(value or None)
        projected.append(projected_row)
    if not changed and len(anchor_cols) == col_count:
        return None, None, None
    return projected, header, {
        "source": "sparse_body_anchor_column_semantic_projection",
        "header_row_count": header_row_count,
        "source_col_count": col_count,
        "projected_col_count": len(anchor_cols),
        "anchor_columns": anchor_cols,
    }


def _sparse_anchor_header_rows_are_really_body_data(
    table: dict[str, Any],
    header_rows: list[list[Any]],
    col_count: int,
) -> bool:
    """Avoid treating the first data row as a wrapped visual header band."""
    if not header_rows:
        return False
    header = _header_texts(table)
    if len(header) != col_count:
        return False
    if _header_metadata_looks_like_data_row(header):
        return False
    return all(_row_looks_like_sparse_numeric_data_row(row, col_count) for row in header_rows)


def _project_external_header_body_grid(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, list[str] | None, dict[str, Any] | None]:
    """Prepend trusted header metadata when the observed grid begins at data."""
    if family not in {"comparison_matrix", "rowspan_grouped_table", "numeric_schedule_table"}:
        return None, None, None
    header = _header_texts(table)
    if len(header) < 3:
        return None, None, None
    if _header_metadata_looks_like_data_row(header):
        return None, None, None
    col_count = _col_count(table, grid)
    if col_count < 3 or len(header) != col_count:
        return None, None, None
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    if not rows:
        return None, None, None
    if _should_defer_external_header_to_sparse_numeric_projection(rows, col_count):
        return None, None, None
    first_row_texts = [_cell_text(cell) for cell in rows[0][:col_count]]
    if [_normalize_text(text) for text in first_row_texts] == [_normalize_text(text) for text in header]:
        return None, None, None
    if _row_looks_like_logical_header(rows[0], col_count) and not _row_looks_like_data_record(rows[0], col_count):
        return None, None, None
    body_start = 0
    search_rows = rows[: min(4, len(rows))]
    for idx, row in enumerate(search_rows):
        if _row_overlaps_header_text(row, header, col_count) and not _row_has_numeric_value_atom(row, col_count):
            continue
        if _row_looks_like_data_record(row, col_count):
            body_start = idx
            break
    else:
        body_start = 0
    body_rows = rows[body_start:]
    if body_start > 0 and not all(_row_overlaps_header_text(row, header, col_count) for row in rows[:body_start]):
        return None, None, None
    body_like = sum(1 for row in body_rows[: min(5, len(body_rows))] if _row_looks_like_data_record(row, col_count))
    min_body_records = 1 if _has_external_table_title_or_note_context(table) else min(2, len(body_rows))
    if body_like < min_body_records:
        return None, None, None
    projected: list[list[str | None]] = [[text or None for text in header]]
    projected.extend([[_cell_text(cell) or None for cell in row[:col_count]] for row in body_rows])
    return projected, header, {
        "source": "external_header_body_grid_projection",
        "header_col_count": len(header),
        "dropped_leading_header_fragment_count": body_start,
        "body_row_count": len(body_rows),
    }


def _should_defer_external_header_to_sparse_numeric_projection(rows: list[list[Any]], col_count: int) -> bool:
    """Keep external one-to-one headers from stealing sparse spacer-column tables."""
    if col_count < 5 or len(rows) < 4:
        return False
    data_rows: list[list[Any]] = []
    tail_seen = False
    for row in rows:
        if not tail_seen and _row_looks_like_sparse_numeric_data_row(row, col_count):
            data_rows.append(row)
            continue
        if data_rows:
            tail_seen = True
    if len(data_rows) < 3:
        return False
    anchor_cols = _sparse_numeric_anchor_columns(data_rows, col_count)
    return len(anchor_cols) >= 3 and len(anchor_cols) < col_count and 0 in anchor_cols


def _has_external_table_title_or_note_context(table: dict[str, Any]) -> bool:
    if str(table.get("title") or table.get("caption_text") or "").strip():
        return True
    title_block = table.get("title_block") if isinstance(table.get("title_block"), dict) else {}
    if str(title_block.get("text") or "").strip():
        return True
    return bool(table.get("note_blocks") or table.get("footer_blocks"))


def _row_looks_like_data_record(row: list[Any], col_count: int) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [text for text in texts if text]
    if len(filled) < max(2, col_count // 2):
        return False
    numeric_like = sum(
        1
        for text in filled
        if _looks_like_compact_table_value(text) or _looks_numeric(text) or _looks_like_atomic_numeric_value(text)
    )
    if numeric_like >= max(1, len(filled) - 1):
        return True
    first = texts[0] if texts else ""
    if first and (_looks_numeric(first) or _looks_like_compact_key_cell(first)):
        right_numeric = sum(
            1
            for text in texts[1:]
            if _looks_like_compact_table_value(text) or _looks_numeric(text) or _looks_like_atomic_numeric_value(text)
        )
        return right_numeric >= max(1, (len(filled) - 1) // 2)
    return False


def _header_metadata_looks_like_data_row(header: list[str]) -> bool:
    if len(header) < 3:
        return False
    numeric_like = sum(
        1
        for text in header
        if _looks_like_atomic_numeric_value(text) or len(_split_numeric_value_atoms(text)) >= 2
    )
    non_numeric = [text for text in header if text and not (_looks_like_atomic_numeric_value(text) or len(_split_numeric_value_atoms(text)) >= 2)]
    if numeric_like >= 1 and len(non_numeric) >= 1:
        if any(_looks_like_compact_key_cell(text) for text in non_numeric):
            return True
    return False


def _row_overlaps_header_text(row: list[Any], header: list[str], col_count: int) -> bool:
    header_tokens = {
        token
        for text in header
        for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", _normalize_text(text))
        if token
    }
    if not header_tokens:
        return False
    row_tokens = {
        token
        for cell in row[:col_count]
        for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", _normalize_text(_cell_text(cell)))
        if token
    }
    return len(row_tokens & header_tokens) >= max(1, min(2, len(row_tokens)))


def _row_has_numeric_value_atom(row: list[Any], col_count: int) -> bool:
    return any(_looks_like_atomic_numeric_value(_cell_text(cell)) for cell in row[:col_count])


def _sparse_anchor_header_row_count(rows: list[list[Any]], col_count: int) -> int:
    max_header_rows = min(4, len(rows) - 1)
    for idx in range(1, max_header_rows + 1):
        body = rows[idx:]
        if not body:
            break
        first_body = body[0][:col_count]
        if _row_starts_sparse_anchor_body(first_body, col_count):
            return idx
    return 0


def _row_starts_sparse_anchor_body(row: list[Any], col_count: int) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [(idx, text) for idx, text in enumerate(texts) if text]
    if len(filled) < 2:
        return False
    compact_values = sum(1 for _idx, text in filled if _looks_like_compact_table_value(text) or _looks_numeric(text))
    if compact_values >= max(1, len(filled) - 1):
        return True
    first_idx, first_text = filled[0]
    return first_idx == 0 and (_looks_numeric(first_text) or re.fullmatch(r"\d{4}", first_text)) and compact_values >= 1


def _sparse_body_anchor_columns(body_rows: list[list[Any]], col_count: int) -> list[int]:
    meaningful = [row[:col_count] for row in body_rows if _non_empty_count(row[:col_count]) >= 2]
    if len(meaningful) < 2:
        return []
    scores: list[tuple[int, int]] = []
    for col_idx in range(col_count):
        count = 0
        for row in meaningful:
            text = _cell_text(row[col_idx] if col_idx < len(row) else None)
            if text and (_looks_like_compact_table_value(text) or _looks_numeric(text) or col_idx == 0):
                count += 1
        scores.append((col_idx, count))
    min_count = max(2, int(len(meaningful) * 0.55))
    return [col_idx for col_idx, count in scores if count >= min_count]


def _sparse_anchor_columns_have_compact_body_values(body_rows: list[list[Any]], anchor_cols: list[int]) -> bool:
    checked = 0
    compact = 0
    for row in body_rows:
        values = [_cell_text(row[col] if col < len(row) else None) for col in anchor_cols]
        values = [value for value in values if value]
        if len(values) < 2:
            continue
        checked += 1
        if sum(1 for value in values[1:] if _looks_like_compact_table_value(value) or _looks_numeric(value)) >= max(1, len(values) - 1):
            compact += 1
    return checked >= 2 and compact >= max(2, int(checked * 0.6))


def _project_sparse_numeric_anchor_columns(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, list[str] | None, dict[str, Any] | None]:
    """Collapse empty spacer columns between stable numeric value anchors."""
    if family not in {"comparison_matrix", "rowspan_grouped_table"}:
        return None, None, None
    col_count = _col_count(table, grid)
    if col_count < 5 or len(grid) < 4:
        return None, None, None
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    header = _header_texts(table)
    header_rows = 0
    if rows and _row_looks_like_logical_header(rows[0], col_count):
        header_rows = 1
        if (
            _sparse_anchor_header_rows_are_really_body_data(table, rows[:1], col_count)
            or (
                _header_metadata_looks_like_data_row(header)
                and _sparse_numeric_external_header_candidates(table)
                and _row_looks_like_sparse_numeric_data_row(rows[0], col_count)
            )
        ):
            header_rows = 0
    body_start = header_rows
    data_rows: list[list[Any]] = []
    tail_rows: list[list[Any]] = []
    tail_started = False
    for row in rows[body_start:]:
        if not tail_started and _row_looks_like_sparse_numeric_data_row(row, col_count):
            data_rows.append(row)
            continue
        tail_started = True
        tail_rows.append(row)
    if len(data_rows) < 3 or not tail_rows:
        return None, None, None

    external_header = _sparse_numeric_external_header_candidates(table)
    external_groups = _sparse_numeric_external_header_column_groups(
        data_rows,
        col_count,
        logical_col_count=len(external_header),
    ) if external_header else None
    if external_groups:
        projected: list[list[str | None]] = [[text or None for text in external_header]]
        for row in data_rows:
            projected.append([
                _sparse_numeric_group_value(row, group) or None
                for group in external_groups
            ])
        return projected, external_header, {
            "source": "sparse_numeric_anchor_column_projection",
            "source_col_count": col_count,
            "projected_col_count": len(external_groups),
            "anchor_columns": [group[-1] for group in external_groups],
            "column_groups": external_groups,
            "header_source": "header_candidates",
            "released_tail_row_count": len([row for row in tail_rows if any(_cell_text(cell) for cell in row)]),
        }

    anchor_cols = _sparse_numeric_anchor_columns(data_rows, col_count)
    if len(anchor_cols) < 3 or len(anchor_cols) >= col_count:
        return None, None, None
    if 0 not in anchor_cols:
        return None, None, None

    projected_header = _project_sparse_numeric_header(header, rows[:header_rows], anchor_cols, col_count)
    if len(projected_header) != len(anchor_cols) or sum(1 for text in projected_header if text) < len(anchor_cols) - 1:
        return None, None, None
    projected: list[list[str | None]] = [[text or None for text in projected_header]]
    for row in data_rows:
        projected.append([_cell_text(row[col]) or None for col in anchor_cols])
    return projected, projected_header, {
        "source": "sparse_numeric_anchor_column_projection",
        "source_col_count": col_count,
        "projected_col_count": len(anchor_cols),
        "anchor_columns": anchor_cols,
        "released_tail_row_count": len([row for row in tail_rows if any(_cell_text(cell) for cell in row)]),
    }


def _sparse_numeric_external_header_candidates(table: dict[str, Any]) -> list[str]:
    candidates = [
        _cell_text(text)
        for text in (table.get("header_candidates") or [])
        if _cell_text(text)
    ]
    if len(candidates) < 3:
        return []
    if _header_metadata_looks_like_data_row(candidates):
        return []
    alpha_like = sum(1 for text in candidates if re.search(r"[A-Za-z\u4e00-\u9fff]", text))
    numeric_like = sum(1 for text in candidates if _looks_like_atomic_numeric_value(text) or _looks_numeric(text))
    if alpha_like < max(2, len(candidates) - 1):
        return []
    if numeric_like > max(0, len(candidates) // 3):
        return []
    return candidates


def _sparse_numeric_external_header_column_groups(
    data_rows: list[list[Any]],
    col_count: int,
    *,
    logical_col_count: int,
) -> list[list[int]] | None:
    if logical_col_count < 3 or col_count <= logical_col_count:
        return None
    numeric_cols: list[int] = []
    for col_idx in range(col_count):
        count = 0
        for row in data_rows:
            text = _cell_text(row[col_idx] if col_idx < len(row) else None)
            if not text:
                continue
            if col_idx == 0 or _looks_like_atomic_numeric_value(text) or _looks_like_compact_table_value(text) or _looks_numeric(text):
                count += 1
        if count:
            numeric_cols.append(col_idx)
    if not numeric_cols or numeric_cols[0] != 0:
        return None

    value_groups: list[list[int]] = []
    for col_idx in numeric_cols[1:]:
        if not value_groups:
            value_groups.append([col_idx])
            continue
        previous_group = value_groups[-1]
        adjacent = col_idx == previous_group[-1] + 1
        mutually_exclusive = all(
            not _sparse_numeric_row_has_any_value(row, previous_group)
            or not _cell_text(row[col_idx] if col_idx < len(row) else None)
            for row in data_rows
        )
        if adjacent and mutually_exclusive:
            previous_group.append(col_idx)
        else:
            value_groups.append([col_idx])

    groups = [[0]] + value_groups
    if len(groups) != logical_col_count:
        return None
    populated_rows = 0
    for row in data_rows:
        populated = sum(1 for group in groups if _sparse_numeric_group_value(row, group))
        if populated >= max(3, logical_col_count - 1):
            populated_rows += 1
    if populated_rows < max(3, int(len(data_rows) * 0.7)):
        return None
    return groups


def _sparse_numeric_row_has_any_value(row: list[Any], columns: list[int]) -> bool:
    return any(_cell_text(row[col_idx] if col_idx < len(row) else None) for col_idx in columns)


def _sparse_numeric_group_value(row: list[Any], columns: list[int]) -> str:
    values = [
        _cell_text(row[col_idx] if col_idx < len(row) else None)
        for col_idx in columns
    ]
    values = [value for value in values if value]
    if not values:
        return ""
    return _join_schema_header_parts(values)


def _row_looks_like_sparse_numeric_data_row(row: list[Any], col_count: int) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [(idx, text) for idx, text in enumerate(texts) if text]
    if len(filled) < 3:
        return False
    if not (_looks_numeric(filled[0][1]) or _looks_like_atomic_numeric_value(filled[0][1])):
        return False
    numeric_like = sum(
        1
        for _idx, text in filled[1:]
        if _looks_like_atomic_numeric_value(text) or _looks_like_compact_table_value(text) or _looks_numeric(text)
    )
    return numeric_like >= max(2, len(filled) - 1)


def _sparse_numeric_anchor_columns(data_rows: list[list[Any]], col_count: int) -> list[int]:
    counts = [0] * col_count
    for row in data_rows:
        for idx in range(col_count):
            text = _cell_text(row[idx] if idx < len(row) else None)
            if idx == 0:
                if text:
                    counts[idx] += 1
            elif _looks_like_atomic_numeric_value(text) or _looks_like_compact_table_value(text) or _looks_numeric(text):
                counts[idx] += 1
    required = max(2, int(len(data_rows) * 0.7))
    return [idx for idx, count in enumerate(counts) if count >= required]


def _project_sparse_numeric_header(
    header: list[str],
    header_rows: list[list[Any]],
    anchor_cols: list[int],
    col_count: int,
) -> list[str]:
    source_rows: list[list[Any]] = []
    if len(header) == col_count:
        source_rows.append(header)
    source_rows.extend(header_rows)
    result: list[str] = []
    previous = -1
    for anchor in anchor_cols:
        parts: list[str] = []
        start = previous + 1
        for row in source_rows:
            for idx in range(start, anchor + 1):
                text = _cell_text(row[idx] if idx < len(row) else None)
                if text:
                    parts.append(text)
        result.append(_join_schema_header_parts(parts))
        previous = anchor
    return result


def _project_packed_leading_stub_marker_column(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, list[str] | None, dict[str, Any] | None]:
    if family not in {"comparison_matrix", "rowspan_grouped_table", "projected_stub_matrix"}:
        return None, None, None
    col_count = _col_count(table, grid)
    if col_count < 5 or len(grid) < 4:
        return None, None, None
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    header_row = rows[0]
    if not (_is_generic_header(_cell_text(header_row[0])) or not _cell_text(header_row[0])):
        return None, None, None

    stub_header, first_value_header = _split_packed_stub_header_and_value(_cell_text(header_row[1]))
    if not (stub_header and first_value_header):
        return None, None, None

    body_rows = [row for row in rows[1:] if any(_cell_text(cell) for cell in row[:col_count])]
    if len(body_rows) < 3:
        return None, None, None
    marker_rows = 0
    packed_first_rows = 0
    for row in body_rows:
        left = _cell_text(row[0])
        second = _cell_text(row[1])
        if left and _looks_like_boolean_marker_value(second):
            marker_rows += 1
            continue
        packed_label, packed_marker = _split_trailing_boolean_marker(second)
        if not left and packed_label and packed_marker:
            marker_rows += 1
            packed_first_rows += 1
    if marker_rows < max(3, len(body_rows) - 1) or packed_first_rows < 1:
        return None, None, None

    header = [stub_header, first_value_header, *[_cell_text(cell) for cell in header_row[2:col_count]]]
    header = _repair_header_continuation_fragments_in_packed_marker_projection(header)
    projected: list[list[str | None]] = [[text or None for text in header]]
    split_body_rows = 0
    for row in body_rows:
        left = _cell_text(row[0])
        second = _cell_text(row[1])
        if not left:
            packed_label, packed_marker = _split_trailing_boolean_marker(second)
            if packed_label and packed_marker:
                projected.append([packed_label, packed_marker, *[_cell_text(cell) or None for cell in row[2:col_count]]])
                split_body_rows += 1
                continue
        projected.append([left or None, second or None, *[_cell_text(cell) or None for cell in row[2:col_count]]])

    if split_body_rows < 1:
        return None, None, None
    return projected, header, {
        "source": "packed_leading_stub_marker_semantic_projection",
        "header_source_col": 2,
        "body_marker_col": 2,
        "split_body_row_count": split_body_rows,
        "marker_row_count": marker_rows,
    }


def _split_packed_stub_header_and_value(text: str) -> tuple[str, str]:
    cleaned = _cell_text(text)
    if not cleaned:
        return "", ""
    parts = cleaned.split(maxsplit=1)
    if len(parts) != 2:
        return "", ""
    left, right = parts[0].strip(), parts[1].strip()
    if not left or not right:
        return "", ""
    if _looks_numeric(left) or _looks_like_boolean_marker_value(left):
        return "", ""
    if not _looks_like_comparison_axis_header(left):
        return "", ""
    if not _looks_like_comparison_axis_header(right):
        return "", ""
    return left, right


def _split_trailing_boolean_marker(text: str) -> tuple[str, str]:
    cleaned = _cell_text(text)
    if not cleaned:
        return "", ""
    parts = cleaned.rsplit(maxsplit=1)
    if len(parts) != 2:
        return "", ""
    label, marker = parts[0].strip(), parts[1].strip()
    if not label or not marker:
        return "", ""
    if _looks_numeric(label) or _row_text_looks_like_prose_sentence(label):
        return "", ""
    if not _looks_like_boolean_marker_value(marker):
        return "", ""
    return label, marker


def _looks_like_boolean_marker_value(text: str | None) -> bool:
    cleaned = _cell_text(text).strip()
    if not cleaned:
        return False
    normalized = cleaned.lower()
    if normalized in {"o", "x", "y", "n", "yes", "no", "-", "--"}:
        return True
    return bool(re.fullmatch(r"[✓✗✔✘鉁撯湕鉁斺湗閴?]", cleaned))


def _repair_header_continuation_fragments_in_packed_marker_projection(header: list[str]) -> list[str]:
    repaired = list(header)
    for idx in range(2, len(repaired) - 1):
        current = _cell_text(repaired[idx])
        following = _cell_text(repaired[idx + 1])
        if not current or not following:
            continue
        if "(" in current or ")" in current:
            continue
        parts = current.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue
        prefix, tail = parts[0].strip(), parts[1].strip()
        if not prefix or not re.fullmatch(r"[A-Z][A-Za-z]{2,}\.", tail):
            continue
        if not re.search(r"[A-Za-z]", following) or _looks_like_boolean_marker_value(following):
            continue
        repaired[idx] = prefix
        repaired[idx + 1] = _join_schema_header_parts([tail, following])
        break
    return repaired


def _merge_split_leading_stub_label_column(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    """Merge a spurious second physical column that only completes row labels.

    Text-aligned extraction can create an extra anchor inside the leading stub
    column when labels contain spaces. The raw/display grids stay unchanged;
    this only projects the logical table view used by AST consumers.
    """
    if family not in {"comparison_matrix", "rowspan_grouped_table", "projected_stub_matrix"}:
        return None, None
    col_count = _col_count(table, grid)
    if col_count < 4 or len(grid) < 4:
        return None, None

    rows = _rectangular_grid(_clone_grid(grid), col_count)
    first_row = rows[0] if rows else []
    if (
        len(first_row) > 1
        and _cell_text(first_row[0])
        and _cell_text(first_row[1])
        and _looks_like_independent_second_column_header(first_row)
    ):
        return None, None

    mergeable_indices: list[int] = []
    value_rich_indices: list[int] = []
    for row_idx, row in enumerate(rows):
        left = _cell_text(row[0] if len(row) > 0 else None)
        second = _cell_text(row[1] if len(row) > 1 else None)
        right_values = [_cell_text(cell) for cell in row[2:]]
        right_non_empty = [text for text in right_values if text]
        compact_values = sum(1 for text in right_non_empty if _looks_like_compact_table_value(text))
        value_rich = len(right_non_empty) >= max(2, col_count // 3) and compact_values >= max(2, len(right_non_empty) // 2)
        if value_rich:
            value_rich_indices.append(row_idx)
        if not (left and second and right_non_empty and value_rich):
            continue
        if not (_looks_like_stub_label_fragment(left) and _looks_like_stub_label_fragment(second)):
            continue
        mergeable_indices.append(row_idx)

    if len(mergeable_indices) < 2:
        return None, None
    if len(mergeable_indices) < max(2, len(value_rich_indices) // 2):
        return None, None

    multilevel_projected, multilevel_projection = _project_split_stub_with_preserved_multilevel_header_leaf(
        rows,
        col_count,
        mergeable_indices,
    )
    if multilevel_projected:
        return multilevel_projected, multilevel_projection

    projected: list[list[str | None]] = []
    for row_idx, row in enumerate(rows):
        next_row: list[str | None] = []
        left = _cell_text(row[0] if len(row) > 0 else None)
        second = _cell_text(row[1] if len(row) > 1 else None)
        if row_idx in mergeable_indices:
            next_row.append(_join_schema_header_parts([left, second]) or None)
        else:
            next_row.append(left or None)
        for cell in row[2:]:
            next_row.append(_cell_text(cell) or None)
        projected.append(next_row)

    return projected, {
        "source": "split_leading_stub_label_column_semantic_projection",
        "removed_physical_col": 2,
        "merged_row_count": len(mergeable_indices),
        "candidate_row_indices": mergeable_indices,
    }


def _project_split_stub_with_preserved_multilevel_header_leaf(
    rows: list[list[Any]],
    col_count: int,
    mergeable_indices: list[int],
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    header_depth = _sparse_header_band_depth(rows, col_count)
    if header_depth < 3 or header_depth >= len(rows):
        return None, None
    leaf_row_idx = header_depth - 1
    if any(row_idx < header_depth for row_idx in mergeable_indices):
        return None, None
    leaf_row = rows[leaf_row_idx]
    if _cell_text(leaf_row[0]):
        return None, None
    if not _cell_text(leaf_row[1]):
        return None, None
    if _cell_text(leaf_row[col_count - 1]):
        return None, None
    leaf_values = [
        _cell_text(leaf_row[col_idx])
        for col_idx in range(1, col_count - 1)
        if _cell_text(leaf_row[col_idx])
    ]
    if len(leaf_values) < max(2, (col_count - 2) // 2):
        return None, None

    projected: list[list[str | None]] = []
    for row_idx, row in enumerate(rows):
        if row_idx < leaf_row_idx:
            projected.append([_cell_text(row[0]) or None, *[_cell_text(cell) or None for cell in row[2:]]])
            continue
        if row_idx == leaf_row_idx:
            projected.append([_cell_text(row[0]) or None, *[_cell_text(row[col_idx]) or None for col_idx in range(1, col_count - 1)]])
            continue

        left = _cell_text(row[0] if len(row) > 0 else None)
        second = _cell_text(row[1] if len(row) > 1 else None)
        next_row: list[str | None] = []
        if row_idx in mergeable_indices:
            next_row.append(_join_schema_header_parts([left, second]) or None)
        else:
            next_row.append(left or None)
        for cell in row[2:]:
            next_row.append(_cell_text(cell) or None)
        projected.append(next_row)

    return projected, {
        "source": "split_leading_stub_label_column_semantic_projection",
        "removed_physical_col": 2,
        "merged_row_count": len(mergeable_indices),
        "candidate_row_indices": mergeable_indices,
        "preserved_multilevel_header_leaf": True,
        "header_row_count": header_depth,
    }


def _looks_like_stub_label_fragment(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_numeric(cleaned) or _is_generic_header(cleaned):
        return False
    if _looks_like_boolean_marker_value(cleaned):
        return False
    if re.search(r"\d", cleaned) and _looks_like_compact_table_value(cleaned):
        return False
    if _row_text_looks_like_prose_sentence(cleaned):
        return False
    return len(cleaned) <= 40 and len(cleaned.split()) <= 4 and bool(re.search(r"[A-Za-z\u4e00-\u9fff#]", cleaned))


def _looks_like_independent_second_column_header(row: list[Any]) -> bool:
    first = _cell_text(row[0] if len(row) > 0 else None)
    second = _cell_text(row[1] if len(row) > 1 else None)
    if not (_looks_like_stub_label_fragment(first) and _looks_like_stub_label_fragment(second)):
        return False
    following = [_cell_text(cell) for cell in row[2:] if _cell_text(cell)]
    if len(following) < 2:
        return False
    if sum(1 for text in following if _looks_like_compact_table_value(text)) >= max(2, len(following) // 2):
        return False
    return True


def _looks_like_compact_table_value(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_numeric(cleaned):
        return True
    value_core = _strip_table_value_note_markers(cleaned)
    if value_core != cleaned and _looks_like_compact_table_value(value_core):
        return True
    if _looks_like_compact_numeric_value_with_unit(cleaned):
        return True
    if re.fullmatch(r"[~∼≈]?\s*\d+(?:[.,]\d+)?\s*[KMBkmb]?", cleaned):
        return True
    if re.fullmatch(r"[OX✓✗✔✘鉁?]\s*", cleaned, re.IGNORECASE):
        return True
    if re.fullmatch(r"\d+(?:[.,]\d+)?\s*%", cleaned):
        return True
    return len(cleaned) <= 8 and bool(re.fullmatch(r"[A-Za-z0-9+./%~∼≈-]+", cleaned))


def _strip_table_value_note_markers(text: str | None) -> str:
    cleaned = _cell_text(text)
    if not cleaned:
        return ""
    return re.sub(r"^[*#\$†‡§¶]+(?:\s*)", "", cleaned).strip()


def _looks_like_compact_numeric_value_with_unit(text: str | None) -> bool:
    cleaned = _strip_table_value_note_markers(text)
    if not cleaned or len(cleaned) > 32:
        return False
    if re.fullmatch(
        r"(?:[<>≤≥=]\s*)?[+\-−]?\$?\d+(?:,\d{3})*(?:[.,]\d+)?"
        r"(?:\s*(?:%|[A-Za-zµμ][A-Za-z0-9µμ/%.\-]*(?:/[A-Za-z0-9µμ%.\-]+)*))?",
        cleaned,
        re.IGNORECASE,
    ):
        return True
    return bool(re.fullmatch(r"p\s*[<=>≤≥]\s*\d+(?:[.,]\d+)?", cleaned, re.IGNORECASE))


def _project_flowchart_connector_semantic_grid(
    table: dict[str, Any],
    grid: list[list[Any]],
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 2:
        return None, None
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0][:col_count]]
    if len(header) < col_count:
        return None, None
    connector_positions = _flowchart_header_connector_positions(header[:col_count])
    if not connector_positions:
        return None, None
    semantic_header = _flowchart_expand_row_with_connectors(header[:col_count], connector_positions, is_header=True)
    if len(semantic_header) <= col_count:
        return None, None
    word_projected_grid, word_projection = _project_flowchart_from_word_evidence(
        table,
        semantic_header,
        connector_positions,
    )
    if word_projected_grid:
        return word_projected_grid, {
            "source": "flowchart_connector_column_projection",
            "connector_positions": connector_positions,
            "source_col_count": col_count,
            "projected_col_count": len(semantic_header),
            "word_projection": word_projection,
        }
    rows = _clone_grid(grid)
    header_matches_grid = [_cell_text(cell) for cell in rows[0][:col_count]] == [_cell_text(cell) for cell in header[:col_count]]
    body_rows = rows[1:] if header_matches_grid else _body_rows(table, grid)
    semantic_grid: list[list[str | None]] = [[cell or None for cell in semantic_header]]
    for row in body_rows:
        expanded = _flowchart_expand_row_with_connectors(row[:col_count], connector_positions, is_header=False)
        if any(_cell_text(cell) for cell in expanded):
            semantic_grid.append([_cell_text(cell) or None for cell in expanded])
    if len(semantic_grid) < 2:
        return None, None
    return semantic_grid, {
        "source": "flowchart_connector_column_projection",
        "connector_positions": connector_positions,
        "source_col_count": col_count,
        "projected_col_count": len(semantic_header),
    }


def _project_flowchart_from_word_evidence(
    table: dict[str, Any],
    semantic_header: list[str | None],
    connector_positions: list[int],
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    words = _table_word_evidence(table)
    if not words:
        return None, None
    table_bbox = _bbox(table)
    if not table_bbox:
        return None, None
    header_bottom = _flowchart_header_bottom_y(words, semantic_header, table_bbox)
    connector_words = [
        word for word in words
        if _extract_flowchart_connector(str(word.get("text") or ""))
    ]
    if len(connector_words) < len(connector_positions):
        return None, None

    header_connector_words = [
        word for word in connector_words
        if _word_y_center(word) <= header_bottom + max(2.0, _word_height(word) * 0.35)
    ]
    connector_source_words = header_connector_words if len(header_connector_words) >= len(connector_positions) else connector_words
    connector_centers = sorted({round(_word_x_center(word), 2) for word in connector_source_words})
    if len(connector_centers) < len(connector_positions):
        return None, None
    if len(connector_centers) > len(connector_positions):
        connector_centers = _select_connector_centers(connector_centers, len(connector_positions))

    semantic_width = len(semantic_header)
    content_cols = [idx for idx in range(semantic_width) if idx % 2 == 0]
    connector_cols = [idx for idx in range(semantic_width) if idx % 2 == 1]
    if len(content_cols) != len(connector_centers) + 1 or len(connector_cols) != len(connector_centers):
        return None, None

    body_words = [
        word for word in words
        if _word_y_center(word) > header_bottom + 1.0
        and not _extract_flowchart_connector(str(word.get("text") or ""))
    ]
    if len(body_words) < 6:
        return None, None

    word_lines = _cluster_flowchart_word_lines(body_words)
    if len(word_lines) < 2:
        return None, None
    line_records = [_flowchart_line_record(line, connector_centers, content_cols, semantic_width) for line in word_lines]
    line_records = [record for record in line_records if any(record["cells"])]
    if len(line_records) < 2:
        return None, None

    row_groups = _group_flowchart_word_lines_into_records(line_records, content_cols)
    if not row_groups:
        return None, None
    semantic_grid: list[list[str | None]] = [[cell or None for cell in semantic_header]]
    source_groups: list[dict[str, Any]] = []
    for group in row_groups:
        row = _merge_flowchart_line_records(group, content_cols, semantic_width)
        if any(_cell_text(row[idx]) for idx in content_cols):
            semantic_grid.append(row)
            source_groups.append({
                "target_row": len(semantic_grid) - 1,
                "line_count": len(group),
                "y_range": [
                    round(min(float(item["y0"]) for item in group), 3),
                    round(max(float(item["y1"]) for item in group), 3),
                ],
            })
    if len(semantic_grid) < 2:
        return None, None
    return semantic_grid, {
        "source": "flowchart_word_evidence_projection",
        "word_count": len(words),
        "body_word_count": len(body_words),
        "connector_centers": connector_centers,
        "connector_center_source": "header_words" if header_connector_words else "all_words",
        "line_count": len(line_records),
        "row_group_count": len(source_groups),
        "source_groups": source_groups,
    }


def _flowchart_header_connector_positions(header: list[Any]) -> list[int]:
    positions: list[int] = []
    for idx, value in enumerate(header):
        text = _cell_text(value)
        if idx <= 0:
            continue
        if _extract_flowchart_connector(text):
            positions.append(idx)
    return positions


def _table_word_evidence(table: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for item in table.get("word_evidence") or []:
        if not isinstance(item, dict):
            continue
        text = _cell_text(item.get("text"))
        bbox = item.get("bbox")
        if not text or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            words.append({
                "text": text,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            })
        except (TypeError, ValueError):
            continue
    return words


def _word_x_center(word: dict[str, Any]) -> float:
    bbox = word.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return (float(bbox[0]) + float(bbox[2])) / 2.0


def _word_y_center(word: dict[str, Any]) -> float:
    bbox = word.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return (float(bbox[1]) + float(bbox[3])) / 2.0


def _word_height(word: dict[str, Any]) -> float:
    bbox = word.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _select_connector_centers(centers: list[float], expected_count: int) -> list[float]:
    if expected_count <= 0 or len(centers) <= expected_count:
        return centers[:expected_count]
    if expected_count == 1:
        mid = (centers[0] + centers[-1]) / 2.0
        return [min(centers, key=lambda value: abs(value - mid))]
    selected: list[float] = []
    for idx in range(expected_count):
        target = idx * (len(centers) - 1) / max(1, expected_count - 1)
        selected.append(centers[round(target)])
    return sorted(set(selected))[:expected_count]


def _flowchart_header_bottom_y(
    words: list[dict[str, Any]],
    semantic_header: list[str | None],
    table_bbox: tuple[float, float, float, float],
) -> float:
    header_terms = {
        term
        for cell in semantic_header
        for term in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", _cell_text(cell))
        if term and term != "→"
    }
    near_top_words = [
        word for word in words
        if _word_y_center(word) <= table_bbox[1] + max(36.0, (table_bbox[3] - table_bbox[1]) * 0.18)
    ]
    top_lines = _cluster_flowchart_word_lines(near_top_words)
    if top_lines:
        first_line = top_lines[0]
        first_line_terms = {str(word.get("text") or "") for word in first_line}
        header_overlap = len(first_line_terms & header_terms)
        if header_overlap >= max(1, min(2, len(header_terms))):
            return max(float((word.get("bbox") or [0, 0, 0, table_bbox[1]])[3]) for word in first_line)
    header_words = [word for word in near_top_words if str(word.get("text") or "") in header_terms]
    if header_words:
        first_y = min(_word_y_center(word) for word in header_words)
        same_line_words = [
            word for word in header_words
            if abs(_word_y_center(word) - first_y) <= max(2.0, _word_height(word) * 0.55)
        ]
        return max(float((word.get("bbox") or [0, 0, 0, table_bbox[1]])[3]) for word in same_line_words)
    return table_bbox[1] + max(10.0, (table_bbox[3] - table_bbox[1]) * 0.08)


def _cluster_flowchart_word_lines(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not words:
        return []
    ordered = sorted(words, key=lambda item: (_word_y_center(item), _word_x_center(item)))
    heights = sorted(_word_height(word) for word in ordered if _word_height(word) > 0)
    median_height = heights[len(heights) // 2] if heights else 8.0
    y_tolerance = max(2.0, median_height * 0.55)
    lines: list[list[dict[str, Any]]] = []
    centers: list[float] = []
    for word in ordered:
        center = _word_y_center(word)
        if not lines:
            lines.append([word])
            centers.append(center)
            continue
        if abs(center - centers[-1]) <= y_tolerance:
            lines[-1].append(word)
            centers[-1] = sum(_word_y_center(item) for item in lines[-1]) / len(lines[-1])
            continue
        lines.append([word])
        centers.append(center)
    return [sorted(line, key=_word_x_center) for line in lines]


def _flowchart_line_record(
    line: list[dict[str, Any]],
    connector_centers: list[float],
    content_cols: list[int],
    semantic_width: int,
) -> dict[str, Any]:
    cells: list[str | None] = [None] * semantic_width
    words_by_col: dict[int, list[dict[str, Any]]] = {idx: [] for idx in content_cols}
    for word in line:
        content_col = _flowchart_content_col_for_word(word, connector_centers, content_cols)
        words_by_col.setdefault(content_col, []).append(word)
    for col_idx, col_words in words_by_col.items():
        text = _cell_text(" ".join(str(word.get("text") or "") for word in sorted(col_words, key=_word_x_center)))
        cells[col_idx] = text or None
    y_values = [float((word.get("bbox") or [0, 0, 0, 0])[1]) for word in line]
    y2_values = [float((word.get("bbox") or [0, 0, 0, 0])[3]) for word in line]
    return {
        "cells": cells,
        "y0": min(y_values) if y_values else 0.0,
        "y1": max(y2_values) if y2_values else 0.0,
        "filled": [idx for idx in content_cols if _cell_text(cells[idx])],
    }


def _flowchart_content_col_for_word(
    word: dict[str, Any],
    connector_centers: list[float],
    content_cols: list[int],
) -> int:
    x_center = _word_x_center(word)
    before = sum(1 for center in connector_centers if x_center > center)
    return content_cols[min(before, len(content_cols) - 1)]


def _group_flowchart_word_lines_into_records(
    line_records: list[dict[str, Any]],
    content_cols: list[int],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    left_col = content_cols[0]
    for record in line_records:
        filled = list(record.get("filled") or [])
        starts_new = bool(_cell_text((record.get("cells") or [])[left_col] if record.get("cells") else None))
        if starts_new and current and _flowchart_record_is_new_logical_row(record, current, content_cols):
            groups.append(current)
            current = [record]
            continue
        current.append(record)
    if current:
        groups.append(current)
    return groups


def _flowchart_record_is_new_logical_row(
    record: dict[str, Any],
    current: list[dict[str, Any]],
    content_cols: list[int],
) -> bool:
    cells = record.get("cells") or []
    left_text = _cell_text(cells[content_cols[0]] if len(cells) > content_cols[0] else None)
    if not left_text:
        return False
    if _flowchart_left_stub_fragment_continues(left_text, _merged_flowchart_group_col_text(current, content_cols[0])):
        return False
    filled = [idx for idx in content_cols if len(cells) > idx and _cell_text(cells[idx])]
    if len(filled) >= 2:
        return True
    return _flowchart_left_stub_looks_like_logical_row_start(left_text)


def _flowchart_left_stub_looks_like_logical_row_start(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if re.match(r"^(?:that|which|who|whose|where|with|without|for|of|to|in|on|and|or)\b", cleaned, re.IGNORECASE):
        return False
    if re.match(r"^\([A-Za-z0-9]{1,8}\)$", cleaned):
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", cleaned))


def _merged_flowchart_group_col_text(group: list[dict[str, Any]], col_idx: int) -> str:
    parts: list[str] = []
    for record in group:
        cells = record.get("cells") or []
        if col_idx < len(cells):
            text = _cell_text(cells[col_idx])
            if text:
                parts.append(text)
    return _cell_text(" ".join(parts))


def _merge_flowchart_line_records(
    group: list[dict[str, Any]],
    content_cols: list[int],
    semantic_width: int,
) -> list[str | None]:
    merged: list[str | None] = [None] * semantic_width
    for record in group:
        cells = record.get("cells") or []
        for idx in content_cols:
            if idx >= len(cells):
                continue
            text = _cell_text(cells[idx])
            if not text:
                continue
            current = _cell_text(merged[idx])
            merged[idx] = _join_wrapped_description_text(current, text) if current else text
    for idx in range(1, semantic_width, 2):
        merged[idx] = "→"
    return [_cell_text(cell) or None for cell in merged]


def _flowchart_expand_row_with_connectors(
    row: list[Any],
    connector_positions: list[int],
    *,
    is_header: bool,
) -> list[str | None]:
    connectors = set(connector_positions)
    expanded: list[str | None] = []
    for idx, value in enumerate(row):
        text = _cell_text(value)
        if idx in connectors:
            connector = _extract_flowchart_connector(text) or "→"
            expanded.append(connector)
            text = _remove_flowchart_connector(text) if is_header else text
        expanded.append(text or None)
    return expanded


def _extract_flowchart_connector(text: str) -> str:
    cleaned = _cell_text(text)
    if not cleaned:
        return ""
    if "→" in cleaned or "¡ú" in cleaned:
        return "→"
    if "↔" in cleaned:
        return "↔"
    if "=>" in cleaned:
        return "=>"
    if "->" in cleaned:
        return "→"
    if "<-" in cleaned:
        return "←"
    if re.search(r"\barrow\b", cleaned, re.IGNORECASE):
        return "→"
    return ""


def _remove_flowchart_connector(text: str) -> str:
    cleaned = _cell_text(text)
    cleaned = cleaned.replace("¡ú", "→")
    cleaned = re.sub(r"\s*(?:→|↔|=>|->|<-)\s*", " ", cleaned)
    cleaned = re.sub(r"\barrow\b", " ", cleaned, flags=re.IGNORECASE)
    return _cell_text(cleaned)


def _compact_flowchart_matrix_wrapped_rows(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    if family != "flowchart_matrix":
        return None, None
    projected_width = max((len(row) for row in grid if isinstance(row, list)), default=0)
    col_count = max(_col_count(table, grid), projected_width)
    if col_count < 5 or len(grid) < 5:
        return None, None
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    header = [_cell_text(cell) or None for cell in rows[0]]
    connector_cols = [idx for idx, cell in enumerate(header) if _extract_flowchart_connector(_cell_text(cell))]
    if not connector_cols:
        connector_cols = [idx for idx in range(col_count) if idx % 2 == 1]
    content_cols = [idx for idx in range(col_count) if idx not in set(connector_cols)]
    if len(content_cols) < 3:
        return None, None

    compacted: list[list[str | None]] = [header]
    groups: list[dict[str, Any]] = []
    current: list[str | None] | None = None
    current_sources: list[int] = []
    pending_prefix: list[str | None] = [None] * col_count
    pending_sources: list[int] = []
    changed = False

    for source_idx, row in enumerate(rows[1:], start=1):
        normalized = [_cell_text(cell) or None for cell in row[:col_count]]
        filled_content = [idx for idx in content_cols if _cell_text(normalized[idx])]
        if not filled_content:
            continue
        if current is None:
            if _flowchart_row_is_prefix_fragment(normalized, content_cols):
                pending_prefix = _merge_flowchart_rows(pending_prefix, normalized, content_cols)
                pending_sources.append(source_idx)
                changed = True
                continue
            current = normalized
            current_sources = [source_idx]
            if pending_sources:
                current = _merge_flowchart_rows(pending_prefix, current, content_cols)
                current_sources = [*pending_sources, *current_sources]
                pending_prefix = [None] * col_count
                pending_sources = []
                changed = True
            continue
        if _flowchart_row_continues_current(current, normalized, content_cols):
            current = _merge_flowchart_rows(current, normalized, content_cols)
            current_sources.append(source_idx)
            changed = True
            continue
        compacted.append(current)
        if len(current_sources) > 1:
            groups.append({"target_row": len(compacted) - 1, "source_rows": current_sources})
        current = normalized
        current_sources = [source_idx]

    if current is not None:
        compacted.append(current)
        if len(current_sources) > 1:
            groups.append({"target_row": len(compacted) - 1, "source_rows": current_sources})
    elif pending_sources:
        compacted.append(pending_prefix)
        groups.append({"target_row": len(compacted) - 1, "source_rows": pending_sources})

    if not changed:
        return None, None
    return compacted, {
        "source": "flowchart_wrapped_row_compaction",
        "source_row_count": len(rows),
        "compacted_row_count": len(compacted),
        "content_cols": content_cols,
        "groups": groups,
    }


def _flowchart_row_is_prefix_fragment(row: list[str | None], content_cols: list[int]) -> bool:
    filled = [idx for idx in content_cols if _cell_text(row[idx])]
    if not filled or filled[0] != content_cols[0]:
        return False
    if len([idx for idx in filled if idx != content_cols[0]]) > 1:
        return False
    first_text = _cell_text(row[content_cols[0]])
    if not first_text or _looks_numeric(first_text):
        return False
    return len(first_text) <= 80 and len(first_text.split()) <= 8


def _flowchart_row_continues_current(
    current: list[str | None],
    row: list[str | None],
    content_cols: list[int],
) -> bool:
    filled = [idx for idx in content_cols if _cell_text(row[idx])]
    if not filled:
        return False
    if content_cols[0] in filled:
        previous = _cell_text(current[content_cols[0]])
        candidate = _cell_text(row[content_cols[0]])
        if previous and candidate and _flowchart_left_stub_fragment_continues(candidate, previous):
            return True
    for idx in filled:
        if idx == content_cols[0]:
            continue
        candidate = _cell_text(row[idx])
        previous = _cell_text(current[idx])
        if previous and candidate and (
            _looks_like_wrapped_description_continuation(candidate, previous)
            or _flowchart_cell_fragment_continues(candidate, previous)
        ):
            return True
    if len(filled) == 1:
        idx = filled[0]
        previous = _cell_text(current[idx])
        candidate = _cell_text(row[idx])
        return bool(previous and _flowchart_cell_fragment_continues(candidate, previous))
    return False


def _flowchart_left_stub_fragment_continues(candidate: str, previous: str) -> bool:
    text = _cell_text(candidate)
    if not text:
        return False
    if re.match(r"^(?:that|which|who|whose|where|with|without|for|of|to|in|on|and|or)\b", text, re.IGNORECASE):
        return True
    if re.match(r"^\([A-Za-z0-9]{1,8}\)$", text):
        return True
    return len(text.split()) <= 4 and not re.search(r"[.?!;銆傦紒锛燂紱]\s*$", _cell_text(previous))


def _flowchart_cell_fragment_continues(candidate: str, previous: str) -> bool:
    text = _cell_text(candidate)
    prev = _cell_text(previous)
    if not text or not prev:
        return False
    if _extract_flowchart_connector(text):
        return True
    if len(text.split()) <= 7 and not re.match(r"^[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,3}$", text):
        return True
    return _row_text_looks_like_prose_sentence(_join_wrapped_description_text(prev, text))


def _merge_flowchart_rows(
    left: list[str | None],
    right: list[str | None],
    content_cols: list[int],
) -> list[str | None]:
    width = max(len(left), len(right))
    merged = list(left) + [None] * max(0, width - len(left))
    candidate = list(right) + [None] * max(0, width - len(right))
    for idx in content_cols:
        if idx >= width:
            continue
        left_text = _cell_text(merged[idx])
        right_text = _cell_text(candidate[idx])
        if not right_text:
            continue
        if not left_text:
            merged[idx] = right_text
            continue
        if _normalize_text(right_text) in _normalize_text(left_text):
            continue
        merged[idx] = _join_wrapped_description_text(left_text, right_text)
    return [_cell_text(cell) or None for cell in merged]


def _project_compressed_image_measurement_matrix(
    table: dict[str, Any],
    grid: list[list[Any]],
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    rows = _clone_grid(grid)
    data_rows = _compressed_measurement_data_rows(rows)
    if len(data_rows) < 2:
        return None, None
    first_data_index = data_rows[0][0]
    header_rows = rows[:first_data_index]
    value_count = max(len(_measurement_value_tokens(_cell_text(row[1]))) for _, row in data_rows)
    header = _project_compressed_measurement_header(header_rows, value_count)
    if not header or len(header) < value_count + 1:
        return None, None
    semantic_grid: list[list[str | None]] = [header]
    for _, row in data_rows:
        label = _cell_text(row[0])
        values = _measurement_value_tokens(_cell_text(row[1]))
        projected_values = _project_measurement_values_to_header(label, values, header[1:])
        semantic_grid.append([label, *projected_values])
    return semantic_grid, {
        "source": "compressed_image_measurement_matrix_projection",
        "source_row_count": len(rows),
        "header_source_row_count": len(header_rows),
        "projected_col_count": len(header),
        "data_row_count": len(data_rows),
    }


def _compressed_measurement_data_rows(rows: list[list[Any]]) -> list[tuple[int, list[Any]]]:
    result: list[tuple[int, list[Any]]] = []
    for idx, row in enumerate(rows):
        if len(row) < 2:
            continue
        label = _cell_text(row[0])
        values = _measurement_value_tokens(_cell_text(row[1]))
        if not label or len(values) < 2:
            continue
        if _row_text_looks_like_prose_sentence(label):
            continue
        result.append((idx, row))
    return result


def _measurement_value_tokens(text: str) -> list[str]:
    cleaned = _normalize_measurement_text(text)
    tokens = [_normalize_measurement_token(match.group(0)) for match in _MEASUREMENT_VALUE_RE.finditer(cleaned)]
    return [token for token in tokens if token]


def _normalize_measurement_text(text: str) -> str:
    cleaned = _cell_text(text)
    if not cleaned:
        return ""
    return (
        cleaned.replace("¦Ì", "μ")
        .replace("𝜇", "μ")
        .replace("渭", "μ")
        .replace("碌", "μ")
        .replace("µ", "μ")
    )


def _normalize_measurement_token(text: str) -> str:
    cleaned = _normalize_measurement_text(text)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"(?i)^(\d+(?:\.\d+)?)uL$", r"\1μL", cleaned)
    cleaned = re.sub(r"(?i)^(\d+(?:\.\d+)?)ul$", r"\1μL", cleaned)
    cleaned = re.sub(r"(?i)^(\d+(?:\.\d+)?)[?]L$", r"\1μL", cleaned)
    cleaned = re.sub(r"(?i)^(\d+(?:\.\d+)?)μl$", r"\1μL", cleaned)
    cleaned = re.sub(r"(?i)^(\d+(?:\.\d+)?)ml$", r"\1mL", cleaned)
    return re.sub(r"^(\d+(?:\.\d+)?)([A-Za-zμ]+)$", r"\1 \2", cleaned)


def _project_compressed_measurement_header(
    header_rows: list[list[Any]],
    value_count: int,
) -> list[str | None]:
    left_text = " ".join(_cell_text(row[0]) for row in header_rows if len(row) > 0 and _cell_text(row[0]))
    right_text = " ".join(_cell_text(row[1]) for row in header_rows if len(row) > 1 and _cell_text(row[1]))
    all_text = " ".join(part for part in (left_text, right_text) if part)
    tokens = _compressed_header_tokens(all_text)
    if not tokens:
        return []
    row_label = "Tube" if any(_normalize_text(token).startswith("tube") for token in tokens) else tokens[0]
    left_tokens = _compressed_header_tokens(left_text)
    right_tokens = _compressed_header_tokens(right_text)
    reagent = _compressed_left_header_phrase(left_tokens)
    buffer = _compressed_buffer_header_phrase(right_tokens)
    suspect1 = _header_phrase_from_token_pair(right_tokens, "suspect", "1", suffix="DNA")
    suspect2 = _header_phrase_from_token_pair(right_tokens, "suspect", "2", suffix="DNA")
    evidence = _compressed_evidence_header_phrase(right_tokens)
    water = next((token for token in tokens if _normalize_text(token) in {"h20", "h2o", "water"}), "H20")
    columns = [row_label, reagent, buffer, suspect1, suspect2, evidence, water]
    columns = [_cell_text(column) or None for column in columns]
    if sum(1 for column in columns if column) < value_count + 1:
        return []
    return columns


def _compressed_left_header_phrase(tokens: list[str]) -> str:
    useful = [token for token in tokens if _normalize_text(token) not in {"tube", "restriction"}]
    if not useful:
        return ""
    leading = useful[0]
    tail_norm = " ".join(_normalize_text(token) for token in useful[1:])
    tail = "enzyme mixture" if "enzyme" in tail_norm or "mixture" in tail_norm else _cell_text(" ".join(useful[1:]))
    return _cell_text(f"{leading} restriction {tail}".strip())


def _compressed_buffer_header_phrase(tokens: list[str]) -> str:
    buffer_idx = _find_token_index_contains(tokens, "buffer")
    if buffer_idx < 0:
        return ""
    buffer_token = tokens[buffer_idx]
    prefix = next((token for token in tokens[:buffer_idx] if _normalize_text(token) == "restriction"), "")
    return _cell_text(" ".join(token for token in (prefix, buffer_token) if token))


def _compressed_evidence_header_phrase(tokens: list[str]) -> str:
    evidence_idx = _find_token_index(tokens, "evidence")
    if evidence_idx < 0:
        return ""
    phrase_tokens = [tokens[evidence_idx]]
    tail_norms = [_normalize_text(token) for token in tokens[evidence_idx + 1 :]]
    if "a" in tail_norms:
        phrase_tokens.append("A")
        if "or" in tail_norms and "b" in tail_norms:
            phrase_tokens.extend(["or", "B"])
    return _cell_text(" ".join(phrase_tokens))


def _compressed_header_tokens(text: str) -> list[str]:
    cleaned = _cell_text(text)
    if not cleaned:
        return []
    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[()+/]", cleaned)
    tokens: list[str] = []
    for token in raw_tokens:
        lower = token.lower()
        if lower in {"for", "use", "with", "stain", "carolinablu", "carolinablu'm"}:
            continue
        tokens.append(token)
    return tokens


def _header_phrase_before_token(tokens: list[str], token: str, *, fallback_start: int, fallback_end: int) -> str:
    index = _find_token_index(tokens, token)
    if index > 0:
        start = 0 if index <= 2 else max(0, index - 2)
        phrase = " ".join(tokens[start:index + 1])
    else:
        phrase = " ".join(tokens[fallback_start:fallback_end])
    if "enzyme" not in _normalize_text(phrase):
        phrase = f"{phrase} enzyme mixture".strip()
    return _cell_text(phrase)


def _header_phrase_between_tokens(
    tokens: list[str],
    start_token: str,
    end_token: str,
    *,
    include_start: bool = False,
) -> str:
    start = _find_token_index(tokens, start_token)
    end = _find_token_index(tokens, end_token)
    if start < 0 or end < 0 or end <= start:
        return ""
    phrase_tokens = tokens[start:end] if include_start else tokens[start + 1 : end]
    return _cell_text(" ".join(phrase_tokens))


def _header_phrase_from_token_pair(tokens: list[str], left: str, right: str, *, suffix: str) -> str:
    for idx in range(len(tokens) - 1):
        if _normalize_text(tokens[idx]) == left and _normalize_text(tokens[idx + 1]) == right:
            return _cell_text(f"{tokens[idx]} {tokens[idx + 1]} {suffix}")
    return ""


def _find_token_index(tokens: list[str], token: str) -> int:
    target = _normalize_text(token)
    for idx, value in enumerate(tokens):
        if _normalize_text(value) == target:
            return idx
    return -1


def _find_token_index_contains(tokens: list[str], token: str) -> int:
    target = _normalize_text(token)
    for idx, value in enumerate(tokens):
        if target and target in _normalize_text(value):
            return idx
    return -1


def _project_measurement_values_to_header(
    label: str,
    values: list[str],
    columns: list[str | None],
) -> list[str | None]:
    projected: list[str | None] = [None] * len(columns)
    if not values:
        return projected
    role_projected = _project_measurement_values_by_column_roles(label, values, columns)
    if role_projected:
        return role_projected
    shared_indices = [
        idx
        for idx, column in enumerate(columns)
        if any(token in _normalize_text(column or "") for token in ("restriction", "buffer", "enzyme", "mixture", "rnase"))
    ]
    always_prefix_count = min(len(shared_indices), len(values))
    for idx, column_index in enumerate(shared_indices[:always_prefix_count]):
        projected[column_index] = values[idx]
    remaining = values[always_prefix_count:]
    label_norm = _normalize_text(label)
    target_index: int | None = None
    for idx, column in enumerate(columns):
        col_norm = _normalize_text(column or "")
        if label_norm and label_norm in col_norm:
            target_index = idx
            break
    if target_index is None and remaining:
        target_index = always_prefix_count if always_prefix_count < len(projected) else None
    if target_index is not None and remaining:
        projected[target_index] = remaining[0]
        remaining = remaining[1:]
    last_index = next(
        (
            idx
            for idx, column in enumerate(columns)
            if _normalize_text(column or "") in {"h20", "h2o", "water"}
        ),
        len(projected) - 1,
    )
    if remaining and last_index >= 0:
        projected[last_index] = remaining[-1]
    elif len(values) > always_prefix_count + (1 if target_index is not None else 0) and last_index >= 0:
        projected[last_index] = values[-1]
    return projected


def _project_measurement_values_by_column_roles(
    label: str,
    values: list[str],
    columns: list[str | None],
) -> list[str | None] | None:
    shared_indices = [
        idx
        for idx, column in enumerate(columns)
        if any(token in _normalize_text(column or "") for token in ("restriction", "buffer", "enzyme", "mixture", "rnase"))
    ]
    target_index = _measurement_target_column_index(label, columns)
    water_index = _measurement_terminal_diluent_index(columns)
    if len(shared_indices) < 2 or target_index is None or water_index is None or len(values) < 3:
        return None
    measured = [(value, _measurement_numeric_value(value)) for value in values]
    measured = [(value, number) for value, number in measured if number is not None]
    if len(measured) < 3:
        return None
    projected: list[str | None] = [None] * len(columns)
    water_value = min(measured, key=lambda item: item[1])[0]
    projected[water_index] = water_value
    remaining = [item for item in measured if item[0] != water_value]
    if not remaining:
        return None
    target_value = max(remaining, key=lambda item: item[1])[0]
    projected[target_index] = target_value
    shared_candidates = [item[0] for item in remaining if item[0] != target_value]
    shared_value = shared_candidates[0] if shared_candidates else min(remaining, key=lambda item: item[1])[0]
    for idx in shared_indices:
        projected[idx] = shared_value
    return projected


def _measurement_target_column_index(label: str, columns: list[str | None]) -> int | None:
    label_norm = _normalize_text(label)
    if not label_norm:
        return None
    label_tokens = [token for token in re.findall(r"[a-z0-9]+", label_norm) if len(token) > 1 or token.isdigit()]
    best_index: int | None = None
    best_score = 0
    for idx, column in enumerate(columns):
        col_norm = _normalize_text(column or "")
        column_tokens = re.findall(r"[a-z0-9]+", col_norm)
        score = sum(1 for token in label_tokens if token in col_norm)
        score += sum(1 for token in label_tokens if _measurement_label_token_matches_column(token, column_tokens))
        if score > best_score:
            best_score = score
            best_index = idx
    return best_index if best_score > 0 else None


def _measurement_label_token_matches_column(label_token: str, column_tokens: list[str]) -> bool:
    token = _normalize_text(label_token).replace(" ", "")
    if not token or not column_tokens:
        return False
    initials = "".join(part[0] for part in column_tokens if part and part not in {"or", "and", "the"})
    if token and initials.startswith(token):
        return True
    match = re.fullmatch(r"([a-z]+)(\d+)", token)
    if match:
        letters, digits = match.groups()
        return any(part.startswith(letters) for part in initials) and digits in column_tokens
    if len(token) == 2 and token[0].isalpha() and token[1].isalpha():
        return any(part.startswith(token[0]) for part in column_tokens) and token[1] in column_tokens
    return False


def _measurement_terminal_diluent_index(columns: list[str | None]) -> int | None:
    for idx, column in enumerate(columns):
        if _normalize_text(column or "") in {"h20", "h2o", "water"}:
            return idx
    for idx, column in enumerate(columns):
        if any(token in _normalize_text(column or "") for token in ("water", "diluent", "vehicle", "saline")):
            return idx
    return None


def _measurement_numeric_value(value: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", _cell_text(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _project_leading_boundary_to_data_header(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, list[str] | None, dict[str, Any] | None]:
    """Recover a data grid when leading title/caption fragments were promoted to rows.

    This is a semantic-only projection for adjacent-table pages: a prior list tail
    or a multi-line caption may sit above the next table, while the true header is
    the first dense non-caption row followed by value-like rows.
    """
    if family not in {"comparison_matrix", "rowspan_grouped_table"}:
        return None, None, None
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 4:
        return None, None, None
    rows = _clone_grid(grid)
    if rows and _row_looks_like_logical_header(rows[0], col_count):
        return None, None, None
    header_index = _find_late_dense_data_header_index(rows, col_count)
    if header_index is None or header_index <= 0:
        return None, None, None
    leading_rows = rows[:header_index]
    if not leading_rows or not all(_row_is_boundary_before_late_header(row, col_count) for row in leading_rows):
        return None, None, None
    body_rows = rows[header_index + 1 :]
    if not body_rows or not _rows_support_late_header_data(body_rows, col_count):
        return None, None, None
    header = [_cell_text(cell) for cell in rows[header_index][:col_count]]
    semantic_grid = [[text or None for text in header]]
    semantic_grid.extend([[_cell_text(cell) or None for cell in row[:col_count]] for row in body_rows])
    title_text = _cell_text(table.get("title"))
    if (
        title_text
        and _row_text_matches_single_cell_boundary(title_text, leading_rows[0])
        and not _table_title_is_verified_internal_title_row(table)
    ):
        table["title"] = ""
        table["semantic_projection_v2_cleared_spurious_title"] = title_text
    return semantic_grid, header, {
        "source": "leading_boundary_header_projection",
        "dropped_leading_row_count": header_index,
        "header_row_index": header_index,
    }


def _project_compact_single_cell_header_to_data_columns(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, list[str] | None, dict[str, Any] | None]:
    """Recover a logical header when extraction compressed it into one cell.

    The projection is evidence-gated by the following data rows. This keeps page
    headers, logos, and short document titles out of the semantic table while
    preserving the observed display grid for audit.
    """
    if family not in {"comparison_matrix", "rowspan_grouped_table"}:
        return None, None, None
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 4:
        return None, None, None
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    header_index, header = _find_compact_single_cell_header_index(rows, col_count)
    if header_index is None or not header:
        return None, None, None
    leading_rows = rows[:header_index]
    if leading_rows and not all(_row_is_boundary_before_compact_single_cell_header(row, col_count) for row in leading_rows):
        return None, None, None
    body_rows = rows[header_index + 1 :]
    if not _rows_support_compact_single_cell_header_data(body_rows, col_count):
        return None, None, None

    semantic_grid: list[list[str | None]] = [[text or None for text in header]]
    semantic_grid.extend([[_cell_text(cell) or None for cell in row[:col_count]] for row in body_rows])
    title_text = _cell_text(table.get("title"))
    if (
        title_text
        and leading_rows
        and _row_text_matches_single_cell_boundary(title_text, leading_rows[0])
        and not _table_title_is_verified_internal_title_row(table)
    ):
        table["title"] = ""
        table["semantic_projection_v2_cleared_spurious_title"] = title_text
    return semantic_grid, header, {
        "source": "compact_single_cell_header_projection",
        "dropped_leading_row_count": header_index,
        "header_row_index": header_index,
        "source_col_count": col_count,
    }


def _table_title_is_verified_internal_title_row(table: dict[str, Any]) -> bool:
    title_block = table.get("title_block") if isinstance(table.get("title_block"), dict) else {}
    if str(title_block.get("source") or "") == "internal_title_row":
        return True
    try:
        title_row_index = int(table.get("title_row_index"))
    except Exception:
        return False
    return title_row_index == 0 and bool(_cell_text(table.get("title")))


def _find_compact_single_cell_header_index(
    rows: list[list[Any]],
    col_count: int,
) -> tuple[int | None, list[str] | None]:
    for idx in range(0, min(len(rows) - 2, 8)):
        row = rows[idx][:col_count]
        texts = [_cell_text(cell) for cell in row]
        filled = [(col_idx, text) for col_idx, text in enumerate(texts) if text]
        if len(filled) != 1:
            continue
        cell_col, header_text = filled[0]
        if cell_col > 0:
            continue
        body_rows = rows[idx + 1 :]
        if not _rows_support_compact_single_cell_header_data(body_rows, col_count):
            continue
        header = _segment_compact_single_cell_header_text(header_text, col_count, body_rows)
        if header:
            return idx, header
    return None, None


def _segment_compact_single_cell_header_text(
    text: str,
    col_count: int,
    body_rows: list[list[Any]],
) -> list[str] | None:
    cleaned = _cell_text(text)
    if not _looks_like_compact_single_cell_header_text(cleaned, col_count):
        return None
    words = cleaned.split()
    if len(words) == col_count:
        return words
    if len(words) < col_count or len(words) > col_count * 5:
        return None

    first_column_is_id = _body_column_is_identifier_like(body_rows, 0, col_count)
    value_columns = sum(
        1 for col_idx in range(1, col_count)
        if _body_column_is_value_like(body_rows, col_idx, col_count)
    )
    if first_column_is_id and value_columns >= col_count - 2 and len(words) - 1 >= col_count - 1:
        lengths = [1, *_balanced_positive_lengths(len(words) - 1, col_count - 1)]
    else:
        lengths = _balanced_positive_lengths(len(words), col_count)
    if not lengths or sum(lengths) != len(words):
        return None

    labels: list[str] = []
    offset = 0
    for length in lengths:
        labels.append(" ".join(words[offset : offset + length]).strip())
        offset += length
    if len(labels) != col_count or any(not label for label in labels):
        return None
    if any(_row_text_looks_like_prose_sentence(label) for label in labels):
        return None
    return labels


def _balanced_positive_lengths(total: int, groups: int) -> list[int]:
    if total < groups or groups <= 0:
        return []
    base, remainder = divmod(total, groups)
    return [base + (1 if idx < remainder else 0) for idx in range(groups)]


def _looks_like_compact_single_cell_header_text(text: str, col_count: int) -> bool:
    cleaned = _cell_text(text)
    if not cleaned or _looks_numeric(cleaned):
        return False
    if _looks_like_table_caption_like_text(cleaned) or _looks_like_table_note_or_marker_text(cleaned):
        return False
    if _row_text_looks_like_prose_sentence(cleaned):
        return False
    words = cleaned.split()
    if len(words) < col_count or len(words) > col_count * 5:
        return False
    if re.search(r"[.?!;。！？；]\s*$", cleaned):
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned))


def _row_is_boundary_before_compact_single_cell_header(row: list[Any], col_count: int) -> bool:
    if _row_is_boundary_before_late_header(row, col_count):
        return True
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [(idx, text) for idx, text in enumerate(texts) if text]
    if not filled:
        return True
    generic_count = sum(1 for _, text in filled if _is_generic_header(text))
    if generic_count < 1:
        return False
    content = [text for _, text in filled if not _is_generic_header(text)]
    if len(content) > max(2, col_count // 2):
        return False
    return all(_looks_like_page_boundary_atom(text) for text in content)


def _looks_like_page_boundary_atom(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return True
    if _looks_numeric(cleaned) or _MEASUREMENT_VALUE_RE.search(cleaned):
        return False
    if _looks_like_table_note_or_marker_text(cleaned) or _looks_like_table_caption_like_text(cleaned):
        return True
    if _row_text_looks_like_prose_sentence(cleaned):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned)
    if len(words) > 5:
        return False
    if re.fullmatch(r"[A-Z0-9 .:/_-]{2,40}", cleaned):
        return True
    return len(cleaned) <= 40 and len(cleaned.split()) <= 5


def _rows_support_compact_single_cell_header_data(rows: list[list[Any]], col_count: int) -> bool:
    meaningful = [row[:col_count] for row in rows if _non_empty_count(row[:col_count]) > 0]
    if len(meaningful) < 2:
        return False
    dense_rows = [row for row in meaningful if _non_empty_count(row) >= max(2, col_count - 1)]
    if len(dense_rows) < 2:
        return False
    value_like_rows = 0
    for row in dense_rows[:6]:
        values = [_cell_text(cell) for cell in row[:col_count]]
        filled = [value for value in values if value]
        if len(filled) < max(2, col_count - 1):
            continue
        value_like = sum(1 for value in filled if _looks_like_compact_body_value_atom(value))
        if value_like >= max(2, len(filled) - 1):
            value_like_rows += 1
    return value_like_rows >= 2


def _body_column_is_identifier_like(rows: list[list[Any]], col_idx: int, col_count: int) -> bool:
    values = [
        _cell_text(row[col_idx])
        for row in rows[:8]
        if len(row) > col_idx and _cell_text(row[col_idx])
    ]
    if len(values) < 2:
        return False
    matches = sum(1 for value in values if _looks_like_identifier_body_value(value))
    return matches >= max(2, len(values) - 1)


def _body_column_is_value_like(rows: list[list[Any]], col_idx: int, col_count: int) -> bool:
    values = [
        _cell_text(row[col_idx])
        for row in rows[:8]
        if len(row) > col_idx and _cell_text(row[col_idx])
    ]
    if len(values) < 2:
        return False
    matches = sum(1 for value in values if _looks_like_compact_body_value_atom(value))
    return matches >= max(2, len(values) - 1)


def _looks_like_identifier_body_value(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_numeric(cleaned):
        return True
    return len(cleaned) <= 24 and len(cleaned.split()) <= 3 and bool(re.search(r"[A-Za-z0-9]", cleaned))


def _looks_like_compact_body_value_atom(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_numeric(cleaned) or _MEASUREMENT_VALUE_RE.search(cleaned):
        return True
    if _row_text_looks_like_prose_sentence(cleaned):
        return False
    return len(cleaned) <= 24 and len(cleaned.split()) <= 3


def _find_late_dense_data_header_index(rows: list[list[Any]], col_count: int) -> int | None:
    for idx in range(1, min(len(rows), 8)):
        row = rows[idx][:col_count]
        if _row_is_fragmented_caption_before_late_header(rows, idx, col_count):
            continue
        if not _row_looks_like_compact_data_header(row, col_count):
            continue
        body = rows[idx + 1 :]
        if _rows_support_late_header_data(body, col_count):
            return idx
    return None


def _row_is_fragmented_caption_before_late_header(rows: list[list[Any]], row_index: int, col_count: int) -> bool:
    if row_index < 0 or row_index >= len(rows) - 2:
        return False
    row = rows[row_index][:col_count]
    texts = [_cell_text(cell) for cell in row]
    filled = [text for text in texts if text]
    if len(filled) < max(3, col_count - 1):
        return False
    joined = " ".join(filled)
    if not _looks_like_table_caption_like_text(joined) and not _looks_like_table_caption_like_text(filled[0]):
        return False
    next_row = rows[row_index + 1][:col_count]
    next_filled = [_cell_text(cell) for cell in next_row if _cell_text(cell)]
    if next_filled and not _row_is_boundary_before_late_header(next_row, col_count):
        return False
    candidate_index = row_index + 1
    while candidate_index < min(len(rows) - 1, row_index + 4):
        candidate_index += 1
        candidate = rows[candidate_index][:col_count]
        if not _row_looks_like_compact_data_header(candidate, col_count):
            continue
        if _rows_support_late_header_data(rows[candidate_index + 1 :], col_count):
            return True
    return False


def _row_looks_like_compact_data_header(row: list[Any], col_count: int) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [text for text in texts if text]
    if len(filled) < max(3, col_count - 1):
        return False
    if any(_looks_like_table_caption_like_text(text) for text in filled):
        return False
    if any(_row_text_looks_like_prose_sentence(text) for text in filled):
        return False
    if sum(1 for text in filled if _looks_numeric(text)) > 0:
        return False
    return True


def _row_is_boundary_before_late_header(row: list[Any], col_count: int) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [text for text in texts if text]
    if not filled:
        return True
    joined = " ".join(filled)
    if _looks_like_table_caption_like_text(joined):
        return True
    if len(filled) == 1:
        single = filled[0]
        if _looks_like_list_value_atom(single):
            return True
        return _row_text_looks_like_prose_sentence(single)
    if _row_text_looks_like_prose_sentence(joined):
        return True
    if _looks_like_table_caption_like_text(filled[0]):
        return True
    return False


def _rows_support_late_header_data(rows: list[list[Any]], col_count: int) -> bool:
    meaningful = [row[:col_count] for row in rows if _non_empty_count(row[:col_count]) > 0]
    if not meaningful:
        return False
    dense_rows = sum(1 for row in meaningful if _non_empty_count(row) >= max(2, col_count - 1))
    numeric_or_na_rows = sum(1 for row in meaningful if _row_is_numeric_or_na_values(row, col_count))
    return dense_rows >= 1 and (numeric_or_na_rows >= 1 or len(meaningful) >= 2)


def _row_is_numeric_or_na_values(row: list[Any], col_count: int) -> bool:
    values = [_cell_text(cell) for cell in row[:col_count] if _cell_text(cell)]
    if not values:
        return False
    matched = 0
    for value in values:
        if _looks_numeric(value) or re.fullmatch(r"n/?a|na|--|-", value, re.IGNORECASE):
            matched += 1
    return matched >= max(1, len(values) - 1)


def _row_text_matches_single_cell_boundary(text: str, row: list[Any]) -> bool:
    values = [_cell_text(cell) for cell in row if _cell_text(cell)]
    return len(values) == 1 and _normalize_text(values[0]) == _normalize_text(text)


def _row_looks_like_schema_header_fragment(row: list[Any], col_count: int, *, row_idx: int) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [text for text in texts if text]
    if not filled:
        return False
    if row_idx == 0:
        return len(filled) >= max(2, col_count // 2)
    if any(_looks_like_long_body_fragment(text) for text in filled):
        return False
    if any(_looks_numeric(text) and not _looks_like_unit_or_header_marker(text) for text in filled):
        return False
    return all(_looks_like_header_fragment_text(text) for text in filled)


def _row_starts_schema_body(row: list[Any], col_count: int) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [(idx, text) for idx, text in enumerate(texts) if text]
    if not filled:
        return False
    if len(filled) >= max(3, col_count - 1):
        return True
    if len(filled) == 1:
        idx, text = filled[0]
        return idx >= min(2, col_count - 2) and _looks_like_long_body_fragment(text)
    return any(_looks_like_long_body_fragment(text) for _, text in filled)


def _merge_schema_header_rows(header_rows: list[list[Any]], col_count: int) -> list[str]:
    parts_by_col: list[list[str]] = [[] for _ in range(col_count)]
    carry_text = ""
    for row in header_rows:
        row_parts = [_cell_text(cell) for cell in row[:col_count]]
        row_non_empty = [(idx, text) for idx, text in enumerate(row_parts) if text]
        if len(row_non_empty) == 1 and carry_text:
            idx, text = row_non_empty[0]
            if not parts_by_col[idx]:
                parts_by_col[idx].append(carry_text)
            parts_by_col[idx].append(text)
            continue
        for idx, text in row_non_empty:
            parts_by_col[idx].append(text)
        if row_non_empty:
            carry_text = row_non_empty[-1][1]
    return [_join_schema_header_parts(parts) for parts in parts_by_col]


def _join_schema_header_parts(parts: list[str]) -> str:
    result = ""
    for part in parts:
        text = _cell_text(part)
        if not text:
            continue
        if not result:
            result = text
            continue
        compact_result = _normalize_text(result)
        compact_text = _normalize_text(text)
        if compact_text and compact_text in compact_result:
            continue
        if compact_result and compact_result in compact_text:
            result = text
            continue
        if text.startswith(("(", "[", "{")):
            result = f"{result} {text}"
        else:
            result = f"{result} {text}"
    return result.strip()


def _looks_like_header_fragment_text(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_like_unit_or_header_marker(cleaned):
        return True
    if len(cleaned) <= 80 and len(cleaned.split()) <= 8 and bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned)):
        return True
    return False


def _looks_like_unit_or_header_marker(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if re.fullmatch(r"\([^)]{1,40}\)", cleaned):
        return True
    if re.fullmatch(r"\[[^\]]{1,40}\]", cleaned):
        return True
    return bool(re.fullmatch(r"(?:%|[A-Za-z]{1,8}/[A-Za-z0-9]+|[A-Za-z]{1,8})", cleaned))


def _looks_like_long_body_fragment(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if len(re.findall(r"[\u4e00-\u9fff]", cleaned)) >= 12:
        return True
    return len(cleaned) >= 28 and len(cleaned.split()) >= 4


def _schema_header_projection_has_body_evidence(body_rows: list[list[Any]], col_count: int) -> bool:
    probe_rows = [row[:col_count] for row in body_rows[: min(8, len(body_rows))]]
    if not probe_rows:
        return False
    keyed_rows = 0
    continuation_rows = 0
    for row in probe_rows:
        filled = [(idx, _cell_text(cell)) for idx, cell in enumerate(row) if _cell_text(cell)]
        if len(filled) >= max(3, col_count - 1):
            keyed_rows += 1
        elif len(filled) == 1 and filled[0][0] >= min(2, col_count - 2):
            continuation_rows += 1
    return keyed_rows >= 1 or continuation_rows >= 1


def _project_blank_form_comparison_semantic_grid(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, dict[str, Any]]:
    if family != "blank_form_comparison_matrix":
        return None, {}
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0][:2]]
    if len(header) < 2:
        return None, {}
    rows = _body_rows(table, grid)
    projected_body: list[list[str | None]] = []
    pending_stub = ""
    consumed_rows = 0
    for row in rows:
        left = _cell_text(row[0] if len(row) > 0 else None)
        right = _cell_text(row[1] if len(row) > 1 else None)
        if left and right and _semantic_row_is_probable_narrative_tail([left, right, None], 3):
            break
        if right:
            break
        if not left:
            continue
        if _semantic_row_is_probable_narrative_tail([left, None, None], 3):
            break
        if pending_stub and _blank_form_stub_fragments_should_join(pending_stub, left):
            pending_stub = _join_wrapped_description_text(pending_stub, left)
            consumed_rows += 1
            continue
        if pending_stub:
            projected_body.append([pending_stub, None, None])
        pending_stub = left
        consumed_rows += 1
    if pending_stub:
        projected_body.append([pending_stub, None, None])
    if len(projected_body) < 3:
        return None, {}
    projected_grid: list[list[str | None]] = [[None, header[0], header[1]], *projected_body]
    return projected_grid, {
        "source": "blank_form_comparison_projection",
        "source_row_count": consumed_rows,
        "trimmed_row_count": max(0, len(rows) - consumed_rows),
        "projected_col_count": 3,
    }


def _project_two_column_spanning_header_grid(
    table: dict[str, Any],
    grid: list[list[Any]],
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    if _col_count(table, grid) != 2 or len(grid) < 4:
        return None, None
    rows = _rectangular_grid(_clone_grid(grid), 2)
    first = [_cell_text(cell) for cell in rows[0][:2]]
    header_text = _join_schema_header_parts(first)
    if not header_text:
        return None, None
    body_rows = []
    for row in rows[1:]:
        left = _cell_text(row[0])
        right = _cell_text(row[1])
        if not left and not right:
            continue
        if left and right and not _semantic_row_is_probable_narrative_tail([left, right], 2):
            body_rows.append([left or None, right or None])
    if len(body_rows) < 3:
        return None, None
    return [[header_text, None], *body_rows], {
        "source": "two_column_spanning_header_projection",
        "header_colspan": 2,
        "body_row_count": len(body_rows),
    }


def _blank_form_stub_fragments_should_join(left: str, right: str) -> bool:
    left_clean = _cell_text(left)
    right_clean = _cell_text(right)
    if not left_clean or not right_clean:
        return False
    if _looks_like_standalone_form_field_label(right_clean):
        return False
    if re.match(r"^[a-z][a-z-]*$", right_clean) and not _ends_with_strong_separator(left_clean):
        return True
    if left_clean.endswith(("-", "/", "(")):
        return True
    return False


def _looks_like_standalone_form_field_label(text: str) -> bool:
    cleaned = _normalize_text(text)
    if not cleaned or len(cleaned.split()) > 3:
        return False
    generic_fields = {
        "purpose",
        "method",
        "methods",
        "result",
        "results",
        "description",
        "descriptions",
        "comment",
        "comments",
        "remark",
        "remarks",
        "note",
        "notes",
        "conclusion",
        "conclusions",
        "objective",
        "objectives",
        "summary",
        "status",
        "signature",
        "date",
    }
    return cleaned in generic_fields


def _ends_with_strong_separator(text: str) -> bool:
    return bool(re.search(r"[:;.!?銆傦紱锛侊紵]\s*$", _cell_text(text)))


def _refine_semantic_row_run_boundary(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, dict[str, Any]]:
    """Trim narrative tail rows from the semantic table view.

    Table discovery intentionally keeps high-recall raw evidence. When a
    borderless/text-aligned candidate runs past the true grid into following
    numbered prose, this semantic layer narrows the consumer-facing grid while
    leaving raw/display rows available for audit.
    """
    if family not in {"blank_form_comparison_matrix", "comparison_matrix", "rowspan_grouped_table"}:
        return None, {}
    if _looks_like_clustered_listing_table(table, grid):
        return None, {}
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 5:
        return None, {}
    body_start = 1
    rows = _clone_grid(grid)
    if len(rows) <= body_start + 3:
        return None, {}

    stable_rows = 0
    split_index: int | None = None
    for idx in range(body_start, len(rows)):
        row = rows[idx][:col_count]
        if _semantic_row_is_stable_table_body(row, col_count, family):
            stable_rows += 1
            continue
        if stable_rows < 3:
            continue
        if _semantic_row_is_probable_narrative_tail(row, col_count):
            split_index = idx
            break

    if split_index is None:
        return None, {}
    if split_index <= body_start + 2:
        return None, {}
    trimmed = rows[:split_index]
    if len(trimmed) < 3:
        return None, {}
    return _clone_grid(trimmed), {
        "source": "semantic_row_run_boundary_refinement",
        "trimmed_row_count": len(rows) - split_index,
        "split_row_index": split_index,
        "stable_row_count_before_split": stable_rows,
    }


def _semantic_row_is_stable_table_body(row: list[Any], col_count: int, family: str) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [(idx, text) for idx, text in enumerate(texts) if text]
    if not filled:
        return False
    if len(filled) >= max(2, col_count - 1):
        return not _row_text_looks_like_prose_sentence(" ".join(text for _, text in filled))
    first_col = texts[0] if texts else ""
    if first_col and len(filled) == 1:
        if family == "comparison_matrix":
            return _looks_like_stub_label(first_col)
        return not _row_text_looks_like_prose_sentence(first_col)
    return False


def _semantic_row_is_probable_narrative_tail(row: list[Any], col_count: int) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [(idx, text) for idx, text in enumerate(texts) if text]
    if not filled:
        return False
    joined = " ".join(text for _, text in filled)
    if not joined:
        return False
    if _looks_like_table_note_or_marker_text(joined):
        return False
    if _looks_like_table_caption_like_text(joined):
        return True
    if re.match(r"^\s*(?:\d+|[A-Za-z])[\.)]\s+\S+", joined):
        return True
    if len(filled) <= 2 and _row_text_looks_like_prose_sentence(joined):
        return True
    if len(filled) == 1 and len(joined.split()) >= 7:
        return True
    return False


def _looks_like_stub_label(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_numeric(cleaned):
        return False
    if re.match(r"^\s*(?:\d+|[A-Za-z])[\.)]\s+\S+", cleaned):
        return False
    if len(cleaned) <= 80 and len(cleaned.split()) <= 8:
        return True
    if cleaned.startswith("#") and len(cleaned.split()) <= 8:
        return True
    return False


def _row_text_looks_like_prose_sentence(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if re.search(r"[.?!;。！？；]\s*$", cleaned):
        return len(cleaned.split()) >= 5 or len(cleaned) >= 24
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned)
    if len(words) >= 9:
        return True
    if len(words) >= 6 and re.search(
        r"\b(?:the|and|or|of|to|in|on|with|from|for|is|are|was|were|when|that|this|these|those|using|ensuring)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return True
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    return cjk_count >= 18 and bool(re.search(r"[，。；！？]", cleaned))


def _looks_like_table_note_or_marker_text(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    return bool(
        re.match(r"^\s*(?:note|notes|legend|remark|remarks|注|说明|备注|图例)\b[:：]?", cleaned, re.IGNORECASE)
        or re.match(r"^\s*(?:[#\$]|\*+|[+\u2020\u2021])(?=\s*[-:：\]\u4e00-\u9fffA-Za-z])", cleaned)
    )


def _looks_like_table_caption_like_text(text: str) -> bool:
    cleaned = _cell_text(text)
    return bool(re.match(r"^\s*(?:table|tab\.?|fig(?:ure)?\.?|表|图)\s*[\w\d一二三四五六七八九十Xx.-]*", cleaned, re.IGNORECASE))


def _compact_keyed_long_description_rows(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    """Merge wrapped description-line rows under their keyed record.

    Borderless regulatory/IND overview tables often have compact key columns
    followed by one or more long narrative columns. Visual extraction emits each
    wrapped line as a row with only the description column populated. The
    semantic table should expose one logical record per key row while preserving
    raw/display rows for audit.
    """
    if family not in {"comparison_matrix", "rowspan_grouped_table", "projected_stub_matrix"}:
        return None, None
    col_count = _col_count(table, grid)
    if col_count < 4 or len(grid) < 5:
        return None, None
    rows = _clone_grid(grid)
    header_rows = 1 if _row_looks_like_logical_header(rows[0], col_count) else 0
    if not header_rows or _looks_numeric(_cell_text(rows[0][0] if rows[0] else "")):
        return None, None
    body_rows = rows[header_rows:]
    if len(body_rows) < 4:
        return None, None

    description_col = _dominant_long_description_column(body_rows, col_count)
    if description_col is None:
        return None, None

    compacted: list[list[str | None]] = []
    if header_rows:
        compacted.append([_cell_text(cell) or None for cell in rows[0][:col_count]])

    current: list[str | None] | None = None
    current_source_rows: list[int] = []
    groups: list[dict[str, Any]] = []
    changed = False
    for body_offset, row in enumerate(body_rows, start=header_rows):
        normalized_row = [_cell_text(cell) or None for cell in row[:col_count]]
        filled = [idx for idx, text in enumerate(normalized_row) if text]
        if not filled:
            continue
        if _is_keyed_long_description_record_start(normalized_row, description_col, col_count):
            if current is not None:
                compacted.append(current)
                if len(current_source_rows) > 1:
                    groups.append(
                        {
                            "target_row": len(compacted) - 1,
                            "source_rows": current_source_rows,
                            "description_col": description_col,
                        }
                    )
            current = normalized_row
            current_source_rows = [body_offset]
            continue
        if (
            current is not None
            and len(filled) == 1
            and filled[0] == description_col
            and _looks_like_wrapped_description_continuation(normalized_row[description_col], current[description_col])
        ):
            current[description_col] = _join_wrapped_description_text(
                current[description_col],
                normalized_row[description_col],
            )
            current_source_rows.append(body_offset)
            changed = True
            continue
        if (
            current is None
            and len(filled) == 1
            and filled[0] == description_col
            and _looks_like_long_description_continuation(normalized_row[description_col])
        ):
            current = normalized_row
            current_source_rows = [body_offset]
            continue

        if current is not None:
            compacted.append(current)
            if len(current_source_rows) > 1:
                groups.append(
                    {
                        "target_row": len(compacted) - 1,
                        "source_rows": current_source_rows,
                        "description_col": description_col,
                    }
                )
            current = None
            current_source_rows = []
        compacted.append(normalized_row)

    if current is not None:
        compacted.append(current)
        if len(current_source_rows) > 1:
            groups.append(
                {
                    "target_row": len(compacted) - 1,
                    "source_rows": current_source_rows,
                    "description_col": description_col,
                }
            )

    if not changed:
        return None, None
    return compacted, {
        "source_row_count": len(rows),
        "compacted_row_count": len(compacted),
        "description_col": description_col,
        "groups": groups,
        "source": "keyed_long_description_row_compaction",
    }


def _project_compact_single_cell_body_rows(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    if family not in {"comparison_matrix", "rowspan_grouped_table"}:
        return None, None
    col_count = _col_count(table, grid)
    if col_count < 4 or len(grid) < 4:
        return None, None
    rows = _clone_grid(grid)
    changed_rows: list[dict[str, Any]] = []
    projected: list[list[str | None]] = []
    for row_index, row in enumerate(rows):
        normalized = [_cell_text(cell) or None for cell in row[:col_count]]
        filled = [idx for idx, value in enumerate(normalized) if value]
        if filled == [0]:
            expanded = _split_compact_single_cell_schema_row(normalized[0], col_count)
            if expanded:
                projected.append(expanded)
                changed_rows.append({"source_row": row_index, "target_row": len(projected) - 1})
                continue
        projected.append(normalized)

    if not changed_rows:
        return None, None
    return projected, {
        "source": "compact_single_cell_body_row_projection",
        "projected_row_count": len(changed_rows),
        "rows": changed_rows,
    }


def _split_compact_single_cell_schema_row(text: str | None, col_count: int) -> list[str | None] | None:
    cleaned = _cell_text(text)
    if not cleaned or col_count < 2:
        return None
    tokens = cleaned.split()
    if len(tokens) <= col_count - 1:
        return None
    value_count = col_count - 1
    values = tokens[-value_count:]
    if not all(_looks_like_compact_table_value(value) for value in values):
        return None
    label = " ".join(tokens[:-value_count]).strip()
    if not label or _looks_numeric(label):
        return None
    if len(label.split()) > 6:
        return None
    return [label, *values]


def _compact_rowspan_repeated_key_description_rows(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    if family not in {"comparison_matrix", "rowspan_grouped_table"}:
        return None, None
    col_count = _col_count(table, grid)
    if col_count < 4 or len(grid) < 5:
        return None, None
    key_cols = _rowspan_repeated_key_columns(table, col_count)
    if not key_cols:
        return None, None
    description_cols = _rowspan_repeated_description_columns(grid, key_cols, col_count)
    if not description_cols:
        return None, None
    rows = _clone_grid(grid)
    header_rows = 1 if rows and sum(1 for cell in rows[0][:col_count] if _cell_text(cell)) >= max(2, col_count - 1) else 0
    if not header_rows:
        return None, None

    compacted: list[list[str | None]] = [[_cell_text(cell) or None for cell in rows[0][:col_count]]]
    current: list[str | None] | None = None
    current_key: tuple[str, ...] | None = None
    current_source_rows: list[int] = []
    groups: list[dict[str, Any]] = []
    changed = False
    for row_index, row in enumerate(rows[header_rows:], start=header_rows):
        normalized = [_cell_text(cell) or None for cell in row[:col_count]]
        if not any(normalized):
            continue
        key = tuple(_cell_text(normalized[idx]) for idx in key_cols)
        has_key = any(key)
        if current is not None and _rowspan_repeated_row_continues_current(
            normalized,
            key,
            current_key,
            key_cols,
            description_cols,
            col_count,
        ):
            for idx in description_cols:
                text = normalized[idx]
                if text:
                    current[idx] = _join_wrapped_description_text(current[idx], text) if current[idx] else text
            current_source_rows.append(row_index)
            changed = True
            continue
        if current is not None:
            compacted.append(current)
            if len(current_source_rows) > 1:
                groups.append({"target_row": len(compacted) - 1, "source_rows": current_source_rows})
        current = normalized
        current_key = key if has_key else None
        current_source_rows = [row_index]

    if current is not None:
        compacted.append(current)
        if len(current_source_rows) > 1:
            groups.append({"target_row": len(compacted) - 1, "source_rows": current_source_rows})

    if not changed:
        return None, None
    return compacted, {
        "source": "rowspan_key_repetition_compaction",
        "source_row_count": len(rows),
        "compacted_row_count": len(compacted),
        "key_columns": key_cols,
        "description_columns": description_cols,
        "groups": groups,
    }


def _project_merged_adjacent_numeric_value_columns(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    """Split adjacent numeric columns that extraction merged into one cell."""
    if family not in {"comparison_matrix", "rowspan_grouped_table"}:
        return None, None
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 4:
        return None, None
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    header_rows = 1 if _row_looks_like_logical_header(rows[0], col_count) else 0
    body_rows = rows[header_rows:]
    if len(body_rows) < 3:
        return None, None

    candidate_pairs: list[tuple[int, int, int]] = []
    for left_col in range(1, col_count - 1):
        right_col = left_col + 1
        separated = 0
        merged = 0
        for row in body_rows:
            left = _cell_text(row[left_col] if left_col < len(row) else None)
            right = _cell_text(row[right_col] if right_col < len(row) else None)
            if left and right and _looks_like_atomic_numeric_value(left) and _looks_like_atomic_numeric_value(right):
                separated += 1
            elif left and not right and len(_split_numeric_value_atoms(left)) == 2:
                merged += 1
        if separated >= 2 and merged >= 1:
            candidate_pairs.append((left_col, right_col, merged))
    if not candidate_pairs:
        return None, None
    left_col, right_col, merged_count = max(candidate_pairs, key=lambda item: item[2])

    projected: list[list[str | None]] = []
    changed = False
    for row_idx, row in enumerate(rows):
        next_row = [_cell_text(cell) or None for cell in row[:col_count]]
        if row_idx >= header_rows:
            left = _cell_text(next_row[left_col])
            right = _cell_text(next_row[right_col])
            atoms = _split_numeric_value_atoms(left)
            if not right and len(atoms) == 2:
                next_row[left_col] = atoms[0]
                next_row[right_col] = atoms[1]
                changed = True
        projected.append(next_row)
    if not changed:
        return None, None
    return projected, {
        "source": "merged_adjacent_numeric_value_projection",
        "left_col": left_col,
        "right_col": right_col,
        "split_row_count": merged_count,
    }


def _split_numeric_value_atoms(text: str | None) -> list[str]:
    cleaned = _cell_text(text)
    if not cleaned:
        return []
    atoms = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?", cleaned)
    if not atoms:
        return []
    residual = cleaned
    for atom in atoms:
        residual = residual.replace(atom, " ", 1)
    if re.sub(r"[\s/;,\-鈥揃]+", "", residual):
        return []
    return atoms


def _looks_like_atomic_numeric_value(text: str | None) -> bool:
    return len(_split_numeric_value_atoms(text)) == 1


def _rowspan_repeated_key_columns(table: dict[str, Any], col_count: int) -> list[int]:
    counts = [0] * col_count
    for group in table.get("row_groups", []) or []:
        if not isinstance(group, dict):
            continue
        try:
            col = int(group.get("col", -1))
            rowspan = int(group.get("rowspan", 1) or 1)
        except (TypeError, ValueError):
            continue
        text = _cell_text(group.get("text"))
        if 0 <= col < col_count and rowspan >= 2 and text and _looks_like_repeated_key_group_text(text):
            counts[col] += 1
    return [idx for idx, count in enumerate(counts) if count >= 1]


def _looks_like_repeated_key_group_text(text: str | None) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_numeric(cleaned):
        return True
    return _looks_like_compact_key_cell(cleaned) or len(cleaned.split()) <= 4


def _rowspan_repeated_description_columns(
    grid: list[list[Any]],
    key_cols: list[int],
    col_count: int,
) -> list[int]:
    rows = _clone_grid(grid)
    body_rows = rows[1:] if rows and _row_looks_like_logical_header(rows[0], col_count) else rows
    counts = [0] * col_count
    for row in body_rows:
        normalized = [_cell_text(cell) for cell in row[:col_count]]
        if not any(_cell_text(normalized[idx]) for idx in key_cols):
            continue
        for idx, text in enumerate(normalized):
            if idx in key_cols:
                continue
            if _looks_like_long_description_continuation(text):
                counts[idx] += 1
    return [idx for idx, count in enumerate(counts) if idx not in key_cols and count >= 1]


def _rowspan_repeated_row_continues_current(
    row: list[str | None],
    key: tuple[str, ...],
    current_key: tuple[str, ...] | None,
    key_cols: list[int],
    description_cols: list[int],
    col_count: int,
) -> bool:
    if len(row) < col_count:
        row = list(row) + [None] * (col_count - len(row))
    if current_key is None:
        if any(key):
            return False
        non_description_values = [
            idx
            for idx in range(col_count)
            if idx not in key_cols and idx not in description_cols and _cell_text(row[idx])
        ]
        if non_description_values:
            return False
        return any(_cell_text(row[idx]) for idx in description_cols)
    if any(key) and key != current_key:
        return False
    non_description_values = [
        idx
        for idx in range(col_count)
        if idx not in key_cols and idx not in description_cols and _cell_text(row[idx])
    ]
    if non_description_values:
        return False
    return any(_cell_text(row[idx]) for idx in description_cols)


def _compact_multicolumn_wrapped_record_rows(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
    *,
    projected_header_rows: int | None = None,
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    if family not in {"comparison_matrix", "rowspan_grouped_table"}:
        return None, None
    col_count = _col_count(table, grid)
    if col_count < 4 or len(grid) < 4:
        return None, None
    rows = _clone_grid(grid)
    if projected_header_rows is not None:
        header_rows = max(0, min(projected_header_rows, len(rows) - 1))
    else:
        header_rows = 1 if _row_looks_like_logical_header(rows[0], col_count) else 0
    body_rows = rows[header_rows:]
    if len(body_rows) < 3:
        return None, None

    description_cols = _wrapped_record_description_columns(body_rows, col_count)
    if not description_cols:
        return None, None
    first_description_col = min(description_cols)
    if first_description_col <= 0:
        return None, None

    compacted: list[list[str | None]] = []
    if header_rows:
        compacted.extend([[_cell_text(cell) or None for cell in row[:col_count]] for row in rows[:header_rows]])

    current: list[str | None] | None = None
    current_source_rows: list[int] = []
    groups: list[dict[str, Any]] = []
    changed = False
    for body_offset, row in enumerate(body_rows, start=header_rows):
        normalized = [_cell_text(cell) or None for cell in row[:col_count]]
        if not any(normalized):
            continue
        if current is not None and _is_multicolumn_wrapped_record_continuation(
            normalized,
            current,
            first_description_col,
            col_count,
        ):
            for idx, text in enumerate(normalized):
                if not text:
                    continue
                current[idx] = _join_wrapped_description_text(current[idx], text) if current[idx] else text
            current_source_rows.append(body_offset)
            changed = True
            continue
        if current is not None:
            compacted.append(current)
            if len(current_source_rows) > 1:
                groups.append({"target_row": len(compacted) - 1, "source_rows": current_source_rows})
        current = normalized
        current_source_rows = [body_offset]

    if current is not None:
        compacted.append(current)
        if len(current_source_rows) > 1:
            groups.append({"target_row": len(compacted) - 1, "source_rows": current_source_rows})

    if not changed or len(compacted) >= len(rows):
        return None, None
    return compacted, {
        "source": "multicolumn_wrapped_record_compaction",
        "source_row_count": len(rows),
        "compacted_row_count": len(compacted),
        "description_columns": description_cols,
        "groups": groups,
    }


def _wrapped_record_description_columns(rows: list[list[Any]], col_count: int) -> list[int]:
    long_counts = [0] * col_count
    compact_key_counts = [0] * col_count
    continuation_counts = [0] * col_count
    unkeyed_continuation_counts = [0] * col_count
    key_role_counts = [0] * col_count
    for row in rows:
        normalized = [_cell_text(cell) for cell in row[:col_count]]
        filled = [idx for idx, text in enumerate(normalized) if text]
        has_stub_key = bool(_cell_text(normalized[0] if normalized else ""))
        for idx, text in enumerate(normalized):
            if _looks_like_long_description_continuation(text):
                long_counts[idx] += 1
            if _looks_like_compact_key_cell(text):
                compact_key_counts[idx] += 1
            if idx > 0 and text and (len(filled) > 1 or idx != filled[0]) and _looks_like_wrapped_cell_continuation_fragment(text):
                continuation_fragment = True
            else:
                continuation_fragment = False
            if _looks_like_wrapped_record_key_role_cell(text, continuation_fragment=continuation_fragment):
                key_role_counts[idx] += 1
            if idx > 0 and text and (len(filled) > 1 or idx != filled[0]) and _looks_like_wrapped_cell_continuation_fragment(text):
                continuation_counts[idx] += 1
                if not has_stub_key:
                    unkeyed_continuation_counts[idx] += 1
    minimum = 2 if len(rows) < 8 else 3
    candidates = [
        idx
        for idx, count in enumerate(long_counts)
        if (
            (
                count >= minimum
                and count > compact_key_counts[idx]
                and _wrapped_record_column_has_description_dominance(
                    long_counts[idx],
                    continuation_counts[idx],
                    key_role_counts[idx],
                    compact_key_counts[idx],
                    row_count=len(rows),
                )
            )
            or (
                count + unkeyed_continuation_counts[idx] >= minimum
                and continuation_counts[idx] > 0
                and count + continuation_counts[idx] >= compact_key_counts[idx]
                and _wrapped_record_column_has_description_dominance(
                    long_counts[idx] + unkeyed_continuation_counts[idx],
                    continuation_counts[idx],
                    key_role_counts[idx],
                    compact_key_counts[idx],
                    row_count=len(rows),
                )
            )
        )
    ]
    non_stub_candidates = [idx for idx in candidates if idx > 0]
    return non_stub_candidates or candidates


def _wrapped_record_column_has_description_dominance(
    description_count: int,
    continuation_count: int,
    key_role_count: int,
    compact_key_count: int,
    *,
    row_count: int,
) -> bool:
    if key_role_count <= 1:
        return True
    if (
        key_role_count >= max(3, row_count // 4)
        and compact_key_count >= continuation_count
        and description_count <= key_role_count + continuation_count
    ):
        return False
    if description_count >= key_role_count + 2:
        return True
    if description_count >= max(4, key_role_count + 2) and continuation_count >= 2:
        return True
    return description_count / max(1, row_count) >= 0.65 and description_count > key_role_count


def _looks_like_wrapped_record_key_role_cell(text: str | None, *, continuation_fragment: bool = False) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if re.match(r"^\s*(?:\d+(?:\.\d+)*|[A-Za-z]|[ivxlcdmIVXLCDM]+)[\.)]\s+\S+", cleaned):
        return True
    if continuation_fragment and _looks_like_long_description_continuation(cleaned):
        return False
    return _looks_like_compact_key_cell(cleaned) or _looks_like_comparison_axis_header(cleaned)


def _looks_like_wrapped_cell_continuation_fragment(text: str | None) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_numeric(cleaned):
        return False
    if re.match(r"^(?:and|or|of|in|to|for|from|with|within|without|subject|except|resident|those)\b", cleaned, re.IGNORECASE):
        return True
    if cleaned.endswith((",", ";", ":", "-", "/")):
        return True
    if len(cleaned.split()) >= 2 and re.search(r"\b(?:and|or|of|for|to|with|in|on|by)\b$", cleaned, re.IGNORECASE):
        return True
    return len(cleaned.split()) >= 2 and not _looks_like_compact_key_cell(cleaned)


def _is_multicolumn_wrapped_record_continuation(
    row: list[str | None],
    current: list[str | None],
    first_description_col: int,
    col_count: int,
) -> bool:
    filled = [idx for idx, text in enumerate(row[:col_count]) if _cell_text(text)]
    if not filled:
        return False
    key_cols = [idx for idx in filled if idx < first_description_col]
    description_cols = [idx for idx in filled if idx >= first_description_col]
    if not key_cols and description_cols:
        if _row_has_new_record_signal_after_stub(row, current, first_description_col, col_count):
            return False
        return True
    if not key_cols:
        return False
    leftmost_key_col = min(key_cols)
    if _looks_like_wrapped_key_cell_fragment(row[leftmost_key_col], current[leftmost_key_col]):
        for idx in key_cols:
            if idx == leftmost_key_col:
                continue
            if current[idx] and not _looks_like_wrapped_key_cell_fragment(row[idx], current[idx]):
                return False
        return True
    if all(_looks_like_wrapped_key_cell_fragment(row[idx], current[idx]) for idx in key_cols):
        return True
    return False


def _row_has_new_record_signal_after_stub(
    row: list[str | None],
    current: list[str | None],
    first_description_col: int,
    col_count: int,
) -> bool:
    for idx in range(1, min(first_description_col, col_count)):
        text = _cell_text(row[idx])
        if not text:
            continue
        previous = _cell_text(current[idx] if idx < len(current) else None)
        if _looks_like_wrapped_key_cell_fragment(text, previous):
            continue
        if _looks_like_compact_key_cell(text) or _looks_like_comparison_axis_header(text):
            return True
    return False


def _looks_like_wrapped_key_cell_fragment(text: str | None, previous: str | None) -> bool:
    cleaned = _cell_text(text)
    prior = _cell_text(previous)
    if not cleaned or not prior:
        return False
    if re.match(r"^[a-z]", cleaned):
        return True
    if prior.endswith(",") and re.fullmatch(r"\d{2,4}", cleaned):
        return True
    if re.search(r"(?:\b(?:and|or|of|for|to|with)|[/,\-])\s*$", prior, re.IGNORECASE):
        return True
    return len(cleaned.split()) == 1 and len(cleaned) <= 14 and len(prior.split()) >= 3


def _row_looks_like_logical_header(row: list[Any], col_count: int) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [text for text in texts if text]
    if len(filled) < max(2, col_count // 2):
        return False
    if any(_looks_like_long_body_fragment(text) for text in filled):
        return False
    return True


def _dominant_long_description_column(rows: list[list[Any]], col_count: int) -> int | None:
    long_counts = [0] * col_count
    continuation_counts = [0] * col_count
    for row in rows:
        normalized_row = [_cell_text(cell) for cell in row[:col_count]]
        filled = [idx for idx, text in enumerate(normalized_row) if text]
        for idx, text in enumerate(normalized_row):
            if _looks_like_long_description_continuation(text):
                long_counts[idx] += 1
        if len(filled) == 1 and _looks_like_long_description_continuation(normalized_row[filled[0]]):
            continuation_counts[filled[0]] += 1
    candidates = [
        idx
        for idx in range(col_count)
        if long_counts[idx] >= 3 and continuation_counts[idx] >= 2
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda idx: (continuation_counts[idx], long_counts[idx], idx))


def _is_keyed_long_description_record_start(row: list[str | None], description_col: int, col_count: int) -> bool:
    filled = [idx for idx, text in enumerate(row[:col_count]) if _cell_text(text)]
    if len(filled) < 2:
        return False
    if description_col not in filled:
        return False
    keyed_cols = [idx for idx in filled if idx != description_col]
    if not keyed_cols:
        return False
    compact_key_cells = sum(1 for idx in keyed_cols if _looks_like_compact_key_cell(row[idx]))
    return compact_key_cells >= max(1, min(3, len(keyed_cols)))


def _looks_like_compact_key_cell(text: str | None) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if len(cleaned) <= 3 and re.fullmatch(r"[A-Za-z0-9()+/\-]+", cleaned):
        return True
    return len(cleaned) <= 32 and len(cleaned.split()) <= 3 and bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned))


def _looks_like_long_description_continuation(text: str | None) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if len(re.findall(r"[\u4e00-\u9fff]", cleaned)) >= 6:
        return True
    return len(cleaned) >= 16 and len(cleaned.split()) >= 2


def _looks_like_wrapped_description_continuation(text: str | None, previous_text: str | None) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_like_long_description_continuation(cleaned):
        return True
    previous = _cell_text(previous_text)
    if not previous:
        return False
    if len(cleaned.split()) <= 5 and re.match(r"^(?:and|or|of|in|to|for|from|with|within|without|subject|except|resident|those)\b", cleaned, re.IGNORECASE):
        return True
    if len(cleaned) <= 32 and re.search(r"[.;,，。；]\s*$", cleaned):
        return True
    return False


def _join_wrapped_description_text(left: str | None, right: str | None) -> str:
    prefix = _cell_text(left)
    suffix = _cell_text(right)
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    if prefix.endswith("-") and suffix[:1].islower():
        return prefix[:-1] + suffix
    return f"{prefix} {suffix}".strip()


def _build_semantic_header(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
    *,
    header_override: list[str] | None = None,
) -> list[dict[str, Any]]:
    texts = list(header_override or [])
    if not texts:
        texts = _header_texts(table)
    if not texts and grid:
        texts = [_cell_text(cell) for cell in grid[0]]
    col_count = max(_col_count(table, grid), len(texts))
    while len(texts) < col_count:
        texts.append("")
    header: list[dict[str, Any]] = []
    for col_idx, text in enumerate(texts[:col_count]):
        role = "column_header"
        if col_idx == 0 and family in {"comparison_matrix", "rowspan_grouped_table", "spreadsheet_matrix"}:
            role = "projected_row_header"
        header.append({"col": col_idx + 1, "text": text, "role": role})
    return header


def _infer_sparse_header_span_groups(
    grid: list[list[Any]],
    family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    if family not in {"comparison_matrix", "rowspan_grouped_table", "projected_stub_matrix"}:
        return [], [], None
    col_count = max((len(row) for row in grid), default=0)
    if col_count < 3 or len(grid) < 4:
        return [], [], None
    rows = _rectangular_grid(grid, col_count)
    header_depth = _sparse_header_band_depth(rows, col_count)
    if header_depth < 2:
        return [], [], None

    data_start_col = _sparse_header_data_start_col(rows, header_depth, col_count)
    if data_start_col >= col_count - 1:
        return [], [], None

    column_groups: list[dict[str, Any]] = []
    for row_idx in range(header_depth - 1):
        row = rows[row_idx]
        filled = [
            (col_idx, _cell_text(row[col_idx]))
            for col_idx in range(data_start_col, col_count)
            if _cell_text(row[col_idx])
        ]
        if not filled:
            continue
        if len(filled) == 1:
            col_idx, text = filled[0]
            start_col = data_start_col
            end_col = col_count - 1
            if start_col <= col_idx <= end_col and end_col > start_col:
                column_groups.append(
                    {
                        "row": row_idx,
                        "start_col": start_col,
                        "end_col": end_col,
                        "text": text,
                        "source": "sparse_header_colspan_projection",
                    }
                )
            continue
        label_cols = [col_idx for col_idx, _ in filled]
        for index, (col_idx, text) in enumerate(filled):
            start_col = data_start_col if index == 0 else ((label_cols[index - 1] + col_idx) // 2) + 1
            end_col = col_count - 1 if index == len(filled) - 1 else (col_idx + label_cols[index + 1]) // 2
            if start_col <= col_idx <= end_col and end_col > start_col:
                column_groups.append(
                    {
                        "row": row_idx,
                        "start_col": start_col,
                        "end_col": end_col,
                        "text": text,
                        "source": "sparse_header_colspan_projection",
                    }
                )

    row_groups: list[dict[str, Any]] = []
    if data_start_col > 0:
        for col_idx in range(data_start_col):
            filled = [
                (row_idx, _cell_text(rows[row_idx][col_idx]))
                for row_idx in range(header_depth)
                if _cell_text(rows[row_idx][col_idx])
            ]
            if len(filled) != 1:
                continue
            label_row, text = filled[0]
            if header_depth > 1 and label_row > 0:
                row_groups.append(
                    {
                        "col": col_idx,
                        "start_row": 0,
                        "rowspan": header_depth,
                        "text": text,
                        "source": "sparse_header_rowspan_projection",
                    }
                )

    if not column_groups and not row_groups:
        return [], [], None
    return column_groups, row_groups, {
        "source": "sparse_header_span_projection",
        "header_row_count": header_depth,
        "data_start_col": data_start_col,
        "column_group_count": len(column_groups),
        "row_group_count": len(row_groups),
    }


def _fill_sparse_multilevel_blank_header_leaf_from_context(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
    header_column_groups: list[dict[str, Any]],
    header_row_groups: list[dict[str, Any]],
    sparse_header_projection: dict[str, Any] | None,
) -> tuple[list[list[str | None]] | None, dict[str, Any] | None]:
    if family not in {"comparison_matrix", "rowspan_grouped_table", "projected_stub_matrix"}:
        return None, None
    if not sparse_header_projection or not header_column_groups:
        return None, None
    col_count = max((len(row) for row in grid), default=0)
    if col_count < 3:
        return None, None
    rows = _rectangular_grid(grid, col_count)
    try:
        header_depth = int(sparse_header_projection.get("header_row_count") or 0)
    except Exception:
        header_depth = 0
    if header_depth < 2 or header_depth >= len(rows):
        return None, None

    leaf_row_idx = header_depth - 1
    blank_fills: list[tuple[int, str, dict[str, Any]]] = []
    for group in header_column_groups:
        if not isinstance(group, dict):
            continue
        try:
            start_col = int(group.get("start_col"))
            end_col = int(group.get("end_col"))
        except Exception:
            continue
        if start_col < 0 or end_col >= col_count or end_col <= start_col:
            continue
        blank_cols = [col_idx for col_idx in range(start_col, end_col + 1) if not _cell_text(rows[leaf_row_idx][col_idx])]
        if len(blank_cols) != 1:
            continue
        known_leafs = [
            _cell_text(rows[leaf_row_idx][col_idx])
            for col_idx in range(start_col, end_col + 1)
            if _cell_text(rows[leaf_row_idx][col_idx])
        ]
        if len(known_leafs) < 2:
            continue
        fill_value = _unique_caption_context_header_leaf_candidate(
            table,
            rows,
            header_depth,
            start_col,
            end_col,
            known_leafs,
        )
        if not fill_value:
            continue
        blank_fills.append(
            (
                blank_cols[0],
                fill_value,
                {
                    "row": leaf_row_idx,
                    "col": blank_cols[0],
                    "parent_header": _cell_text(group.get("text")),
                    "text": fill_value,
                },
            )
        )

    if not blank_fills:
        return None, None
    filled_grid = _clone_grid(rows)
    filled_columns: list[dict[str, Any]] = []
    for col_idx, fill_value, info in blank_fills:
        filled_grid[leaf_row_idx][col_idx] = fill_value
        filled_columns.append(info)
    return filled_grid, {
        "source": "caption_context_blank_header_leaf_fill",
        "header_row_count": header_depth,
        "filled_column_count": len(filled_columns),
        "filled_columns": filled_columns,
        "row_group_count": len(header_row_groups),
    }


def _project_compressed_multilevel_metric_header_groups(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
    structure_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Infer multilevel header spans when borderless extraction packed header text into the row stub.

    Some IND text-layer tables have clear data columns but no ruling lines. When a
    preceding header/template fragment is reclaimed into the table, the visible
    multilevel header may arrive as text in the first column while the remaining
    header cells are blank. This function records the logical colspan semantics
    from schema evidence and numeric body support without rewriting the observed
    grid.
    """

    if family not in {"comparison_matrix", "projected_stub_matrix", "rowspan_grouped_table"}:
        return [], None
    profile_family = str(structure_profile.get("structure_family") or "")
    if profile_family not in {"aligned_text_grid", "clustered_text_table"}:
        return [], None
    col_count = _col_count(table, grid)
    if col_count < 5 or len(grid) < 6:
        return [], None
    rows = _rectangular_grid(grid, col_count)
    if not _has_compressed_header_stub_rows(rows):
        return [], None

    value_cols = _supported_numeric_value_columns(rows)
    if len(value_cols) < 4:
        return [], None
    contiguous_runs = _contiguous_index_runs(value_cols)
    primary_value_run = max(contiguous_runs, key=len) if contiguous_runs else []
    if len(primary_value_run) < 4:
        return [], None

    metric_text = _metric_header_text_from_compressed_rows(rows)
    if not metric_text:
        return [], None

    group_labels = _compressed_header_group_labels(rows)
    if len(group_labels) < 2:
        return [], None
    groups_by_span = _assign_compressed_group_labels_to_value_columns(group_labels, primary_value_run, rows)
    if len(groups_by_span) < 2:
        return [], None

    metric_start = min(start for _, start, _ in groups_by_span)
    metric_end = max(end for _, _, end in groups_by_span)
    if metric_end - metric_start + 1 < 4:
        return [], None

    header_groups: list[dict[str, Any]] = [
        {
            "row": 0,
            "start_col": metric_start,
            "end_col": metric_end,
            "colspan": metric_end - metric_start + 1,
            "text": metric_text,
            "source": "compressed_multilevel_metric_header_projection",
        }
    ]
    for label, start_col, end_col in groups_by_span:
        if end_col > start_col:
            header_groups.append(
                {
                    "row": 1,
                    "start_col": start_col,
                    "end_col": end_col,
                    "colspan": end_col - start_col + 1,
                    "text": label,
                    "source": "compressed_multilevel_metric_header_projection",
                }
            )

    if len(header_groups) < 3:
        return [], None
    return header_groups, {
        "source": "compressed_multilevel_metric_header_projection",
        "semantic_profile": "compressed_metric_multilevel_header",
        "metric_text": metric_text,
        "metric_start_col": metric_start,
        "metric_end_col": metric_end,
        "value_columns": primary_value_run,
        "group_count": len(groups_by_span),
        "groups": [
            {"text": label, "start_col": start_col, "end_col": end_col}
            for label, start_col, end_col in groups_by_span
        ],
    }


def _has_compressed_header_stub_rows(rows: list[list[Any]]) -> bool:
    if len(rows) < 2:
        return False
    first = [_cell_text(cell) for cell in rows[0]]
    second = [_cell_text(cell) for cell in rows[1]]
    if not first or not second:
        return False
    first_stub = first[0]
    second_stub = second[0]
    if not first_stub or not second_stub:
        return False
    if any(first[col] for col in range(1, len(first))):
        return False
    if any(second[col] for col in range(1, len(second))):
        return False
    return _looks_like_metric_header_text(first_stub) and _looks_like_compressed_group_header_text(second_stub)


def _looks_like_metric_header_text(text: str) -> bool:
    compact = _normalize_text(text)
    if not compact:
        return False
    has_metric = bool(re.search(r"(?:AUC|Cmax|Tmax|T1/2|Css|浓度|含量|暴露量|回收率|百分比)", compact, re.IGNORECASE))
    has_unit = bool(re.search(r"[()（）/]|(?:mg|kg|ml|ng|pg|g|h|µg|μg|ug|%)", compact, re.IGNORECASE))
    return has_metric and has_unit


def _looks_like_compressed_group_header_text(text: str) -> bool:
    compact = _normalize_text(text)
    if not compact:
        return False
    has_stub = bool(re.search(r"(?:日剂量|剂量|浓度|时间|周期|参数|单位)", compact))
    has_group = bool(re.search(r"(?:小鼠|大鼠|犬|猴|兔|人|雄性|雌性|M\b|F\b|Male|Female)", compact, re.IGNORECASE))
    return has_stub and has_group


def _supported_numeric_value_columns(rows: list[list[Any]]) -> list[int]:
    if not rows:
        return []
    col_count = max((len(row) for row in rows), default=0)
    supported: list[int] = []
    body_rows = rows[2:]
    for col_idx in range(1, col_count):
        numeric_count = 0
        non_empty_count = 0
        for row in body_rows:
            text = _cell_text(row[col_idx] if col_idx < len(row) else None)
            if not text:
                continue
            non_empty_count += 1
            if _looks_numeric_or_marker_value(text):
                numeric_count += 1
        if numeric_count >= 2 and numeric_count >= max(1, non_empty_count - 1):
            supported.append(col_idx)
    return supported


def _looks_numeric_or_marker_value(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_numeric(cleaned):
        return True
    return bool(re.fullmatch(r"[\d,.\s]+[A-Za-z]*[、,，;；]?[\d,.\sA-Za-z、,，;；-]*", cleaned))


def _contiguous_index_runs(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    ordered = sorted(set(values))
    runs: list[list[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return runs


def _metric_header_text_from_compressed_rows(rows: list[list[Any]]) -> str:
    text = _cell_text(rows[0][0] if rows and rows[0] else None)
    if not text:
        return ""
    return _normalize_metric_unit_header_text(text)


def _normalize_metric_unit_header_text(text: str) -> str:
    cleaned = _cell_text(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("μ", "µ")
    return cleaned


def _compressed_header_group_labels(rows: list[list[Any]]) -> list[str]:
    second = _cell_text(rows[1][0] if len(rows) > 1 and rows[1] else None)
    if not second:
        return []
    return [
        _cell_text(match.group(0))
        for match in re.finditer(r"(?:小鼠|大鼠|犬|猴|兔|人|雄性|雌性)[A-Za-z]?", second)
    ]


def _assign_compressed_group_labels_to_value_columns(
    labels: list[str],
    value_cols: list[int],
    rows: list[list[Any]],
) -> list[tuple[str, int, int]]:
    if not labels or not value_cols:
        return []
    leaf_count = _infer_repeated_leaf_count_for_compressed_header(rows, value_cols)
    if leaf_count <= 0:
        return []
    max_groups = len(value_cols) // leaf_count
    usable_labels = labels[:max_groups]
    groups: list[tuple[str, int, int]] = []
    for index, label in enumerate(usable_labels):
        start_offset = index * leaf_count
        end_offset = start_offset + leaf_count - 1
        if end_offset >= len(value_cols):
            break
        groups.append((label, value_cols[start_offset], value_cols[end_offset]))
    return groups


def _infer_repeated_leaf_count_for_compressed_header(rows: list[list[Any]], value_cols: list[int]) -> int:
    second = _cell_text(rows[1][0] if len(rows) > 1 and rows[1] else None)
    if re.search(r"\bM\b", second, re.IGNORECASE) and re.search(r"\bF\b", second, re.IGNORECASE):
        return 2
    if re.search(r"雄性", second) and re.search(r"雌性", second):
        return 2
    return 2 if len(value_cols) >= 4 and len(value_cols) % 2 == 0 else 0


def _unique_caption_context_header_leaf_candidate(
    table: dict[str, Any],
    rows: list[list[Any]],
    header_depth: int,
    start_col: int,
    end_col: int,
    known_leafs: list[str],
) -> str:
    context = _table_caption_context_text(table)
    if not context:
        return ""
    context_norm = _normalize_table_header_match_text(context)
    if not context_norm:
        return ""
    known_norms = {_normalize_table_header_match_text(text) for text in known_leafs if _cell_text(text)}
    known_norms.discard("")
    candidates = _caption_context_header_phrase_candidates(context)
    if not candidates:
        return ""
    used_header_norms = {
        _normalize_table_header_match_text(_cell_text(rows[row_idx][col_idx]))
        for row_idx in range(header_depth)
        for col_idx in range(start_col, end_col + 1)
        if _cell_text(rows[row_idx][col_idx])
    }
    used_header_norms |= known_norms
    unused: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = _cell_text(candidate)
        norm = _normalize_table_header_match_text(cleaned)
        if not norm or norm in seen or norm in used_header_norms:
            continue
        if any(norm in known or known in norm for known in known_norms):
            continue
        if not _caption_candidate_matches_header_leaf_style(cleaned, known_leafs):
            continue
        if not _caption_candidate_supported_by_body_values(cleaned, rows, header_depth, start_col, end_col):
            continue
        if _normalize_table_header_match_text(cleaned) not in context_norm:
            continue
        seen.add(norm)
        unused.append(cleaned)
    return unused[0] if len(unused) == 1 else ""


def _table_caption_context_text(table: dict[str, Any]) -> str:
    parts = [
        _cell_text(table.get(key))
        for key in ("title", "caption_text", "caption", "nearby_text", "context_text")
        if _cell_text(table.get(key))
    ]
    result: list[str] = []
    seen: set[str] = set()
    for text in parts:
        signature = _normalize_text(text)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        result.append(text)
    return " ".join(result)


def _caption_context_header_phrase_candidates(text: str) -> list[str]:
    cleaned = _cell_text(text)
    if not cleaned:
        return []
    raw_candidates = re.findall(
        r"[A-Z0-9][A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*(?:\.)?"
        r"(?:\s+[A-Z0-9][A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*(?:\.)?){0,4}",
        cleaned,
    )
    candidates: list[str] = []
    seen: set[str] = set()
    stop_norms = {
        "the",
        "for",
        "and",
        "or",
        "used",
        "training",
        "datasets",
        "properties",
        "total",
        "maximum",
    }
    for candidate in raw_candidates:
        trimmed = _trim_caption_context_candidate(candidate)
        if not trimmed:
            continue
        norm = _normalize_table_header_match_text(trimmed)
        if not norm or norm in seen or norm in stop_norms:
            continue
        if len(trimmed) > 96 or len(trimmed.split()) > 5:
            continue
        if not _looks_like_comparison_axis_header(trimmed):
            continue
        seen.add(norm)
        candidates.append(trimmed)
    return candidates


def _trim_caption_context_candidate(text: str) -> str:
    cleaned = _cell_text(text).strip(" ,;:.()[]{}")
    if not cleaned:
        return ""
    leading_stop = {"For", "The", "A", "An", "And", "Or", "In", "On", "At", "To", "By", "We"}
    trailing_stop = {"and", "or", "the", "a", "an", "for", "of", "to", "by", "with"}
    tokens = cleaned.split()
    while tokens and tokens[0].strip(" ,;:.") in leading_stop:
        tokens = tokens[1:]
    while tokens and tokens[-1].strip(" ,;:.").lower() in trailing_stop:
        tokens = tokens[:-1]
    return _cell_text(" ".join(tokens).strip(" ,;:.()[]{}"))


def _caption_candidate_matches_header_leaf_style(candidate: str, known_leafs: list[str]) -> bool:
    cleaned = _cell_text(candidate)
    if not cleaned or not known_leafs:
        return False
    if _looks_numeric(cleaned) or _row_text_looks_like_prose_sentence(cleaned):
        return False
    known_with_markers = sum(1 for text in known_leafs if re.search(r"[-_/#]|\d", text))
    known_title_like = sum(1 for text in known_leafs if re.search(r"[A-Z]", text))
    if known_with_markers and not re.search(r"[-_/#]|\d", cleaned):
        return False
    if known_title_like and not re.search(r"[A-Z]", cleaned):
        return False
    return True


def _caption_candidate_supported_by_body_values(
    candidate: str,
    rows: list[list[Any]],
    header_depth: int,
    start_col: int,
    end_col: int,
) -> bool:
    candidate_norm = _normalize_table_header_match_text(candidate)
    if not candidate_norm:
        return False
    if header_depth >= len(rows):
        return False
    body = rows[header_depth : min(len(rows), header_depth + 4)]
    target_col = end_col
    if target_col < start_col or target_col >= max((len(row) for row in rows), default=0):
        return False
    values = [_cell_text(row[target_col]) for row in body if len(row) > target_col and _cell_text(row[target_col])]
    if len(values) < 2:
        return False
    value_like = sum(1 for value in values if _looks_like_compact_table_value(value) or _looks_numeric(value))
    return value_like >= max(2, len(values) // 2)


def _normalize_table_header_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", _normalize_text(value))


def _project_grouped_multilevel_borderless_table(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
    structure_profile: dict[str, Any],
) -> tuple[
    list[str] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    """Project weakly ruled visual tables into logical grouped schemas.

    This is intentionally a semantic-only layer. It is routed by evidence
    profile and table shape, then validated by header/body row geometry. It
    does not mutate raw or display grids, which keeps region evidence isolated
    from IND semantic interpretation.
    """

    if family not in {"comparison_matrix", "rowspan_grouped_table", "projected_stub_matrix"}:
        return None, [], [], [], None
    if str(table.get("semantic_role") or "business_table") != "business_table":
        return None, [], [], [], None
    border_model = str(structure_profile.get("border_model") or "")
    structure_family = str(structure_profile.get("structure_family") or "")
    if border_model not in {"borderless_aligned", "sparse_or_visual", "sparse_rules", "horizontal_rules"}:
        return None, [], [], [], None
    if structure_family not in {
        "aligned_text_grid",
        "visual_structure_table",
        "sparse_rule_table",
        "horizontal_rule_table",
    }:
        return None, [], [], [], None

    col_count = _col_count(table, grid)
    if col_count < 5 or len(grid) < 6:
        return None, [], [], [], None
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    header_depth = _sparse_header_band_depth(rows, col_count)
    if header_depth < 2:
        return None, [], [], [], None
    body_rows = rows[header_depth:]
    if len(body_rows) < 3:
        return None, [], [], [], None

    logical_header, diagnostics = _project_grouped_multilevel_header(rows, header_depth, col_count)
    if not logical_header:
        return None, [], [], [], None
    column_groups = _project_grouped_multilevel_header_column_groups(rows, header_depth, logical_header)
    row_header_groups = _project_grouped_multilevel_header_row_groups(rows, header_depth, logical_header)
    row_groups = _project_leading_stub_rowspan_groups_from_body(rows, header_depth, col_count)
    if not column_groups and not row_groups and not row_header_groups:
        return None, [], [], [], None

    filled_header_count = sum(1 for text in logical_header if _cell_text(text))
    if filled_header_count < max(4, len(logical_header) - 2):
        return None, [], [], [], None
    if not _grouped_multilevel_projection_has_body_support(body_rows, len(logical_header)):
        return None, [], [], [], None

    projection = {
        "source": "grouped_multilevel_borderless_semantic_projection",
        "semantic_profile": "grouped_multilevel_borderless_table",
        "header_row_count": header_depth,
        "source_col_count": col_count,
        "projected_col_count": len(logical_header),
        "border_model": border_model,
        "structure_family": structure_family,
        "header_column_groups": column_groups,
        "header_row_groups": row_header_groups,
        "row_groups": row_groups,
        "projection_confidence": "medium" if diagnostics.get("needs_packed_cell_review") else "high",
        **diagnostics,
    }
    return logical_header, column_groups, row_header_groups, row_groups, projection


def _project_grouped_multilevel_header(
    rows: list[list[Any]],
    header_depth: int,
    col_count: int,
) -> tuple[list[str] | None, dict[str, Any]]:
    header_rows = rows[:header_depth]
    body_rows = rows[header_depth:]
    packed_projection = _project_packed_two_row_grouped_header(header_rows, body_rows, col_count)
    if packed_projection:
        packed_header, packed_groups, packed_diagnostics = packed_projection
        return packed_header, packed_diagnostics | {"header_column_groups": packed_groups}

    logical_header: list[str] = []
    vertical_fusions: list[dict[str, Any]] = []
    for col_idx in range(col_count):
        parts = [_cell_text(row[col_idx]) for row in header_rows if col_idx < len(row) and _cell_text(row[col_idx])]
        if not parts:
            logical_header.append("")
            continue
        leaf = _select_grouped_multilevel_leaf_header(parts)
        if len(parts) > 1 and leaf == _join_schema_header_parts(parts):
            vertical_fusions.append({"col": col_idx, "parts": parts, "text": leaf})
        logical_header.append(leaf)

    logical_header = _repair_packed_grouped_multilevel_leaf_headers(logical_header, body_rows)
    logical_header = _expand_leading_stub_header_if_packed(logical_header, body_rows)
    logical_header = _compact_empty_grouped_header_columns(logical_header, body_rows)
    diagnostics = {
        "vertical_header_fusions": vertical_fusions,
        "needs_packed_cell_review": any(_grouped_header_cell_looks_packed(text) for text in logical_header),
    }
    if sum(1 for text in logical_header if text) < 4:
        return None, diagnostics
    return logical_header, diagnostics


def _project_packed_two_row_grouped_header(
    header_rows: list[list[Any]],
    body_rows: list[list[Any]],
    col_count: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]] | None:
    if len(header_rows) != 2 or col_count < 5:
        return None
    top_row = _rectangular_grid([header_rows[0]], col_count)[0]
    bottom_row = _rectangular_grid([header_rows[1]], col_count)[0]
    if not _leading_body_stub_looks_packed(body_rows):
        return None
    packed_header_cells = sum(
        1
        for col_idx in range(col_count)
        if len(_cell_text(bottom_row[col_idx]).split()) >= 2
    )
    parent_like_cells = sum(
        1
        for col_idx in range(col_count)
        if _cell_text(top_row[col_idx]) and _cell_text(bottom_row[col_idx])
    )
    if packed_header_cells < 2 or parent_like_cells < 2:
        return None

    entries: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    vertical_fusions: list[dict[str, Any]] = []

    def append_entry(text: str, source_col: int, role: str = "leaf") -> int:
        entries.append({"text": _cell_text(text), "source_col": source_col, "role": role})
        return len(entries) - 1

    def attach_to_previous_parent(child_index: int) -> bool:
        if not groups:
            return False
        child = entries[child_index]
        if not _grouped_orphan_leaf_can_extend_previous_parent(str(child.get("text") or "")):
            return False
        group = groups[-1]
        child_indices = list(group.get("_child_indices", []) or [])
        if child_index not in child_indices:
            child_indices.append(child_index)
        group["_child_indices"] = child_indices
        group["start_col"] = min(child_indices)
        group["end_col"] = max(child_indices)
        group["child_headers"] = [entries[idx]["text"] for idx in sorted(child_indices)]
        return True

    for col_idx in range(col_count):
        top = _cell_text(top_row[col_idx])
        bottom = _cell_text(bottom_row[col_idx])
        if not top and not bottom:
            continue
        if col_idx == 0 and not top and len(bottom.split()) >= 2:
            for token in bottom.split():
                append_entry(token, col_idx, "stub_header")
            continue
        if top and bottom:
            tokens = bottom.split()
            if len(tokens) > 1 and _grouped_header_tail_completes_previous(top, tokens[0]):
                merged = _join_schema_header_parts([top, tokens[0]])
                append_entry(merged, col_idx)
                vertical_fusions.append({"col": len(entries) - 1, "parts": [top, tokens[0]], "text": merged})
                for token in tokens[1:]:
                    append_entry(token, col_idx, "orphan_leaf")
                continue
            if len(tokens) > 1:
                suffix_count = _packed_header_suffix_child_count_from_body(body_rows, col_idx, col_count)
                prefix_tokens = tokens[:-suffix_count] if suffix_count > 0 else []
                suffix_tokens = tokens[-suffix_count:] if suffix_count > 0 else tokens
                for token in prefix_tokens:
                    child_idx = append_entry(token, col_idx, "orphan_leaf")
                    attach_to_previous_parent(child_idx)
                child_indices = [append_entry(token, col_idx, "child_header") for token in suffix_tokens]
                if len(child_indices) >= 2:
                    groups.append(
                        {
                            "row": 0,
                            "start_col": min(child_indices),
                            "end_col": max(child_indices),
                            "text": top,
                            "child_headers": [entries[idx]["text"] for idx in child_indices],
                            "_child_indices": child_indices,
                            "source": "grouped_multilevel_borderless_colspan_projection",
                        }
                    )
                continue
            if _single_bottom_token_looks_like_child_header(bottom):
                child_indices: list[int] = []
                if entries and entries[-1].get("role") == "orphan_leaf":
                    child_indices.append(len(entries) - 1)
                child_indices.append(append_entry(bottom, col_idx, "child_header"))
                if len(child_indices) >= 2:
                    groups.append(
                        {
                            "row": 0,
                            "start_col": min(child_indices),
                            "end_col": max(child_indices),
                            "text": top,
                            "child_headers": [entries[idx]["text"] for idx in child_indices],
                            "_child_indices": child_indices,
                            "source": "grouped_multilevel_borderless_colspan_projection",
                        }
                    )
                    continue
            merged = _join_schema_header_parts([top, bottom])
            append_entry(merged, col_idx)
            vertical_fusions.append({"col": len(entries) - 1, "parts": [top, bottom], "text": merged})
            continue
        if top:
            append_entry(top, col_idx)
            continue
        for token in bottom.split():
            append_entry(token, col_idx)

    logical_header = [str(entry.get("text") or "") for entry in entries if _cell_text(entry.get("text"))]
    if len(logical_header) < col_count:
        return None
    public_groups: list[dict[str, Any]] = []
    for group in groups:
        public_group = {key: value for key, value in group.items() if key != "_child_indices"}
        if int(public_group.get("end_col", 0) or 0) > int(public_group.get("start_col", 0) or 0):
            public_groups.append(public_group)
    diagnostics = {
        "source": "packed_two_row_grouped_header_projection",
        "vertical_header_fusions": vertical_fusions,
        "packed_header_source_col_count": packed_header_cells,
    }
    return logical_header, public_groups, diagnostics


def _project_grouped_multilevel_word_logical_grid(
    table: dict[str, Any],
    logical_header: list[str] | None,
    grouped_projection: dict[str, Any],
) -> tuple[list[list[str | None]] | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """Re-project packed weak-grid result matrices from word-level anchors.

    Visual structure grids often preserve the region but pack adjacent weakly
    separated columns into one cell. When the grouped-header layer has already
    proven a multilevel schema, word evidence can safely recover the logical
    columns without rewriting the raw/display evidence grid.
    """

    if str(grouped_projection.get("semantic_profile") or "") != "grouped_multilevel_borderless_table":
        return None, [], [], None
    table_bbox = _bbox(table)
    if table_bbox is None:
        return None, [], [], None
    words = [
        word
        for word in _table_word_evidence(table)
        if _word_inside_bbox(word, table_bbox, x_pad=8.0, y_pad=3.0)
    ]
    if len(words) < 20:
        return None, [], [], None
    lines = _cluster_flowchart_word_lines(words)
    if len(lines) < 5:
        return None, [], [], None

    header_lines = lines[:2]
    body_lines = lines[2:]
    leaf_specs = _grouped_word_leaf_specs_from_header_lines(header_lines)
    if len(leaf_specs) < 6:
        return None, [], [], None
    header = [_cell_text(spec.get("text")) for spec in leaf_specs]
    if logical_header:
        logical_terms = {_normalize_table_header_match_text(text) for text in logical_header if _cell_text(text)}
        header_terms = {_normalize_table_header_match_text(text) for text in header if _cell_text(text)}
        if len(logical_terms & header_terms) < max(4, min(len(header_terms), len(logical_terms)) // 2):
            return None, [], [], None

    centers = [float(spec.get("x", 0.0) or 0.0) for spec in leaf_specs]
    if len(centers) != len(header):
        return None, [], [], None
    projected_rows: list[list[str | None]] = [[text or None for text in header]]
    for line in body_lines:
        line_text = _cell_text(" ".join(str(word.get("text") or "") for word in line))
        if not line_text or _grouped_word_line_is_table_note(line_text):
            continue
        projected = _project_grouped_word_line_to_leaf_columns(line, centers)
        if not any(_cell_text(cell) for cell in projected):
            continue
        if _grouped_word_projected_row_is_viable(projected):
            projected_rows.append(projected)

    if len(projected_rows) < 4:
        return None, [], [], None
    body_value_rows = projected_rows[1:]
    numeric_value_rows = sum(
        1
        for row in body_value_rows
        if sum(1 for cell in row[2:] if _cell_text(cell) and (_looks_numeric(_cell_text(cell)) or _looks_like_compact_table_value(_cell_text(cell)))) >= 3
    )
    if numeric_value_rows < 3:
        return None, [], [], None

    header_groups = _grouped_word_header_column_groups(header_lines[0], leaf_specs)
    row_groups = _grouped_word_leading_stub_row_groups(projected_rows)
    return projected_rows, header_groups, row_groups, {
        "source": "grouped_multilevel_word_anchor_projection",
        "semantic_profile": "grouped_multilevel_logical_grid",
        "header_row_count": 1,
        "source_word_line_count": len(lines),
        "projected_row_count": len(projected_rows),
        "projected_col_count": len(header),
        "projection_confidence": "high",
    }


def _word_inside_bbox(
    word: dict[str, Any],
    bbox: tuple[float, float, float, float],
    *,
    x_pad: float = 0.0,
    y_pad: float = 0.0,
) -> bool:
    word_bbox = word.get("bbox")
    if not isinstance(word_bbox, (list, tuple)) or len(word_bbox) < 4:
        return False
    x = _word_x_center(word)
    y = _word_y_center(word)
    return bbox[0] - x_pad <= x <= bbox[2] + x_pad and bbox[1] - y_pad <= y <= bbox[3] + y_pad


def _grouped_word_leaf_specs_from_header_lines(lines: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if len(lines) < 2:
        return []
    top_line = sorted(lines[0], key=_word_x_center)
    bottom_line = sorted(lines[1], key=_word_x_center)
    specs: list[dict[str, Any]] = []
    for word in bottom_line:
        text = _cell_text(word.get("text"))
        if not text:
            continue
        top = _nearest_vertical_header_fragment(word, top_line)
        if top is not None and _grouped_header_word_fragments_should_merge(_cell_text(top.get("text")), text):
            text = _join_schema_header_parts([_cell_text(top.get("text")), text])
        specs.append({"text": text, "x": _word_x_center(word), "source_word": word})
    return specs


def _nearest_vertical_header_fragment(
    word: dict[str, Any],
    top_line: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not top_line:
        return None
    x = _word_x_center(word)
    candidate = min(top_line, key=lambda item: abs(_word_x_center(item) - x))
    gap = abs(_word_x_center(candidate) - x)
    width = max(1.0, float((candidate.get("bbox") or [0, 0, 0, 0])[2]) - float((candidate.get("bbox") or [0, 0, 0, 0])[0]))
    if gap <= max(28.0, width * 0.70):
        return candidate
    return None


def _grouped_header_word_fragments_should_merge(parent: str, child: str) -> bool:
    parent_text = _cell_text(parent)
    child_text = _cell_text(child)
    if not parent_text or not child_text:
        return False
    if parent_text.count("(") > parent_text.count(")") or parent_text.count("（") > parent_text.count("）"):
        return True
    if parent_text.endswith(("/", "或")):
        return True
    if child_text in {"周期", "量%"} and re.search(r"(采样时间|给药剂)$", parent_text):
        return True
    return False


def _grouped_word_line_is_table_note(text: str) -> bool:
    cleaned = _cell_text(text)
    return bool(re.match(r"^(?:附加信息|备注|注释|注)[:：]", cleaned))


def _project_grouped_word_line_to_leaf_columns(
    line: list[dict[str, Any]],
    centers: list[float],
) -> list[str | None]:
    cells: list[list[dict[str, Any]]] = [[] for _ in centers]
    if not centers:
        return []
    boundaries = [
        (centers[index] + centers[index + 1]) / 2.0
        for index in range(len(centers) - 1)
    ]
    for word in sorted(line, key=_word_x_center):
        x = _word_x_center(word)
        col = 0
        while col < len(boundaries) and x > boundaries[col]:
            col += 1
        if 0 <= col < len(cells):
            cells[col].append(word)
    return [
        _cell_text(" ".join(str(word.get("text") or "") for word in sorted(col_words, key=_word_x_center))) or None
        for col_words in cells
    ]


def _grouped_word_projected_row_is_viable(row: list[str | None]) -> bool:
    texts = [_cell_text(cell) for cell in row]
    if not any(texts):
        return False
    if _grouped_word_line_is_table_note(" ".join(texts)):
        return False
    value_count = sum(
        1
        for text in texts[2:]
        if text and (_looks_numeric(text) or _looks_like_compact_table_value(text) or re.fullmatch(r"n\.?d\.?|微量|nd", text, re.IGNORECASE))
    )
    label_count = sum(1 for text in texts[:2] if text and not _looks_numeric(text))
    return value_count >= 2 or (label_count >= 1 and value_count >= 1)


def _grouped_word_header_column_groups(
    top_line: list[dict[str, Any]],
    leaf_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    leaf_texts = [_cell_text(spec.get("text")) for spec in leaf_specs]
    leaf_centers = [float(spec.get("x", 0.0) or 0.0) for spec in leaf_specs]
    if len(leaf_centers) < 2:
        return []
    median_gap = sorted(
        leaf_centers[index + 1] - leaf_centers[index]
        for index in range(len(leaf_centers) - 1)
    )[max(0, (len(leaf_centers) - 1) // 2)]
    for parent in sorted(top_line, key=_word_x_center):
        parent_text = _cell_text(parent.get("text"))
        if not parent_text:
            continue
        parent_norm = _normalize_table_header_match_text(parent_text)
        if any(
            _grouped_header_word_fragments_should_merge(parent_text, leaf)
            or (parent_norm and parent_norm in _normalize_table_header_match_text(leaf))
            for leaf in leaf_texts
        ):
            continue
        parent_x = _word_x_center(parent)
        child_indices = [
            index
            for index, center in enumerate(leaf_centers)
            if abs(center - parent_x) <= max(42.0, median_gap * 1.15)
        ]
        if len(child_indices) < 2:
            continue
        children = [leaf_texts[index] for index in child_indices if leaf_texts[index]]
        if len(children) < 2:
            continue
        groups.append(
            {
                "row": 0,
                "start_col": min(child_indices),
                "end_col": max(child_indices),
                "text": parent_text,
                "child_headers": children,
                "source": "grouped_multilevel_word_header_colspan_projection",
            }
        )
    return _merge_same_span_grouped_word_parent_headers(_dedupe_span_groups(groups))


def _merge_same_span_grouped_word_parent_headers(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not groups:
        return []
    merged: dict[tuple[int, int, int], dict[str, Any]] = {}
    for group in groups:
        key = (
            int(group.get("row", 0) or 0),
            int(group.get("start_col", 0) or 0),
            int(group.get("end_col", 0) or 0),
        )
        if key not in merged:
            merged[key] = dict(group)
            continue
        existing = merged[key]
        existing["text"] = _join_schema_header_parts([_cell_text(existing.get("text")), _cell_text(group.get("text"))])
        children = list(existing.get("child_headers", []) or [])
        for child in group.get("child_headers", []) or []:
            if child not in children:
                children.append(child)
        existing["child_headers"] = children
    return list(merged.values())


def _grouped_word_leading_stub_row_groups(rows: list[list[str | None]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    active_text = ""
    active_start: int | None = None
    for row_index, row in enumerate(rows[1:], start=1):
        text = _cell_text(row[0] if row else None)
        if text:
            if active_text and active_start is not None and row_index - active_start > 1:
                groups.append(
                    {
                        "row": active_start,
                        "start_row": active_start,
                        "col": 0,
                        "rowspan": row_index - active_start,
                        "text": active_text,
                        "source": "leading_stub_word_rowspan_projection",
                    }
                )
            active_text = text
            active_start = row_index
    if active_text and active_start is not None and len(rows) - active_start > 1:
        groups.append(
            {
                "row": active_start,
                "start_row": active_start,
                "col": 0,
                "rowspan": len(rows) - active_start,
                "text": active_text,
                "source": "leading_stub_word_rowspan_projection",
            }
        )
    return groups


def _packed_header_suffix_child_count_from_body(
    body_rows: list[list[Any]],
    col_idx: int,
    col_count: int,
) -> int:
    candidates: list[int] = []
    for probe_col in (col_idx, col_idx + 1):
        if probe_col >= col_count:
            continue
        counts = [
            len(_split_numeric_value_atoms(_cell_text(row[probe_col])))
            for row in body_rows[:8]
            if probe_col < len(row) and _cell_text(row[probe_col])
        ]
        counts = [count for count in counts if count >= 2]
        if counts:
            candidates.append(max(set(counts), key=counts.count))
    return max(candidates, default=0)


def _grouped_orphan_leaf_can_extend_previous_parent(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned or _looks_numeric(cleaned):
        return False
    if re.fullmatch(r"[A-Za-z]{1,4}\d?", cleaned):
        return True
    if re.fullmatch(r"[A-Za-z]\d{1,3}", cleaned):
        return True
    return False


def _single_bottom_token_looks_like_child_header(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned or _looks_numeric(cleaned):
        return False
    if len(cleaned.split()) > 1:
        return False
    if re.fullmatch(r"[A-Za-z]{1,4}\d?", cleaned):
        return True
    compact = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff%]+", "", cleaned)
    return 1 <= len(compact) <= 12


def _select_grouped_multilevel_leaf_header(parts: list[str]) -> str:
    cleaned = [_cell_text(part) for part in parts if _cell_text(part)]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if all(_grouped_header_fragment_can_vertically_merge(part) for part in cleaned):
        return _join_schema_header_parts(cleaned)
    last = cleaned[-1]
    if _looks_like_compact_table_value(last) or _looks_numeric(last):
        return _join_schema_header_parts(cleaned)
    return last


def _grouped_header_fragment_can_vertically_merge(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_numeric(cleaned):
        return False
    if re.search(r"[。；;!?！？]\s*$", cleaned):
        return False
    if len(cleaned.split()) > 5:
        return False
    return True


def _repair_packed_grouped_multilevel_leaf_headers(
    header: list[str],
    body_rows: list[list[Any]],
) -> list[str]:
    result = list(header)
    for col_idx, text in enumerate(list(result)):
        cleaned = _cell_text(text)
        if not cleaned:
            continue
        tokens = cleaned.split()
        if len(tokens) < 2:
            continue
        previous = _cell_text(result[col_idx - 1]) if col_idx > 0 else ""
        if previous and _grouped_header_tail_completes_previous(previous, tokens[0]):
            result[col_idx - 1] = _join_schema_header_parts([previous, tokens[0]])
            result[col_idx] = " ".join(tokens[1:]).strip()
            continue
        if col_idx + 1 < len(result) and _cell_text(result[col_idx + 1]):
            continue
        if _grouped_header_contains_multiple_leafs(cleaned) and _grouped_body_column_has_packed_values(body_rows, col_idx):
            first, rest = tokens[0], " ".join(tokens[1:]).strip()
            result[col_idx] = first
            if col_idx + 1 < len(result) and not _cell_text(result[col_idx + 1]):
                result[col_idx + 1] = rest
    return result


def _grouped_header_tail_completes_previous(previous: str, tail: str) -> bool:
    prev = _cell_text(previous)
    next_part = _cell_text(tail)
    if not prev or not next_part:
        return False
    if len(next_part) > 12 or _looks_numeric(next_part):
        return False
    if re.search(r"[%）)]$", next_part):
        return True
    if re.search(r"(?:或|的|剂|给药|采样)$", prev):
        return True
    return False


def _grouped_header_contains_multiple_leafs(text: str) -> bool:
    tokens = _cell_text(text).split()
    if len(tokens) < 2:
        return False
    if any(_looks_numeric(token) for token in tokens):
        return False
    compact_tokens = [re.sub(r"[^A-Za-z0-9\u4e00-\u9fff%]+", "", token) for token in tokens]
    return sum(1 for token in compact_tokens if token) >= 2


def _grouped_body_column_has_packed_values(body_rows: list[list[Any]], col_idx: int) -> bool:
    packed = 0
    for row in body_rows[:8]:
        if col_idx >= len(row):
            continue
        text = _cell_text(row[col_idx])
        if len(_split_numeric_value_atoms(text)) >= 2:
            packed += 1
    return packed >= 1


def _expand_leading_stub_header_if_packed(
    header: list[str],
    body_rows: list[list[Any]],
) -> list[str]:
    if not header:
        return header
    first = _cell_text(header[0])
    if not first or len(first.split()) < 2:
        return header
    if not _leading_body_stub_looks_packed(body_rows):
        return header
    parts = first.split()
    return [parts[0], " ".join(parts[1:]).strip(), *header[1:]]


def _leading_body_stub_looks_packed(body_rows: list[list[Any]]) -> bool:
    packed = 0
    for row in body_rows[:8]:
        first = _cell_text(row[0] if row else None)
        if len(first.split()) >= 2 and not _row_text_looks_like_prose_sentence(first):
            packed += 1
    return packed >= 2


def _compact_empty_grouped_header_columns(
    header: list[str],
    body_rows: list[list[Any]],
) -> list[str]:
    keep: list[int] = []
    for col_idx, text in enumerate(header):
        if _cell_text(text):
            keep.append(col_idx)
            continue
        body_values = [
            _cell_text(row[col_idx])
            for row in body_rows[:8]
            if col_idx < len(row) and _cell_text(row[col_idx])
        ]
        if body_values:
            keep.append(col_idx)
    return [header[idx] for idx in keep]


def _project_grouped_multilevel_header_column_groups(
    rows: list[list[Any]],
    header_depth: int,
    logical_header: list[str],
) -> list[dict[str, Any]]:
    if header_depth < 2 or len(logical_header) < 4:
        return []
    top_texts = [_cell_text(cell) for cell in rows[0]]
    groups: list[dict[str, Any]] = []
    for col_idx, text in enumerate(top_texts):
        if not text or col_idx >= len(logical_header):
            continue
        child_start = _grouped_header_child_start_col(col_idx, logical_header)
        child_end = _grouped_header_child_end_col(col_idx, logical_header)
        if child_end <= child_start:
            continue
        children = [_cell_text(logical_header[idx]) for idx in range(child_start, child_end + 1) if _cell_text(logical_header[idx])]
        if len(children) < 2:
            continue
        parent_norm = _normalize_table_header_match_text(text)
        child_norms = {_normalize_table_header_match_text(child) for child in children}
        if parent_norm in child_norms:
            continue
        groups.append(
            {
                "row": 0,
                "start_col": child_start,
                "end_col": child_end,
                "text": text,
                "child_headers": children,
                "source": "grouped_multilevel_borderless_colspan_projection",
            }
        )
    return _dedupe_span_groups(groups)


def _grouped_header_child_start_col(parent_col: int, logical_header: list[str]) -> int:
    if _cell_text(logical_header[parent_col]):
        return parent_col
    for col_idx in range(parent_col + 1, len(logical_header)):
        if _cell_text(logical_header[col_idx]):
            return col_idx
    return parent_col


def _grouped_header_child_end_col(parent_col: int, logical_header: list[str]) -> int:
    start = _grouped_header_child_start_col(parent_col, logical_header)
    end = start
    for col_idx in range(start + 1, len(logical_header)):
        text = _cell_text(logical_header[col_idx])
        if not text:
            continue
        if _grouped_header_leaf_starts_new_parent(text):
            break
        end = col_idx
    return end


def _grouped_header_leaf_starts_new_parent(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if len(cleaned.split()) > 1 and not re.fullmatch(r"[A-Za-z0-9_.%-]+", cleaned):
        return True
    compact = re.sub(r"[^A-Za-z\u4e00-\u9fff]+", "", cleaned)
    return len(compact) > 8 and not re.fullmatch(r"[A-Za-z]{1,4}\d?", cleaned)


def _project_grouped_multilevel_header_row_groups(
    rows: list[list[Any]],
    header_depth: int,
    logical_header: list[str],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for col_idx, text in enumerate(logical_header):
        if not _cell_text(text):
            continue
        parts = [
            _cell_text(rows[row_idx][col_idx])
            for row_idx in range(min(header_depth, len(rows)))
            if col_idx < len(rows[row_idx]) and _cell_text(rows[row_idx][col_idx])
        ]
        if len(parts) >= 2 and _join_schema_header_parts(parts) == text:
            groups.append(
                {
                    "col": col_idx,
                    "start_row": 0,
                    "rowspan": len(parts),
                    "text": text,
                    "source": "grouped_multilevel_borderless_row_header_projection",
                }
            )
    return groups


def _project_leading_stub_rowspan_groups_from_body(
    rows: list[list[Any]],
    header_depth: int,
    col_count: int,
) -> list[dict[str, Any]]:
    body_rows = rows[header_depth:]
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for offset, row in enumerate(body_rows, start=header_depth):
        first = _cell_text(row[0] if row else None)
        first_label, residual = _split_leading_group_label_from_cell(first, row)
        if first_label:
            if current and int(current.get("rowspan", 1) or 1) > 1:
                groups.append(current)
            current = {
                "row": offset,
                "start_row": offset,
                "col": 0,
                "rowspan": 1,
                "text": first_label,
                "source": "leading_stub_rowspan_projection",
                "residual_first_cell": residual,
            }
            continue
        if current and _row_continues_leading_stub_group(row, col_count):
            current["rowspan"] = int(current.get("rowspan", 1) or 1) + 1
            continue
        if current and int(current.get("rowspan", 1) or 1) > 1:
            groups.append(current)
        current = None
    if current and int(current.get("rowspan", 1) or 1) > 1:
        groups.append(current)
    return groups


def _split_leading_group_label_from_cell(text: str, row: list[Any]) -> tuple[str, str]:
    cleaned = _cell_text(text)
    if not cleaned:
        return "", ""
    tokens = cleaned.split()
    if len(tokens) < 2:
        return "", ""
    first, rest = tokens[0], " ".join(tokens[1:]).strip()
    if not rest:
        return "", ""
    if _looks_numeric(first) or _row_text_looks_like_prose_sentence(cleaned):
        return "", ""
    right_values = [_cell_text(cell) for cell in row[1:] if _cell_text(cell)]
    value_like = sum(1 for value in right_values if _looks_numeric(value) or _looks_like_compact_table_value(value))
    if value_like < 2:
        return "", ""
    if len(first) > 12 or re.search(r"[。；;!?！？]", first):
        return "", ""
    return first, rest


def _row_continues_leading_stub_group(row: list[Any], col_count: int) -> bool:
    first = _cell_text(row[0] if row else None)
    if not first:
        return False
    if len(first.split()) >= 2:
        return False
    if _looks_numeric(first) or _row_text_looks_like_prose_sentence(first):
        return False
    right_values = [_cell_text(cell) for cell in row[1:col_count] if _cell_text(cell)]
    if not right_values:
        return False
    value_like = sum(1 for value in right_values if _looks_numeric(value) or _looks_like_compact_table_value(value))
    return value_like >= 1


def _grouped_multilevel_projection_has_body_support(body_rows: list[list[Any]], logical_col_count: int) -> bool:
    supported_rows = 0
    for row in body_rows[:10]:
        values = [_cell_text(cell) for cell in row[:logical_col_count] if _cell_text(cell)]
        if len(values) < 3:
            continue
        value_like = sum(1 for value in values[1:] if _looks_numeric(value) or _looks_like_compact_table_value(value))
        if value_like >= 2:
            supported_rows += 1
    return supported_rows >= 3


def _grouped_header_cell_looks_packed(text: str) -> bool:
    cleaned = _cell_text(text)
    return bool(cleaned and len(cleaned.split()) >= 3 and not _row_text_looks_like_prose_sentence(cleaned))


def _dedupe_span_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for group in groups:
        key = (
            int(group.get("start_col", 0) or 0),
            int(group.get("end_col", 0) or 0),
            _normalize_table_header_match_text(str(group.get("text") or "")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(group)
    return result


def _project_missing_leading_stub_header_cell(
    table: dict[str, Any],
    grid: list[list[Any]],
    family: str,
) -> tuple[list[list[str | None]] | None, list[str] | None, dict[str, Any] | None]:
    """Restore a blank row-stub header cell when extraction shifted labels left."""
    if family not in {"comparison_matrix", "rowspan_grouped_table", "projected_stub_matrix", "spreadsheet_matrix"}:
        return None, None, None
    col_count = _col_count(table, grid)
    if col_count < 3 or len(grid) < 3:
        return None, None, None
    rows = _rectangular_grid(_clone_grid(grid), col_count)
    if _should_defer_missing_stub_projection_to_sparse_numeric_external_header(table, rows, col_count):
        return None, None, None
    first_row = rows[0]
    if not _cell_text(first_row[0]):
        return None, None, None
    if _cell_text(first_row[1]):
        return None, None, None
    if sum(1 for col_idx in range(1, col_count) if _cell_text(first_row[col_idx])) < 1:
        return None, None, None

    body_probe = rows[1 : min(len(rows), 6)]
    if len(body_probe) < 2:
        return None, None, None
    spreadsheet_profile = family == "spreadsheet_matrix"
    stub_label_rows = 0
    value_rows = 0
    for offset, row in enumerate(body_probe, start=1):
        first_text = _cell_text(row[0])
        if first_text and (
            not _looks_numeric(first_text)
            or (spreadsheet_profile and _looks_like_spreadsheet_row_number(first_text, expected=offset))
        ):
            stub_label_rows += 1
        if spreadsheet_profile and offset == 1:
            value_like = sum(1 for cell in row[1:col_count] if _cell_text(cell) and not _looks_numeric(_cell_text(cell)))
        else:
            value_like = sum(
                1
                for cell in row[1:col_count]
                if _cell_text(cell) and (_looks_numeric(_cell_text(cell)) or _looks_like_compact_table_value(_cell_text(cell)))
            )
        if value_like >= 1:
            value_rows += 1
    if stub_label_rows < 2 or value_rows < 2:
        return None, None, None

    header_values = [_cell_text(cell) for cell in first_row if _cell_text(cell)]
    if len(header_values) > col_count - 1:
        return None, None, None
    shifted_first_row: list[str | None] = [None, *header_values]
    while len(shifted_first_row) < col_count:
        shifted_first_row.append(None)
    projected_rows: list[list[str | None]] = [shifted_first_row]
    projected_rows.extend([_cell_text(cell) or None for cell in row[:col_count]] for row in rows[1:])
    header_override = [cell or "" for cell in shifted_first_row]
    return projected_rows, header_override, {
        "source": "missing_leading_stub_header_projection",
        "reason": "first_header_label_shifted_left_over_blank_row_stub",
        "column_count": col_count,
        "stub_label_row_count": stub_label_rows,
        "value_row_count": value_rows,
    }


def _should_defer_missing_stub_projection_to_sparse_numeric_external_header(
    table: dict[str, Any],
    rows: list[list[Any]],
    col_count: int,
) -> bool:
    external_header = _sparse_numeric_external_header_candidates(table)
    if not external_header:
        return False
    if not _header_metadata_looks_like_data_row(_header_texts(table)):
        return False
    data_rows: list[list[Any]] = []
    for row in rows:
        if _row_looks_like_sparse_numeric_data_row(row, col_count):
            data_rows.append(row)
            continue
        if data_rows:
            break
    if len(data_rows) < 3:
        return False
    return bool(
        _sparse_numeric_external_header_column_groups(
            data_rows,
            col_count,
            logical_col_count=len(external_header),
        )
    )


def _sparse_header_band_depth(rows: list[list[Any]], col_count: int) -> int:
    max_probe = min(4, len(rows) - 1)
    depth = 0
    for row_idx in range(max_probe):
        row = rows[row_idx][:col_count]
        if not any(_cell_text(cell) for cell in row):
            break
        if row_idx > 0 and _row_has_body_value_signature(row, col_count):
            break
        depth += 1
    if depth < 2:
        return 0
    if depth >= len(rows):
        return 0
    if not _row_has_body_value_signature(rows[depth][:col_count], col_count):
        return 0
    sparse_header_rows = 0
    for row in rows[:depth]:
        filled = sum(1 for cell in row[:col_count] if _cell_text(cell))
        if filled < col_count:
            sparse_header_rows += 1
    return depth if sparse_header_rows else 0


def _row_has_body_value_signature(row: list[Any], col_count: int) -> bool:
    texts = [_cell_text(cell) for cell in row[:col_count]]
    filled = [text for text in texts if text]
    if len(filled) < max(2, col_count // 3):
        return False
    value_like = 0
    for text in filled[1:] if len(filled) > 1 else filled:
        if (
            _looks_numeric(text)
            or _looks_like_compact_table_value(text)
            or re.fullmatch(r"(?:[ox×✗✓✔]|yes|no|y|n|--|-)", text, re.IGNORECASE)
        ):
            value_like += 1
    return value_like >= max(2, min(4, col_count // 2))


def _sparse_header_data_start_col(rows: list[list[Any]], header_depth: int, col_count: int) -> int:
    first_col_header_texts = [_cell_text(rows[row_idx][0]) for row_idx in range(header_depth)]
    has_sparse_first_col_header = any(first_col_header_texts) and not all(first_col_header_texts)
    body_probe = rows[header_depth : min(len(rows), header_depth + 3)]
    first_col_body_labels = [
        _cell_text(row[0])
        for row in body_probe
        if _cell_text(row[0]) and not _looks_numeric(_cell_text(row[0]))
    ]
    non_stub_header_text = any(
        _cell_text(rows[row_idx][col_idx])
        for row_idx in range(header_depth)
        for col_idx in range(1, col_count)
    )
    if has_sparse_first_col_header and first_col_body_labels and non_stub_header_text:
        return 1
    return 0


def _build_logical_cells(
    table: dict[str, Any],
    grid: list[list[Any]],
    *,
    include_row_groups: bool = True,
    extra_row_groups: Any = None,
    extra_header_column_groups: Any = None,
    extra_header_row_groups: Any = None,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            text = _cell_text(cell)
            if not text:
                continue
            cells.append(
                {
                    "row": row_idx,
                    "col": col_idx,
                    "rowspan": 1,
                    "colspan": 1,
                    "text": text,
                    "source": "semantic_grid",
                }
            )
    row_group_sources: list[Any] = []
    if include_row_groups:
        row_group_sources.extend(table.get("row_groups", []) or [])
    if isinstance(extra_row_groups, list):
        row_group_sources.extend(extra_row_groups)
    if row_group_sources:
        for group in row_group_sources:
            if not isinstance(group, dict):
                continue
            try:
                col = int(group.get("col", 0) or 0)
                start_row = int(group.get("row", group.get("start_data_row", group.get("start_row", 0))) or 0)
                rowspan = int(group.get("rowspan", 1) or 1)
            except Exception:
                continue
            text = _cell_text(group.get("text"))
            if not text or rowspan <= 1:
                continue
            cells.append(
                {
                    "row": start_row,
                    "col": col,
                    "rowspan": rowspan,
                    "colspan": 1,
                    "text": text,
                    "source": str(group.get("source") or "row_group"),
                }
            )
    header_column_group_sources: list[Any] = []
    header_column_group_sources.extend(table.get("header_column_groups", []) or [])
    if isinstance(extra_header_column_groups, list):
        header_column_group_sources.extend(extra_header_column_groups)
    for group in header_column_group_sources:
        if not isinstance(group, dict):
            continue
        try:
            start_col = int(group.get("start_col", group.get("col", 0)) or 0)
            end_col = int(group.get("end_col", start_col) or start_col)
        except Exception:
            continue
        text = _cell_text(group.get("text"))
        if text and end_col >= start_col:
            cells.append(
                {
                    "row": int(group.get("row", 0) or 0),
                    "col": start_col,
                    "rowspan": 1,
                    "colspan": end_col - start_col + 1,
                    "text": text,
                    "source": str(group.get("source") or "header_column_group"),
                }
            )
    header_row_group_sources: list[Any] = []
    header_row_group_sources.extend(table.get("header_row_groups", []) or [])
    if isinstance(extra_header_row_groups, list):
        header_row_group_sources.extend(extra_header_row_groups)
    for group in header_row_group_sources:
        if not isinstance(group, dict):
            continue
        try:
            col = int(group.get("col", 0) or 0)
            start_row = int(group.get("start_row", 0) or 0)
            rowspan = int(group.get("rowspan", 1) or 1)
        except Exception:
            continue
        text = _cell_text(group.get("text"))
        if text and rowspan > 1:
            cells.append(
                {
                    "row": start_row,
                    "col": col,
                    "rowspan": rowspan,
                    "colspan": 1,
                    "text": text,
                    "source": str(group.get("source") or "header_row_group"),
                }
            )
    return _dedupe_logical_cells(cells)


def _logical_cells_with_two_column_spanning_header(
    cells: list[dict[str, Any]],
    grid: list[list[Any]],
) -> list[dict[str, Any]]:
    if not grid:
        return cells
    header_text = _cell_text(grid[0][0] if grid[0] else None)
    if not header_text:
        return cells
    filtered = [
        cell
        for cell in cells
        if not (
            int(cell.get("row", -1) or -1) == 0
            and int(cell.get("col", -1) or -1) in {0, 1}
            and str(cell.get("source") or "") == "semantic_grid"
        )
    ]
    filtered.append(
        {
            "row": 0,
            "col": 0,
            "rowspan": 1,
            "colspan": 2,
            "text": header_text,
            "source": "two_column_spanning_header_projection",
        }
    )
    return _dedupe_logical_cells(filtered)


def _dedupe_semantic_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        key = (
            group.get("row"),
            group.get("col"),
            group.get("start_col"),
            group.get("end_col"),
            group.get("start_row"),
            group.get("rowspan"),
            group.get("colspan"),
            group.get("text"),
            group.get("source"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(group)
    return result


def _build_projected_stub_semantic_grid(
    table: dict[str, Any],
    grid: list[list[Any]],
) -> tuple[list[list[str | None]], dict[str, Any]] | tuple[None, None]:
    headers = _header_texts(table) or [_cell_text(cell) for cell in grid[0]]
    body_rows = _body_rows(table, grid)
    anchors = _projected_stub_label_anchors(table, grid)
    if not headers or not body_rows or not anchors:
        return None, None

    normalized_body = [[_cell_text(cell) or None for cell in row[: len(headers)]] for row in body_rows]
    for row in normalized_body:
        while len(row) < len(headers):
            row.append(None)

    anchor_specs = _projected_stub_group_boundaries(anchors, len(normalized_body))
    if not anchor_specs:
        return None, None

    semantic_grid: list[list[str | None]] = [[None, *headers]]
    groups: list[dict[str, Any]] = []
    for anchor_index, start, end in anchor_specs:
        anchor = anchors[anchor_index]
        label = str(anchor["label"])
        source_rows = list(range(start, end))
        if not source_rows:
            continue
        merged_row: list[str | None] = [label]
        for col_idx in range(len(headers)):
            values = [
                normalized_body[row_idx][col_idx]
                for row_idx in source_rows
                if row_idx < len(normalized_body)
            ]
            values = _projected_stub_remove_anchor_label(values, label, col_idx == int(anchor.get("col", 0)))
            merged_row.append(_merge_projected_stub_cell_text(values))
        if not any(_cell_text(cell) for cell in merged_row[1:]):
            continue
        semantic_grid.append(merged_row)
        groups.append(
            {
                "label": label,
                "source_data_row_start": start + 1,
                "source_data_row_end": end,
                "anchor_source": str(anchor.get("source") or ""),
            }
        )

    if len(semantic_grid) <= 1 or len(semantic_grid) >= len(grid):
        return None, None
    return semantic_grid, {
        "strategy": "projected_stub_matrix_visual_wrap_compaction",
        "source_data_row_count": len(normalized_body),
        "compacted_data_row_count": len(semantic_grid) - 1,
        "groups": groups,
    }


def _projected_stub_group_boundaries(
    anchors: list[dict[str, Any]],
    row_count: int,
) -> list[tuple[int, int, int]]:
    ordered = sorted(
        [
            (idx, int(anchor.get("row", 0) or 0))
            for idx, anchor in enumerate(anchors)
            if 0 <= int(anchor.get("row", 0) or 0) < row_count
        ],
        key=lambda item: item[1],
    )
    if not ordered:
        return []
    starts: list[int] = []
    for pos, (_, row_idx) in enumerate(ordered):
        if pos == 0:
            starts.append(0)
            continue
        previous_row = ordered[pos - 1][1]
        anchor = anchors[ordered[pos][0]]
        if str(anchor.get("source") or "") == "cell_tail_label" and row_idx - previous_row <= 2:
            starts.append(row_idx)
        else:
            starts.append((previous_row + row_idx + 1) // 2)
    boundaries: list[tuple[int, int, int]] = []
    for pos, (anchor_index, _) in enumerate(ordered):
        start = starts[pos]
        end = starts[pos + 1] if pos + 1 < len(starts) else row_count
        if end > start:
            boundaries.append((anchor_index, start, end))
    return boundaries


def _projected_stub_label_anchors(table: dict[str, Any], grid: list[list[Any]]) -> list[dict[str, Any]]:
    rows = _body_rows(table, grid)
    anchors: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    anchored_rows: set[int] = set()
    for group in table.get("row_groups", []) or []:
        if not isinstance(group, dict):
            continue
        text = _cell_text(group.get("text"))
        if not _looks_like_projected_stub_label(text):
            continue
        try:
            row_idx = int(group.get("start_data_row", group.get("start_row", 1)) or 1) - 1
            col_idx = int(group.get("col", 0) or 0)
        except Exception:
            continue
        if row_idx < 0 or row_idx >= len(rows):
            continue
        key = (row_idx, _normalize_text(text))
        if key in seen:
            continue
        seen.add(key)
        anchored_rows.add(row_idx)
        anchors.append({"row": row_idx, "col": col_idx, "label": text, "source": str(group.get("source") or "row_group")})

    for row_idx, row in enumerate(rows):
        if row_idx in anchored_rows:
            continue
        for col_idx, value in enumerate(row):
            text = _cell_text(value)
            label, prefix = _split_projected_stub_tail_label(text)
            if not label:
                continue
            key = (row_idx, _normalize_text(label))
            if key in seen:
                continue
            seen.add(key)
            anchored_rows.add(row_idx)
            anchors.append({"row": row_idx, "col": col_idx, "label": label, "prefix": prefix, "source": "cell_tail_label"})
            break
    anchors.sort(key=lambda item: (int(item.get("row", 0) or 0), int(item.get("col", 0) or 0)))
    return anchors


def _projected_stub_remove_anchor_label(values: list[str | None], label: str, anchor_column: bool) -> list[str | None]:
    if not anchor_column or not label:
        return values
    result: list[str | None] = []
    label_norm = _normalize_text(label)
    for value in values:
        text = _cell_text(value)
        if not text:
            result.append(None)
            continue
        if _normalize_text(text) == label_norm:
            result.append(None)
            continue
        tail_label, prefix = _split_projected_stub_tail_label(text)
        if tail_label and _normalize_text(tail_label) == label_norm:
            result.append(prefix or None)
            continue
        result.append(text)
    return result


def _split_projected_stub_tail_label(text: str) -> tuple[str, str]:
    cleaned = _cell_text(text)
    if not cleaned:
        return "", ""
    tokens = cleaned.split()
    candidates: list[tuple[float, int, str, str]] = []
    for size in range(1, min(4, len(tokens) - 1) + 1):
        candidate = " ".join(tokens[-size:])
        prefix = " ".join(tokens[:-size]).strip()
        if prefix and _looks_like_projected_stub_label(candidate):
            candidates.append((_projected_stub_label_title_quality(candidate), size, candidate, prefix))
    if candidates:
        _, _, candidate, prefix = max(candidates, key=lambda item: (item[0], item[1]))
        return candidate, prefix
    return "", ""


def _projected_stub_label_title_quality(text: str) -> float:
    words = [
        word
        for word in _cell_text(text).split()
        if word.lower() not in {"and", "or", "of", "the", "to", "for", "in", "on", "by", "with"}
    ]
    if not words:
        return 0.0
    title_like = 0
    for word in words:
        stripped = re.sub(r"^[^A-Za-z\u4e00-\u9fff]+|[^A-Za-z\u4e00-\u9fff]+$", "", word)
        if not stripped:
            continue
        if re.search(r"[\u4e00-\u9fff]", stripped):
            title_like += 1
        elif stripped.isupper() and len(stripped) <= 8:
            title_like += 1
        elif stripped[:1].isupper():
            title_like += 1
    return title_like / max(1, len(words))


def _looks_like_projected_stub_label(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if len(cleaned) > 36:
        return False
    if _looks_numeric(cleaned):
        return False
    if any(mark in cleaned for mark in ".;!?。；！？"):
        return False
    words = cleaned.split()
    if len(words) > 4:
        return False
    compact = re.sub(r"[^A-Za-z\u4e00-\u9fff]+", "", cleaned)
    if len(compact) < 3:
        return False
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return True
    first_word = words[0] if words else ""
    if not first_word[:1].isupper():
        return False
    lowercase_function_words = {"and", "or", "of", "the", "to", "for", "in", "on", "by", "with"}
    content_words = [word for word in words if word.lower() not in lowercase_function_words]
    if len(content_words) > 1:
        uppercase_starts = sum(1 for word in content_words if word[:1].isupper())
        return uppercase_starts >= 1
    return True


def _merge_projected_stub_cell_text(values: list[str | None]) -> str | None:
    merged = ""
    seen: set[str] = set()
    for value in values:
        text = _cell_text(value)
        if not text:
            continue
        signature = _normalize_text(text)
        if signature in seen:
            continue
        seen.add(signature)
        merged = text if not merged else _join_projected_stub_text(merged, text)
    return merged or None


def _join_projected_stub_text(left: str, right: str) -> str:
    prefix = left.rstrip()
    suffix = right.lstrip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    if prefix.endswith("-") and suffix[:1].islower():
        return prefix[:-1] + suffix
    if prefix[-1].isspace() or suffix[0].isspace():
        return prefix + suffix
    if prefix[-1] in {",", ";", ":"}:
        return f"{prefix} {suffix}"
    if re.match(r"[A-Za-z0-9)]", prefix[-1]) and re.match(r"[A-Za-z0-9(]", suffix[0]):
        return f"{prefix} {suffix}"
    return prefix + suffix


def _dedupe_logical_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cell in cells:
        key = (
            cell.get("row"),
            cell.get("col"),
            cell.get("rowspan"),
            cell.get("colspan"),
            cell.get("text"),
            cell.get("source"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(cell)
    return result


def _header_texts(table: dict[str, Any]) -> list[str]:
    header = table.get("header")
    if isinstance(header, list) and header:
        values: list[tuple[int, str]] = []
        for index, item in enumerate(header):
            if isinstance(item, dict):
                col_value = item.get("col", index + 1)
                try:
                    col = int(col_value or index + 1)
                except Exception:
                    col = index + 1
                values.append((col, _cell_text(item.get("text"))))
            else:
                values.append((index + 1, _cell_text(item)))
        values.sort(key=lambda item: item[0])
        return [text for _, text in values]
    return []


def _body_rows(table: dict[str, Any], grid: list[list[Any]]) -> list[list[Any]]:
    data_grid = _clone_grid(table.get("data_grid"))
    if data_grid:
        return data_grid
    if len(grid) <= 1:
        return []
    header = _header_texts(table)
    if header and [_cell_text(cell) for cell in grid[0]][: len(header)] == header[: len(grid[0])]:
        return grid[1:]
    return grid[1:]


def _remove_adjacent_duplicate_rows(grid: list[list[Any]]) -> tuple[list[list[Any]], int]:
    result: list[list[Any]] = []
    removed = 0
    previous_signature: tuple[str, ...] | None = None
    for row in grid:
        signature = tuple(_cell_text(cell).lower() for cell in row)
        if result and signature == previous_signature:
            removed += 1
            continue
        if result and _is_single_cell_repeat(result[-1], row):
            removed += 1
            continue
        result.append(list(row))
        previous_signature = signature
    return result, removed


def _project_two_column_label_after_value_pairs(
    table: dict[str, Any],
    grid: list[list[Any]],
) -> tuple[list[list[str | None]] | None, dict[str, Any]]:
    if _col_count(table, grid) != 2 or len(grid) < 5:
        return None, {}
    header = _semantic_two_column_inventory_header(table, grid)
    if len(header) < 2:
        return None, {}
    body = _body_rows(table, grid)
    specs = _two_column_label_after_value_pair_specs(body)
    if len(specs) < 2:
        return None, {}

    spec_by_label_row = {label_row_idx: (value_row_idx, label, value) for value_row_idx, label_row_idx, label, value in specs}
    consumed_value_rows = {value_row_idx for value_row_idx, _, _, _ in specs}
    consumed_label_rows: set[int] = set()
    semantic_grid: list[list[str | None]] = [[header[0] or None, header[1] or None]]
    pending_value: str | None = None
    for row_idx, row in enumerate(body):
        left = _cell_text(row[0] if len(row) > 0 else None)
        right = _cell_text(row[1] if len(row) > 1 else None)
        if [left, right] == header[:2]:
            pending_value = None
            continue
        if row_idx in consumed_value_rows:
            pending_value = right
            continue
        pair = spec_by_label_row.get(row_idx)
        if pair:
            _value_row_idx, label, value = pair
            semantic_grid.append([label, value])
            consumed_label_rows.add(row_idx)
            pending_value = None
            continue
        if left and not right:
            if _looks_like_two_column_inventory_label(left):
                semantic_grid.append([left, None])
            pending_value = None
            continue
        if left and right:
            semantic_grid.append([left, right])
            pending_value = None
            continue
        if right and not left:
            pending_value = right

    if pending_value and not consumed_label_rows:
        return None, {}
    if len(semantic_grid) >= len([row for row in grid if any(_cell_text(cell) for cell in row)]):
        return None, {}
    return semantic_grid, {
        "source": "two_column_label_after_value_pair_projection",
        "projected_pair_count": len(specs),
        "source_row_count": len(body),
    }


def _project_two_column_hierarchy_child_column_grid(
    table: dict[str, Any],
    grid: list[list[Any]],
) -> tuple[list[list[str | None]] | None, dict[str, Any]]:
    if _col_count(table, grid) != 2 or len(grid) < 4 or not _has_body_rowspan_groups(table):
        return None, {}
    header = _semantic_two_column_inventory_header(table, grid)
    if len(header) < 2:
        return None, {}
    source_rows = _body_rows(table, grid)
    if not source_rows:
        return None, {}

    changed = False
    projected: list[list[str | None]] = [[header[0] or None, header[1] or None]]
    current_parent_prefix = ""
    current_parent_text = ""
    moved_count = 0
    split_count = 0
    for row in source_rows:
        left = _cell_text(row[0] if len(row) > 0 else None)
        right = _cell_text(row[1] if len(row) > 1 else None)
        if [left, right] == header[:2]:
            changed = True
            continue
        if left and right:
            current_parent_prefix = _leading_outline_prefix(left)
            current_parent_text = left
            projected.append([left, right])
            continue
        if left and not right:
            parent_part, child_part = _split_parent_and_child_marker(left)
            if parent_part and child_part:
                current_parent_prefix = _leading_outline_prefix(parent_part)
                current_parent_text = parent_part
                projected.append([parent_part, child_part])
                changed = True
                split_count += 1
                continue
            if (
                current_parent_prefix
                and _leading_outline_prefix(left).startswith(current_parent_prefix + ".")
                and _looks_like_hierarchy_child_item(left, parent_prefix=current_parent_prefix)
            ):
                projected.append([None, left])
                changed = True
                moved_count += 1
                continue
            current_parent_prefix = _leading_outline_prefix(left) or current_parent_prefix
            current_parent_text = left or current_parent_text
            projected.append([left, None])
            continue
        projected.append([None, right or None])

    if not changed or len(projected) != len([row for row in grid if any(_cell_text(cell) for cell in row)]):
        return None, {}
    row_groups = _semantic_two_column_hierarchy_row_groups(projected)
    return projected, {
        "source": "two_column_hierarchy_child_column_projection",
        "moved_child_item_count": moved_count,
        "split_parent_child_item_count": split_count,
        "row_groups": row_groups,
    }


def _semantic_two_column_hierarchy_row_groups(grid: list[list[str | None]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    starts: list[tuple[int, str]] = []
    for row_idx, row in enumerate(grid[1:], start=1):
        left = _cell_text(row[0] if row else None)
        if left:
            starts.append((row_idx, left))
    for index, (start, text) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(grid)
        rowspan = end - start
        if rowspan <= 1:
            continue
        groups.append(
            {
                "row": start,
                "col": 0,
                "rowspan": rowspan,
                "text": text,
                "source": "semantic_hierarchy_row_group",
            }
        )
    return groups


def _leading_outline_prefix(text: str) -> str:
    cleaned = _cell_text(text)
    match = re.match(r"^\s*(\d+(?:\.\d+)+|\d+)[.)]?\s+\S+", cleaned)
    return match.group(1) if match else ""


def _split_parent_and_child_marker(text: str) -> tuple[str, str]:
    cleaned = _cell_text(text)
    match = re.match(
        r"^\s*(?P<parent>\d+\.\s+\S.+?)\s+(?P<child>\d+\.\d+(?:\.\d+)*\s+\S.+)$",
        cleaned,
    )
    if not match:
        return "", ""
    parent = _cell_text(match.group("parent"))
    child = _cell_text(match.group("child"))
    parent_prefix = _leading_outline_prefix(parent)
    child_prefix = _leading_outline_prefix(child)
    if not parent_prefix or not child_prefix.startswith(parent_prefix.rstrip(".") + "."):
        return "", ""
    return parent, child


def _looks_like_hierarchy_child_item(text: str, *, parent_prefix: str) -> bool:
    cleaned = _cell_text(text)
    prefix = _leading_outline_prefix(cleaned)
    if not prefix or not prefix.startswith(parent_prefix + "."):
        return False
    if _row_text_looks_like_prose_sentence(cleaned):
        return False
    return len(cleaned) <= 120


def _semantic_two_column_inventory_header(table: dict[str, Any], grid: list[list[Any]]) -> list[str]:
    explicit = _header_texts(table)
    if len(explicit) >= 2 and not any(_is_generic_header(text) for text in explicit[:2]):
        return explicit[:2]
    for row in grid:
        if len(row) < 2:
            continue
        left = _cell_text(row[0])
        right = _cell_text(row[1])
        if not left or not right:
            continue
        if _is_generic_header(left) or _is_generic_header(right):
            continue
        if _looks_like_two_column_inventory_label(left):
            return [left, right]
        break
    return []


def _two_column_label_after_value_pair_specs(rows: list[list[Any]]) -> list[tuple[int, int, str, str]]:
    specs: list[tuple[int, int, str, str]] = []
    for row_idx in range(len(rows) - 1):
        row = rows[row_idx]
        next_row = rows[row_idx + 1]
        left = _cell_text(row[0] if len(row) > 0 else None)
        right = _cell_text(row[1] if len(row) > 1 else None)
        next_left = _cell_text(next_row[0] if len(next_row) > 0 else None)
        next_right = _cell_text(next_row[1] if len(next_row) > 1 else None)
        if left or not right or not next_left or next_right:
            continue
        if not _looks_like_two_column_inventory_value(right):
            continue
        if not _looks_like_two_column_inventory_label(next_left):
            continue
        specs.append((row_idx, row_idx + 1, next_left, right))
    return specs


def _looks_like_two_column_inventory_label(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned or _looks_numeric(cleaned):
        return False
    if _looks_like_table_caption_like_text(cleaned) or _looks_like_table_note_or_marker_text(cleaned):
        return False
    if re.search(r"[.;!?銆傦紒锛燂紱]\s*$", cleaned):
        return False
    if len(cleaned) > 90 or len(cleaned.split()) > 8:
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned))


def _looks_like_two_column_inventory_value(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if _looks_like_table_caption_like_text(cleaned) or _looks_like_table_note_or_marker_text(cleaned):
        return False
    if _looks_numeric(cleaned):
        return False
    if re.search(r"[●•路♂]", cleaned):
        return True
    if _row_text_looks_like_prose_sentence(cleaned):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned)
    if len(words) >= 5 and re.search(r"\b(?:to|and|or|of|the|in|into|with|for|from)\b", cleaned, re.IGNORECASE):
        return True
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    return cjk_count >= 12 and bool(re.search(r"[，、；。]", cleaned))


def _compact_parallel_inventory_lists(
    table: dict[str, Any],
    grid: list[list[Any]],
) -> tuple[list[list[str | None]] | None, dict[str, Any]]:
    if _col_count(table, grid) != 2 or len(grid) < 5:
        return None, {}
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0][:2]]
    if len(header) < 2 or any(_is_generic_header(text) for text in header[:2]):
        return None, {}
    body = _body_rows(table, grid)
    if len(body) < 4:
        return None, {}
    meaningful = [row[:2] for row in body if _non_empty_count(row[:2]) >= 1]
    if len(meaningful) < 4:
        return None, {}
    if any(_looks_numeric(_cell_text(cell)) for row in meaningful for cell in row[:2] if _cell_text(cell)):
        return None, {}
    if _has_grouped_hierarchical_inventory_rows(table, meaningful):
        return None, {}
    if _looks_like_key_value_inventory_rows(meaningful):
        return None, {}
    if not _looks_like_parallel_inventory_columns(header[:2], meaningful):
        return None, {}

    left_items, right_items = _collect_parallel_inventory_column_items(meaningful)
    if len(left_items) < 3 or len(right_items) < 3:
        return None, {}
    compacted = [
        [header[0], header[1]],
        [_join_parallel_inventory_items(left_items), _join_parallel_inventory_items(right_items)],
    ]
    return compacted, {
        "source_row_count": len(meaningful),
        "compacted_row_count": 1,
        "source": "parallel_inventory_list_compaction",
    }


def _has_grouped_hierarchical_inventory_rows(table: dict[str, Any], rows: list[list[Any]]) -> bool:
    if _has_body_rowspan_groups(table):
        first_col_values = [_cell_text(row[0]) for row in rows if _cell_text(row[0])]
        numbered = sum(1 for text in first_col_values if re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S+", text))
        if numbered >= 2:
            return True
        if numbered >= 1 and len(first_col_values) <= max(3, len(rows) // 2):
            return True
    return False


def _collect_parallel_inventory_column_items(rows: list[list[Any]]) -> tuple[list[str], list[str]]:
    columns: list[list[str]] = [[], []]
    for row in rows:
        row_filled_count = _non_empty_count(row[:2])
        for col in range(2):
            text = _cell_text(row[col] if len(row) > col else None)
            if not text:
                continue
            if row_filled_count == 1 and _looks_like_inventory_continuation_fragment(
                text,
                columns[col][-1] if columns[col] else "",
            ):
                columns[col][-1] = _join_wrapped_description_text(columns[col][-1], text)
            else:
                columns[col].append(text)
    return columns[0], columns[1]


def _looks_like_inventory_continuation_fragment(text: str, previous: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned or not previous:
        return False
    if cleaned.endswith(":"):
        return False
    if re.search(r"[*#+$†‡]\s*$", previous):
        return False
    if re.search(r"[*#+$†‡]\s*$", cleaned):
        return False
    if len(cleaned) <= 28 and len(cleaned.split()) <= 3:
        return True
    return bool(re.match(r"^(?:and|or|of|in|with|for|enzyme|mixture)\b", cleaned, re.IGNORECASE))


def _looks_like_key_value_inventory_rows(rows: list[list[Any]]) -> bool:
    right_lengths = [len(_cell_text(row[1])) for row in rows if len(row) > 1 and _cell_text(row[1])]
    left_lengths = [len(_cell_text(row[0])) for row in rows if _cell_text(row[0])]
    if not right_lengths or not left_lengths:
        return False
    if max(right_lengths) >= max(40, max(left_lengths) * 2):
        return True
    right_sentence_like = sum(1 for row in rows if len(row) > 1 and _looks_like_long_body_fragment(_cell_text(row[1])))
    return right_sentence_like >= max(2, len(rows) // 2)


def _looks_like_parallel_inventory_columns(headers: list[str], rows: list[list[Any]]) -> bool:
    header_text = " ".join(headers).lower()
    inventory_header_signal = any(
        token in header_text
        for token in (
            "reagent",
            "supply",
            "equipment",
            "material",
            "consumable",
            "component",
            "file",
            "document",
            "item",
            "试剂",
            "设备",
            "材料",
            "耗材",
            "文件",
            "资料",
            "项目",
        )
    )
    column_scores = [_parallel_inventory_column_score([_cell_text(row[col]) for row in rows]) for col in range(2)]
    if min(column_scores) < 0.55:
        return False
    if inventory_header_signal:
        return True
    return min(column_scores) >= 0.72


def _parallel_inventory_column_score(items: list[str]) -> float:
    texts = [text for text in items if text]
    if len(texts) < 3:
        return 0.0
    compact = sum(1 for text in texts if len(text) <= 64 and len(text.split()) <= 8)
    marker_like = sum(1 for text in texts if _looks_like_inventory_marker_or_item(text))
    return (compact + marker_like) / max(1, len(texts) * 2)


def _looks_like_inventory_marker_or_item(text: str) -> bool:
    cleaned = _cell_text(text)
    if not cleaned:
        return False
    if cleaned.endswith(":"):
        return True
    if re.search(r"[*#+$†‡]\s*$", cleaned):
        return True
    if len(cleaned) <= 64 and len(cleaned.split()) <= 7:
        return True
    return False


def _join_parallel_inventory_items(items: list[str]) -> str:
    merged = ""
    for item in items:
        merged = _join_wrapped_description_text(merged, item)
    return merged


def _is_single_cell_repeat(left: list[Any], right: list[Any]) -> bool:
    if len(left) != len(right):
        return False
    left_non_empty = [(idx, _cell_text(cell)) for idx, cell in enumerate(left) if _cell_text(cell)]
    right_non_empty = [(idx, _cell_text(cell)) for idx, cell in enumerate(right) if _cell_text(cell)]
    return len(left_non_empty) == 1 and left_non_empty == right_non_empty


def _rewrite_lossless_duplicate_public_grids(table: dict[str, Any]) -> None:
    for key in ("display_grid", "data_grid", "grid"):
        value = table.get(key)
        if not isinstance(value, list) or not value:
            continue
        deduped, removed = _remove_adjacent_duplicate_rows(_clone_grid(value))
        if removed:
            table.setdefault("raw_grid_before_semantic_projection_v2", deepcopy(value))
            table[key] = deduped
            table[f"{key}_semantic_duplicate_rows_removed"] = removed


def _should_merge_same_row_wide_table_fragments(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left.get("page", 0) or 0) != int(right.get("page", 0) or 0):
        return False
    ordered_left, ordered_right = _order_left_right_tables(left, right)
    left_bbox = _bbox(ordered_left)
    right_bbox = _bbox(ordered_right)
    if not left_bbox or not right_bbox:
        return False
    if not _same_row_table_band_is_compatible(left_bbox, right_bbox):
        return False
    if not _horizontal_fragment_gap_is_compatible(left_bbox, right_bbox):
        return False

    left_grid = _grid_with_header_row(ordered_left)
    right_grid = _grid_with_header_row(ordered_right)
    if len(left_grid) < 3 or len(right_grid) < 3:
        return False
    if abs(len(left_grid) - len(right_grid)) > 1:
        return False
    left_cols = max((len(row) for row in left_grid), default=0)
    right_cols = max((len(row) for row in right_grid), default=0)
    if left_cols < 2 or right_cols < 2 or left_cols + right_cols < 4:
        return False
    if not _fragment_titles_are_compatible(ordered_left, ordered_right):
        return False
    if not _wide_fragment_headers_are_complementary(ordered_left, ordered_right, left_grid, right_grid):
        return False
    if not _wide_fragment_rows_align(left_grid, right_grid):
        return False

    shared_title = _same_nonempty_title(ordered_left, ordered_right)
    right_numeric_density = _fragment_numeric_density(right_grid[1:])
    left_stub_support = _left_fragment_has_stub_or_label_column(left_grid)
    if shared_title and right_numeric_density >= 0.45:
        return True
    return left_stub_support and right_numeric_density >= 0.65


def _merge_same_row_wide_table_fragment_into_left(left: dict[str, Any], right: dict[str, Any]) -> None:
    ordered_left, ordered_right = _order_left_right_tables(left, right)
    left_grid = _grid_with_header_row(ordered_left)
    right_grid = _grid_with_header_row(ordered_right)
    merged_grid = _merge_same_row_wide_grids(left_grid, right_grid)
    target = ordered_left
    other = ordered_right
    target.setdefault(
        "raw_grid_before_semantic_projection_v2",
        deepcopy(target.get("display_grid") or target.get("raw_grid") or left_grid),
    )
    target["display_grid"] = merged_grid
    target["grid"] = deepcopy(merged_grid)
    target["data_grid"] = deepcopy(merged_grid[1:] if merged_grid else [])
    target["raw_grid"] = deepcopy(merged_grid)
    target["col_count"] = max((len(row) for row in merged_grid), default=0)
    target["logical_col_count"] = target["col_count"]
    target["physical_col_count"] = target["col_count"]
    target["row_count"] = len(merged_grid)
    target["display_row_count"] = len(merged_grid)
    target["raw_row_count"] = len(merged_grid)
    target["data_row_count"] = max(0, len(merged_grid) - 1)
    target["logical_row_count"] = len(merged_grid)
    target["data_start_row"] = 1
    target["header"] = [
        {"col": idx + 1, "text": _cell_text(cell) or f"Column {idx + 1}"}
        for idx, cell in enumerate(merged_grid[0] if merged_grid else [])
    ]
    row_texts = [" | ".join(_cell_text(cell) for cell in row) for row in merged_grid]
    target["raw_row_texts"] = list(row_texts)
    target["display_row_texts"] = list(row_texts)
    target["row_texts"] = list(row_texts)
    target["data_row_texts"] = list(row_texts[1:])
    target["bbox"] = _merge_bboxes(_bbox(target), _bbox(other)) or target.get("bbox")
    _merge_word_evidence(target, other)
    if not _cell_text(target.get("title")) and _cell_text(other.get("title")):
        target["title"] = other.get("title")
    target.setdefault("same_row_wide_fragment_merge", {})
    target["same_row_wide_fragment_merge"] = {
        "source": "table_semantic_projection_v2",
        "merge_type": "same_row_wide_table_fragments",
        "merged_fragment_id": str(other.get("table_id") or ""),
    }
    _record_semantic_fragment_merge(target, other, "same_row_wide_table_fragments")
    if target is not left:
        left.clear()
        left.update(target)


def _order_left_right_tables(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    left_bbox = _bbox(left)
    right_bbox = _bbox(right)
    if not left_bbox or not right_bbox or left_bbox[0] <= right_bbox[0]:
        return left, right
    return right, left


def _same_row_table_band_is_compatible(
    left_bbox: tuple[float, float, float, float],
    right_bbox: tuple[float, float, float, float],
) -> bool:
    vertical_overlap = max(0.0, min(left_bbox[3], right_bbox[3]) - max(left_bbox[1], right_bbox[1]))
    min_height = max(1.0, min(left_bbox[3] - left_bbox[1], right_bbox[3] - right_bbox[1]))
    if vertical_overlap / min_height < 0.62:
        return False
    center_delta = abs(((left_bbox[1] + left_bbox[3]) / 2.0) - ((right_bbox[1] + right_bbox[3]) / 2.0))
    return center_delta <= max(10.0, min_height * 0.35)


def _horizontal_fragment_gap_is_compatible(
    left_bbox: tuple[float, float, float, float],
    right_bbox: tuple[float, float, float, float],
) -> bool:
    gap = right_bbox[0] - left_bbox[2]
    if gap < -12.0:
        return False
    left_width = max(1.0, left_bbox[2] - left_bbox[0])
    right_width = max(1.0, right_bbox[2] - right_bbox[0])
    band_height = max(1.0, max(left_bbox[3], right_bbox[3]) - min(left_bbox[1], right_bbox[1]))
    return gap <= max(36.0, min(left_width, right_width) * 0.35, band_height * 1.5)


def _grid_with_header_row(table: dict[str, Any]) -> list[list[Any]]:
    grid = _primary_grid(table)
    if not grid:
        return []
    header = _header_texts(table)
    if not header:
        return _rectangular_grid(grid, _col_count(table, grid))
    col_count = max(_col_count(table, grid), len(header), max((len(row) for row in grid), default=0))
    rectangular = _rectangular_grid(grid, col_count)
    header_row = list(header[:col_count])
    while len(header_row) < col_count:
        header_row.append(None)
    first_row = rectangular[0] if rectangular else []
    if _row_signature(first_row[: len(header_row)]) == _row_signature(header_row):
        return rectangular
    first_text = " ".join(_cell_text(cell) for cell in first_row)
    header_text = " ".join(_cell_text(cell) for cell in header_row)
    if _normalize_text(header_text) and _normalize_text(header_text) in _normalize_text(first_text):
        return rectangular
    return [header_row, *rectangular]


def _fragment_titles_are_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_title = _normalize_text(_cell_text(left.get("title") or left.get("caption_text") or left.get("caption")))
    right_title = _normalize_text(_cell_text(right.get("title") or right.get("caption_text") or right.get("caption")))
    if not left_title or not right_title:
        return True
    if left_title == right_title or left_title in right_title or right_title in left_title:
        return True
    return False


def _same_nonempty_title(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_title = _normalize_text(_cell_text(left.get("title") or left.get("caption_text") or left.get("caption")))
    right_title = _normalize_text(_cell_text(right.get("title") or right.get("caption_text") or right.get("caption")))
    return bool(left_title and right_title and (left_title == right_title or left_title in right_title or right_title in left_title))


def _wide_fragment_headers_are_complementary(
    left: dict[str, Any],
    right: dict[str, Any],
    left_grid: list[list[Any]],
    right_grid: list[list[Any]],
) -> bool:
    left_header = _header_texts(left) or [_cell_text(cell) for cell in left_grid[0]]
    right_header = _header_texts(right) or [_cell_text(cell) for cell in right_grid[0]]
    left_values = [_normalize_text(text) for text in left_header if _cell_text(text) and not _is_generic_header(_cell_text(text))]
    right_values = [_normalize_text(text) for text in right_header if _cell_text(text) and not _is_generic_header(_cell_text(text))]
    if len(left_values) < 2 or len(right_values) < 2:
        return False
    overlap = set(left_values) & set(right_values)
    if len(overlap) >= max(2, min(len(left_values), len(right_values)) // 2):
        return False
    combined = len(left_values) + len(right_values)
    return combined >= 4


def _wide_fragment_rows_align(left_grid: list[list[Any]], right_grid: list[list[Any]]) -> bool:
    left_body = [row for row in left_grid[1:] if _non_empty_count(row) > 0]
    right_body = [row for row in right_grid[1:] if _non_empty_count(row) > 0]
    if len(left_body) < 2 or len(right_body) < 2:
        return False
    if abs(len(left_body) - len(right_body)) > 1:
        return False
    comparable = min(len(left_body), len(right_body))
    aligned = 0
    for left_row, right_row in zip(left_body[:comparable], right_body[:comparable]):
        left_filled = _non_empty_count(left_row)
        right_filled = _non_empty_count(right_row)
        if left_filled <= 0 or right_filled <= 0:
            continue
        if left_filled >= 2 and right_filled >= 2:
            aligned += 1
    return aligned >= max(2, int(comparable * 0.7))


def _fragment_numeric_density(rows: list[list[Any]]) -> float:
    values = [_cell_text(cell) for row in rows for cell in row if _cell_text(cell)]
    if not values:
        return 0.0
    numeric = sum(1 for value in values if _looks_numeric(value))
    return numeric / len(values)


def _left_fragment_has_stub_or_label_column(grid: list[list[Any]]) -> bool:
    rows = [row for row in grid[1:] if _non_empty_count(row) > 0]
    if len(rows) < 2:
        return False
    first_col = [_cell_text(row[0] if row else None) for row in rows]
    non_empty = [text for text in first_col if text]
    if len(non_empty) < 2:
        return False
    non_numeric = sum(1 for text in non_empty if not _looks_numeric(text))
    return non_numeric >= max(2, int(len(non_empty) * 0.6))


def _merge_same_row_wide_grids(left_grid: list[list[Any]], right_grid: list[list[Any]]) -> list[list[Any]]:
    row_count = max(len(left_grid), len(right_grid))
    left_width = max((len(row) for row in left_grid), default=0)
    right_width = max((len(row) for row in right_grid), default=0)
    merged: list[list[Any]] = []
    for row_idx in range(row_count):
        left_row = list(left_grid[row_idx]) if row_idx < len(left_grid) else []
        right_row = list(right_grid[row_idx]) if row_idx < len(right_grid) else []
        while len(left_row) < left_width:
            left_row.append(None)
        while len(right_row) < right_width:
            right_row.append(None)
        merged.append(left_row + right_row)
    return merged


def _should_merge_stacked_header_body_table_fragments(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left.get("page", 0) or 0) != int(right.get("page", 0) or 0):
        return False
    top, bottom = _order_top_bottom_tables(left, right)
    top_bbox = _bbox(top)
    bottom_bbox = _bbox(bottom)
    if not top_bbox or not bottom_bbox:
        return False
    if top is not left:
        return False
    if not _fragment_titles_are_compatible(top, bottom):
        return False
    if not _same_nonempty_title(top, bottom):
        return False
    top_grid = _primary_grid(top)
    bottom_grid = _primary_grid(bottom)
    if not top_grid or not bottom_grid:
        return False
    if len(top_grid) > 4 or len(bottom_grid) < 3:
        return False
    top_cols = max((len(row) for row in top_grid), default=0)
    bottom_cols = max((len(row) for row in bottom_grid), default=0)
    if top_cols < 2 or bottom_cols < 2 or bottom_cols < top_cols:
        return False
    if not _stacked_fragments_geometry_is_compatible(top_bbox, bottom_bbox):
        return False
    if not _looks_like_stacked_header_fragment(top_grid):
        return False
    if not _looks_like_stacked_body_fragment(bottom_grid):
        return False
    return True


def _should_drop_header_only_fragment_before_body_table(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left.get("page", 0) or 0) != int(right.get("page", 0) or 0):
        return False
    left_bbox = _bbox(left)
    right_bbox = _bbox(right)
    if not left_bbox or not right_bbox:
        return False
    if left_bbox[1] > right_bbox[1]:
        return False
    gap = right_bbox[1] - left_bbox[3]
    if gap < -2.0 or gap > 36.0:
        return False
    left_grid = _primary_grid(left)
    right_grid = _primary_grid(right)
    if not _looks_like_header_only_table_fragment(left, left_grid):
        return False
    if len(right_grid) < 2:
        return False
    left_header = _header_texts(left) or [_cell_text(cell) for row in left_grid for cell in row if _cell_text(cell)]
    right_header = _header_texts(right)
    if not right_header:
        right_header = [_cell_text(cell) for cell in right_grid[0] if _cell_text(cell)]
    if not _header_signatures_compatible(left_header, right_header):
        return False
    right_cols = _col_count(right, right_grid)
    right_rows = _rectangular_grid(_clone_grid(right_grid), right_cols)
    return sum(1 for row in right_rows[: min(5, len(right_rows))] if _row_looks_like_data_record(row, right_cols)) >= 2


def _looks_like_header_only_table_fragment(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    rows = [row for row in grid if _non_empty_count(row) > 0]
    if not rows or len(rows) > 2:
        return False
    col_count = _col_count(table, grid)
    if col_count < 2:
        return False
    if any(_row_looks_like_data_record(row, col_count) for row in rows):
        return False
    values = [_cell_text(cell) for row in rows for cell in row if _cell_text(cell)]
    if not values:
        return False
    if any(_row_text_looks_like_prose_sentence(value) for value in values):
        return False
    return sum(1 for value in values if _looks_like_header_fragment_text(value)) >= max(2, len(values) - 1)


def _header_signatures_compatible(left_header: list[str], right_header: list[str]) -> bool:
    left_tokens = [_normalize_text(text) for text in left_header if _normalize_text(text)]
    right_tokens = [_normalize_text(text) for text in right_header if _normalize_text(text)]
    if not left_tokens or not right_tokens:
        return False
    left_joined = " ".join(left_tokens)
    right_joined = " ".join(right_tokens)
    if left_joined == right_joined:
        return True
    left_set = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", left_joined))
    right_set = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", right_joined))
    if not left_set or not right_set:
        return False
    overlap = len(left_set & right_set)
    return overlap / max(1, min(len(left_set), len(right_set))) >= 0.75


def _merge_stacked_header_body_table_fragment_into_left(left: dict[str, Any], right: dict[str, Any]) -> None:
    top, bottom = _order_top_bottom_tables(left, right)
    top_grid = _primary_grid(top)
    bottom_grid = _primary_grid(bottom)
    if not top_grid or not bottom_grid:
        return
    body_width = max((len(row) for row in bottom_grid), default=0)
    header_grid, header_spans = _project_stacked_header_fragment_to_body_width(top_grid, body_width)
    bottom_rows = _rectangular_grid(bottom_grid, body_width)
    merged_grid = [*header_grid, *bottom_rows]
    top.setdefault(
        "raw_grid_before_semantic_projection_v2",
        deepcopy(top.get("display_grid") or top.get("raw_grid") or top_grid),
    )
    top["display_grid"] = deepcopy(merged_grid)
    top["grid"] = deepcopy(merged_grid)
    top["data_grid"] = deepcopy(merged_grid[len(header_grid):])
    top["raw_grid"] = deepcopy(merged_grid)
    top["col_count"] = body_width
    top["logical_col_count"] = body_width
    top["physical_col_count"] = body_width
    top["row_count"] = len(merged_grid)
    top["display_row_count"] = len(merged_grid)
    top["raw_row_count"] = len(merged_grid)
    top["data_row_count"] = max(0, len(merged_grid) - len(header_grid))
    top["logical_row_count"] = len(merged_grid)
    top["data_start_row"] = len(header_grid)
    top["header"] = [
        {"col": idx + 1, "text": _cell_text(cell) or f"Column {idx + 1}"}
        for idx, cell in enumerate(merged_grid[0] if merged_grid else [])
    ]
    row_texts = [" | ".join(_cell_text(cell) for cell in row) for row in merged_grid]
    top["raw_row_texts"] = list(row_texts)
    top["display_row_texts"] = list(row_texts)
    top["row_texts"] = list(row_texts)
    top["data_row_texts"] = list(row_texts[len(header_grid):])
    top["bbox"] = _merge_bboxes(_bbox(top), _bbox(bottom)) or top.get("bbox")
    _merge_word_evidence(top, bottom)
    if not _cell_text(top.get("title")) and _cell_text(bottom.get("title")):
        top["title"] = bottom.get("title")
    if header_spans:
        top["header_column_groups"] = [dict(item) for item in header_spans]
    top["stacked_header_body_fragment_merge"] = {
        "source": "table_semantic_projection_v2",
        "merge_type": "stacked_header_body_table_fragments",
        "merged_fragment_id": str(bottom.get("table_id") or ""),
        "header_row_count": len(header_grid),
    }
    _record_semantic_fragment_merge(top, bottom, "stacked_header_body_table_fragments")
    if top is not left:
        left.clear()
        left.update(top)


def _order_top_bottom_tables(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    left_bbox = _bbox(left)
    right_bbox = _bbox(right)
    if not left_bbox or not right_bbox or left_bbox[1] <= right_bbox[1]:
        return left, right
    return right, left


def _stacked_fragments_geometry_is_compatible(
    top_bbox: tuple[float, float, float, float],
    bottom_bbox: tuple[float, float, float, float],
) -> bool:
    top_width = max(1.0, top_bbox[2] - top_bbox[0])
    bottom_width = max(1.0, bottom_bbox[2] - bottom_bbox[0])
    overlap = max(0.0, min(top_bbox[2], bottom_bbox[2]) - max(top_bbox[0], bottom_bbox[0]))
    if overlap / max(top_width, bottom_width) < 0.72:
        return False
    gap = bottom_bbox[1] - top_bbox[3]
    top_height = max(1.0, top_bbox[3] - top_bbox[1])
    return -3.0 <= gap <= max(42.0, top_height * 0.9)


def _looks_like_stacked_header_fragment(grid: list[list[Any]]) -> bool:
    rows = [row for row in grid if _non_empty_count(row) > 0]
    if not rows or len(rows) > 4:
        return False
    values = [_cell_text(cell) for row in rows for cell in row if _cell_text(cell)]
    if not values:
        return False
    prose_like = sum(1 for value in values if _row_text_looks_like_prose_sentence(value))
    numeric_like = sum(1 for value in values if _looks_numeric(value))
    generic_like = sum(1 for value in values if _is_generic_header(value))
    header_like = sum(1 for value in values if _looks_like_header_fragment_text(value) or _looks_like_compact_table_value(value))
    return prose_like == 0 and header_like + generic_like >= max(2, len(values) - numeric_like)


def _looks_like_stacked_body_fragment(grid: list[list[Any]]) -> bool:
    rows = [row for row in grid if _non_empty_count(row) > 0]
    if len(rows) < 3:
        return False
    dense = sum(1 for row in rows[:8] if _non_empty_count(row) >= max(2, len(row) // 2))
    value_like = 0
    for row in rows[:8]:
        values = [_cell_text(cell) for cell in row if _cell_text(cell)]
        if len(values) >= 2 and any(_looks_numeric(value) or _looks_like_compact_table_value(value) for value in values[1:]):
            value_like += 1
    return dense >= 2 and value_like >= 2


def _project_stacked_header_fragment_to_body_width(
    header_grid: list[list[Any]],
    body_width: int,
) -> tuple[list[list[str | None]], list[dict[str, Any]]]:
    if body_width <= 0:
        return [], []
    rows = _rectangular_grid(_clone_grid(header_grid), max((len(row) for row in header_grid), default=0))
    if not rows:
        return [], []
    if max((len(row) for row in rows), default=0) == body_width:
        return [[_cell_text(cell) or None for cell in row[:body_width]] for row in rows], []
    if len(rows) >= 2 and len([cell for cell in rows[1] if _cell_text(cell)]) == body_width - 2:
        first_row_values = [_cell_text(cell) for cell in rows[0] if _cell_text(cell)]
        second_row_values = [_cell_text(cell) for cell in rows[1] if _cell_text(cell)]
        stub, group = _split_stacked_stub_and_group_header(first_row_values[0] if first_row_values else "")
        trailing = first_row_values[-1] if len(first_row_values) > 1 else ""
        middle_parts = first_row_values[1:-1]
        if not group:
            group = _join_schema_header_parts([text for text in middle_parts if not _is_generic_header(text)])
        top: list[str | None] = [None for _ in range(body_width)]
        top[0] = stub or (first_row_values[0] if first_row_values else None)
        if body_width > 2:
            top[1] = group or None
        if trailing and _normalize_text(trailing) != _normalize_text(top[0]):
            top[-1] = trailing
        second: list[str | None] = [None, *second_row_values[: body_width - 2], None]
        spans: list[dict[str, Any]] = []
        if top[1]:
            spans.append(
                {
                    "row": 0,
                    "col": 1,
                    "rowspan": 1,
                    "colspan": max(1, body_width - 2),
                    "text": top[1],
                    "source": "header_column_group",
                }
            )
        if top[-1]:
            spans.append(
                {
                    "row": 0,
                    "col": body_width - 1,
                    "rowspan": 2,
                    "colspan": 1,
                    "text": top[-1],
                    "source": "header_row_group",
                }
            )
        return [top, second], spans

    projected = []
    for row in rows:
        next_row = [_cell_text(cell) or None for cell in row[:body_width]]
        while len(next_row) < body_width:
            next_row.append(None)
        projected.append(next_row)
    return projected, []


def _split_stacked_stub_and_group_header(text: str) -> tuple[str, str]:
    cleaned = _cell_text(text)
    if not cleaned:
        return "", ""
    parts = cleaned.split()
    if len(parts) < 3:
        return cleaned, ""
    first = parts[0]
    if re.fullmatch(r"[A-Z]{2,8}s?", first) or (len(first) <= 6 and first[:1].isupper()):
        return first, " ".join(parts[1:])
    return cleaned, ""


def _should_merge_two_column_inventory_fragments(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left.get("page", 0) or 0) != int(right.get("page", 0) or 0):
        return False
    left_grid = _primary_grid(left)
    right_grid = _primary_grid(right)
    if not (
        _looks_like_two_column_inventory_compatible(left, left_grid)
        and _looks_like_two_column_inventory_compatible(right, right_grid)
    ):
        return False
    if _has_independent_title(right):
        return False
    if not _fragment_geometry_is_compatible(left, right):
        return False
    left_last = _last_meaningful_row(left_grid)
    right_first = _first_meaningful_row(right_grid)
    if left_last and right_first and _row_signature(left_last) == _row_signature(right_first):
        return True
    if right_first and (_non_empty_count(right_first) == 1 or any(_SECTION_LABEL_RE.search(_cell_text(cell)) for cell in right_first)):
        return True
    left_header = _header_texts(left) or [_cell_text(cell) for cell in left_grid[0]]
    right_header = _header_texts(right) or [_cell_text(cell) for cell in right_grid[0]]
    return _compatible_two_column_headers(left_header, right_header)


def _merge_table_fragment_into_left(left: dict[str, Any], right: dict[str, Any]) -> None:
    left_grid = _primary_grid(left)
    right_grid = _primary_grid(right)
    if not left_grid or not right_grid:
        return
    right_rows = list(right_grid)
    left_header = _header_texts(left) or [_cell_text(cell) for cell in left_grid[0]]
    right_header = _header_texts(right) or [_cell_text(cell) for cell in right_grid[0]]
    if _compatible_two_column_headers(left_header, right_header):
        right_rows = right_rows[1:]
    merged_grid, removed = _remove_adjacent_duplicate_rows([*left_grid, *right_rows])
    left.setdefault("semantic_fragment_merge_v2", {})
    left["semantic_fragment_merge_v2"] = {
        "merged_fragment_ids": [
            *list((left.get("semantic_fragment_merge_v2") or {}).get("merged_fragment_ids", []) or []),
            str(right.get("table_id") or ""),
        ],
        "duplicate_boundary_rows_removed": removed,
        "source": "table_semantic_projection_v2",
    }
    left.setdefault("raw_grid_before_semantic_projection_v2", deepcopy(left.get("display_grid") or left.get("raw_grid") or left_grid))
    left["display_grid"] = merged_grid
    left["grid"] = deepcopy(merged_grid)
    left["data_grid"] = merged_grid[1:] if merged_grid else []
    left["col_count"] = 2
    _merge_word_evidence(left, right)
    if not left.get("header") and merged_grid:
        left["header"] = [{"col": idx + 1, "text": _cell_text(cell)} for idx, cell in enumerate(merged_grid[0])]


def _should_merge_flowchart_tail_fragment(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left.get("page", 0) or 0) != int(right.get("page", 0) or 0):
        return False
    if str(left.get("table_family") or "") != "flowchart_matrix":
        return False
    right_grid = _primary_grid(right)
    if _col_count(right, right_grid) != 1 or len(right_grid) < 3:
        return False
    if not _fragment_geometry_is_compatible(left, right):
        return False
    title = _cell_text(right.get("title") or right.get("caption_text"))
    rows = [_cell_text(row[0] if row else None) for row in right_grid]
    values = [value for value in rows if value]
    if title and values and _normalize_text(values[0]) == _normalize_text(title):
        values = values[1:]
    if len(values) < 2:
        return False
    text = " ".join([title, *values]).strip()
    return bool(_extract_flowchart_connector(text)) or _looks_like_flowchart_tail_text(text)


def _merge_flowchart_tail_fragment_into_left(left: dict[str, Any], right: dict[str, Any]) -> None:
    left_grid = _primary_grid(left)
    right_grid = _primary_grid(right)
    if not left_grid or not right_grid:
        return
    col_count = _col_count(left, left_grid)
    if col_count < 3:
        return
    title = _cell_text(right.get("title") or right.get("caption_text"))
    values = [_cell_text(row[0] if row else None) for row in right_grid]
    values = [value for value in values if value]
    if title and values and _normalize_text(values[0]) == _normalize_text(title):
        values = values[1:]
    merged_text = _join_wrapped_description_text(title, " ".join(values))
    if not merged_text:
        return
    tail_row: list[str | None] = [None] * col_count
    tail_row[col_count - 1] = merged_text
    merged_grid = [*left_grid, tail_row]
    left.setdefault("raw_grid_before_semantic_projection_v2", deepcopy(left.get("display_grid") or left.get("raw_grid") or left_grid))
    left["display_grid"] = merged_grid
    left["grid"] = deepcopy(merged_grid)
    left["data_grid"] = merged_grid[1:] if merged_grid else []
    _merge_word_evidence(left, right)
    left.setdefault("semantic_fragment_merge_v2", {})
    left["semantic_fragment_merge_v2"] = {
        "merged_fragment_ids": [
            *list((left.get("semantic_fragment_merge_v2") or {}).get("merged_fragment_ids", []) or []),
            str(right.get("table_id") or ""),
        ],
        "source": "flowchart_tail_fragment_merge",
    }


def _should_merge_matrix_right_edge_tail_fragment(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left.get("page", 0) or 0) != int(right.get("page", 0) or 0):
        return False
    left_grid = _primary_grid(left)
    right_grid = _primary_grid(right)
    if len(left_grid) < 2 or _col_count(left, left_grid) < 3:
        return False
    if not _can_own_matrix_right_edge_tail(left, left_grid):
        return False
    if not _looks_like_single_column_tail_fragment(right, right_grid):
        return False
    if not _matrix_tail_geometry_is_compatible(left, right, left_grid):
        return False
    tail_parts = _tail_fragment_text_parts(right, right_grid)
    if len(tail_parts) < 2:
        return False
    target = _matrix_right_edge_tail_target_cell(left_grid)
    if target is None:
        return False
    target_text = _cell_text(left_grid[target[0]][target[1]])
    if not target_text:
        return False
    first_tail = _cell_text(tail_parts[0])
    title = _cell_text(right.get("title") or right.get("caption_text") or right.get("caption"))
    if title and first_tail and _normalize_text(title) not in {_normalize_text(first_tail), _normalize_text(target_text)}:
        return False
    return _tail_text_continues_owner_cell(target_text, tail_parts)


def _can_own_matrix_right_edge_tail(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    family = str(table.get("table_family") or (table.get("semantic_projection_v2") or {}).get("table_family") or "")
    if family in {"comparison_matrix", "flowchart_matrix", "projected_stub_matrix", "rowspan_grouped_table"}:
        return True
    if _col_count(table, grid) < 3:
        return False
    text = _grid_text(grid)
    if _ARROW_RE.search(text):
        return True
    header = _header_texts(table) or [_cell_text(cell) for cell in grid[0]]
    non_empty_headers = [text for text in header if text and not _is_generic_header(text)]
    return len(non_empty_headers) >= 2 and _looks_like_comparison_matrix(table, grid)


def _looks_like_single_column_tail_fragment(table: dict[str, Any], grid: list[list[Any]]) -> bool:
    if not grid:
        return False
    source = str(table.get("detection_source") or table.get("detection_method") or table.get("source") or "")
    if source not in {"structured_text_region", "text_aligned_borderless_grid", "visual_structure_grid"}:
        return False
    col_count = max((len(row) for row in grid if isinstance(row, list)), default=0)
    if col_count != 1 or len(grid) < 2:
        return False
    parts = _tail_fragment_text_parts(table, grid)
    if len(parts) < 2:
        return False
    long_parts = [text for text in parts if len(text) >= 18 or len(text.split()) >= 3]
    if not long_parts:
        return False
    numeric_parts = [text for text in parts if _looks_numeric(text)]
    return len(numeric_parts) <= max(1, len(parts) // 3)


def _matrix_tail_geometry_is_compatible(
    left: dict[str, Any],
    right: dict[str, Any],
    left_grid: list[list[Any]],
) -> bool:
    left_bbox = _bbox(left)
    right_bbox = _bbox(right)
    if not left_bbox or not right_bbox:
        return True
    lx0, ly0, lx1, ly1 = left_bbox
    rx0, ry0, rx1, ry1 = right_bbox
    left_width = max(1.0, lx1 - lx0)
    right_width = max(1.0, rx1 - rx0)
    horizontal_overlap = max(0.0, min(lx1, rx1) - max(lx0, rx0)) / right_width
    right_center = (rx0 + rx1) / 2.0
    in_owner_right_band = right_center >= lx0 + left_width * 0.45 or rx1 >= lx0 + left_width * 0.70
    if horizontal_overlap < 0.25 or not in_owner_right_band:
        return False

    vertical_overlap = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    if vertical_overlap > 0.0:
        return True
    vertical_gap = ry0 - ly1
    if vertical_gap < 0.0:
        return False
    owner_height = max(1.0, ly1 - ly0)
    row_height = owner_height / max(1, len(left_grid))
    return vertical_gap <= max(row_height * 2.5, owner_height * 0.35)


def _tail_fragment_text_parts(table: dict[str, Any], grid: list[list[Any]]) -> list[str]:
    parts: list[str] = []
    title = _cell_text(table.get("title") or table.get("caption_text") or table.get("caption"))
    if title:
        parts.append(title)
    for row in grid:
        if not isinstance(row, list):
            continue
        text = _cell_text(row[0] if row else None)
        if text:
            parts.append(text)
    result: list[str] = []
    seen: set[str] = set()
    for text in parts:
        signature = _normalize_text(text)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        result.append(text)
    return result


def _matrix_right_edge_tail_target_cell(grid: list[list[Any]]) -> tuple[int, int] | None:
    col_count = max((len(row) for row in grid if isinstance(row, list)), default=0)
    if col_count <= 0:
        return None
    right_edge_start = max(0, col_count - 2)
    fallback: tuple[int, int] | None = None
    for row_idx in range(len(grid) - 1, 0, -1):
        row = grid[row_idx]
        if not isinstance(row, list):
            continue
        for col_idx in range(min(len(row), col_count) - 1, right_edge_start - 1, -1):
            if _cell_text(row[col_idx]):
                return row_idx, col_idx
        if fallback is None:
            for col_idx in range(min(len(row), col_count) - 1, -1, -1):
                if _cell_text(row[col_idx]):
                    fallback = (row_idx, col_idx)
                    break
    return fallback


def _tail_text_continues_owner_cell(owner_text: str, tail_parts: list[str]) -> bool:
    owner_norm = _normalize_text(owner_text)
    if not owner_norm:
        return False
    first_tail_norm = _normalize_text(tail_parts[0]) if tail_parts else ""
    if first_tail_norm and (first_tail_norm in owner_norm or owner_norm in first_tail_norm):
        return True
    if len(tail_parts) >= 2 and _looks_like_wrapped_description_continuation(tail_parts[1], owner_text):
        return True
    joined_tail = " ".join(tail_parts[: min(3, len(tail_parts))])
    return _row_text_looks_like_prose_sentence(_join_wrapped_description_text(owner_text, joined_tail))


def _merge_matrix_right_edge_tail_fragment_into_left(left: dict[str, Any], right: dict[str, Any]) -> None:
    grid = _primary_grid(left)
    target = _matrix_right_edge_tail_target_cell(grid)
    if target is None:
        return
    tail_parts = _tail_fragment_text_parts(right, _primary_grid(right))
    row_idx, col_idx = target
    owner_text = _cell_text(grid[row_idx][col_idx])
    merged_text = owner_text
    for part in tail_parts:
        normalized_part = _normalize_text(part)
        if not normalized_part:
            continue
        normalized_merged = _normalize_text(merged_text)
        if normalized_part in normalized_merged:
            continue
        merged_text = _join_wrapped_description_text(merged_text, part)
    grid[row_idx][col_idx] = merged_text

    left.setdefault("raw_grid_before_semantic_projection_v2", deepcopy(left.get("display_grid") or left.get("raw_grid") or grid))
    left["display_grid"] = grid
    left["grid"] = deepcopy(grid)
    left["data_grid"] = grid[1:] if grid else []
    left["col_count"] = max((len(row) for row in grid if isinstance(row, list)), default=0)
    left["bbox"] = _merge_bboxes(_bbox(left), _bbox(right)) or left.get("bbox")
    _merge_word_evidence(left, right)
    _record_semantic_fragment_merge(left, right, "matrix_right_edge_tail_fragment")


def _merge_bboxes(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> list[float] | None:
    if not left:
        return list(right) if right else None
    if not right:
        return list(left)
    return [
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    ]


def _merge_word_evidence(left: dict[str, Any], right: dict[str, Any]) -> None:
    words = [*_table_word_evidence(left), *_table_word_evidence(right)]
    if not words:
        return
    seen: set[tuple[str, float, float, float, float]] = set()
    merged: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda item: (_word_y_center(item), _word_x_center(item), str(item.get("text") or ""))):
        bbox = word.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        key = (
            _cell_text(word.get("text")),
            round(float(bbox[0]), 3),
            round(float(bbox[1]), 3),
            round(float(bbox[2]), 3),
            round(float(bbox[3]), 3),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append({"text": key[0], "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]})
    left["word_evidence"] = merged


def _record_semantic_fragment_merge(left: dict[str, Any], right: dict[str, Any], merge_type: str) -> None:
    metadata = left.get("semantic_fragment_merge_v2")
    if not isinstance(metadata, dict):
        metadata = {}
    merged_ids = list(metadata.get("merged_fragment_ids", []) or [])
    right_id = str(right.get("table_id") or "").strip()
    if right_id and right_id not in merged_ids:
        merged_ids.append(right_id)
    merge_types = list(metadata.get("merge_types", []) or [])
    if merge_type not in merge_types:
        merge_types.append(merge_type)
    metadata.update(
        {
            "merged_fragment_ids": merged_ids,
            "merge_types": merge_types,
            "source": "table_semantic_projection_v2",
        }
    )
    left["semantic_fragment_merge_v2"] = metadata


def _looks_like_flowchart_tail_text(text: str) -> bool:
    cleaned = _cell_text(text).lower()
    if not cleaned:
        return False
    flow_tokens = ("blood", "cell", "protein", "gene", "hemoglobin", "vessel", "step", "result", "outcome")
    return sum(1 for token in flow_tokens if token in cleaned) >= 2


def _fragment_geometry_is_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_bbox = _bbox(left)
    right_bbox = _bbox(right)
    if not left_bbox or not right_bbox:
        return True
    lx0, ly0, lx1, ly1 = left_bbox
    rx0, ry0, rx1, _ = right_bbox
    left_width = max(1.0, lx1 - lx0)
    right_width = max(1.0, rx1 - rx0)
    overlap = max(0.0, min(lx1, rx1) - max(lx0, rx0)) / max(left_width, right_width)
    vertical_gap = ry0 - ly1
    left_height = max(1.0, ly1 - ly0)
    return overlap >= 0.45 and vertical_gap <= max(80.0, left_height * 0.8)


def _bbox(table: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = table.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except Exception:
        return None


def _has_independent_title(table: dict[str, Any]) -> bool:
    title = _cell_text(table.get("title"))
    if not title:
        return False
    return bool(re.match(r"^(?:table|表)\s*[\w\d一二三四五六七八九十xX.-]+", title, re.IGNORECASE))


def _compatible_two_column_headers(left: list[str], right: list[str]) -> bool:
    if len(left) < 2 or len(right) < 2:
        return False
    left_norm = [_normalize_text(text) for text in left[:2]]
    right_norm = [_normalize_text(text) for text in right[:2]]
    if left_norm == right_norm:
        return True
    if any(_is_generic_header(text) for text in right[:2]):
        return True
    return left_norm[0] == right_norm[0] or left_norm[1] == right_norm[1]


def _first_meaningful_row(grid: list[list[Any]]) -> list[Any] | None:
    for row in grid:
        if _non_empty_count(row) > 0:
            return row
    return None


def _last_meaningful_row(grid: list[list[Any]]) -> list[Any] | None:
    for row in reversed(grid):
        if _non_empty_count(row) > 0:
            return row
    return None


def _row_signature(row: list[Any]) -> tuple[str, ...]:
    return tuple(_normalize_text(_cell_text(cell)) for cell in row)


def _grid_text(grid: list[list[Any]]) -> str:
    return " ".join(_cell_text(cell) for row in grid for cell in row if _cell_text(cell))


def _non_empty_count(row: list[Any]) -> int:
    return sum(1 for cell in row if _cell_text(cell))


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _is_generic_header(value: str) -> bool:
    return bool(_GENERIC_HEADER_RE.match(_cell_text(value)))


def _looks_numeric(value: str) -> bool:
    text = _cell_text(value)
    return bool(text) and bool(_NUMERIC_RE.match(text))
