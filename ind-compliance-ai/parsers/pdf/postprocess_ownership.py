from __future__ import annotations

from copy import deepcopy
from typing import Any

_OWNED_TEXT_EDGE_PROJECTION_SPECS: dict[str, tuple[dict[str, Any], ...]] = {
    "table": (
        {
            "source_key": "note_blocks",
            "relation": "table_note",
        },
        {
            "source_key": "header_note_refs",
            "source_role": "table_header_note_ref",
            "relation": "table_header_note_ref",
            "source_id_keys": ("note_source_block_id", "note_block_id", "source_block_id", "source_block_ids"),
            "extra_edge_fields": (
                "marker",
                "header_row",
                "header_col",
                "header_col_1based",
                "header_text",
                "note_text",
            ),
        },
        {
            "source_key": "cell_note_refs",
            "source_role": "table_cell_note_ref",
            "relation": "table_cell_note_ref",
            "source_id_keys": ("note_source_block_id", "note_block_id", "source_block_id", "source_block_ids"),
            "extra_edge_fields": (
                "marker",
                "data_row",
                "data_col",
                "data_col_1based",
                "cell_text",
                "note_text",
            ),
        },
    ),
    "image": (
        {
            "source_key": "caption_blocks",
            "relation": "figure_caption",
        },
        {
            "source_key": "content_segments",
            "relation": "figure_legend",
            "role_allowlist": ("legend", "figure_legend", "caption_legend"),
        },
    ),
    "structure_template": (
        {
            "source_key": "note_blocks",
            "relation": "structure_template_note",
        },
        {
            "source_key": "local_note_refs",
            "source_role": "local_note_ref",
            "relation": "structure_template_local_note_ref",
            "source_id_keys": ("note_block_id", "note_source_block_id", "source_block_id", "source_block_ids"),
            "extra_edge_fields": ("marker", "anchor_text", "anchor_row_index", "note_text", "confidence"),
        },
    ),
}


def _copy_metadata_reference_edges_to_evidence(evidence: dict[str, Any], source: dict[str, Any]) -> None:
    edges = [
        dict(edge)
        for edge in source.get("metadata_reference_edges", []) or []
        if isinstance(edge, dict)
    ]
    if edges:
        evidence["metadata_reference_edges"] = edges
    source_ids = [
        str(block_id).strip()
        for block_id in source.get("title_metadata_source_block_ids", []) or []
        if str(block_id).strip()
    ]
    if source_ids:
        evidence["title_metadata_source_block_ids"] = source_ids


def _project_owned_text_metadata_reference_edges(
    evidence: dict[str, Any],
    *,
    source_object: dict[str, Any],
    target_object_type: str,
    target_object_id: str,
) -> None:
    for spec in _OWNED_TEXT_EDGE_PROJECTION_SPECS.get(target_object_type, ()):
        source_key = str(spec.get("source_key") or "").strip()
        if not source_key:
            continue
        role_allowlist = {
            str(role or "").strip()
            for role in spec.get("role_allowlist", ()) or ()
            if str(role or "").strip()
        }
        source_role = str(spec.get("source_role") or "").strip()
        segments: list[dict[str, Any]] = []
        for item in source_object.get(source_key, []) or []:
            if not isinstance(item, dict):
                continue
            segment_role = str(item.get("role") or "").strip()
            if role_allowlist and segment_role not in role_allowlist:
                continue
            segment = dict(item)
            if source_role:
                segment["role"] = source_role
            segments.append(segment)
        if not segments:
            continue
        _append_owned_text_metadata_reference_edges(
            evidence,
            target_object_type=target_object_type,
            target_object_id=target_object_id,
            relation=str(spec.get("relation") or "").strip(),
            segments=segments,
            source_id_keys=tuple(spec.get("source_id_keys", ("source_block_id", "source_block_ids"))),
            extra_edge_fields=tuple(spec.get("extra_edge_fields", ())),
        )


def validate_ownership_graph_invariants(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    blocks_by_id = {
        block_id: block
        for block in blocks or []
        if isinstance(block, dict)
        for block_id in [_ownership_graph_block_id(block)]
        if block_id
    }
    seen_edges: set[tuple[str, str, str, str]] = set()
    metadata_only_sources: dict[str, dict[str, Any]] = {}
    visible_title_owners: dict[str, list[str]] = {}
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        owner_id = _ownership_graph_block_id(block)
        visible_title_owner = str(block.get("visible_title_owner") or "").strip()
        if visible_title_owner:
            visible_title_owners.setdefault(visible_title_owner, []).append(owner_id)
        for edge in block.get("metadata_reference_edges", []) or []:
            if not isinstance(edge, dict):
                continue
            source_block_id = str(edge.get("source_block_id") or "").strip()
            relation = str(edge.get("relation") or "").strip()
            target_object_type = str(edge.get("target_object_type") or "").strip()
            target_object_id = str(edge.get("target_object_id") or "").strip()
            if not source_block_id or not relation or not target_object_type or not target_object_id:
                continue
            key = (source_block_id, relation, target_object_type, target_object_id)
            if relation == "title_metadata_continuation" and str(edge.get("visible_render_policy") or "").strip() != "metadata_only":
                diagnostics.append(
                    {
                        "invariant": "title_metadata_continuation_not_metadata_only",
                        "source_block_id": source_block_id,
                        "relation": relation,
                        "target_object_type": target_object_type,
                        "target_object_id": target_object_id,
                        "owner_block_id": owner_id,
                        "severity": "warning",
                    }
                )
            if str(edge.get("visible_render_policy") or "").strip() == "metadata_only":
                metadata_only_sources.setdefault(
                    source_block_id,
                    {
                        "source_block_id": source_block_id,
                        "relation": relation,
                        "target_object_type": target_object_type,
                        "target_object_id": target_object_id,
                        "owner_block_id": owner_id,
                    },
                )
            if target_object_id not in blocks_by_id:
                diagnostics.append(
                    {
                        "invariant": "metadata_edge_target_owner_missing",
                        "source_block_id": source_block_id,
                        "relation": relation,
                        "target_object_type": target_object_type,
                        "target_object_id": target_object_id,
                        "owner_block_id": owner_id,
                        "severity": "warning",
                    }
                )
            if key in seen_edges:
                diagnostics.append(
                    {
                        "invariant": "duplicate_metadata_reference_edge",
                        "source_block_id": source_block_id,
                        "relation": relation,
                        "target_object_type": target_object_type,
                        "target_object_id": target_object_id,
                        "owner_block_id": owner_id,
                        "severity": "warning",
                    }
                )
                continue
            seen_edges.add(key)
    for source_block_id, source_context in metadata_only_sources.items():
        source_block = blocks_by_id.get(source_block_id)
        if not source_block or _ownership_graph_block_metadata_only(source_block):
            continue
        diagnostics.append(
            {
                "invariant": "metadata_only_source_has_visible_rendering",
                "source_block_id": source_block_id,
                "relation": source_context.get("relation"),
                "target_object_type": source_context.get("target_object_type"),
                "target_object_id": source_context.get("target_object_id"),
                "owner_block_id": source_context.get("owner_block_id"),
                "source_block_type": source_block.get("block_type"),
                "source_semantic_role": source_block.get("semantic_role"),
                "severity": "warning",
            }
        )
    for visible_title_owner, owner_block_ids in visible_title_owners.items():
        unique_owner_block_ids = [block_id for block_id in dict.fromkeys(owner_block_ids) if block_id]
        if len(unique_owner_block_ids) <= 1:
            continue
        diagnostics.append(
            {
                "invariant": "multiple_visible_title_owners",
                "visible_title_owner": visible_title_owner,
                "owner_block_ids": unique_owner_block_ids,
                "severity": "warning",
            }
        )
    absorbed_structure_templates = {
        block_id
        for block in blocks or []
        if isinstance(block, dict)
        for block_id in [_ownership_graph_block_id(block)]
        if block_id
        and str(block.get("block_type") or "").strip() == "structure_template"
        and _ownership_graph_block_is_absorbed(block)
    }
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        block_id = _ownership_graph_block_id(block)
        if not block_id or str(block.get("block_type") or "").strip() != "table":
            continue
        if str(block.get("visible_render_policy") or "").strip() == "metadata_only":
            continue
        absorbed_owner_ids = [
            str(owner_id).strip()
            for owner_id in block.get("absorbed_structure_template_ids", []) or []
            if str(owner_id).strip()
        ]
        if not absorbed_owner_ids or not any(owner_id in absorbed_structure_templates for owner_id in absorbed_owner_ids):
            continue
        diagnostics.append(
            {
                "invariant": "absorbed_structure_template_has_visible_table_surface",
                "owner_block_id": absorbed_owner_ids[0],
                "related_block_id": block_id,
                "severity": "warning",
            }
        )
    return diagnostics


def _ownership_graph_block_id(block: dict[str, Any]) -> str:
    for key in ("block_id", "table_id", "image_id", "structure_template_id", "source_id", "evidence_id"):
        value = str(block.get(key) or "").strip()
        if value:
            return value
    return ""


def _ownership_graph_block_metadata_only(block: dict[str, Any]) -> bool:
    if str(block.get("visible_render_policy") or "").strip() == "metadata_only":
        return True
    composite = block.get("composite_object")
    if isinstance(composite, dict) and str(composite.get("visible_render_policy") or "").strip() == "metadata_only":
        return True
    semantic_role = str(block.get("semantic_role") or "").strip()
    unit_role = str(block.get("unit_role") or "").strip()
    if semantic_role.startswith("metadata_only") or unit_role.startswith("metadata_only"):
        return True
    if semantic_role in {"absorbed_structure_template", "absorbed_structure_template_fragment"}:
        return True
    if str(block.get("ownership_domain") or "").strip() == "absorbed_by_business_table":
        return True
    return False


def _ownership_graph_block_is_absorbed(block: dict[str, Any]) -> bool:
    if str(block.get("ownership_domain") or "").strip() == "absorbed_by_business_table":
        return True
    if str(block.get("semantic_role") or "").strip() in {"absorbed_structure_template", "absorbed_structure_template_fragment"}:
        return True
    return False


def _metadata_edge_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if str(value or "").strip():
        return [str(value).strip()]
    return []


def _append_owned_text_metadata_reference_edges(
    evidence: dict[str, Any],
    *,
    target_object_type: str,
    target_object_id: str,
    relation: str,
    segments: list[dict[str, Any]],
    source_id_keys: tuple[str, ...] = ("source_block_id", "source_block_ids"),
    extra_edge_fields: tuple[str, ...] = (),
) -> None:
    if not target_object_id:
        return
    existing_edges = [
        dict(edge)
        for edge in evidence.get("metadata_reference_edges", []) or []
        if isinstance(edge, dict)
    ]
    existing_keys = {
        (
            str(edge.get("source_block_id") or "").strip(),
            str(edge.get("relation") or "").strip(),
            str(edge.get("target_object_id") or "").strip(),
        )
        for edge in existing_edges
    }
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        source_ids: list[str] = []
        for source_id_key in source_id_keys:
            source_ids.extend(_metadata_edge_string_list(segment.get(source_id_key)))
        for source_block_id in source_ids:
            key = (source_block_id, relation, target_object_id)
            if not source_block_id or key in existing_keys:
                continue
            existing_keys.add(key)
            edge = {
                "source_block_id": source_block_id,
                "source_role": str(segment.get("role") or relation).strip() or relation,
                "relation": relation,
                "target_object_type": target_object_type,
                "target_object_id": target_object_id,
                "visible_render_policy": "metadata_only",
                "metadata_title_reference_policy": "may_reference_without_visible_rendering",
            }
            for field_name in extra_edge_fields:
                value = segment.get(field_name)
                if value is None or value == "":
                    continue
                edge[field_name] = deepcopy(value) if isinstance(value, (dict, list)) else value
            existing_edges.append(edge)
    if existing_edges:
        evidence["metadata_reference_edges"] = existing_edges


def _mark_metadata_only_visibility(item: dict[str, Any]) -> None:
    item["visible_render_policy"] = "metadata_only"
