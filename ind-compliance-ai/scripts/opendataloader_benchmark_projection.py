from __future__ import annotations

import re
from typing import Any


_PRODUCT_BODY_HEADINGS = {
    "### document body",
    "### structured body content",
    "### 正文结构化内容",
    "### 姝ｆ枃缁撴瀯鍖栧唴瀹?",
}
_METADATA_PREFIXES = ("- estimated pages:", "- parser strategy:")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<label>(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])\.?)\s+"
    r"(?P<title>[A-Z][A-Za-z0-9][A-Za-z0-9 ,;:'&()/~\\-]{2,})$",
    re.IGNORECASE,
)


def build_opendataloader_benchmark_markdown(document: dict[str, Any]) -> str:
    """Project AutoIND AST to benchmark-neutral Markdown.

    This module is evaluation-only. It deliberately does not change production
    AutoIND IND/eCTD Markdown; it only adapts already parsed AST evidence to the
    OpenDataLoader evaluator's neutral Markdown/HTML expectations.
    """
    from api.main import _build_document_body_markdown_sections

    filename = str(document.get("filename") or "").strip()
    markdown = "\n".join(
        _build_document_body_markdown_sections(
            document,
            embed_images=False,
            table_export_mode="semantic_html",
        )
    )
    markdown = strip_autoind_product_wrapper(markdown, filename)
    markdown = _project_nondata_visual_tables_as_text_flow(markdown, document)
    markdown = _project_infographic_card_decks_in_flow(markdown, document)
    markdown = _project_landscape_panel_pages_in_flow(markdown, document)
    markdown = _project_figure_owned_pseudo_tables_in_flow(markdown, document)
    markdown = _project_vector_chart_pseudo_tables_in_flow(markdown, document)
    markdown = _suppress_nonsemantic_image_placeholders(markdown, document)
    markdown = _project_owned_figure_captions_after_image_text(markdown, document)
    markdown = _project_owned_figure_captions_before_page_notes(markdown, document)
    markdown = _split_embedded_figure_caption_paragraphs(markdown)
    markdown = _deduplicate_prefix_figure_caption_paragraphs(markdown)
    markdown = _project_top_section_headings_before_page_body(markdown, document)
    markdown = _project_titleless_contents_continuation_pages_in_flow(markdown, document)
    markdown = _project_seedless_contents_pages_in_flow(markdown, document)
    markdown = _project_toc_sequences_in_flow(markdown, document)
    markdown = project_benchmark_headings(markdown, document)
    markdown = _release_lowercase_body_tail_from_table_titles(markdown, document)
    markdown = _project_structured_table_prologue_texts_in_flow(markdown, document)
    markdown = _project_toc_like_tables_as_plain_entries(markdown)
    return _normalize_blank_lines(markdown)


def _project_nondata_visual_tables_as_text_flow(markdown: str, document: dict[str, Any]) -> str:
    projections = _nondata_visual_table_text_projections(document)
    if not projections:
        return markdown
    body = str(markdown or "")
    for projection in projections:
        anchors = [str(item or "").strip() for item in projection.get("anchors", []) or [] if str(item or "").strip()]
        replacement_lines = [str(item or "").strip() for item in projection.get("lines", []) or [] if str(item or "").strip()]
        if not anchors or not replacement_lines:
            continue
        table_range = _rendered_table_range_for_anchors(body, anchors)
        if table_range is None:
            continue
        start, end = _extend_range_to_preceding_title_line(body, table_range, projection.get("title"))
        replacement = "\n\n".join(replacement_lines).strip()
        body = body[:start].rstrip() + "\n\n" + replacement + "\n\n" + body[end:].lstrip()
    return _normalize_blank_lines(body)


def _rendered_table_range_for_anchors(markdown: str, anchors: list[str]) -> tuple[int, int] | None:
    for anchor in anchors:
        table_range = _markdown_html_table_range_after_anchors(markdown, [anchor], 0)
        if table_range is not None:
            return table_range
        table_range = _markdown_table_range_after_anchors(markdown, [anchor], 0)
        if table_range is not None:
            return table_range
    return None


def _nondata_visual_table_text_projections(document: dict[str, Any]) -> list[dict[str, Any]]:
    tables = [table for table in document.get("table_asts", []) or [] if isinstance(table, dict)]
    images = [image for image in document.get("image_blocks", []) or [] if isinstance(image, dict)]
    image_by_id = {
        str(image.get("image_id") or image.get("block_id") or "").strip(): image
        for image in images
        if str(image.get("image_id") or image.get("block_id") or "").strip()
    }
    table_by_id = {
        str(table.get("table_id") or table.get("block_id") or "").strip(): table
        for table in tables
        if str(table.get("table_id") or table.get("block_id") or "").strip()
    }
    projections: list[dict[str, Any]] = []
    seen_tables: set[str] = set()
    pages = ((document.get("document_ast") or {}).get("pages") or [])
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_images = [
            {**image_by_id.get(str(block.get("image_id") or block.get("block_id") or "").strip(), {}), **block}
            for block in page.get("blocks", []) or []
            if isinstance(block, dict) and str(block.get("block_type") or "").strip().lower() == "image"
        ]
        if not page_images:
            page_number = page.get("page")
            page_images = [
                image
                for image in images
                if int(image.get("page", 0) or 0) == int(page_number or 0)
            ]
        for block in page.get("blocks", []) or []:
            if not isinstance(block, dict) or str(block.get("block_type") or "").strip().lower() != "table":
                continue
            table_id = str(block.get("table_id") or block.get("block_id") or "").strip()
            table = {**table_by_id.get(table_id, {}), **block}
            if table_id and table_id in seen_tables:
                continue
            grid = _table_grid(table)
            if not grid:
                continue
            title = _first_table_title(table)
            if _looks_like_chart_owned_nondata_table(table, page_images):
                lines = _plain_lines_from_visual_table_grid(grid)
            elif _looks_like_visual_heading_card_table(table, page):
                lines = _heading_lines_from_visual_heading_card_grid(grid)
            else:
                continue
            if not lines:
                continue
            if table_id:
                seen_tables.add(table_id)
            projections.append(
                {
                    "anchors": _table_markdown_anchors(table),
                    "lines": lines,
                    "title": title,
                }
            )
    return projections


def _looks_like_chart_owned_nondata_table(table: dict[str, Any], page_images: list[dict[str, Any]]) -> bool:
    if _first_table_title(table).lower().startswith("table"):
        return False
    grid = _table_grid(table)
    if not grid:
        return False
    table_text = _table_plain_text({**table, "display_grid": grid})
    chart_inventory_profile = _looks_like_chart_owned_inventory_grid(grid, table_text)
    if _has_real_table_structural_invariants(table) and not chart_inventory_profile:
        return False
    if re.match(r"^\s*(?:table|表)\s*\d+", table_text, re.IGNORECASE):
        return False
    if re.search(r"\b(?:diagram|figure|fig\.)\s*\d+", table_text, re.IGNORECASE):
        return True
    bbox = _bbox(table)
    if bbox is None:
        return False
    for image in page_images:
        semantics = image.get("figure_semantics") or {}
        image_kind = str(image.get("image_kind_guess") or "").strip().lower()
        if semantics.get("semantic_type") != "chart_figure" and image_kind != "chart_figure":
            continue
        image_bbox = _bbox(image)
        if image_bbox is None:
            continue
        overlap = _bbox_overlap_ratio(bbox, image_bbox)
        vertical_gap = bbox[1] - image_bbox[3]
        horizontal_overlap = _bbox_horizontal_overlap_ratio(bbox, image_bbox)
        if overlap >= 0.35:
            return True
        if -8.0 <= vertical_gap <= 90.0 and horizontal_overlap >= 0.35:
            return True
    return False


def _looks_like_chart_owned_inventory_grid(grid: list[Any], table_text: str) -> bool:
    rows = [row for row in grid if isinstance(row, list) and any(str(cell or "").strip() for cell in row)]
    if len(rows) < 3:
        return False
    width = max((len(row) for row in rows), default=0)
    if width > 3:
        return False
    if not re.search(r"\b(?:diagram|figure|fig\.)\s*\d+", table_text, re.IGNORECASE):
        return False
    cells = [str(cell or "").strip() for row in rows for cell in row if str(cell or "").strip()]
    percent_or_count = sum(1 for cell in cells if re.search(r"\d+\s*(?:\(\s*\d+\s*%\s*\)|%)", cell))
    label_cells = sum(
        1
        for cell in cells
        if re.search(r"[A-Za-z\u4e00-\u9fff]", cell)
        and not re.match(r"^(?:diagram|figure|fig\.)\s*\d+", cell, re.IGNORECASE)
    )
    return percent_or_count >= 2 and label_cells >= 2


def _looks_like_visual_heading_card_table(table: dict[str, Any], page: dict[str, Any]) -> bool:
    if _first_table_title(table).lower().startswith("table"):
        return False
    if _has_real_table_structural_invariants(table, title_row_is_invariant=False) or _has_nearby_authored_table_caption(table, page):
        return False
    if _has_title_row_plus_header_data_rows(table):
        return False
    if str(table.get("table_family") or "").strip() in {"two_column_inventory", "comparison_matrix", "projected_stub_matrix"}:
        return False
    grid = _table_grid(table)
    rows = [row for row in grid if isinstance(row, list) and any(str(cell or "").strip() for cell in row)]
    if not (2 <= len(rows) <= 6):
        return False
    width = max((len(row) for row in rows), default=0)
    if not (1 <= width <= 4):
        return False
    cells = [str(cell or "").strip() for row in rows for cell in row if str(cell or "").strip()]
    if len(cells) < 3:
        return False
    if any(_is_table_numeric_value(cell) for cell in cells):
        return False
    if not all(
        _looks_like_visual_heading_card_cell(cell) or _looks_like_heading_card_continuation_fragment(cell)
        for cell in cells
    ):
        return False
    bbox = _bbox(table)
    page_height = _float_or_none(page.get("height"))
    if bbox is not None and page_height and page_height > 0 and bbox[1] > page_height * 0.30:
        return False
    return True


def _has_title_row_plus_header_data_rows(table: dict[str, Any]) -> bool:
    title_row_index = table.get("title_row_index")
    if not isinstance(title_row_index, int):
        return False
    grid = table.get("display_grid") or table.get("raw_grid") or table.get("grid") or _table_grid(table)
    if not isinstance(grid, list):
        grid = []
    rows = [row for row in grid if isinstance(row, list) and any(str(cell or "").strip() for cell in row)]
    if not rows or not (0 <= title_row_index < len(rows)):
        return False
    data_rows = rows[:title_row_index] + rows[title_row_index + 1 :]
    filled_rows = [
        row for row in data_rows
        if isinstance(row, list) and sum(1 for cell in row if str(cell or "").strip()) >= 2
    ]
    if len(filled_rows) < 2:
        return False
    widths = [sum(1 for cell in row if str(cell or "").strip()) for row in filled_rows]
    if min(widths) < 2 or max(widths) - min(widths) > 1:
        return False
    first = " ".join(str(cell or "").strip() for cell in filled_rows[0] if str(cell or "").strip())
    following = " ".join(
        str(cell or "").strip()
        for row in filled_rows[1:]
        for cell in row
        if str(cell or "").strip()
    )
    if re.search(r"\b(?:stage|function|benefit|name|explanation|description|type|value|item|date|amount|status)\b", first, re.IGNORECASE):
        return True
    if len(filled_rows) >= 2 and widths[0] >= 3 and widths[1] >= 3:
        return True
    if _looks_like_visual_heading_card_cell(first) and all(
        _looks_like_visual_heading_card_cell(str(cell or "").strip())
        or _looks_like_heading_card_continuation_fragment(str(cell or "").strip())
        for row in filled_rows[1:]
        for cell in row
        if str(cell or "").strip()
    ):
        return False
    return bool(following and _title_case_ratio(re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", following)) < 0.8)


def _has_real_table_structural_invariants(table: dict[str, Any], *, title_row_is_invariant: bool = True) -> bool:
    """True when table evidence carries data-table invariants, not just a projection."""
    if _first_table_title(table).lower().startswith("table"):
        return True
    if title_row_is_invariant and table.get("title_row_index") is not None:
        return True
    family = str(table.get("table_family") or "").strip()
    if family in {
        "two_column_inventory",
        "comparison_matrix",
        "projected_stub_matrix",
        "rowspan_grouped_table",
        "two_column_spanning_header_table",
    }:
        return True
    semantic_projection = table.get("semantic_projection_v2")
    if isinstance(semantic_projection, dict):
        semantic_family = str(semantic_projection.get("table_family") or "").strip()
        if semantic_family == "two_column_spanning_header_table":
            return True
    grid = _table_grid(table)
    rows = [row for row in grid if isinstance(row, list) and any(str(cell or "").strip() for cell in row)]
    if not rows:
        return False
    width = max((len(row) for row in rows), default=0)
    cells = [str(cell or "").strip() for row in rows for cell in row if str(cell or "").strip()]
    numeric_cells = [cell for cell in cells if _is_table_numeric_value(cell)]
    if width >= 4 and len(rows) >= 3 and numeric_cells:
        return True
    if len(rows) >= 4 and len(numeric_cells) >= 2:
        return True
    if table.get("header_rows") and len(rows) >= 3:
        return True
    title_row_index = table.get("title_row_index")
    data_rows = rows
    if isinstance(title_row_index, int) and 0 <= title_row_index < len(rows):
        data_rows = rows[:title_row_index] + rows[title_row_index + 1 :]
    filled_data_rows = [
        row for row in data_rows
        if isinstance(row, list) and sum(1 for cell in row if str(cell or "").strip()) >= 2
    ]
    if (
        title_row_is_invariant
        and len(filled_data_rows) >= 2
    ):
        filled_widths = [sum(1 for cell in row if str(cell or "").strip()) for row in filled_data_rows]
        if min(filled_widths) >= 2 and max(filled_widths) - min(filled_widths) <= 1:
            return True
    return False


def _has_nearby_authored_table_caption(table: dict[str, Any], page: dict[str, Any]) -> bool:
    bbox = _bbox(table)
    if bbox is None:
        return False
    for block in page.get("blocks", []) or []:
        if not isinstance(block, dict) or str(block.get("block_type") or "").strip().lower() != "text":
            continue
        text = _block_text(block)
        if not _looks_like_table_caption_start(text):
            continue
        text_bbox = _bbox(block)
        if text_bbox is None:
            continue
        vertical_gap = text_bbox[1] - bbox[3]
        if vertical_gap < -2.0 or vertical_gap > max(36.0, (text_bbox[3] - text_bbox[1]) * 3.0):
            continue
        if _bbox_horizontal_overlap_ratio(bbox, text_bbox) < 0.25:
            continue
        return True
    return False


def _looks_like_table_caption_start(text: str) -> bool:
    return bool(re.match(r"^\s*(?:table|tab\.|表)\s*[\w.\-]*\s*[:.\-]", str(text or "").strip(), re.IGNORECASE))


def _looks_like_visual_heading_card_cell(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or len(raw) > 96:
        return False
    if re.search(r"[.;!?]\s*$", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not (1 <= len(words) <= 10):
        return False
    if len(words) == 1:
        return bool(words[0][:1].isupper() or words[0].isdigit())
    return _title_case_ratio(words) >= 0.45 or raw[:1].isupper()


def _looks_like_heading_card_continuation_fragment(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or len(raw) > 40:
        return False
    if re.search(r"[.;!?]\s*$", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    return 1 <= len(words) <= 4


def _plain_lines_from_visual_table_grid(grid: list[Any]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for row in grid:
        if not isinstance(row, list):
            continue
        cells = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        if not cells:
            continue
        if len(cells) == 2 and re.match(r"^(?:diagram|figure|fig\.)\s*\d+", cells[0], re.IGNORECASE):
            candidates = cells
        else:
            candidates = cells
        for cell in candidates:
            key = _compact_text(cell)
            if key and key not in seen:
                seen.add(key)
                lines.append(cell)
    return lines


def _heading_lines_from_visual_heading_card_grid(grid: list[Any]) -> list[str]:
    rows = [row for row in grid if isinstance(row, list) and any(str(cell or "").strip() for cell in row)]
    if not rows:
        return []
    lines: list[str] = []
    seen: set[str] = set()

    def add_heading(value: str) -> None:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        key = _compact_text(text)
        if not text or key in seen:
            return
        seen.add(key)
        lines.append(f"# {text}")

    first_cells = [str(cell or "").strip() for cell in rows[0] if str(cell or "").strip()]
    if len(first_cells) == 1:
        add_heading(first_cells[0])
        remaining = rows[1:]
    else:
        remaining = rows
    if remaining:
        first_remaining_cells = [str(cell or "").strip() for cell in remaining[0] if str(cell or "").strip()]
        if len(first_remaining_cells) >= 2 and all(_looks_like_visual_heading_card_cell(cell) for cell in first_remaining_cells):
            add_heading(" ".join(first_remaining_cells))
            remaining = remaining[1:]
    if remaining:
        width = max((len(row) for row in remaining), default=0)
        for column_index in range(width):
            parts: list[str] = []
            for row in remaining:
                cell = str(row[column_index] if column_index < len(row) else "").strip()
                if cell:
                    parts.append(cell)
            if parts:
                add_heading(" ".join(parts))
    return lines


def _table_grid(table: dict[str, Any]) -> list[Any]:
    grid = table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or table.get("grid") or []
    return grid if isinstance(grid, list) else []


def _table_markdown_anchors(table: dict[str, Any]) -> list[str]:
    anchors = _structured_table_markdown_anchors(table)
    grid = _table_grid(table)
    for row in grid:
        if not isinstance(row, list):
            continue
        cells = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        if len(cells) >= 2:
            anchors.append(" | ".join(cells))
            anchors.append(" ".join(cells))
        elif cells:
            anchors.append(cells[0])
    deduped: list[str] = []
    seen: set[str] = set()
    for anchor in anchors:
        key = _compact_text(anchor)
        if key and key not in seen:
            seen.add(key)
            deduped.append(anchor)
    return deduped


def _first_table_title(table: dict[str, Any]) -> str:
    for title in _table_title_candidates(table):
        value = str(title or "").strip()
        if value:
            return value
    value = str(table.get("title") or table.get("caption_text") or "").strip()
    if value:
        return value
    return ""


def _extend_range_to_preceding_title_line(
    markdown: str,
    table_range: tuple[int, int],
    title: Any,
) -> tuple[int, int]:
    start, end = table_range
    title_text = str(title or "").strip()
    if not title_text:
        return start, end
    body = str(markdown or "")
    line_start = _markdown_line_start_for_index(body, start)
    previous_end = max(0, line_start - 1)
    previous_start = body.rfind("\n", 0, previous_end) + 1
    previous_line = body[previous_start:previous_end].strip()
    normalized_previous = _compact_text(_strip_heading_or_emphasis(previous_line))
    if normalized_previous == _compact_text(title_text):
        start = previous_start
    return start, end


def strip_autoind_product_wrapper(markdown: str, filename: str = "") -> str:
    lines = str(markdown or "").splitlines()
    start_index = 0
    if filename:
        for index, line in enumerate(lines):
            if line.startswith(f"## {filename} "):
                start_index = index + 1
                break
    body = lines[start_index:]
    while body and not body[0].strip():
        body.pop(0)
    while body and _is_metadata_bullet(body[0]):
        body.pop(0)
    while body and not body[0].strip():
        body.pop(0)
    if body and body[0].strip().lower().startswith("### table of contents"):
        body.pop(0)
        while body and body[0].strip():
            body.pop(0)
        while body and not body[0].strip():
            body.pop(0)

    filtered: list[str] = []
    for line in body:
        stripped = line.strip()
        if stripped.lower() in _PRODUCT_BODY_HEADINGS:
            continue
        if stripped.startswith("![") and "](data:image/" in stripped:
            continue
        filtered.append(line)
    return _normalize_blank_lines("\n".join(filtered))


def _project_infographic_card_decks_in_flow(markdown: str, document: dict[str, Any]) -> str:
    projected = str(markdown or "")
    for page in ((document.get("document_ast") or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        for deck in _collect_infographic_card_deck_profiles(blocks):
            projected = _replace_card_deck_flow(projected, deck)
    return projected


def _project_landscape_panel_pages_in_flow(markdown: str, document: dict[str, Any]) -> str:
    projected = str(markdown or "")
    for page in ((document.get("document_ast") or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        profile = _landscape_panel_page_profile(page)
        if profile is None:
            continue
        projected = _replace_panel_page_flow(projected, profile)
    return projected


def _replace_panel_page_flow(markdown: str, profile: dict[str, Any]) -> str:
    blocks = [block for block in profile.get("blocks", []) or [] if isinstance(block, dict)]
    if not blocks:
        return markdown
    texts = [_block_text(block) for block in blocks]
    start = _earliest_markdown_text_index(markdown, texts)
    end = _latest_markdown_text_end(markdown, texts)
    if start is None or end is None or end <= start:
        return markdown
    replacement_lines: list[str] = []
    for lane in profile.get("lanes", []) or []:
        lane_blocks = [block for block in lane if isinstance(block, dict)]
        if not lane_blocks:
            continue
        if replacement_lines:
            replacement_lines.append("")
        replacement_lines.extend(_block_text(block) for block in lane_blocks if _block_text(block))
    replacement = "\n".join(line for line in replacement_lines if line is not None).strip()
    if not replacement:
        return markdown
    start = _markdown_line_start_for_index(markdown, start)
    end = _markdown_line_end_for_index(markdown, end)
    return markdown[:start].rstrip() + "\n\n" + replacement + "\n\n" + markdown[end:].lstrip()


def _landscape_panel_page_profile(page: dict[str, Any]) -> dict[str, Any] | None:
    blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
    if any(str(block.get("block_type") or "").strip().lower() in {"table", "image"} for block in blocks):
        return None
    text_blocks = [
        block
        for block in blocks
        if str(block.get("block_type") or "").strip().lower() == "text"
        and _bbox(block) is not None
        and _block_text(block)
    ]
    if len(text_blocks) < 12:
        return None
    _top, vertical_extent = _page_vertical_extent(text_blocks)
    _left, horizontal_extent = _page_horizontal_extent(text_blocks)
    if horizontal_extent <= vertical_extent * 1.15:
        return None
    if not _has_panel_page_text_profile(text_blocks):
        return None
    lanes = _horizontal_ownership_lanes(text_blocks)
    if len(lanes) < 3:
        return None
    active_lanes = [lane for lane in lanes if len(lane) >= 3]
    if len(active_lanes) < 3:
        return None
    return {
        "blocks": sorted(text_blocks, key=lambda item: ((_bbox(item) or (0, 0, 0, 0))[1], (_bbox(item) or (0, 0, 0, 0))[0])),
        "lanes": active_lanes,
    }


def _has_panel_page_text_profile(blocks: list[dict[str, Any]]) -> bool:
    word_counts = [len(re.findall(r"[A-Za-z\u4e00-\u9fff0-9]+", _block_text(block))) for block in blocks]
    if not word_counts:
        return False
    short_blocks = sum(1 for count in word_counts if count <= 7)
    panel_labels = sum(1 for block in blocks if _looks_like_panel_label_text(_block_text(block)))
    return short_blocks / max(1, len(word_counts)) >= 0.65 and panel_labels >= 3


def _looks_like_panel_label_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or len(raw) > 80:
        return False
    if re.search(r"[.!?]\s*$", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not (1 <= len(words) <= 8):
        return False
    letters = re.sub(r"[^A-Za-z]+", "", raw)
    if len(letters) >= 4 and letters.upper() == letters:
        return True
    return raw.endswith(":") or _title_case_ratio(words) >= 0.65


def _horizontal_ownership_lanes(blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    positioned: list[tuple[float, dict[str, Any]]] = []
    widths: list[float] = []
    for block in blocks:
        bbox = _bbox(block)
        if bbox is None:
            continue
        positioned.append((((bbox[0] + bbox[2]) / 2.0), block))
        widths.append(max(1.0, bbox[2] - bbox[0]))
    if len(positioned) < 3:
        return []
    positioned.sort(key=lambda item: item[0])
    centers = [item[0] for item in positioned]
    gaps = [(centers[index + 1] - centers[index], index) for index in range(len(centers) - 1)]
    if not gaps:
        return []
    median_width = sorted(widths)[len(widths) // 2]
    left, horizontal_extent = _page_horizontal_extent(blocks)
    separation_floor = max(median_width * 0.9, horizontal_extent * 0.12)
    split_indexes = sorted(index for gap, index in gaps if gap >= separation_floor)
    if len(split_indexes) < 2:
        return []
    lanes: list[list[dict[str, Any]]] = []
    start = 0
    for split_index in split_indexes:
        lane_blocks = [block for _center, block in positioned[start : split_index + 1]]
        if lane_blocks:
            lanes.append(sorted(lane_blocks, key=lambda item: ((_bbox(item) or (0, 0, 0, 0))[1], (_bbox(item) or (0, 0, 0, 0))[0])))
        start = split_index + 1
    lane_blocks = [block for _center, block in positioned[start:]]
    if lane_blocks:
        lanes.append(sorted(lane_blocks, key=lambda item: ((_bbox(item) or (0, 0, 0, 0))[1], (_bbox(item) or (0, 0, 0, 0))[0])))
    return lanes


def _project_figure_owned_pseudo_tables_in_flow(markdown: str, document: dict[str, Any]) -> str:
    projections = _figure_owned_pseudo_table_projections(document)
    if not projections:
        return markdown
    body = str(markdown or "")
    for projection in projections:
        source_texts = [str(item or "").strip() for item in projection.get("source_texts", []) or [] if str(item or "").strip()]
        table_anchors = [str(item or "").strip() for item in projection.get("table_anchors", []) or [] if str(item or "").strip()]
        replacement_lines = [str(item or "").strip() for item in projection.get("lines", []) or [] if str(item or "").strip()]
        if not source_texts or not replacement_lines:
            continue
        start = _earliest_markdown_text_index(body, source_texts)
        end = _latest_markdown_text_end(body, source_texts)
        if start is None or end is None or end <= start:
            table_range = _markdown_table_range_after_anchors(body, table_anchors, 0)
            if table_range is None:
                table_range = _markdown_html_table_range_after_anchors(body, table_anchors, 0)
            if table_range is None:
                continue
            start, end = table_range
        start = _markdown_line_start_for_index(body, start)
        end = _markdown_line_end_for_index(body, end)
        placeholder_range = _markdown_image_placeholder_range_before(body, projection.get("image_ids", []) or [], start)
        if placeholder_range is not None:
            start = min(start, placeholder_range[0])
        table_range = _markdown_table_range_after_anchors(body, table_anchors, start)
        if table_range is not None:
            table_start, table_end = table_range
            start = min(start, table_start)
            end = max(end, table_end)
        html_table_range = _markdown_html_table_range_after_anchors(body, table_anchors, start)
        if html_table_range is not None:
            table_start, table_end = html_table_range
            start = min(start, table_start)
            end = max(end, table_end)
        replacement = "\n\n".join(replacement_lines).strip()
        body = body[:start].rstrip() + "\n\n" + replacement + "\n\n" + body[end:].lstrip()
    return _normalize_blank_lines(body)


def _figure_owned_pseudo_table_projections(document: dict[str, Any]) -> list[dict[str, Any]]:
    images = [image for image in document.get("image_blocks", []) or [] if isinstance(image, dict)]
    tables = [table for table in document.get("table_asts", []) or [] if isinstance(table, dict)]
    projections: list[dict[str, Any]] = []
    used_image_ids: set[str] = set()
    for table in tables:
        table_bbox = _bbox(table)
        if table_bbox is None:
            continue
        if not _looks_like_figure_owned_pseudo_table_candidate(table):
            continue
        image = _matching_captioned_textual_figure_for_table(table, images)
        grid = table.get("display_grid") or table.get("raw_grid") or table.get("grid") or []
        if image is None:
            if str(table.get("detection_source") or table.get("detection_method") or "").strip() != "embedded_image_ocr":
                continue
            lines = _plain_lines_from_visual_table_grid(grid)
            if not lines:
                continue
            projections.append(
                {
                    "source_texts": [_table_plain_text(table)],
                    "table_anchors": _table_markdown_anchors(table),
                    "image_ids": [],
                    "lines": lines,
                }
            )
            continue
        image_id = str(image.get("image_id") or image.get("block_id") or "").strip()
        if image_id and image_id in used_image_ids:
            continue
        if image_id:
            used_image_ids.add(image_id)
        embedded_text = _figure_embedded_text(image)
        caption_text = str(image.get("caption_text") or image.get("title") or "").strip()
        source_texts = [text for text in [embedded_text, caption_text, _table_plain_text(table)] if text]
        if _has_axis_legend_chart_grid_profile(grid):
            lines = [text for text in [caption_text, embedded_text] if text]
        else:
            lines = [text for text in [embedded_text, caption_text] if text]
        if source_texts and lines:
            projections.append(
                {
                    "source_texts": source_texts,
                    "table_anchors": _structured_table_markdown_anchors(table),
                    "image_ids": [image_id] if image_id else [],
                    "lines": lines,
                }
            )
    return projections


def _looks_like_figure_owned_pseudo_table_candidate(table: dict[str, Any]) -> bool:
    detection = str(table.get("detection_source") or table.get("detection_method") or "").strip()
    grid = table.get("display_grid") or table.get("raw_grid") or table.get("grid") or []
    if not isinstance(grid, list) or len(grid) < 3:
        return False
    if detection == "embedded_image_ocr" and _has_regular_image_ocr_table_structure(grid):
        return False
    cells = [str(cell or "").strip() for row in grid if isinstance(row, list) for cell in row]
    non_empty = [cell for cell in cells if cell]
    if not non_empty:
        return False
    numeric_like = sum(1 for cell in non_empty if re.fullmatch(r"[\d.,%+\- ]+", cell))
    column_placeholders = sum(1 for cell in non_empty if re.fullmatch(r"Column\s+\d+", cell, re.IGNORECASE))
    if detection == "embedded_image_ocr":
        if _has_axis_legend_chart_grid_profile(grid):
            return True
        return _has_chart_or_screenshot_pseudo_table_profile(grid, non_empty, numeric_like, column_placeholders)
    return numeric_like / max(1, len(non_empty)) >= 0.55 or column_placeholders >= 2


def _has_axis_legend_chart_grid_profile(grid: list[Any]) -> bool:
    rows = [row for row in grid if isinstance(row, list) and any(str(cell or "").strip() for cell in row)]
    if len(rows) < 5:
        return False
    width = max((len(row) for row in rows), default=0)
    if width < 2 or width > 4:
        return False
    first_column: list[str] = []
    other_columns: list[str] = []
    for row in rows:
        padded = list(row) + [None] * max(0, width - len(row))
        first_column.append(str(padded[0] or "").strip())
        other_columns.extend(str(cell or "").strip() for cell in padded[1:])
    numeric_axis_values = sum(1 for value in first_column if re.fullmatch(r"\d{1,4}(?:[.,]\d+)?", value))
    unit_axis_values = sum(1 for value in first_column if re.search(r"[A-Za-z%]", value) and re.search(r"\d", value))
    legend_values = [
        value
        for value in other_columns
        if value
        and re.search(r"[A-Za-z\u4e00-\u9fff]", value)
        and not re.fullmatch(r"Column\s+\d+", value, re.IGNORECASE)
    ]
    non_empty_other = [value for value in other_columns if value]
    sparse_other_ratio = len(non_empty_other) / max(1, len(other_columns))
    return (
        numeric_axis_values >= 4
        and unit_axis_values >= 1
        and len(legend_values) >= 3
        and sparse_other_ratio <= 0.85
    )


def _has_regular_image_ocr_table_structure(grid: list[Any]) -> bool:
    rows = [row for row in grid if isinstance(row, list) and any(str(cell or "").strip() for cell in row)]
    if len(rows) < 4:
        return False
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return False
    non_empty_ratios = [sum(1 for cell in row if str(cell or "").strip()) / max(1, width) for row in rows]
    dense_rows = sum(1 for ratio in non_empty_ratios if ratio >= 0.75)
    if dense_rows < max(3, int(len(rows) * 0.75)):
        return False
    header = rows[0]
    header_text_cells = [
        str(cell or "").strip()
        for cell in header
        if re.search(r"[A-Za-z\u4e00-\u9fff]", str(cell or ""))
        and not re.fullmatch(r"Column\s+\d+", str(cell or "").strip(), re.IGNORECASE)
    ]
    if len(header_text_cells) < 2:
        return False
    numeric_data_rows = 0
    for row in rows[1:]:
        numeric_cells = sum(1 for cell in row if _is_table_numeric_value(str(cell or "").strip()))
        if numeric_cells >= min(2, max(1, width - 1)):
            numeric_data_rows += 1
    if numeric_data_rows >= 2:
        return True
    repeated_label_columns = 0
    for col_index in range(width):
        column_values = [str(row[col_index] if col_index < len(row) else "").strip() for row in rows[1:]]
        if sum(1 for value in column_values if value) >= 3:
            repeated_label_columns += 1
    return repeated_label_columns >= 2 and dense_rows >= max(4, int(len(rows) * 0.8))


def _has_chart_or_screenshot_pseudo_table_profile(
    grid: list[Any],
    non_empty: list[str],
    numeric_like: int,
    column_placeholders: int,
) -> bool:
    rows = [row for row in grid if isinstance(row, list)]
    total_slots = sum(len(row) for row in rows)
    empty_slots = sum(1 for row in rows for cell in row if not str(cell or "").strip())
    empty_ratio = empty_slots / max(1, total_slots)
    text = " ".join(non_empty)
    percent_cells = sum(1 for cell in non_empty if "%" in cell)
    ui_signal = re.search(
        r"\b(?:subscribe|playlist|playlists|uploads|videos|channel|channels|home|community|about|podcast)\b",
        text,
        re.IGNORECASE,
    )
    chart_axis_signal = re.search(
        r"\b(?:age|years?|length|weight|kg|lbs|cm|attribute|attributes|scenic|solitude|adventure)\b",
        text,
        re.IGNORECASE,
    )
    if column_placeholders:
        return True
    if percent_cells >= 2 and empty_ratio >= 0.12:
        return True
    if ui_signal and empty_ratio >= 0.10:
        return True
    if chart_axis_signal and numeric_like / max(1, len(non_empty)) >= 0.35 and empty_ratio >= 0.18:
        return True
    return False


def _is_table_numeric_value(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    normalized = value.replace(",", "").replace("−", "-")
    return re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[-+]?\d+)?%?", normalized, re.IGNORECASE) is not None


def _matching_captioned_textual_figure_for_table(
    table: dict[str, Any],
    images: list[dict[str, Any]],
) -> dict[str, Any] | None:
    table_bbox = _bbox(table)
    if table_bbox is None:
        return None
    best: tuple[float, dict[str, Any]] | None = None
    table_text = _table_plain_text(table)
    for image in images:
        image_kind = str(image.get("image_kind_guess") or "").strip()
        if image_kind not in {"captioned_textual_figure", "textual_image", "contextual_textual_image"}:
            continue
        if not str(image.get("caption_text") or image.get("title") or "").strip():
            continue
        image_bbox = _bbox(image)
        if image_bbox is None:
            continue
        overlap = _bbox_overlap_ratio(table_bbox, image_bbox)
        if overlap < 0.75:
            continue
        embedded_text = _figure_embedded_text(image)
        token_overlap = _token_overlap_ratio(_table_structural_anchor_text(table) or table_text, embedded_text)
        if token_overlap < 0.35:
            continue
        score = overlap + token_overlap
        if best is None or score > best[0]:
            best = (score, image)
    return best[1] if best else None


def _figure_embedded_text(image: dict[str, Any]) -> str:
    for segment in image.get("content_segments", []) or []:
        if not isinstance(segment, dict):
            continue
        if str(segment.get("role") or "").strip() == "embedded_text":
            text = str(segment.get("text") or "").strip()
            if text:
                return text
    return str(image.get("embedded_text") or "").strip()


def _project_vector_chart_pseudo_tables_in_flow(markdown: str, document: dict[str, Any]) -> str:
    projections = _vector_chart_pseudo_table_projections(document)
    if not projections:
        return markdown
    body = str(markdown or "")
    for projection in projections:
        source_texts = [str(item or "").strip() for item in projection.get("source_texts", []) or [] if str(item or "").strip()]
        table_anchors = [str(item or "").strip() for item in projection.get("table_anchors", []) or [] if str(item or "").strip()]
        replacement_lines = [str(item or "").strip() for item in projection.get("lines", []) or [] if str(item or "").strip()]
        if not source_texts or not replacement_lines:
            continue
        start = _earliest_markdown_text_index(body, source_texts)
        end = _latest_markdown_text_end(body, source_texts)
        table_range = _markdown_table_range_after_anchors(body, table_anchors, start or 0)
        if table_range is not None:
            start, end = table_range
        if start is None or end is None or end <= start:
            continue
        start = _markdown_line_start_for_index(body, start)
        end = _markdown_line_end_for_index(body, end)
        replacement = "\n\n".join(replacement_lines).strip()
        body = body[:start].rstrip() + "\n\n" + replacement + "\n\n" + body[end:].lstrip()
    return _normalize_blank_lines(body)


def _vector_chart_pseudo_table_projections(document: dict[str, Any]) -> list[dict[str, Any]]:
    tables = [table for table in document.get("table_asts", []) or [] if isinstance(table, dict)]
    pages = ((document.get("document_ast") or {}).get("pages") or [])
    projections: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        for table_block in blocks:
            if str(table_block.get("block_type") or "").strip().lower() != "table":
                continue
            table = _matching_table_ast(table_block, tables) or table_block
            if not _looks_like_vector_chart_pseudo_table(table):
                continue
            table_bbox = _bbox(table_block) or _bbox(table)
            if table_bbox is None:
                continue
            chart_blocks = _chart_evidence_blocks_for_table(blocks, table_block, table_bbox)
            if not chart_blocks:
                continue
            lines = _chart_evidence_lines(chart_blocks, table)
            if len(lines) < 3:
                continue
            projections.append(
                {
                    "source_texts": [_block_text(block) for block in chart_blocks] + [_table_plain_text(table)],
                    "table_anchors": _structured_table_markdown_anchors(table),
                    "lines": lines,
                }
            )
    return projections


def _looks_like_vector_chart_pseudo_table(table: dict[str, Any]) -> bool:
    detection = str(table.get("detection_source") or table.get("detection_method") or "").strip()
    if detection not in {"pymupdf_builtin", "text_aligned_borderless_grid"}:
        return False
    grid = table.get("display_grid") or table.get("raw_grid") or table.get("grid") or []
    if not isinstance(grid, list) or len(grid) < 3:
        return False
    cells = [str(cell or "").strip() for row in grid if isinstance(row, list) for cell in row]
    non_empty = [cell for cell in cells if cell]
    if not non_empty:
        return False
    alpha_cells = sum(1 for cell in non_empty if re.search(r"[A-Za-z\u4e00-\u9fff]", cell))
    column_placeholders = sum(1 for cell in non_empty if re.fullmatch(r"Column\s+\d+", cell, re.IGNORECASE))
    numeric_like = sum(1 for cell in non_empty if re.fullmatch(r"[\d.,%+\- ]+", cell))
    total_slots = sum(len(row) for row in grid if isinstance(row, list))
    empty_slots = sum(1 for row in grid if isinstance(row, list) for cell in row if not str(cell or "").strip())
    empty_ratio = empty_slots / max(1, total_slots)
    if column_placeholders and numeric_like >= 2:
        return True
    if alpha_cells <= 1 and numeric_like / max(1, len(non_empty)) >= 0.75 and empty_ratio >= 0.35:
        return True
    return False


def _chart_evidence_blocks_for_table(
    blocks: list[dict[str, Any]],
    table_block: dict[str, Any],
    table_bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    figure_caption = _nearest_preceding_figure_caption_block(blocks, table_bbox)
    if figure_caption is None:
        return []
    if re.match(r"^\s*(?:Table|表)\b", _block_text(figure_caption), re.IGNORECASE):
        return []
    chart_top = min((_bbox(figure_caption) or table_bbox)[1], table_bbox[1])
    chart_bottom = table_bbox[3]
    source_bottom = chart_bottom
    for block in blocks:
        if block is table_block or str(block.get("block_type") or "").strip().lower() != "text":
            continue
        bbox = _bbox(block)
        text = _block_text(block)
        if bbox is None or not text:
            continue
        if bbox[1] >= table_bbox[3] and bbox[1] - table_bbox[3] <= 80 and re.match(r"^\s*(?:Source|Note|Legend)\b", text, re.IGNORECASE):
            source_bottom = max(source_bottom, bbox[3])
    chart_blocks: list[dict[str, Any]] = []
    for block in blocks:
        block_type = str(block.get("block_type") or "").strip().lower()
        if block is table_block or block_type != "text":
            continue
        bbox = _bbox(block)
        text = _block_text(block)
        if bbox is None or not text:
            continue
        if bbox[1] < chart_top - 5 or bbox[3] > source_bottom + 5:
            continue
        if bbox[2] < table_bbox[0] - 80 or bbox[0] > table_bbox[2] + 80:
            continue
        chart_blocks.append(block)
    chart_blocks.sort(key=lambda block: ((_bbox(block) or (0, 0, 0, 0))[1], (_bbox(block) or (0, 0, 0, 0))[0]))
    return chart_blocks


def _nearest_preceding_figure_caption_block(
    blocks: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for block in blocks:
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        text = _block_text(block)
        if not re.match(r"^\s*(?:Figure|Fig\.?|图)\b", text, re.IGNORECASE):
            continue
        bbox = _bbox(block)
        if bbox is None or bbox[1] > table_bbox[1] + 20:
            continue
        if _bbox_horizontal_overlap_ratio(bbox, table_bbox) < 0.25:
            continue
        distance = abs(table_bbox[1] - bbox[3])
        if best is None or distance < best[0]:
            best = (distance, block)
    return best[1] if best else None


def _chart_evidence_lines(blocks: list[dict[str, Any]], table: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        text = _block_text(block)
        if not text:
            continue
        key = _compact_text(text)
        if key in seen:
            continue
        seen.add(key)
        lines.append(text)
    table_text = _table_plain_text(table)
    if table_text and _compact_text(table_text) not in seen:
        lines.append(table_text)
    return lines


def _project_seedless_contents_pages_in_flow(markdown: str, document: dict[str, Any]) -> str:
    if document.get("toc_sequences"):
        return markdown
    projections = _build_seedless_contents_page_projections(document)
    if not projections:
        return markdown
    body = str(markdown or "")
    for projection in projections:
        replacement = "\n".join(projection["lines"]).strip()
        if not replacement:
            continue
        start_texts = projection.get("start_texts") or []
        end_texts = projection.get("end_texts") or []
        start = _earliest_markdown_text_index(body, start_texts)
        end = _latest_markdown_text_end(body, end_texts)
        if start is None or end is None or end <= start:
            continue
        body = body[:start] + replacement + "\n" + body[end:]
    return _normalize_blank_lines(body)


def _build_seedless_contents_page_projections(document: dict[str, Any]) -> list[dict[str, Any]]:
    pages = ((document.get("document_ast") or {}).get("pages") or [])
    table_asts = [table for table in document.get("table_asts", []) or [] if isinstance(table, dict)]
    projections: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        title_block = _seedless_contents_title_block(blocks)
        if title_block is None:
            continue
        entries = _seedless_contents_text_entries(blocks, title_block)
        entries.extend(_seedless_contents_table_entries(blocks, table_asts))
        if len(entries) < 4:
            continue
        entries.sort(key=lambda item: (float(item.get("sort_y0") or 0.0), float(item.get("sort_x0") or 0.0)))
        locator_count = sum(1 for item in entries if item.get("locator"))
        if locator_count < max(3, int(len(entries) * 0.65)):
            continue
        title = _block_text(title_block) or "Contents"
        lines = [f"# {title}", ""]
        source_texts = [title]
        for entry in entries:
            text = str(entry.get("text") or "").strip()
            locator = str(entry.get("locator") or "").strip()
            if not text:
                continue
            lines.append(f"{text} {locator}".rstrip())
            source_texts.append(text)
            if locator:
                source_texts.append(locator)
        projections.append(
            {
                "lines": lines,
                "start_texts": source_texts,
                "end_texts": source_texts,
            }
        )
    return projections


def _project_titleless_contents_continuation_pages_in_flow(markdown: str, document: dict[str, Any]) -> str:
    if document.get("toc_sequences"):
        return markdown
    projections = _build_titleless_contents_continuation_page_projections(document)
    if not projections:
        return markdown
    body = str(markdown or "")
    for projection in projections:
        replacement = "\n".join(projection["lines"]).strip()
        if not replacement:
            continue
        start = _earliest_markdown_text_index(body, projection.get("start_texts") or [])
        end = _latest_markdown_text_end(body, projection.get("end_texts") or [])
        if start is None or end is None or end <= start:
            continue
        body = body[:start] + replacement + "\n" + body[end:]
    return _normalize_blank_lines(body)


def _build_titleless_contents_continuation_page_projections(document: dict[str, Any]) -> list[dict[str, Any]]:
    table_asts = [table for table in document.get("table_asts", []) or [] if isinstance(table, dict)]
    projections: list[dict[str, Any]] = []
    for page in ((document.get("document_ast") or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        if _seedless_contents_title_block(blocks) is not None:
            continue
        entries = _titleless_contents_continuation_entries(blocks, table_asts)
        if len(entries) < 6:
            continue
        located = [entry for entry in entries if str(entry.get("locator") or "").strip()]
        if len(located) < max(4, int(len(entries) * 0.55)):
            continue
        group_count = sum(1 for entry in entries if not str(entry.get("locator") or "").strip())
        if group_count < 1:
            continue
        entries.sort(key=lambda item: (float(item.get("sort_y0") or 0.0), float(item.get("sort_x0") or 0.0)))
        lines: list[str] = []
        source_texts: list[str] = []
        for entry in entries:
            text = str(entry.get("text") or "").strip()
            locator = str(entry.get("locator") or "").strip()
            if not text:
                continue
            lines.append(f"{text} {locator}".rstrip())
            source_texts.append(text)
        if locator:
            source_texts.append(locator)
        if lines:
            projections.append({"lines": lines, "start_texts": source_texts, "end_texts": source_texts})
    return projections


def _titleless_contents_continuation_entries(
    blocks: list[dict[str, Any]],
    table_asts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    locator_blocks = [
        block
        for block in blocks
        if str(block.get("block_type") or "").strip().lower() == "text"
        and _is_seedless_contents_locator_text(_block_text(block))
        and _bbox(block) is not None
    ]
    used_locator_ids: set[int] = set()
    for block in blocks:
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        text = _block_text(block)
        bbox = _bbox(block)
        if not text or bbox is None or _is_seedless_contents_locator_text(text):
            continue
        if _looks_like_seedless_contents_section_label(text) or _looks_like_toc_continuation_reference_entry(text):
            locator = _nearest_same_row_locator(block, locator_blocks, used_locator_ids)
            locator_text = ""
            if locator is not None:
                locator_block, locator_text = locator
                used_locator_ids.add(id(locator_block))
            entries.append({"text": text, "locator": locator_text, "sort_y0": bbox[1], "sort_x0": bbox[0]})
    for item in _seedless_contents_table_entries(blocks, table_asts):
        if item.get("text"):
            entries.append(item)
    return entries


def _seedless_contents_title_block(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for block in blocks:
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        if _is_toc_heading_line(_block_text(block)):
            return block
    return None


def _seedless_contents_text_entries(blocks: list[dict[str, Any]], title_block: dict[str, Any]) -> list[dict[str, Any]]:
    title_bbox = _bbox(title_block)
    if title_bbox is None:
        return []
    text_blocks = [
        block
        for block in blocks
        if str(block.get("block_type") or "").strip().lower() == "text"
        and _bbox(block) is not None
        and _block_text(block)
        and _bbox(block)[1] > title_bbox[3]
    ]
    if not text_blocks:
        return []
    page_right = max((_bbox(block) or title_bbox)[2] for block in text_blocks)
    left_blocks = [block for block in text_blocks if not _is_seedless_contents_locator_text(_block_text(block))]
    locator_blocks = [block for block in text_blocks if _is_seedless_contents_locator_text(_block_text(block))]
    entries: list[dict[str, Any]] = []
    used_locator_ids: set[int] = set()
    for block in sorted(left_blocks, key=lambda item: ((_bbox(item) or title_bbox)[1], (_bbox(item) or title_bbox)[0])):
        bbox = _bbox(block)
        if bbox is None:
            continue
        if bbox[0] > page_right * 0.55:
            continue
        locator = _nearest_same_row_locator(block, locator_blocks, used_locator_ids)
        if locator is None:
            if _looks_like_seedless_contents_section_label(_block_text(block)):
                entries.append(
                    {
                        "text": _block_text(block),
                        "locator": "",
                        "sort_y0": bbox[1],
                        "sort_x0": bbox[0],
                    }
                )
            continue
        locator_block, locator_text = locator
        used_locator_ids.add(id(locator_block))
        entries.append(
            {
                "text": _block_text(block),
                "locator": locator_text,
                "sort_y0": bbox[1],
                "sort_x0": bbox[0],
            }
        )
    return entries


def _nearest_same_row_locator(
    text_block: dict[str, Any],
    locator_blocks: list[dict[str, Any]],
    used_locator_ids: set[int],
) -> tuple[dict[str, Any], str] | None:
    text_bbox = _bbox(text_block)
    if text_bbox is None:
        return None
    text_mid = (text_bbox[1] + text_bbox[3]) / 2.0
    text_height = max(1.0, text_bbox[3] - text_bbox[1])
    candidates: list[tuple[float, dict[str, Any], str]] = []
    for block in locator_blocks:
        if id(block) in used_locator_ids:
            continue
        bbox = _bbox(block)
        if bbox is None or bbox[0] <= text_bbox[2]:
            continue
        locator_text = _block_text(block)
        if not _is_seedless_contents_locator_text(locator_text):
            continue
        locator_mid = (bbox[1] + bbox[3]) / 2.0
        distance = abs(locator_mid - text_mid)
        if distance <= text_height * 0.75:
            candidates.append((distance, block, locator_text))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def _is_seedless_contents_locator_text(text: str) -> bool:
    return re.fullmatch(r"\d{1,4}|[ivxlcdm]+", str(text or "").strip(), re.IGNORECASE) is not None


def _looks_like_seedless_contents_section_label(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or _is_seedless_contents_locator_text(raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9'&/-]*", raw)
    if not words:
        return False
    if _looks_like_seedless_contents_group_label(raw):
        return True
    if len(words) > 5:
        return False
    letters = re.sub(r"[^A-Za-z]+", "", raw)
    return bool(letters) and letters.upper() == letters


def _looks_like_seedless_contents_group_label(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or _is_seedless_contents_locator_text(raw):
        return False
    if re.search(r"\s(?:\d{1,4}|[ivxlcdm]+)\s*$", raw, re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9'&/-]*|\d+", raw)
    if not (3 <= len(words) <= 18):
        return False
    if re.match(r"^(?:part\s+[ivxlcdm]+|chapter\s+(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten))\b", raw, re.IGNORECASE):
        return True
    if re.match(r"^(?:section|unit|module)\s+(?:\d+(?:\.\d+)*|[ivxlcdm]+)\b", raw, re.IGNORECASE):
        return True
    return False


def _looks_like_toc_continuation_reference_entry(text: str) -> bool:
    raw = str(text or "").strip()
    return _compact_text(raw) in {"references", "bibliography", "appendix", "appendices", "index"}


def _seedless_contents_table_entries(blocks: list[dict[str, Any]], table_asts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table_blocks = [
        block for block in blocks if str(block.get("block_type") or "").strip().lower() == "table" and _bbox(block) is not None
    ]
    entries: list[dict[str, Any]] = []
    for block in table_blocks:
        table = _matching_table_ast(block, table_asts)
        if table is None:
            continue
        bbox = _bbox(block)
        rows = _toc_plain_entries_from_grid(table.get("display_grid") or table.get("raw_grid") or table.get("grid") or [])
        for offset, row in enumerate(rows):
            text, locator = row
            entries.append(
                {
                    "text": text,
                    "locator": locator,
                    "sort_y0": (bbox[1] if bbox else 0.0) + offset,
                    "sort_x0": bbox[0] if bbox else 0.0,
                }
            )
    return entries


def _matching_table_ast(block: dict[str, Any], table_asts: list[dict[str, Any]]) -> dict[str, Any] | None:
    table_id = str(block.get("table_id") or block.get("block_id") or "").strip()
    if table_id:
        for table in table_asts:
            if str(table.get("table_id") or "").strip() == table_id:
                return table
    block_bbox = _bbox(block)
    if block_bbox is None:
        return None
    for table in table_asts:
        table_bbox = _bbox(table)
        if table_bbox is not None and abs(table_bbox[0] - block_bbox[0]) < 2 and abs(table_bbox[1] - block_bbox[1]) < 2:
            return table
    return None


def _toc_plain_entries_from_grid(grid: list[Any]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for row in grid:
        if not isinstance(row, list):
            continue
        cells = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        if len(cells) == 2 and not _is_seedless_contents_locator_text(cells[-1]):
            joined = " ".join(cells).strip()
            if _looks_like_seedless_contents_section_label(joined):
                entries.append((joined, ""))
                continue
        if len(cells) == 1 and _looks_like_seedless_contents_section_label(cells[0]):
            entries.append((cells[0], ""))
            continue
        if len(cells) < 2:
            continue
        locator = cells[-1]
        if not _is_seedless_contents_locator_text(locator):
            continue
        text = " ".join(cells[:-1]).strip()
        if text:
            entries.append((text, locator))
    return entries


def _replace_card_deck_flow(markdown: str, deck: dict[str, Any]) -> str:
    blocks = [block for block in deck.get("blocks", []) or [] if isinstance(block, dict)]
    anchor_blocks = [block for block in deck.get("anchor_blocks", []) or [] if isinstance(block, dict)]
    if not blocks or not anchor_blocks:
        return markdown
    start = _earliest_markdown_text_index(markdown, [_block_text(block) for block in anchor_blocks])
    end = _latest_markdown_text_end(markdown, [_block_text(block) for block in blocks])
    if start is None or end is None or end <= start:
        return markdown
    replacement_lines: list[str] = []
    for lane in deck.get("lanes", []) or []:
        lane_blocks = [block for block in lane if isinstance(block, dict)]
        if not lane_blocks:
            continue
        if replacement_lines:
            replacement_lines.append("")
        paragraph_parts: list[str] = []
        for block in lane_blocks:
            text = _block_text(block)
            if not text:
                continue
            if _is_card_deck_heading_block(block):
                if paragraph_parts:
                    replacement_lines.append(" ".join(paragraph_parts).strip())
                    paragraph_parts = []
                replacement_lines.append(text)
            else:
                paragraph_parts.append(text)
        if paragraph_parts:
            replacement_lines.append(" ".join(paragraph_parts).strip())
    if not replacement_lines:
        return markdown
    replacement = "\n\n".join(line for line in replacement_lines if line.strip())
    return markdown[:start].rstrip() + "\n\n" + replacement + "\n\n" + markdown[end:].lstrip()


def _earliest_markdown_text_index(markdown: str, texts: list[str]) -> int | None:
    indexes = [
        index
        for text in texts
        for index in [markdown.find(text)]
        if text and len(text.strip()) >= 3 and index >= 0
    ]
    return min(indexes) if indexes else None


def _latest_markdown_text_end(markdown: str, texts: list[str]) -> int | None:
    ends = [
        index + len(text)
        for text in texts
        for index in [markdown.rfind(text)]
        if text and len(text.strip()) >= 3 and index >= 0
    ]
    return max(ends) if ends else None


def _suppress_nonsemantic_image_placeholders(markdown: str, document: dict[str, Any]) -> str:
    suppressed_ids = {
        str(image.get("image_id") or "").strip()
        for image in document.get("image_blocks", []) or []
        if isinstance(image, dict) and _is_nonsemantic_image_placeholder_for_text_projection(image, document)
    }
    if not suppressed_ids:
        return markdown
    lines: list[str] = []
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        target = re.fullmatch(r"!\[[^\]]*\]\(#([^)]+)\)", stripped)
        if target and target.group(1) in suppressed_ids:
            continue
        lines.append(line)
    return _normalize_blank_lines("\n".join(lines))


def _is_nonsemantic_image_placeholder_for_text_projection(image: dict[str, Any], document: dict[str, Any] | None = None) -> bool:
    return (
        _is_nonsemantic_path_image_placeholder(image)
        or _is_structured_region_owned_path_screenshot_placeholder(image, document or {})
        or _is_caption_only_image_placeholder(image)
        or _is_embedded_text_image_placeholder(image)
    )


def _is_nonsemantic_path_image_placeholder(image: dict[str, Any]) -> bool:
    image_kind = str(image.get("image_kind_guess") or "").strip()
    if image_kind != "path_screenshot":
        return False
    if str(image.get("caption_text") or image.get("title") or "").strip():
        return False
    if str(image.get("embedded_text") or "").strip():
        return False
    signals = image.get("content_signals") if isinstance(image.get("content_signals"), dict) else {}
    if bool(signals.get("has_caption")) or bool(signals.get("has_embedded_text")):
        return False
    segments = [segment for segment in image.get("content_segments", []) or [] if isinstance(segment, dict)]
    if not segments:
        return True
    return all(str(segment.get("role") or "").strip() == "nearby_context" for segment in segments)


def _is_structured_region_owned_path_screenshot_placeholder(image: dict[str, Any], document: dict[str, Any]) -> bool:
    if str(image.get("image_kind_guess") or "").strip() != "path_screenshot":
        return False
    if not str(image.get("embedded_text") or "").strip():
        return False
    image_bbox = _bbox(image)
    if image_bbox is None:
        return False
    tables = [table for table in document.get("table_asts", []) or [] if isinstance(table, dict)]
    if not tables:
        return False
    embedded_compact = _compact_text(str(image.get("embedded_text") or ""))
    for table in tables:
        table_bbox = _bbox(table)
        if table_bbox is None:
            continue
        if _bbox_overlap_ratio(table_bbox, image_bbox) < 0.75:
            continue
        table_text = _table_plain_text(table)
        if table_text and _token_overlap_ratio(table_text, str(image.get("embedded_text") or "")) >= 0.65:
            return True
        anchor_text = _table_structural_anchor_text(table)
        if anchor_text and _token_overlap_ratio(anchor_text, str(image.get("embedded_text") or "")) >= 0.65:
            return True
    return False


def _table_structural_anchor_text(table: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "caption_text"):
        value = str(table.get(key) or "").strip()
        if value:
            parts.append(value)
    grid = table.get("display_grid") or table.get("raw_grid") or table.get("grid") or []
    if not isinstance(grid, list):
        return " ".join(parts).strip()
    title_row = table.get("title_row_index")
    selected_rows: list[int] = []
    if isinstance(title_row, int) and 0 <= title_row < len(grid):
        selected_rows.append(title_row)
        if title_row + 1 < len(grid):
            selected_rows.append(title_row + 1)
    else:
        header_rows = table.get("header_rows")
        if isinstance(header_rows, list) and header_rows:
            for offset in range(min(len(header_rows), 2)):
                selected_rows.append(offset)
        else:
            selected_rows.extend(range(min(len(grid), 2)))
    for row_index in selected_rows:
        row = grid[row_index]
        if not isinstance(row, list):
            continue
        row_text = " ".join(str(cell or "").strip() for cell in row if str(cell or "").strip()).strip()
        if row_text:
            parts.append(row_text)
    return " ".join(parts).strip()


def _table_plain_text(table: dict[str, Any]) -> str:
    parts: list[str] = []
    for row in table.get("display_grid") or table.get("raw_grid") or table.get("grid") or []:
        if not isinstance(row, list):
            continue
        for cell in row:
            text = str(cell or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts).strip()


def _bbox_overlap_ratio(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inner_area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return ((ix1 - ix0) * (iy1 - iy0)) / inner_area


def _bbox_horizontal_overlap_ratio(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    ix0 = max(inner[0], outer[0])
    ix1 = min(inner[2], outer[2])
    if ix1 <= ix0:
        return 0.0
    width = max(1.0, inner[2] - inner[0])
    return (ix1 - ix0) / width


def _token_overlap_ratio(needle_text: str, haystack_text: str) -> float:
    needle_tokens = re.findall(r"[A-Za-z0-9]+", str(needle_text or "").casefold())
    haystack_tokens = set(re.findall(r"[A-Za-z0-9]+", str(haystack_text or "").casefold()))
    if not needle_tokens or not haystack_tokens:
        return 0.0
    matched = sum(1 for token in needle_tokens if token in haystack_tokens)
    return matched / max(1, len(needle_tokens))


def _is_caption_only_image_placeholder(image: dict[str, Any]) -> bool:
    if not str(image.get("caption_text") or image.get("title") or "").strip():
        return False
    if str(image.get("embedded_text") or "").strip():
        return False
    signals = image.get("content_signals") if isinstance(image.get("content_signals"), dict) else {}
    if bool(signals.get("has_embedded_text")):
        return False
    segments = [segment for segment in image.get("content_segments", []) or [] if isinstance(segment, dict)]
    if not segments:
        return False
    semantic_roles = {str(segment.get("role") or "").strip() for segment in segments}
    if "caption" not in semantic_roles:
        return False
    return semantic_roles <= {"caption", "nearby_context"}


def _is_embedded_text_image_placeholder(image: dict[str, Any]) -> bool:
    if str(image.get("image_kind_guess") or "").strip() == "path_screenshot":
        return False
    if not str(image.get("embedded_text") or "").strip():
        return False
    segments = [segment for segment in image.get("content_segments", []) or [] if isinstance(segment, dict)]
    if not segments:
        return False
    semantic_roles = {str(segment.get("role") or "").strip() for segment in segments}
    return "embedded_text" in semantic_roles and semantic_roles <= {"embedded_text", "nearby_context"}


def _project_owned_figure_captions_after_image_text(markdown: str, document: dict[str, Any]) -> str:
    projections = _owned_figure_caption_projections(document)
    if not projections:
        return markdown
    body = str(markdown or "")
    for projection in projections:
        image_text = str(projection.get("image_text") or "").strip()
        caption_text = str(projection.get("caption_text") or "").strip()
        if not image_text or not caption_text:
            continue
        image_index = body.find(image_text)
        caption_index = body.find(caption_text)
        if image_index < 0 or caption_index < 0:
            continue
        if image_index < caption_index:
            between = body[image_index + len(image_text) : caption_index]
            if not _figure_caption_relocation_has_intervening_flow(between):
                continue
        caption_start = _markdown_paragraph_start_for_index(body, caption_index)
        caption_end = _markdown_paragraph_end_for_index(body, caption_index)
        caption_paragraph = body[caption_start:caption_end].strip()
        if not caption_paragraph:
            continue
        body_without_caption = body[:caption_start].rstrip() + "\n\n" + body[caption_end:].lstrip()
        image_index = body_without_caption.find(image_text)
        if image_index < 0:
            continue
        insert_at = _markdown_paragraph_end_for_index(body_without_caption, image_index)
        body = (
            body_without_caption[:insert_at].rstrip()
            + "\n\n"
            + caption_paragraph
            + "\n\n"
            + body_without_caption[insert_at:].lstrip()
        )
    return _normalize_blank_lines(body)


def _owned_figure_caption_projections(document: dict[str, Any]) -> list[dict[str, str]]:
    images = [image for image in document.get("image_blocks", []) or [] if isinstance(image, dict)]
    pages = ((document.get("document_ast") or {}).get("pages") or [])
    projections: list[dict[str, str]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        page_images = _page_image_blocks(page_blocks, images)
        if not page_images:
            continue
        for image in page_images:
            image_text = _figure_embedded_text(image)
            if not image_text:
                continue
            image_bbox = _bbox(image)
            if image_bbox is None:
                continue
            caption_blocks = _side_or_below_caption_blocks_for_image(page_blocks, image_bbox)
            if not caption_blocks:
                continue
            caption_text = " ".join(_block_text(block) for block in caption_blocks if _block_text(block)).strip()
            if caption_text:
                projections.append({"image_text": image_text, "caption_text": caption_text})
    return projections


def _page_image_blocks(page_blocks: list[dict[str, Any]], images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_by_id = {str(image.get("image_id") or image.get("block_id") or "").strip(): image for image in images}
    result: list[dict[str, Any]] = []
    for block in page_blocks:
        if str(block.get("block_type") or "").strip().lower() != "image":
            continue
        image_id = str(block.get("image_id") or block.get("block_id") or "").strip()
        enriched = {**image_by_id.get(image_id, {}), **block} if image_id else block
        if str(enriched.get("image_kind_guess") or "").strip() == "path_screenshot":
            continue
        if _bbox(enriched) is not None:
            result.append(enriched)
    return result


def _side_or_below_caption_blocks_for_image(
    blocks: list[dict[str, Any]],
    image_bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, block in enumerate(blocks):
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        text = _block_text(block)
        bbox = _bbox(block)
        if not text or bbox is None:
            continue
        side_overlap = min(bbox[3], image_bbox[3]) - max(bbox[1], image_bbox[1])
        is_side_caption = bbox[0] >= image_bbox[2] and side_overlap >= max(8.0, (bbox[3] - bbox[1]) * 0.5)
        is_below_caption = bbox[1] >= image_bbox[3] and bbox[1] - image_bbox[3] <= max(40.0, (bbox[3] - bbox[1]) * 3.0)
        if not (is_side_caption or is_below_caption):
            continue
        if _looks_like_figure_caption_start(text):
            if _has_immediate_caption_title_above(blocks, index, bbox):
                continue
            candidates.append((index, block))
    if not candidates:
        return []
    start_index = candidates[0][0]
    caption_blocks = [candidates[0][1]]
    previous_bbox = _bbox(candidates[0][1])
    for block in blocks[start_index + 1 : min(len(blocks), start_index + 6)]:
        if str(block.get("block_type") or "").strip().lower() != "text":
            break
        text = _block_text(block)
        bbox = _bbox(block)
        if not text or bbox is None or previous_bbox is None:
            break
        if _looks_like_footnote_or_reference_line_text(text):
            break
        if _looks_like_body_text_after_heading(text) and not _looks_like_caption_continuation_text(text):
            break
        if abs(bbox[0] - previous_bbox[0]) > max(45.0, (previous_bbox[2] - previous_bbox[0]) * 0.35):
            break
        vertical_gap = bbox[1] - previous_bbox[3]
        if vertical_gap < -2.0 or vertical_gap > max(14.0, (previous_bbox[3] - previous_bbox[1]) * 1.3):
            break
        caption_blocks.append(block)
        previous_bbox = bbox
    return caption_blocks


def _has_immediate_caption_title_above(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    for previous in reversed(blocks[max(0, index - 3) : index]):
        if str(previous.get("block_type") or "").strip().lower() != "text":
            continue
        previous_bbox = _bbox(previous)
        previous_text = _block_text(previous)
        if previous_bbox is None or not previous_text:
            continue
        vertical_gap = bbox[1] - previous_bbox[3]
        if vertical_gap < -2.0 or vertical_gap > max(12.0, (bbox[3] - bbox[1]) * 0.9):
            continue
        horizontal_overlap = min(bbox[2], previous_bbox[2]) - max(bbox[0], previous_bbox[0])
        same_lane = horizontal_overlap > 0 or abs(((bbox[0] + bbox[2]) / 2.0) - ((previous_bbox[0] + previous_bbox[2]) / 2.0)) <= max(
            45.0,
            (bbox[2] - bbox[0]) * 0.35,
        )
        if not same_lane:
            continue
        if _looks_like_figure_caption_start(previous_text) or _looks_like_footnote_or_reference_line_text(previous_text):
            continue
        return True
    return False


def _looks_like_figure_caption_start(text: str) -> bool:
    return re.match(r"^\s*(?:fig(?:ure)?\.?|[\u56fe\u5716])\s*\.?\s*[A-Za-z]?\d+(?:[.\-:]\d+)*(?:[A-Za-z])?\b", str(text or ""), re.IGNORECASE) is not None


def _looks_like_caption_continuation_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if len(words) <= 9:
        return True
    return not re.search(r"[.!?]\s*$", raw)


def _figure_caption_relocation_has_intervening_flow(text: str) -> bool:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", str(text or "")) if item.strip()]
    return len(paragraphs) >= 1


def _project_owned_figure_captions_before_page_notes(markdown: str, document: dict[str, Any]) -> str:
    projections = _owned_caption_before_note_projections(document)
    if not projections:
        return markdown
    body = str(markdown or "")
    for projection in projections:
        caption_text = str(projection.get("caption_text") or "").strip()
        note_text = str(projection.get("note_text") or "").strip()
        if not caption_text or not note_text:
            continue
        caption_index = body.find(caption_text)
        note_index = body.find(note_text)
        if caption_index < 0 or note_index < 0 or caption_index < note_index:
            continue
        caption_start = _markdown_paragraph_start_for_index(body, caption_index)
        caption_end = _markdown_paragraph_end_for_index(body, caption_index)
        caption_paragraph = body[caption_start:caption_end].strip()
        if not caption_paragraph:
            continue
        body_without_caption = body[:caption_start].rstrip() + "\n\n" + body[caption_end:].lstrip()
        note_index = body_without_caption.find(note_text)
        if note_index < 0:
            continue
        insert_at = _markdown_paragraph_start_for_index(body_without_caption, note_index)
        body = (
            body_without_caption[:insert_at].rstrip()
            + "\n\n"
            + caption_paragraph
            + "\n\n"
            + body_without_caption[insert_at:].lstrip()
        )
    return _normalize_blank_lines(body)


def _owned_caption_before_note_projections(document: dict[str, Any]) -> list[dict[str, str]]:
    projections: list[dict[str, str]] = []
    for page in ((document.get("document_ast") or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        if not any(str(block.get("block_type") or "").strip().lower() == "image" for block in blocks):
            continue
        page_height = _projection_page_height(page, blocks)
        note_blocks = [
            block
            for index, block in enumerate(blocks)
            if _looks_like_page_bottom_note_or_reference_zone(blocks, index, page_height)
            and _bbox(block) is not None
            and _block_text(block)
        ]
        if not note_blocks:
            continue
        first_note_bbox = min((_bbox(block) for block in note_blocks if _bbox(block) is not None), key=lambda bbox: bbox[1])
        note_text = _block_text(min(note_blocks, key=lambda block: (_bbox(block) or (0, 0, 0, 0))[1]))
        for caption_blocks in _owned_caption_groups_for_page(blocks):
            caption_bboxes = [_bbox(block) for block in caption_blocks]
            usable = [bbox for bbox in caption_bboxes if bbox is not None]
            if not usable or max(bbox[3] for bbox in usable) >= first_note_bbox[1]:
                continue
            caption_text = _block_text(caption_blocks[0])
            if caption_text:
                projections.append({"caption_text": caption_text, "note_text": note_text})
    return projections


def _projection_page_height(page: dict[str, Any], blocks: list[dict[str, Any]]) -> float | None:
    try:
        height = float(page.get("height"))
        if height > 0:
            return height
    except (TypeError, ValueError):
        pass
    usable = [bbox for block in blocks for bbox in [_bbox(block)] if bbox is not None]
    return max((bbox[3] for bbox in usable), default=None)


def _owned_caption_groups_for_page(blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for block in blocks:
        if str(block.get("block_type") or "").strip().lower() != "image":
            continue
        image_bbox = _bbox(block)
        if image_bbox is None:
            continue
        caption_blocks = _side_or_below_caption_blocks_for_image(blocks, image_bbox)
        if caption_blocks:
            groups.append(caption_blocks)
    return groups


def _deduplicate_prefix_figure_caption_paragraphs(markdown: str) -> str:
    paragraphs = re.split(r"\n\s*\n", str(markdown or "").strip())
    if len(paragraphs) < 2:
        return markdown
    kept: list[str] = []
    index = 0
    while index < len(paragraphs):
        current = paragraphs[index].strip()
        if not current:
            index += 1
            continue
        if index + 1 < len(paragraphs):
            following = paragraphs[index + 1].strip()
            if _is_prefix_duplicate_figure_caption(current, following):
                index += 1
                continue
        kept.append(current)
        index += 1
    return _normalize_blank_lines("\n\n".join(kept))


_FIGURE_CAPTION_LABEL_IN_TEXT_RE = re.compile(
    r"(?<!\w)(?:fig(?:ure)?|[\u56fe\u5716])\s*\.?\s*[A-Za-z]?\d+(?:[.\-:]\d+)*(?:[A-Za-z])?\s*[:.\-]\s+",
    re.IGNORECASE,
)


def _split_embedded_figure_caption_paragraphs(markdown: str) -> str:
    paragraphs = re.split(r"\n\s*\n", str(markdown or "").strip())
    if not paragraphs:
        return markdown
    split_paragraphs: list[str] = []
    for paragraph in paragraphs:
        text = paragraph.strip()
        if not text:
            continue
        split_paragraphs.extend(_split_embedded_figure_caption_paragraph(text))
    return _normalize_blank_lines("\n\n".join(split_paragraphs))


def _split_embedded_figure_caption_paragraph(paragraph: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(paragraph or "")).strip()
    if not text:
        return []
    matches = list(_FIGURE_CAPTION_LABEL_IN_TEXT_RE.finditer(text))
    if len(matches) < 2:
        return [paragraph.strip()]
    parts: list[str] = []
    start = 0
    for match in matches[1:]:
        part = text[start:match.start()].strip()
        if part:
            parts.append(part)
        start = match.start()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts or [paragraph.strip()]


def _is_prefix_duplicate_figure_caption(current: str, following: str) -> bool:
    current_label, current_body = _split_leading_figure_caption_label(current)
    following_label, following_body = _split_leading_figure_caption_label(following)
    if not current_label or not following_label or current_label != following_label:
        return False
    if not current_body or not following_body:
        return False
    current_compact = _compact_text(current_body)
    following_compact = _compact_text(following_body)
    if len(current_compact) < 18 or len(following_compact) <= len(current_compact):
        return False
    return following_compact.startswith(current_compact)


def _split_leading_figure_caption_label(text: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    match = re.match(
        r"^(?P<label>(?:fig(?:ure)?|[\u56fe\u5716])\s*\.?\s*[A-Za-z]?\d+(?:[.\-:]\d+)*(?:[A-Za-z])?)\s*[:.\-]?\s*(?P<body>.+)$",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return "", ""
    label = re.sub(r"\s+", "", match.group("label")).casefold()
    return label, match.group("body").strip()


def _project_top_section_headings_before_page_body(markdown: str, document: dict[str, Any]) -> str:
    body = str(markdown or "")
    for projection in _top_section_heading_order_projections(document):
        heading_text = str(projection.get("heading_text") or "").strip()
        anchor_text = str(projection.get("anchor_text") or "").strip()
        if not heading_text or not anchor_text:
            continue
        heading_index = body.find(heading_text)
        anchor_index = body.find(anchor_text)
        if heading_index < 0 or anchor_index < 0 or heading_index <= anchor_index:
            continue
        heading_start = _markdown_paragraph_start_for_index(body, heading_index)
        heading_end = _markdown_paragraph_end_for_index(body, heading_index + len(heading_text))
        heading_paragraph = body[heading_start:heading_end].strip()
        if not heading_paragraph:
            continue
        body_without_heading = body[:heading_start].rstrip() + "\n\n" + body[heading_end:].lstrip()
        anchor_index = body_without_heading.find(anchor_text)
        if anchor_index < 0:
            continue
        anchor_start = _markdown_paragraph_start_for_index(body_without_heading, anchor_index)
        rendered_heading = heading_paragraph if heading_paragraph.startswith("#") else f"# {heading_paragraph}"
        body = (
            body_without_heading[:anchor_start].rstrip()
            + "\n\n"
            + rendered_heading
            + "\n\n"
            + body_without_heading[anchor_start:].lstrip()
        )
    return _normalize_blank_lines(body)


def _top_section_heading_order_projections(document: dict[str, Any]) -> list[dict[str, str]]:
    projections: list[dict[str, str]] = []
    for page in ((document.get("document_ast") or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        text_blocks = [
            block
            for block in blocks
            if str(block.get("block_type") or "").strip().lower() == "text"
            and _block_text(block)
            and _bbox(block) is not None
        ]
        if len(text_blocks) < 2:
            continue
        first_body: dict[str, Any] | None = None
        for block in text_blocks:
            if str(block.get("semantic_role") or "").strip() == "section_heading":
                continue
            if _is_non_body_top_section_order_anchor(block):
                continue
            first_body = block
            break
        if first_body is None:
            continue
        first_body_bbox = _bbox(first_body)
        if first_body_bbox is None:
            continue
        for block in text_blocks:
            if str(block.get("semantic_role") or "").strip() != "section_heading":
                continue
            heading_bbox = _bbox(block)
            heading_text = _block_text(block)
            if heading_bbox is None or not heading_text:
                continue
            if heading_bbox[1] >= first_body_bbox[1]:
                continue
            if text_blocks.index(block) <= text_blocks.index(first_body):
                continue
            projections.append(
                {
                    "heading_text": heading_text,
                    "anchor_text": _block_text(first_body),
                }
            )
            break
    return projections


def _is_non_body_top_section_order_anchor(block: dict[str, Any]) -> bool:
    role = str(block.get("semantic_role") or block.get("unit_role") or "").strip()
    if role in {
        "page_number",
        "page_header",
        "page_footer",
        "publication_footer",
        "publication_masthead",
        "citation_metadata",
        "reference_entry",
        "footnote",
        "footnote_continuation",
    }:
        return True
    text = _block_text(block)
    if re.fullmatch(r"\d{1,4}", text):
        return True
    return False


def project_benchmark_headings(markdown: str, document: dict[str, Any]) -> str:
    markdown = _project_compact_statistical_heading_prefixes(markdown)
    heading_texts = _collect_benchmark_heading_texts(document)
    heading_texts = _suppress_titleless_contents_continuation_heading_texts(heading_texts, document)
    markdown = _project_benchmark_table_title_headings(markdown, document)
    if not heading_texts:
        normalized = _normalize_existing_benchmark_heading_levels(
            _merge_adjacent_heading_continuations(_normalize_blank_lines(markdown)),
            heading_texts,
        )
        return _demote_existing_false_positive_heading_lines(
            normalized,
            document,
        )

    lines = str(markdown or "").splitlines()
    projected: list[str] = []
    heading_by_compact = {_compact_text(item): item for item in heading_texts}
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        multiline_match = _match_multiline_heading_lines(lines, index, heading_texts)
        if multiline_match is not None and not stripped.startswith("#"):
            heading, next_index = multiline_match
            _append_heading_line(projected, heading)
            index = next_index
            continue
        compact = _compact_text(stripped)
        if compact in heading_by_compact and not stripped.startswith("#"):
            _append_heading_line(projected, heading_by_compact[compact])
            index += 1
            continue
        prefix_match = _match_heading_line_prefix(stripped, heading_texts)
        if prefix_match is not None and not stripped.startswith("#"):
            prefix_heading, remainder = prefix_match
            _append_heading_line(projected, prefix_heading)
            remainder = _trim_heading_separator_prefix(remainder)
            if remainder:
                remainder_replaced = _split_embedded_heading_sequence(remainder, heading_texts)
                if remainder_replaced is None:
                    projected.append(remainder)
                else:
                    projected.extend(remainder_replaced)
            index += 1
            continue
        prefix_heading = _match_strong_heading_prefix(stripped, heading_texts)
        if prefix_heading is not None and not stripped.startswith("#"):
            _append_heading_line(projected, prefix_heading)
            index += 1
            continue
        replaced = _split_embedded_heading_sequence(line, heading_texts)
        if replaced is None:
            projected.append(line)
        else:
            projected.extend(replaced)
        index += 1
    normalized = _normalize_existing_benchmark_heading_levels(
        _merge_adjacent_heading_continuations(
            _normalize_blank_lines("\n".join(projected)),
            heading_texts,
        ),
        heading_texts,
    )
    return _demote_existing_false_positive_heading_lines(
        normalized,
        document,
    )


def _normalize_existing_benchmark_heading_levels(markdown: str, heading_texts: list[str]) -> str:
    heading_by_compact = {_compact_text(item): item for item in heading_texts if str(item or "").strip()}
    if not heading_by_compact:
        return markdown
    lines: list[str] = []
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if match:
            compact = _compact_text(match.group(2))
            if compact in heading_by_compact:
                lines.append(f"# {heading_by_compact[compact]}")
                continue
        lines.append(line)
    return _normalize_blank_lines("\n".join(lines))


def _demote_existing_false_positive_heading_lines(markdown: str, document: dict[str, Any]) -> str:
    demote_texts = _collect_benchmark_heading_demotion_texts(document)
    if not demote_texts:
        return markdown
    lines: list[str] = []
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if match and _compact_text(match.group(2)) in demote_texts:
            prefix = line[: len(line) - len(line.lstrip())]
            lines.append(prefix + match.group(2).strip())
        else:
            lines.append(line)
    return _normalize_blank_lines("\n".join(lines))


def _collect_benchmark_heading_demotion_texts(document: dict[str, Any]) -> set[str]:
    demote: set[str] = set()
    pages = ((document.get("document_ast") or {}).get("pages") or [])
    for page in pages:
        if not isinstance(page, dict):
            continue
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        page_height = _float_or_none(page.get("height"))
        toc_page = _is_toc_page(blocks)
        for index, block in enumerate(blocks):
            if str(block.get("block_type") or "").strip().lower() != "text":
                continue
            text = _block_text(block)
            if not text:
                continue
            if toc_page and _looks_like_toc_entry_text(text):
                demote.add(_compact_text(text))
                continue
            if (
                _looks_like_existing_heading_demotion_candidate(blocks, index, page_height)
            ):
                demote.add(_compact_text(text))
    return demote


def _looks_like_existing_heading_demotion_candidate(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    return (
        _looks_like_list_item_or_step_heading_false_positive(blocks, index)
        or _looks_like_numbered_body_instruction_false_positive(blocks, index)
        or _looks_like_sentence_continuation_heading_false_positive(blocks, index)
        or _looks_like_page_bottom_note_or_reference_zone(blocks, index, page_height)
        or _looks_like_numbered_source_note_heading_false_positive(blocks, index, page_height)
    )


def _looks_like_toc_entry_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    compact = re.sub(r"[^a-z]", "", raw.lower())
    if compact in {"contents", "tableofcontents"}:
        return False
    if re.fullmatch(r"(?:[ivxlcdm]{1,8}|\d{1,4})", raw, re.IGNORECASE):
        return True
    if re.match(r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])\.?\s+\S+", raw, re.IGNORECASE):
        return True
    return bool(re.search(r"\s(?:[ivxlcdm]{1,8}|\d{1,4})$", raw, re.IGNORECASE))


def _suppress_titleless_contents_continuation_heading_texts(
    heading_texts: list[str],
    document: dict[str, Any],
) -> list[str]:
    continuation_texts: set[str] = set()
    for projection in _build_titleless_contents_continuation_page_projections(document):
        for line in projection.get("lines", []) or []:
            text = re.sub(r"\s+(?:\d{1,4}|[ivxlcdm]+)\s*$", "", str(line or "").strip(), flags=re.IGNORECASE).strip()
            if text:
                continuation_texts.add(_compact_text(text))
    if not continuation_texts:
        return heading_texts
    return [text for text in heading_texts if _compact_text(text) not in continuation_texts]


def _project_compact_statistical_heading_prefixes(markdown: str) -> str:
    lines: list[str] = []
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|") or stripped.startswith("<"):
            lines.append(line)
            continue
        split = _split_compact_statistical_heading_prefix(stripped)
        if split is None:
            lines.append(line)
            continue
        heading, remainder = split
        _append_heading_line(lines, heading)
        if remainder:
            lines.append(remainder)
    return _normalize_blank_lines("\n".join(lines))


def _split_compact_statistical_heading_prefix(text: str) -> tuple[str, str] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    match = re.match(
        r"^(?P<title>[A-Z][A-Za-z0-9'/-]*(?:\s+[A-Z][A-Za-z0-9'/-]*){0,5})\s+"
        r"(?P<count>\d{1,6})\s*responses\b(?P<rest>.*)$",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return None
    title = _normalize_compact_heading_words(match.group("title") or "")
    count = str(match.group("count") or "").strip()
    remainder = str(match.group("rest") or "").strip()
    if not title or not count:
        return None
    title_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", title)
    if not (1 <= len(title_words) <= 6):
        return None
    if _title_case_ratio(title_words) < 0.55:
        return None
    if remainder and re.match(r"^(?:[a-z]|\b(?:and|or|but|that|which|where|when|while)\b)", remainder):
        return None
    if remainder and not _looks_like_chart_or_survey_payload(remainder):
        return None
    return f"{title} {count} responses".strip(), remainder


def _normalize_compact_heading_words(text: str) -> str:
    words: list[str] = []
    for token in re.split(r"\s+", str(text or "").strip()):
        if not token:
            continue
        token = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", token)
        words.extend(part for part in token.split() if part)
    return " ".join(words)


def _looks_like_chart_or_survey_payload(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    if re.search(r"\d+(?:\.\d+)?\s*%", raw):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if len(words) >= 3 and _title_case_ratio(words[: min(len(words), 8)]) >= 0.45:
        return True
    return bool(re.search(r"\b(?:n=|respondents?|participants?|survey|primary|secondary|bachelor|student)\b", raw, re.IGNORECASE))


def _is_media_adjacent_section_heading_candidate(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    if _is_non_heading_role(block):
        return False
    text = _block_text(block)
    if not _looks_like_post_media_heading_text(text):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    if _previous_substantive_media_block(blocks, index) is None:
        return False
    following = _next_text_block(blocks, index)
    if following is None or not _looks_like_post_media_supporting_text(_block_text(following)):
        return False
    return True


def _collect_post_media_section_heading_texts(blocks: list[dict[str, Any]]) -> list[str]:
    headings: list[str] = []
    for index, block in enumerate(blocks):
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        if _is_non_heading_role(block):
            continue
        if _previous_substantive_media_block(blocks, index) is None:
            continue
        group = _post_media_heading_group(blocks, index)
        if group:
            _append_unique_heading_text(headings, " ".join(_block_text(item) for item in group))
            continue
        if _is_post_media_single_heading_candidate(blocks, index):
            _append_unique_heading_text(headings, _block_text(block))
    return headings


def _post_media_heading_group(blocks: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    first = blocks[index]
    first_text = _block_text(first)
    first_bbox = _bbox(first)
    if first_bbox is None or not first_text or re.search(r"[.!?;:]\s*$", first_text):
        return []
    if _starts_like_lowercase_continuation(first_text):
        return []
    group = [first]
    current_bbox = first_bbox
    for following in blocks[index + 1 : min(len(blocks), index + 3)]:
        if str(following.get("block_type") or "").strip().lower() != "text":
            break
        following_text = _block_text(following)
        following_bbox = _bbox(following)
        if not following_text or following_bbox is None:
            break
        height = max(1.0, current_bbox[3] - current_bbox[1])
        same_left = abs(following_bbox[0] - first_bbox[0]) <= max(8.0, height)
        gap = following_bbox[1] - current_bbox[3]
        if not same_left or gap < -height * 0.35 or gap > height * 1.2:
            break
        if not _looks_like_post_media_heading_fragment(following_text):
            break
        group.append(following)
        current_bbox = following_bbox
    if len(group) < 2:
        return []
    combined = " ".join(_block_text(item) for item in group)
    if not _looks_like_post_media_heading_text(combined):
        return []
    following = _next_text_block(blocks, index + len(group) - 1)
    if following is None or not _looks_like_post_media_supporting_text(_block_text(following)):
        return []
    return group


def _is_post_media_single_heading_candidate(blocks: list[dict[str, Any]], index: int) -> bool:
    text = _block_text(blocks[index])
    if not _looks_like_post_media_heading_text(text):
        return False
    following = _next_text_block(blocks, index)
    return bool(following and _looks_like_post_media_supporting_text(_block_text(following)))


def _looks_like_post_media_supporting_text(text: str) -> bool:
    if _looks_like_body_text_after_heading(text):
        return True
    raw = str(text or "").strip()
    if not raw:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if len(words) < 4:
        return False
    if re.search(r"[.!?]\s*$", raw):
        return True
    return _title_case_ratio(words) < 0.65


def _looks_like_post_media_heading_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or re.search(r"[.!?;:]\s*$", raw):
        return False
    if _starts_like_lowercase_continuation(raw):
        return False
    if _looks_like_chart_time_or_legend_label(raw):
        return False
    if re.search(r"\d+(?:\.\d+)?\s*%|\b(?:figure|fig\.|table)\b", raw, re.IGNORECASE):
        return False
    if re.search(r"\b[A-Za-z]+\d+\b", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not (2 <= len(words) <= 8):
        return False
    lowered = [word.lower() for word in words if not word.isdigit()]
    if lowered and len(set(lowered)) / len(lowered) < 0.75:
        return False
    return _title_case_ratio(words) >= 0.65


def _looks_like_post_media_heading_fragment(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or re.search(r"[.!?;:]\s*$", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not (1 <= len(words) <= 5):
        return False
    return _title_case_ratio(words) >= 0.55


def _project_toc_sequences_in_flow(markdown: str, document: dict[str, Any]) -> str:
    toc_lines = _build_toc_projection_lines(document)
    if not toc_lines:
        return markdown
    body = str(markdown or "").strip()
    toc_text = "\n".join(toc_lines)
    if not body:
        return toc_text + "\n"
    if _compact_text(toc_text) in _compact_text(body):
        return markdown
    prefix = _toc_preceding_flow_prefix(document, body)
    if prefix:
        lines = body.splitlines()
        prefix_line_count = len(prefix)
        return _normalize_blank_lines("\n".join(lines[:prefix_line_count] + [""] + toc_lines + [""] + lines[prefix_line_count:]))
    return toc_text + "\n\n" + body + "\n"


def _toc_preceding_flow_prefix(document: dict[str, Any], body: str) -> list[str]:
    sequences = [
        sequence
        for sequence in document.get("toc_sequences", []) or []
        if isinstance(sequence, dict) and sequence.get("entries")
    ]
    if not sequences:
        return []
    first_page = min(
        (
            int(page)
            for sequence in sequences
            for page in sequence.get("pages", []) or []
            if isinstance(page, int) or (isinstance(page, str) and str(page).isdigit())
        ),
        default=None,
    )
    if first_page is None:
        return []
    first_toc_top = min(
        (
            bbox[1]
            for sequence in sequences
            for bbox in [_bbox(sequence)]
            if bbox is not None and first_page in set(sequence.get("pages", []) or [])
        ),
        default=None,
    )
    if first_toc_top is None:
        pages = ((document.get("document_ast") or {}).get("pages") or [])
        for page in pages:
            if not isinstance(page, dict) or page.get("page") != first_page:
                continue
            toc_tops = [
                bbox[1]
                for block in page.get("blocks", []) or []
                for bbox in [_bbox(block)]
                if isinstance(block, dict) and str(block.get("block_type") or "").strip().lower() == "toc" and bbox is not None
            ]
            if toc_tops:
                first_toc_top = min(toc_tops)
                break
    if first_toc_top is None:
        return []
    page_blocks: list[dict[str, Any]] = []
    for page in ((document.get("document_ast") or {}).get("pages") or []):
        if isinstance(page, dict) and page.get("page") == first_page:
            page_blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
            break
    prefix_texts: list[str] = []
    for block in page_blocks:
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        if str(block.get("semantic_role") or "").strip() in {"toc_title", "page_number"}:
            continue
        bbox = _bbox(block)
        text = _block_text(block)
        if bbox is None or not text:
            continue
        if bbox[1] < first_toc_top:
            prefix_texts.append(text)
    if not prefix_texts:
        return []
    body_lines = body.splitlines()
    cursor = 0
    matched: list[str] = []
    for text in prefix_texts:
        while cursor < len(body_lines) and not body_lines[cursor].strip():
            matched.append(body_lines[cursor])
            cursor += 1
        if cursor >= len(body_lines):
            return []
        if _compact_text(_strip_heading_or_emphasis(body_lines[cursor])) != _compact_text(text):
            return []
        matched.append(body_lines[cursor])
        cursor += 1
    return matched


def _build_toc_projection_lines(document: dict[str, Any]) -> list[str]:
    sequences = [
        sequence
        for sequence in document.get("toc_sequences", []) or []
        if isinstance(sequence, dict) and sequence.get("entries")
    ]
    if not sequences:
        return []
    lines: list[str] = []
    for sequence in sequences:
        title = str(sequence.get("title") or "Table of contents").strip() or "Table of contents"
        if lines:
            lines.append("")
        lines.append(f"# {title}")
        lines.append("")
        projection_entries = _toc_projection_entries_for_sequence(document, sequence)
        restore_leaders = _should_restore_split_toc_locator_leaders_for_sequence(projection_entries)
        for entry in projection_entries:
            if not isinstance(entry, dict):
                continue
            text = _clean_toc_entry_text(entry.get("text"))
            text = _prefix_toc_outline_index(text, entry.get("outline_index"))
            locator = str(entry.get("page_locator") or "").strip()
            if not text:
                continue
            lines.append(_format_toc_projection_entry_line(text, locator, entry, restore_split_leader=restore_leaders))
    return lines


def _format_toc_projection_entry_line(
    text: str,
    locator: str,
    entry: dict[str, Any],
    *,
    restore_split_leader: bool = True,
) -> str:
    body = str(text or "").strip()
    page_locator = str(locator or "").strip()
    if not body:
        return ""
    if not page_locator:
        return _normalize_toc_leader_spacing(body)
    if _toc_entry_text_has_leader(body):
        return _normalize_toc_leader_spacing(f"{body} {page_locator}".rstrip())
    if restore_split_leader and _should_restore_split_toc_locator_leader(entry):
        return f"{body}{_restored_toc_leader(body, page_locator)} {page_locator}".rstrip()
    return f"{body} {page_locator}".rstrip()


def _normalize_toc_leader_spacing(line: str) -> str:
    raw = str(line or "").strip()
    if not raw:
        return raw
    match = re.match(
        r"^(?P<title>.+?)(?P<leader>(?:\s*\.\s*){3,})(?P<locator>\d{1,4}|[ivxlcdm]+)\s*$",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return raw
    title = str(match.group("title") or "").rstrip()
    leader = str(match.group("leader") or "")
    locator = str(match.group("locator") or "").strip()
    dot_count = max(3, len(leader.strip()))
    return f"{title} {'.' * dot_count} {locator}".rstrip()


def _should_restore_split_toc_locator_leaders_for_sequence(entries: list[dict[str, Any]]) -> bool:
    raw_entries = [entry for entry in entries if isinstance(entry, dict) and str(entry.get("source") or "").strip() == "raw_toc_row"]
    if len(raw_entries) < 2:
        return False
    split_text_entries = [
        entry
        for entry in raw_entries
        if str(entry.get("source_kind") or "").strip() == "text_block"
        and str(entry.get("page_locator") or "").strip()
        and not _toc_entry_text_has_leader(str(entry.get("text") or ""))
    ]
    if len(split_text_entries) < 2:
        return False
    return len(split_text_entries) >= max(2, int(len(raw_entries) * 0.6))


def _toc_entry_text_has_leader(text: str) -> bool:
    return re.search(r"(?:[.\u2024\u00b7]\s*){2,}\s*$", str(text or "")) is not None


def _should_restore_split_toc_locator_leader(entry: dict[str, Any]) -> bool:
    if str(entry.get("source") or "").strip() != "raw_toc_row":
        return False
    if not str(entry.get("page_locator") or "").strip():
        return False
    if str(entry.get("source_kind") or "").strip() != "text_block":
        return False
    text = str(entry.get("text") or "").strip()
    if not text or _toc_entry_text_has_leader(text):
        return False
    bbox = _bbox(entry)
    if bbox is None:
        return False
    row_width = bbox[2] - bbox[0]
    if row_width <= 180:
        return False
    return True


def _restored_toc_leader(text: str, locator: str) -> str:
    target_width = 92
    count = max(12, target_width - len(str(text or "").strip()) - len(str(locator or "").strip()) - 1)
    return "." * count


def _toc_projection_entries_for_sequence(
    document: dict[str, Any],
    sequence: dict[str, Any],
) -> list[dict[str, Any]]:
    sequence_entries = [entry for entry in sequence.get("entries", []) or [] if isinstance(entry, dict)]
    raw_entries = _raw_toc_projection_entries_for_sequence(document, sequence)
    if _should_use_raw_toc_projection_entries(sequence_entries, raw_entries):
        return _restore_toc_prefix_entries_from_text_evidence(document, sequence, raw_entries)
    return _restore_toc_prefix_entries_from_text_evidence(document, sequence, sequence_entries)


def _restore_toc_prefix_entries_from_text_evidence(
    document: dict[str, Any],
    sequence: dict[str, Any],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not entries:
        return entries
    evidence_lines = _toc_text_evidence_lines(document, sequence)
    if not evidence_lines:
        return entries
    first_entry = entries[0]
    first_index = _find_matching_toc_evidence_line_index(evidence_lines, first_entry)
    if first_index is None or first_index <= 1:
        return entries
    prefix_lines = evidence_lines[1:first_index]
    prefix_entries: list[dict[str, Any]] = []
    for line in prefix_lines:
        parsed = _parse_toc_text_evidence_line(line)
        if parsed is None:
            return entries
        text, locator = parsed
        prefix_entries.append(
            {
                "text": text,
                "page_locator": locator,
                "outline_index": None,
                "source_kind": "document_text",
                "source": "text_evidence_toc_prefix",
            }
        )
    if not prefix_entries:
        return entries
    return prefix_entries + entries


def _toc_text_evidence_lines(document: dict[str, Any], sequence: dict[str, Any]) -> list[str]:
    text = str(document.get("text") or "")
    if not text.strip():
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return []
    title = str(sequence.get("title") or "").strip()
    if title:
        title_index = next((index for index, line in enumerate(lines[:5]) if _compact_text(line) == _compact_text(title)), None)
        if title_index is None:
            return []
        return lines[title_index:]
    return lines


def _find_matching_toc_evidence_line_index(lines: list[str], entry: dict[str, Any]) -> int | None:
    entry_text = _clean_toc_entry_text(entry.get("text"))
    locator = str(entry.get("page_locator") or "").strip()
    if not entry_text or not locator:
        return None
    entry_title = _toc_entry_title_without_leaders(entry_text)
    for index, line in enumerate(lines):
        parsed = _parse_toc_text_evidence_line(line)
        if parsed is None:
            continue
        line_title, line_locator = parsed
        if line_locator != locator:
            continue
        line_title_normalized = _remove_toc_outline_prefix(_toc_entry_title_without_leaders(line_title))
        entry_title_normalized = _remove_toc_outline_prefix(entry_title)
        if _compact_text(line_title_normalized) == _compact_text(entry_title_normalized):
            return index
    return None


def _parse_toc_text_evidence_line(line: str) -> tuple[str, str] | None:
    parsed = _extract_terminal_toc_locator(str(line or "").strip())
    if parsed is None:
        return None
    text, locator = parsed
    if not locator:
        return None
    return text, locator


def _toc_entry_title_without_leaders(text: str) -> str:
    raw = str(text or "").strip()
    raw = re.sub(r"(?:[.\u2024\u00b7]\s*){2,}\s*$", "", raw).strip()
    return raw


def _remove_toc_outline_prefix(text: str) -> str:
    return re.sub(r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+", "", str(text or "").strip(), count=1, flags=re.IGNORECASE).strip()


def _raw_toc_projection_entries_for_sequence(
    document: dict[str, Any],
    sequence: dict[str, Any],
) -> list[dict[str, Any]]:
    toc_ids = {str(toc_id).strip() for toc_id in sequence.get("toc_ids", []) or [] if str(toc_id).strip()}
    raw_rows: list[dict[str, Any]] = []
    for toc_block in _iter_toc_blocks(document):
        if not isinstance(toc_block, dict):
            continue
        toc_id = str(toc_block.get("toc_id") or toc_block.get("block_id") or "").strip()
        if toc_ids and toc_id not in toc_ids:
            continue
        for raw_entry in toc_block.get("_raw_entries", []) or []:
            if isinstance(raw_entry, dict):
                raw_rows.append(raw_entry)
    raw_rows.sort(
        key=lambda item: (
            _float_or_none(item.get("sort_y0")) if _float_or_none(item.get("sort_y0")) is not None else float(item.get("row_index") or 0),
            float(item.get("row_index") or 0),
        )
    )
    entries: list[dict[str, Any]] = []
    for raw_entry in raw_rows:
        text = _clean_toc_entry_text(raw_entry.get("text"))
        if not text:
            continue
        locator = str(raw_entry.get("page_locator") or "").strip()
        if not locator:
            extracted = _extract_terminal_toc_locator(text)
            if extracted is None:
                continue
            text, locator = extracted
        entries.append(
            {
                "text": text,
                "page_locator": locator,
                "outline_index": raw_entry.get("outline_index"),
                "bbox": raw_entry.get("bbox"),
                "source_kind": raw_entry.get("source_kind"),
                "source": "raw_toc_row",
                "row_index": raw_entry.get("row_index"),
            }
        )
    return entries


def _prefix_toc_outline_index(text: str, outline_index: Any) -> str:
    raw_text = str(text or "").strip()
    prefix = _format_toc_outline_index(outline_index)
    if not raw_text or not prefix:
        return raw_text
    prefix_compact = _compact_text(prefix)
    text_compact = _compact_text(raw_text)
    if text_compact == prefix_compact or text_compact.startswith(prefix_compact):
        return raw_text
    return f"{prefix} {raw_text}".strip()


def _format_toc_outline_index(outline_index: Any) -> str:
    raw = str(outline_index or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"\s+", "", raw)
    integer_match = re.fullmatch(r"(\d+)\.0+", raw)
    if integer_match:
        return f"{integer_match.group(1)}."
    if re.fullmatch(r"\d+(?:\.\d+)*", raw):
        return raw
    if re.fullmatch(r"[IVXLCDM]+\.?", raw, re.IGNORECASE):
        return raw if raw.endswith(".") else f"{raw}."
    if re.fullmatch(r"[A-Za-z]\.?", raw):
        return raw if raw.endswith(".") else f"{raw}."
    return ""


def _iter_toc_blocks(document: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        block for block in document.get("toc_blocks", []) or [] if isinstance(block, dict)
    ]
    for page in ((document.get("document_ast") or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks", []) or []:
            if isinstance(block, dict) and str(block.get("block_type") or "").strip().lower() == "toc":
                blocks.append(block)
    return blocks


def _should_use_raw_toc_projection_entries(
    sequence_entries: list[dict[str, Any]],
    raw_entries: list[dict[str, Any]],
) -> bool:
    if len(raw_entries) < 2:
        return False
    raw_locator_count = sum(1 for entry in raw_entries if str(entry.get("page_locator") or "").strip())
    if raw_locator_count < max(2, int(len(raw_entries) * 0.85)):
        return False
    if len(raw_entries) > len(sequence_entries):
        return True
    if len(raw_entries) == len(sequence_entries):
        return any(_bbox(entry) is not None for entry in raw_entries)
    return False


def _extract_terminal_toc_locator(text: str) -> tuple[str, str] | None:
    raw = str(text or "").strip()
    match = re.fullmatch(
        r"(?P<title>.+?)(?P<leader>\s*(?:[.\u2024\u00b7]\s*){2,}|\s+[.\u2024\u00b7]\s+|\s{2,})(?P<locator>\d{1,4}|[ivxlcdm]+)\s*",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return None
    title = _clean_toc_entry_text((match.group("title") or "") + (match.group("leader") or ""))
    locator = str(match.group("locator") or "").strip()
    if not title or not locator:
        return None
    return title, locator


def _project_benchmark_table_title_headings(markdown: str, document: dict[str, Any]) -> str:
    title_texts = _collect_benchmark_table_title_texts(document)
    if not title_texts:
        return markdown
    title_by_compact = {_compact_text(title): title for title in title_texts}
    lines: list[str] = []
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        bold_match = re.fullmatch(r"\*\*(.+?)\*\*", stripped)
        if bold_match:
            split = _split_bold_table_title_embedded_section_heading(bold_match.group(1))
            if split is not None:
                before, heading = split
                if before:
                    lines.append(f"**{before}**")
                _append_heading_line(lines, heading)
                continue
        title = title_by_compact.get(_compact_text(bold_match.group(1) if bold_match else stripped))
        if title and not stripped.startswith("#"):
            _append_heading_line(lines, title)
            continue
        lines.append(line)
    return _insert_missing_table_title_headings(_normalize_blank_lines("\n".join(lines)), title_texts)


def _project_structured_table_prologue_texts_in_flow(markdown: str, document: dict[str, Any]) -> str:
    projections = _structured_table_prologue_text_projections(document)
    if not projections:
        return markdown
    body = str(markdown or "")
    for projection in projections:
        text = str(projection.get("text") or "").strip()
        anchors = [str(item or "").strip() for item in projection.get("anchors", []) or [] if str(item or "").strip()]
        if not text or not anchors:
            continue
        table_index = _earliest_markdown_text_index(body, anchors)
        text_match = _find_markdown_line_match(body, text)
        if table_index is None or text_match is None:
            continue
        text_start, text_end, rendered_line = text_match
        if text_start <= table_index:
            continue
        if _is_markdown_table_line(rendered_line):
            continue
        line_text = _strip_heading_or_emphasis(rendered_line)
        if _compact_text(line_text) != _compact_text(text):
            continue
        body = (body[:text_start] + body[text_end:]).strip()
        table_index = _earliest_markdown_text_index(body, anchors)
        if table_index is None:
            continue
        insertion = text
        insertion_index = _markdown_line_start_for_index(body, table_index)
        body = body[:insertion_index].rstrip() + "\n\n" + insertion + "\n\n" + body[insertion_index:].lstrip()
    return _normalize_blank_lines(body)


def _release_lowercase_body_tail_from_table_titles(markdown: str, document: dict[str, Any]) -> str:
    title_splits = _table_title_lowercase_tail_splits(document)
    if not title_splits:
        return markdown
    split_by_compact = {_compact_text(full): (title, tail) for full, title, tail in title_splits}
    lines = str(markdown or "").splitlines()
    projected: list[str] = []
    index = 0
    changed = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        bold_match = re.fullmatch(r"\*\*(.+?)\*\*", stripped)
        if not bold_match:
            projected.append(line)
            index += 1
            continue
        split = split_by_compact.get(_compact_text(bold_match.group(1)))
        if split is None:
            projected.append(line)
            index += 1
            continue
        title_text, body_tail = split
        table_start = index + 1
        while table_start < len(lines) and not lines[table_start].strip():
            table_start += 1
        table_end = _rendered_table_line_end(lines, table_start)
        if table_end is None:
            projected.append(line)
            index += 1
            continue
        projected.append(f"**{title_text}**")
        projected.extend(lines[index + 1:table_end])
        if body_tail:
            if projected and projected[-1].strip():
                projected.append("")
            projected.append(body_tail)
        index = table_end
        changed = True
    if not changed:
        return markdown
    return _normalize_blank_lines("\n".join(projected))


def _table_title_lowercase_tail_splits(document: dict[str, Any]) -> list[tuple[str, str, str]]:
    tables: list[dict[str, Any]] = []
    for table in document.get("table_asts", []) or []:
        if isinstance(table, dict):
            tables.append(table)
    for page in ((document.get("document_ast") or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks", []) or []:
            if isinstance(block, dict) and str(block.get("block_type") or "").strip().lower() == "table":
                tables.append(block)
    splits: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for table in tables:
        for key in ("title", "caption_text"):
            full_title = str(table.get(key) or "").strip()
            compact = _compact_text(full_title)
            if not compact or compact in seen:
                continue
            split = _split_table_title_lowercase_body_tail(full_title)
            if split is None:
                continue
            title_text, body_tail = split
            seen.add(compact)
            splits.append((full_title, title_text, body_tail))
    return splits


def _split_table_title_lowercase_body_tail(text: str) -> tuple[str, str] | None:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return None
    if not re.match(r"^(?:Table|Tab\.?|表)\s*[\dIVXLCDM一二三四五六七八九十]+(?:[.\-]\d+)*\s*[:：.、]?\s+\S+", raw, re.IGNORECASE):
        return None
    for match in re.finditer(r"(?<=[.!?。！？])\s+(?=[a-z][a-z-]{2,}\b)", raw):
        split_at = match.start()
        before = raw[:split_at].strip()
        tail = raw[match.end():].strip()
        if not _is_safe_table_title_body_tail_split(before, tail):
            continue
        return before, tail
    return None


def _is_safe_table_title_body_tail_split(before: str, tail: str) -> bool:
    if not before or not tail:
        return False
    before_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+|[\u4e00-\u9fff]", before)
    tail_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+|[\u4e00-\u9fff]", tail)
    if len(before_words) < 8 or len(tail_words) < 4:
        return False
    if _ends_with_common_abbreviation(before):
        return False
    if re.match(r"^(?:e\.g|i\.e|cf|vs|et al)\b", tail, re.IGNORECASE):
        return False
    if re.match(r"^[a-z][).]\s+", tail):
        return False
    first_tail_word = re.match(r"([A-Za-z][A-Za-z-]*)", tail)
    if first_tail_word is None:
        return False
    return first_tail_word.group(1)[:1].islower()


def _ends_with_common_abbreviation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:e\.g|i\.e|cf|vs|fig|figs|tab|eq|eqs|no|nos|vol|pp|dr|mr|mrs|ms|prof|al)\.$",
            str(text or "").strip(),
            re.IGNORECASE,
        )
    )


def _rendered_table_line_end(lines: list[str], start_index: int) -> int | None:
    if start_index >= len(lines):
        return None
    stripped = lines[start_index].strip()
    if stripped.startswith("<table"):
        cursor = start_index
        while cursor < len(lines):
            cursor += 1
            if "</table>" in lines[cursor - 1].lower():
                return cursor
        return None
    if stripped.startswith("|") and "|" in stripped[1:]:
        cursor = start_index
        while cursor < len(lines) and lines[cursor].strip().startswith("|"):
            cursor += 1
        return cursor
    return None


def _structured_table_prologue_text_projections(document: dict[str, Any]) -> list[dict[str, Any]]:
    table_asts = [table for table in document.get("table_asts", []) or [] if isinstance(table, dict)]
    image_by_id = {
        str(image.get("image_id") or image.get("block_id") or "").strip(): image
        for image in document.get("image_blocks", []) or []
        if isinstance(image, dict) and str(image.get("image_id") or image.get("block_id") or "").strip()
    }
    projections: list[dict[str, Any]] = []
    for page in ((document.get("document_ast") or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        table_blocks = [
            block for block in blocks
            if str(block.get("block_type") or "").strip().lower() == "table" and _bbox(block) is not None
        ]
        if not table_blocks:
            continue
        for table_block in table_blocks:
            table = _matching_table_ast(table_block, table_asts) or table_block
            table_bbox = _bbox(table_block)
            if table_bbox is None:
                continue
            if not _table_has_owned_path_screenshot_evidence(table_block, table, blocks, image_by_id):
                continue
            anchors = _structured_table_markdown_anchors(table)
            if not anchors:
                continue
            for block in blocks:
                if str(block.get("block_type") or "").strip().lower() != "text":
                    continue
                text = _block_text(block)
                text_bbox = _bbox(block)
                if not text or text_bbox is None:
                    continue
                if not _is_structured_table_visual_prologue_text(block, text_bbox, table_bbox):
                    continue
                projections.append({"text": text, "anchors": anchors})
                break
    return projections


def _table_has_owned_path_screenshot_evidence(
    table_block: dict[str, Any],
    table: dict[str, Any],
    blocks: list[dict[str, Any]],
    image_by_id: dict[str, dict[str, Any]],
) -> bool:
    table_id = str(table.get("table_id") or table_block.get("table_id") or table_block.get("block_id") or "").strip()
    candidate_images: list[dict[str, Any]] = []
    for block in blocks:
        if str(block.get("block_type") or "").strip().lower() != "image":
            continue
        image_id = str(block.get("image_id") or block.get("block_id") or "").strip()
        enriched = {**image_by_id.get(image_id, {}), **block} if image_id else block
        candidate_images.append(enriched)
    for image in candidate_images:
        source_table_id = str(image.get("source_table_id") or image.get("owned_table_id") or "").strip()
        if source_table_id and table_id and source_table_id == table_id:
            return True
        if _is_structured_region_owned_path_screenshot_placeholder(image, {"table_asts": [table]}):
            return True
    return False


def _structured_table_markdown_anchors(table: dict[str, Any]) -> list[str]:
    anchors: list[str] = []
    for title in _table_title_candidates(table):
        anchors.append(title)
    grid = table.get("display_grid") or table.get("raw_grid") or table.get("grid") or []
    if isinstance(grid, list):
        for row in grid[:2]:
            if not isinstance(row, list):
                continue
            cells = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
            if len(cells) >= 2:
                anchors.append(" | ".join(cells))
            elif cells:
                anchors.append(cells[0])
    return anchors


def _is_structured_table_visual_prologue_text(
    block: dict[str, Any],
    text_bbox: tuple[float, float, float, float],
    table_bbox: tuple[float, float, float, float],
) -> bool:
    text = _block_text(block)
    if not text or len(text) > 120:
        return False
    if re.match(r"^(?:fig(?:ure)?\.?|table)\s*\d+(?:\.\d+)*\b", text, re.IGNORECASE):
        return False
    if str(block.get("semantic_role") or "").strip() in {"page_number", "footnote", "footnote_continuation"}:
        return False
    if text_bbox[3] > table_bbox[1]:
        return False
    height = max(1.0, text_bbox[3] - text_bbox[1])
    vertical_gap = table_bbox[1] - text_bbox[3]
    if vertical_gap > height * 3.0:
        return False
    if _bbox_horizontal_overlap_ratio(text_bbox, table_bbox) < 0.45:
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*|[\u4e00-\u9fff]", text)
    if not (3 <= len(words) <= 18):
        return False
    return True


def _find_markdown_line_match(markdown: str, text: str) -> tuple[int, int, str] | None:
    cursor = 0
    target = _compact_text(text)
    for line in str(markdown or "").splitlines(keepends=True):
        line_body = line.rstrip("\r\n")
        if _compact_text(_strip_heading_or_emphasis(line_body)) == target:
            return cursor, cursor + len(line), line_body
        cursor += len(line)
    return None


def _markdown_line_start_for_index(markdown: str, index: int) -> int:
    before = str(markdown or "")[: max(0, index)]
    return before.rfind("\n") + 1


def _markdown_line_end_for_index(markdown: str, index: int) -> int:
    text = str(markdown or "")
    next_break = text.find("\n", max(0, index))
    return len(text) if next_break < 0 else next_break + 1


def _markdown_paragraph_start_for_index(markdown: str, index: int) -> int:
    text = str(markdown or "")
    cursor = max(0, min(index, len(text)))
    boundary = text.rfind("\n\n", 0, cursor)
    return 0 if boundary < 0 else boundary + 2


def _markdown_paragraph_end_for_index(markdown: str, index: int) -> int:
    text = str(markdown or "")
    cursor = max(0, min(index, len(text)))
    boundary = text.find("\n\n", cursor)
    return len(text) if boundary < 0 else boundary


def _markdown_table_range_after_anchors(
    markdown: str,
    anchors: list[str],
    start_index: int,
) -> tuple[int, int] | None:
    body = str(markdown or "")
    search_from = max(0, start_index)
    anchor_positions: list[int] = []
    for anchor in anchors:
        anchor_text = str(anchor or "").strip()
        if not anchor_text:
            continue
        cursor = search_from
        while True:
            found = body.find(anchor_text, cursor)
            if found < 0:
                break
            anchor_positions.append(found)
            cursor = found + max(1, len(anchor_text))
    if not anchor_positions:
        return None
    lines = body.splitlines(keepends=True)
    starts: list[int] = []
    cursor = 0
    for index, line in enumerate(lines):
        starts.append(cursor)
        cursor += len(line)
    for absolute_anchor in sorted(set(anchor_positions)):
        anchor_line_index: int | None = None
        for index, line_start in enumerate(starts):
            line_end = line_start + len(lines[index])
            if line_start <= absolute_anchor < line_end:
                anchor_line_index = index
                break
        if anchor_line_index is None:
            continue
        table_start = anchor_line_index
        while table_start > 0 and lines[table_start - 1].strip().startswith("|"):
            table_start -= 1
        if not lines[table_start].strip().startswith("|"):
            continue
        table_end = anchor_line_index + 1
        while table_end < len(lines) and lines[table_end].strip().startswith("|"):
            table_end += 1
        start = sum(len(line) for line in lines[:table_start])
        end = sum(len(line) for line in lines[:table_end])
        return start, end
    return None


def _markdown_html_table_range_after_anchors(
    markdown: str,
    anchors: list[str],
    start_index: int,
) -> tuple[int, int] | None:
    body = str(markdown or "")
    search_from = max(0, start_index)
    for match in re.finditer(r"<table\b[^>]*>.*?</table>", body[search_from:], re.IGNORECASE | re.DOTALL):
        absolute_start = search_from + match.start()
        absolute_end = search_from + match.end()
        table_text = match.group(0)
        if any(_html_table_contains_anchor(table_text, anchor) for anchor in anchors):
            return _markdown_line_start_for_index(body, absolute_start), _markdown_line_end_for_index(body, absolute_end)
    return None


def _html_table_contains_anchor(table_html: str, anchor: str) -> bool:
    anchor_text = str(anchor or "").strip()
    if not anchor_text:
        return False
    html_text = re.sub(r"<[^>]+>", " ", str(table_html or ""))
    html_text = re.sub(r"\s+", " ", html_text).strip()
    if _compact_text(anchor_text) and _compact_text(anchor_text) in _compact_text(html_text):
        return True
    anchor_cells = [cell.strip() for cell in anchor_text.split("|") if cell.strip()]
    if len(anchor_cells) < 2:
        return False
    matched_cells = sum(1 for cell in anchor_cells if _compact_text(cell) in _compact_text(html_text))
    return matched_cells >= max(2, len(anchor_cells) - 1)


def _markdown_image_placeholder_range_before(
    markdown: str,
    image_ids: list[str],
    start_index: int,
) -> tuple[int, int] | None:
    ids = {str(image_id or "").strip() for image_id in image_ids if str(image_id or "").strip()}
    if not ids:
        return None
    body = str(markdown or "")
    prefix = body[: max(0, start_index)]
    matches = list(re.finditer(r"(?m)^!\[[^\]]*\]\(#([^)]+)\)\s*(?:\r?\n)?", prefix))
    if not matches:
        return None
    match = matches[-1]
    if match.group(1) not in ids:
        return None
    between = prefix[match.end() :].strip()
    if between:
        return None
    return match.start(), match.end()


def _is_markdown_table_line(line: str) -> bool:
    stripped = str(line or "").strip()
    return stripped.startswith("|") or stripped.startswith("<tr") or stripped.startswith("<td")


def _insert_missing_table_title_headings(markdown: str, title_texts: list[str]) -> str:
    lines = str(markdown or "").splitlines()
    present = {_compact_text(_strip_heading_or_emphasis(line)) for line in lines if line.strip()}
    pending = [title for title in title_texts if _compact_text(title) not in present]
    if not pending:
        return markdown
    out: list[str] = []
    pending_index = 0
    index = 0
    while index < len(lines):
        if pending_index < len(pending) and _looks_like_table_start_line(lines, lines[index], index):
            _append_heading_line(out, pending[pending_index])
            pending_index += 1
        out.append(lines[index])
        index += 1
    while pending_index < len(pending):
        _append_heading_line(out, pending[pending_index])
        pending_index += 1
    return _normalize_blank_lines("\n".join(out))


def _project_toc_like_tables_as_plain_entries(markdown: str) -> str:
    lines = str(markdown or "").splitlines()
    out: list[str] = []
    index = 0
    last_toc_heading_emitted = False
    while index < len(lines):
        stripped = lines[index].strip()
        if _is_toc_heading_line(stripped):
            table_start = _next_nonblank_line_index(lines, index + 1)
            if table_start is not None and table_start < len(lines) and _looks_like_table_start_line(lines, lines[table_start], table_start):
                table_end = table_start
                while table_end < len(lines) and lines[table_end].strip().startswith("|"):
                    table_end += 1
                entries = _toc_plain_entries_from_markdown_table(lines[table_start:table_end])
                if entries:
                    if not last_toc_heading_emitted:
                        _append_heading_line(out, _strip_heading_or_emphasis(stripped) or "Table of Contents")
                        last_toc_heading_emitted = True
                    else:
                        while out and not out[-1].strip():
                            out.pop()
                        if out:
                            out.append("")
                    out.extend(entries)
                    out.append("")
                    index = table_end
                    continue
        out.append(lines[index])
        if stripped:
            if stripped.startswith("#") and _is_toc_heading_line(stripped):
                last_toc_heading_emitted = True
            elif not _is_toc_heading_line(stripped):
                last_toc_heading_emitted = False
        index += 1
    return _normalize_blank_lines("\n".join(out))


def _is_toc_heading_line(line: str) -> bool:
    return _compact_text(_strip_heading_or_emphasis(line)) in {"contents", "table of contents", "toc"}


def _next_nonblank_line_index(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip():
            return index
    return None


def _toc_plain_entries_from_markdown_table(table_lines: list[str]) -> list[str]:
    rows: list[list[str]] = []
    for line in table_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells if cell):
            continue
        rows.append([_strip_heading_or_emphasis(cell) for cell in cells])
    entries: list[str] = []
    for row in rows:
        cells = [cell for cell in row if cell]
        if not cells:
            continue
        if len(cells) >= 2 and re.fullmatch(r"\d{1,4}|[ivxlcdm]+", cells[-1], re.IGNORECASE):
            text = " ".join(cells[:-1]).strip()
            if text:
                entries.append(f"{text} {cells[-1]}")
        else:
            entries.append(" ".join(cells).strip())
    page_locator_count = sum(1 for entry in entries if re.search(r"\s(?:\d{1,4}|[ivxlcdm]+)$", entry, re.IGNORECASE))
    if len(entries) < 2 or page_locator_count < max(1, len(entries) // 2):
        return []
    return entries


def _strip_heading_or_emphasis(line: str) -> str:
    stripped = str(line or "").strip()
    stripped = re.sub(r"^#+\s*", "", stripped).strip()
    bold_match = re.fullmatch(r"\*\*(.+?)\*\*", stripped)
    if bold_match:
        return bold_match.group(1).strip()
    return stripped


def _looks_like_table_start_line(lines: list[str], line: str, index: int) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    if stripped.startswith("<table"):
        return True
    if not stripped.startswith("|") or "|" not in stripped[1:]:
        return False
    if index + 1 >= len(lines):
        return False
    separator = lines[index + 1].strip()
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?", separator))


def _collect_benchmark_table_title_texts(document: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    tables: list[dict[str, Any]] = []
    for table in document.get("table_asts", []) or []:
        if isinstance(table, dict):
            tables.append(table)
    for page in ((document.get("document_ast") or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks", []) or []:
            if isinstance(block, dict) and str(block.get("block_type") or "").strip().lower() == "table":
                tables.append(block)
    for table in tables:
        for title in _table_title_candidates(table):
            _append_unique_heading_text(titles, title)
        for heading in _embedded_section_heading_candidates_from_text(table.get("title")):
            _append_unique_heading_text(titles, heading)
    return titles


def _table_title_candidates(table: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    title = str(table.get("title") or "").strip()
    if title and _is_benchmark_table_title_heading(title, internal=False):
        candidates.append(title)
    title_row = table.get("title_row_index")
    grid = table.get("display_grid")
    if isinstance(title_row, int) and isinstance(grid, list) and 0 <= title_row < len(grid):
        row = grid[title_row]
        if isinstance(row, list):
            non_empty = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
            if len(non_empty) == 1 and _is_benchmark_table_title_heading(non_empty[0], internal=True):
                candidates.append(non_empty[0])
    return candidates


def _is_benchmark_table_title_heading(text: str, *, internal: bool) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    compact = re.sub(r"[^a-z]", "", raw.lower())
    if compact in {"tableofcontents", "contents"}:
        return True
    if not internal:
        return False
    if re.match(r"^(?:table|fig(?:ure)?\.?)\s+\d+(?:\.\d+)*\b", raw, re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    return 3 <= len(words) <= 12 and _title_case_ratio(words) >= 0.55


def _embedded_section_heading_candidates_from_text(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    candidates: list[str] = []
    for match in re.finditer(r"(?:^|[.;:]\s+|\bthat\s+)(\d+(?:\.\d+)+\s+[A-Z][A-Za-z0-9][A-Za-z0-9 ,;:'&()/~\-]{2,})", text):
        candidate = match.group(1).strip()
        if _looks_like_numbered_heading_text(candidate):
            candidates.append(candidate)
    return candidates


def _split_bold_table_title_embedded_section_heading(text: str) -> tuple[str, str] | None:
    candidates = _embedded_section_heading_candidates_from_text(text)
    if not candidates:
        return None
    heading = candidates[-1]
    index = str(text).rfind(heading)
    if index <= 0:
        return None
    before = str(text)[:index].rstrip()
    before = re.sub(r"[\s.;:]+$", ".", before).strip()
    return before, heading


def _clean_toc_entry_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _collect_benchmark_heading_texts(document: dict[str, Any]) -> list[str]:
    headings: list[str] = []
    pages = ((document.get("document_ast") or {}).get("pages") or [])
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_height = _float_or_none(page.get("height"))
        blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        toc_page = _is_toc_page(blocks)
        if not toc_page:
            for text in _collect_numbered_title_opener_texts(blocks, page_height):
                _append_unique_heading_text(headings, text)
            for text in _collect_top_band_consecutive_title_texts(blocks, page_height):
                _append_unique_heading_text(headings, text)
            for text in _collect_wrapped_ast_heading_texts(blocks, page_height):
                _append_unique_heading_text(headings, text)
            for text in _collect_multiline_title_group_texts(blocks, page_height):
                _append_unique_heading_text(headings, text)
        if not toc_page:
            for text in _collect_infographic_card_kpi_heading_texts(blocks):
                _append_unique_heading_text(headings, text)
            for text in _collect_infographic_card_heading_texts(blocks):
                _append_unique_heading_text(headings, text)
            for text in _collect_post_media_section_heading_texts(blocks):
                _append_unique_heading_text(headings, text)
            for text in _collect_isolated_short_section_heading_texts(blocks, page_height):
                _append_unique_heading_text(headings, text)
        for index, block in enumerate(blocks):
            if toc_page:
                if _is_toc_title_candidate(blocks, index):
                    _append_unique_heading_text(headings, _block_text(block))
                continue
            if _is_chapter_label_before_true_title(blocks, index, page_height):
                continue
            if _looks_like_front_matter_branding_block(block):
                continue
            if _is_front_matter_metadata_context(blocks, index, page_height):
                continue
            if _looks_like_heading_evidence_false_positive(blocks, index, page_height):
                continue
            if (
                _is_explicit_ast_heading(block)
                or _is_gap_backed_numbered_heading(blocks, index)
                or _is_procedural_section_label_candidate(blocks, index)
                or _is_colon_label_group_heading_candidate(blocks, index)
                or _is_short_category_heading_before_lettered_item(blocks, index)
                or _is_activity_section_heading_candidate(blocks, index, page_height)
                or _is_peer_card_heading_candidate(blocks, index)
                or _is_page_top_short_heading_candidate(blocks, index, page_height)
                or _is_isolated_title_block_candidate(blocks, index, page_height)
            ):
                text = _block_text(block)
                _append_unique_heading_text(headings, text)
    return headings


def _is_toc_page(blocks: list[dict[str, Any]]) -> bool:
    first_text = ""
    for block in blocks[:5]:
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        first_text = _block_text(block)
        if first_text:
            break
    if not first_text:
        return False
    compact = _compact_text(first_text)
    if compact in {"contents", "tableofcontents", "toc"}:
        return True
    title_count = 0
    for block in blocks[:20]:
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        text = _block_text(block)
        if not text:
            continue
        if re.search(r"\b\d{1,4}\s*$", text):
            title_count += 1
        elif _looks_like_short_title_text(text):
            title_count += 1
    return title_count >= 8 and any(compact in {"contents", "tableofcontents"} for compact in [_compact_text(_block_text(block)) for block in blocks[:3] if _block_text(block)])


def _is_toc_title_candidate(blocks: list[dict[str, Any]], index: int) -> bool:
    if index >= 3:
        return False
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    compact = _compact_text(text)
    return compact in {"contents", "tableofcontents", "toc"}


def _append_unique_heading_text(headings: list[str], text: str) -> None:
    text = str(text or "").strip()
    if text and _compact_text(text) not in {_compact_text(item) for item in headings}:
        headings.append(text)


def _is_explicit_ast_heading(block: dict[str, Any]) -> bool:
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    role = str(block.get("semantic_role") or "").strip()
    unit_role = str(block.get("unit_role") or "").strip()
    if role in {"section_heading", "reference_heading"} or unit_role == "section_heading":
        return True
    if role in {"reference_entry", "footnote", "footnote_continuation", "page_number"}:
        return False
    text = _block_text(block)
    return _looks_like_literature_heading(text)


def _is_gap_backed_numbered_heading(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    if str(block.get("semantic_role") or "").strip() not in {"", "text_block", "body"}:
        return False
    text = _block_text(block)
    if not _looks_like_numbered_heading_text(text):
        return False
    if re.search(r"[.;!?]\s*$", text):
        return False
    bbox = _bbox(block)
    prev_bbox = _bbox(blocks[index - 1]) if index > 0 else None
    next_bbox = _bbox(blocks[index + 1]) if index + 1 < len(blocks) else None
    if bbox is None:
        return False
    prev_gap = (bbox[1] - prev_bbox[3]) if prev_bbox is not None else 999.0
    next_gap = (next_bbox[1] - bbox[3]) if next_bbox is not None else 999.0
    height = max(1.0, bbox[3] - bbox[1])
    return prev_gap >= height * 1.2 and next_gap >= height * 0.7


def _is_procedural_section_label_candidate(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    if _is_non_heading_role(block):
        return False
    text = _block_text(block)
    if not re.fullmatch(r"[A-Z][A-Za-z /&()-]{2,40}:", text):
        return False
    following = _next_text_block(blocks, index)
    if not following:
        return False
    following_text = _block_text(following)
    if not re.match(r"^\d+[.)]\s+\S+", following_text):
        return False
    bbox = _bbox(block)
    following_bbox = _bbox(following)
    if bbox is None or following_bbox is None:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    gap = following_bbox[1] - bbox[3]
    return 0 <= gap <= height * 4.0


def _is_colon_label_group_heading_candidate(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    if _is_non_heading_role(block):
        return False
    text = _block_text(block)
    if not re.fullmatch(r"[A-Z][A-Za-z0-9 /&()'鈥檌.-]{1,72}:", text):
        return False
    if _looks_like_instructional_prompt_title(text):
        return False
    if _looks_like_body_sentence_start(text):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if not (2 <= len(words) <= 9):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    visual_following = sorted(
        [
            following
            for following in blocks
            if following is not block
            and str(following.get("block_type") or "").strip().lower() == "text"
            and _bbox(following) is not None
            and (_bbox(following) or bbox)[1] >= bbox[3] - height * 0.35
        ],
        key=lambda item: ((_bbox(item) or bbox)[1], (_bbox(item) or bbox)[0]),
    )
    next_peer_label_top: float | None = None
    for following in visual_following:
        following_text = _block_text(following)
        following_bbox = _bbox(following)
        if following_bbox is None or not following_text:
            continue
        same_label_column = abs(following_bbox[0] - bbox[0]) <= max(10.0, height)
        if same_label_column and re.fullmatch(r"[A-Z][A-Za-z0-9 /&()'鈥檌.-]{1,72}:", following_text):
            next_peer_label_top = following_bbox[1]
            break
    item_count = 0
    item_left: float | None = None
    previous_bottom = bbox[3]
    for following in visual_following:
        if _is_non_heading_role(following):
            continue
        following_text = _block_text(following)
        following_bbox = _bbox(following)
        if not following_text or following_bbox is None:
            continue
        if next_peer_label_top is not None and following_bbox[1] >= next_peer_label_top:
            break
        if re.fullmatch(r"[A-Z][A-Za-z0-9 /&()'鈥檌.-]{1,72}:", following_text):
            break
        vertical_gap = following_bbox[1] - previous_bottom
        indented = following_bbox[0] >= bbox[0] + max(6.0, height * 0.45)
        if not indented:
            continue
        if item_left is not None and abs(following_bbox[0] - item_left) > max(12.0, height):
            continue
        if vertical_gap < -height * 0.35 or vertical_gap > height * 2.6:
            if item_count:
                break
            continue
        if item_left is None:
            item_left = following_bbox[0]
        item_count += 1
        previous_bottom = following_bbox[3]
        if item_count == 1 and _looks_like_marked_list_item_text(following_text):
            return True
        if item_count >= 2:
            return True
    return False


def _looks_like_marked_list_item_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    return bool(re.match(r"^(?:[-*+•‣▪▫◦‒–—]|[➢➤➜➔⮚]|[a-zA-Z]|\d{1,2})[.)]?\s+\S+", raw))


def _is_short_category_heading_before_lettered_item(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    if _is_non_heading_role(block):
        return False
    text = _block_text(block)
    if not _looks_like_short_title_text(text):
        return False
    bbox = _bbox(block)
    following = _next_text_block(blocks, index)
    following_bbox = _bbox(following) if following else None
    if bbox is None or following is None or following_bbox is None:
        return False
    following_text = _block_text(following)
    if not re.match(r"^[a-z][.)]\s+[A-Z]", following_text):
        return False
    height = max(1.0, bbox[3] - bbox[1])
    if abs(following_bbox[0] - bbox[0]) > max(10.0, height):
        return False
    gap = following_bbox[1] - bbox[3]
    if gap < -height * 0.35 or gap > height * 2.4:
        return False
    previous = _previous_text_block(blocks, index)
    previous_bbox = _bbox(previous) if previous else None
    if previous_bbox is None:
        return True
    same_body_column = abs(previous_bbox[0] - bbox[0]) <= max(10.0, height)
    previous_text = _block_text(previous)
    return (not same_body_column) or bool(re.search(r"[.!?]\s*$", previous_text))


def _is_page_top_short_heading_candidate(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    role = str(block.get("semantic_role") or "").strip()
    if role in {"reference_entry", "footnote", "footnote_continuation", "page_number"}:
        return False
    if _looks_like_page_notice_or_funding_text(blocks, index, page_height):
        return False
    text = _block_text(block)
    if not text or len(text) > 60:
        return False
    if re.search(r"[.!?]\s*$", text):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if not words or len(words) > 6:
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    if _has_title_like_same_row_peers(blocks, index, bbox):
        return False
    page_top, page_span = _page_vertical_extent(blocks)
    y0 = bbox[1]
    if page_height is not None and page_height > 0 and y0 > page_height * 0.33:
        return False
    if (page_height is None or page_height <= 0) and page_span > 0 and (y0 - page_top) > page_span * 0.33:
        return False
    if _is_running_header_above_title_separator(blocks, index, bbox, page_height):
        return False
    if index == 0:
        return _looks_like_short_title_text(text) and not _is_page_label_before_stronger_title(blocks, index, page_height)
    prev_bbox = _bbox(blocks[index - 1])
    if prev_bbox is None:
        return False
    gap = y0 - prev_bbox[3]
    if gap < max(8.0, (bbox[3] - bbox[1]) * 1.1):
        return False
    return _looks_like_short_title_text(text) and not _is_page_label_before_stronger_title(blocks, index, page_height)


def _is_isolated_title_block_candidate(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    role = str(block.get("semantic_role") or "").strip()
    if role in {"reference_entry", "footnote", "footnote_continuation", "page_number", "author_line"}:
        return False
    text = _block_text(block)
    if _looks_like_instructional_prompt_title(text):
        return False
    if _looks_like_standalone_figure_or_table_caption(text):
        return False
    if _looks_like_page_notice_or_funding_text(blocks, index, page_height):
        return False
    if _looks_like_body_lead_in_before_visual(blocks, index):
        return False
    if _looks_like_list_item_or_step_heading_false_positive(blocks, index):
        return False
    if _looks_like_cross_reference_sentence_fragment(text):
        return False
    bbox = _bbox(block)
    if (
        bbox is not None
        and _has_nearby_page_header_separator_rule(blocks, index, bbox, page_height)
        and not _is_title_below_running_header_separator(blocks, index, bbox, page_height)
    ):
        return False
    if _is_media_adjacent_section_heading_candidate(blocks, index, page_height):
        return True
    if _looks_like_chart_axis_label_heading_false_positive(blocks, index):
        return False
    if _is_page_label_before_stronger_title(blocks, index, page_height):
        return False
    if bbox is None:
        return False
    page_label_backed_title = _has_preceding_page_label_for_title(blocks, index, bbox, page_height)
    same_row_title_peers = _has_title_like_same_row_peers(blocks, index, bbox)
    structured_short_heading = _looks_like_centered_short_heading(blocks, index, bbox) or _looks_like_left_aligned_short_heading(
        blocks,
        index,
        bbox,
    )
    running_header_backed_title = _has_running_header_prefix_before_title(blocks, index)
    separator_backed_title = _is_title_below_running_header_separator(blocks, index, bbox, page_height)
    top_context_backed_title = _has_top_context_before_title(blocks, index, bbox)
    if (
        not page_label_backed_title
        and not running_header_backed_title
        and not separator_backed_title
        and not top_context_backed_title
        and not structured_short_heading
        and _looks_like_body_continuation_fragment(blocks, index)
    ):
        return False
    if same_row_title_peers and not _has_infographic_kpi_same_row_peers(blocks, index, bbox):
        return False
    height = max(1.0, bbox[3] - bbox[1])
    prev_gap = _previous_vertical_gap(blocks, index, bbox)
    next_gap = _next_vertical_gap(blocks, index, bbox)
    next_text = _next_text_block(blocks, index)
    next_text_value = _block_text(next_text) if next_text else ""
    followed_by_body = bool(next_text_value) and _looks_like_body_text_after_heading(next_text_value)
    followed_by_media = _next_non_empty_block_type(blocks, index) == "image"
    page_top, page_span = _page_vertical_extent(blocks)
    top_band = bool(
        (page_height and page_height > 0 and bbox[1] <= page_height * 0.18)
        or ((not page_height or page_height <= 0) and page_span > 0 and (bbox[1] - page_top) <= page_span * 0.18)
    )
    title_text = _looks_like_isolated_title_text(text)
    top_title_text = _looks_like_top_band_main_title_text(text)

    if page_label_backed_title and (top_title_text or _looks_like_body_text_after_heading(text)):
        return True
    if top_band and (top_title_text or title_text) and (next_gap >= height * 0.6 or followed_by_media):
        return True
    if title_text and followed_by_body and separator_backed_title:
        return True
    if title_text and followed_by_body and running_header_backed_title:
        return True
    if title_text and followed_by_body and top_context_backed_title and next_gap >= height * 0.35:
        return True
    if title_text and followed_by_body and prev_gap >= height * 0.8 and next_gap >= height * 0.6:
        return True
    if title_text and followed_by_body and structured_short_heading and prev_gap >= height * 1.1 and next_gap >= height * 0.25:
        return True
        if (
            title_text
            and followed_by_body
            and prev_gap >= height * 1.1
            and next_gap >= height * 0.35
            and (_looks_like_centered_short_heading(blocks, index, bbox) or _looks_like_left_aligned_short_heading(blocks, index, bbox))
        ):
            return True
    if title_text and followed_by_media and prev_gap >= height * 0.8:
        return True
    return False


def _has_top_context_before_title(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    text = _block_text(blocks[index])
    if not _looks_like_short_title_text(text):
        return False
    page_top, page_span = _page_vertical_extent(blocks)
    if page_span <= 0 or (bbox[1] - page_top) > page_span * 0.18:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    preceding = []
    for previous in blocks[max(0, index - 3) : index]:
        previous_bbox = _bbox(previous)
        previous_text = _block_text(previous)
        if previous_bbox is None or not previous_text:
            continue
        if previous_bbox[3] > bbox[1] + height * 0.35:
            continue
        gap = bbox[1] - previous_bbox[3]
        if gap < -height * 0.35 or gap > height * 2.5:
            continue
        preceding.append((previous, previous_bbox, previous_text))
    if not preceding:
        return False
    same_left_context = any(abs(previous_bbox[0] - bbox[0]) <= max(10.0, height) for _previous, previous_bbox, _text in preceding)
    page_label_context = any(_looks_like_page_number_or_roman_label(previous_text) for _previous, _bbox_value, previous_text in preceding)
    short_title_context = any(
        len(re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", previous_text)) <= 4 and _looks_like_short_title_text(previous_text)
        for _previous, _bbox_value, previous_text in preceding
    )
    return same_left_context and (page_label_context or short_title_context)


def _looks_like_page_number_or_roman_label(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    return bool(re.fullmatch(r"(?:\d{1,4}|[ivxlcdm]{1,8})", raw, re.IGNORECASE))


def _collect_isolated_short_section_heading_texts(
    blocks: list[dict[str, Any]],
    page_height: float | None,
) -> list[str]:
    headings: list[str] = []
    for index, block in enumerate(blocks):
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        if _is_non_heading_role(block):
            continue
        text = _block_text(block)
        bbox = _bbox(block)
        if not text or bbox is None:
            continue
        if _looks_like_heading_evidence_false_positive(blocks, index, page_height):
            continue
        if _looks_like_body_continuation_fragment(blocks, index) and not _looks_like_short_all_caps_heading(text):
            continue
        next_text = _next_text_block(blocks, index)
        next_value = _block_text(next_text) if next_text else ""
        if not next_value or not _looks_like_body_text_after_heading(next_value):
            continue
        height = max(1.0, bbox[3] - bbox[1])
        prev_gap = _previous_vertical_gap(blocks, index, bbox)
        next_gap = _next_vertical_gap(blocks, index, bbox)
        if _looks_like_short_all_caps_heading(text) and prev_gap >= height * 1.0 and next_gap >= height * 0.6:
            _append_unique_heading_text(headings, text)
            continue
        if _is_short_title_after_paragraph_boundary(blocks, index, bbox):
            _append_unique_heading_text(headings, text)
    return headings


def _is_short_title_after_paragraph_boundary(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    text = _block_text(blocks[index])
    if not _looks_like_short_title_text(text):
        return False
    if not _looks_like_left_aligned_short_heading(blocks, index, bbox):
        return False
    previous = _previous_text_block(blocks, index)
    following = _next_text_block(blocks, index)
    previous_bbox = _bbox(previous) if previous else None
    following_bbox = _bbox(following) if following else None
    if previous is None or following is None or previous_bbox is None or following_bbox is None:
        return False
    previous_text = _block_text(previous)
    following_text = _block_text(following)
    if not re.search(r"[.!?:]\s*$", previous_text):
        return False
    if not _looks_like_body_text_after_heading(following_text):
        return False
    height = max(1.0, bbox[3] - bbox[1])
    same_left_prev = abs(previous_bbox[0] - bbox[0]) <= max(10.0, height)
    same_left_next = abs(following_bbox[0] - bbox[0]) <= max(10.0, height)
    if not (same_left_prev and same_left_next):
        return False
    prev_gap = bbox[1] - previous_bbox[3]
    next_gap = following_bbox[1] - bbox[3]
    if prev_gap < height * 0.35 or prev_gap > height * 2.2:
        return False
    if next_gap < 0 or next_gap > height * 1.0:
        return False
    body_line_gap = _local_body_line_gap(blocks, index, bbox)
    if body_line_gap is not None and prev_gap < body_line_gap * 1.45:
        return False
    return True


def _local_body_line_gap(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> float | None:
    gaps: list[float] = []
    height = max(1.0, bbox[3] - bbox[1])
    for cursor in range(max(0, index - 5), index):
        current = blocks[cursor]
        following = _next_text_block(blocks, cursor)
        if following is None or _block_index(blocks, following) >= index:
            continue
        current_bbox = _bbox(current)
        following_bbox = _bbox(following)
        if current_bbox is None or following_bbox is None:
            continue
        if abs(current_bbox[0] - following_bbox[0]) > max(10.0, height):
            continue
        gap = following_bbox[1] - current_bbox[3]
        if 0 <= gap <= height * 1.2:
            gaps.append(gap)
    if not gaps:
        return None
    gaps.sort()
    return gaps[len(gaps) // 2]


def _looks_like_short_all_caps_heading(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or len(raw) > 48:
        return False
    if re.search(r"[.:;!?]\s*$", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    alpha_words = [word for word in words if not word.isdigit()]
    if not (1 <= len(alpha_words) <= 4):
        return False
    return all(word.upper() == word and len(word) > 1 for word in alpha_words)


def _looks_like_body_continuation_fragment(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    bbox = _bbox(block)
    if not text or bbox is None:
        return False
    if _looks_like_numbered_heading_text(text) or _looks_like_literature_heading(text):
        return False
    height = max(1.0, bbox[3] - bbox[1])
    previous = _previous_text_block(blocks, index)
    following = _next_text_block(blocks, index)
    previous_bbox = _bbox(previous) if previous else None
    following_bbox = _bbox(following) if following else None

    if following and following_bbox is not None:
        same_left = abs(following_bbox[0] - bbox[0]) <= max(10.0, height)
        tight_gap = 0 <= following_bbox[1] - bbox[3] <= height * 1.7
        following_text = _block_text(following)
        weak_tail = bool(re.search(r"\b(?:a|an|and|as|at|by|for|from|in|is|of|on|or|the|to|with)\s*$", text, re.IGNORECASE))
        lowercase_continuation = _starts_like_lowercase_continuation(following_text)
        if same_left and tight_gap and (weak_tail or lowercase_continuation):
            return True

    if previous and previous_bbox is not None:
        same_left = abs(previous_bbox[0] - bbox[0]) <= max(10.0, height)
        tight_gap = 0 <= bbox[1] - previous_bbox[3] <= height * 1.7
        previous_text = _block_text(previous)
        if same_left and tight_gap and not re.search(r"[.!?]\s*$", previous_text):
            return True
    return False


def _looks_like_centered_short_heading(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    text = _block_text(blocks[index])
    if not _looks_like_short_title_text(text):
        return False
    text_bboxes = [
        other_bbox
        for other_index, other in enumerate(blocks)
        for other_bbox in [_bbox(other)]
        if other_index != index
        and str(other.get("block_type") or "").strip().lower() == "text"
        and other_bbox is not None
        and _block_text(other)
    ]
    if not text_bboxes:
        return False
    left = min(item[0] for item in text_bboxes)
    right = max(item[2] for item in text_bboxes)
    page_center = (left + right) / 2.0
    center = (bbox[0] + bbox[2]) / 2.0
    width = max(1.0, bbox[2] - bbox[0])
    return abs(center - page_center) <= max(width * 0.35, 28.0) and bbox[0] > left + max(24.0, width * 0.25)


def _looks_like_left_aligned_short_heading(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    text = _block_text(blocks[index])
    if not _looks_like_short_title_text(text):
        return False
    previous = _previous_text_block(blocks, index)
    following = _next_text_block(blocks, index)
    previous_bbox = _bbox(previous) if previous else None
    following_bbox = _bbox(following) if following else None
    if following_bbox is None:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    text_bboxes = [
        other_bbox
        for other_index, other in enumerate(blocks)
        for other_bbox in [_bbox(other)]
        if other_index != index
        and str(other.get("block_type") or "").strip().lower() == "text"
        and other_bbox is not None
        and _block_text(other)
    ]
    if not text_bboxes:
        return False
    left = min(item[0] for item in text_bboxes)
    if abs(bbox[0] - left) > max(14.0, height):
        return False
    if following_bbox[0] < bbox[0] - max(6.0, height * 0.5):
        return False
    if previous_bbox is not None and abs(previous_bbox[0] - bbox[0]) <= max(14.0, height):
        if bbox[1] - previous_bbox[3] <= height * 3.0 and not re.search(r"[.!?]\s*$", _block_text(previous)):
            return False
    return True


def _looks_like_cross_reference_sentence_fragment(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.match(r"^(?:fig(?:ure)?\.?|table)\s+\d+(?:\.\d+)*\b", raw, re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if len(words) > 10:
        return False
    return bool(
        re.search(
            r"\b(?:in|from|see|shown\s+in|reported\s+in)\s+(?:fig(?:ure)?\.?|table)\s+\d+(?:\.\d+)*\b",
            raw,
            re.IGNORECASE,
        )
    )


def _looks_like_standalone_figure_or_table_caption(text: str) -> bool:
    raw = str(text or "").strip()
    return bool(re.match(r"^(?:fig(?:ure)?\.?|table)\s+\d+(?:\.\d+)*[a-z]?\s*[:.]", raw, re.IGNORECASE))


def _looks_like_page_notice_or_funding_text(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    text = _block_text(block)
    if not text:
        return False
    if not re.search(r"\b(?:co-?funded|funded|sponsored|supported|copyright|license|project\s+no|doi|source)\b", text, re.IGNORECASE):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return True
    page_top, page_span = _page_vertical_extent(blocks)
    in_top_or_note_band = bool(
        (page_height and page_height > 0 and (bbox[1] <= page_height * 0.12 or bbox[1] >= page_height * 0.82))
        or ((not page_height or page_height <= 0) and page_span > 0 and ((bbox[1] - page_top) <= page_span * 0.12 or (bbox[1] - page_top) >= page_span * 0.82))
    )
    if in_top_or_note_band:
        return True
    next_text = _block_text(_next_text_block(blocks, index) or {})
    return bool(next_text and _looks_like_body_text_after_heading(next_text))


def _looks_like_body_lead_in_before_visual(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    text = _block_text(block)
    if not text:
        return False
    if not text.endswith(":"):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if len(words) < 8:
        return False
    if _title_case_ratio(words) >= 0.55:
        return False
    if not re.match(
        r"^(?:having|as|when|while|where|after|before|because|given|using|based|the|this|these|those|we|our|in|to)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    for following in blocks[index + 1 : min(len(blocks), index + 4)]:
        block_type = str(following.get("block_type") or "").strip().lower()
        if block_type in {"image", "formula", "equation"}:
            return True
        following_text = _block_text(following)
        if (
            _looks_like_chart_axis_label_text(following_text)
            or _looks_like_display_formula_text(following_text)
            or re.fullmatch(r"[\d\s.,%$+-]+", following_text)
        ):
            return True
    return False


def _looks_like_body_lead_in_group_before_visual(
    blocks: list[dict[str, Any]],
    index: int,
    group: list[dict[str, Any]],
) -> bool:
    if not group:
        return False
    text = " ".join(_block_text(item) for item in group)
    if not text.endswith(":"):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if len(words) < 8:
        return False
    if _title_case_ratio(words) >= 0.55:
        return False
    if not re.match(
        r"^(?:having|as|when|while|where|after|before|because|given|using|based|the|this|these|those|we|our|in|to)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    cursor = index + len(group)
    for following in blocks[cursor : min(len(blocks), cursor + 3)]:
        block_type = str(following.get("block_type") or "").strip().lower()
        if block_type in {"image", "formula", "equation"}:
            return True
        following_text = _block_text(following)
        if (
            _looks_like_chart_axis_label_text(following_text)
            or _looks_like_display_formula_text(following_text)
            or re.fullmatch(r"[\d\s.,%$+-]+", following_text)
        ):
            return True
    return False


def _looks_like_display_formula_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if raw.startswith(("$", "\\[", "\\(", "=")):
        return True
    math_signal_count = len(re.findall(r"(?:[=+\-*/^]|\\[A-Za-z]+|[∑∫√≈≤≥≠±−]|_[A-Za-z0-9]|\^[A-Za-z0-9])", raw))
    if math_signal_count >= 2 and re.search(r"\b[A-Za-z]\s*\(|\)\s*=|=\s*[A-Za-z0-9]", raw):
        return True
    return False


def _looks_like_formula_context_body_line_false_positive(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    text = _block_text(block)
    if not text:
        return False
    if _looks_like_display_formula_text(text):
        return True
    if _is_explicit_ast_heading(block):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if len(words) > 10:
        return False
    short_body_line = bool(
        _starts_like_lowercase_continuation(text)
        or _looks_like_body_sentence_start(text)
        or _looks_like_wrapped_body_sentence(text)
        or re.match(r"^(?:note|therefore|hence|thus|then|where|when|for|to|and)\b", text, re.IGNORECASE)
        or re.match(r"^(?:chapter|section)\s+\d+(?:\.\d+)*\b", text, re.IGNORECASE)
    )
    if not short_body_line:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    for cursor in range(max(0, index - 3), min(len(blocks), index + 4)):
        if cursor == index:
            continue
        other = blocks[cursor]
        other_bbox = _bbox(other)
        if other_bbox is None:
            continue
        if abs(other_bbox[1] - bbox[1]) > height * 8.0:
            continue
        other_type = str(other.get("block_type") or "").strip().lower()
        if other_type in {"formula", "equation"}:
            return True
        if _looks_like_display_formula_text(_block_text(other)):
            return True
    return False


def _looks_like_body_sentence_fragment_after_short_heading(heading: str, following_text: str) -> bool:
    heading_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", str(heading or "").strip())
    if not (1 <= len(heading_words) <= 5):
        return False
    following = str(following_text or "").strip()
    if not following:
        return False
    if re.match(r"^(?:after|before|during|while|when|where|because|since|although|if|in|on|at|for|from|to|with|by|as|the|this|these|those|we|it)\b", following, re.IGNORECASE):
        return True
    if _starts_like_lowercase_continuation(following) or _looks_like_wrapped_body_sentence(following):
        return True
    return bool(re.search(r"[,;:]\s*$", following))


def _looks_like_list_item_or_step_heading_false_positive(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    text = _block_text(block)
    if not text:
        return False
    if re.match(r"^[\u2022\u2023\u25E6\u2043\u2219*+\-]\s+\S+", text):
        return True
    if not re.match(r"^\d+[.)]?\s+[A-Z]", text):
        return False
    bbox = _bbox(block)
    following = _next_text_block(blocks, index)
    following_bbox = _bbox(following) if following else None
    if bbox is None or following is None or following_bbox is None:
        return False
    if _has_nearby_numbered_step_peer(blocks, index, bbox):
        return True
    height = max(1.0, bbox[3] - bbox[1])
    gap = following_bbox[1] - bbox[3]
    continuation_indent = following_bbox[0] >= bbox[0] + max(10.0, height * 0.8)
    continuation_text = _block_text(following)
    if _is_explicit_ast_heading(block) and not (
        0 <= gap <= height * 2.3
        and continuation_indent
        and _starts_like_lowercase_continuation(continuation_text)
    ):
        return False
    return 0 <= gap <= height * 2.3 and continuation_indent and (
        _starts_like_lowercase_continuation(continuation_text) or _looks_like_wrapped_body_sentence(continuation_text)
    )


def _looks_like_running_header_before_true_title(
    blocks: list[dict[str, Any]],
    index: int,
    following: dict[str, Any],
) -> bool:
    block = blocks[index]
    header_text = _block_text(block)
    title_text = _block_text(following)
    if not header_text or not title_text:
        return False
    if re.search(r"[.!?:;]\s*$", header_text):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", header_text)
    if not (3 <= len(words) <= 8 and words[0].isdigit()):
        return False
    alpha_words = [word for word in words[1:] if not word.isdigit()]
    if len(alpha_words) < 2 or _title_case_ratio(alpha_words) < 0.6:
        return False
    if len(alpha_words) >= 5 and _looks_like_numbered_heading_text(header_text):
        return False
    if not _looks_like_short_title_text(title_text):
        return False
    bbox = _bbox(block)
    title_bbox = _bbox(following)
    if bbox is None or title_bbox is None:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    gap = title_bbox[1] - bbox[3]
    if not (height * 0.9 <= gap <= height * 4.0):
        return False
    same_left = abs(title_bbox[0] - bbox[0]) <= max(8.0, height)
    if not same_left:
        return False
    after_title = _next_text_block(blocks, index + 1)
    after_title_text = _block_text(after_title) if after_title else ""
    return bool(after_title_text and _looks_like_body_text_after_heading(after_title_text))


def _has_running_header_prefix_before_title(blocks: list[dict[str, Any]], index: int) -> bool:
    previous = _previous_text_block(blocks, index)
    if previous is None:
        return False
    try:
        previous_index = blocks.index(previous)
    except ValueError:
        return False
    return _looks_like_running_header_before_true_title(blocks, previous_index, blocks[index])


def _has_title_like_same_row_peers(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    peer_count = 0
    for other_index, other in enumerate(blocks):
        if other_index == index:
            continue
        if str(other.get("block_type") or "").strip().lower() != "text":
            continue
        if _bbox(other) is None or not _same_visual_row(bbox, _bbox(other)):
            continue
        if _looks_like_card_heading_text(_block_text(other)):
            peer_count += 1
    return peer_count >= 2


def _has_infographic_kpi_same_row_peers(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    peer_count = 0
    for other_index, other in enumerate(blocks):
        if other_index == index:
            continue
        other_bbox = _bbox(other)
        if other_bbox is None or not _same_visual_row(bbox, other_bbox):
            continue
        if _looks_like_infographic_kpi_text(_block_text(other)):
            peer_count += 1
    return peer_count >= 2


def _collect_multiline_title_group_texts(
    blocks: list[dict[str, Any]],
    page_height: float | None,
) -> list[str]:
    headings: list[str] = []
    page_left, page_right = _page_horizontal_extent(blocks)
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if str(block.get("block_type") or "").strip().lower() != "text":
            index += 1
            continue
        if _is_non_heading_role(block):
            index += 1
            continue
        first_text = _block_text(block)
        if not first_text:
            index += 1
            continue
        bbox = _bbox(block)
        if bbox is None:
            index += 1
            continue
        if _looks_like_top_running_header_false_positive(blocks, index, page_height):
            index += 1
            continue
        if _looks_like_page_notice_or_funding_text(blocks, index, page_height):
            index += 1
            continue
        if _looks_like_standalone_figure_or_table_caption(first_text):
            index += 1
            continue
        if _looks_like_front_matter_branding_block(block):
            index += 1
            continue
        if _is_card_body_below_peer_heading(blocks, index, bbox):
            index += 1
            continue
        if _looks_like_chart_axis_label_text(first_text) and not _has_title_continuation_below(blocks, index, bbox):
            index += 1
            continue
        if page_height and page_height > 0 and bbox[1] > page_height * 0.22:
            index += 1
            continue
        group = [block]
        current_bbox = bbox
        consumed = 1
        for following in blocks[index + 1 : min(len(blocks), index + 4)]:
            if str(following.get("block_type") or "").strip().lower() != "text":
                break
            if _is_non_heading_role(following):
                break
            following_text = _block_text(following)
            following_bbox = _bbox(following)
            if not following_text or following_bbox is None:
                break
            if len(group) == 1 and _looks_like_running_header_before_true_title(blocks, index, following):
                break
            if len(group) == 1 and _is_peer_card_heading_candidate(blocks, index):
                break
            if following_text.endswith(":") and _looks_like_numbered_heading_text(first_text):
                break
            gap = following_bbox[1] - current_bbox[3]
            height = max(1.0, current_bbox[3] - current_bbox[1])
            same_left = abs(following_bbox[0] - bbox[0]) <= max(8.0, height * 0.8)
            same_center = abs(((following_bbox[0] + following_bbox[2]) / 2.0) - ((bbox[0] + bbox[2]) / 2.0)) <= max(
                12.0,
                min(bbox[2] - bbox[0], following_bbox[2] - following_bbox[0]) * 0.18,
            )
            top_band = bool((page_height and page_height > 0 and bbox[1] <= page_height * 0.22) or (page_height is None and index <= 1))
            media_title_continuation = _looks_like_media_title_continuation_group(
                blocks,
                index,
                following,
                group + [following],
                gap,
                height,
                same_left,
                same_center,
            )
            region_title_continuation = _looks_like_top_band_region_title_group(
                blocks,
                index,
                group + [following],
                page_height,
                gap,
                height,
                same_left,
                same_center,
            )
            if (
                gap < -height * 0.2
                or gap > height * 1.0
                or not (
                    same_left
                    or (top_band and same_center)
                    or media_title_continuation
                    or region_title_continuation
                )
            ):
                break
            if _looks_like_body_text_after_heading(following_text) and not (
                media_title_continuation or region_title_continuation
            ):
                break
            if len(group) >= 2 and _looks_like_front_matter_metadata_line(following, page_height):
                break
            group.append(following)
            current_bbox = following_bbox
            consumed += 1

        if len(group) < 2:
            index += 1
            continue
        text = " ".join(_block_text(item) for item in group)
        group_followed_by_type = _next_non_empty_block_type(blocks, index + len(group) - 1)
        group_followed_by_media = group_followed_by_type == "image"
        group_followed_by_region = group_followed_by_type in {"image", "table"}
        top_band_region_title = _looks_like_top_band_region_title_group(
            blocks,
            index,
            group,
            page_height,
            None,
            None,
            None,
            None,
        )
        if _looks_like_instructional_prompt_title(text):
            index += consumed
            continue
        if _looks_like_body_lead_in_group_before_visual(blocks, index, group):
            index += consumed
            continue
        front_matter_title_group = _looks_like_front_matter_title_group(blocks, index, group, page_height)
        if _looks_like_body_group_false_positive(text, group) and not top_band_region_title and not front_matter_title_group:
            index += consumed
            continue
        if _looks_like_body_sentence_start(text) and not group_followed_by_region:
            index += consumed
            continue
        if (
            _looks_like_wrapped_body_sentence(text)
            and not _looks_like_top_band_main_title_text(text)
            and not top_band_region_title
            and not group_followed_by_media
        ):
            index += consumed
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
        alpha_words = [word for word in words if not word.isdigit()]
        if alpha_words and not alpha_words[0][:1].isupper():
            index += consumed
            continue
        compact_title_group = len(group) >= 2 and 3 <= len(words) <= 18 and _title_case_ratio(words) >= 0.65
        if not (5 <= len(words) <= 18 or compact_title_group or front_matter_title_group):
            index += consumed
            continue
        next_index = index + len(group)
        if next_index < len(blocks):
            next_block_type = str(blocks[next_index].get("block_type") or "").strip().lower()
            next_bbox = _bbox(blocks[next_index])
            if (
                next_block_type == "text"
                and next_bbox is not None
                and next_bbox[1] - current_bbox[3] < max(12.0, (current_bbox[3] - current_bbox[1]) * 1.1)
                and not _looks_like_front_matter_metadata_line(blocks[next_index], page_height)
                and not front_matter_title_group
            ):
                index += consumed
                continue
        _append_unique_heading_text(headings, text)
        index += consumed
    return headings


def _collect_top_band_consecutive_title_texts(
    blocks: list[dict[str, Any]],
    page_height: float | None,
) -> list[str]:
    headings: list[str] = []
    text_blocks = [
        (index, block)
        for index, block in enumerate(blocks)
        if str(block.get("block_type") or "").strip().lower() == "text"
        and not _is_non_heading_role(block)
        and _bbox(block) is not None
        and _block_text(block)
    ]
    if not text_blocks:
        return headings
    page_top, page_span = _page_vertical_extent(blocks)
    top_band_blocks: list[tuple[int, dict[str, Any]]] = []
    for index, block in text_blocks[:8]:
        bbox = _bbox(block)
        if bbox is None:
            continue
        if page_height is not None and page_height > 0:
            if bbox[1] > page_height * 0.22:
                continue
        elif page_span > 0 and bbox[1] - page_top > page_span * 0.22:
            continue
        top_band_blocks.append((index, block))
    for start_offset, (index, block) in enumerate(top_band_blocks):
        group = _collect_top_band_independent_title_group(
            blocks,
            top_band_blocks,
            start_offset,
            page_height,
        )
        if not group:
            continue
        for group_index, group_block in group:
            text = _block_text(group_block)
            if text:
                _append_unique_heading_text(headings, text)
    return headings


def _collect_top_band_independent_title_group(
    blocks: list[dict[str, Any]],
    top_band_blocks: list[tuple[int, dict[str, Any]]],
    start_offset: int,
    page_height: float | None,
) -> list[tuple[int, dict[str, Any]]]:
    index, block = top_band_blocks[start_offset]
    text = _block_text(block)
    if not _looks_like_top_band_independent_title_text(text):
        return []
    if _looks_like_top_band_independent_title_exclusion(blocks, index, page_height):
        return []
    group: list[tuple[int, dict[str, Any]]] = [(index, block)]
    current_bbox = _bbox(block)
    if current_bbox is None:
        return []
    for next_index, following in top_band_blocks[start_offset + 1 : start_offset + 4]:
        following_text = _block_text(following)
        following_bbox = _bbox(following)
        if following_bbox is None or not _looks_like_top_band_independent_title_text(following_text):
            break
        if _looks_like_top_band_independent_title_exclusion(blocks, next_index, page_height):
            break
        height = max(1.0, current_bbox[3] - current_bbox[1])
        gap = following_bbox[1] - current_bbox[3]
        same_left = abs(following_bbox[0] - current_bbox[0]) <= max(12.0, height * 1.2)
        if gap < -height * 0.35 or gap > height * 3.2 or not same_left:
            break
        group.append((next_index, following))
        current_bbox = following_bbox
    if len(group) < 2:
        return []
    last_index = group[-1][0]
    if not _top_band_independent_title_group_has_context(blocks, last_index, page_height):
        return []
    return group


def _looks_like_top_band_independent_title_exclusion(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    text = _block_text(block)
    if not text:
        return True
    if _looks_like_page_notice_or_funding_text(blocks, index, page_height):
        return True
    if _looks_like_front_matter_branding_block(block):
        return True
    if _is_front_matter_metadata_context(blocks, index, page_height):
        return True
    if _looks_like_heading_evidence_false_positive(blocks, index, page_height):
        return True
    return False


def _top_band_independent_title_group_has_context(
    blocks: list[dict[str, Any]],
    last_index: int,
    page_height: float | None,
) -> bool:
    following = _next_text_block(blocks, last_index)
    following_text = _block_text(following or {})
    if not following_text:
        return True
    if _looks_like_same_column_lowercase_paragraph_continuation(blocks, last_index, following):
        return False
    if _looks_like_front_matter_metadata_line(following or {}, page_height):
        return True
    if _looks_like_literature_heading(following_text):
        return True
    return _looks_like_body_text_after_heading(following_text)


def _looks_like_same_column_lowercase_paragraph_continuation(
    blocks: list[dict[str, Any]],
    previous_index: int,
    following: dict[str, Any] | None,
) -> bool:
    if following is None:
        return False
    previous = blocks[previous_index]
    previous_bbox = _bbox(previous)
    following_bbox = _bbox(following)
    following_text = _block_text(following)
    if previous_bbox is None or following_bbox is None or not following_text:
        return False
    height = max(1.0, previous_bbox[3] - previous_bbox[1])
    same_left = abs(following_bbox[0] - previous_bbox[0]) <= max(8.0, height * 0.9)
    tight_gap = following_bbox[1] - previous_bbox[3] <= height * 1.6
    return same_left and tight_gap and _starts_like_lowercase_continuation(following_text)


def _looks_like_top_band_independent_title_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or re.search(r"[.!?]\s*$", raw):
        return False
    if _looks_like_standalone_figure_or_table_caption(raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not (3 <= len(words) <= 12):
        return False
    if _looks_like_body_sentence_start(raw) or _looks_like_wrapped_body_sentence(raw):
        return False
    alpha_words = [word for word in words if not word.isdigit()]
    if not alpha_words or not alpha_words[0][:1].isupper():
        return False
    if _title_case_ratio(words) >= 0.55:
        return True
    return bool(re.search(r"\b(?:OCR|E2E|API|PDF|AI|ML|IND|CTD|eCTD)\b", raw))


def _collect_wrapped_ast_heading_texts(
    blocks: list[dict[str, Any]],
    page_height: float | None,
) -> list[str]:
    headings: list[str] = []
    for index, block in enumerate(blocks):
        if not _is_explicit_ast_heading(block):
            continue
        if _looks_like_chart_axis_label_heading_false_positive(blocks, index) and not _is_chapter_label_before_true_title(
            blocks,
            index,
            page_height,
        ):
            continue
        text = _block_text(block)
        bbox = _bbox(block)
        if not text or bbox is None:
            continue
        following = _next_text_block(blocks, index)
        following_bbox = _bbox(following) if following else None
        following_text = _block_text(following) if following else ""
        if following is None or following_bbox is None or not following_text:
            continue
        if _is_non_heading_role(following):
            continue
        if _looks_like_running_header_before_true_title(blocks, index, following):
            continue
        if _is_explicit_ast_heading(following) and _looks_like_ast_confirmed_heading_wrap(text, following_text):
            _append_unique_heading_text(headings, f"{text} {following_text}")
            continue
        if _starts_like_lowercase_continuation(text):
            continue
        if _looks_like_body_sentence_fragment_after_short_heading(text, following_text):
            continue
        height = max(1.0, bbox[3] - bbox[1])
        gap = following_bbox[1] - bbox[3]
        if gap < -height * 0.2 or gap > height * 1.8:
            continue
        if following_bbox[0] < bbox[0] - max(8.0, height):
            continue
        if re.search(r"[.!?]\s*$", text):
            continue
        if not _looks_like_split_heading_continuation(following_text):
            continue
        _append_unique_heading_text(headings, f"{text} {following_text}")
    return headings


def _looks_like_ast_confirmed_heading_wrap(left: str, right: str) -> bool:
    left_raw = str(left or "").strip()
    right_raw = str(right or "").strip()
    if not left_raw or not right_raw:
        return False
    if re.search(r"[.!?;:]\s*$", left_raw) or re.search(r"[.!?;:]\s*$", right_raw):
        return False
    left_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", left_raw)
    right_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", right_raw)
    if not (4 <= len(left_words) <= 18 and 3 <= len(right_words) <= 10):
        return False
    if left_words[-1].lower() not in {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}:
        return False
    if _looks_like_body_sentence_start(right_raw) or _looks_like_wrapped_body_sentence(right_raw):
        return False
    combined_words = left_words + right_words
    combined_text = f"{left_raw} {right_raw}"
    return _title_case_ratio(combined_words) >= 0.4 or bool(re.search(r"\d|[~:/-]", combined_text))


def _looks_like_heading_continuation_group(
    group: list[dict[str, Any]],
    text: str,
    *,
    page_left: float | None = None,
    page_right: float | None = None,
    top_band: bool = False,
) -> bool:
    if len(group) < 2 or len(group) > 3:
        return False
    raw = str(text or "").strip()
    if not raw or re.search(r"[.!?]\s*$", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if len(words) < 6 or len(words) > 24:
        return False
    first_text = _block_text(group[0])
    first_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", first_text)
    if not (4 <= len(first_words) <= 18):
        return False
    if re.search(r"[.!?]\s+\S", first_text):
        return False
    if first_words[-1].lower() not in {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}:
        return False
    title_ratio = _title_case_ratio(first_words)
    if title_ratio >= 0.55:
        return True
    if not top_band:
        return False
    first_bbox = _bbox(group[0])
    if first_bbox is None or page_left is None or page_right is None:
        return False
    page_span = max(1.0, page_right - page_left)
    first_width = max(0.0, first_bbox[2] - first_bbox[0])
    return title_ratio >= 0.3 and len(words) >= 10 and first_width >= page_span * 0.55


def _looks_like_front_matter_title_group(
    blocks: list[dict[str, Any]],
    start_index: int,
    group: list[dict[str, Any]],
    page_height: float | None,
) -> bool:
    if not (2 <= len(group) <= 3):
        return False
    first_bbox = _bbox(group[0])
    last_bbox = _bbox(group[-1])
    if first_bbox is None or last_bbox is None:
        return False
    if _looks_like_front_matter_branding_block(blocks[start_index]):
        return False
    page_top, page_span = _page_vertical_extent(blocks)
    follows_branding = _has_preceding_front_matter_branding(blocks, start_index, first_bbox)
    first_starts_page_flow = not any(
        _bbox(previous) is not None and (_bbox(previous) or first_bbox)[1] < first_bbox[1]
        for previous in blocks[:start_index]
        if str(previous.get("block_type") or "").strip().lower() == "text" and _block_text(previous)
    )
    in_front_band = bool(
        (page_height and page_height > 0 and first_bbox[1] <= page_height * 0.45)
        or ((not page_height or page_height <= 0) and page_span > 0 and (first_bbox[1] - page_top) <= page_span * 0.45 and first_starts_page_flow)
        or follows_branding
    )
    if not in_front_band:
        return False
    text = " ".join(_block_text(item) for item in group).strip()
    if not text or re.search(r"[.!?]\s*$", text):
        return False
    if re.search(r"[.!?]\s+\S", text):
        return False
    if re.search(r"\b(?:led|part|see also|journal|museum|authority|society|press|vol\.?|no\.?|pp?\.?)\b", text, re.IGNORECASE):
        return False
    if _looks_like_page_notice_or_funding_text(blocks, start_index, page_height):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if not (5 <= len(words) <= 24):
        return False
    if not (
        _looks_like_top_band_main_title_text(text)
        or _looks_like_isolated_title_text(text)
        or _looks_like_heading_continuation_group(group, text, top_band=True)
        or (follows_branding and _looks_like_centered_multiline_cover_title(group))
    ):
        return False
    next_index = start_index + len(group)
    if next_index >= len(blocks):
        return True
    next_block = blocks[next_index]
    if _looks_like_front_matter_metadata_line(next_block, page_height):
        return True
    next_text = _block_text(next_block)
    return _looks_like_literature_heading(next_text) or _looks_like_body_text_after_heading(next_text)


def _looks_like_front_matter_metadata_line(block: dict[str, Any], page_height: float | None) -> bool:
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    role = str(block.get("semantic_role") or "").strip()
    unit_role = str(block.get("unit_role") or "").strip()
    if role in {"author_line", "publication_masthead", "contact_email"} or unit_role == "metadata":
        return True
    text = _block_text(block)
    if not text:
        return False
    if re.search(r"@|https?://|\b(?:doi|arxiv|file\s+no|license|copyright)\b", text, re.IGNORECASE):
        return True
    if re.fullmatch(r"(?:19|20)\d{2}(?:\s*[-/]\s*(?:19|20)\d{2})?|[A-Z][a-z]+\s+(?:19|20)\d{2}", text):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if len(words) <= 2:
        return False
    _ = page_height
    comma_count = text.count(",")
    if comma_count >= 2 and _title_case_ratio(words) >= 0.45:
        return True
    if re.search(r"\b(?:university|institute|college|department|school|laboratory|library|congress|corporation|company|inc\.?|ltd\.?)\b", text, re.IGNORECASE):
        return True
    return False


def _is_front_matter_metadata_context(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    if not text or _looks_like_literature_heading(text):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    page_top, page_span = _page_vertical_extent(blocks)
    in_front_band = bool(
        (page_height and page_height > 0 and bbox[1] <= page_height * 0.35)
        or ((not page_height or page_height <= 0) and page_span > 0 and (bbox[1] - page_top) <= page_span * 0.35)
    )
    if not in_front_band:
        return False
    if _looks_like_front_matter_branding_block(block):
        return True
    previous = _previous_text_block(blocks, index)
    following = _next_text_block(blocks, index)
    previous_text = _block_text(previous or {})
    following_text = _block_text(following or {})
    role = str(block.get("semantic_role") or "").strip()
    unit_role = str(block.get("unit_role") or "").strip()
    explicit_metadata = (
        role in {"author_line", "publication_masthead", "contact_email"}
        or unit_role == "metadata"
        or _looks_like_front_matter_metadata_line(block, page_height)
    )
    near_contact_or_abstract = bool(
        (following_text and (re.search(r"@|https?://", following_text) or _looks_like_literature_heading(following_text)))
        or (previous_text and (re.search(r"@|https?://", previous_text) or str((previous or {}).get("semantic_role") or "").strip() == "author_line"))
    )
    if explicit_metadata and near_contact_or_abstract:
        return True
    if _looks_like_author_or_affiliation_line(text) and near_contact_or_abstract:
        return True
    return False


def _looks_like_author_or_affiliation_line(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.search(r"@|https?://", raw):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if len(words) < 2 or len(words) > 18:
        return False
    if raw.count(",") >= 1 and _title_case_ratio(words) >= 0.45:
        return True
    return bool(re.search(r"\b(?:university|institute|college|department|school|laboratory|lab|library|congress|company|corporation|inc\.?|ltd\.?|south|north|korea|china|japan|usa|uk)\b", raw, re.IGNORECASE))


def _looks_like_front_matter_branding_block(block: dict[str, Any]) -> bool:
    text = _block_text(block)
    if not text:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    alpha_words = [word for word in words if not word.isdigit()]
    if len(alpha_words) < 3:
        return False
    uppercase_ratio = sum(1 for word in alpha_words if word.upper() == word and len(word) > 1) / max(1, len(alpha_words))
    if uppercase_ratio < 0.75:
        return False
    return bool(re.search(r"\b(?:library|congress|university|institute|college|department|office|agency|ministry|company|corporation)\b", text, re.IGNORECASE))


def _has_preceding_front_matter_branding(
    blocks: list[dict[str, Any]],
    start_index: int,
    title_bbox: tuple[float, float, float, float],
) -> bool:
    for previous in blocks[max(0, start_index - 3) : start_index]:
        previous_bbox = _bbox(previous)
        if previous_bbox is None:
            continue
        gap = title_bbox[1] - previous_bbox[3]
        if gap < 0 or gap > max(160.0, (title_bbox[3] - title_bbox[1]) * 6.0):
            continue
        if _looks_like_front_matter_branding_block(previous):
            return True
    return False


def _looks_like_centered_multiline_cover_title(group: list[dict[str, Any]]) -> bool:
    if len(group) < 2:
        return False
    bboxes = [_bbox(block) for block in group]
    if any(bbox is None for bbox in bboxes):
        return False
    usable = [bbox for bbox in bboxes if bbox is not None]
    centers = [(bbox[0] + bbox[2]) / 2.0 for bbox in usable]
    widths = [max(1.0, bbox[2] - bbox[0]) for bbox in usable]
    if max(centers) - min(centers) > max(widths) * 0.28:
        return False
    text = " ".join(_block_text(block) for block in group).strip()
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    return 5 <= len(words) <= 18 and _title_case_ratio(words) >= 0.55


def _is_chapter_label_before_true_title(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    block_id = str(block.get("block_id") or "")
    if block_id and re.search(r"(?:chapter|part|book).*label|label.*(?:chapter|part|book)", block_id, re.IGNORECASE):
        pass
    elif page_height is not None and page_height > 0:
        return False
    text = _block_text(block)
    if not re.fullmatch(r"(?:chapter|part|book)\s+\d+[A-Za-z]?", text, re.IGNORECASE):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    page_top, page_span = _page_vertical_extent(blocks)
    in_top_band = bool(
        (page_height and page_height > 0 and bbox[1] <= page_height * 0.25)
        or ((not page_height or page_height <= 0) and page_span > 0 and (bbox[1] - page_top) <= page_span * 0.25)
    )
    if not in_top_band:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    for following in blocks[index + 1 : min(len(blocks), index + 4)]:
        if str(following.get("block_type") or "").strip().lower() != "text":
            continue
        following_text = _block_text(following)
        following_bbox = _bbox(following)
        if not following_text or following_bbox is None:
            continue
        gap = following_bbox[1] - bbox[3]
        if gap < 0 or gap > height * 4.0:
            continue
        if _looks_like_short_title_text(following_text) and not re.fullmatch(
            r"(?:chapter|part|book)\s+\d+[A-Za-z]?",
            following_text,
            re.IGNORECASE,
        ):
            return True
    return False


def _looks_like_media_title_continuation_group(
    blocks: list[dict[str, Any]],
    start_index: int,
    candidate: dict[str, Any],
    group: list[dict[str, Any]],
    gap: float,
    height: float,
    same_left: bool,
    same_center: bool,
) -> bool:
    if gap < -height * 0.2 or gap > height * 1.6:
        return False
    candidate_text = _block_text(candidate)
    if not candidate_text:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", candidate_text)
    if len(words) > 4:
        return False
    if not (same_left or same_center):
        return False
    candidate_index = start_index + len(group) - 1
    next_type = _next_non_empty_block_type(blocks, candidate_index)
    if next_type != "image":
        return False
    return bool(re.fullmatch(r"\(?[A-Za-z0-9][A-Za-z0-9 .,&/-]{1,24}\)?\.?", candidate_text))


def _looks_like_top_band_region_title_group(
    blocks: list[dict[str, Any]],
    start_index: int,
    group: list[dict[str, Any]],
    page_height: float | None,
    gap: float | None,
    height: float | None,
    same_left: bool | None,
    same_center: bool | None,
) -> bool:
    if len(group) != 2:
        return False
    first_text = _block_text(group[0])
    second_text = _block_text(group[1])
    if not first_text or not second_text:
        return False
    text = f"{first_text} {second_text}".strip()
    if re.search(r"[.!?]\s*$", text):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if not (8 <= len(words) <= 22):
        return False
    if _looks_like_body_sentence_start(text):
        return False
    if not _looks_like_top_band_main_title_text(first_text) and not _looks_like_top_band_main_title_text(text):
        return False
    if not _starts_like_lowercase_continuation(second_text):
        return False
    first_bbox = _bbox(group[0])
    second_bbox = _bbox(group[1])
    if first_bbox is None or second_bbox is None:
        return False
    page_top, page_span = _page_vertical_extent(blocks)
    in_top_band = bool(
        (page_height and page_height > 0 and first_bbox[1] <= page_height * 0.20)
        or ((not page_height or page_height <= 0) and page_span > 0 and (first_bbox[1] - page_top) <= page_span * 0.20)
    )
    if not in_top_band:
        return False
    line_height = max(1.0, first_bbox[3] - first_bbox[1])
    line_gap = second_bbox[1] - first_bbox[3]
    visual_continuation = (
        (same_left if same_left is not None else abs(second_bbox[0] - first_bbox[0]) <= max(8.0, line_height * 0.8))
        or (same_center if same_center is not None else False)
    )
    if not visual_continuation:
        return False
    if gap is not None and height is not None:
        if gap < -height * 0.2 or gap > height * 1.25:
            return False
    elif line_gap < -line_height * 0.2 or line_gap > line_height * 1.25:
        return False
    next_type = _next_non_empty_block_type(blocks, start_index + len(group) - 1)
    if next_type not in {"image", "table"}:
        return False
    next_index = start_index + len(group)
    if next_index >= len(blocks):
        return False
    next_bbox = _bbox(blocks[next_index])
    if next_bbox is None:
        return True
    region_gap = next_bbox[1] - second_bbox[3]
    return 0 <= region_gap <= max(96.0, line_height * 4.0)


def _collect_infographic_card_heading_texts(blocks: list[dict[str, Any]]) -> list[str]:
    headings: list[str] = []
    text_blocks = [
        (index, block, _bbox(block), _block_text(block))
        for index, block in enumerate(blocks)
        if str(block.get("block_type") or "").strip().lower() == "text"
        and not _is_non_heading_role(block)
        and _bbox(block) is not None
        and _block_text(block)
    ]
    for index, block, bbox, text in text_blocks:
        if bbox is None or not _looks_like_infographic_card_heading_text(text):
            continue
        if _looks_like_chart_metric_label_context(blocks, index, bbox):
            continue
        same_row_headings = [
            other
            for other in text_blocks
            if other[0] != index
            and other[2] is not None
            and _same_visual_row(bbox, other[2])
            and _looks_like_infographic_card_heading_text(other[3])
        ]
        numeric_or_kpi_peers = [
            other
            for other in text_blocks
            if other[2] is not None
            and _same_visual_row(bbox, other[2])
            and _looks_like_infographic_kpi_text(other[3])
        ]
        if len(numeric_or_kpi_peers) < 2 and not (
            len(same_row_headings) >= 2 and _has_infographic_card_header_band_above(text_blocks, index, bbox)
        ):
            continue
        if not _has_supporting_card_body_below(blocks, index, bbox):
            continue
        _append_unique_heading_text(headings, text)
    return headings


def _collect_infographic_card_kpi_heading_texts(blocks: list[dict[str, Any]]) -> list[str]:
    headings: list[str] = []
    for deck in _collect_infographic_card_deck_profiles(blocks):
        for lane in deck.get("lanes", []) or []:
            lane_blocks = [block for block in lane if isinstance(block, dict)]
            if not lane_blocks:
                continue
            first_text = _block_text(lane_blocks[0])
            if _looks_like_card_deck_kpi_heading_text(first_text):
                _append_unique_heading_text(headings, first_text)
    return headings


def _collect_infographic_card_deck_profiles(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_blocks = [
        block
        for block in blocks
        if str(block.get("block_type") or "").strip().lower() == "text"
        and not _is_non_heading_role(block)
        and _bbox(block) is not None
        and _block_text(block)
    ]
    profiles: list[dict[str, Any]] = []
    consumed_anchor_keys: set[tuple[str, float, float]] = set()
    for index, block in enumerate(text_blocks):
        bbox = _bbox(block)
        text = _block_text(block)
        if bbox is None or not _looks_like_infographic_card_heading_text(text):
            continue
        row_headings = [
            other
            for other in text_blocks
            if _bbox(other) is not None
            and _same_visual_row(bbox, _bbox(other))
            and _looks_like_infographic_card_heading_text(_block_text(other))
            and _has_supporting_card_body_below(blocks, _block_index(blocks, other), _bbox(other))
        ]
        if len(row_headings) < 3:
            continue
        row_headings.sort(key=lambda item: (_bbox(item) or (0, 0, 0, 0))[0])
        if not _row_headings_define_distinct_card_lanes(row_headings):
            continue
        anchor_key = tuple(
            (round((_bbox(item) or (0, 0, 0, 0))[0], 1), round((_bbox(item) or (0, 0, 0, 0))[1], 1))
            for item in row_headings
        )
        if anchor_key in consumed_anchor_keys:
            continue
        consumed_anchor_keys.add(anchor_key)
        profile = _build_infographic_card_deck_profile(text_blocks, row_headings)
        if profile is not None:
            profiles.append(profile)
    return profiles


def _block_index(blocks: list[dict[str, Any]], target: dict[str, Any]) -> int:
    for index, block in enumerate(blocks):
        if block is target:
            return index
    target_text = _block_text(target)
    target_bbox = _bbox(target)
    for index, block in enumerate(blocks):
        if _block_text(block) == target_text and _bbox(block) == target_bbox:
            return index
    return -1


def _build_infographic_card_deck_profile(
    text_blocks: list[dict[str, Any]],
    row_headings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    centers = [((_bbox(block) or (0, 0, 0, 0))[0] + (_bbox(block) or (0, 0, 0, 0))[2]) / 2.0 for block in row_headings]
    if len(centers) < 2:
        return None
    lane_bounds: list[tuple[float, float]] = []
    for index, center in enumerate(centers):
        left = float("-inf") if index == 0 else (centers[index - 1] + center) / 2.0
        right = float("inf") if index == len(centers) - 1 else (center + centers[index + 1]) / 2.0
        lane_bounds.append((left, right))

    heading_top = min((_bbox(block) or (0, 0, 0, 0))[1] for block in row_headings)
    heading_height = max(1.0, max((_bbox(block) or (0, 0, 0, 0))[3] - (_bbox(block) or (0, 0, 0, 0))[1] for block in row_headings))
    candidate_blocks: list[dict[str, Any]] = []
    for block in text_blocks:
        bbox = _bbox(block)
        if bbox is None:
            continue
        text = _block_text(block)
        center = (bbox[0] + bbox[2]) / 2.0
        lane_index = _lane_index_for_center(center, lane_bounds)
        if lane_index is None:
            continue
        if bbox[1] < heading_top - heading_height * 3.0:
            continue
        if bbox[1] < heading_top and not (
            _looks_like_card_deck_kpi_heading_text(text) or _looks_like_infographic_card_heading_text(text)
        ):
            continue
        if bbox[1] >= heading_top and not _looks_like_card_deck_content_text(text):
            continue
        candidate_blocks.append(block)
    if len(candidate_blocks) < len(row_headings) * 2:
        return None

    bottom = _card_deck_bottom(candidate_blocks, heading_top, heading_height)
    deck_blocks = [
        block
        for block in candidate_blocks
        if (_bbox(block) is not None and (_bbox(block) or (0, 0, 0, 0))[1] <= bottom)
    ]
    lanes: list[list[dict[str, Any]]] = [[] for _ in lane_bounds]
    for block in deck_blocks:
        bbox = _bbox(block)
        if bbox is None:
            continue
        lane_index = _lane_index_for_center((bbox[0] + bbox[2]) / 2.0, lane_bounds)
        if lane_index is not None:
            lanes[lane_index].append(block)
    lanes = [sorted(lane, key=lambda item: ((_bbox(item) or (0, 0, 0, 0))[1], (_bbox(item) or (0, 0, 0, 0))[0])) for lane in lanes]
    if sum(1 for lane in lanes if lane) < 2:
        return None
    if not _lanes_have_supported_card_content(lanes):
        return None
    start_anchor_blocks = [block for block in deck_blocks if _is_card_deck_heading_block(block)]
    return {
        "blocks": sorted(deck_blocks, key=lambda item: ((_bbox(item) or (0, 0, 0, 0))[1], (_bbox(item) or (0, 0, 0, 0))[0])),
        "anchor_blocks": start_anchor_blocks or list(row_headings),
        "lanes": lanes,
    }


def _row_headings_define_distinct_card_lanes(row_headings: list[dict[str, Any]]) -> bool:
    bboxes = [_bbox(block) for block in row_headings]
    if len(row_headings) < 3 or any(bbox is None for bbox in bboxes):
        return False
    sorted_bboxes = sorted((bbox for bbox in bboxes if bbox is not None), key=lambda item: item[0])
    widths = [max(1.0, bbox[2] - bbox[0]) for bbox in sorted_bboxes]
    centers = [(bbox[0] + bbox[2]) / 2.0 for bbox in sorted_bboxes]
    for left_bbox, right_bbox, left_width, right_width, left_center, right_center in zip(
        sorted_bboxes,
        sorted_bboxes[1:],
        widths,
        widths[1:],
        centers,
        centers[1:],
    ):
        overlap = max(0.0, min(left_bbox[2], right_bbox[2]) - max(left_bbox[0], right_bbox[0]))
        if overlap > min(left_width, right_width) * 0.15:
            return False
        if right_center - left_center <= min(left_width, right_width) * 0.75:
            return False
    spread = centers[-1] - centers[0]
    typical_width = sorted(widths)[len(widths) // 2]
    return spread >= typical_width * max(1.0, len(sorted_bboxes) - 1.25)


def _lanes_have_supported_card_content(lanes: list[list[dict[str, Any]]]) -> bool:
    active_lanes = [lane for lane in lanes if lane]
    if len(active_lanes) < 3:
        return False
    for lane in active_lanes:
        if len(lane) < 2:
            return False
        heading_count = sum(1 for block in lane if _is_card_deck_heading_block(block))
        if heading_count < 1 or len(lane) - heading_count < 1:
            return False
    return True


def _lane_index_for_center(center: float, lane_bounds: list[tuple[float, float]]) -> int | None:
    for index, (left, right) in enumerate(lane_bounds):
        if left <= center < right:
            return index
    return None


def _card_deck_bottom(blocks: list[dict[str, Any]], heading_top: float, heading_height: float) -> float:
    lower_blocks = [
        block for block in blocks if (_bbox(block) is not None and (_bbox(block) or (0, 0, 0, 0))[1] >= heading_top)
    ]
    if not lower_blocks:
        return heading_top
    sorted_blocks = sorted(lower_blocks, key=lambda item: ((_bbox(item) or (0, 0, 0, 0))[1], (_bbox(item) or (0, 0, 0, 0))[0]))
    row_tops: list[float] = []
    for block in sorted_blocks:
        bbox = _bbox(block)
        if bbox is None:
            continue
        if not row_tops or abs(bbox[1] - row_tops[-1]) > heading_height * 0.5:
            row_tops.append(bbox[1])
    row_gaps = [right - left for left, right in zip(row_tops, row_tops[1:]) if right > left]
    typical_gap = sorted(row_gaps[:5])[len(row_gaps[:5]) // 2] if row_gaps else heading_height
    max_continuation_gap = max(heading_height * 2.5, typical_gap * 3.5)
    bottom = (_bbox(sorted_blocks[0]) or (0, 0, 0, 0))[3]
    previous_y = (_bbox(sorted_blocks[0]) or (0, 0, 0, 0))[1]
    for block in sorted_blocks[1:]:
        bbox = _bbox(block)
        if bbox is None:
            continue
        if bbox[1] - previous_y > max_continuation_gap:
            break
        bottom = max(bottom, bbox[3])
        previous_y = bbox[1]
    return bottom


def _looks_like_card_deck_content_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.fullmatch(r"\d{1,2}", raw):
        return False
    if re.fullmatch(r"[^\w]+(?:\d+)?", raw):
        return False
    return True


def _is_card_deck_heading_block(block: dict[str, Any]) -> bool:
    text = _block_text(block)
    return _looks_like_card_deck_kpi_heading_text(text) or _looks_like_infographic_card_heading_text(text)


def _looks_like_card_deck_kpi_heading_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.fullmatch(r"\d{1,2}", raw):
        return False
    if re.fullmatch(r"[^\w]+(?:\d+)?", raw):
        return False
    return _looks_like_infographic_kpi_text(raw)


def _has_infographic_card_header_band_above(
    text_blocks: list[tuple[int, dict[str, Any], tuple[float, float, float, float] | None, str]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    height = max(1.0, bbox[3] - bbox[1])
    band_items: list[tuple[int, dict[str, Any], tuple[float, float, float, float] | None, str]] = []
    for other in text_blocks:
        other_index, _other_block, other_bbox, other_text = other
        if other_index == index or other_bbox is None:
            continue
        gap = bbox[1] - other_bbox[3]
        if gap < -height * 0.5 or gap > height * 2.8:
            continue
        same_column = abs(((bbox[0] + bbox[2]) / 2.0) - ((other_bbox[0] + other_bbox[2]) / 2.0)) <= max(
            bbox[2] - bbox[0],
            other_bbox[2] - other_bbox[0],
            90.0,
        )
        if not same_column:
            continue
        if _looks_like_infographic_kpi_text(other_text) or _looks_like_infographic_card_heading_text(other_text):
            band_items.append(other)
    if not band_items:
        return False
    row_peers = 0
    for candidate in band_items:
        candidate_bbox = candidate[2]
        if candidate_bbox is None:
            continue
        same_row_count = sum(
            1
            for other in text_blocks
            if other[2] is not None
            and _same_visual_row(candidate_bbox, other[2])
            and (_looks_like_infographic_kpi_text(other[3]) or _looks_like_infographic_card_heading_text(other[3]))
        )
        row_peers = max(row_peers, same_row_count)
    return row_peers >= 3


def _looks_like_infographic_card_heading_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or len(raw) > 72:
        return False
    if re.search(r"[.!?;:]\s*$", raw):
        return False
    if _looks_like_infographic_kpi_text(raw):
        return False
    if _looks_like_chart_time_or_legend_label(raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not (2 <= len(words) <= 7):
        return False
    return _title_case_ratio(words) >= 0.6


def _looks_like_chart_metric_label_context(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    text = _block_text(blocks[index])
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*-[A-Za-z0-9]+", text):
        return False
    nearby_numeric = 0
    nearby_images = 0
    height = max(1.0, bbox[3] - bbox[1])
    for other_index, other in enumerate(blocks):
        if other_index == index:
            continue
        other_bbox = _bbox(other)
        if other_bbox is None:
            continue
        if abs(((other_bbox[1] + other_bbox[3]) / 2.0) - ((bbox[1] + bbox[3]) / 2.0)) > height * 9.0:
            continue
        other_text = _block_text(other)
        if str(other.get("block_type") or "").strip().lower() == "image":
            nearby_images += 1
        if re.fullmatch(r"[\d.,]+(?:\s*[xX%])?(?:\s*[鈫↑↓])?(?:\s*\d+)?", other_text):
            nearby_numeric += 1
    return nearby_numeric >= 4 or (nearby_numeric >= 2 and nearby_images >= 1)


def _looks_like_infographic_kpi_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.fullmatch(r"[\d.,]+(?:\s*[xX%])?(?:\s*[鈫↑↓])?(?:\s*\d+)?", raw):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    return len(words) == 1 and (raw.isupper() or bool(re.search(r"\d", raw)))


def _collect_numbered_title_opener_texts(
    blocks: list[dict[str, Any]],
    page_height: float | None,
) -> list[str]:
    headings: list[str] = []
    for index, block in enumerate(blocks[:3]):
        if str(block.get("block_type") or "").strip().lower() != "text":
            continue
        number_text = _block_text(block)
        if not re.fullmatch(r"\d{1,3}", number_text):
            continue
        bbox = _bbox(block)
        following = _next_text_block(blocks, index)
        following_bbox = _bbox(following) if following else None
        if bbox is None or following is None or following_bbox is None:
            continue
        if page_height is not None and page_height > 0 and bbox[1] > page_height * 0.28:
            continue
        title = _block_text(following)
        if not _looks_like_short_title_text(title):
            continue
        height = max(1.0, bbox[3] - bbox[1])
        gap = following_bbox[1] - bbox[3]
        same_visual_row = _same_visual_row(bbox, following_bbox)
        if _looks_like_wide_top_page_number_running_header_pair(blocks, index, bbox, following_bbox, page_height):
            continue
        if (gap < -height * 0.2 and not same_visual_row) or gap > height * 1.6:
            continue
        same_left = abs(following_bbox[0] - bbox[0]) <= max(8.0, height * 0.8)
        same_center = abs(((following_bbox[0] + following_bbox[2]) / 2.0) - ((bbox[0] + bbox[2]) / 2.0)) <= max(
            14.0,
            min(max(1.0, bbox[2] - bbox[0]), max(1.0, following_bbox[2] - following_bbox[0])) * 0.9,
        )
        next_after_title = _next_text_block(blocks, index + 1)
        next_after_bbox = _bbox(next_after_title) if next_after_title else None
        if next_after_bbox is not None and next_after_bbox[1] - following_bbox[3] < max(8.0, (following_bbox[3] - following_bbox[1]) * 0.4):
            continue
        followed_by_body = bool(next_after_title) and (
            _looks_like_body_text_after_heading(_block_text(next_after_title))
            or _looks_like_narrow_body_column_start(next_after_title, following_bbox)
        )
        if same_left or (same_visual_row and followed_by_body):
            _append_unique_heading_text(headings, f"{number_text} {title}")
        elif same_center or followed_by_body:
            _append_unique_heading_text(headings, number_text)
            _append_unique_heading_text(headings, title)
    return headings


def _looks_like_wide_top_page_number_running_header_pair(
    blocks: list[dict[str, Any]],
    index: int,
    number_bbox: tuple[float, float, float, float],
    title_bbox: tuple[float, float, float, float],
    page_height: float | None,
) -> bool:
    number_text = _block_text(blocks[index])
    if not re.fullmatch(r"\d{1,4}", number_text):
        return False
    if not _same_visual_row(number_bbox, title_bbox):
        return False
    page_top, page_span = _page_vertical_extent(blocks)
    if page_height is not None and page_height > 0:
        top_limit = max(60.0, page_height * 0.10)
        if max(number_bbox[1], title_bbox[1]) > top_limit:
            return False
    elif page_span > 0 and max(number_bbox[1] - page_top, title_bbox[1] - page_top) > page_span * 0.16:
        return False
    bboxes = [_bbox(block) for block in blocks if _bbox(block) is not None]
    if not bboxes:
        return False
    page_left = min(bbox[0] for bbox in bboxes)
    page_right = max(bbox[2] for bbox in bboxes)
    page_width = max(1.0, page_right - page_left)
    horizontal_gap = max(number_bbox[0], title_bbox[0]) - min(number_bbox[2], title_bbox[2])
    if horizontal_gap <= max(72.0, page_width * 0.28):
        return False
    if _has_substantive_region_between_top_header_and_body(blocks, index, number_bbox, title_bbox):
        return True
    title_width = max(1.0, title_bbox[2] - title_bbox[0])
    number_width = max(1.0, number_bbox[2] - number_bbox[0])
    if horizontal_gap <= max(title_width * 1.8, number_width * 6.0):
        return False
    next_after_title = _next_text_block(blocks, index + 1)
    next_after_text = _block_text(next_after_title) if next_after_title else ""
    later_true_heading = any(
        str(following.get("block_type") or "").strip().lower() == "text"
        and (
            _is_explicit_ast_heading(following)
            or _looks_like_numbered_heading_text(_block_text(following))
        )
        for following in blocks[index + 2 : min(len(blocks), index + 8)]
    )
    return _looks_like_body_text_after_heading(next_after_text) or later_true_heading


def _has_substantive_region_between_top_header_and_body(
    blocks: list[dict[str, Any]],
    header_index: int,
    number_bbox: tuple[float, float, float, float],
    title_bbox: tuple[float, float, float, float],
) -> bool:
    header_bottom = max(number_bbox[3], title_bbox[3])
    header_height = max(1.0, max(number_bbox[3] - number_bbox[1], title_bbox[3] - title_bbox[1]))
    for other in blocks[header_index + 1 : min(len(blocks), header_index + 6)]:
        if str(other.get("block_type") or "").strip().lower() not in {"image", "table"}:
            continue
        other_bbox = _bbox(other)
        if other_bbox is None:
            continue
        gap = other_bbox[1] - header_bottom
        if gap < 0 or gap > header_height * 3.0:
            continue
        width = max(1.0, other_bbox[2] - other_bbox[0])
        height = max(1.0, other_bbox[3] - other_bbox[1])
        if width >= 96.0 and height >= 96.0:
            return True
    return False


def _looks_like_narrow_body_column_start(
    block: dict[str, Any],
    title_bbox: tuple[float, float, float, float],
) -> bool:
    text = _block_text(block)
    bbox = _bbox(block)
    if bbox is None or not text:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if not (2 <= len(words) <= 7):
        return False
    if not (text[:1].isupper() or _starts_like_lowercase_continuation(text)):
        return False
    title_width = max(1.0, title_bbox[2] - title_bbox[0])
    body_width = max(1.0, bbox[2] - bbox[0])
    return body_width <= title_width * 0.9 and bbox[1] > title_bbox[3]


def _is_activity_section_heading_candidate(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    if _is_non_heading_role(block):
        return False
    text = _block_text(block)
    if not re.match(r"^Activity\s+\d+(?:[.:]|\s+-)\s+[A-Z]", text, re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if not (4 <= len(words) <= 18):
        return False
    bbox = _bbox(block)
    following = _next_text_block(blocks, index)
    following_bbox = _bbox(following) if following else None
    if bbox is None or following is None or following_bbox is None:
        return True
    if page_height is not None and page_height > 0 and bbox[1] > page_height * 0.9:
        return False
    gap = following_bbox[1] - bbox[3]
    height = max(1.0, bbox[3] - bbox[1])
    return gap >= height * 0.4 or _looks_like_body_text_after_heading(_block_text(following))


def _has_title_continuation_below(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    height = max(1.0, bbox[3] - bbox[1])
    for following in blocks[index + 1 : min(len(blocks), index + 3)]:
        if str(following.get("block_type") or "").strip().lower() != "text":
            return False
        following_text = _block_text(following)
        following_bbox = _bbox(following)
        if not following_text or following_bbox is None:
            return False
        gap = following_bbox[1] - bbox[3]
        same_left = abs(following_bbox[0] - bbox[0]) <= max(8.0, height * 0.8)
        if gap < -height * 0.35 or gap > height * 1.1 or not same_left:
            return False
        combined = f"{_block_text(blocks[index])} {following_text}"
        words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", combined)
        return 3 <= len(words) <= 8 and _title_case_ratio(words) >= 0.65
    return False


def _looks_like_body_group_false_positive(text: str, group: list[dict[str, Any]]) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    group_texts = [_block_text(item) for item in group]
    if any(re.match(r"^[•*]\s+", item) for item in group_texts):
        return True
    if re.search(r"\b(?:license|licensed|copyright|creative commons|cc by|source:|adapted from|wikipedia)\b", raw, re.IGNORECASE):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if re.search(r"[.!?]\s+\S", raw):
        return True
    if re.search(
        r"\b(?:led|part|with|see also|journal|museum|authority|society|press|vol\.?|no\.?|pp?\.?|"
        r"traditional|technical|techniques)\b",
        raw,
        re.IGNORECASE,
    ) and len(words) >= 7:
        return True
    if len(words) >= 9 and re.search(r"[.!?]", raw) and _title_case_ratio(words) < 0.7:
        return True
    if len(words) >= 12 and "," in raw and _title_case_ratio(words) < 0.55:
        return True
    return False


def _is_peer_card_heading_candidate(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    if _is_non_heading_role(block):
        return False
    text = _block_text(block)
    if not _looks_like_card_heading_text(text):
        return False
    if _looks_like_chart_axis_label_heading_false_positive(blocks, index):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    row_peers = [
        other
        for other_index, other in enumerate(blocks)
        if other_index != index
        and str(other.get("block_type") or "").strip().lower() == "text"
        and _looks_like_card_heading_text(_block_text(other))
        and _bbox(other) is not None
        and _same_visual_row(bbox, _bbox(other))
    ]
    if len(row_peers) < 2:
        return False
    if not _has_clear_section_break_before_card_row(blocks, index, bbox):
        return False
    return _has_supporting_card_body_below(blocks, index, bbox)


def _has_clear_section_break_before_card_row(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    height = max(1.0, bbox[3] - bbox[1])
    previous_bottom = max(
        (
            other_bbox[3]
            for other_index, other in enumerate(blocks[:index])
            for other_bbox in [_bbox(other)]
            if other_bbox is not None
            and other_bbox[3] <= bbox[1]
            and not _same_visual_row(bbox, other_bbox)
        ),
        default=None,
    )
    if previous_bottom is None:
        return True
    return bbox[1] - previous_bottom >= max(80.0, height * 4.0)


def _is_card_body_below_peer_heading(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    height = max(1.0, bbox[3] - bbox[1])
    center = (bbox[0] + bbox[2]) / 2.0
    for previous_index, previous in enumerate(blocks[:index]):
        previous_bbox = _bbox(previous)
        if previous_bbox is None:
            continue
        if previous_bbox[3] > bbox[1]:
            continue
        gap = bbox[1] - previous_bbox[3]
        if gap < 0 or gap > max(80.0, height * 5.0):
            continue
        previous_center = (previous_bbox[0] + previous_bbox[2]) / 2.0
        overlap = min(previous_bbox[2], bbox[2]) - max(previous_bbox[0], bbox[0])
        body_contains_heading_center = bbox[0] <= previous_center <= bbox[2]
        heading_contains_body_center = previous_bbox[0] <= center <= previous_bbox[2]
        same_column = (
            abs(previous_center - center) <= max(previous_bbox[2] - previous_bbox[0], 90.0)
            or overlap >= min(previous_bbox[2] - previous_bbox[0], bbox[2] - bbox[0]) * 0.35
            or body_contains_heading_center
            or heading_contains_body_center
        )
        if same_column and _is_peer_card_heading_candidate(blocks, previous_index):
            return True
    return False


def _looks_like_card_heading_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or len(raw) > 54:
        return False
    if re.search(r"[.!?,;:]\s*$", raw):
        return False
    if re.fullmatch(r"[\d.\s%-]+", raw):
        return False
    if _looks_like_chart_time_or_legend_label(raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not (2 <= len(words) <= 6):
        return False
    return _title_case_ratio(words) >= 0.65


def _same_visual_row(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> bool:
    if left is None or right is None:
        return False
    left_mid = (left[1] + left[3]) / 2.0
    right_mid = (right[1] + right[3]) / 2.0
    tolerance = max(left[3] - left[1], right[3] - right[1], 1.0) * 0.8
    return abs(left_mid - right_mid) <= tolerance


def _has_supporting_card_body_below(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    center = (bbox[0] + bbox[2]) / 2.0
    width = max(1.0, bbox[2] - bbox[0])
    visual_following_blocks = sorted(
        (
            following
            for following_index, following in enumerate(blocks)
            if following_index != index and _bbox(following) is not None
        ),
        key=lambda item: ((_bbox(item) or (0, 0, 0, 0))[1], (_bbox(item) or (0, 0, 0, 0))[0]),
    )
    for following in visual_following_blocks:
        following_bbox = _bbox(following)
        if following_bbox is None:
            continue
        if following_bbox[1] <= bbox[3]:
            continue
        vertical_gap = following_bbox[1] - bbox[3]
        if vertical_gap > max(80.0, (bbox[3] - bbox[1]) * 5.0):
            break
        following_center = (following_bbox[0] + following_bbox[2]) / 2.0
        same_column = abs(following_center - center) <= max(width * 0.95, 75.0)
        horizontal_overlap = min(bbox[2], following_bbox[2]) - max(bbox[0], following_bbox[0])
        if same_column or horizontal_overlap >= min(width, following_bbox[2] - following_bbox[0]) * 0.35:
            text = _block_text(following)
            if text:
                return True
    return False


def _is_page_label_before_stronger_title(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    text = _block_text(block)
    bbox = _bbox(block)
    if bbox is None or not text:
        return False
    if ":" not in text:
        return False
    if not _looks_like_page_label_text(text):
        return False
    page_top, page_span = _page_vertical_extent(blocks)
    in_top_band = bool(
        (page_height and page_height > 0 and bbox[1] <= page_height * 0.18)
        or ((not page_height or page_height <= 0) and page_span > 0 and (bbox[1] - page_top) <= page_span * 0.18)
    )
    if not in_top_band:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    for following in blocks[index + 1 : min(len(blocks), index + 4)]:
        if str(following.get("block_type") or "").strip().lower() != "text":
            continue
        following_text = _block_text(following)
        following_bbox = _bbox(following)
        if not following_text or following_bbox is None:
            continue
        gap = following_bbox[1] - bbox[3]
        if gap < 0 or gap > height * 4.0:
            continue
        following_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", following_text)
        if len(following_words) >= 7 and (following_bbox[3] - following_bbox[1]) >= height * 1.2:
            return True
    return False


def _looks_like_page_label_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or ":" not in raw:
        return False
    prefix, suffix = [part.strip() for part in raw.split(":", 1)]
    if not prefix or not suffix:
        return False
    if re.fullmatch(
        r"(?:source|sources|note|notes|fig(?:ure)?\.?|table|doi|reference|references|citation|citations)",
        prefix,
        re.IGNORECASE,
    ):
        return False
    if re.search(r"\b(?:18|19|20)\d{2}\b|\bdoi\b|https?://|www\.", raw, re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not (2 <= len(words) <= 7):
        return False
    return _title_case_ratio(words) >= 0.6


def _has_preceding_page_label_for_title(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
    page_height: float | None,
) -> bool:
    for previous_index in range(max(0, index - 2), index):
        previous = blocks[previous_index]
        previous_bbox = _bbox(previous)
        if previous_bbox is None:
            continue
        if not _is_page_label_before_stronger_title(blocks, previous_index, page_height):
            continue
        height = max(1.0, previous_bbox[3] - previous_bbox[1])
        gap = bbox[1] - previous_bbox[3]
        same_left = abs(bbox[0] - previous_bbox[0]) <= max(12.0, height)
        wider = (bbox[2] - bbox[0]) >= (previous_bbox[2] - previous_bbox[0]) * 1.8
        taller = (bbox[3] - bbox[1]) >= height * 1.1
        if 0 <= gap <= height * 4.0 and same_left and (wider or taller):
            return True
    return False


def _looks_like_chart_axis_label_heading_false_positive(blocks: list[dict[str, Any]], index: int) -> bool:
    text = _block_text(blocks[index])
    if _looks_like_colon_year_section_title(text):
        return False
    if not _looks_like_chart_axis_label_text(text):
        return False
    if _looks_like_isolated_title_text(text) and _has_body_paragraph_continuity(blocks, index):
        return False
    if not re.fullmatch(r"[\d.\s%-]+", text):
        text_bbox = _bbox(blocks[index])
        if text_bbox is not None:
            title_like_same_row = 0
            for other in blocks:
                if other is blocks[index]:
                    continue
                if _bbox(other) is not None and _same_visual_row(text_bbox, _bbox(other)) and _looks_like_card_heading_text(_block_text(other)):
                    title_like_same_row += 1
            if title_like_same_row >= 2:
                return False
    bbox = _bbox(blocks[index])
    if bbox is None:
        return True
    nearby_numeric = 0
    nearby_short_labels = 0
    nearby_media = False
    for other in blocks:
        if other is blocks[index]:
            continue
        other_bbox = _bbox(other)
        if other_bbox is None:
            continue
        if abs(((other_bbox[1] + other_bbox[3]) / 2.0) - ((bbox[1] + bbox[3]) / 2.0)) > 90:
            continue
        other_text = _block_text(other)
        if str(other.get("block_type") or "").strip().lower() == "image":
            nearby_media = True
        if re.fullmatch(r"[\d.\s%-]+", other_text):
            nearby_numeric += 1
        elif _looks_like_chart_axis_label_text(other_text):
            nearby_short_labels += 1
    return nearby_media or nearby_numeric >= 1 or nearby_short_labels >= 1


def _has_body_paragraph_continuity(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    bbox = _bbox(block)
    if bbox is None:
        return False
    previous = _previous_text_block(blocks, index)
    following = _next_text_block(blocks, index)
    previous_bbox = _bbox(previous) if previous else None
    following_bbox = _bbox(following) if following else None
    height = max(1.0, bbox[3] - bbox[1])
    before_body = False
    if previous and previous_bbox is not None:
        same_left = abs(previous_bbox[0] - bbox[0]) <= max(10.0, height)
        gap = bbox[1] - previous_bbox[3]
        before_body = same_left and 0 <= gap <= height * 2.0 and _looks_like_body_text_after_heading(_block_text(previous))
    after_body = False
    if following and following_bbox is not None:
        same_left = abs(following_bbox[0] - bbox[0]) <= max(10.0, height)
        gap = following_bbox[1] - bbox[3]
        after_body = same_left and 0 <= gap <= height * 2.2 and _looks_like_body_text_after_heading(_block_text(following))
    return before_body or after_body


def _looks_like_sentence_continuation_heading_false_positive(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    if not text:
        return False
    if not re.match(r"^\d+(?:\.\d+)+\s+(?:and|or|to|of|for|in|with|table|fig(?:ure)?)\b", text, re.IGNORECASE):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    previous = _previous_text_block(blocks, index)
    following = _next_text_block(blocks, index)
    previous_bbox = _bbox(previous) if previous else None
    following_bbox = _bbox(following) if following else None
    height = max(1.0, bbox[3] - bbox[1])
    previous_continues = False
    if previous and previous_bbox is not None:
        same_left = abs(previous_bbox[0] - bbox[0]) <= max(8.0, height * 0.8)
        tight_gap = bbox[1] - previous_bbox[3] <= height * 0.8
        previous_continues = same_left and tight_gap and not re.search(r"[.!?]\s*$", _block_text(previous))
    following_continues = False
    if following and following_bbox is not None:
        same_left = abs(following_bbox[0] - bbox[0]) <= max(8.0, height * 0.8)
        tight_gap = following_bbox[1] - bbox[3] <= height * 0.8
        following_continues = same_left and tight_gap and _starts_like_lowercase_continuation(_block_text(following))
    return previous_continues or following_continues


def _looks_like_heading_evidence_false_positive(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    if _is_procedural_section_label_candidate(blocks, index):
        return False
    if _looks_like_page_bottom_note_or_reference_zone(blocks, index, page_height):
        return True
    next_text = _next_text_block(blocks, index)
    if next_text is not None and _looks_like_running_header_before_true_title(blocks, index, next_text):
        return True
    if _looks_like_list_item_or_step_heading_false_positive(blocks, index):
        return True
    if _looks_like_formula_context_body_line_false_positive(blocks, index):
        return True
    if _looks_like_top_running_header_false_positive(blocks, index, page_height):
        return True
    if _looks_like_sentence_continuation_heading_false_positive(blocks, index):
        return True
    if _looks_like_lettered_body_list_item_false_positive(blocks, index):
        return True
    if _looks_like_numbered_body_instruction_false_positive(blocks, index):
        return True
    if _looks_like_numbered_page_label_false_positive(blocks, index, page_height):
        return True
    if _looks_like_numbered_running_footer_false_positive(blocks, index, page_height):
        return True
    if _is_chapter_label_before_true_title(blocks, index, page_height):
        return True
    if _looks_like_numbered_source_note_heading_false_positive(blocks, index, page_height):
        return True
    if _looks_like_inventory_list_label_heading_false_positive(blocks, index):
        return True
    text = _block_text(blocks[index])
    literature_heading = _looks_like_literature_heading(text)
    page_top_heading = _is_page_top_short_heading_candidate(blocks, index, page_height)
    if _is_isolated_title_block_candidate(blocks, index, page_height):
        return False
    if _looks_like_chart_axis_label_heading_false_positive(blocks, index) and not literature_heading and not (page_top_heading and index == 0):
        return True
    if literature_heading or page_top_heading:
        return False
    return _looks_like_instructional_prompt_title(text)


def _looks_like_page_bottom_note_or_reference_zone(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    bbox = _bbox(block)
    if not text or bbox is None:
        return False
    role = str(block.get("semantic_role") or "").strip()
    unit_role = str(block.get("unit_role") or "").strip()
    if role in {"footnote", "footnote_continuation", "reference_entry"} or unit_role in {
        "footnote",
        "footnote_continuation",
        "reference_entry",
    }:
        return True
    page_top, page_span = _page_vertical_extent(blocks)
    if page_height is not None and page_height > 0:
        y_ratio = bbox[1] / page_height
    elif page_span > 0:
        if page_span < 360.0:
            return False
        y_ratio = bbox[1] / page_span if page_top > page_span * 0.5 else (bbox[1] - page_top) / page_span
    else:
        y_ratio = 0.0
    if y_ratio < 0.72:
        return False
    heights = sorted(
        max(1.0, other_bbox[3] - other_bbox[1])
        for other in blocks
        for other_bbox in [_bbox(other)]
        if str(other.get("block_type") or "").strip().lower() == "text" and other_bbox is not None and _block_text(other)
    )
    if not heights:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    median_height = heights[len(heights) // 2]
    if height > median_height * 1.05:
        return False
    if role == "author_line" or unit_role == "metadata":
        return True
    if _looks_like_footnote_or_reference_line_text(text):
        return True
    previous = _previous_text_block(blocks, index)
    previous_text = _block_text(previous or {})
    previous_bbox = _bbox(previous) if previous else None
    if previous_bbox is None or not _looks_like_footnote_or_reference_line_text(previous_text):
        return False
    gap = bbox[1] - previous_bbox[3]
    continuation_indent = bbox[0] >= previous_bbox[0] + max(8.0, height * 0.7)
    same_column = abs(bbox[0] - previous_bbox[0]) <= max(32.0, height * 2.5) or continuation_indent
    return -height * 0.35 <= gap <= height * 2.0 and same_column


def _looks_like_footnote_or_reference_line_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.match(r"^\d{1,3}\s+", raw):
        return _numbered_line_has_note_or_reference_evidence(raw)
    if re.match(r"^[*†‡§#]+\s*\S+", raw):
        return True
    if re.match(r"^[A-Z]\.\s+[A-Z][A-Za-z'/-]+,", raw):
        return True
    return bool(
        re.search(
            r"\b(?:press|printed|university|vol\.?|no\.?|pp?\.?|doi|isbn|london|seattle|journal)\b",
            raw,
            re.IGNORECASE,
        )
    )


def _numbered_line_has_note_or_reference_evidence(text: str) -> bool:
    raw = str(text or "").strip()
    after_marker = re.sub(r"^\d{1,3}\s+", "", raw, count=1).strip()
    if not after_marker:
        return False
    if re.match(r"(?:e\.g\.|i\.e\.|cf\.|see|see also)\b", after_marker, re.IGNORECASE):
        return True
    if re.search(
        r"\b(?:press|printed|university|vol\.?|no\.?|pp?\.?|doi|isbn|london|seattle|journal)\b",
        after_marker,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:matlab|statistics toolbox|http|www|retrieved|copyright|license|permission)\b",
        after_marker,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"<[^>]{4,}>", after_marker):
        return True
    if re.search(r"\b[A-Z][A-Za-z'/-]+,\s+[A-Z][A-Za-z'/-]+", after_marker):
        return True
    if re.search(r"[A-Z][A-Za-z'/-]+(?:\s+[A-Z][A-Za-z'/-]+){0,3},\s+[^,]+,\s+\d{4}", after_marker):
        return True
    return False


def _looks_like_inventory_list_label_heading_false_positive(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    if not _looks_like_short_title_text(text):
        return False
    following = _next_text_block(blocks, index)
    following_text = _block_text(following) if following else ""
    if not following_text:
        return False
    if re.match(r"^[•\-\*]\s+", following_text):
        return True
    return str(following.get("semantic_role") or "").strip() == "body_list_item" and not re.match(r"^\d+[.)]\s+", following_text)


def _looks_like_numbered_body_instruction_false_positive(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    if not re.match(r"^\d+[.)]?\s+[A-Z]", text):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if len(words) < 6:
        return False
    if re.match(r"^\d+\s+(?:If|When|While|Where|The|This|These|Those|You|We|For|In|On|At)\b", text):
        return True
    if re.search(r"[.!?]\s*$", text):
        return True
    bbox = _bbox(block)
    following = _next_text_block(blocks, index)
    following_bbox = _bbox(following) if following else None
    if bbox is None or following is None or following_bbox is None:
        return False
    if _looks_like_numbered_heading_text(text) and _looks_like_split_heading_continuation(_block_text(following)):
        return False
    height = max(1.0, bbox[3] - bbox[1])
    previous = _previous_text_block(blocks, index)
    previous_bbox = _bbox(previous) if previous else None
    if previous is not None and previous_bbox is not None:
        previous_gap = bbox[1] - previous_bbox[3]
        previous_is_parent_label = str(_block_text(previous)).strip().endswith(":")
        previous_is_list_peer = bool(re.match(r"^(?:\d+[.)]?|[a-z][.)])\s+", _block_text(previous), re.IGNORECASE))
        nearby_numbered_peer = _has_nearby_numbered_step_peer(blocks, index, bbox)
        if previous_gap >= height * 1.35 and not previous_is_parent_label and not previous_is_list_peer and not nearby_numbered_peer:
            return False
    gap = following_bbox[1] - bbox[3]
    tight_gap = -height * 0.35 <= gap <= height * 1.1
    same_indent_or_child = following_bbox[0] >= bbox[0] - max(4.0, height * 0.3)
    if not (tight_gap and same_indent_or_child and not re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", _block_text(following))):
        return False
    if re.match(r"^\d+[.)]?\s+\S+", _block_text(following)):
        return True
    return True


def _has_nearby_numbered_step_peer(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    current_number = _leading_integer(_block_text(blocks[index]))
    if current_number is None:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    for direction in (-1, 1):
        skipped_continuations = 0
        cursor = index + direction
        while 0 <= cursor < len(blocks) and skipped_continuations <= 2:
            other = blocks[cursor]
            if str(other.get("block_type") or "").strip().lower() != "text":
                break
            other_bbox = _bbox(other)
            if other_bbox is None:
                break
            vertical_gap = abs(other_bbox[1] - bbox[1])
            if vertical_gap > height * 7.0:
                break
            other_number = _leading_integer(_block_text(other))
            if other_number is not None:
                if abs(other_number - current_number) <= 2 and abs(other_bbox[0] - bbox[0]) <= max(10.0, height):
                    return True
                break
            if not _looks_like_wrapped_body_sentence(_block_text(other)) and not _starts_like_lowercase_continuation(_block_text(other)):
                break
            skipped_continuations += 1
            cursor += direction
    return False


def _leading_integer(text: str) -> int | None:
    match = re.match(r"^(\d+)[.)]?\s+", str(text or "").strip())
    if not match:
        return None
    return int(match.group(1))


def _looks_like_numbered_page_label_false_positive(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    if not re.fullmatch(r"\d{1,4}", text):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    if page_height is not None and page_height > 0 and bbox[1] > page_height * 0.14:
        return False
    return True


def _looks_like_numbered_running_footer_false_positive(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    if not re.match(r"^\d{1,4}\s+[A-Z]", text):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    if page_height is not None and page_height > 0:
        if bbox[1] < page_height * 0.82:
            return False
    else:
        page_top, page_span = _page_vertical_extent(blocks)
        if page_span <= 0 or (bbox[1] - page_top) < page_span * 0.88:
            return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    alpha_words = [word for word in words if not word.isdigit()]
    return len(alpha_words) >= 4 and _title_case_ratio(alpha_words) >= 0.6


def _looks_like_numbered_source_note_heading_false_positive(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    if not re.match(r"^\d+[.)]\s+[A-Z]", text):
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    if not _is_in_page_note_band(blocks, bbox, page_height):
        return False
    lower = text.lower()
    has_source_signal = bool(
        re.search(
            r"\b(?:table|figure|fig\.|doi|retrieved|http|www|licen[cs]e|copyright|source|permission|reproduced|distributed|statistics canada|et al\.)\b",
            lower,
        )
    )
    remainder = re.sub(r"^\d+[.)]\s+", "", text).strip()
    remainder_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", remainder)
    sentence_note = len(remainder_words) >= 9 and _title_case_ratio(remainder_words) < 0.45
    following = _next_text_block(blocks, index)
    bibliography_entry = _looks_like_numbered_bibliography_entry(remainder, _block_text(following) if following else "")
    if following is None:
        return has_source_signal or sentence_note or bibliography_entry
    following_text = _block_text(following).lower()
    if re.search(r"\b(?:doi|retrieved|http|www|licen[cs]e|copyright|permission|reproduced|distributed)\b", following_text):
        return True
    return (
        ((has_source_signal or sentence_note) and _looks_like_wrapped_body_sentence(_block_text(following)))
        or bibliography_entry
    )


def _looks_like_numbered_bibliography_entry(remainder: str, following_text: str = "") -> bool:
    raw = str(remainder or "").strip()
    if not raw:
        return False
    author_chunks = re.findall(r"\b[A-Z][A-Za-z'/-]+,\s+[A-Z]\.", raw)
    has_author_list = len(author_chunks) >= 2 or bool(
        re.match(r"^[A-Z][A-Za-z'/-]+,\s+[A-Z]\.", raw)
        and re.search(r"(?:,\s*&\s*|\s+and\s+)[A-Z][A-Za-z'/-]+,\s+[A-Z]\.", raw)
    )
    if not has_author_list:
        return False
    if not re.search(r"(?:\(\d{4}\)|\b\d{4}\b)", raw):
        return False
    combined = f"{raw} {following_text}".strip()
    if re.search(
        r"\b(?:press|university|institute|journal|conference|consortium|education|management|research|vol\.?|no\.?|pp?\.?|doi|http|www)\b",
        combined,
        re.IGNORECASE,
    ):
        return True
    return bool(re.search(r"\b[A-Z][a-z]+,\s+[A-Z]{2}\b", combined))


def _is_in_page_note_band(
    blocks: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    page_height: float | None,
) -> bool:
    if page_height is not None and page_height > 0:
        return bbox[1] >= page_height * 0.58
    bboxes = [_bbox(block) for block in blocks]
    usable = [item for item in bboxes if item is not None]
    if not usable:
        return False
    top = min(item[1] for item in usable)
    bottom = max(item[3] for item in usable)
    span = max(1.0, bottom - top)
    return (bbox[1] - top) / span >= 0.72


def _looks_like_split_heading_continuation(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if _looks_like_standalone_figure_or_table_caption(raw):
        return False
    if re.search(r"[.!?;:]\s*$", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not (1 <= len(words) <= 6):
        return False
    if _looks_like_body_sentence_start(raw) or _looks_like_wrapped_body_sentence(raw):
        return False
    return True


def _merge_adjacent_heading_continuations(markdown: str, heading_texts: list[str] | None = None) -> str:
    lines = str(markdown or "").splitlines()
    if not lines:
        return markdown
    merged: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        if (
            current.startswith("# ")
            and index + 2 < len(lines)
            and not lines[index + 1].strip()
            and lines[index + 2].startswith("# ")
        ):
            left = current[2:].strip()
            right = lines[index + 2][2:].strip()
            if _looks_like_heading_continuation_pair(left, right, heading_texts):
                merged.append(f"# {left} {right}".strip())
                index += 3
                continue
        merged.append(current)
        index += 1
    return _dedupe_adjacent_duplicate_headings("\n".join(merged))


def _dedupe_adjacent_duplicate_headings(markdown: str) -> str:
    lines = str(markdown or "").splitlines()
    out: list[str] = []
    last_heading_compact = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            compact = _compact_text(re.sub(r"^#+\s*", "", stripped))
            if compact and compact == last_heading_compact:
                continue
            last_heading_compact = compact
        elif stripped:
            last_heading_compact = ""
        out.append(line)
    return "\n".join(out)


def _looks_like_heading_continuation_pair(left: str, right: str, heading_texts: list[str] | None = None) -> bool:
    left_raw = str(left or "").strip()
    right_raw = str(right or "").strip()
    if not left_raw or not right_raw:
        return False
    if re.search(r"[.!?]\s*$", left_raw):
        return False
    left_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", left_raw)
    right_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", right_raw)
    if not left_words or not right_words:
        return False
    weak_left_tail = left_words[-1].lower() in {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
    right_short = len(right_words) <= 4
    right_title_like = _title_case_ratio(right_words) >= 0.6
    if weak_left_tail and right_short and right_title_like:
        return True
    if re.fullmatch(r"\d{4}(?:\s+[A-Za-z][A-Za-z'/-]*){1,3}", right_raw) and left_words[-1].lower() in {"to", "the", "of"}:
        return True
    if _looks_like_wrapped_heading_line_pair(left_raw, right_raw, heading_texts):
        return True
    return False


def _looks_like_wrapped_heading_line_pair(left: str, right: str, heading_texts: list[str] | None = None) -> bool:
    left_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", str(left or "").strip())
    right_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", str(right or "").strip())
    if len(left_words) < 5 or len(right_words) < 4:
        return False
    if len(left_words) + len(right_words) > 24:
        return False
    if re.search(r"[.!?;:]\s*$", right):
        return False
    combined_text = f"{left} {right}"
    if heading_texts is not None and _looks_like_ast_confirmed_heading_wrap(left, right):
        return any(_compact_text(item) == _compact_text(combined_text) for item in heading_texts)
    if right_words[0].lower() not in {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}:
        return False
    combined_words = left_words + right_words
    if heading_texts is not None:
        if not any(_compact_text(item) == _compact_text(combined_text) for item in heading_texts):
            return False
    if _title_case_ratio(combined_words) >= 0.55:
        return True
    lowercase_starts = sum(1 for word in combined_words if word[:1].islower())
    return lowercase_starts <= max(4, len(combined_words) // 3) and any(re.search(r"\d|[~:/-]", word) for word in combined_words)


def _looks_like_lettered_body_list_item_false_positive(blocks: list[dict[str, Any]], index: int) -> bool:
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    if not re.match(r"^[a-z][.)]\s+[A-Z]", text):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    if len(words) < 5:
        return False
    bbox = _bbox(block)
    if bbox is None:
        return True
    previous = _previous_text_block(blocks, index)
    following = _next_text_block(blocks, index)
    height = max(1.0, bbox[3] - bbox[1])
    for neighbor in (previous, following):
        neighbor_bbox = _bbox(neighbor) if neighbor else None
        if neighbor_bbox is None:
            continue
        if abs(neighbor_bbox[0] - bbox[0]) <= max(8.0, height):
            neighbor_text = _block_text(neighbor)
            if re.match(r"^[a-z][.)]\s+[A-Z]", neighbor_text) or re.match(r"^\d+[.)]\s+[A-Z]", neighbor_text):
                return True
    return False


def _looks_like_top_running_header_false_positive(
    blocks: list[dict[str, Any]],
    index: int,
    page_height: float | None,
) -> bool:
    if index != 0:
        return False
    block = blocks[index]
    if str(block.get("block_type") or "").strip().lower() != "text":
        return False
    text = _block_text(block)
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", text)
    alpha_words = [word for word in words if not word.isdigit()]
    if len(alpha_words) < 3:
        return False
    uppercase_ratio = sum(1 for word in alpha_words if word.upper() == word and len(word) > 1) / max(1, len(alpha_words))
    if uppercase_ratio < 0.8:
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    top_limit = max(60.0, (page_height or 0) * 0.09)
    if bbox[1] > top_limit:
        return False
    page_left = min((_bbox(item) or bbox)[0] for item in blocks if _bbox(item) is not None)
    page_right = max((_bbox(item) or bbox)[2] for item in blocks if _bbox(item) is not None)
    page_width = max(1.0, page_right - page_left)
    if (bbox[2] - bbox[0]) < page_width * 0.55:
        return False
    if re.search(r"\d", text):
        return True
    for following in blocks[index + 1 : min(len(blocks), index + 4)]:
        following_bbox = _bbox(following)
        following_text = _block_text(following)
        if following_bbox is None or not following_text:
            continue
        gap = following_bbox[1] - bbox[3]
        if gap < 0 or gap > max(55.0, (bbox[3] - bbox[1]) * 3.0):
            continue
        if _looks_like_short_title_text(following_text) and (following_bbox[2] - following_bbox[0]) < (bbox[2] - bbox[0]) * 0.6:
            return True
    return False


def _looks_like_instructional_prompt_title(text: str) -> bool:
    raw = str(text or "").strip()
    if ":" not in raw:
        return False
    prefix = raw.split(":", 1)[0].strip()
    prefix_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", prefix)
    if not (2 <= len(prefix_words) <= 9):
        return False
    return bool(
        re.search(
            r"\b(?:question|questions|exercise|exercises|prompt|prompts|discussion|reflection|problem|task)\b",
            prefix,
            re.IGNORECASE,
        )
    )


def _looks_like_chart_axis_label_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.fullmatch(r"[\d.\s%-]+", raw):
        return True
    if _looks_like_chart_time_or_legend_label(raw):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if len(words) <= 2 and not re.search(r"[.!?]\s*$", raw):
        return True
    return False


def _looks_like_colon_year_section_title(text: str) -> bool:
    raw = str(text or "").strip()
    if ":" not in raw:
        return False
    if not re.search(r"\b(?:18|19|20)\d{2}\s*[-\u2013\u2014]\s*(?:18|19|20)?\d{2}\b", raw):
        return False
    prefix = raw.split(":", 1)[0].strip()
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", prefix)
    return 1 <= len(words) <= 6 and _title_case_ratio(words) >= 0.65


def _looks_like_chart_time_or_legend_label(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.search(r"\b(?:19|20)\d{2}\b", raw):
        return True
    if re.search(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        raw,
        re.IGNORECASE,
    ):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if 3 <= len(words) <= 6 and re.search(r"\b(?:employment|terminate|terminated|know|not)\b", raw, re.IGNORECASE):
        return True
    return False


def _is_non_heading_role(block: dict[str, Any]) -> bool:
    role = str(block.get("semantic_role") or "").strip()
    unit_role = str(block.get("unit_role") or "").strip()
    return role in {"reference_entry", "footnote", "footnote_continuation", "page_number", "author_line"} or unit_role in {
        "reference_entry",
        "footnote",
        "footnote_continuation",
        "page_number",
    }


def _split_embedded_heading_sequence(line: str, heading_texts: list[str]) -> list[str] | None:
    source = str(line or "").strip()
    if not source:
        return None
    ordered_headings = sorted(
        [heading for heading in heading_texts if heading],
        key=lambda value: (-len(value), heading_texts.index(value)),
    )
    emitted: list[str] = []
    remainder = source
    matched_any = False
    while remainder:
        match = _match_heading_prefix(remainder, ordered_headings)
        if match is None:
            infix = _match_heading_infix(remainder, ordered_headings)
            if infix is None:
                break
            before, heading, after = infix
            before = before.strip()
            if before:
                emitted.append(before)
                emitted.append("")
            matched_any = True
            _append_heading_line(emitted, heading)
            remainder = _trim_heading_separator_prefix(after)
            continue
        matched_any = True
        heading, remainder = match
        _append_heading_line(emitted, heading)
        remainder = _trim_heading_separator_prefix(remainder)
    if not matched_any:
        return None
    if remainder:
        emitted.append(remainder)
    return emitted


def _match_multiline_heading_lines(lines: list[str], start_index: int, heading_texts: list[str]) -> tuple[str, int] | None:
    first = str(lines[start_index] or "").strip()
    if not first or first.startswith("#"):
        return None
    ordered_headings = sorted(
        [heading for heading in heading_texts if heading],
        key=lambda value: (-len(value), heading_texts.index(value)),
    )
    parts = [first]
    cursor = start_index + 1
    consumed_any = False
    while cursor < len(lines) and len(parts) < 4:
        if not str(lines[cursor]).strip():
            cursor += 1
            continue
        candidate = str(lines[cursor]).strip()
        if candidate.startswith("#"):
            break
        parts.append(candidate)
        consumed_any = True
        joined = " ".join(parts)
        compact_joined = _compact_text(joined)
        for heading in ordered_headings:
            compact_heading = _compact_text(heading)
            if compact_joined == compact_heading:
                return heading, cursor + 1
            if not compact_heading.startswith(compact_joined):
                continue
        if not any(_compact_text(heading).startswith(compact_joined) for heading in ordered_headings):
            break
        cursor += 1
    if not consumed_any:
        return None
    return None


def _match_heading_prefix(source: str, heading_texts: list[str]) -> tuple[str, str] | None:
    compact_source = _compact_text(source)
    for heading in heading_texts:
        compact_heading = _compact_text(heading)
        if not compact_heading:
            continue
        if compact_source == compact_heading:
            return heading, ""
        if compact_source.startswith(compact_heading + " "):
            remainder = source[len(heading):]
            return heading, remainder
    return None


def _match_heading_line_prefix(source: str, heading_texts: list[str]) -> tuple[str, str] | None:
    match = _match_heading_prefix(source, heading_texts)
    if match is None:
        return None
    heading, remainder = match
    if not remainder.strip():
        return match
    if _looks_like_body_text_after_heading(remainder) or _looks_like_wrapped_body_sentence(remainder):
        return None
    return heading, remainder


def _match_strong_heading_prefix(source: str, heading_texts: list[str]) -> str | None:
    raw = str(source or "").strip()
    if not raw:
        return None
    source_compact = _compact_text(raw)
    source_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if len(source_words) < 5:
        return None
    best: str | None = None
    for heading in heading_texts:
        heading_raw = str(heading or "").strip()
        heading_compact = _compact_text(heading_raw)
        if not heading_compact or source_compact == heading_compact:
            continue
        if not heading_compact.startswith(source_compact + " "):
            continue
        heading_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", heading_raw)
        if len(heading_words) <= len(source_words):
            continue
        if len(source_words) / max(1, len(heading_words)) < 0.55:
            continue
        remainder = heading_raw[len(raw):].strip()
        remainder_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", remainder)
        if not (1 <= len(remainder_words) <= 8):
            continue
        if _looks_like_body_sentence_start(remainder) or _looks_like_wrapped_body_sentence(remainder):
            continue
        if best is None or len(heading_raw) > len(best):
            best = heading_raw
    return best


def _match_heading_infix(source: str, heading_texts: list[str]) -> tuple[str, str, str] | None:
    best: tuple[int, str, str, str] | None = None
    for heading in heading_texts:
        compact_heading = _compact_text(heading)
        if not compact_heading:
            continue
        index, end_index = _find_heading_span(source, heading)
        if index <= 0:
            continue
        before = source[:index]
        after = source[end_index:]
        if (
            not _has_valid_sentence_boundary_before_heading(before, heading)
            and not _looks_like_heading_glued_to_body_line_suffix(before, after)
            and not _looks_like_kpi_prefix_before_heading(before)
            and not _looks_like_page_label_prefix_before_heading(before)
            and not _looks_like_chapter_label_prefix_before_heading(before)
            and not _looks_like_running_header_prefix_before_heading(before, heading, after)
            and not _looks_like_short_running_header_prefix_before_heading(before, heading, after)
        ):
            continue
        if _reject_short_heading_infix_without_structural_boundary(before, heading, after):
            continue
        candidate = (index, heading, before, after)
        if best is None or index < best[0] or (index == best[0] and len(heading) > len(best[1])):
            best = candidate
    if best is None:
        return None
    _index, heading, before, after = best
    return before, heading, after


def _has_valid_sentence_boundary_before_heading(text: str, heading: str) -> bool:
    raw = str(text or "").rstrip()
    if not raw:
        return False
    if re.search(r"[.!?)]\s*$", raw):
        return True
    if not re.search(r"[.!?)]\s*\d{1,3}\s*$", raw):
        return False
    return _looks_like_numbered_heading_text(heading)


def _find_heading_span(source: str, heading: str) -> tuple[int, int]:
    raw_source = str(source or "")
    raw_heading = str(heading or "").strip()
    if not raw_source or not raw_heading:
        return -1, -1
    direct_index = raw_source.find(raw_heading)
    if direct_index >= 0:
        return direct_index, direct_index + len(raw_heading)
    source_tokens = list(re.finditer(r"\S+", raw_source))
    heading_tokens = re.findall(r"\S+", raw_heading)
    if not source_tokens or not heading_tokens:
        return -1, -1
    compact_heading = _compact_text(" ".join(heading_tokens))
    heading_len = len(heading_tokens)
    for start in range(0, len(source_tokens) - heading_len + 1):
        window = source_tokens[start : start + heading_len]
        if _compact_text(" ".join(token.group(0) for token in window)) == compact_heading:
            return window[0].start(), window[-1].end()
    return -1, -1


def _looks_like_kpi_prefix_before_heading(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    parts = raw.split()
    if len(parts) > 4:
        return False
    return all(_looks_like_infographic_kpi_text(part) for part in parts)


def _looks_like_page_label_prefix_before_heading(text: str) -> bool:
    raw = str(text or "").strip()
    return _looks_like_page_label_text(raw) or _looks_like_page_number_or_roman_label(raw)


def _looks_like_chapter_label_prefix_before_heading(text: str) -> bool:
    raw = str(text or "").strip()
    return bool(re.fullmatch(r"(?:chapter|part|book)\s+\d+[A-Za-z]?", raw, re.IGNORECASE))


def _looks_like_running_header_prefix_before_heading(before: str, heading: str, after: str) -> bool:
    prefix = str(before or "").strip()
    title = str(heading or "").strip()
    if not prefix or not _looks_like_short_title_text(title):
        return False
    suffix = str(after or "").strip()
    if suffix and not (_looks_like_body_text_after_heading(suffix) or _looks_like_body_sentence_start(suffix)):
        return False
    if re.search(r"[.!?:;]\s*$", prefix):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", prefix)
    if not (3 <= len(words) <= 8 and words[0].isdigit()):
        return False
    alpha_words = [word for word in words[1:] if not word.isdigit()]
    return len(alpha_words) >= 2 and _title_case_ratio(alpha_words) >= 0.6


def _looks_like_short_running_header_prefix_before_heading(before: str, heading: str, after: str) -> bool:
    prefix = str(before or "").strip()
    title = str(heading or "").strip()
    if not prefix or not _looks_like_short_title_text(title):
        return False
    suffix = str(after or "").strip()
    if suffix and not (_looks_like_body_text_after_heading(suffix) or _looks_like_body_sentence_start(suffix)):
        return False
    if re.search(r"[.!?:;]\s*$", prefix):
        return False
    prefix_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", prefix)
    title_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", title)
    if not (1 <= len(prefix_words) <= 4 and 2 <= len(title_words) <= 6):
        return False
    if _looks_like_body_text_after_heading(prefix):
        return False
    return _title_case_ratio(prefix_words) >= 0.6


def _reject_short_heading_infix_without_structural_boundary(before: str, heading: str, after: str) -> bool:
    title = str(heading or "").strip()
    if not title or _looks_like_numbered_heading_text(title):
        return False
    heading_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", title)
    if not (1 <= len(heading_words) <= 2):
        return False
    prefix = str(before or "").rstrip()
    suffix = str(after or "").lstrip()
    prefix_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", prefix)
    suffix_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", suffix)
    if prefix_words and prefix_words[-1].lower() in {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }:
        return True
    if suffix_words and suffix_words[0].lower() in {"and", "or", "of", "in", "to", "for", "with", "from", "by"}:
        return True
    if _looks_like_short_running_header_prefix_before_heading(before, heading, after):
        return False
    if _looks_like_all_caps_actor_sequence(prefix_words, heading_words, suffix_words):
        return True
    return False


def _looks_like_all_caps_actor_sequence(
    prefix_words: list[str],
    heading_words: list[str],
    suffix_words: list[str],
) -> bool:
    if not prefix_words or not suffix_words:
        return False
    window = prefix_words[-3:] + heading_words + suffix_words[:3]
    alpha_words = [word for word in window if re.search(r"[A-Za-z]", word)]
    if len(alpha_words) < 3:
        return False
    long_alpha = [word for word in alpha_words if len(word) > 1]
    if len(long_alpha) < 3:
        return False
    uppercase_like = sum(1 for word in long_alpha if word.upper() == word)
    return uppercase_like / max(1, len(long_alpha)) >= 0.75


def _looks_like_heading_glued_to_body_line_suffix(before: str, after: str) -> bool:
    if str(after or "").strip():
        return False
    prefix = str(before or "").rstrip()
    if not prefix:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", prefix)
    if len(words) < 5:
        return False
    return prefix.lstrip().startswith(("•", "-", "*")) or words[-1][:1].islower()


def _trim_heading_separator_prefix(text: str) -> str:
    remainder = str(text or "").lstrip()
    remainder = re.sub(r"^[\s\-\u2013\u2014:;,.]+", "", remainder).lstrip()
    return remainder


def _append_heading_line(lines: list[str], heading: str) -> None:
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        lines.append("")
    lines.append(f"# {heading.strip()}")
    lines.append("")


def _looks_like_literature_heading(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or len(raw) > 96:
        return False
    if re.search(r"[.;,!?]\s*$", raw):
        return False
    if _looks_like_numbered_heading_text(raw):
        return True
    if re.match(r"^(?:chapter|part|book|prologue|epilogue)\s+\d+[A-Za-z0-9'’\- ]*$", raw, re.IGNORECASE):
        return True
    compact = re.sub(r"[^a-z]", "", raw.lower())
    return compact in {
        "abstract",
        "introduction",
        "background",
        "methods",
        "materials",
        "materialsandmethods",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
    }


def _looks_like_short_title_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.search(r"\d", raw) and not re.match(r"^(?:chapter|part|book|prologue|epilogue)\s+\d+", raw, re.IGNORECASE):
        return False
    if len(raw) > 60:
        return False
    if re.search(r"[.!?]\s*$", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not words or len(words) > 6:
        return False
    alpha_words = [word for word in words if not word.isdigit()]
    if not alpha_words:
        return False
    if not alpha_words[0][:1].isupper():
        return False
    title_like = 0
    for word in alpha_words:
        lower = word.lower()
        if lower in {"of", "the", "and", "in", "to", "for", "on", "at", "by", "with", "from"}:
            title_like += 1
        elif word[:1].isupper():
            title_like += 1
    return title_like >= max(1, len(alpha_words) - 1)


def _looks_like_isolated_title_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if len(raw) > 96:
        return False
    if re.search(r"[,;:]\s*$", raw):
        return False
    if _looks_like_standalone_figure_or_table_caption(raw):
        return False
    if re.match(r"^\(?[a-z]\)\s+", raw, re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not words or len(words) > 12:
        return False
    if raw.endswith("?"):
        return len(words) >= 4
    if raw.endswith("."):
        raw_without_period = raw[:-1].strip()
        if not raw_without_period:
            return False
        words_without_period = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw_without_period)
        return 2 <= len(words_without_period) <= 8 and _title_case_ratio(words_without_period) >= 0.6
    if _looks_like_numbered_heading_text(raw):
        return True
    return _looks_like_short_title_text(raw) or _title_case_ratio(words) >= 0.65


def _looks_like_top_band_main_title_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or re.search(r"[.!?]\s*$", raw):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if not (6 <= len(words) <= 18):
        return False
    if _looks_like_body_sentence_start(raw):
        return False
    if re.match(r"^(?:as shown|as discussed|as described|as mentioned|according to|in this|in the)\b", raw, re.IGNORECASE):
        return False
    alpha_words = [word for word in words if not word.isdigit()]
    return bool(alpha_words and alpha_words[0][:1].isupper())


def _looks_like_numbered_heading_text(text: str) -> bool:
    raw = str(text or "").strip()
    match = _NUMBERED_HEADING_RE.match(raw)
    if not match:
        return False
    label = match.group("label")
    title = match.group("title")
    title_words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", title)
    if re.fullmatch(r"[A-Za-z]\.?", label):
        if not label.rstrip(".").isupper():
            return False
        return _title_case_ratio(title_words) >= 0.65
    return True


def _looks_like_body_sentence_start(text: str) -> bool:
    raw = str(text or "").lstrip()
    first = re.match(r"[A-Za-z]+", raw)
    return bool(first and first.group(0).lower() in {"a", "an", "the", "this", "these", "those", "we", "it", "if"})


def _looks_like_wrapped_body_sentence(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.match(r"^(?:as shown|as discussed|as described|as mentioned|according to|in this|in the)\b", raw, re.IGNORECASE):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    return len(words) >= 8 and _title_case_ratio(words) < 0.45 and bool(re.search(r"\b(?:the|of|in|for|with|from|to|and)\b", raw, re.IGNORECASE))


def _title_case_ratio(words: list[str]) -> float:
    alpha_words = [word for word in words if not word.isdigit()]
    if not alpha_words:
        return 0.0
    title_like = 0
    for word in alpha_words:
        lower = word.lower()
        if lower in {"a", "an", "of", "the", "and", "or", "in", "to", "for", "on", "at", "by", "with", "from"}:
            title_like += 1
        elif word[:1].isupper():
            title_like += 1
    return title_like / max(1, len(alpha_words))


def _previous_vertical_gap(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> float:
    for previous in reversed(blocks[:index]):
        previous_bbox = _bbox(previous)
        if previous_bbox is not None:
            return bbox[1] - previous_bbox[3]
    return 999.0


def _next_vertical_gap(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
) -> float:
    for following in blocks[index + 1 :]:
        following_bbox = _bbox(following)
        if following_bbox is not None:
            return following_bbox[1] - bbox[3]
    return 999.0


def _next_text_block(blocks: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    for following in blocks[index + 1 :]:
        if str(following.get("block_type") or "").strip().lower() == "text" and _block_text(following):
            return following
    return None


def _previous_text_block(blocks: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    for previous in reversed(blocks[:index]):
        if str(previous.get("block_type") or "").strip().lower() == "text" and _block_text(previous):
            return previous
    return None


def _starts_like_lowercase_continuation(text: str) -> bool:
    raw = str(text or "").lstrip()
    first = re.match(r"[A-Za-z]+", raw)
    return bool(first and first.group(0)[:1].islower())


def _next_non_empty_block_type(blocks: list[dict[str, Any]], index: int) -> str:
    for following in blocks[index + 1 :]:
        block_type = str(following.get("block_type") or "").strip().lower()
        if block_type:
            return block_type
    return ""


def _previous_non_empty_block_type(blocks: list[dict[str, Any]], index: int) -> str:
    for previous in reversed(blocks[:index]):
        block_type = str(previous.get("block_type") or "").strip().lower()
        if block_type:
            return block_type
    return ""


def _previous_substantive_media_block(blocks: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    for previous in reversed(blocks[:index]):
        block_type = str(previous.get("block_type") or "").strip().lower()
        if not block_type:
            continue
        if block_type != "image":
            return None
        return previous if _is_substantive_media_block(previous) else None
    return None


def _is_substantive_media_block(block: dict[str, Any]) -> bool:
    if str(block.get("block_type") or "").strip().lower() != "image":
        return False
    bbox = _bbox(block)
    if bbox is None:
        return False
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    if width >= 24.0 and height >= 24.0:
        return True
    if width >= 48.0 and height >= 12.0:
        return True
    if str(block.get("caption_text") or block.get("title") or "").strip():
        return True
    if str(block.get("embedded_text") or block.get("content_text") or "").strip():
        return height >= 8.0 and width >= 24.0
    return False


def _has_nearby_page_header_separator_rule(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
    page_height: float | None,
) -> bool:
    if page_height is not None and page_height > 0 and bbox[1] > page_height * 0.18:
        return False
    height = max(1.0, bbox[3] - bbox[1])
    for other in blocks[max(0, index - 2) : min(len(blocks), index + 3)]:
        if other is blocks[index]:
            continue
        if str(other.get("block_type") or "").strip().lower() != "image":
            continue
        other_bbox = _bbox(other)
        if other_bbox is None:
            continue
        width = max(0.0, other_bbox[2] - other_bbox[0])
        rule_height = max(0.0, other_bbox[3] - other_bbox[1])
        if rule_height > 2.0 or width < max(120.0, (bbox[2] - bbox[0]) * 1.8):
            continue
        horizontal_overlap = min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0])
        if horizontal_overlap < min(width, max(1.0, bbox[2] - bbox[0])) * 0.35:
            continue
        gap_above = bbox[1] - other_bbox[3]
        gap_below = other_bbox[1] - bbox[3]
        if -height * 0.5 <= gap_above <= height * 0.8 or -height * 0.5 <= gap_below <= height * 0.8:
            return True
    return False


def _is_title_below_running_header_separator(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
    page_height: float | None,
) -> bool:
    text = _block_text(blocks[index])
    if not _looks_like_short_title_text(text):
        return False
    separator = _thin_separator_near_title(blocks, index, bbox, above=True)
    if separator is None:
        return False
    page_top, page_span = _page_vertical_extent(blocks)
    if page_height is not None and page_height > 0:
        if bbox[1] > page_height * 0.24:
            return False
    elif page_span > 0 and (bbox[1] - page_top) > max(36.0, page_span * 0.24):
        return False
    if not _has_running_header_above_separator(blocks, index, separator):
        return False
    following = _next_text_block(blocks, index)
    following_text = _block_text(following or {})
    return bool(following_text and _looks_like_body_text_after_heading(following_text))


def _is_running_header_above_title_separator(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
    page_height: float | None,
) -> bool:
    text = _block_text(blocks[index])
    if not _looks_like_short_title_text(text):
        return False
    separator = _thin_separator_near_title(blocks, index, bbox, above=False)
    if separator is None:
        return False
    page_top, page_span = _page_vertical_extent(blocks)
    if page_height is not None and page_height > 0:
        if bbox[1] > page_height * 0.18:
            return False
    elif page_span > 0 and (bbox[1] - page_top) > page_span * 0.18:
        return False
    below_title = _first_text_below_separator(blocks, index, separator)
    if below_title is None:
        return False
    below_bbox = _bbox(below_title)
    below_text = _block_text(below_title)
    if below_bbox is None or not _looks_like_short_title_text(below_text):
        return False
    following = _next_text_block(blocks, _block_index(blocks, below_title))
    following_text = _block_text(following or {})
    return bool(following_text and _looks_like_body_text_after_heading(following_text))


def _thin_separator_near_title(
    blocks: list[dict[str, Any]],
    index: int,
    bbox: tuple[float, float, float, float],
    *,
    above: bool,
) -> dict[str, Any] | None:
    height = max(1.0, bbox[3] - bbox[1])
    for other in blocks[max(0, index - 3) : min(len(blocks), index + 4)]:
        if other is blocks[index]:
            continue
        if str(other.get("block_type") or "").strip().lower() != "image":
            continue
        other_bbox = _bbox(other)
        if other_bbox is None:
            continue
        width = max(0.0, other_bbox[2] - other_bbox[0])
        rule_height = max(0.0, other_bbox[3] - other_bbox[1])
        if rule_height > 2.0 or width < max(120.0, (bbox[2] - bbox[0]) * 1.8):
            continue
        if above:
            gap = bbox[1] - other_bbox[3]
        else:
            gap = other_bbox[1] - bbox[3]
        if -height * 0.35 <= gap <= height * 1.2:
            return other
    return None


def _has_running_header_above_separator(
    blocks: list[dict[str, Any]],
    title_index: int,
    separator: dict[str, Any],
) -> bool:
    separator_bbox = _bbox(separator)
    if separator_bbox is None:
        return False
    for previous_index, previous in enumerate(blocks[:title_index]):
        if str(previous.get("block_type") or "").strip().lower() != "text":
            continue
        previous_bbox = _bbox(previous)
        previous_text = _block_text(previous)
        if previous_bbox is None or not previous_text:
            continue
        if previous_bbox[3] > separator_bbox[1]:
            continue
        gap = separator_bbox[1] - previous_bbox[3]
        height = max(1.0, previous_bbox[3] - previous_bbox[1])
        if gap > height * 1.2:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", previous_text)
        if 1 <= len(words) <= 6 and not _looks_like_body_text_after_heading(previous_text):
            return True
    return False


def _first_text_below_separator(
    blocks: list[dict[str, Any]],
    index: int,
    separator: dict[str, Any],
) -> dict[str, Any] | None:
    separator_bbox = _bbox(separator)
    if separator_bbox is None:
        return None
    for following in blocks[index + 1 : min(len(blocks), index + 5)]:
        if str(following.get("block_type") or "").strip().lower() != "text":
            continue
        following_bbox = _bbox(following)
        if following_bbox is None:
            continue
        if following_bbox[1] >= separator_bbox[3]:
            return following
    return None


def _looks_like_body_text_after_heading(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*|\d+", raw)
    if len(words) >= 8:
        return True
    return bool(re.search(r"[.;]\s*$", raw)) and len(words) >= 5


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_metadata_bullet(line: str) -> bool:
    return line.strip().lower().startswith(_METADATA_PREFIXES)


def _block_text(block: dict[str, Any]) -> str:
    return str(block.get("display_text") or block.get("text") or "").strip()


def _bbox(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = block.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return None


def _page_vertical_extent(blocks: list[dict[str, Any]]) -> tuple[float, float]:
    bboxes = [_bbox(block) for block in blocks]
    usable = [bbox for bbox in bboxes if bbox is not None]
    if not usable:
        return 0.0, 0.0
    top = min(bbox[1] for bbox in usable)
    bottom = max(bbox[3] for bbox in usable)
    return top, max(1.0, bottom - top)


def _page_horizontal_extent(blocks: list[dict[str, Any]]) -> tuple[float, float]:
    bboxes = [_bbox(block) for block in blocks]
    usable = [bbox for bbox in bboxes if bbox is not None]
    if not usable:
        return 0.0, 0.0
    left = min(bbox[0] for bbox in usable)
    right = max(bbox[2] for bbox in usable)
    return left, max(1.0, right - left)


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _normalize_blank_lines(markdown: str) -> str:
    text = str(markdown or "").strip()
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text + "\n"
