from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import fitz


_VALIDATION_STANDARD_FILENAME = bytes(
    "eCTD\\u9a8c\\u8bc1\\u6807\\u51c6.pdf",
    "ascii",
).decode("unicode_escape")
_ITEM_NUMBER_PATTERN = re.compile(r"^(?P<number>\d+(?:\.\d+)+)$")
_CHAPTER_FOOTER_PATTERN = re.compile(r"^(?P<number>\d)\s*-\s*(?P<title>.+)$")
_SKIP_LINES = {"序号", "描述", "说明", "严重程度"}
_SEVERITY_VALUES = ("错误", "警告", "提示信息")
_TITLE_OVERRIDES = {
    "6.22": "PDF应该设置启用“快速Web访问（Fast Web Access）”",
}
_FULL_TITLE_ONLY_ITEMS = {
    "3.7",
    "3.8",
    "3.9",
    "4.1.8",
    "4.1.9",
    "4.1.10",
}
_TAIL_STOP_MARKERS = {
    "必须遵守的关键验证标准",
    "建议遵守的验证标准",
    "用于收集信息的验证标准",
    "说明:",
}


@dataclass(slots=True)
class ValidationReviewItem:
    clause_id: str
    chapter_no: int
    chapter_heading: str
    item_no: str
    title: str
    severity: str
    detail_lines: list[str]
    normalized_text: str
    structured_list_lines: list[str]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _load_pdf_lines(path: Path) -> list[str]:
    doc = fitz.open(path)
    lines: list[str] = []
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            for raw_line in page.get_text("text").splitlines():
                line = _normalize_text(raw_line)
                if line:
                    lines.append(line)
    finally:
        doc.close()
    return lines


def _load_validation_standard_payload(normalized_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(
        (normalized_root / "cn_ectd_validation_standard.document.json").read_text(encoding="utf-8")
    )
    clauses = json.loads(
        (normalized_root / "cn_ectd_validation_standard.clauses.json").read_text(encoding="utf-8")
    )
    return document, clauses


def _extract_chapter_heading_map(lines: list[str]) -> dict[str, str]:
    chapter_title_map: dict[str, str] = {}
    for line in lines:
        match = _CHAPTER_FOOTER_PATTERN.match(line)
        if not match:
            continue
        title = _normalize_text(match.group("title"))
        if title.isdigit():
            continue
        chapter_title_map[_normalize_text(match.group("number"))] = title
    return chapter_title_map


def _extract_review_items(lines: list[str], clauses_payload: dict[str, Any]) -> list[ValidationReviewItem]:
    clauses = list(clauses_payload.get("clauses") or [])
    normalized_by_heading = {
        str(item.get("heading") or "").strip(): str(item.get("normalized_text") or "").strip()
        for item in clauses
    }
    normalized_by_item_no = {
        str(item.get("heading") or "").strip().split(" ", 1)[0]: str(item.get("normalized_text") or "").strip()
        for item in clauses
    }
    title_by_item_no = {
        str(item.get("heading") or "").strip().split(" ", 1)[0]: (
            str(item.get("heading") or "").strip().split(" ", 1)[1]
            if " " in str(item.get("heading") or "").strip()
            else ""
        )
        for item in clauses
    }
    clause_id_by_heading = {
        str(item.get("heading") or "").strip(): str(item.get("clause_id") or "").strip()
        for item in clauses
    }
    clause_id_by_item_no = {
        str(item.get("heading") or "").strip().split(" ", 1)[0]: str(item.get("clause_id") or "").strip()
        for item in clauses
    }

    chapter_title_map = _extract_chapter_heading_map(lines)
    items: list[ValidationReviewItem] = []
    index = 0

    while index < len(lines):
        number_match = _ITEM_NUMBER_PATTERN.match(lines[index])
        if not number_match:
            index += 1
            continue

        item_no = _normalize_text(number_match.group("number"))
        chapter_no = int(item_no.split(".", 1)[0])
        chapter_heading = f"{chapter_no}. {chapter_title_map.get(str(chapter_no), '')}".strip()

        title = ""
        title_index = index + 1
        while title_index < len(lines):
            candidate = lines[title_index]
            if candidate in _SKIP_LINES:
                title_index += 1
                continue
            if candidate.startswith("第 ") or candidate.startswith("eCTD验证标准"):
                title_index += 1
                continue
            if _CHAPTER_FOOTER_PATTERN.match(candidate) or _ITEM_NUMBER_PATTERN.match(candidate):
                break
            title = candidate
            break

        if not title:
            index += 1
            continue

        expected_title = _TITLE_OVERRIDES.get(item_no) or title_by_item_no.get(item_no, "")
        prefixed_detail = ""
        if expected_title and item_no in _FULL_TITLE_ONLY_ITEMS:
            title = expected_title
        elif expected_title and expected_title.startswith(title):
            title = expected_title
        elif expected_title and title.startswith(expected_title) and title != expected_title:
            prefixed_detail = title[len(expected_title) :].strip()
            title = expected_title
        elif "”" in title:
            quote_index = title.find("”")
            tail = title[quote_index + 1 :].strip()
            if tail and ("。" in tail or "不能" in tail or "必须" in tail or "不允许" in tail):
                prefixed_detail = tail
                title = title[: quote_index + 1].strip()

        detail_lines: list[str] = []
        severity = ""
        if prefixed_detail:
            detail_lines.append(prefixed_detail)
        body_index = title_index + 1
        while body_index < len(lines):
            candidate = lines[body_index]
            if _ITEM_NUMBER_PATTERN.match(candidate):
                break
            if candidate in _SKIP_LINES:
                body_index += 1
                continue
            if candidate.startswith("第 ") or candidate.startswith("eCTD验证标准"):
                body_index += 1
                continue
            if _CHAPTER_FOOTER_PATTERN.match(candidate):
                body_index += 1
                continue
            if candidate in _TAIL_STOP_MARKERS:
                break
            if candidate in _SEVERITY_VALUES:
                severity = candidate
                body_index += 1
                break
            detail_lines.append(candidate)
            body_index += 1

        heading = f"{item_no} {title}"
        structured_list_lines: list[str] = []
        marker_index = -1
        for i, candidate in enumerate(detail_lines):
            if candidate == "标准字体列表如下：":
                marker_index = i
                break
            if candidate.endswith("标准字体列") and i + 1 < len(detail_lines) and detail_lines[i + 1] == "表如下：":
                marker_index = i + 1
                break
        if marker_index >= 0:
            list_index = marker_index + 1
            while list_index < len(detail_lines):
                candidate = detail_lines[list_index]
                if candidate.startswith("参考文献（"):
                    break
                structured_list_lines.append(candidate)
                list_index += 1

        items.append(
            ValidationReviewItem(
                clause_id=clause_id_by_heading.get(heading, "") or clause_id_by_item_no.get(item_no, ""),
                chapter_no=chapter_no,
                chapter_heading=chapter_heading,
                item_no=item_no,
                title=title,
                severity=severity,
                detail_lines=detail_lines,
                normalized_text=normalized_by_heading.get(heading, "") or normalized_by_item_no.get(item_no, ""),
                structured_list_lines=structured_list_lines,
            )
        )
        index = body_index

    return items


def build_validation_standard_review_markdown(
    regulations_root: Path,
    normalized_root: Path,
) -> str:
    regulations_root = Path(regulations_root)
    normalized_root = Path(normalized_root)
    source_path = regulations_root / _VALIDATION_STANDARD_FILENAME
    lines = _load_pdf_lines(source_path)
    document_payload, clauses_payload = _load_validation_standard_payload(normalized_root)
    review_items = _extract_review_items(lines, clauses_payload)
    chapters = list(document_payload.get("chapters") or [])

    output: list[str] = []
    output.append("# eCTD验证标准 提取校对稿")
    output.append("")
    output.append(f"- 来源文件: `{document_payload['regulation']['source_filename']}`")
    output.append(f"- regulation_id: `{document_payload['regulation']['regulation_id']}`")
    output.append(f"- 提取章节数: `{document_payload['regulation']['chapter_count']}`")
    output.append(f"- 提取验证项数: `{document_payload['regulation']['article_count']}`")
    output.append("")
    output.append("## 章节目录")
    output.append("")
    for chapter in chapters:
        output.append(f"- {chapter['heading']}")
    output.append("")

    current_chapter_no = None
    for item in review_items:
        if item.chapter_no != current_chapter_no:
            current_chapter_no = item.chapter_no
            output.append(f"## {item.chapter_heading}")
            output.append("")

        output.append(f"### {item.item_no} {item.title}")
        output.append("")
        output.append(f"- `clause_id`: `{item.clause_id}`")
        output.append(f"- `chapter_no`: `{item.chapter_no}`")
        output.append(f"- `source_heading`: `{item.chapter_heading}`")
        output.append(f"- `严重程度`: `{item.severity or '未识别'}`")
        output.append("- `原始说明行`:")
        output.append("```text")
        if item.detail_lines:
            output.extend(item.detail_lines)
        output.append("```")
        if item.structured_list_lines:
            output.append("- `结构化列表项`:")
            output.append("```text")
            output.extend(item.structured_list_lines)
            output.append("```")
            output.append(
                "- `结构化单行比对串`: "
                + " | ".join(item.structured_list_lines)
            )
        output.append(f"- `规范化全文`: {item.normalized_text}")
        output.append("")

    return "\n".join(output)


def write_validation_standard_review_markdown(
    regulations_root: Path,
    normalized_root: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    regulations_root = Path(regulations_root)
    normalized_root = Path(normalized_root)
    if output_path is None:
        output_path = normalized_root / "cn_ectd_validation_standard.extracted_review.md"
    output_path = Path(output_path)
    output_path.write_text(
        build_validation_standard_review_markdown(regulations_root, normalized_root),
        encoding="utf-8",
    )
    return output_path
