from __future__ import annotations

import re
from typing import Any

from .shared import _clean_text

_BILINGUAL_CJK_ENGLISH_PAIR_RE = re.compile(
    r"(?P<cjk>[\u4e00-\u9fff]{2,12})\s+"
    r"(?P<english>[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,5})"
)


def _normalize_bilingual_english_gloss(value: str) -> str:
    words = [
        token.lower().strip("-")
        for token in re.findall(r"[A-Za-z][A-Za-z-]*", str(value or ""))
        if token.strip("-")
    ]
    return " ".join(words)


def _iter_bilingual_cjk_english_pairs(text: str) -> list[tuple[str, str]]:
    normalized = _clean_text(str(text or ""))
    if not normalized or not re.search(r"[\u4e00-\u9fff]", normalized) or not re.search(r"[A-Za-z]", normalized):
        return []

    pairs: list[tuple[str, str]] = []
    for match in _BILINGUAL_CJK_ENGLISH_PAIR_RE.finditer(normalized):
        cjk_text = str(match.group("cjk") or "")
        english_words = [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z-]*", str(match.group("english") or ""))
            if token.strip("-")
        ]
        if not cjk_text or not english_words:
            continue
        cjk_candidates = [
            cjk_text[-length:]
            for length in range(2, min(8, len(cjk_text)) + 1)
        ]
        english_candidates = [
            _normalize_bilingual_english_gloss(" ".join(english_words[:length]))
            for length in range(1, min(4, len(english_words)) + 1)
        ]
        for cjk_candidate in cjk_candidates:
            for english_candidate in english_candidates:
                if cjk_candidate and english_candidate:
                    pairs.append((cjk_candidate, english_candidate))
    return pairs


def _cjk_edit_distance(left: str, right: str) -> int:
    left_text = str(left or "")
    right_text = str(right or "")
    if left_text == right_text:
        return 0
    if not left_text:
        return len(right_text)
    if not right_text:
        return len(left_text)

    previous = list(range(len(right_text) + 1))
    for left_index, left_char in enumerate(left_text, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right_text, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (0 if left_char == right_char else 1),
                )
            )
        previous = current
    return previous[-1]


def _iter_state_bilingual_source_texts(state: Any) -> list[str]:
    texts: list[str] = []
    for page_payload in state.page_payloads:
        for block in page_payload.get("text_blocks", []) or []:
            text = _clean_text(str(block.get("text", "")))
            if text:
                texts.append(text)
        for table in page_payload.get("tables", []) or []:
            title = _clean_text(str(table.get("title", "")))
            if title:
                texts.append(title)
            for cell in table.get("header", []) or []:
                if isinstance(cell, dict):
                    cell_text = _clean_text(str(cell.get("text", "")))
                else:
                    cell_text = _clean_text(str(cell or ""))
                if cell_text:
                    texts.append(cell_text)
            for row in table.get("data_grid", []) or table.get("grid", []) or []:
                for cell in row:
                    cell_text = _clean_text(str(cell or ""))
                    if cell_text:
                        texts.append(cell_text)
    return texts


def _build_document_bilingual_ocr_term_corrections(state: Any) -> dict[str, dict[str, str]]:
    by_gloss: dict[str, dict[str, int]] = {}
    for text in _iter_state_bilingual_source_texts(state):
        for cjk_text, english_gloss in _iter_bilingual_cjk_english_pairs(text):
            by_gloss.setdefault(english_gloss, {})
            by_gloss[english_gloss][cjk_text] = by_gloss[english_gloss].get(cjk_text, 0) + 1

    corrections: dict[str, dict[str, str]] = {}
    for english_gloss, counts in by_gloss.items():
        if len(english_gloss.split()) < 2 or len(counts) < 2:
            continue
        counts_by_length: dict[int, dict[str, int]] = {}
        for cjk_text, count in counts.items():
            counts_by_length.setdefault(len(cjk_text), {})[cjk_text] = count
        for same_length_counts in counts_by_length.values():
            if len(same_length_counts) < 2:
                continue
            canonical, canonical_count = max(
                same_length_counts.items(),
                key=lambda item: (item[1], len(item[0]), item[0]),
            )
            if canonical_count < 2:
                continue
            for variant, variant_count in same_length_counts.items():
                if variant == canonical:
                    continue
                if variant_count >= canonical_count:
                    continue
                if canonical_count < variant_count + 2 and canonical_count < variant_count * 2:
                    continue
                if not (set(variant) & set(canonical)):
                    continue
                distance = _cjk_edit_distance(variant, canonical)
                max_distance = 1 if len(canonical) <= 2 else 2
                if 1 <= distance <= max_distance:
                    corrections.setdefault(english_gloss, {})[variant] = canonical
    return corrections


def _apply_document_bilingual_ocr_term_corrections(
    text_blocks: list[dict[str, Any]],
    corrections: dict[str, dict[str, str]],
) -> None:
    if not text_blocks or not corrections:
        return

    for block in text_blocks:
        if str(block.get("source", "") or "") != "body-ocr-repair":
            continue
        text = str(block.get("text", "") or "")
        if not text:
            continue
        repaired = text
        applied: list[dict[str, str]] = []
        for english_gloss, replacements in corrections.items():
            gloss_pattern = re.escape(english_gloss).replace(r"\ ", r"\s+")
            for variant, canonical in replacements.items():
                pattern = re.compile(
                    rf"(?<![\u4e00-\u9fff]){re.escape(variant)}(?=\s+{gloss_pattern}\b)",
                    re.IGNORECASE,
                )
                repaired, count = pattern.subn(canonical, repaired)
                if count:
                    applied.append(
                        {
                            "from": variant,
                            "to": canonical,
                            "english_gloss": english_gloss,
                        }
                    )
        if repaired != text:
            block["text"] = _clean_text(repaired)
            existing = list(block.get("ocr_term_corrections", []) or [])
            block["ocr_term_corrections"] = existing + applied
