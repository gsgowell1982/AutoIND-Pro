from __future__ import annotations

import re
from typing import Any

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_END_RE = re.compile(r"[\u4e00-\u9fff]$")
_CJK_START_RE = re.compile(r"^[\u4e00-\u9fff]")
_WORD_END_RE = re.compile(r"[A-Za-z0-9]$")
_WORD_START_RE = re.compile(r"^[A-Za-z0-9]")
_PLAIN_TOKEN_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9_.-]+$")
_PUNCTUATION_RE = re.compile(r"[\u3002\uff1b;:\uff1a\uff0c,\u3001.!?\uff01\uff1f\uff08\uff09()]")
_ASCII_RE = re.compile(r"[A-Za-z0-9]")
_DEPENDENT_PREFIX_RE = re.compile(
    r"^(?:和|及|与|或|的|之|及其|以及|并|且|用于|适用于|包括|其中|因为|如果|按照|根据)"
)
_DEPENDENT_SUFFIX_FRAGMENTS = {
    "申请",
    "类型",
    "描述",
    "资料",
    "文件",
    "内容",
    "要求",
    "规则",
}
_HEAD_NOUN_OR_VERB_FRAGMENTS = {
    "办理",
    "补充",
    "提交",
    "递交",
    "申报",
    "登记",
    "审批",
    "审评",
    "评价",
    "检查",
    "验证",
    "确认",
    "说明",
    "通知",
}

_STANDALONE_ENUM_TOKENS = {
    "回复",
    "撤回",
    "报告",
    "补充",
    "替换",
    "删除",
    "新增",
    "更新",
    "首次提交",
    "再次提交",
    "初始提交",
    "再注册",
    "新适应症",
    "联合用药",
    "批准",
    "不批准",
    "通过",
    "不通过",
    "适用",
    "不适用",
}


def project_table_cell_display_text(value: Any) -> Any:
    """Project raw table-cell text into reviewer-facing display/data text.

    Raw table evidence may contain in-cell newlines from PDF visual wrapping.
    Keep raw grids unchanged and apply this projection only to consumer-facing
    table views such as display_grid/data_grid.
    """
    if value is None or not isinstance(value, str):
        return value
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" not in text:
        return project_pdf_math_symbol_display_text(text)

    parts = [part.strip() for part in text.split("\n") if part.strip()]
    if not parts:
        return ""

    enum_parts: list[str] = []
    for part in parts:
        enum_parts.extend(_split_cell_enumeration(part))
    if _looks_like_short_cell_enumeration(enum_parts):
        return project_pdf_math_symbol_display_text(" / ".join(enum_parts))
    return project_pdf_math_symbol_display_text(_join_continuation_parts(parts))


def project_table_grid_display_text(grid: list[list[Any]] | None) -> list[list[Any]]:
    return [
        [project_table_cell_display_text(cell) for cell in row]
        for row in (grid or [])
    ]


def project_pdf_math_symbol_display_text(value: Any) -> Any:
    """Normalize common PDF math-font decode artifacts for display text.

    Some embedded symbolic fonts expose plus/equal/minus signs as legacy glyph
    codes through PyMuPDF text extraction. Keep raw evidence unchanged and only
    repair high-confidence mathematical contexts in display/data projections.
    """
    if value is None or not isinstance(value, str):
        return value
    text = value.replace("\x00", " ").replace("\x03", "-")
    if not text.strip():
        return text

    text = re.sub(r"(?<=\()[þ镁](?=\))", "+", text)
    text = re.sub(r"(?<=\()[¼录](?=\))", "=", text)
    text = re.sub(r"(?<=[A-Za-z0-9)\]])[þ镁](?=[A-Za-z0-9(\[])", "+", text)
    text = re.sub(r"(?<=[A-Za-z0-9)\]])[¼录](?=[A-Za-z0-9(\[])", "=", text)
    text = re.sub(r"(?i)(\d(?:\.\d+)?e)\s+(\d{1,3})(?=\b)", r"\1-\2", text)

    has_statistical_context = bool(
        re.search(r"\d(?:\.\d+)?e-\d+\b", text, re.IGNORECASE)
        or re.search(r"\b\d*\.\d+\b", text)
        or re.search(r"\([+=-]\)", text)
    )
    if has_statistical_context:
        text = re.sub(r"\(\s+\)(?=\s*\d)", "(-)", text)
        text = re.sub(r"(?<=\d)\(\s+\)", "(-)", text)
    return text


def _split_cell_enumeration(text: str) -> list[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    parts = [part.strip() for part in re.split(r"\s*(?:/|\u3001)\s*", normalized) if part.strip()]
    return parts or [normalized]


def _looks_like_short_cell_enumeration(parts: list[str]) -> bool:
    if len(parts) < 2 or len(parts) > 8:
        return False
    if not all(_CJK_RE.search(part) for part in parts):
        return False
    if any(len(part) > 18 for part in parts):
        return False
    if any(_PUNCTUATION_RE.search(part) for part in parts):
        return False
    if not _has_clear_parallel_item_evidence(parts):
        return False
    continuity_score = _score_continuous_cell_lines(parts)
    independence_score = _score_independent_cell_lines(parts)
    return independence_score >= continuity_score + 3


def _score_continuous_cell_lines(parts: list[str]) -> int:
    score = 0
    if len(parts) == 2 and _looks_like_suffix_completion(parts[0], parts[1]):
        score += 5
    if len(parts) == 2 and _looks_like_modifier_head_completion(parts[0], parts[1]):
        score += 5
    if any(_looks_like_dependent_fragment(part) for part in parts[1:]):
        score += 5
    if any(len(part) == 1 and _CJK_RE.fullmatch(part) for part in parts[1:]):
        score += 4
    if any(len(part) >= 12 for part in parts):
        score += 3
    if any(re.search(r"(?:因为|如果|其中|包括|按照|根据|适用于|用于)", part) for part in parts[1:]):
        score += 3
    if _looks_like_split_cjk_term(parts):
        score += 3
    return score


def _score_independent_cell_lines(parts: list[str]) -> int:
    score = 0
    if all(_looks_like_complete_short_item(part) for part in parts):
        score += 4
    if len(parts) >= 3:
        score += 2
    known_count = sum(1 for part in parts if _looks_like_standalone_short_enum_token(part))
    score += min(known_count, 3)
    if known_count >= 2:
        score += 2
    return score


def _has_clear_parallel_item_evidence(parts: list[str]) -> bool:
    if len(parts) < 2:
        return False
    known_count = sum(1 for part in parts if _looks_like_standalone_short_enum_token(part))
    if known_count >= 2:
        return True
    if len(parts) >= 3 and known_count >= 1 and all(_looks_like_complete_short_item(part) for part in parts):
        return True
    if len(parts) == 2 and all(_looks_like_complete_short_item(part) for part in parts):
        return _looks_like_parallel_short_term_pair(parts[0], parts[1])
    return False


def _looks_like_parallel_short_term_pair(left: str, right: str) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    if any(char.isspace() for char in left_text + right_text):
        return False
    suffix = _common_suffix(left_text, right_text)
    if not suffix:
        return False
    if suffix in {"性"} and len(left_text) <= 4 and len(right_text) <= 4:
        return True
    return len(suffix) >= 2 and min(len(left_text), len(right_text)) > len(suffix)


def _common_suffix(left: str, right: str) -> str:
    chars: list[str] = []
    for left_char, right_char in zip(reversed(left), reversed(right)):
        if left_char != right_char:
            break
        chars.append(left_char)
    return "".join(reversed(chars))


def _looks_like_continuous_cjk_phrase(parts: list[str]) -> bool:
    if len(parts) != 2:
        return False
    if not all(_PLAIN_TOKEN_RE.fullmatch(part or "") for part in parts):
        return False
    if any(_looks_like_standalone_short_enum_token(part) for part in parts):
        return False
    left, right = parts
    return len(left) <= 6 and len(right) <= 6


def _looks_like_dependent_fragment(part: str) -> bool:
    token = str(part or "").strip()
    if not token:
        return False
    if _DEPENDENT_PREFIX_RE.search(token):
        return True
    if len(token) == 1 and _CJK_RE.fullmatch(token):
        return True
    return False


def _looks_like_suffix_completion(left: str, right: str) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or right_text not in _DEPENDENT_SUFFIX_FRAGMENTS:
        return False
    if not _CJK_RE.search(left_text):
        return False
    if _looks_like_standalone_short_enum_token(left_text) or _looks_like_standalone_short_enum_token(right_text):
        return False
    return True


def _looks_like_modifier_head_completion(left: str, right: str) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    if not all(_PLAIN_TOKEN_RE.fullmatch(part) for part in (left_text, right_text)):
        return False
    if len(left_text) > 6 or len(right_text) > 6:
        return False
    if right_text in _HEAD_NOUN_OR_VERB_FRAGMENTS:
        return True
    return False


def _looks_like_complete_short_item(part: str) -> bool:
    token = str(part or "").strip()
    if not token or len(token) > 8:
        return False
    if _looks_like_dependent_fragment(token):
        return False
    if not _PLAIN_TOKEN_RE.fullmatch(token):
        return False
    if _ASCII_RE.search(token) and not _CJK_RE.search(token):
        return False
    return True


def _looks_like_split_cjk_term(parts: list[str]) -> bool:
    if len(parts) < 2:
        return False
    return any(_looks_like_dependent_fragment(part) for part in parts[1:])


def _looks_like_standalone_short_enum_token(part: str) -> bool:
    token = str(part or "").strip()
    if not token or len(token) > 8:
        return False
    if token in _STANDALONE_ENUM_TOKENS:
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,8}", token))


def _join_continuation_parts(parts: list[str]) -> str:
    merged = ""
    for part in parts:
        if not merged:
            merged = part
            continue
        if _should_join_without_space(merged, part):
            merged = f"{merged}{part}"
        else:
            merged = f"{merged} {part}"
    return merged


def _should_join_without_space(left: str, right: str) -> bool:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left or not right:
        return False
    if _CJK_END_RE.search(left) and (_CJK_START_RE.search(right) or _WORD_START_RE.search(right)):
        return True
    if _WORD_END_RE.search(left) and _CJK_START_RE.search(right):
        return True
    return False
