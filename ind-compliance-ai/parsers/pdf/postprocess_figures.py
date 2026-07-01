from __future__ import annotations

import re

from .shared import _clean_text

_FIGURE_LABEL_RE = re.compile(
    r"^\s*(?:fig(?:ure)?\.?|图|圖|插图|圖表)\s*[0-9A-Za-z一二三四五六七八九十零〇]+(?:[.\-:：、]|\s)\b",
    re.IGNORECASE,
)
_FIGURE_BODY_REFERENCE_LABEL_RE = re.compile(
    r"^\s*figure\s*\d+\s+"
    r"(?:shows?|showed|shown|depicts?|depicted|illustrates?|illustrated|presents?|presented|"
    r"reports?|reported|indicates?|indicated|demonstrates?|demonstrated|compares?|compared|"
    r"summarizes?|summarized|describes?|described|gives?|gave|provides?|provided)\b",
    re.IGNORECASE,
)
_FIGURE_LEGEND_CUE_RE = re.compile(
    r"(?:\[参考\]|参考|p\s*[<≤]\s*0\.\d+|统计学显著|平均值|标准误差|自发性高血压|"
    r"组\）|组\)|n\s*=\s*\d+|"
    r"\b(?:legend|reference|statistically|significant|mean|standard\s+error)\b)",
    re.IGNORECASE,
)
_FIGURE_BODY_REFERENCE_CUE_RE = re.compile(
    r"(?:\b(?:as\s+shown\s+in|shown\s+in|see|as\s+illustrated\s+in|as\s+depicted\s+in)\s+"
    r"(?:fig(?:ure)?\.?\s*)\d+\b|"
    r"(?:\u5982\u4e0b|\u5982|\u89c1|\u53c2\u89c1)\s*(?:\u56fe|\u5716)\s*"
    r"[0-9A-Za-z\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u96f6]+"
    r"\s*(?:\u6240\u793a|\u663e\u793a))",
    re.IGNORECASE,
)
_FIGURE_REFERENCE_TOKEN_RE = re.compile(
    r"(?:\bfig(?:ure)?\.?\s*\d+\b|(?:\u56fe|\u5716)\s*"
    r"[0-9A-Za-z\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u96f6]+)",
    re.IGNORECASE,
)


def _looks_like_explicit_figure_caption_text(text: str) -> bool:
    cleaned = _clean_text(text)
    if not _FIGURE_LABEL_RE.match(cleaned):
        return False
    if _FIGURE_BODY_REFERENCE_LABEL_RE.match(cleaned):
        return False
    return True


def _looks_like_figure_body_reference_cue(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    if _FIGURE_BODY_REFERENCE_LABEL_RE.match(cleaned):
        return True
    return bool(_FIGURE_BODY_REFERENCE_CUE_RE.search(cleaned))


def _figure_reference_token_count(text: str) -> int:
    cleaned = _clean_text(text)
    if not cleaned:
        return 0
    return len(_FIGURE_REFERENCE_TOKEN_RE.findall(cleaned))


def _looks_like_contaminated_figure_caption_text(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned or not _looks_like_figure_body_reference_cue(cleaned):
        return False
    return not _looks_like_explicit_figure_caption_text(cleaned) or _figure_reference_token_count(cleaned) >= 2


def _looks_like_figure_legend_text(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    if _looks_like_explicit_figure_caption_text(cleaned):
        return True
    return bool(_FIGURE_LEGEND_CUE_RE.search(cleaned))
