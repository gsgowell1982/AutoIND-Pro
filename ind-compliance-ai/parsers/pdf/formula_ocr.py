from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    import pymupdf
except ImportError:  # pragma: no cover - optional runtime dependency
    pymupdf = None  # type: ignore[assignment]


_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_FLATTENED_TOKEN_RE = re.compile(r"\b(?:sum|frac|sqrt|begin|end)\b", re.IGNORECASE)
_MATH_SYMBOL_RE = re.compile(r"[=+\-*/^_{}\\]|\\(?:sum|frac|sqrt|begin|end|left|right|mathbf|mathrm)")
_TEXT_MATH_SYMBOL_RE = re.compile(r"[=∑Σ∏√≤≥<>+\-*/]")


@dataclass(frozen=True, slots=True)
class FormulaOcrCandidate:
    latex: str
    confidence: float = 0.0
    source: str = "formula_ocr"


def formula_ocr_enabled() -> bool:
    return str(os.environ.get("IND_FORMULA_OCR_ENABLED", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def texteller_ocr_enabled() -> bool:
    return str(os.environ.get("IND_FORMULA_OCR_BACKEND", "")).strip().lower() in {
        "texteller",
        "texteller_cli",
    }


def paddle_formula_ocr_enabled() -> bool:
    return str(os.environ.get("IND_FORMULA_OCR_BACKEND", "")).strip().lower() in {
        "paddle",
        "paddle_formula",
        "pp_formulanet",
        "pp_formulanet_plus_m",
    }


def build_formula_latex_fallback(source: str = "none") -> dict[str, Any]:
    return {
        "latex_text": None,
        "latex_confidence": 0.0,
        "latex_source": source,
        "latex_render_policy": "image_primary_latex_enhancement",
        "latex_validation": {
            "accepted": False,
            "score": 0.0,
            "signals": [],
            "issues": ["no_latex_candidate"],
        },
    }


def build_formula_latex_enhancement(
    equation_block: dict[str, Any],
    candidate: FormulaOcrCandidate | None,
) -> dict[str, Any]:
    if candidate is None or not str(candidate.latex or "").strip():
        return build_formula_latex_fallback("none")

    latex = _normalize_latex(str(candidate.latex or ""))
    latex, tag_signals = _normalize_latex_tag_from_equation_label(
        latex,
        str(equation_block.get("equation_label", "") or ""),
    )
    latex, normalization_signals = _trim_repetitive_latex_tail(latex)
    validation = _validate_latex_candidate(
        equation_text=str(equation_block.get("text", "") or ""),
        latex=latex,
        ocr_confidence=float(candidate.confidence or 0.0),
        normalization_signals=[*tag_signals, *normalization_signals],
    )
    accepted = bool(validation["accepted"])
    return {
        "latex_text": latex if accepted else None,
        "latex_confidence": float(validation["score"]) if accepted else 0.0,
        "latex_source": str(candidate.source or "formula_ocr").strip() or "formula_ocr",
        "latex_candidate_text": latex,
        "latex_render_policy": "image_primary_latex_enhancement",
        "latex_validation": validation,
    }


def build_inline_formula_latex_enhancement(
    inline_span: dict[str, Any],
    candidate: FormulaOcrCandidate | None,
) -> dict[str, Any]:
    equation_like = {
        "text": str(inline_span.get("content", "") or ""),
        "equation_label": "",
    }
    enhancement = build_formula_latex_enhancement(equation_like, candidate)
    enhancement["latex_render_policy"] = "text_primary_latex_enhancement"
    return enhancement


def enhance_equation_blocks_with_formula_ocr(
    path: Path,
    equation_blocks: list[dict[str, Any]],
) -> None:
    if not formula_ocr_enabled():
        for equation in equation_blocks:
            _apply_formula_latex_enhancement(equation, build_formula_latex_fallback("disabled"))
        return
    if not equation_blocks:
        return

    for equation in equation_blocks:
        candidate = _recognize_equation_latex(path, equation)
        enhancement = build_formula_latex_enhancement(equation, candidate)
        _apply_formula_latex_enhancement(equation, enhancement)


def enhance_inline_formula_spans_with_formula_ocr(
    path: Path | str,
    content_evidence: list[dict[str, Any]],
    enhancement_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> None:
    if not content_evidence:
        return
    if not formula_ocr_enabled():
        return

    enhancement_cache = enhancement_cache if enhancement_cache is not None else {}
    budget = _InlineFormulaOcrBudget.from_env()
    for evidence in content_evidence:
        page_number = int(evidence.get("page", 0) or 0)
        block_attempts = 0
        for span in _iter_inline_formula_span_dicts(evidence):
            if not _inline_span_is_formula_ocr_candidate(span):
                continue
            cache_key = _inline_formula_span_cache_key(page_number, span)
            enhancement = enhancement_cache.get(cache_key)
            if enhancement is None:
                skip_reason = budget.skip_reason(page_number, block_attempts)
                if skip_reason:
                    _mark_inline_formula_latex_skipped(span, skip_reason)
                    continue
                candidate = _recognize_inline_formula_latex(Path(path), page_number, span)
                budget.record_attempt(page_number)
                block_attempts += 1
                enhancement = build_inline_formula_latex_enhancement(span, candidate)
                enhancement_cache[cache_key] = enhancement
            _apply_inline_formula_latex_enhancement(span, enhancement)


def _apply_formula_latex_enhancement(equation_block: dict[str, Any], enhancement: dict[str, Any]) -> None:
    equation_block["latex_text"] = enhancement.get("latex_text")
    equation_block["latex_confidence"] = float(enhancement.get("latex_confidence", 0.0) or 0.0)
    equation_block["latex_source"] = enhancement.get("latex_source")
    if enhancement.get("latex_candidate_text"):
        equation_block["latex_candidate_text"] = enhancement.get("latex_candidate_text")
    equation_block["latex_render_policy"] = "image_primary_latex_enhancement"
    equation_block["latex_validation"] = dict(enhancement.get("latex_validation", {}) or {})


def _apply_inline_formula_latex_enhancement(inline_span: dict[str, Any], enhancement: dict[str, Any]) -> None:
    inline_span["latex_text"] = enhancement.get("latex_text")
    inline_span["latex_confidence"] = float(enhancement.get("latex_confidence", 0.0) or 0.0)
    inline_span["latex_source"] = enhancement.get("latex_source")
    if enhancement.get("latex_candidate_text"):
        inline_span["latex_candidate_text"] = enhancement.get("latex_candidate_text")
    inline_span["latex_render_policy"] = "text_primary_latex_enhancement"
    inline_span["latex_validation"] = dict(enhancement.get("latex_validation", {}) or {})


def _mark_inline_formula_latex_skipped(inline_span: dict[str, Any], reason: str) -> None:
    inline_span["latex_skip_reason"] = reason
    inline_span["latex_source"] = "formula_ocr_budget"
    inline_span["latex_render_policy"] = "text_primary_latex_enhancement"
    inline_span["latex_validation"] = {
        "accepted": False,
        "score": 0.0,
        "signals": [],
        "issues": [reason],
    }


@dataclass(slots=True)
class _InlineFormulaOcrBudget:
    max_document_spans: int
    max_page_spans: int
    max_block_spans: int
    document_attempts: int
    page_attempts: dict[int, int]

    @classmethod
    def from_env(cls) -> "_InlineFormulaOcrBudget":
        return cls(
            max_document_spans=_env_nonnegative_int("IND_FORMULA_OCR_MAX_INLINE_SPANS", 24),
            max_page_spans=_env_nonnegative_int("IND_FORMULA_OCR_MAX_INLINE_SPANS_PER_PAGE", 6),
            max_block_spans=_env_nonnegative_int("IND_FORMULA_OCR_MAX_INLINE_SPANS_PER_BLOCK", 2),
            document_attempts=0,
            page_attempts={},
        )

    def skip_reason(self, page_number: int, block_attempts: int) -> str | None:
        if self.max_document_spans <= 0:
            return "inline_formula_ocr_document_budget_exceeded"
        if self.max_page_spans <= 0:
            return "inline_formula_ocr_page_budget_exceeded"
        if self.max_block_spans <= 0:
            return "inline_formula_ocr_block_budget_exceeded"
        if self.document_attempts >= self.max_document_spans:
            return "inline_formula_ocr_document_budget_exceeded"
        if self.page_attempts.get(page_number, 0) >= self.max_page_spans:
            return "inline_formula_ocr_page_budget_exceeded"
        if block_attempts >= self.max_block_spans:
            return "inline_formula_ocr_block_budget_exceeded"
        return None

    def record_attempt(self, page_number: int) -> None:
        self.document_attempts += 1
        self.page_attempts[page_number] = self.page_attempts.get(page_number, 0) + 1


def _env_nonnegative_int(name: str, default: int) -> int:
    raw_value = str(os.environ.get(name, "") or "").strip()
    if not raw_value:
        return max(0, int(default))
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _iter_inline_formula_span_dicts(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for span in evidence.get("inline_formula_spans", []) or []:
        if isinstance(span, dict):
            spans.append(span)
    for segment in evidence.get("segments", []) or []:
        if not isinstance(segment, dict):
            continue
        for span in segment.get("inline_formula_spans", []) or []:
            if isinstance(span, dict):
                spans.append(span)
    return spans


def _inline_span_is_formula_ocr_candidate(span: dict[str, Any]) -> bool:
    if not bool(span.get("ocr_candidate")):
        return False
    if str(span.get("formula_complexity", "") or "") != "inline_formula":
        return False
    bbox = list(span.get("bbox", []) or [])
    if len(bbox) != 4:
        return False
    if not str(span.get("content", "") or "").strip():
        return False
    return True


def _inline_formula_span_cache_key(page_number: int, span: dict[str, Any]) -> tuple[Any, ...]:
    bbox = list(span.get("bbox", []) or [])
    rounded_bbox = tuple(round(float(value), 2) for value in bbox[:4])
    content = re.sub(r"\s+", " ", str(span.get("content", "") or "")).strip()
    return (page_number, rounded_bbox, content)


def _normalize_latex(latex: str) -> str:
    text = re.sub(r"\s+", " ", str(latex or "")).strip()
    text = text.strip("$ ")
    text = _normalize_spaced_latex_operator_names(text)
    text = re.sub(r"(?:\\qquad\s*){3,}", r"\\qquad\\qquad ", text)
    text = re.sub(r"(?:\\quad\s*){4,}", r"\\quad\\quad ", text)
    text = re.sub(r"\\eqno\s*\(?\s*([0-9]+)\s*\)?", r"\\tag{\1}", text)
    text = re.sub(r"\\tag\{\s*\)?\s*([0-9]+)\s*\}", r"\\tag{\1}", text)
    text = re.sub(r"(?<![A-Za-z0-9\\])\(\s*([0-9]{1,3})\s*\)\s*$", r"\\tag{\1}", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_latex_tag_from_equation_label(latex: str, equation_label: str) -> tuple[str, list[str]]:
    text = str(latex or "")
    match = re.fullmatch(r"\(\s*(\d{1,3})\s*\)", str(equation_label or "").strip())
    if not text or not match:
        return text, []
    expected_tag = str(match.group(1) or "").strip()
    if not expected_tag:
        return text, []
    tag_pattern = re.compile(r"\\tag\{\s*\d{1,3}\s*\}")
    tag_match = tag_pattern.search(text)
    if not tag_match:
        return text, []
    normalized = tag_pattern.sub(rf"\\tag{{{expected_tag}}}", text, count=1)
    if normalized == text:
        return text, []
    return _normalize_latex(normalized), ["normalized_formula_tag"]


def _normalize_spaced_latex_operator_names(latex: str) -> str:
    known_operators = {
        "argmax",
        "argmin",
        "cos",
        "det",
        "exp",
        "lim",
        "log",
        "max",
        "min",
        "sin",
        "softmax",
        "sup",
        "tan",
    }

    def replace(match: re.Match[str]) -> str:
        command = str(match.group("command") or "")
        body = str(match.group("body") or "")
        compact = re.sub(r"\s+", "", body).lower()
        if compact in known_operators:
            return f"{command}{{{compact}}}"
        return str(match.group(0) or "")

    return re.sub(
        r"(?P<command>\\(?:operatorname\*?|mathrm|text))\{(?P<body>(?:[A-Za-z]\s+){1,}[A-Za-z])\}",
        replace,
        str(latex or ""),
    )


def _validate_latex_candidate(
    *,
    equation_text: str,
    latex: str,
    ocr_confidence: float,
    normalization_signals: list[str] | None = None,
) -> dict[str, Any]:
    signals: list[str] = []
    issues: list[str] = []
    latex = _normalize_latex(latex)
    for signal in normalization_signals or []:
        if signal not in signals:
            signals.append(signal)
    if not latex:
        return {
            "accepted": False,
            "score": 0.0,
            "signals": [],
            "issues": ["empty_latex_candidate"],
        }

    syntax_score = _latex_syntax_score(latex)
    if syntax_score >= 0.85:
        signals.append("syntax_valid")
    else:
        issues.append("latex_syntax_weak")

    structure_score = _latex_structure_score(latex)
    if structure_score >= 0.35:
        signals.append("math_structure")
    else:
        issues.append("insufficient_math_structure")

    overlap_score = _symbol_overlap_score(equation_text, latex)
    if overlap_score >= 0.35:
        signals.append("symbol_overlap")
    else:
        issues.append("low_symbol_overlap")

    flattened_penalty = 0.0
    if _looks_like_flattened_plain_text(latex):
        flattened_penalty = 0.35
        issues.append("flattened_plain_text_candidate")

    hallucination_issues = _latex_hallucination_issues(equation_text=equation_text, latex=latex)
    if hallucination_issues:
        issues.extend(hallucination_issues)

    score = (
        max(0.0, min(1.0, float(ocr_confidence or 0.0))) * 0.35
        + syntax_score * 0.25
        + structure_score * 0.25
        + overlap_score * 0.15
        - flattened_penalty
    )
    score = round(max(0.0, min(1.0, score)), 4)
    blocking_issues = {
        "insufficient_math_structure",
        "latex_syntax_weak",
        "flattened_plain_text_candidate",
        "repetitive_latex_candidate",
        "excessive_latex_expansion",
        "explanatory_prose_in_formula_candidate",
        "unexplained_symbol_tokens",
    }
    acceptance_threshold = 0.85
    if len(latex) <= 80 and syntax_score >= 0.85 and structure_score >= 0.35 and overlap_score >= 0.65:
        acceptance_threshold = 0.79
    if len(latex) <= 80 and syntax_score >= 0.85 and structure_score >= 0.35 and not issues:
        acceptance_threshold = min(acceptance_threshold, 0.80)
    accepted = score >= acceptance_threshold and not (blocking_issues & set(issues))
    return {
        "accepted": accepted,
        "score": score,
        "signals": signals,
        "issues": issues,
        "ocr_confidence": round(max(0.0, min(1.0, float(ocr_confidence or 0.0))), 4),
        "syntax_score": round(syntax_score, 4),
        "structure_score": round(structure_score, 4),
        "symbol_overlap_score": round(overlap_score, 4),
    }


def _trim_repetitive_latex_tail(latex: str) -> tuple[str, list[str]]:
    text = _normalize_latex(latex)
    if not text:
        return text, []

    infix_match = _find_repetitive_latex_tail_start(text)
    if infix_match is not None:
        body = text[:infix_match].rstrip(" ,;:")
        tail = text[infix_match:]
        if _can_trim_repetitive_latex_tail(body, tail):
            return _normalize_latex(body), ["trimmed_repetitive_tail"]

    patterns = [
        re.compile(r"(?:\s*\\quad\s*\\(?:mathrm|operatorname|text)\{[^{}]{1,24}\}){4,}\s*$"),
        re.compile(r"(?:\s*\\qquad\s*\\(?:mathrm|operatorname|text)\{[^{}]{1,24}\}){4,}\s*$"),
        re.compile(r"(?:\s*\\(?:mathrm|operatorname|text)\{[^{}]{1,24}\}){4,}\s*$"),
        re.compile(r"(?:\s*\\quad){6,}\s*$"),
        re.compile(r"(?:\s*\\qquad){6,}\s*$"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        tail = str(match.group(0) or "")
        body = text[: match.start()].rstrip(" ,;:")
        if not _can_trim_repetitive_latex_tail(body, tail):
            continue
        return _normalize_latex(body), ["trimmed_repetitive_tail"]
    return text, []


def _find_repetitive_latex_tail_start(latex: str) -> int | None:
    repeated_token_re = re.compile(
        r"(?:\\quad|\\qquad)\s*\\(?:mathrm|operatorname|text)\{([^{}]{1,24})\}"
    )
    matches = list(repeated_token_re.finditer(latex))
    if len(matches) < 4:
        return None
    normalized = [
        _collapse_spaced_letter_words(str(match.group(1) or "")).replace("~", " ").strip().lower()
        for match in matches
    ]
    for start in range(0, len(matches) - 3):
        window = [item for item in normalized[start : start + 4] if item]
        if len(window) < 4:
            continue
        if len(set(window)) == 1:
            return int(matches[start].start())
    return None


def _can_trim_repetitive_latex_tail(body: str, tail: str) -> bool:
    if len(_clean_latex_for_tail_signal(body)) < 18:
        return False
    tail_units = re.findall(r"\\(?:mathrm|operatorname|text)\{([^{}]{1,24})\}|\\q?quad", tail)
    if len(tail_units) < 4:
        return False
    normalized_units = [
        _collapse_spaced_letter_words(str(unit or "")).replace("~", " ").strip().lower()
        for unit in tail_units
    ]
    normalized_units = [unit for unit in normalized_units if unit or r"\quad" in tail]
    distinct_units = {unit for unit in normalized_units if unit}
    if distinct_units and len(distinct_units) <= 2:
        return True
    tail_commands = _LATEX_COMMAND_RE.findall(tail)
    if tail_commands:
        counts = Counter(tail_commands)
        _, most_common_count = counts.most_common(1)[0]
        return most_common_count >= 4 and most_common_count / max(1, len(tail_commands)) >= 0.6
    return False


def _clean_latex_for_tail_signal(latex: str) -> str:
    text = re.sub(r"\\(?:quad|qquad)\b", " ", str(latex or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _latex_hallucination_issues(*, equation_text: str, latex: str) -> list[str]:
    issues: list[str] = []
    if _looks_like_formula_candidate_contains_explanatory_prose(latex):
        issues.append("explanatory_prose_in_formula_candidate")
    if _has_unexplained_latex_symbol_tokens(equation_text=equation_text, latex=latex):
        issues.append("unexplained_symbol_tokens")

    commands = _LATEX_COMMAND_RE.findall(latex)
    if commands:
        counts = Counter(commands)
        total = max(1, len(commands))
        most_common_command, most_common_count = counts.most_common(1)[0]
        if most_common_count >= 8 and most_common_count / total >= 0.35:
            issues.append("repetitive_latex_candidate")
        if most_common_command in {r"\sqrt", r"\frac"} and most_common_count >= 7:
            if "repetitive_latex_candidate" not in issues:
                issues.append("repetitive_latex_candidate")

    if re.search(r"(\\[A-Za-z]+\{){5,}", latex):
        if "repetitive_latex_candidate" not in issues:
            issues.append("repetitive_latex_candidate")

    text_token_count = len(_math_identity_tokens(equation_text))
    latex_token_count = len(_math_identity_tokens(latex))
    if text_token_count >= 4 and latex_token_count >= max(60, text_token_count * 8):
        issues.append("excessive_latex_expansion")
    if len(str(equation_text or "").strip()) >= 12 and len(latex) >= max(500, len(equation_text) * 10):
        if "excessive_latex_expansion" not in issues:
            issues.append("excessive_latex_expansion")
    return issues


def _has_unexplained_latex_symbol_tokens(*, equation_text: str, latex: str) -> bool:
    source_tokens = _math_identity_tokens(equation_text)
    if len(source_tokens) < 3:
        return False

    latex_tokens = _latex_noncommand_symbol_words(latex)
    if not latex_tokens:
        return False

    allowed_tokens = {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "varepsilon",
        "zeta",
        "eta",
        "theta",
        "vartheta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "varphi",
        "chi",
        "psi",
        "omega",
        "min",
        "max",
        "argmin",
        "argmax",
        "log",
        "exp",
        "sin",
        "cos",
        "tan",
        "softmax",
        "relu",
        "sigmoid",
        "tanh",
        "det",
        "trace",
        "tr",
    }
    unexplained = [
        token
        for token in latex_tokens
        if len(token) >= 4 and token not in source_tokens and token not in allowed_tokens
    ]
    return bool(unexplained)


def _latex_noncommand_symbol_words(latex: str) -> list[str]:
    text = str(latex or "")
    text = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z]+\*?", " ", text)
    text = text.replace("\\", " ")
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z]{2,}", text)
        if token.strip()
    ]
    tokens.extend(
        "".join(part.lower() for part in match.group(0).split())
        for match in re.finditer(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b", text)
    )
    return tokens


def _looks_like_formula_candidate_contains_explanatory_prose(latex: str) -> bool:
    text = str(latex or "")
    if not text.strip():
        return False
    prose_wrapped = " ".join(re.findall(r"\\(?:operatorname|mathrm|text)\s*\{([^{}]{8,})\}", text))
    prose_wrapped = prose_wrapped.replace("~", " ")
    prose_wrapped = _collapse_spaced_letter_words(prose_wrapped)
    wrapped_words = [word.lower() for word in re.findall(r"[A-Za-z]{3,}", prose_wrapped)]
    prose_markers = {
        "where",
        "defined",
        "definition",
        "obtained",
        "carrying",
        "direction",
        "search",
        "respectively",
        "within",
        "class",
        "distance",
        "following",
        "given",
        "thus",
        "therefore",
    }
    if sum(1 for word in wrapped_words if word in prose_markers) >= 1 and len(wrapped_words) >= 3:
        return True
    if len(wrapped_words) >= 8:
        return True
    compact_wrapped = re.sub(r"[^A-Za-z]", "", prose_wrapped).lower()
    if any(marker in compact_wrapped for marker in prose_markers) and len(compact_wrapped) >= 24:
        return True

    protected = re.sub(r"\\(?:operatorname|mathrm|mathbf|mathit|text)\s*\{[^{}]{0,80}\}", " ", text)
    protected = _collapse_spaced_letter_words(protected)
    protected = re.sub(r"\\[A-Za-z]+", " ", protected)
    protected = re.sub(r"\{|\}|\^|_|=|\+|-|/|\*|\||,|\.|:|;|&|\\\\", " ", protected)
    words = [word.lower() for word in re.findall(r"[A-Za-z]{3,}", protected)]
    if not words:
        return False
    marker_count = sum(1 for word in words if word in prose_markers)
    if marker_count >= 1 and len(words) >= 3:
        return True
    return len(words) >= 8


def _collapse_spaced_letter_words(text: str) -> str:
    def collapse(match: re.Match[str]) -> str:
        return str(match.group(0) or "").replace(" ", "").replace("~", "")

    return re.sub(r"\b(?:[A-Za-z][ ~]){2,}[A-Za-z]\b", collapse, str(text or ""))


def _latex_syntax_score(latex: str) -> float:
    if not latex:
        return 0.0
    stack: list[str] = []
    pairs = {"}": "{", "]": "[", ")": "("}
    escaped = False
    for char in latex:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in "{[(":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return 0.2
    if stack:
        return 0.55
    return 1.0 if _MATH_SYMBOL_RE.search(latex) else 0.35


def _latex_structure_score(latex: str) -> float:
    score = 0.0
    commands = set(_LATEX_COMMAND_RE.findall(latex))
    if commands:
        score += 0.25
    if any(command in commands for command in {r"\frac", r"\dfrac", r"\tfrac"}):
        score += 0.25
    if any(command in commands for command in {r"\sum", r"\prod", r"\int"}):
        score += 0.20
    if "_" in latex or "^" in latex:
        score += 0.15
    if any(command in commands for command in {r"\begin", r"\left", r"\right", r"\cases"}):
        score += 0.15
    if any(command in commands for command in {r"\tag", r"\eqno"}):
        score += 0.10
    if r"\|" in latex or "||" in latex or any(char in latex for char in "{}"):
        score += 0.10
    if _symbolic_operator_count(latex) >= 4:
        score += 0.10
    return min(1.0, score)


def _symbolic_operator_count(text: str) -> int:
    return len(re.findall(r"[=+\-*/^_{}|]|\|\||\\\||\\(?:sum|prod|int|log|exp|min|max)", str(text or "")))


def _symbol_overlap_score(equation_text: str, latex: str) -> float:
    text_tokens = _math_identity_tokens(equation_text)
    latex_tokens = _math_identity_tokens(latex)
    if not text_tokens:
        return 0.5 if latex_tokens else 0.0
    if not latex_tokens:
        return 0.0
    return len(text_tokens & latex_tokens) / max(1, len(text_tokens))


def _math_identity_tokens(text: str) -> set[str]:
    normalized = str(text or "")
    replacements = {
        "∑": "sum",
        "Σ": "sum",
        r"\sum": "sum",
        r"\frac": "frac",
        "−": "-",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, f" {new} ")
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z]+|\d+|sum|frac", normalized)
        if token.strip()
    }
    return tokens


def _looks_like_flattened_plain_text(latex: str) -> bool:
    text = str(latex or "").strip()
    if not text:
        return True
    has_latex_command = bool(_LATEX_COMMAND_RE.search(text))
    has_math_symbol = bool(_MATH_SYMBOL_RE.search(text))
    has_flattened_keyword = bool(_FLATTENED_TOKEN_RE.search(text))
    if not has_latex_command and has_flattened_keyword:
        return True
    if not has_latex_command and len(re.findall(r"\s+", text)) >= 6 and has_math_symbol:
        return True
    return False


def _recognize_equation_latex(path: Path, equation_block: dict[str, Any]) -> FormulaOcrCandidate | None:
    image_bytes = _render_equation_crop_png(path, equation_block)
    if not image_bytes:
        return None

    if paddle_formula_ocr_enabled():
        candidate = _recognize_with_paddle_formula(image_bytes)
        if candidate is not None:
            return candidate
    if texteller_ocr_enabled():
        candidate = _recognize_with_texteller_cli(image_bytes)
        if candidate is not None:
            return candidate
    candidate = _recognize_with_rapid_latex_ocr(image_bytes)
    if candidate is not None:
        return candidate
    return None


def _recognize_inline_formula_latex(
    path: Path,
    page_number: int,
    inline_span: dict[str, Any],
) -> FormulaOcrCandidate | None:
    image_bytes = _render_inline_formula_crop_png(path, page_number, inline_span)
    if not image_bytes:
        return None

    if paddle_formula_ocr_enabled():
        candidate = _recognize_with_paddle_formula(image_bytes)
        if candidate is not None:
            return candidate
    if texteller_ocr_enabled():
        candidate = _recognize_with_texteller_cli(image_bytes)
        if candidate is not None:
            return candidate
    candidate = _recognize_with_rapid_latex_ocr(image_bytes)
    if candidate is not None:
        return candidate
    return None


def _render_equation_crop_png(path: Path, equation_block: dict[str, Any], scale: float = 3.0) -> bytes | None:
    if pymupdf is None or not Path(path).exists():
        return None
    bbox = list(equation_block.get("ocr_bbox") or equation_block.get("bbox", []) or [])
    if len(bbox) != 4:
        return None
    page_number = int(equation_block.get("page", 0) or 0)
    if page_number <= 0:
        return None
    try:
        with pymupdf.open(path) as document:
            page = document.load_page(page_number - 1)
            clip = pymupdf.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])) & page.rect
            if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                return None
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False)
            return pixmap.tobytes("png")
    except Exception:
        return None


def _render_inline_formula_crop_png(
    path: Path,
    page_number: int,
    inline_span: dict[str, Any],
    scale: float = 4.0,
) -> bytes | None:
    if pymupdf is None or not Path(path).exists():
        return None
    bbox = list(inline_span.get("ocr_bbox") or inline_span.get("bbox", []) or [])
    if len(bbox) != 4 or page_number <= 0:
        return None
    try:
        with pymupdf.open(path) as document:
            page = document.load_page(page_number - 1)
            clip = pymupdf.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            pad_x = max(2.0, min(8.0, clip.width * 0.06))
            pad_y = max(1.5, min(5.0, clip.height * 0.25))
            clip = pymupdf.Rect(
                clip.x0 - pad_x,
                clip.y0 - pad_y,
                clip.x1 + pad_x,
                clip.y1 + pad_y,
            ) & page.rect
            if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                return None
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False)
            return pixmap.tobytes("png")
    except Exception:
        return None


def _texteller_cli_path() -> str | None:
    configured = str(os.environ.get("IND_TEXTELLER_CLI", "")).strip()
    if configured:
        return configured
    return shutil.which("texteller")


def _extract_texteller_latex(stdout: str) -> str:
    text = str(stdout or "").strip()
    if not text:
        return ""
    fenced = re.search(r"Predicted\s+LaTeX:\s*```\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return str(fenced.group(1) or "").strip()
    generic_fenced = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fenced:
        return str(generic_fenced.group(1) or "").strip()
    marker = re.search(r"Predicted\s+LaTeX:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if marker:
        return str(marker.group(1) or "").strip()
    return text


def _recognize_with_texteller_cli(image_bytes: bytes) -> FormulaOcrCandidate | None:
    cli_path = _texteller_cli_path()
    if not cli_path:
        return None

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)
        completed = subprocess.run(
            [cli_path, "inference", str(temp_path), "--output-format", "latex"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(os.environ.get("IND_TEXTELLER_TIMEOUT_SECONDS", "120") or "120"),
        )
    except Exception:
        return None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if int(getattr(completed, "returncode", 1) or 0) != 0:
        return None
    latex = _extract_texteller_latex(str(getattr(completed, "stdout", "") or ""))
    if not latex:
        return None
    return FormulaOcrCandidate(latex=latex, confidence=0.95, source="texteller_cli")


_PADDLE_FORMULA_MODEL: Any | None = None


def _paddle_formula_model_name() -> str:
    return str(os.environ.get("IND_PADDLE_FORMULA_MODEL", "") or "PP-FormulaNet_plus-M").strip()


def _load_paddle_formula_model() -> Any | None:
    global _PADDLE_FORMULA_MODEL
    if _PADDLE_FORMULA_MODEL is not None:
        return _PADDLE_FORMULA_MODEL

    try:
        from paddleocr import FormulaRecognition
    except Exception:
        return None

    try:
        _PADDLE_FORMULA_MODEL = FormulaRecognition(model_name=_paddle_formula_model_name())
    except Exception:
        return None
    return _PADDLE_FORMULA_MODEL


def _extract_paddle_formula_latex(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in (
            "rec_formula",
            "formula",
            "latex",
            "latex_text",
            "text",
            "pred",
            "prediction",
            "result",
        ):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("res", "data", "output", "outputs"):
            value = _extract_paddle_formula_latex(result.get(key))
            if value:
                return value
        for value in result.values():
            latex = _extract_paddle_formula_latex(value)
            if latex:
                return latex
        return ""
    if isinstance(result, (list, tuple)):
        for item in result:
            latex = _extract_paddle_formula_latex(item)
            if latex:
                return latex
    return ""


def _recognize_with_paddle_formula(image_bytes: bytes) -> FormulaOcrCandidate | None:
    model = _load_paddle_formula_model()
    if model is None:
        return None

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)
        result = model.predict(input=str(temp_path), batch_size=1)
    except Exception:
        return None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    latex = _extract_paddle_formula_latex(result)
    if not latex:
        return None
    return FormulaOcrCandidate(
        latex=latex,
        confidence=0.96,
        source=f"paddleocr_formula:{_paddle_formula_model_name()}",
    )


def _recognize_with_rapid_latex_ocr(image_bytes: bytes) -> FormulaOcrCandidate | None:
    try:
        from rapid_latex_ocr import LaTeXOCR
    except Exception:
        return None

    try:
        engine = LaTeXOCR()
    except Exception:
        return None

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)
        result = engine(str(temp_path))
    except Exception:
        return None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    latex = ""
    confidence = 0.0
    if isinstance(result, tuple):
        if result:
            latex = str(result[0] or "")
        if len(result) >= 2:
            try:
                confidence = float(result[1] or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
    else:
        latex = str(result or "")
        confidence = 0.70
    if not latex.strip():
        return None
    return FormulaOcrCandidate(latex=latex, confidence=confidence, source="rapid_latex_ocr")
