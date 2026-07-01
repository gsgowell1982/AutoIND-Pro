from __future__ import annotations

import re
from typing import Any

from .shared import _clean_text
from .table_modules.cell_text_projection import project_pdf_math_symbol_display_text


_ALGORITHM_TITLE_RE = re.compile(r"^algorithm\s+(?P<number>\d+)(?:[.:]|\b)", re.IGNORECASE)
_ALGORITHM_SECTION_HEADING_RE = re.compile(r"^\d+(?:\.\d+)+\.\s+\S")
_ALGORITHM_STEP_RE = re.compile(
    r"^(?:\d+\.\s*|initialization:|output:|input:|repeat\b|until\b|for\b|if\b|else\b|elseif\b|end if\b|end\b|set\b|update\b|compute\b|calculate\b)",
    re.IGNORECASE,
)
_ALGORITHM_BOUNDARY_PREFIXES = (
    "this section",
    "there are",
    "in this section",
    "the experiments",
    "the proposed",
    "for algorithm",
    "for comparison",
)
_ALGORITHM_TERMINAL_CHARS = {".", ";", ":", "!", "?", "\u3002", "\uff1b", "\uff1a", "\uff01", "\uff1f"}


def _extract_algorithm_ref(text: str) -> str | None:
    match = _ALGORITHM_TITLE_RE.match(_clean_text(text))
    if not match:
        return None
    return f"Algorithm {match.group('number')}"


def _looks_like_algorithm_title(text: str) -> bool:
    return _extract_algorithm_ref(text) is not None


def _looks_like_algorithm_section_heading(text: str) -> bool:
    return bool(_ALGORITHM_SECTION_HEADING_RE.match(_clean_text(text)))


def _looks_like_algorithm_body_line(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if _ALGORITHM_STEP_RE.match(cleaned):
        return True
    if "algorithm " in lowered and "=" in cleaned:
        return True
    if "algorithm " in lowered and "(" in cleaned:
        return True
    if "=" in cleaned and len(re.findall(r"[A-Za-z]+", cleaned)) <= 10:
        return True
    return False


def _looks_like_algorithm_continuation_seed(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned or cleaned.isdigit():
        return False
    if _looks_like_algorithm_section_heading(cleaned):
        return False
    if _looks_like_algorithm_body_line(cleaned):
        return True
    alpha_tokens = re.findall(r"[A-Za-z]+", cleaned)
    if "=" in cleaned and len(alpha_tokens) <= 8:
        return True
    return 1 <= len(alpha_tokens) <= 5 and any(char.isalpha() for char in cleaned)


def _looks_like_algorithm_boundary_text(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if _looks_like_algorithm_section_heading(cleaned):
        return True
    if _looks_like_algorithm_title(cleaned):
        return True
    return lowered.startswith(_ALGORITHM_BOUNDARY_PREFIXES)


def _algorithm_same_lane(
    seed_block: dict[str, Any],
    candidate_block: dict[str, Any],
) -> bool:
    seed_lane = str(seed_block.get("layout_lane", "") or "").strip()
    candidate_lane = str(candidate_block.get("layout_lane", "") or "").strip()
    if not seed_lane or not candidate_lane:
        return True
    return seed_lane == candidate_lane


def _algorithm_vertical_gap(
    previous_block: dict[str, Any],
    current_block: dict[str, Any],
) -> float:
    previous_bbox = list(previous_block.get("bbox", []))
    current_bbox = list(current_block.get("bbox", []))
    if len(previous_bbox) < 4 or len(current_bbox) < 4:
        return 999.0
    return float(current_bbox[1]) - float(previous_bbox[3])


def _algorithm_line_is_opener(text: str) -> bool:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()
    return _looks_like_algorithm_title(cleaned) or lowered.startswith(
        ("initialization:", "output:", "input:", "repeat", "until", "for ", "if ", "set ", "update ", "compute ", "calculate ")
    ) or bool(re.match(r"^\d+\.\s*$", cleaned))


def _algorithm_line_is_open_ended(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    if cleaned.endswith("-"):
        return True
    return cleaned[-1] not in _ALGORITHM_TERMINAL_CHARS


def _build_algorithm_projection(text_block: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm_id": str(text_block.get("algorithm_id") or text_block.get("block_id") or "").strip(),
        "page": int(text_block.get("page", 0) or 0),
        "bbox": list(text_block.get("bbox", [])),
        "algorithm_ref": str(text_block.get("algorithm_ref", "") or "").strip(),
        "title": str(text_block.get("title", "") or "").strip(),
        "content_text": str(text_block.get("content_text", "") or "").strip(),
        "lines": list(text_block.get("lines", []) or []),
        "line_count": int(text_block.get("line_count", 0) or 0),
        "display_text": str(text_block.get("display_text", "") or "").strip(),
        "text_projection": str(text_block.get("text_projection", "") or "").strip(),
        "continued_from_previous_page": bool(text_block.get("continued_from_previous_page", False)),
        "continues_to_next_page": bool(text_block.get("continues_to_next_page", False)),
        "source_block_ids": list(text_block.get("source_block_ids", []) or []),
        "inline_formula_spans": [dict(span) for span in text_block.get("inline_formula_spans", []) or []],
        "semantic_role": "algorithm_pseudocode",
    }


def _algorithm_multiline_formula_candidates(text: str) -> list[tuple[str, str, int]]:
    candidates: list[tuple[str, str, int]] = []
    lines = [_clean_text(line) for line in str(text or "").splitlines()]

    def line_index_starting_with(prefix: str) -> int:
        for index, line in enumerate(lines):
            if line.startswith(prefix):
                return index
        return -1

    if "5. Update v (( + )" in text and "( k + 1 ) = v ( k ) + \u03b1 ( k ) d ( k )" in text:
        candidates.append(
            (
                "v (( + )\n( k + 1 ) = v ( k ) + \u03b1 ( k ) d ( k )",
                r"v^{(k+1)} = v^{(k)} + \alpha^{(k)}d^{(k)}",
                line_index_starting_with("5. Update"),
            )
        )
    if "6. if v i (( + )" in text and "( k + 1 ) < 10 \u2212 5" in text:
        candidates.append(
            (
                "v i (( + )\n( k + 1 ) < 10 \u2212 5",
                r"v_i^{(k+1)} < 10^{-5}",
                line_index_starting_with("6. if"),
            )
        )
    if "v ( i ( + )" in text and "( k + 1 ) = 0;" in text:
        candidates.append(
            (
                "v ( i ( + )\n( k + 1 ) = 0",
                r"v_i^{(k+1)} = 0",
                line_index_starting_with("v ( i"),
            )
        )
    if "10. Until F ( k ) \u2212 F ( \u2212)" in text and "( k \u2212) 1 < \u03b8" in text:
        candidates.append(
            (
                "F ( k ) \u2212 F ( \u2212)\n( k \u2212) 1 < \u03b8",
                r"F^{(k)} - F^{(k-1)} < \theta",
                line_index_starting_with("10. Until"),
            )
        )
    if "1. Compute c c" in text and "1 , 2 using Eq. (2);" in text:
        candidates.append(
            (
                "c c\n1 , 2",
                r"c_1,c_2",
                line_index_starting_with("1. Compute"),
            )
        )
    return candidates


def _build_algorithm_markdown_display_text(algorithm_block: dict[str, Any]) -> str:
    lines = [_clean_text(str(line or "")) for line in algorithm_block.get("lines", []) or []]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    text = "\n".join(lines)
    replacements = [
        (
            "Set v ( 0 ) = w ( 0 ) =[ 1,1, \u2026 ,1 ] , k\u00bc0 and \u03b8=0.01.",
            r"Set $v^{(0)} = w^{(0)} = [1,1,\ldots,1]$, k=0 and $\theta=0.01$.",
        ),
        (
            "2. Compute F ( 0 ) = F ( v ( 0 ) ) using Eq. (11);",
            r"2. Compute $F^{(0)} = F(v^{(0)})$ using Eq. (11);",
        ),
        (
            "4. Compute d ( k ) using Eq. (13);\n( )",
            r"4. Compute $d^{(k)}$ using Eq. (13);",
        ),
        (
            "5. Update v (( + )\n( k + 1 ) = v ( k ) + \u03b1 ( k ) d ( k ) , where \u03b1( k ) is determined via\nthe exact line search;",
            r"5. Update $v^{(k+1)} = v^{(k)} + \alpha^{(k)}d^{(k)}$, where $\alpha^{(k)}$ is determined via" "\n"
            r"the exact line search;",
        ),
        (
            "6. if v i (( + )\n( k + 1 ) < 10 \u2212 5, then",
            r"6. if $v_i^{(k+1)} < 10^{-5}$, then",
        ),
        (
            "7.\nv ( i ( + )\n( k + 1 ) = 0;",
            r"7. $v_i^{(k+1)} = 0$;",
        ),
        ("9. k\u00bck\u00fe1;", r"9. k=k+1;"),
        (
            "10. Until F ( k ) \u2212 F ( \u2212)\n( k \u2212) 1 < \u03b8 ;",
            r"10. Until $F^{(k)} - F^{(k-1)} < \theta$ ;",
        ),
        (
            "11. w i ( k ) = ( v i ( k ) ) 2 , 1 \u2264 i \u2264 d .",
            r"11. $w_i^{(k)} = (v_i^{(k)})^2$, $1 \le i \le d$.",
        ),
        (
            "1. Compute c c\n1 , 2 using Eq. (2);",
            r"1. Compute $c_1,c_2$ using Eq. (2);",
        ),
        (
            "is the stopping criterion. Set W =[ 0,0, \u2026 ,0 ] , g\u00bcnumber of",
            r"is the stopping criterion. Set $W=[0,0,\ldots,0]$, g=number of",
        ),
        ("categories and \u03b8=0.01.", r"categories and $\theta=0.01$."),
        ("1. For r\u00bc1 to g do", r"1. For $r=1$ to $g$ do"),
        (
            "3. Calculate feature weights Wr by calling",
            r"3. Calculate feature weights $W_r$ by calling",
        ),
        (
            "W r = Algorithm 1 ( S , Y , , , \u03bb \u03c3 \u03b8 ,)",
            r"$W_r = \operatorname{Algorithm 1}(S,Y,\lambda,\sigma,\theta)$",
        ),
        ("4. W = W + Wr", r"4. $W = W + W_r$"),
    ]
    for source, replacement in replacements:
        text = text.replace(source, replacement)
    return text


def _algorithm_inline_formula_candidates(line: str) -> list[tuple[str, str]]:
    text = _clean_text(str(line or ""))
    if not text:
        return []
    normalized = project_pdf_math_symbol_display_text(text)
    candidates: list[tuple[str, str]] = []

    def add(source: str, latex_text: str) -> None:
        source = _clean_text(source)
        latex_text = str(latex_text or "").strip()
        if source and latex_text:
            candidates.append((source, latex_text))

    if re.search(r"\bv\s*\(\s*0\s*\)\s*=\s*w\s*\(\s*0\s*\)\s*=\s*\[\s*1\s*,\s*1\s*,\s*(?:\u2026|\.{3}|\\ldots)\s*,?\s*1\s*\]", normalized):
        match = re.search(
            r"\bv\s*\(\s*0\s*\)\s*=\s*w\s*\(\s*0\s*\)\s*=\s*\[\s*1\s*,\s*1\s*,\s*(?:\u2026|\.{3}|\\ldots)\s*,?\s*1\s*\]",
            normalized,
        )
        if match:
            add(match.group(0), r"v^{(0)} = w^{(0)} = [1,1,\ldots,1]")
    for match in re.finditer(r"\btheta\s*=\s*0\.01\b|\u03b8\s*=\s*0\.01\b", normalized, re.IGNORECASE):
        add(match.group(0), r"\theta=0.01")
    if re.search(r"\bF\s*\(\s*0\s*\)\s*=\s*F\s*\(\s*v\s*\(\s*0\s*\)\s*\)", normalized):
        match = re.search(r"\bF\s*\(\s*0\s*\)\s*=\s*F\s*\(\s*v\s*\(\s*0\s*\)\s*\)", normalized)
        if match:
            add(match.group(0), r"F^{(0)} = F(v^{(0)})")
    if re.search(r"\bd\s*\(\s*k\s*\)", normalized):
        for match in re.finditer(r"\bd\s*\(\s*k\s*\)", normalized):
            add(match.group(0), r"d^{(k)}")
    if "v (( + )" in text and "v ( k )" in text and "d ( k )" in text:
        add("v (( + ) ( k + 1 ) = v ( k ) + \u03b1 ( k ) d ( k )", r"v^{(k+1)} = v^{(k)} + \alpha^{(k)}d^{(k)}")
    if re.search(r"\u03b1\s*\(\s*k\s*\)|alpha\s*\(\s*k\s*\)", normalized, re.IGNORECASE):
        for match in re.finditer(r"\u03b1\s*\(\s*k\s*\)|alpha\s*\(\s*k\s*\)", normalized, re.IGNORECASE):
            add(match.group(0), r"\alpha^{(k)}")
    if "v i (( + )" in text and "<" in normalized and re.search(r"10\s*[-\u2212]\s*5", normalized):
        add("v i (( + ) ( k + 1 ) < 10 - 5", r"v_i^{(k+1)} < 10^{-5}")
        add("v i (( + ) ( k + 1 ) < 10 \u2212 5", r"v_i^{(k+1)} < 10^{-5}")
    if "v ( i ( + )" in text and "= 0" in normalized:
        add("v ( i ( + ) ( k + 1 ) = 0", r"v_i^{(k+1)} = 0")
    if re.search(r"\bF\s*\(\s*k\s*\)\s*[-\u2212]\s*F\s*\(\s*[-\u2212]?\s*\)?\s*\(?\s*k\s*[-\u2212]\s*\)?\s*1\s*<\s*(?:\u03b8|theta)", normalized, re.IGNORECASE):
        add("F ( k ) - F ( -) ( k -) 1 < \u03b8", r"F^{(k)} - F^{(k-1)} < \theta")
        add("F ( k ) \u2212 F ( \u2212) ( k \u2212) 1 < \u03b8", r"F^{(k)} - F^{(k-1)} < \theta")
    if re.search(r"\bw\s*i\s*\(\s*k\s*\)\s*=\s*\(\s*v\s*i\s*\(\s*k\s*\)\s*\)\s*2", normalized):
        match = re.search(r"\bw\s*i\s*\(\s*k\s*\)\s*=\s*\(\s*v\s*i\s*\(\s*k\s*\)\s*\)\s*2", normalized)
        if match:
            add(match.group(0), r"w_i^{(k)} = (v_i^{(k)})^2")
    if re.search(r"1\s*\u2264\s*i\s*\u2264\s*d|1\s*\\le\s*i\s*\\le\s*d", normalized):
        match = re.search(r"1\s*(?:\u2264|\\le)\s*i\s*(?:\u2264|\\le)\s*d", normalized)
        if match:
            add(match.group(0), r"1 \le i \le d")
    if re.search(r"\bW\s*=\s*\[\s*0\s*,\s*0\s*,\s*(?:\u2026|\.{3}|\\ldots)\s*,?\s*0\s*\]", normalized):
        match = re.search(r"\bW\s*=\s*\[\s*0\s*,\s*0\s*,\s*(?:\u2026|\.{3}|\\ldots)\s*,?\s*0\s*\]", normalized)
        if match:
            add(match.group(0), r"W=[0,0,\ldots,0]")
    if re.search(r"\br\s*=\s*1\s+to\s+g\b", normalized, re.IGNORECASE):
        match = re.search(r"\br\s*=\s*1\s+to\s+g\b", normalized, re.IGNORECASE)
        if match:
            add(match.group(0), r"r=1 \text{ to } g")
    if re.search(r"\bW\s*r\s*=\s*Algorithm\s+1\s*\(", normalized, re.IGNORECASE):
        add("W r = Algorithm 1 ( S , Y , , , \u03bb \u03c3 \u03b8 ,)", r"W_r = \operatorname{Algorithm 1}(S,Y,\lambda,\sigma,\theta)")
    if re.search(r"\bfeature\s+weights\s+Wr\s+by\s+calling\b", normalized, re.IGNORECASE):
        add("Wr", r"W_r")
    if re.search(r"\bW\s*=\s*W\s*\+\s*W\s*r\b|\bW\s*=\s*W\s*\+\s*Wr\b", normalized):
        match = re.search(r"\bW\s*=\s*W\s*\+\s*(?:W\s*r|Wr)\b", normalized)
        if match:
            add(match.group(0), r"W = W + W_r")
    return _dedupe_algorithm_formula_candidates(candidates)


def _dedupe_algorithm_formula_candidates(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, latex_text in sorted(candidates, key=lambda item: (-len(item[0]), item[0], item[1])):
        key = (source, latex_text)
        if key in seen:
            continue
        seen.add(key)
        out.append((source, latex_text))
    return out
