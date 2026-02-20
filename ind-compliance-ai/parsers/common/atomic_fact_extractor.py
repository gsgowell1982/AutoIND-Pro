import re
from typing import Pattern


FACT_PATTERNS: dict[str, Pattern[str]] = {
    "drug_name": re.compile(r"(?:药品名称|Drug Name)\s*[:：]\s*([^\n,;；，]+)", re.IGNORECASE),
    "dosage_form": re.compile(r"(?:剂型|Dosage Form)\s*[:：]\s*([^\n,;；，]+)", re.IGNORECASE),
    "batch_number": re.compile(
        r"(?:批号|Batch(?:\s*(?:No\.?|Number))?)\s*[:：]\s*([A-Za-z0-9._/-]+)",
        re.IGNORECASE,
    ),
    "manufacturing_site": re.compile(
        r"(?:生产地|生产地址|Manufacturing Site)\s*[:：]\s*([^\n,;；，]+)",
        re.IGNORECASE,
    ),
    "strength": re.compile(r"(?:规格|Strength)\s*[:：]\s*([^\n,;；，]+)", re.IGNORECASE),
}


def extract_atomic_facts(text: str) -> dict[str, str]:
    """Extract a compact set of cross-module comparison facts from raw text."""
    facts: dict[str, str] = {}
    for fact_key, pattern in FACT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            facts[fact_key] = match.group(1).strip()
    return facts
