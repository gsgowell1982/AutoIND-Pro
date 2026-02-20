from typing import Any


def compare_atomic_facts(module_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect fact inconsistencies across CTD modules."""
    findings: list[dict[str, Any]] = []
    baseline = module_payloads.get("module_3", {})
    baseline_facts = baseline.get("atomic_facts", {})
    for module_name, payload in module_payloads.items():
        current_facts = payload.get("atomic_facts", {})
        for key, base_value in baseline_facts.items():
            current_value = current_facts.get(key)
            if current_value is not None and current_value != base_value:
                findings.append(
                    {
                        "module": module_name,
                        "fact": key,
                        "expected": base_value,
                        "actual": current_value,
                    }
                )
    return findings
