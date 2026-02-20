from typing import Any


def map_ectd_to_ctd(ectd_index: dict[str, Any]) -> dict[str, Any]:
    """Map eCTD directory structure to CTD module buckets."""
    return {
        "module_1": ectd_index.get("module_1", []),
        "module_2": ectd_index.get("module_2", []),
        "module_3": ectd_index.get("module_3", []),
        "module_4": ectd_index.get("module_4", []),
        "module_5": ectd_index.get("module_5", []),
    }
