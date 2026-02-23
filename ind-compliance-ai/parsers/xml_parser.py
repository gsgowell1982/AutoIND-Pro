from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from parsers.common.atomic_fact_extractor import extract_atomic_facts


def _walk_xml(node: ET.Element, pointer: str, nodes: list[dict[str, Any]]) -> None:
    text = " ".join((node.text or "").split())
    nodes.append(
        {
            "pointer": pointer,
            "tag": node.tag,
            "attributes": dict(node.attrib),
            "text": text,
        }
    )
    for child_index, child in enumerate(list(node)):
        child_pointer = f"{pointer}/{child.tag}[{child_index}]"
        _walk_xml(child, child_pointer, nodes)


def parse_xml(path: Path) -> dict[str, Any]:
    """Parse XML/eCTD metadata into normalized nodes."""
    xml_text = path.read_text(encoding="utf-8", errors="ignore")
    nodes: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
        _walk_xml(root, f"/{root.tag}[0]", nodes)
        parser_hint = "xml-tree-v1"
    except ET.ParseError:
        parser_hint = "xml-fallback-plain-text"
        for line_index, raw_line in enumerate(xml_text.splitlines()):
            cleaned = " ".join(raw_line.split())
            if cleaned:
                nodes.append({"pointer": f"/raw[{line_index}]", "tag": "raw", "attributes": {}, "text": cleaned})

    merged_text = "\n".join(node["text"] for node in nodes if node.get("text"))
    return {
        "source_path": str(path),
        "source_type": "xml",
        "nodes": nodes,
        "document_ast": {
            "source_type": "xml",
            "nodes": nodes,
            "node_count": len(nodes),
        },
        "text": merged_text,
        "atomic_facts": extract_atomic_facts(merged_text),
        "metadata": {"node_count": len(nodes), "parser_hint": parser_hint},
    }
