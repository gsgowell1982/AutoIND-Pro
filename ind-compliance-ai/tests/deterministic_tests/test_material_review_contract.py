from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.material_review_contract import (
    MATERIAL_REVIEW_CONTRACT_VERSION,
    build_material_review_contract,
)
from core.run_manager import create_run_context, persist_document_artifacts, persist_normalized_artifacts
from parsers.parser_registry import parse_file


def _build_sample_document() -> dict[str, object]:
    return {
        "filename": "sample-ind.pdf",
        "file_id": "file_001",
        "source_type": "pdf",
        "source_path": "D:\\sample-ind.pdf",
        "metadata": {
            "page_count": 2,
            "table_count": 1,
            "image_count": 0,
            "figure_count": 0,
            "toc_count": 1,
            "toc_sequence_count": 1,
            "content_evidence_count": 3,
            "content_unit_count": 2,
            "fact_extraction_unit_count": 1,
            "review_required_table_count": 1,
            "review_required_toc_count": 1,
            "toc_review_item_count": 1,
            "aggregated_toc_review_item_count": 1,
            "continuation_table_count": 0,
            "cross_page_table_links": 0,
            "cross_page_boundary_row_merge_count": 0,
            "low_confidence_table_count": 0,
            "diagnostic_table_count": 1,
            "content_evidence_counts": {"text": 1, "table": 1, "toc": 1},
            "content_unit_counts": {"text": 1, "toc": 1},
        },
        "document_ast": {
            "pages": [{"page": 1}, {"page": 2}],
            "table_refs": ["tbl_001"],
            "toc_refs": ["toc_001"],
            "toc_sequence_refs": ["tocseq_001"],
            "image_refs": [],
            "content_evidence_refs": ["ce_text_001", "ce_table_001", "ce_toc_001"],
            "content_unit_refs": ["cu_text_001", "cu_toc_001"],
        },
        "content_units": [
            {
                "unit_id": "cu_text_001",
                "evidence_id": "ce_text_001",
                "source_type": "text",
                "source_id": "txt_001",
                "page": 1,
                "bbox": [10.0, 10.0, 90.0, 20.0],
                "semantic_role": "text_block",
                "unit_role": "body",
                "unit_index": 1,
                "text": "Drug Name: ExampleDrug\nDosage Form: Injection",
                "attributes": {},
                "section_context": {
                    "module_label": "M3",
                    "outline_index": "3.2.S.1",
                    "section_title": "General Information",
                    "anchor_source": "heading",
                    "anchor_confidence": 0.93,
                },
                "fact_extraction_eligible": True,
            },
            {
                "unit_id": "cu_toc_001",
                "evidence_id": "ce_toc_001",
                "source_type": "toc",
                "source_id": "toc_001",
                "page": 2,
                "bbox": [10.0, 120.0, 220.0, 220.0],
                "semantic_role": "toc_outline",
                "unit_role": "entry",
                "unit_index": 1,
                "text": "1.0 | Overview | 3",
                "attributes": {"entry_index": 1},
                "fact_extraction_eligible": False,
            },
            {
                "unit_id": "cu_eq_001",
                "evidence_id": "ce_eq_001",
                "source_type": "text",
                "source_id": "eq_001",
                "page": 1,
                "bbox": [10.0, 210.0, 220.0, 240.0],
                "semantic_role": "display_equation",
                "unit_role": "equation",
                "unit_index": 3,
                "text": "fr(h, t) = rT(h ⋆t) (1)",
                "attributes": {"equation_label": "(1)"},
                "fact_extraction_eligible": False,
            },
        ],
        "content_evidence": [
            {
                "evidence_id": "ce_text_001",
                "source_type": "text",
                "source_id": "txt_001",
                "page": 1,
                "bbox": [10.0, 10.0, 90.0, 20.0],
                "semantic_role": "text_block",
                "content_text": "Drug Name: ExampleDrug\nDosage Form: Injection",
                "section_context": {
                    "module_label": "M3",
                    "outline_index": "3.2.S.1",
                    "section_title": "General Information",
                    "anchor_source": "heading",
                    "anchor_confidence": 0.93,
                },
                "segments": [
                    {"role": "body", "text": "Drug Name: ExampleDrug\nDosage Form: Injection"},
                ],
            },
            {
                "evidence_id": "ce_toc_001",
                "source_type": "toc",
                "source_id": "toc_001",
                "page": 2,
                "bbox": [10.0, 120.0, 220.0, 220.0],
                "semantic_role": "toc_outline",
                "content_text": "TABLE OF CONTENTS\n1.0 | Overview | 3",
                "segments": [
                    {"role": "title", "text": "TABLE OF CONTENTS"},
                    {"role": "entry", "text": "1.0 | Overview | 3"},
                ],
            },
            {
                "evidence_id": "ce_table_001",
                "source_type": "table",
                "source_id": "tbl_001",
                "page": 1,
                "bbox": [10.0, 40.0, 200.0, 100.0],
                "semantic_role": "business_table",
                "content_text": "Batch Inventory\nBatch Number | Strength\nBatch Number: B-001 | Strength: 100 mg",
                "segments": [
                    {"role": "title", "text": "Batch Inventory"},
                    {"role": "header", "text": "Batch Number | Strength"},
                    {"role": "row", "text": "Batch Number: B-001 | Strength: 100 mg"},
                ],
            },
            {
                "evidence_id": "ce_eq_001",
                "source_type": "text",
                "source_id": "eq_001",
                "page": 1,
                "bbox": [10.0, 210.0, 220.0, 240.0],
                "semantic_role": "display_equation",
                "content_text": "fr(h, t) = rT(h ⋆t) (1)",
                "segments": [
                    {"role": "equation", "text": "fr(h, t) = rT(h ⋆t) (1)", "equation_label": "(1)"},
                ],
            },
        ],
        "toc_sequences": [
            {
                "toc_sequence_id": "tocseq_001",
                "semantic_role": "toc_outline_sequence",
                "title": "TABLE OF CONTENTS",
                "toc_ids": ["toc_001"],
                "pages": [2],
                "page_count": 1,
                "page_span": [2, 2],
                "bbox": [10.0, 120.0, 220.0, 220.0],
                "entry_count": 1,
                "root_entry_count": 1,
                "leaf_entry_count": 1,
                "max_branching_factor": 0,
                "max_entry_level": 1,
                "max_outline_depth": 2,
                "missing_page_locator_count": 0,
                "page_locator_kinds": {"arabic": 1, "roman": 0, "unknown": 0},
                "review_required": True,
                "entries": [
                    {
                        "sequence_entry_index": 1,
                        "toc_id": "toc_001",
                        "page": 2,
                        "outline_index": "1.0",
                        "text": "Overview",
                        "page_locator": "3",
                        "page_locator_kind": "arabic",
                        "page_locator_value": 3,
                        "level": 1,
                        "parent_sequence_entry_index": None,
                        "parent_toc_id": None,
                        "section_anchor_sequence_entry_index": 1,
                        "review_required": False,
                        "audit_flags": [],
                    }
                ],
                "root_nodes": [
                    {
                        "sequence_entry_index": 1,
                        "outline_index": "1.0",
                        "text": "Overview",
                        "page": 2,
                        "toc_id": "toc_001",
                        "child_count": 0,
                        "has_children": False,
                        "children": [],
                    }
                ],
                "navigation_summary": {
                    "page_entry_spans": [
                        {
                            "page": 2,
                            "toc_ids": ["toc_001"],
                            "entry_count": 1,
                            "first_sequence_entry_index": 1,
                            "last_sequence_entry_index": 1,
                            "root_entry_indices": [1],
                            "cross_page_parent_entry_count": 0,
                        }
                    ],
                    "root_sections": [],
                    "outline_index_lookup": {"1.0": [1]},
                    "outline_path_lookup": {"1.0": [1]},
                    "cross_page_parent_link_count": 0,
                    "cross_page_root_section_count": 0,
                },
                "toc_sequence_diagnostics": {
                    "review_required": True,
                    "review_item_count": 1,
                    "aggregated_review_item_count": 1,
                    "mixed_page_numbering": False,
                },
            }
        ],
        "atomic_facts": {
            "drug_name": "ExampleDrug",
            "dosage_form": "Injection",
        },
        "equation_blocks": [
            {
                "equation_id": "eq_001",
                "page": 1,
                "bbox": [10.0, 210.0, 220.0, 240.0],
                "text": "fr(h, t) = rT(h ⋆t) (1)",
                "equation_label": "(1)",
                "semantic_role": "display_equation",
            }
        ],
        "table_asts": [
            {
                "table_id": "tbl_001",
                "page": 1,
                "bbox": [10.0, 40.0, 200.0, 100.0],
                "review_required": True,
                "review_reasons": ["Table structure is unstable."],
                "diagnostics": {
                    "possible_missing_table_content": True,
                    "candidate_count": 2,
                    "candidates": [{"row_index": 3}],
                },
            }
        ],
        "toc_blocks": [
            {
                "toc_id": "toc_001",
                "page": 2,
                "bbox": [10.0, 120.0, 220.0, 220.0],
                "toc_diagnostics": {
                    "review_required": True,
                    "review_item_count": 1,
                    "aggregated_review_item_count": 1,
                    "review_items": [
                        {
                            "code": "missing_page_locator",
                            "severity": "medium",
                            "entry_index": 1,
                            "message": "TOC entry is missing a visible page locator.",
                        }
                    ],
                    "aggregated_review_items": [
                        {
                            "code": "missing_page_locator",
                            "severity": "medium",
                            "item_count": 2,
                            "entry_index_range": [1, 2],
                            "message": "Section contains 2 child TOC entries without visible page locators.",
                        }
                    ],
                },
            }
        ],
    }


def _build_word_like_document() -> dict[str, object]:
    return {
        "filename": "m2-overview.docx",
        "file_id": "file_word_001",
        "source_type": "word",
        "source_path": "D:\\submission\\m2\\m2-overview.docx",
        "paragraphs": [
            {"index": 0, "text": "1. Overview"},
            {"index": 1, "text": "This section summarizes the dossier scope."},
            {"index": 2, "text": "1.1 Scope"},
            {"index": 3, "text": "The filing currently covers quality and administrative content."},
        ],
        "document_ast": {
            "source_type": "docx",
            "blocks": [
                {"block_type": "paragraph", "block_id": "p_0001", "text": "1. Overview"},
                {"block_type": "paragraph", "block_id": "p_0002", "text": "This section summarizes the dossier scope."},
                {"block_type": "paragraph", "block_id": "p_0003", "text": "1.1 Scope"},
                {"block_type": "paragraph", "block_id": "p_0004", "text": "The filing currently covers quality and administrative content."},
            ],
            "block_count": 4,
        },
        "text": "\n".join(
            [
                "1. Overview",
                "This section summarizes the dossier scope.",
                "1.1 Scope",
                "The filing currently covers quality and administrative content.",
            ]
        ),
        "atomic_facts": {},
        "metadata": {
            "paragraph_count": 4,
            "page_count": None,
            "parser_hint": "docx-native",
        },
    }


def _build_word_like_document_with_semantic_outline_sorting() -> dict[str, object]:
    return {
        "filename": "m2-semantic-order.docx",
        "file_id": "file_word_002",
        "source_type": "word",
        "source_path": "D:\\submission\\m2\\m2-semantic-order.docx",
        "paragraphs": [
            {"index": 0, "text": "1. Overview"},
            {"index": 1, "text": "Parent overview body."},
            {"index": 2, "text": "1.10 Late subsection"},
            {"index": 3, "text": "Late subsection body."},
            {"index": 4, "text": "1.2 Early subsection"},
            {"index": 5, "text": "Early subsection body."},
        ],
        "document_ast": {
            "source_type": "docx",
            "blocks": [
                {"block_type": "paragraph", "block_id": "p_0001", "text": "1. Overview"},
                {"block_type": "paragraph", "block_id": "p_0002", "text": "Parent overview body."},
                {"block_type": "paragraph", "block_id": "p_0003", "text": "1.10 Late subsection"},
                {"block_type": "paragraph", "block_id": "p_0004", "text": "Late subsection body."},
                {"block_type": "paragraph", "block_id": "p_0005", "text": "1.2 Early subsection"},
                {"block_type": "paragraph", "block_id": "p_0006", "text": "Early subsection body."},
            ],
            "block_count": 6,
        },
        "text": "\n".join(
            [
                "1. Overview",
                "Parent overview body.",
                "1.10 Late subsection",
                "Late subsection body.",
                "1.2 Early subsection",
                "Early subsection body.",
            ]
        ),
        "atomic_facts": {},
        "metadata": {
            "paragraph_count": 6,
            "page_count": None,
            "parser_hint": "docx-native",
        },
    }


def _build_pdf_like_document_with_semantic_outline_sorting() -> dict[str, object]:
    return {
        "filename": "semantic-outline-order.pdf",
        "file_id": "file_pdf_002",
        "source_type": "pdf",
        "source_path": "D:\\submission\\semantic-outline-order.pdf",
        "metadata": {"page_count": 2},
        "document_ast": {
            "pages": [{"page": 1}, {"page": 2}],
            "content_evidence_refs": ["ce_text_001", "ce_text_002", "ce_text_003"],
            "content_unit_refs": ["cu_text_001", "cu_text_002", "cu_text_003"],
        },
        "content_units": [
            {
                "unit_id": "cu_text_001",
                "evidence_id": "ce_text_001",
                "source_type": "text",
                "source_id": "txt_001",
                "page": 1,
                "bbox": [10.0, 10.0, 200.0, 30.0],
                "semantic_role": "section_heading",
                "unit_role": "section_heading",
                "unit_index": 1,
                "text": "1 Overview",
                "attributes": {},
                "section_context": {
                    "module_label": "DOC-1",
                    "outline_index": "1",
                    "section_title": "Overview",
                    "anchor_source": "heading",
                },
                "fact_extraction_eligible": False,
            },
            {
                "unit_id": "cu_text_002",
                "evidence_id": "ce_text_002",
                "source_type": "text",
                "source_id": "txt_002",
                "page": 2,
                "bbox": [10.0, 40.0, 200.0, 60.0],
                "semantic_role": "section_heading",
                "unit_role": "section_heading",
                "unit_index": 2,
                "text": "1.10 Late subsection",
                "attributes": {},
                "section_context": {
                    "module_label": "DOC-1",
                    "outline_index": "1.10",
                    "section_title": "Late subsection",
                    "anchor_source": "heading",
                },
                "fact_extraction_eligible": False,
            },
            {
                "unit_id": "cu_text_003",
                "evidence_id": "ce_text_003",
                "source_type": "text",
                "source_id": "txt_003",
                "page": 2,
                "bbox": [10.0, 70.0, 200.0, 90.0],
                "semantic_role": "section_heading",
                "unit_role": "section_heading",
                "unit_index": 3,
                "text": "1.2 Early subsection",
                "attributes": {},
                "section_context": {
                    "module_label": "DOC-1",
                    "outline_index": "1.2",
                    "section_title": "Early subsection",
                    "anchor_source": "heading",
                },
                "fact_extraction_eligible": False,
            },
        ],
        "content_evidence": [
            {
                "evidence_id": "ce_text_001",
                "source_type": "text",
                "source_id": "txt_001",
                "page": 1,
                "bbox": [10.0, 10.0, 200.0, 30.0],
                "semantic_role": "section_heading",
                "content_text": "1 Overview",
                "section_context": {
                    "module_label": "DOC-1",
                    "outline_index": "1",
                    "section_title": "Overview",
                    "anchor_source": "heading",
                },
                "segments": [{"role": "heading", "text": "1 Overview"}],
            },
            {
                "evidence_id": "ce_text_002",
                "source_type": "text",
                "source_id": "txt_002",
                "page": 2,
                "bbox": [10.0, 40.0, 200.0, 60.0],
                "semantic_role": "section_heading",
                "content_text": "1.10 Late subsection",
                "section_context": {
                    "module_label": "DOC-1",
                    "outline_index": "1.10",
                    "section_title": "Late subsection",
                    "anchor_source": "heading",
                },
                "segments": [{"role": "heading", "text": "1.10 Late subsection"}],
            },
            {
                "evidence_id": "ce_text_003",
                "source_type": "text",
                "source_id": "txt_003",
                "page": 2,
                "bbox": [10.0, 70.0, 200.0, 90.0],
                "semantic_role": "section_heading",
                "content_text": "1.2 Early subsection",
                "section_context": {
                    "module_label": "DOC-1",
                    "outline_index": "1.2",
                    "section_title": "Early subsection",
                    "anchor_source": "heading",
                },
                "segments": [{"role": "heading", "text": "1.2 Early subsection"}],
            },
        ],
        "atomic_facts": {},
    }


class MaterialReviewContractTests(unittest.TestCase):
    def test_contract_projects_rule_ready_documents_units_navigation_facts_and_diagnostics(self) -> None:
        contract = build_material_review_contract(
            [_build_sample_document()],
            generated_at="2026-03-21T00:00:00Z",
        )

        self.assertEqual(contract["schema_version"], MATERIAL_REVIEW_CONTRACT_VERSION)
        self.assertEqual(contract["generated_at"], "2026-03-21T00:00:00Z")
        self.assertEqual(contract["document_count"], 1)
        self.assertEqual(contract["index_summary"]["evidence_count"], 4)
        self.assertEqual(contract["index_summary"]["unit_count"], 3)
        self.assertEqual(contract["index_summary"]["navigation_sequence_count"], 1)
        self.assertEqual(contract["index_summary"]["section_anchor_count"], 1)
        self.assertEqual(contract["index_summary"]["paragraph_count"], 1)
        self.assertEqual(contract["index_summary"]["section_tree_node_count"], 1)
        self.assertEqual(contract["index_summary"]["fact_count"], 2)
        self.assertEqual(contract["index_summary"]["fact_signal_count"], 2)
        self.assertEqual(contract["index_summary"]["diagnostic_count"], 4)

        document = contract["documents"][0]
        self.assertEqual(document["document_id"], "doc_001")
        self.assertEqual(document["classification"]["module_label"], "DOC-1")
        self.assertFalse(document["classification"]["quality_overview_candidate"])
        self.assertEqual(document["summary"]["fact_signal_count"], 2)
        self.assertEqual(document["summary"]["fact_with_unit_provenance_count"], 2)
        self.assertEqual(document["summary"]["fact_without_unit_provenance_count"], 0)
        self.assertEqual(document["summary"]["toc_sequence_count"], 1)
        self.assertEqual(document["parser_diagnostics_summary"]["review_required_table_count"], 1)
        self.assertEqual(document["structure_refs"]["toc_sequence_refs"], ["tocseq_001"])
        self.assertEqual(document["navigation_sequence_ids"], ["tocseq_001"])
        self.assertEqual(document["fact_keys"], ["dosage_form", "drug_name"])

        unit = contract["unit_index"][0]
        self.assertEqual(unit["document_id"], "doc_001")
        self.assertEqual(unit["filename"], "sample-ind.pdf")
        self.assertEqual(unit["section_context"]["outline_index"], "3.2.S.1")

        evidence = contract["evidence_index"][0]
        self.assertEqual(evidence["document_id"], "doc_001")
        self.assertEqual(evidence["filename"], "sample-ind.pdf")
        self.assertEqual(evidence["segment_count"], 1)
        self.assertEqual(evidence["section_context"]["section_title"], "General Information")

        fact_signal = contract["fact_signal_index"][0]
        self.assertEqual(fact_signal["document_id"], "doc_001")
        self.assertEqual(fact_signal["fact_key"], "drug_name")
        self.assertEqual(fact_signal["fact_value"], "ExampleDrug")
        self.assertEqual(fact_signal["unit_id"], "cu_text_001")

        navigation = contract["navigation_index"][0]
        self.assertEqual(navigation["document_id"], "doc_001")
        self.assertEqual(navigation["toc_sequence_id"], "tocseq_001")
        self.assertEqual(navigation["entry_count"], 1)
        self.assertEqual(navigation["entries"][0]["outline_index"], "1.0")
        self.assertEqual(navigation["navigation_summary"]["outline_path_lookup"]["1.0"], [1])

        section_anchor = contract["section_index"][0]
        self.assertEqual(section_anchor["document_id"], "doc_001")
        self.assertEqual(section_anchor["module_label"], "M3")
        self.assertEqual(section_anchor["outline_index"], "3.2.S.1")
        self.assertEqual(section_anchor["section_title"], "General Information")
        self.assertIsNone(section_anchor["parent_outline_index"])
        self.assertEqual(section_anchor["outline_path"], "3.2.S.1")
        self.assertEqual(section_anchor["tree_depth"], 1)

        paragraph_entry = contract["paragraph_index"][0]
        self.assertEqual(paragraph_entry["document_id"], "doc_001")
        self.assertEqual(paragraph_entry["module_label"], "M3")
        self.assertEqual(paragraph_entry["outline_index"], "3.2.S.1")
        self.assertEqual(paragraph_entry["page"], 1)
        self.assertIn("Drug Name: ExampleDrug", paragraph_entry["text"])
        self.assertIsNone(paragraph_entry["parent_outline_index"])
        self.assertEqual(paragraph_entry["outline_path"], "3.2.S.1")
        self.assertEqual(paragraph_entry["tree_depth"], 1)

        section_tree = contract["section_tree"]
        self.assertEqual(len(section_tree), 1)
        self.assertEqual(section_tree[0]["outline_index"], "3.2.S.1")
        self.assertEqual(section_tree[0]["section_title"], "General Information")
        self.assertEqual(section_tree[0]["paragraph_count"], 1)
        self.assertEqual(section_tree[0]["page_span"], [1, 1])
        self.assertIsNone(section_tree[0]["parent_outline_index"])
        self.assertIsNone(section_tree[0]["parent_section_node_id"])
        self.assertEqual(section_tree[0]["outline_path"], "3.2.S.1")
        self.assertEqual(section_tree[0]["tree_depth"], 1)
        self.assertEqual(section_tree[0]["ancestor_outline_indices"], [])

        fact_index = {
            (item["fact_key"], item["fact_value"])
            for item in contract["fact_index"]
        }
        self.assertEqual(
            fact_index,
            {
                ("drug_name", "ExampleDrug"),
                ("dosage_form", "Injection"),
            },
        )
        fact_provenance_by_key = {
            item["fact_key"]: item["provenance"]
            for item in contract["fact_index"]
        }
        self.assertEqual(fact_provenance_by_key["drug_name"]["granularity"], "unit_level")
        self.assertEqual(fact_provenance_by_key["dosage_form"]["granularity"], "unit_level")
        self.assertEqual(fact_provenance_by_key["drug_name"]["supporting_unit_count"], 1)
        self.assertEqual(fact_provenance_by_key["dosage_form"]["supporting_unit_count"], 1)

        diagnostic_codes = [
            (item["source_type"], item["diagnostic_scope"], item["code"])
            for item in contract["diagnostic_index"]
        ]
        self.assertEqual(
            diagnostic_codes,
            [
                ("table", "parser", "possible_missing_table_content"),
                ("table", "parser", "table_review_reason"),
                ("toc", "parser", "missing_page_locator"),
                ("toc", "parser_aggregated", "missing_page_locator"),
            ],
        )

    def test_persist_normalized_artifacts_emits_review_contract_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            context = create_run_context(
                project_root=project_root,
                job_id="job12345678",
                file_records=[],
            )

            persist_normalized_artifacts(context, [_build_sample_document()])

            normalized_refs = [
                Path(ref).as_posix()
                for ref in context.manifest["artifacts_index"]["normalized"]
            ]
            self.assertEqual(
                normalized_refs,
                [
                    "artifacts/normalized/material_normalized.json",
                    "artifacts/normalized/material_review_contract.json",
                ],
            )

            review_contract_path = context.run_dir / "artifacts" / "normalized" / "material_review_contract.json"
            normalized_path = context.run_dir / "artifacts" / "normalized" / "material_normalized.json"
            atomic_path = context.run_dir / "artifacts" / "atomic_facts" / "atomic_facts.json"

            self.assertTrue(review_contract_path.exists())
            self.assertTrue(normalized_path.exists())
            self.assertTrue(atomic_path.exists())

            review_contract_payload = json.loads(review_contract_path.read_text(encoding="utf-8"))
            self.assertEqual(review_contract_payload["schema_version"], MATERIAL_REVIEW_CONTRACT_VERSION)
            self.assertEqual(review_contract_payload["document_count"], 1)

            normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
            self.assertEqual(normalized_payload["documents"][0]["toc_count"], 1)
            self.assertEqual(normalized_payload["documents"][0]["toc_sequence_count"], 1)

    def test_persist_document_artifacts_emits_equation_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            context = create_run_context(
                project_root=project_root,
                job_id="job12345678",
                file_records=[{"filename": "sample-ind.pdf", "suffix": ".pdf", "path": "D:\\sample-ind.pdf"}],
            )

            persist_document_artifacts(
                context,
                doc_index=0,
                parsed_document=_build_sample_document(),
                file_record={"filename": "sample-ind.pdf", "suffix": ".pdf", "path": "D:\\sample-ind.pdf"},
            )

            self.assertIn("equations", context.manifest["artifacts_index"])
            equation_refs = [Path(ref).as_posix() for ref in context.manifest["artifacts_index"]["equations"]]
            self.assertEqual(equation_refs, ["artifacts/equations/eq_001.json"])

            equation_path = context.run_dir / "artifacts" / "equations" / "eq_001.json"
            self.assertTrue(equation_path.exists())
            equation_payload = json.loads(equation_path.read_text(encoding="utf-8"))
            self.assertEqual(equation_payload["equation_id"], "eq_001")
            self.assertEqual(equation_payload["equation_label"], "(1)")

    def test_persist_document_artifacts_emits_algorithm_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            context = create_run_context(
                project_root=project_root,
                job_id="job12345678",
                file_records=[{"filename": "sample-ind.pdf", "suffix": ".pdf", "path": "D:\\sample-ind.pdf"}],
            )

            document = _build_sample_document()
            document["algorithm_blocks"] = [
                {
                    "algorithm_id": "alg_001",
                    "page": 1,
                    "bbox": [10.0, 245.0, 220.0, 320.0],
                    "algorithm_ref": "Algorithm 1",
                    "title": "Algorithm 1. Example procedure",
                    "content_text": "Algorithm 1. Example procedure\n1. Initialize W",
                    "lines": ["Algorithm 1. Example procedure", "1. Initialize W"],
                    "line_count": 2,
                    "continued_from_previous_page": False,
                    "continues_to_next_page": False,
                    "semantic_role": "algorithm_pseudocode",
                }
            ]

            persist_document_artifacts(
                context,
                doc_index=0,
                parsed_document=document,
                file_record={"filename": "sample-ind.pdf", "suffix": ".pdf", "path": "D:\\sample-ind.pdf"},
            )

            self.assertIn("algorithms", context.manifest["artifacts_index"])
            algorithm_refs = [Path(ref).as_posix() for ref in context.manifest["artifacts_index"]["algorithms"]]
            self.assertEqual(algorithm_refs, ["artifacts/algorithms/alg_001.json"])

            algorithm_path = context.run_dir / "artifacts" / "algorithms" / "alg_001.json"
            self.assertTrue(algorithm_path.exists())
            algorithm_payload = json.loads(algorithm_path.read_text(encoding="utf-8"))
            self.assertEqual(algorithm_payload["algorithm_id"], "alg_001")
            self.assertEqual(algorithm_payload["algorithm_ref"], "Algorithm 1")

    def test_contract_projects_algorithm_refs_and_algorithm_evidence_as_first_class_structure(self) -> None:
        document = _build_sample_document()
        document["metadata"]["algorithm_count"] = 1
        document["metadata"]["content_evidence_counts"]["algorithm"] = 1
        document["metadata"]["content_unit_counts"]["algorithm"] = 1
        document["document_ast"]["algorithm_refs"] = ["alg_001"]
        document["document_ast"]["content_evidence_refs"].append("ce_algorithm_001")
        document["document_ast"]["content_unit_refs"].append("cu_algorithm_001")
        document["content_evidence"].append(
            {
                "evidence_id": "ce_algorithm_001",
                "source_type": "algorithm",
                "source_id": "alg_001",
                "page": 1,
                "bbox": [10.0, 245.0, 220.0, 320.0],
                "semantic_role": "algorithm_pseudocode",
                "content_text": "Algorithm 1. Example procedure\n1. Initialize W",
                "segments": [
                    {"role": "title", "text": "Algorithm 1. Example procedure"},
                    {"role": "step", "text": "1. Initialize W"},
                ],
            }
        )
        document["content_units"].append(
            {
                "unit_id": "cu_algorithm_001",
                "evidence_id": "ce_algorithm_001",
                "source_type": "algorithm",
                "source_id": "alg_001",
                "page": 1,
                "bbox": [10.0, 245.0, 220.0, 320.0],
                "semantic_role": "algorithm_pseudocode",
                "unit_role": "step",
                "unit_index": 1,
                "text": "1. Initialize W",
                "attributes": {"algorithm_ref": "Algorithm 1"},
                "fact_extraction_eligible": False,
            }
        )

        contract = build_material_review_contract([document], generated_at="2026-03-21T00:00:00Z")

        structure_refs = contract["documents"][0]["structure_refs"]
        self.assertEqual(structure_refs["algorithm_refs"], ["alg_001"])

        algorithm_evidence = next(
            item for item in contract["evidence_index"] if item["source_type"] == "algorithm"
        )
        self.assertEqual(algorithm_evidence["source_id"], "alg_001")
        self.assertEqual(algorithm_evidence["segment_roles"], ["title", "step"])

        algorithm_unit = next(
            item for item in contract["unit_index"] if item["source_type"] == "algorithm"
        )
        self.assertEqual(algorithm_unit["unit_role"], "step")
        self.assertFalse(algorithm_unit["fact_extraction_eligible"])

    def test_contract_projects_explicit_ctd_quality_document_classification(self) -> None:
        document = _build_sample_document()
        document["filename"] = "23-quality-overview.pdf"
        document["source_path"] = "D:\\submission\\m2\\23-quality-overview.pdf"

        contract = build_material_review_contract([document], generated_at="2026-03-21T00:00:00Z")

        classification = contract["documents"][0]["classification"]
        self.assertEqual(classification["module_label"], "M2")
        self.assertEqual(classification["module_signal_source"], "source_path")
        self.assertTrue(classification["quality_overview_candidate"])
        self.assertIn("quality_overview", classification["document_tags"])
        self.assertTrue(
            any(hit in {"ctd_2_3", "ctd_2_3_flat"} for hit in classification["signal_hits"])
        )

    def test_contract_builds_section_tree_and_paragraph_index_for_word_like_documents(self) -> None:
        contract = build_material_review_contract(
            [_build_word_like_document()],
            generated_at="2026-04-08T00:00:00Z",
        )

        self.assertEqual(contract["index_summary"]["paragraph_count"], 4)
        self.assertEqual(contract["index_summary"]["section_tree_node_count"], 2)

        paragraph_index = contract["paragraph_index"]
        self.assertEqual(len(paragraph_index), 4)
        self.assertEqual(paragraph_index[0]["outline_index"], "1")
        self.assertEqual(paragraph_index[1]["outline_index"], "1")
        self.assertEqual(paragraph_index[2]["outline_index"], "1.1")
        self.assertEqual(paragraph_index[3]["outline_index"], "1.1")
        self.assertEqual(paragraph_index[0]["module_label"], "M2")
        self.assertIsNone(paragraph_index[0]["page"])
        self.assertIsNone(paragraph_index[0]["parent_outline_index"])
        self.assertEqual(paragraph_index[0]["outline_path"], "1")
        self.assertEqual(paragraph_index[0]["tree_depth"], 1)
        self.assertEqual(paragraph_index[2]["parent_outline_index"], "1")
        self.assertEqual(paragraph_index[2]["outline_path"], "1 > 1.1")
        self.assertEqual(paragraph_index[2]["tree_depth"], 2)

        section_tree = contract["section_tree"]
        self.assertEqual(len(section_tree), 1)
        self.assertEqual(section_tree[0]["outline_index"], "1")
        self.assertEqual(section_tree[0]["paragraph_count"], 2)
        self.assertEqual(len(section_tree[0]["children"]), 1)
        self.assertEqual(section_tree[0]["children"][0]["outline_index"], "1.1")
        self.assertEqual(section_tree[0]["children"][0]["paragraph_count"], 2)
        self.assertIsNone(section_tree[0]["parent_outline_index"])
        self.assertEqual(section_tree[0]["outline_path"], "1")
        self.assertEqual(section_tree[0]["tree_depth"], 1)
        self.assertEqual(section_tree[0]["ancestor_outline_indices"], [])
        self.assertEqual(section_tree[0]["child_outline_indices"], ["1.1"])
        self.assertEqual(section_tree[0]["children"][0]["parent_outline_index"], "1")
        self.assertEqual(section_tree[0]["children"][0]["parent_section_node_id"], "doc_001:sec_1")
        self.assertEqual(section_tree[0]["children"][0]["outline_path"], "1 > 1.1")
        self.assertEqual(section_tree[0]["children"][0]["tree_depth"], 2)
        self.assertEqual(section_tree[0]["children"][0]["ancestor_outline_indices"], ["1"])

    def test_contract_section_tree_sorts_numeric_outline_indices_semantically(self) -> None:
        contract = build_material_review_contract(
            [_build_word_like_document_with_semantic_outline_sorting()],
            generated_at="2026-04-09T00:00:00Z",
        )

        section_tree = contract["section_tree"]
        self.assertEqual(len(section_tree), 1)
        self.assertEqual(section_tree[0]["outline_index"], "1")
        self.assertEqual(
            [child["outline_index"] for child in section_tree[0]["children"]],
            ["1.2", "1.10"],
        )
        self.assertEqual(
            [child["section_title"] for child in section_tree[0]["children"]],
            ["Early subsection", "Late subsection"],
        )

    def test_contract_section_index_sorts_numeric_outline_indices_semantically(self) -> None:
        contract = build_material_review_contract(
            [_build_pdf_like_document_with_semantic_outline_sorting()],
            generated_at="2026-04-09T00:00:00Z",
        )

        self.assertEqual(
            [row["outline_index"] for row in contract["section_index"]],
            ["1", "1.2", "1.10"],
        )
        self.assertEqual(
            [row["section_title"] for row in contract["section_index"]],
            ["Overview", "Early subsection", "Late subsection"],
        )
        self.assertEqual(
            [row["parent_outline_index"] for row in contract["section_index"]],
            [None, "1", "1"],
        )
        self.assertEqual(
            [row["outline_path"] for row in contract["section_index"]],
            ["1", "1 > 1.2", "1 > 1.10"],
        )
        self.assertEqual(
            [row["tree_depth"] for row in contract["section_index"]],
            [1, 2, 2],
        )

    def test_contract_builds_single_document_scope_without_sequence_package_context(self) -> None:
        document = {
            "filename": "cn-regional.xml",
            "source_type": "xml",
            "source_path": "D:\\submission\\x202112345\\0000\\m1\\cn\\cn-regional.xml",
            "metadata": {
                "ectd_application_number": "x202112345",
                "ectd_sequence_number": "0000",
                "ectd_related_sequence_number": "0000",
                "ectd_schema_version": "1.0",
                "ectd_envelope_attributes": {
                    "application-number": "x202112345",
                    "regulatory-activity-type": "initial-application",
                },
            },
        }

        contract = build_material_review_contract([document], generated_at="2026-04-13T00:00:00Z")

        submission_scope = contract["submission_scope"]
        self.assertEqual(submission_scope["upload_mode"], "single_document")
        self.assertEqual(submission_scope["available_scopes"], ["document"])
        self.assertFalse(submission_scope["scope_status"]["sequence"])
        self.assertEqual(submission_scope["ectd_project_context"]["sequence_package_count"], 1)
        self.assertEqual(submission_scope["ectd_project_context"]["regulatory_activity_count"], 1)
        self.assertEqual(submission_scope["ectd_project_context"]["application_project_count"], 1)

    def test_contract_builds_sequence_package_regulatory_activity_and_application_project_context(self) -> None:
        cn_regional = {
            "filename": "cn-regional.xml",
            "source_type": "xml",
            "source_path": "D:\\submission\\x202112345\\0000\\m1\\cn\\cn-regional.xml",
            "metadata": {
                "ectd_application_number": "x202112345",
                "ectd_sequence_number": "0000",
                "ectd_related_sequence_number": "0000",
                "ectd_schema_version": "1.0",
                "ectd_envelope_attributes": {
                    "application-number": "x202112345",
                    "regulatory-activity-type": "initial-application",
                },
            },
        }
        index_xml = {
            "filename": "index.xml",
            "source_type": "xml",
            "source_path": "D:\\submission\\x202112345\\0000\\index.xml",
            "metadata": {
                "ectd_application_number": "x202112345",
                "ectd_sequence_number": "0000",
                "ectd_related_sequence_number": "0000",
            },
        }
        pdf_document = {
            "filename": "study.pdf",
            "source_type": "pdf",
            "source_path": "D:\\submission\\x202112345\\0000\\m5\\study.pdf",
            "metadata": {"page_count": 1},
        }

        contract = build_material_review_contract(
            [cn_regional, index_xml, pdf_document],
            generated_at="2026-04-13T00:00:00Z",
        )

        submission_scope = contract["submission_scope"]
        self.assertEqual(submission_scope["upload_mode"], "ectd_sequence_package")
        self.assertEqual(submission_scope["available_scopes"], ["document", "sequence", "activity", "application"])
        self.assertTrue(submission_scope["scope_status"]["sequence"])
        self.assertTrue(submission_scope["scope_status"]["activity"])
        self.assertTrue(submission_scope["scope_status"]["application"])

        project_context = submission_scope["ectd_project_context"]
        self.assertEqual(project_context["sequence_package_count"], 1)
        self.assertEqual(project_context["regulatory_activity_count"], 1)
        self.assertEqual(project_context["application_project_count"], 1)

        sequence_package = project_context["sequence_packages"][0]
        self.assertEqual(sequence_package["application_key"], "x202112345")
        self.assertEqual(sequence_package["sequence_number"], "0000")
        self.assertEqual(sequence_package["related_sequence_number"], "0000")
        self.assertEqual(sequence_package["regulatory_activity_type"], "initial-application")
        self.assertEqual(set(sequence_package["filenames"]), {"cn-regional.xml", "index.xml"})

        regulatory_activity = project_context["regulatory_activities"][0]
        self.assertEqual(regulatory_activity["application_key"], "x202112345")
        self.assertEqual(regulatory_activity["related_sequence_number"], "0000")
        self.assertEqual(regulatory_activity["regulatory_activity_type"], "initial-application")
        self.assertEqual(regulatory_activity["sequence_numbers"], ["0000"])

        application_project = project_context["application_projects"][0]
        self.assertEqual(application_project["application_key"], "x202112345")
        self.assertEqual(application_project["sequence_count"], 1)
        self.assertEqual(application_project["regulatory_activity_count"], 1)

    def test_contract_builds_sequence_package_context_from_parsed_cn_envelope_child_fields(self) -> None:
        cn_regional_payload = """<?xml version="1.0" encoding="UTF-8"?>
<cn_ectd xmlns:xlink="http://www.w3.org/1999/xlink">
  <cn-envelope>
    <application-id>x202112345</application-id>
    <application-type code="cnapt2" version="1.0" />
    <product-type code="cnprt1" version="1.0" />
    <product-number>ORI-001</product-number>
    <related-sequence>0000</related-sequence>
    <regulatory-activity-type code="cnrat1" version="1.0" />
    <sequence-number>0001</sequence-number>
    <sequence-type code="cnsqt2" version="1.0" />
    <sequence-description>补充提交</sequence-description>
    <sequence-contact>
      <name>张三</name>
      <phone>13600000000</phone>
      <email>a@example.com</email>
    </sequence-contact>
  </cn-envelope>
</cn_ectd>
"""
        index_payload = """<?xml version="1.0" encoding="UTF-8"?>
<ectd:ectd xmlns:ectd="http://www.ich.org/eCTD" xmlns:xlink="http://www.w3.org/1999/xlink">
  <envelope application-number="x202112345" sequence-number="0001" related-sequence="0000" />
</ectd:ectd>
"""
        with TemporaryDirectory() as temp_dir:
            application_root = Path(temp_dir) / "x202112345"
            sequence_root = application_root / "0001"
            cn_regional_path = sequence_root / "m1" / "cn" / "cn-regional.xml"
            cn_regional_path.parent.mkdir(parents=True, exist_ok=True)
            cn_regional_path.write_text(cn_regional_payload, encoding="utf-8")
            index_path = sequence_root / "index.xml"
            index_path.write_text(index_payload, encoding="utf-8")

            parsed_cn_regional = parse_file(cn_regional_path)
            parsed_index = parse_file(index_path)

        contract = build_material_review_contract(
            [parsed_cn_regional, parsed_index],
            generated_at="2026-04-24T00:00:00Z",
        )

        submission_scope = contract["submission_scope"]
        self.assertEqual(submission_scope["upload_mode"], "ectd_sequence_package")
        self.assertTrue(submission_scope["scope_status"]["sequence"])
        sequence_package = submission_scope["ectd_project_context"]["sequence_packages"][0]
        self.assertEqual(sequence_package["application_key"], "x202112345")
        self.assertEqual(sequence_package["sequence_number"], "0001")
        self.assertEqual(sequence_package["related_sequence_number"], "0000")
        self.assertEqual(sequence_package["sequence_type"], "cnsqt2")
        self.assertEqual(sequence_package["application_type"], "cnapt2")
        self.assertEqual(sequence_package["product_type"], "cnprt1")
        self.assertEqual(sequence_package["original_number"], "ORI-001")
        self.assertEqual(sequence_package["regulatory_activity_type"], "cnrat1")

    def test_contract_builds_application_project_context_across_multiple_sequences(self) -> None:
        cn_regional_0000 = {
            "filename": "cn-regional.xml",
            "source_type": "xml",
            "source_path": "D:\\submission\\x202112345\\0000\\m1\\cn\\cn-regional.xml",
            "metadata": {
                "ectd_application_number": "x202112345",
                "ectd_sequence_number": "0000",
                "ectd_related_sequence_number": "0000",
                "ectd_schema_version": "1.0",
                "ectd_envelope_attributes": {
                    "application-number": "x202112345",
                    "regulatory-activity-type": "initial-application",
                },
            },
        }
        cn_regional_0001 = {
            "filename": "cn-regional.xml",
            "source_type": "xml",
            "source_path": "D:\\submission\\x202112345\\0001\\m1\\cn\\cn-regional.xml",
            "metadata": {
                "ectd_application_number": "x202112345",
                "ectd_sequence_number": "0001",
                "ectd_related_sequence_number": "0000",
                "ectd_schema_version": "1.1",
                "ectd_envelope_attributes": {
                    "application-number": "x202112345",
                    "regulatory-activity-type": "initial-application",
                },
            },
        }

        contract = build_material_review_contract(
            [cn_regional_0000, cn_regional_0001],
            generated_at="2026-04-13T00:00:00Z",
        )

        submission_scope = contract["submission_scope"]
        self.assertEqual(submission_scope["upload_mode"], "ectd_application_project")
        project_context = submission_scope["ectd_project_context"]
        self.assertEqual(project_context["sequence_package_count"], 2)
        self.assertEqual(project_context["regulatory_activity_count"], 1)
        self.assertEqual(project_context["application_project_count"], 1)
        self.assertEqual(
            sorted(project_context["application_projects"][0]["sequence_numbers"]),
            ["0000", "0001"],
        )
        packages_by_sequence = {
            package["sequence_number"]: package
            for package in project_context["sequence_packages"]
        }
        self.assertEqual(packages_by_sequence["0000"]["schema_version"], "1.0")
        self.assertEqual(packages_by_sequence["0001"]["schema_version"], "1.1")

    def test_contract_schema_declares_structure_indexes(self) -> None:
        schema_path = Path(__file__).resolve().parents[2] / "schemas" / "material" / "material_review_contract.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertIn("section_index", schema["required"])
        self.assertIn("paragraph_index", schema["required"])
        self.assertIn("section_tree", schema["required"])
        self.assertIn("section_index", schema["properties"])
        self.assertIn("paragraph_index", schema["properties"])
        self.assertIn("section_tree", schema["properties"])

    def test_contract_preserves_xml_schema_location_evidence_for_ectd_documents(self) -> None:
        document = {
            "filename": "cn-regional.xml",
            "source_type": "xml",
            "source_path": "D:\\submission\\0001\\cn-regional.xml",
            "metadata": {
                "xml_schema_location_raw": "cn_ectd util/dtd/cn-regional-1-0.xsd",
                "xml_schema_location_count": 1,
                "xml_schema_location_records": [
                    {
                        "attribute_name": "schemaLocation",
                        "attribute_namespace": "http://www.w3.org/2001/XMLSchema-instance",
                        "namespace": "cn_ectd",
                        "schema_location": "util/dtd/cn-regional-1-0.xsd",
                        "schema_location_normalized": "util/dtd/cn-regional-1-0.xsd",
                        "schema_resolved_path": "D:\\submission\\0001\\util\\dtd\\cn-regional-1-0.xsd",
                        "schema_resolved_path_exists": True,
                        "schema_resolved_filename": "cn-regional-1-0.xsd",
                        "schema_location_is_local_path": True,
                        "schema_location_points_to_util": True,
                    }
                ],
                "xml_schema_locations_all_local": True,
                "xml_schema_locations_all_resolve": True,
                "xml_schema_locations_all_point_to_util": True,
                "xml_schema_validation_attempted": True,
                "xml_schema_validation_schema_path": "D:\\submission\\0001\\util\\dtd\\cn-regional-1-0.xsd",
                "xml_schema_is_valid": True,
                "xml_schema_validation_error_count": 0,
                "xml_schema_validation_errors": [],
                "xml_schema_validation_prerequisite_missing": "",
                "xml_schema_validation_error": "",
            },
            "document_ast": {},
            "content_units": [],
            "content_evidence": [],
            "atomic_facts": {},
        }

        contract = build_material_review_contract([document], generated_at="2026-04-30T00:00:00Z")

        metadata = contract["documents"][0]["ectd_submission_metadata"]
        self.assertEqual(metadata["xml_schema_location_raw"], "cn_ectd util/dtd/cn-regional-1-0.xsd")
        self.assertEqual(metadata["xml_schema_location_count"], 1)
        self.assertTrue(metadata["xml_schema_locations_all_point_to_util"])
        self.assertEqual(
            metadata["xml_schema_location_records"][0]["schema_location"],
            "util/dtd/cn-regional-1-0.xsd",
        )
        self.assertTrue(metadata["xml_schema_validation_attempted"])
        self.assertTrue(metadata["xml_schema_is_valid"])
        self.assertEqual(metadata["xml_schema_validation_error_count"], 0)
        self.assertEqual(metadata.get("xml_schema_validation_prerequisite_missing", ""), "")


if __name__ == "__main__":
    unittest.main()
