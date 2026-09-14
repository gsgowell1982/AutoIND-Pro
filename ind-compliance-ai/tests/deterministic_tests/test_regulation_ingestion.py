from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unicodedata
import unittest
import yaml

from core.regulation_ingestion import (
    REGULATION_CAPABILITY_CROSSWALK_VERSION,
    REGULATION_COVERAGE_REPORT_VERSION,
    REGULATION_GLOSSARY_VERSION,
    REGULATION_LIBRARY_VERSION,
    REGULATION_REFERENCE_MANIFEST_VERSION,
    build_regulation_corpus_entry,
    write_regulation_corpus_entry,
)
from parsers.parser_registry import parse_file


def _resolve_implementation_regulation_doc() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    matches = [
        path
        for path in regulations_root.iterdir()
        if path.suffix.lower() == ".doc"
        and not path.name.startswith("~$")
        and "药品管理法实施条例" in path.name
    ]
    if not matches:
        raise unittest.SkipTest("Implementation regulation doc not found under data/regulations.")
    return matches[0]


def _resolve_registration_classification_doc() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    matches = [
        path
        for path in regulations_root.iterdir()
        if path.suffix.lower() == ".doc"
        and not path.name.startswith("~$")
        and "注册分类" in path.name
    ]
    if not matches:
        raise unittest.SkipTest("Registration classification doc not found under data/regulations.")
    return matches[0]


def _resolve_ectd_technical_spec_pdf() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    expected_name = bytes("eCTD\\u6280\\u672f\\u89c4\\u8303.pdf", "ascii").decode("unicode_escape")
    matches = [
        path
        for path in regulations_root.iterdir()
        if path.suffix.lower() == ".pdf" and path.name == expected_name
    ]
    if matches:
        return matches[0]
    matches = sorted(
        [
            path
            for path in regulations_root.iterdir()
            if path.suffix.lower() == ".pdf"
            and path.name.startswith("eCTD")
            and "实施" not in path.name
            and "验证" not in path.name
            and "Specification" not in path.name
        ],
        key=lambda path: path.name,
    )
    if not matches:
        raise unittest.SkipTest("eCTD technical specification pdf not found under data/regulations.")
    return matches[0]


def _resolve_ectd_validation_standard_pdf() -> Path:
    regulations_root = Path(__file__).resolve().parents[2] / "data" / "regulations"
    expected_name = bytes("eCTD\\u9a8c\\u8bc1\\u6807\\u51c6.pdf", "ascii").decode("unicode_escape")
    matches = [path for path in regulations_root.iterdir() if path.suffix.lower() == ".pdf" and path.name == expected_name]
    if not matches:
        raise unittest.SkipTest("eCTD validation standard pdf not found under data/regulations.")
    return matches[0]


class RegulationIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_path = _resolve_implementation_regulation_doc()
        cls.parsed = parse_file(cls.source_path)
        cls.payload = build_regulation_corpus_entry(cls.source_path)
        cls.registration_source_path = _resolve_registration_classification_doc()
        cls.registration_parsed = parse_file(cls.registration_source_path)
        cls.registration_payload = build_regulation_corpus_entry(cls.registration_source_path)
        cls.ectd_technical_spec_source_path = _resolve_ectd_technical_spec_pdf()
        cls.ectd_technical_spec_parsed = parse_file(cls.ectd_technical_spec_source_path)
        cls.ectd_technical_spec_payload = build_regulation_corpus_entry(cls.ectd_technical_spec_source_path)
        cls.ectd_validation_standard_source_path = _resolve_ectd_validation_standard_pdf()
        cls.ectd_validation_standard_parsed = parse_file(cls.ectd_validation_standard_source_path)
        cls.ectd_validation_standard_payload = build_regulation_corpus_entry(cls.ectd_validation_standard_source_path)

    def _get_ectd_validation_clause(self, article_no_raw: str) -> dict[str, object]:
        for clause in self.ectd_validation_standard_payload["clauses"]:
            if clause["article_no_raw"] == article_no_raw:
                return clause
        self.fail(f"Validation-standard clause not found: {article_no_raw}")

    def test_word_parse_result_preserves_filename_for_regulation_source(self) -> None:
        self.assertEqual(self.parsed.get("filename"), self.source_path.name)
        self.assertEqual(self.parsed.get("source_path"), str(self.source_path))
        self.assertEqual(self.parsed.get("source_type"), "word")

    def test_registration_classification_doc_parses_as_word_and_keeps_ctd_requirement_text(self) -> None:
        self.assertEqual(self.registration_parsed.get("filename"), self.registration_source_path.name)
        self.assertEqual(self.registration_parsed.get("source_path"), str(self.registration_source_path))
        self.assertEqual(self.registration_parsed.get("source_type"), "word")
        self.assertIn("化学药品注册分类", str(self.registration_parsed.get("text") or ""))
        self.assertIn("CTD", str(self.registration_parsed.get("text") or ""))

    def test_build_regulation_corpus_entry_extracts_nine_chapters_and_eighty_nine_articles(self) -> None:
        regulation = self.payload["regulation"]
        chapters = self.payload["chapters"]
        clauses = self.payload["clauses"]

        self.assertEqual(self.payload["schema_version"], REGULATION_LIBRARY_VERSION)
        self.assertEqual(regulation["regulation_id"], "cn_drug_administration_law_implementation_regulation")
        self.assertEqual(regulation["title"], "中华人民共和国药品管理法实施条例")
        self.assertEqual(regulation["chapter_count"], 9)
        self.assertEqual(regulation["article_count"], 89)
        self.assertEqual(len(chapters), 9)
        self.assertEqual(len(clauses), 89)
        self.assertEqual(chapters[0]["heading"], "第一章 总则")
        self.assertEqual(chapters[-1]["heading"], "第九章 附则")
        self.assertEqual(clauses[0]["article_no"], 1)
        self.assertTrue(clauses[0]["normalized_text"].startswith("第一条 根据《中华人民共和国药品管理法》"))
        self.assertEqual(clauses[-1]["article_no"], 89)
        self.assertEqual(clauses[-1]["normalized_text"], "第八十九条 本条例自 2026 年 5 月 15 日起施行。")

    def test_clause_classification_marks_dossier_facing_and_penalty_articles_differently(self) -> None:
        clauses_by_article = {
            int(clause["article_no"]): clause
            for clause in self.payload["clauses"]
        }

        article_6 = clauses_by_article[6]
        article_7 = clauses_by_article[7]
        article_79 = clauses_by_article[79]

        self.assertEqual(article_6["classification"]["material_checkability"], "direct")
        self.assertEqual(article_6["classification"]["recommended_rule_mode"], "hard")
        self.assertIn("general_registration_dossier", article_7["classification"]["expected_material_evidence"])
        self.assertEqual(article_79["classification"]["recommended_rule_mode"], "citation_only")
        self.assertEqual(article_79["clause_type"], "penalty")

    def test_regulation_outputs_keep_stable_cjk_text_for_article_four_and_have_no_replacement_chars(self) -> None:
        clauses_by_article = {
            int(clause["article_no"]): clause
            for clause in self.payload["clauses"]
        }
        article_4 = clauses_by_article[4]["normalized_text"]
        self.assertEqual(
            article_4,
            "第四条 县级以上人民政府承担药品监督管理职责的部门（以下称药品监督管理部门）负责药品监督管理工作。"
            "县级以上人民政府其他有关部门在各自职责范围内负责与药品有关的监督管理工作。",
        )
        for payload_part in (
            self.payload["regulation"]["title"],
            *[chapter["heading"] for chapter in self.payload["chapters"]],
            *[clause["normalized_text"] for clause in self.payload["clauses"]],
        ):
            self.assertNotIn("\ufffd", payload_part)
            self.assertFalse(
                any(
                    unicodedata.category(ch).startswith("C") and ch not in {"\n", "\r", "\t"}
                    for ch in payload_part
                )
            )

    def test_rule_candidates_cover_all_articles_and_preserve_citation_links(self) -> None:
        candidates = self.payload["rule_candidates"]
        self.assertEqual(len(candidates), 89)
        self.assertEqual(candidates[0]["rule_candidate_id"], "cn_drug_administration_law_implementation_regulation:rule_001")
        self.assertEqual(candidates[0]["citation_anchor"], "cn_drug_administration_law_implementation_regulation#art_001")
        self.assertTrue(any(candidate["automation_ready"] for candidate in candidates))
        self.assertTrue(any(candidate["recommended_rule_mode"] == "citation_only" for candidate in candidates))

    def test_direct_rule_drafts_only_promote_direct_candidates(self) -> None:
        draft_catalog = self.payload["direct_rule_drafts"]
        self.assertEqual(draft_catalog["schema_version"], "regulation-rule-draft-v1")
        self.assertEqual(draft_catalog["draft_rule_count"], 9)
        self.assertEqual(draft_catalog["category_counts"], {"hard": 6, "soft": 3})
        self.assertEqual(
            draft_catalog["rules"][0]["rule_id"],
            "DRAFT-HR-cn_drug_administration_law_implementation_regulation-006",
        )
        self.assertTrue(all(rule["status"] == "draft" for rule in draft_catalog["rules"]))
        self.assertTrue(all(rule["implementation_state"] == "planned" for rule in draft_catalog["rules"]))
        self.assertTrue(all(rule["material_checkability"] == "direct" for rule in draft_catalog["rules"]))

    def test_registration_classification_doc_builds_requirement_matrix(self) -> None:
        regulation = self.registration_payload["regulation"]
        requirement_matrix = self.registration_payload["requirement_matrix"]

        self.assertEqual(
            regulation["regulation_id"],
            "cn_drug_registration_classification_and_dossier_requirements",
        )
        self.assertEqual(requirement_matrix["schema_version"], REGULATION_LIBRARY_VERSION)
        self.assertGreaterEqual(requirement_matrix["requirement_count"], 6)

        requirement_ids = {item["requirement_id"] for item in requirement_matrix["requirements"]}
        self.assertIn(
            "cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission",
            requirement_ids,
        )
        self.assertIn(
            "cn_drug_registration_classification_and_dossier_requirements:req_electronic_clinical_trial_database",
            requirement_ids,
        )

        ctd_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission"
        )
        self.assertEqual(ctd_requirement["requirement_type"], "submission_structure")
        self.assertEqual(ctd_requirement["applicable_stage"], "clinical_and_marketing")
        self.assertEqual(ctd_requirement["requirement_level"], "required")
        self.assertIn("CTD", ctd_requirement["requirement_text"])

    def test_registration_classification_doc_extracts_section_fallback_structure(self) -> None:
        regulation = self.registration_payload["regulation"]
        chapters = self.registration_payload["chapters"]
        clauses = self.registration_payload["clauses"]

        self.assertEqual(regulation["chapter_count"], 3)
        self.assertEqual(regulation["article_count"], 17)
        self.assertEqual(len(chapters), 3)
        self.assertEqual(len(clauses), 17)
        self.assertEqual(chapters[0]["chapter_title"], "化学药品注册分类")
        self.assertEqual(chapters[-1]["chapter_title"], "申报资料要求")
        self.assertEqual(clauses[14]["heading"], "（一）")
        self.assertIn("CTD", clauses[14]["normalized_text"])

    def test_write_regulation_corpus_entry_emits_requirement_matrix_for_registration_doc(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            draft_rules_root = output_root / "rules"
            outputs = write_regulation_corpus_entry(
                self.registration_source_path,
                output_root=output_root,
                draft_rules_root=draft_rules_root,
            )

            self.assertIn("requirement_matrix", outputs)
            for path in outputs.values():
                self.assertTrue(path.exists())

            requirement_matrix_payload = json.loads(
                outputs["requirement_matrix"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                requirement_matrix_payload["regulation_id"],
                "cn_drug_registration_classification_and_dossier_requirements",
            )
            self.assertGreaterEqual(requirement_matrix_payload["requirement_count"], 6)

    def test_registration_classification_doc_parses_from_relative_path(self) -> None:
        relative_path = Path("data") / "regulations" / self.registration_source_path.name
        payload = build_regulation_corpus_entry(relative_path)
        self.assertEqual(
            payload["regulation"]["regulation_id"],
            "cn_drug_registration_classification_and_dossier_requirements",
        )
        self.assertGreaterEqual(payload["requirement_matrix"]["requirement_count"], 6)

    def test_ectd_technical_spec_pdf_parses_as_pdf_and_keeps_filename(self) -> None:
        self.assertEqual(
            self.ectd_technical_spec_source_path.name,
            bytes("eCTD\\u6280\\u672f\\u89c4\\u8303.pdf", "ascii").decode("unicode_escape"),
        )
        self.assertEqual(
            self.ectd_technical_spec_parsed.get("filename"),
            self.ectd_technical_spec_source_path.name,
        )
        self.assertEqual(
            self.ectd_technical_spec_parsed.get("source_path"),
            str(self.ectd_technical_spec_source_path),
        )
        self.assertEqual(self.ectd_technical_spec_parsed.get("source_type"), "pdf")
        self.assertIn("eCTD技术规范", str(self.ectd_technical_spec_parsed.get("text") or ""))

    def test_ectd_technical_spec_pdf_extracts_numbered_structure_into_chapters_and_clauses(self) -> None:
        regulation = self.ectd_technical_spec_payload["regulation"]
        chapters = self.ectd_technical_spec_payload["chapters"]
        clauses = self.ectd_technical_spec_payload["clauses"]

        self.assertEqual(regulation["regulation_id"], "cn_ectd_technical_specification")
        self.assertEqual(regulation["title"], "eCTD技术规范")
        self.assertEqual(regulation["chapter_count"], 6)
        self.assertEqual(regulation["article_count"], 38)
        self.assertEqual(regulation["article_count"], len(clauses))
        self.assertEqual(chapters[0]["heading"], "1. 介绍")
        self.assertEqual(chapters[1]["heading"], "2. eCTD 申报资料结构")
        self.assertEqual(chapters[4]["heading"], "5. 参考")
        self.assertEqual(chapters[-1]["heading"], "6. 术语表")
        self.assertTrue(any(clause["heading"] == "1.1 目的" for clause in clauses))
        self.assertTrue(any(clause["heading"] == "2.1.1 申请编号" for clause in clauses))
        self.assertTrue(any(clause["heading"] == "3.3.1 内容文件的格式" for clause in clauses))
        self.assertFalse(
            any(
                clause["heading"].startswith("5.4 参考文献可以不使用STF")
                for clause in clauses
            )
        )

    def test_ectd_technical_spec_pdf_rule_candidates_preserve_numbered_heading_traceability(self) -> None:
        clauses = self.ectd_technical_spec_payload["clauses"]
        candidates = self.ectd_technical_spec_payload["rule_candidates"]
        purpose_clause = next(clause for clause in clauses if clause["heading"] == "1.1 目的")
        sequence_number_clause = next(clause for clause in clauses if clause["heading"] == "2.3.1 序列号")

        self.assertIn("本文档为eCTD 技术规范", purpose_clause["normalized_text"])
        self.assertEqual(
            purpose_clause["source_locator"]["citation_anchor"],
            "cn_ectd_technical_specification#sec_1_1",
        )
        self.assertEqual(
            sequence_number_clause["source_locator"]["citation_anchor"],
            "cn_ectd_technical_specification#sec_2_3_1",
        )
        self.assertEqual(len(candidates), len(clauses))
        self.assertTrue(any(candidate["automation_ready"] for candidate in candidates))
        self.assertTrue(
            any(
                candidate["citation_anchor"] == "cn_ectd_technical_specification#sec_2_3_1"
                for candidate in candidates
            )
        )

    def test_ectd_technical_spec_pdf_builds_requirement_matrix_for_deterministic_submission_checks(self) -> None:
        requirement_matrix = self.ectd_technical_spec_payload["requirement_matrix"]

        self.assertEqual(requirement_matrix["schema_version"], REGULATION_LIBRARY_VERSION)
        self.assertEqual(requirement_matrix["regulation_id"], "cn_ectd_technical_specification")
        self.assertGreaterEqual(requirement_matrix["requirement_count"], 22)

        requirement_ids = {item["requirement_id"] for item in requirement_matrix["requirements"]}
        self.assertIn(
            "cn_ectd_technical_specification:req_application_number_format",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_sequence_number_progression",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_sequence_description_length",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_sequence_contact_information_presence",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_sequence_information_completeness",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_regulatory_activity_information_completeness",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_application_type_controlled_vocabulary_validity",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_product_type_controlled_vocabulary_validity",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_regulatory_activity_type_controlled_vocabulary_validity",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_sequence_type_controlled_vocabulary_validity",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_application_registration_sequence_type_compatibility",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_file_name_character_constraints",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_leaf_xml_lang_attribute_classification",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_replace_leaf_language_class_consistency",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_foreign_reference_leaf_sibling_structure",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_long_pdf_navigation_aids",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_hyperlink_navigation_support",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_no_cross_application_pdf_hyperlinks",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_pdf_hyperlink_target_file_presence",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_pdf_hyperlink_target_resolvability",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_envelope_attributes_required",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_ich_indication_attribute_required",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_ich_manufacturer_attribute_required",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_ich_substance_attribute_required",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_ich_attribute_edge_whitespace_warning",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_stf_required_zone_structure",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_allowed_lifecycle_operation_values",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_non_stf_append_warning",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_cn_regional_xml_root_element",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_cn_regional_xml_schema_version",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_related_sequence_reference",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_clinical_trial_sequence_table1_semantics",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_new_drug_sequence_table2_semantics",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_application_activity_sequence_relationship_examples",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_sequence_description_non_substitution",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_application_information_core_completeness",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_index_dtd_reference_points_to_util_dtd",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_index_xml_valid_against_ich_dtd",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_32r_node_extension_structure_and_title_compliance",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_node_extension_scope_boundary",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_content_file_format_allowed",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_file_and_folder_packaging_boundaries",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_no_empty_directories",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_no_placeholder_documents",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_duplicate_entity_file_submission_within_sequence_warning",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_no_cross_application_leaf_reference",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_leaf_checksum_md5",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_leaf_href_resolves_to_present_file",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_package_xml_envelope_consistency",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_package_xml_leaf_set_consistency",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_package_xml_leaf_checksum_consistency",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_package_declared_file_coverage",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_module1_package_backbone_composition",
            requirement_ids,
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_envelope_controlled_vocabulary_validity",
            requirement_ids,
        )

        sequence_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_sequence_number_progression"
        )
        self.assertEqual(sequence_requirement["requirement_type"], "sequence_numbering")
        self.assertEqual(sequence_requirement["citation_anchor"], "cn_ectd_technical_specification#sec_2_3_1")
        self.assertEqual(sequence_requirement["source_heading"], "2.3.1 序列号")
        self.assertIn("0000", sequence_requirement["requirement_text"])

        sequence_contact_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_sequence_contact_information_presence"
        )
        self.assertEqual(
            sequence_contact_requirement["requirement_type"],
            "sequence_contact_information_presence",
        )
        self.assertEqual(
            sequence_contact_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_2_3_4",
        )
        self.assertIn("sequence-contact", sequence_contact_requirement["requirement_text"])
        self.assertIn("name", sequence_contact_requirement["requirement_text"])
        self.assertIn("phone", sequence_contact_requirement["requirement_text"])
        self.assertIn("email", sequence_contact_requirement["requirement_text"])
        self.assertIn("local cn-regional.xml", sequence_contact_requirement["review_focus"])
        self.assertIn("envelope metadata prerequisite", sequence_contact_requirement["review_focus"])
        self.assertIn("internal cn-contact structure", sequence_contact_requirement["review_focus"])

        sequence_information_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_sequence_information_completeness"
        )
        self.assertEqual(
            sequence_information_requirement["requirement_type"],
            "sequence_information_completeness",
        )
        self.assertEqual(sequence_information_requirement["requirement_level"], "warning")
        self.assertEqual(
            sequence_information_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_2_3",
        )
        self.assertIn("sequence-number", sequence_information_requirement["requirement_text"])
        self.assertIn("sequence-type", sequence_information_requirement["requirement_text"])
        self.assertIn("sequence-description", sequence_information_requirement["requirement_text"])
        self.assertIn("sequence-contact", sequence_information_requirement["requirement_text"])
        self.assertIn("Bounded executable subset", sequence_information_requirement["review_focus"])
        self.assertIn("local sequence package context", sequence_information_requirement["review_focus"])
        self.assertIn("cn-regional.xml envelope metadata", sequence_information_requirement["review_focus"])
        self.assertIn("sec_2_3_1", sequence_information_requirement["review_focus"])
        self.assertIn("sec_2_3_4", sequence_information_requirement["review_focus"])
        self.assertIn("prerequisite guidance", sequence_information_requirement["review_focus"])

        leaf_xml_lang_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_leaf_xml_lang_attribute_classification"
        )
        self.assertEqual(
            leaf_xml_lang_requirement["requirement_type"],
            "leaf_xml_lang_attribute_classification",
        )
        self.assertEqual(
            leaf_xml_lang_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_5_1",
        )
        self.assertIn("xml:lang", leaf_xml_lang_requirement["requirement_text"])
        self.assertIn("ISO639-1", leaf_xml_lang_requirement["requirement_text"])
        self.assertIn("zh", leaf_xml_lang_requirement["requirement_text"])
        self.assertIn("local leaf metadata", leaf_xml_lang_requirement["review_focus"])
        self.assertIn("language classification", leaf_xml_lang_requirement["review_focus"])
        self.assertIn("content-language adequacy", leaf_xml_lang_requirement["review_focus"])

        replace_language_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_replace_leaf_language_class_consistency"
        )
        self.assertEqual(
            replace_language_requirement["requirement_type"],
            "replace_leaf_language_class_consistency",
        )
        self.assertEqual(
            replace_language_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_5_2",
        )
        self.assertIn("replace", replace_language_requirement["requirement_text"])
        self.assertIn("language class", replace_language_requirement["requirement_text"])
        self.assertIn("unique prior-sequence", replace_language_requirement["review_focus"])
        self.assertIn("same application", replace_language_requirement["review_focus"])
        self.assertIn("content-language adequacy", replace_language_requirement["review_focus"])

        foreign_reference_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_foreign_reference_leaf_sibling_structure"
        )
        self.assertEqual(
            foreign_reference_requirement["requirement_type"],
            "foreign_reference_leaf_sibling_structure",
        )
        self.assertEqual(
            foreign_reference_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_5",
        )
        self.assertIn("foreign-reference", foreign_reference_requirement["requirement_text"])
        self.assertIn("Chinese dossier", foreign_reference_requirement["requirement_text"])
        self.assertIn("same-parent", foreign_reference_requirement["review_focus"])
        self.assertIn("local leaf metadata", foreign_reference_requirement["review_focus"])
        self.assertIn("content-language adequacy", foreign_reference_requirement["review_focus"])

        node_extension_scope_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_node_extension_scope_boundary"
        )
        self.assertEqual(
            node_extension_scope_requirement["requirement_type"],
            "node_extension_scope_boundary",
        )
        self.assertEqual(
            node_extension_scope_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_7",
        )
        self.assertIn("node-extension", node_extension_scope_requirement["requirement_text"])
        self.assertIn("biologic", node_extension_scope_requirement["requirement_text"])
        self.assertIn("3.2.R", node_extension_scope_requirement["requirement_text"])
        self.assertIn("product-type evidence", node_extension_scope_requirement["review_focus"])
        self.assertIn("3.2.R", node_extension_scope_requirement["review_focus"])
        self.assertIn("prerequisite guidance", node_extension_scope_requirement["review_focus"])

        checksum_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_leaf_checksum_md5"
        )
        self.assertEqual(checksum_requirement["requirement_type"], "checksum_rule")
        self.assertEqual(checksum_requirement["citation_anchor"], "cn_ectd_technical_specification#sec_4_4")
        self.assertIn("MD5", checksum_requirement["requirement_text"])

        stf_required_zone_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_stf_required_zone_structure"
        )
        self.assertEqual(stf_required_zone_requirement["requirement_type"], "stf_required_zone_structure")
        self.assertEqual(
            stf_required_zone_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_8",
        )
        self.assertIn("Module 4 section 4.2", stf_required_zone_requirement["requirement_text"])
        self.assertIn("external STF specification", stf_required_zone_requirement["review_focus"])

        lifecycle_operation_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_allowed_lifecycle_operation_values"
        )
        self.assertEqual(lifecycle_operation_requirement["requirement_type"], "lifecycle_operation_values")
        self.assertEqual(
            lifecycle_operation_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_9",
        )
        self.assertIn("new", lifecycle_operation_requirement["requirement_text"])
        self.assertIn("replace", lifecycle_operation_requirement["requirement_text"])
        self.assertIn("delete", lifecycle_operation_requirement["requirement_text"])
        self.assertIn("append", lifecycle_operation_requirement["requirement_text"])

        non_stf_append_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_non_stf_append_warning"
        )
        self.assertEqual(non_stf_append_requirement["requirement_type"], "non_stf_append_warning")
        self.assertEqual(non_stf_append_requirement["requirement_level"], "warning")
        self.assertEqual(
            non_stf_append_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_9",
        )
        self.assertIn("non-STF", non_stf_append_requirement["review_focus"])
        self.assertIn("explanation-letter", non_stf_append_requirement["review_focus"])

        cn_regional_root_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_cn_regional_xml_root_element"
        )
        self.assertEqual(cn_regional_root_requirement["requirement_type"], "xml_root_element")
        self.assertEqual(
            cn_regional_root_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_4_2",
        )
        self.assertIn("cn_ectd", cn_regional_root_requirement["requirement_text"])
        self.assertIn("local parsed cn-regional.xml evidence", cn_regional_root_requirement["review_focus"])
        self.assertIn("namespace declaration", cn_regional_root_requirement["review_focus"])

        cn_regional_schema_version_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_cn_regional_xml_schema_version"
        )
        self.assertEqual(cn_regional_schema_version_requirement["requirement_type"], "xml_schema_version")
        self.assertEqual(
            cn_regional_schema_version_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_4_2",
        )
        self.assertIn("schema-version", cn_regional_schema_version_requirement["requirement_text"])
        self.assertIn(
            "namespace declaration prose remains outside hard enforcement",
            cn_regional_schema_version_requirement["review_focus"],
        )

        module1_backbone_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_module1_package_backbone_composition"
        )
        self.assertEqual(
            module1_backbone_requirement["requirement_type"],
            "module1_package_backbone_composition",
        )
        self.assertEqual(module1_backbone_requirement["requirement_level"], "warning")
        self.assertEqual(
            module1_backbone_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_4_1",
        )
        self.assertIn("module-1", module1_backbone_requirement["requirement_text"])
        self.assertIn("cn-regional.xml", module1_backbone_requirement["requirement_text"])
        self.assertIn("index.xml", module1_backbone_requirement["requirement_text"])
        self.assertIn("envelope", module1_backbone_requirement["requirement_text"])
        self.assertIn("content", module1_backbone_requirement["requirement_text"])
        self.assertIn("Bounded executable subset", module1_backbone_requirement["review_focus"])
        self.assertIn("local package evidence", module1_backbone_requirement["review_focus"])
        self.assertIn("sec_4_2", module1_backbone_requirement["review_focus"])
        self.assertIn("sec_4_3", module1_backbone_requirement["review_focus"])
        self.assertIn("sec_4_4", module1_backbone_requirement["review_focus"])
        self.assertIn("prerequisite guidance", module1_backbone_requirement["review_focus"])

        related_sequence_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_related_sequence_reference"
        )
        self.assertEqual(related_sequence_requirement["requirement_type"], "related_sequence_reference")
        self.assertEqual(related_sequence_requirement["requirement_level"], "warning")
        self.assertEqual(
            related_sequence_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_2_2_2",
        )
        self.assertIn("related-sequence", related_sequence_requirement["requirement_text"])
        self.assertIn("not after the current sequence", related_sequence_requirement["requirement_text"])
        self.assertIn("same regulatory activity", related_sequence_requirement["review_focus"])
        self.assertIn("not a generic previous-sequence pointer", related_sequence_requirement["review_focus"])

        clinical_trial_sequence_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_clinical_trial_sequence_table1_semantics"
        )
        self.assertEqual(clinical_trial_sequence_requirement["requirement_type"], "sequence_semantics")
        self.assertEqual(clinical_trial_sequence_requirement["source_clause_id"], "cn_ectd_technical_specification:sec_2_2_2")
        self.assertIn("0000", clinical_trial_sequence_requirement["requirement_text"])
        self.assertIn("0009", clinical_trial_sequence_requirement["requirement_text"])
        self.assertIn("not a maximum", clinical_trial_sequence_requirement["requirement_text"])
        self.assertIn("HR-ECTD-118", clinical_trial_sequence_requirement["review_focus"])

        regulatory_activity_information_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_regulatory_activity_information_completeness"
        )
        self.assertEqual(
            regulatory_activity_information_requirement["requirement_type"],
            "regulatory_activity_information_completeness",
        )
        self.assertEqual(regulatory_activity_information_requirement["requirement_level"], "warning")
        self.assertEqual(
            regulatory_activity_information_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_2_2",
        )
        self.assertIn(
            "regulatory-activity-type",
            regulatory_activity_information_requirement["requirement_text"],
        )
        self.assertIn(
            "related-sequence",
            regulatory_activity_information_requirement["requirement_text"],
        )
        self.assertIn(
            "Bounded executable subset",
            regulatory_activity_information_requirement["review_focus"],
        )
        self.assertIn(
            "local regulatory activity context",
            regulatory_activity_information_requirement["review_focus"],
        )
        self.assertIn(
            "cn-regional.xml envelope metadata",
            regulatory_activity_information_requirement["review_focus"],
        )
        self.assertIn(
            "same regulatory activity",
            regulatory_activity_information_requirement["review_focus"],
        )
        self.assertIn(
            "not a generic previous-sequence pointer",
            regulatory_activity_information_requirement["review_focus"],
        )
        self.assertIn("sec_2_2_1", regulatory_activity_information_requirement["review_focus"])
        self.assertIn("sec_2_2_2", regulatory_activity_information_requirement["review_focus"])
        self.assertIn("prerequisite guidance", regulatory_activity_information_requirement["review_focus"])

        application_information_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_application_information_core_completeness"
        )
        self.assertEqual(
            application_information_requirement["requirement_type"],
            "application_information_core_completeness",
        )
        self.assertEqual(application_information_requirement["requirement_level"], "warning")
        self.assertEqual(
            application_information_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_2_1",
        )
        self.assertIn("application-number", application_information_requirement["requirement_text"])
        self.assertIn("application-type", application_information_requirement["requirement_text"])
        self.assertIn("product-type", application_information_requirement["requirement_text"])
        self.assertIn("2.1.4", application_information_requirement["review_focus"])
        self.assertIn("review-only", application_information_requirement["review_focus"])

        index_dtd_reference_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_index_dtd_reference_points_to_util_dtd"
        )
        self.assertEqual(
            index_dtd_reference_requirement["requirement_type"],
            "index_dtd_reference",
        )
        self.assertEqual(
            index_dtd_reference_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_1_3",
        )
        self.assertIn("index.xml", index_dtd_reference_requirement["requirement_text"])
        self.assertIn("util/dtd/ich-ectd-3-2.dtd", index_dtd_reference_requirement["requirement_text"])
        self.assertIn("local package evidence", index_dtd_reference_requirement["review_focus"])

        index_xml_validity_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_index_xml_valid_against_ich_dtd"
        )
        self.assertEqual(
            index_xml_validity_requirement["requirement_type"],
            "index_xml_dtd_validity",
        )
        self.assertEqual(index_xml_validity_requirement["requirement_level"], "warning")
        self.assertEqual(
            index_xml_validity_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_1_3",
        )
        self.assertIn("well-formed", index_xml_validity_requirement["requirement_text"])
        self.assertIn("ich-ectd-3-2.dtd", index_xml_validity_requirement["requirement_text"])
        self.assertIn("resolvable local DTD", index_xml_validity_requirement["review_focus"])
        self.assertIn("prerequisite", index_xml_validity_requirement["review_focus"])

        node_extension_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_32r_node_extension_structure_and_title_compliance"
        )
        self.assertEqual(
            node_extension_requirement["requirement_type"],
            "node_extension_structure_and_title",
        )
        self.assertEqual(node_extension_requirement["requirement_level"], "warning")
        self.assertEqual(
            node_extension_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_2",
        )
        self.assertIn("3.2.R", node_extension_requirement["requirement_text"])
        self.assertIn("node-extension", node_extension_requirement["requirement_text"])
        self.assertIn("m3/32-body-data/32r-*", node_extension_requirement["requirement_text"])
        self.assertIn("local index.xml", node_extension_requirement["review_focus"])
        self.assertIn(
            "not infer full reviewer-facing content-category adequacy",
            node_extension_requirement["review_focus"],
        )

        content_file_format_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_content_file_format_allowed"
        )
        self.assertEqual(
            content_file_format_requirement["requirement_type"],
            "content_file_format",
        )
        self.assertEqual(
            content_file_format_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_3_1",
        )
        self.assertIn("PDF", content_file_format_requirement["requirement_text"])
        self.assertIn("XML", content_file_format_requirement["requirement_text"])
        self.assertIn("XPT", content_file_format_requirement["requirement_text"])
        self.assertIn("TXT", content_file_format_requirement["requirement_text"])
        self.assertIn("XSL", content_file_format_requirement["requirement_text"])
        self.assertIn("local leaf href", content_file_format_requirement["review_focus"])
        self.assertIn("content adequacy", content_file_format_requirement["review_focus"])

        file_folder_packaging_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_file_and_folder_packaging_boundaries"
        )
        self.assertEqual(
            file_folder_packaging_requirement["requirement_type"],
            "file_and_folder_packaging_rollup",
        )
        self.assertEqual(file_folder_packaging_requirement["requirement_level"], "warning")
        self.assertEqual(
            file_folder_packaging_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_3",
        )
        self.assertIn("content-file formats", file_folder_packaging_requirement["requirement_text"])
        self.assertIn("empty directories", file_folder_packaging_requirement["requirement_text"])
        self.assertIn("cross-application", file_folder_packaging_requirement["requirement_text"])
        self.assertIn("parent traceability rollup", file_folder_packaging_requirement["review_focus"])
        self.assertIn(
            "does not create new runtime verdicts",
            file_folder_packaging_requirement["review_focus"],
        )
        self.assertIn("content adequacy", file_folder_packaging_requirement["review_focus"])
        self.assertIn("advisory reuse", file_folder_packaging_requirement["review_focus"])

        empty_directory_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_no_empty_directories"
        )
        self.assertEqual(
            empty_directory_requirement["requirement_type"],
            "empty_directory_policy",
        )
        self.assertEqual(
            empty_directory_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_3_3",
        )
        self.assertIn("empty directories", empty_directory_requirement["requirement_text"])
        self.assertIn("local sequence package", empty_directory_requirement["review_focus"])
        self.assertIn("missing dossier-section adequacy", empty_directory_requirement["review_focus"])

        placeholder_document_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_no_placeholder_documents"
        )
        self.assertEqual(
            placeholder_document_requirement["requirement_type"],
            "placeholder_document_policy",
        )
        self.assertEqual(
            placeholder_document_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_3_3",
        )
        self.assertIn("placeholder", placeholder_document_requirement["requirement_text"])
        self.assertIn("not applicable", placeholder_document_requirement["requirement_text"])
        self.assertIn("local parsed document text", placeholder_document_requirement["review_focus"])
        self.assertIn("prerequisite", placeholder_document_requirement["review_focus"])

        duplicate_entity_reuse_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_duplicate_entity_file_submission_within_sequence_warning"
        )
        self.assertEqual(
            duplicate_entity_reuse_requirement["requirement_type"],
            "duplicate_entity_file_reuse_warning",
        )
        self.assertEqual(duplicate_entity_reuse_requirement["requirement_level"], "warning")
        self.assertEqual(
            duplicate_entity_reuse_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_3_4",
        )
        self.assertIn("same checksum", duplicate_entity_reuse_requirement["requirement_text"])
        self.assertIn("reuse", duplicate_entity_reuse_requirement["requirement_text"])
        self.assertIn("advisory", duplicate_entity_reuse_requirement["review_focus"])
        self.assertIn("not a hard invalidation", duplicate_entity_reuse_requirement["review_focus"])

        cross_application_reference_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"] == "cn_ectd_technical_specification:req_no_cross_application_leaf_reference"
        )
        self.assertEqual(
            cross_application_reference_requirement["requirement_type"],
            "cross_application_reference_prohibition",
        )
        self.assertEqual(
            cross_application_reference_requirement["citation_anchor"],
            "cn_ectd_technical_specification#sec_3_3_4",
        )
        self.assertIn("cross-application", cross_application_reference_requirement["requirement_text"])
        self.assertIn("leaf href", cross_application_reference_requirement["review_focus"])
        self.assertIn("local application root", cross_application_reference_requirement["review_focus"])

        application_type_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_application_type_controlled_vocabulary_validity"
        )
        self.assertEqual(
            application_type_requirement["source_heading"],
            "2.1.2 申请类型",
        )
        self.assertEqual(
            application_type_requirement["requirement_type"],
            "controlled_vocabulary",
        )
        self.assertIn(
            "cv-application-type.xml",
            application_type_requirement["review_focus"],
        )

        compatibility_requirement = next(
            item
            for item in requirement_matrix["requirements"]
            if item["requirement_id"]
            == "cn_ectd_technical_specification:req_application_registration_sequence_type_compatibility"
        )
        self.assertEqual(
            compatibility_requirement["source_heading"],
            "2.4 申请、注册行为和序列的关系",
        )
        self.assertEqual(
            compatibility_requirement["requirement_type"],
            "type_compatibility",
        )
        self.assertIn(
            "depend-apt-rat-sqt.xml",
            compatibility_requirement["review_focus"],
        )

    def test_ectd_technical_spec_pdf_builds_glossary_artifact(self) -> None:
        glossary = self.ectd_technical_spec_payload["glossary"]

        self.assertEqual(glossary["schema_version"], REGULATION_LIBRARY_VERSION)
        self.assertEqual(glossary["glossary_schema_version"], REGULATION_GLOSSARY_VERSION)
        self.assertEqual(glossary["regulation_id"], "cn_ectd_technical_specification")
        self.assertEqual(glossary["source_heading"], "6. 术语表")
        self.assertEqual(glossary["term_count"], 19)

        terms_by_no = {item["term_no"]: item for item in glossary["terms"]}
        self.assertEqual(terms_by_no[1]["term"], "电子通用技术文档（eCTD）")
        self.assertIn("XML", terms_by_no[1]["definition"])
        self.assertEqual(terms_by_no[9]["term"], "叶元素（leaf element）")
        self.assertIn("生命周期操作", terms_by_no[9]["definition"])
        self.assertEqual(terms_by_no[17]["term"], "DTD")
        self.assertIn("Document Type Definition", terms_by_no[17]["definition"])
        self.assertEqual(terms_by_no[19]["term"], "OCR")
        self.assertIn("Optical Character Recognition", terms_by_no[19]["definition"])
        self.assertIn("扫描的PDF文件", terms_by_no[19]["definition"])
        self.assertEqual(
            terms_by_no[19]["source_clause_id"],
            "cn_ectd_implementation_guide:sec_11",
        )
        self.assertEqual(terms_by_no[19]["source_filename"], "eCTD实施指南.pdf")

    def test_write_regulation_corpus_entry_emits_glossary_for_ectd_technical_spec(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            draft_rules_root = output_root / "rules"
            outputs = write_regulation_corpus_entry(
                self.ectd_technical_spec_source_path,
                output_root=output_root,
                draft_rules_root=draft_rules_root,
            )

            self.assertIn("glossary", outputs)
            glossary_payload = json.loads(outputs["glossary"].read_text(encoding="utf-8"))
            self.assertEqual(glossary_payload["regulation_id"], "cn_ectd_technical_specification")
            self.assertEqual(glossary_payload["term_count"], 19)
            self.assertEqual(glossary_payload["terms"][-1]["term"], "OCR")

    def test_ectd_technical_spec_pdf_builds_reference_manifest_artifact(self) -> None:
        reference_manifest = self.ectd_technical_spec_payload["reference_manifest"]

        self.assertEqual(reference_manifest["schema_version"], REGULATION_LIBRARY_VERSION)
        self.assertEqual(
            reference_manifest["reference_manifest_schema_version"],
            REGULATION_REFERENCE_MANIFEST_VERSION,
        )
        self.assertEqual(reference_manifest["regulation_id"], "cn_ectd_technical_specification")
        self.assertEqual(reference_manifest["source_heading"], "5. 参考")
        self.assertEqual(reference_manifest["reference_count"], 8)

        references_by_no = {item["ref_no"]: item for item in reference_manifest["references"]}
        self.assertEqual(
            references_by_no[2]["title"],
            "ICH Electronic Common Technical Document Specification V3.2.2",
        )
        self.assertEqual(references_by_no[2]["authority"], "ICH")
        self.assertEqual(references_by_no[2]["normative_role"], "external_normative_dependency")
        self.assertEqual(
            references_by_no[7]["title"],
            "《M4 模块一行政文件和药品信息》",
        )
        self.assertEqual(references_by_no[7]["authority"], "NMPA/CTD")
        self.assertEqual(references_by_no[8]["normative_role"], "supporting_guidance")

    def test_ectd_technical_spec_pdf_builds_chapter_coverage_report_for_chapters_one_to_four(self) -> None:
        coverage_report = self.ectd_technical_spec_payload["coverage_report"]

        self.assertEqual(coverage_report["schema_version"], REGULATION_LIBRARY_VERSION)
        self.assertEqual(coverage_report["coverage_schema_version"], REGULATION_COVERAGE_REPORT_VERSION)
        self.assertEqual(coverage_report["regulation_id"], "cn_ectd_technical_specification")
        self.assertEqual(coverage_report["scope"], "chapter_1_to_4")
        self.assertEqual(coverage_report["chapter_count"], 4)
        self.assertGreaterEqual(coverage_report["clause_count"], 30)
        self.assertEqual(
            coverage_report["coverage_counts"],
            {
                "citation_only_recorded": 2,
                "partially_covered": 12,
                "covered": 20,
                "deferred": 4,
            },
        )
        self.assertEqual(coverage_report["traceability_gap_clause_count"], 0)
        self.assertEqual(coverage_report["traceability_gap_clause_ids"], [])
        self.assertEqual(
            coverage_report["recommended_next_direction"]["primary"],
            "run_technical_specification_closure_qa",
        )
        self.assertIn(
            "partially_covered",
            coverage_report["recommended_next_direction"]["primary_reason"],
        )
        self.assertIn(
            "do not inflate",
            coverage_report["recommended_next_direction"]["primary_reason"],
        )
        self.assertEqual(
            coverage_report["recommended_next_direction"]["secondary"],
            "keep_validation_standard_closed_unless_artifacts_change",
        )

        coverages_by_clause = {
            item["clause_id"]: item
            for item in coverage_report["clause_coverages"]
        }
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_1_1"]["coverage_status"],
            "citation_only_recorded",
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_2_1"]["coverage_status"],
            "partially_covered",
        )
        self.assertIn(
            "SR-ECTD-006",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_1"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_2_1"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_application_information_core_completeness",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_1"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_2_2"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-005",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_2"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_2_2"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_regulatory_activity_information_completeness",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_2"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_1_3"]["coverage_status"],
            "partially_covered",
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_2_2_2"]["coverage_status"],
            "partially_covered",
        )
        self.assertIn(
            "SR-ECTD-003",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_2_2"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_2_2_2"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_related_sequence_reference",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_2_2"]["requirement_ids"],
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_clinical_trial_sequence_table1_semantics",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_2_2"]["requirement_ids"],
        )
        self.assertIn(
            "HR-ECTD-118",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_2_2"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-078",
            coverages_by_clause["cn_ectd_technical_specification:sec_1_3"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-079",
            coverages_by_clause["cn_ectd_technical_specification:sec_1_3"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_1_3"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_index_dtd_reference_points_to_util_dtd",
            coverages_by_clause["cn_ectd_technical_specification:sec_1_3"]["requirement_ids"],
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_index_xml_valid_against_ich_dtd",
            coverages_by_clause["cn_ectd_technical_specification:sec_1_3"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_2_3"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-004",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_3"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_2_3"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_sequence_information_completeness",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_3"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_2_3_4"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-014",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_3_4"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_2_3_4"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_sequence_contact_information_presence",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_3_4"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_2"]["coverage_status"],
            "partially_covered",
        )
        self.assertIn(
            "SR-ECTD-007",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_2"]["implemented_rule_ids"],
        )
        self.assertIn(
            "SR-ECTD-026",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_2"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_2"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_32r_node_extension_structure_and_title_compliance",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_2"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3"]["coverage_status"],
            "partially_covered",
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_file_and_folder_packaging_boundaries",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-024",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_foreign_reference_leaf_sibling_structure",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5_1"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-023",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5_1"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5_1"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_leaf_xml_lang_attribute_classification",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5_1"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5_2"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-026",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5_2"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5_2"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_replace_leaf_language_class_consistency",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_5_2"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_7"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-008",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_7"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-074",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_7"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_7"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_node_extension_scope_boundary",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_7"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_1"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-019",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_1"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_1"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_content_file_format_allowed",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_1"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_3"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-021",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_3"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-022",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_3"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_3"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_no_empty_directories",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_3"]["requirement_ids"],
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_no_placeholder_documents",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_3"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_4"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-011",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_4"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-027",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_4"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_4"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_duplicate_entity_file_submission_within_sequence_warning",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_4"]["requirement_ids"],
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_no_cross_application_leaf_reference",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_3_4"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_6"]["coverage_status"],
            "partially_covered",
        )
        self.assertIn(
            "HR-ECTD-075",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_6"]["implemented_rule_ids"],
        )
        self.assertIn(
            "SR-ECTD-027",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_6"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_6"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_ich_indication_attribute_required",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_6"]["requirement_ids"],
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_ich_attribute_edge_whitespace_warning",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_6"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_8"]["coverage_status"],
            "partially_covered",
        )
        self.assertIn(
            "SR-ECTD-010",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_8"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_8"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_stf_required_zone_structure",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_8"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_9"]["coverage_status"],
            "partially_covered",
        )
        self.assertIn(
            "HR-ECTD-025",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_9"]["implemented_rule_ids"],
        )
        self.assertIn(
            "SR-ECTD-009",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_9"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_3_9"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_allowed_lifecycle_operation_values",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_9"]["requirement_ids"],
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_non_stf_append_warning",
            coverages_by_clause["cn_ectd_technical_specification:sec_3_9"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_4_1"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-015",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_1"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-016",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_1"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-030",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_1"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-031",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_1"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-032",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_1"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_4_1"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_module1_package_backbone_composition",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_1"]["requirement_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_technical_specification:sec_4_2"]["coverage_status"],
            "partially_covered",
        )
        self.assertIn(
            "HR-ECTD-030",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_2"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-036",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_2"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_4_2"]["traceability_gap"]
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_cn_regional_xml_root_element",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_2"]["requirement_ids"],
        )
        self.assertIn(
            "cn_ectd_technical_specification:req_cn_regional_xml_schema_version",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_2"]["requirement_ids"],
        )
        self.assertIn(
            "HR-ECTD-037",
            coverages_by_clause["cn_ectd_technical_specification:sec_2_1_2"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-115",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_3"]["implemented_rule_ids"],
        )
        self.assertIn(
            "HR-ECTD-116",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_3"]["implemented_rule_ids"],
        )
        self.assertFalse(
            coverages_by_clause["cn_ectd_technical_specification:sec_4_3"]["traceability_gap"]
        )
        self.assertIn(
            "cross-sequence immutability",
            coverages_by_clause["cn_ectd_technical_specification:sec_4_3"]["coverage_note"],
        )

        chapter_summaries_by_no = {
            item["chapter_no"]: item
            for item in coverage_report["chapter_summaries"]
        }
        self.assertEqual(chapter_summaries_by_no[1]["citation_only_clause_count"], 2)
        self.assertGreaterEqual(chapter_summaries_by_no[3]["covered_clause_count"], 5)
        self.assertEqual(chapter_summaries_by_no[4]["traceability_gap_clause_count"], 0)
        artifact_ids = {
            item["attachment_id"]
            for item in coverage_report["supporting_artifact_coverage"]
        }
        self.assertIn("cn_ectd_attachment_1_3", artifact_ids)
        self.assertIn("cn_ectd_attachment_2_6", artifact_ids)

        self.assertEqual(
            coverage_report["recommended_next_direction"]["primary"],
            "run_technical_specification_closure_qa",
        )

    def test_ectd_validation_standard_builds_full_six_chapter_coverage_report(self) -> None:
        coverage_report = self.ectd_validation_standard_payload["coverage_report"]

        self.assertEqual(coverage_report["schema_version"], REGULATION_LIBRARY_VERSION)
        self.assertEqual(coverage_report["coverage_schema_version"], REGULATION_COVERAGE_REPORT_VERSION)
        self.assertEqual(coverage_report["regulation_id"], "cn_ectd_validation_standard")
        self.assertEqual(coverage_report["scope"], "chapter_1_to_6")
        self.assertEqual(coverage_report["chapter_count"], 6)
        self.assertEqual(coverage_report["clause_count"], 149)

        coverages_by_clause = {
            item["clause_id"]: item
            for item in coverage_report["clause_coverages"]
        }
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_1_1"]["coverage_status"],
            "citation_only_recorded",
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_2_1"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-021",
            coverages_by_clause["cn_ectd_validation_standard:sec_2_1"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_2_2"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-015",
            coverages_by_clause["cn_ectd_validation_standard:sec_2_2"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_2_6"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-046",
            coverages_by_clause["cn_ectd_validation_standard:sec_2_6"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_2_7"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-047",
            coverages_by_clause["cn_ectd_validation_standard:sec_2_7"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_2_10"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-001",
            coverages_by_clause["cn_ectd_validation_standard:sec_2_10"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_1"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-015",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_1"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_4"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-006",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_4"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_13"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-003",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_13"]["implemented_rule_ids"],
        )

        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_14"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-049",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_14"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_15"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-048",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_15"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_1"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-015",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_1"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_2"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-112",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_2"]["implemented_rule_ids"],
        )
        self.assertIn(
            "schema-location",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_2"]["coverage_note"],
        )
        self.assertIn(
            "util",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_2"]["coverage_note"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_3"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-113",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_3"]["implemented_rule_ids"],
        )
        self.assertIn(
            "schema validation",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_3"]["coverage_note"],
        )
        self.assertIn(
            "prerequisite",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_3"]["coverage_note"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_4"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-114",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_4"]["implemented_rule_ids"],
        )
        self.assertIn(
            "prior sequence",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_4"]["coverage_note"],
        )
        self.assertIn(
            "schema-version",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_4"]["coverage_note"],
        )
        self.assertIn(
            "na",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_4"]["coverage_note"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_2_13"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-115",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_2_13"]["implemented_rule_ids"],
        )
        self.assertIn(
            "application-level envelope",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_2_13"]["coverage_note"],
        )
        self.assertIn(
            "initial sequence",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_2_13"]["coverage_note"],
        )
        self.assertIn(
            "na",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_2_13"]["coverage_note"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_2_14"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-116",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_2_14"]["implemented_rule_ids"],
        )
        self.assertIn(
            "same regulatory activity",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_2_14"]["coverage_note"],
        )
        self.assertIn(
            "related-sequence",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_2_14"]["coverage_note"],
        )
        self.assertIn(
            "na",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_2_14"]["coverage_note"],
        )

        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_5"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-006",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_5"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_6"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-059",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_6"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_8"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-055",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_8"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_9"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-056",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_9"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_10"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-057",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_10"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_11"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-058",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_11"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_12"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-003",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_12"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_13"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-050",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_13"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_14"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-051",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_14"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_15"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-052",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_15"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_16"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-053",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_16"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_17"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-054",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_17"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_18"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-060",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_18"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_19"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-024",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_19"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_21"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-016",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_21"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_22"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-016",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_22"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_23"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-015",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_23"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_1"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-090",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_1"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_2"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-050",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_2"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_17"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-041",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_17"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_18"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-017",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_18"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_16"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-018",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_16"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_20"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-019",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_20"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_24"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-020",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_24"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_9"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-021",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_9"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_10"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-022",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_10"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_1"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-044",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_1"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_11"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-045",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_11"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_12"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-023",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_12"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_13"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-023",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_13"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_19"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-042",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_19"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_21"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-043",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_21"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_23"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-016",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_23"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_25"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-045",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_25"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_22"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-046",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_22"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_26"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-047",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_26"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_14"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-048",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_14"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_7"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-049",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_7"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_4"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-091",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_4"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_2"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-050",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_2"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_3"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-092",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_3"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_5"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-051",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_5"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_6"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-052",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_6"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_8"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-053",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_8"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_6_15"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-054",
            coverages_by_clause["cn_ectd_validation_standard:sec_6_15"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_20"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-061",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_20"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_21"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-062",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_21"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_22"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-086",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_22"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_23"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-028",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_23"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_24"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-029",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_24"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_29"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-087",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_29"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_24"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-093",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_24"]["implemented_rule_ids"],
        )
        for clause_id, rule_id in (
            ("cn_ectd_validation_standard:sec_4_1_25", "HR-ECTD-094"),
            ("cn_ectd_validation_standard:sec_4_1_26", "HR-ECTD-095"),
            ("cn_ectd_validation_standard:sec_4_1_27", "HR-ECTD-096"),
            ("cn_ectd_validation_standard:sec_4_1_28", "HR-ECTD-097"),
            ("cn_ectd_validation_standard:sec_4_1_29", "HR-ECTD-098"),
        ):
            self.assertEqual(coverages_by_clause[clause_id]["coverage_status"], "covered")
            self.assertIn(rule_id, coverages_by_clause[clause_id]["implemented_rule_ids"])
        for clause_id, rule_id in (
            ("cn_ectd_validation_standard:sec_4_2_1", "HR-ECTD-002"),
            ("cn_ectd_validation_standard:sec_4_2_2", "HR-ECTD-037"),
            ("cn_ectd_validation_standard:sec_4_2_3", "HR-ECTD-038"),
            ("cn_ectd_validation_standard:sec_4_2_4", "HR-ECTD-103"),
            ("cn_ectd_validation_standard:sec_4_2_5", "HR-ECTD-102"),
            ("cn_ectd_validation_standard:sec_4_2_6", "HR-ECTD-039"),
            ("cn_ectd_validation_standard:sec_4_2_7", "HR-ECTD-001"),
            ("cn_ectd_validation_standard:sec_4_2_8", "HR-ECTD-040"),
            ("cn_ectd_validation_standard:sec_4_2_9", "HR-ECTD-099"),
            ("cn_ectd_validation_standard:sec_4_2_10", "HR-ECTD-018"),
            ("cn_ectd_validation_standard:sec_4_2_11", "HR-ECTD-100"),
            ("cn_ectd_validation_standard:sec_4_2_12", "HR-ECTD-101"),
        ):
            self.assertEqual(coverages_by_clause[clause_id]["coverage_status"], "covered")
            self.assertIn(rule_id, coverages_by_clause[clause_id]["implemented_rule_ids"])
        for clause_id, rule_id in (
            ("cn_ectd_validation_standard:sec_4_1_30", "SR-ECTD-009"),
            ("cn_ectd_validation_standard:sec_4_1_31", "HR-ECTD-026"),
        ):
            self.assertEqual(coverages_by_clause[clause_id]["coverage_status"], "covered")
            self.assertIn(rule_id, coverages_by_clause[clause_id]["implemented_rule_ids"])
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_3_9"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-055",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_3_9"]["implemented_rule_ids"],
        )
        for clause_id, rule_id in (
            ("cn_ectd_validation_standard:sec_4_3_1", "HR-ECTD-106"),
            ("cn_ectd_validation_standard:sec_4_3_2", "HR-ECTD-107"),
            ("cn_ectd_validation_standard:sec_4_3_3", "HR-ECTD-108"),
            ("cn_ectd_validation_standard:sec_4_3_4", "HR-ECTD-109"),
            ("cn_ectd_validation_standard:sec_4_3_5", "HR-ECTD-110"),
            ("cn_ectd_validation_standard:sec_4_3_6", "HR-ECTD-111"),
            ("cn_ectd_validation_standard:sec_4_3_7", "HR-ECTD-104"),
            ("cn_ectd_validation_standard:sec_4_3_8", "HR-ECTD-105"),
        ):
            self.assertEqual(coverages_by_clause[clause_id]["coverage_status"], "covered")
            self.assertIn(rule_id, coverages_by_clause[clause_id]["implemented_rule_ids"])
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_6"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-063",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_6"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_7"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-064",
            coverages_by_clause["cn_ectd_validation_standard:sec_4_1_7"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_5"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-065",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_5"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_7"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-066",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_7"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_8"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-067",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_8"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_9"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-068",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_9"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_10"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-071",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_10"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_11"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-072",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_11"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_12"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-073",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_12"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_16"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-074",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_16"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_2"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-078",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_2"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_3"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-079",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_3"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_17"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-026",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_17"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_18"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-069",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_18"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_19"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-070",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_19"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_20"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-025",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_20"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_25"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-075",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_25"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_26"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-076",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_26"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_27"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-077",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_27"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_28"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-027",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_28"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_30"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-080",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_30"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_31"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-081",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_31"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_32"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-082",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_32"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_33"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-083",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_33"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_34"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-084",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_34"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_35"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-085",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_35"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_3_36"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-026",
            coverages_by_clause["cn_ectd_validation_standard:sec_3_36"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_2"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-006",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_2"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_3"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-030",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_3"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_4"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-088",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_4"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_5"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-031",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_5"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_6"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-032",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_6"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_7"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-037",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_7"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_8"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-036",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_8"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_18"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-040",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_18"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_19"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-041",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_19"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_20"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-042",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_20"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_10"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-038",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_10"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_13"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-039",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_13"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_12"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-034",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_12"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_14"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-035",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_14"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_15"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-033",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_15"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_16"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "HR-ECTD-089",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_16"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_17"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-010",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_17"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_9"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-043",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_9"]["implemented_rule_ids"],
        )
        self.assertEqual(
            coverages_by_clause["cn_ectd_validation_standard:sec_5_11"]["coverage_status"],
            "covered",
        )
        self.assertIn(
            "SR-ECTD-044",
            coverages_by_clause["cn_ectd_validation_standard:sec_5_11"]["implemented_rule_ids"],
        )

        chapter_summaries_by_no = {
            item["chapter_no"]: item
            for item in coverage_report["chapter_summaries"]
        }
        self.assertEqual(chapter_summaries_by_no[1]["citation_only_clause_count"], 3)
        self.assertGreaterEqual(chapter_summaries_by_no[2]["covered_clause_count"], 10)
        self.assertGreaterEqual(chapter_summaries_by_no[2]["partially_covered_clause_count"], 0)
        self.assertGreaterEqual(chapter_summaries_by_no[3]["covered_clause_count"], 36)
        self.assertGreaterEqual(chapter_summaries_by_no[3]["partially_covered_clause_count"], 0)
        self.assertGreaterEqual(chapter_summaries_by_no[4]["covered_clause_count"], 49)
        self.assertLessEqual(chapter_summaries_by_no[4]["partially_covered_clause_count"], 5)
        self.assertGreaterEqual(chapter_summaries_by_no[5]["covered_clause_count"], 20)
        self.assertGreaterEqual(chapter_summaries_by_no[5]["partially_covered_clause_count"], 0)
        self.assertGreaterEqual(chapter_summaries_by_no[6]["covered_clause_count"], 26)
        self.assertGreaterEqual(chapter_summaries_by_no[6]["partially_covered_clause_count"], 0)
        self.assertEqual(
            coverage_report["recommended_next_direction"]["primary"],
            "resolve_validation_standard_chapter4_prerequisite_evidence_before_more_automation",
        )
        self.assertIn(
            "human-review",
            coverage_report["recommended_next_direction"]["primary_reason"],
        )
        self.assertIn(
            "schema",
            coverage_report["recommended_next_direction"]["primary_reason"],
        )
        self.assertIn(
            "historical",
            coverage_report["recommended_next_direction"]["primary_reason"],
        )

    def test_ectd_validation_standard_coverage_rule_ids_exist_in_rule_metadata(self) -> None:
        coverage_report = self.ectd_validation_standard_payload["coverage_report"]
        metadata_path = Path(__file__).resolve().parents[2] / "rules" / "rule_metadata.yaml"
        rule_metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        metadata_rule_ids = {
            str(item.get("rule_id") or "").strip()
            for item in list(rule_metadata.get("rules") or [])
            if str(item.get("rule_id") or "").strip()
        }
        coverage_rule_ids = {
            str(rule_id or "").strip()
            for row in list(coverage_report.get("clause_coverages") or [])
            for rule_id in list(row.get("implemented_rule_ids") or [])
            if str(rule_id or "").strip()
        }

        self.assertEqual(sorted(coverage_rule_ids - metadata_rule_ids), [])

    def test_ectd_validation_standard_builds_capability_crosswalk_to_technical_spec(self) -> None:
        crosswalk = self.ectd_validation_standard_payload["capability_crosswalk"]

        self.assertEqual(crosswalk["schema_version"], REGULATION_LIBRARY_VERSION)
        self.assertEqual(
            crosswalk["capability_crosswalk_schema_version"],
            REGULATION_CAPABILITY_CROSSWALK_VERSION,
        )
        self.assertEqual(crosswalk["source_regulation_id"], "cn_ectd_validation_standard")
        self.assertEqual(crosswalk["target_regulation_id"], "cn_ectd_technical_specification")
        self.assertEqual(crosswalk["relation_basis"], "shared_runtime_rule_ids")
        self.assertGreater(crosswalk["relation_count"], 0)

        relations_by_source_target = {
            (item["source_clause_id"], item["target_clause_id"]): item
            for item in crosswalk["relations"]
        }
        regional_md5_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_4_1_13",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-050", regional_md5_relation["shared_rule_ids"])
        regional_leaf_descendant_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_4_1_20",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-061", regional_leaf_descendant_relation["shared_rule_ids"])
        index_leaf_descendant_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_21",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-062", index_leaf_descendant_relation["shared_rule_ids"])
        index_replace_append_checksum_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_6",
                "cn_ectd_technical_specification:sec_3_3_4",
            )
        ]
        self.assertIn("HR-ECTD-063", index_replace_append_checksum_relation["shared_rule_ids"])
        regional_replace_append_checksum_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_4_1_7",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-064", regional_replace_append_checksum_relation["shared_rule_ids"])
        index_single_operation_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_5",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-065", index_single_operation_relation["shared_rule_ids"])
        index_href_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_7",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-066", index_href_relation["shared_rule_ids"])
        index_delete_href_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_8",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-067", index_delete_href_relation["shared_rule_ids"])
        index_modified_file_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_9",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-068", index_modified_file_relation["shared_rule_ids"])
        index_initial_sequence_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_10",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-071", index_initial_sequence_relation["shared_rule_ids"])
        index_modified_file_target_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_11",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-072", index_modified_file_target_relation["shared_rule_ids"])
        index_path_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_12",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-073", index_path_relation["shared_rule_ids"])
        node_extension_usage_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_16",
                "cn_ectd_technical_specification:sec_3_7",
            )
        ]
        self.assertIn("HR-ECTD-074", node_extension_usage_relation["shared_rule_ids"])
        node_extension_title_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_17",
                "cn_ectd_technical_specification:sec_3_2",
            )
        ]
        self.assertIn("SR-ECTD-026", node_extension_title_relation["shared_rule_ids"])
        index_dtd_reference_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_2",
                "cn_ectd_technical_specification:sec_1_3",
            )
        ]
        self.assertIn("HR-ECTD-078", index_dtd_reference_relation["shared_rule_ids"])
        index_dtd_validity_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_3",
                "cn_ectd_technical_specification:sec_1_3",
            )
        ]
        self.assertIn("HR-ECTD-079", index_dtd_validity_relation["shared_rule_ids"])
        indication_attribute_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_25",
                "cn_ectd_technical_specification:sec_3_6",
            )
        ]
        self.assertIn("HR-ECTD-075", indication_attribute_relation["shared_rule_ids"])
        manufacturer_attribute_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_26",
                "cn_ectd_technical_specification:sec_3_6",
            )
        ]
        self.assertIn("HR-ECTD-076", manufacturer_attribute_relation["shared_rule_ids"])
        substance_attribute_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_27",
                "cn_ectd_technical_specification:sec_3_6",
            )
        ]
        self.assertIn("HR-ECTD-077", substance_attribute_relation["shared_rule_ids"])
        attribute_whitespace_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_28",
                "cn_ectd_technical_specification:sec_3_6",
            )
        ]
        self.assertIn("SR-ECTD-027", attribute_whitespace_relation["shared_rule_ids"])
        index_leaf_title_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_18",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-069", index_leaf_title_relation["shared_rule_ids"])
        index_delete_title_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_19",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-070", index_delete_title_relation["shared_rule_ids"])
        index_title_whitespace_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_20",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("SR-ECTD-025", index_title_whitespace_relation["shared_rule_ids"])
        replace_language_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_3_36",
                "cn_ectd_technical_specification:sec_3_5_2",
            )
        ]
        self.assertIn("HR-ECTD-026", replace_language_relation["shared_rule_ids"])
        checksum_type_relation = relations_by_source_target[
            (
                "cn_ectd_validation_standard:sec_4_1_12",
                "cn_ectd_technical_specification:sec_4_4",
            )
        ]
        self.assertIn("HR-ECTD-003", checksum_type_relation["shared_rule_ids"])
        for source_clause_id, target_clause_id, rule_id in (
            (
                "cn_ectd_validation_standard:sec_4_2_1",
                "cn_ectd_technical_specification:sec_2_1_1",
                "HR-ECTD-002",
            ),
            (
                "cn_ectd_validation_standard:sec_4_2_2",
                "cn_ectd_technical_specification:sec_2_1_2",
                "HR-ECTD-037",
            ),
            (
                "cn_ectd_validation_standard:sec_4_2_3",
                "cn_ectd_technical_specification:sec_2_1_3",
                "HR-ECTD-038",
            ),
            (
                "cn_ectd_validation_standard:sec_4_2_6",
                "cn_ectd_technical_specification:sec_2_2_1",
                "HR-ECTD-039",
            ),
            (
                "cn_ectd_validation_standard:sec_4_2_7",
                "cn_ectd_technical_specification:sec_2_3_1",
                "HR-ECTD-001",
            ),
            (
                "cn_ectd_validation_standard:sec_4_2_8",
                "cn_ectd_technical_specification:sec_2_3_2",
                "HR-ECTD-040",
            ),
            (
                "cn_ectd_validation_standard:sec_4_2_10",
                "cn_ectd_technical_specification:sec_2_4",
                "HR-ECTD-018",
            ),
        ):
            relation = relations_by_source_target[(source_clause_id, target_clause_id)]
            self.assertIn(rule_id, relation["shared_rule_ids"])
        for source_clause_id, target_clause_id, rule_id in (
            (
                "cn_ectd_validation_standard:sec_4_1_30",
                "cn_ectd_technical_specification:sec_3_9",
                "SR-ECTD-009",
            ),
            (
                "cn_ectd_validation_standard:sec_4_1_31",
                "cn_ectd_technical_specification:sec_3_5_2",
                "HR-ECTD-026",
            ),
        ):
            relation = relations_by_source_target[(source_clause_id, target_clause_id)]
            self.assertIn(rule_id, relation["shared_rule_ids"])
        for source_clause_id, rule_id in (
            ("cn_ectd_validation_standard:sec_4_2_13", "HR-ECTD-115"),
            ("cn_ectd_validation_standard:sec_4_2_14", "HR-ECTD-116"),
        ):
            relation = relations_by_source_target[
                (source_clause_id, "cn_ectd_technical_specification:sec_4_3")
            ]
            self.assertIn(rule_id, relation["shared_rule_ids"])

        unlinked_sources = {
            item["source_clause_id"]: item
            for item in crosswalk["source_clauses_without_technical_spec_links"]
        }
        self.assertIn("cn_ectd_validation_standard:sec_3_14", unlinked_sources)
        self.assertIn("HR-ECTD-049", unlinked_sources["cn_ectd_validation_standard:sec_3_14"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_2_9", unlinked_sources)
        self.assertIn("HR-ECTD-099", unlinked_sources["cn_ectd_validation_standard:sec_4_2_9"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_2_4", unlinked_sources)
        self.assertIn("HR-ECTD-103", unlinked_sources["cn_ectd_validation_standard:sec_4_2_4"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_2_5", unlinked_sources)
        self.assertIn("HR-ECTD-102", unlinked_sources["cn_ectd_validation_standard:sec_4_2_5"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_2_11", unlinked_sources)
        self.assertIn("HR-ECTD-100", unlinked_sources["cn_ectd_validation_standard:sec_4_2_11"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_2_12", unlinked_sources)
        self.assertIn("HR-ECTD-101", unlinked_sources["cn_ectd_validation_standard:sec_4_2_12"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_3_9", unlinked_sources)
        self.assertIn("SR-ECTD-055", unlinked_sources["cn_ectd_validation_standard:sec_4_3_9"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_3_7", unlinked_sources)
        self.assertIn("HR-ECTD-104", unlinked_sources["cn_ectd_validation_standard:sec_4_3_7"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_3_8", unlinked_sources)
        self.assertIn("HR-ECTD-105", unlinked_sources["cn_ectd_validation_standard:sec_4_3_8"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_3_1", unlinked_sources)
        self.assertIn("HR-ECTD-106", unlinked_sources["cn_ectd_validation_standard:sec_4_3_1"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_3_2", unlinked_sources)
        self.assertIn("HR-ECTD-107", unlinked_sources["cn_ectd_validation_standard:sec_4_3_2"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_3_3", unlinked_sources)
        self.assertIn("HR-ECTD-108", unlinked_sources["cn_ectd_validation_standard:sec_4_3_3"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_3_4", unlinked_sources)
        self.assertIn("HR-ECTD-109", unlinked_sources["cn_ectd_validation_standard:sec_4_3_4"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_3_5", unlinked_sources)
        self.assertIn("HR-ECTD-110", unlinked_sources["cn_ectd_validation_standard:sec_4_3_5"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_3_6", unlinked_sources)
        self.assertIn("HR-ECTD-111", unlinked_sources["cn_ectd_validation_standard:sec_4_3_6"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_1_2", unlinked_sources)
        self.assertIn("HR-ECTD-112", unlinked_sources["cn_ectd_validation_standard:sec_4_1_2"]["implemented_rule_ids"])
        self.assertIn("cn_ectd_validation_standard:sec_4_1_3", unlinked_sources)
        self.assertIn("HR-ECTD-113", unlinked_sources["cn_ectd_validation_standard:sec_4_1_3"]["implemented_rule_ids"])

    def test_ectd_validation_standard_pdf_parses_as_pdf_and_keeps_filename(self) -> None:
        self.assertEqual(
            self.ectd_validation_standard_parsed.get("filename"),
            self.ectd_validation_standard_source_path.name,
        )
        self.assertEqual(
            self.ectd_validation_standard_parsed.get("source_path"),
            str(self.ectd_validation_standard_source_path),
        )
        self.assertEqual(self.ectd_validation_standard_parsed.get("source_type"), "pdf")
        self.assertIn("eCTD验证标准", str(self.ectd_validation_standard_parsed.get("text") or ""))

    def test_ectd_validation_standard_pdf_extracts_validation_catalog_structure(self) -> None:
        regulation = self.ectd_validation_standard_payload["regulation"]
        chapters = self.ectd_validation_standard_payload["chapters"]
        clauses = self.ectd_validation_standard_payload["clauses"]

        self.assertEqual(regulation["regulation_id"], "cn_ectd_validation_standard")
        self.assertEqual(regulation["title"], "eCTD验证标准")
        self.assertEqual(regulation["chapter_count"], 6)
        self.assertGreaterEqual(regulation["article_count"], 100)
        self.assertEqual(regulation["article_count"], len(clauses))
        self.assertEqual(chapters[0]["chapter_no"], 1)
        self.assertEqual(chapters[0]["chapter_title"], "基础识别")
        self.assertEqual(chapters[-1]["chapter_no"], 6)
        self.assertEqual(chapters[-1]["chapter_title"], "PDF分析")
        self.assertEqual(clauses[0]["heading"], "1.1 文件数量统计")
        self.assertTrue(any(clause["heading"] == "4.1.1 模块一的区域骨架文件必须存在" for clause in clauses))
        self.assertTrue(any(clause["heading"] == "5.8 标签属性和类别元素的值" for clause in clauses))
        self.assertTrue(any(clause["heading"] == "6.25 PDF内容可搜索" for clause in clauses))
        self.assertGreaterEqual(len(self.ectd_validation_standard_payload["rule_candidates"]), len(clauses))

    def test_ectd_validation_standard_clauses_preserve_structured_regulation_severity(self) -> None:
        info_clause = self._get_ectd_validation_clause("1.1")
        error_clause = self._get_ectd_validation_clause("2.1")
        warning_clause = self._get_ectd_validation_clause("2.2")

        self.assertEqual(info_clause["severity"], "提示信息")
        self.assertEqual(error_clause["severity"], "错误")
        self.assertEqual(warning_clause["severity"], "警告")

    def test_write_regulation_corpus_entry_emits_artifacts_for_ectd_validation_standard(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            draft_rules_root = output_root / "rules"
            outputs = write_regulation_corpus_entry(
                self.ectd_validation_standard_source_path,
                output_root=output_root,
                draft_rules_root=draft_rules_root,
            )

            self.assertIn("document", outputs)
            self.assertIn("clauses", outputs)
            self.assertIn("rule_candidates", outputs)
            self.assertIn("coverage_report", outputs)
            self.assertIn("capability_crosswalk", outputs)
            document_payload = json.loads(outputs["document"].read_text(encoding="utf-8"))
            clauses_payload = json.loads(outputs["clauses"].read_text(encoding="utf-8"))
            crosswalk_payload = json.loads(outputs["capability_crosswalk"].read_text(encoding="utf-8"))
            self.assertEqual(document_payload["regulation"]["regulation_id"], "cn_ectd_validation_standard")
            self.assertGreaterEqual(document_payload["regulation"]["article_count"], 100)
            self.assertGreaterEqual(clauses_payload["clause_count"], 100)
            self.assertEqual(crosswalk_payload["target_regulation_id"], "cn_ectd_technical_specification")

    def test_ectd_validation_standard_handles_wrapped_headings_without_truncation(self) -> None:
        expected_pairs = {
            "3.7": "3.7 \u53f6\u5143\u7d20\uff1a\u65b0\u5efa\u3001\u66ff\u6362\u6216\u589e\u8865\u7684\u53f6\u5143\u7d20\uff0c\u5fc5\u987b\u6709\u201c\u6587\u4ef6\u5f15\u7528\uff08xlink:href\uff09\u201d\u503c",
            "3.8": "3.8 \u53f6\u5143\u7d20\uff1a\u5220\u9664\u7684\u53f6\u5143\u7d20\u4e0d\u80fd\u5305\u542b\u201c\u6587\u4ef6\u5f15\u7528\uff08xlink:href\uff09\u201d\u503c",
            "3.9": "3.9 \u53f6\u5143\u7d20\uff1a\u5bf9\u66ff\u6362\u3001\u5220\u9664\u548c\u589e\u8865\u7684\u53f6\u5143\u7d20\uff0c\u5fc5\u987b\u6709\u5bf9\u5e94\u7684\u6587\u4ef6\u201c\u64cd\u4f5c\uff08operation\uff09\u201d\u5c5e\u6027\u503c\u4e3a\u201c\u66ff\u6362\uff08replace\uff09\u201d\u3001\u201c\u5220\u9664\uff08delete\uff09\u201d\u6216\u201c\u589e\u8865\uff08append\uff09\u201d\u7684\u6240\u6709\u53f6\u5143\u7d20\uff0c\u5bf9\u5e94\u7684\u201c\u88ab\u4fee\u6539\u6587\u4ef6\u5bf9\u8c61\uff08modified-file\uff09\u201d\u5fc5\u987b\u6709\u503c\u3002",
            "4.1.8": "4.1.8 \u53f6\u5143\u7d20\uff1a\u65b0\u5efa\u3001\u66ff\u6362\u6216\u589e\u8865\u7684\u53f6\u5143\u7d20\uff0c\u5fc5\u987b\u6709\u201c\u6587\u4ef6\u5f15\u7528\uff08xlink:href\uff09\u201d\u503c",
            "4.1.9": "4.1.9 \u53f6\u5143\u7d20\uff1a\u5220\u9664\u7684\u53f6\u5143\u7d20\u4e0d\u80fd\u5305\u542b\u201c\u6587\u4ef6\u5f15\u7528\uff08xlink:href\uff09\u201d\u503c",
            "4.1.10": "4.1.10 \u53f6\u5143\u7d20\uff1a\u5bf9\u66ff\u6362\u3001\u5220\u9664\u548c\u589e\u8865\u7684\u53f6\u5143\u7d20\uff0c\u5fc5\u987b\u6709\u5bf9\u5e94\u7684\u6587\u4ef6\u201c\u64cd\u4f5c\uff08operation\uff09\u201d\u5c5e\u6027\u503c\u4e3a\u201c\u66ff\u6362\uff08replace\uff09\u201d\u3001\u201c\u5220\u9664\uff08delete\uff09\u201d\u6216\u201c\u589e\u8865\uff08append\uff09\u201d\u7684\u6240\u6709\u53f6\u5143\u7d20\uff0c\u5bf9\u5e94\u7684\u201c\u88ab\u4fee\u6539\u6587\u4ef6\u5bf9\u8c61\uff08modified-file\uff09\u201d\u5fc5\u987b\u6709\u503c\u3002",
        }

        for article_no_raw, expected_heading in expected_pairs.items():
            with self.subTest(article_no_raw=article_no_raw):
                clause = self._get_ectd_validation_clause(article_no_raw)
                self.assertEqual(clause["heading"], expected_heading)

    def test_ectd_validation_standard_splits_inline_heading_and_body_for_622(self) -> None:
        clause = self._get_ectd_validation_clause("6.22")
        self.assertEqual(
            clause["heading"],
            "6.22 PDF\u5e94\u8be5\u8bbe\u7f6e\u542f\u7528\u201c\u5feb\u901fWeb\u8bbf\u95ee\uff08Fast Web Access\uff09\u201d",
        )
        self.assertIn(
            "\u4e0d\u80fd\u63d0\u4ea4\u672a\u542f\u7528\u201c\u5feb\u901fWeb\u8bbf\u95ee\uff08Fast Web Access\uff09\u201d\u60c5\u51b5\u4e0b\u521b\u5efa\u7684PDF\u6587\u4ef6\u3002",
            clause["normalized_text"],
        )
        self.assertNotIn(
            "6.22 PDF\u5e94\u8be5\u8bbe\u7f6e\u542f\u7528\u201c\u5feb\u901fWeb\u8bbf\u95ee\uff08Fast Web Access\uff09\u201d \u4e0d\u80fd\u63d0\u4ea4",
            clause["heading"],
        )

    def test_ectd_validation_standard_stops_626_before_page_legend(self) -> None:
        clause = self._get_ectd_validation_clause("6.26")
        self.assertEqual(
            clause["heading"],
            "6.26 \u5982\u4f7f\u7528\u975e\u6807\u51c6\u5b57\u4f53\uff0c\u9700\u5d4c\u5165\u5728PDF\u6587\u4ef6\u4e2d",
        )
        self.assertIn("Zapf Dingbats", clause["normalized_text"])
        self.assertNotIn("\u5fc5\u987b\u9075\u5b88\u7684\u5173\u952e\u9a8c\u8bc1\u6807\u51c6", clause["normalized_text"])
        self.assertNotIn("\u7528\u4e8e\u6536\u96c6\u4fe1\u606f\u7684\u9a8c\u8bc1\u6807\u51c6", clause["normalized_text"])
        self.assertNotIn("\u8bf4\u660e:", clause["normalized_text"])

    def test_write_regulation_corpus_entry_emits_reference_manifest_for_ectd_technical_spec(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            draft_rules_root = output_root / "rules"
            outputs = write_regulation_corpus_entry(
                self.ectd_technical_spec_source_path,
                output_root=output_root,
                draft_rules_root=draft_rules_root,
            )

            self.assertIn("reference_manifest", outputs)
            reference_manifest_payload = json.loads(
                outputs["reference_manifest"].read_text(encoding="utf-8")
            )
            self.assertEqual(reference_manifest_payload["regulation_id"], "cn_ectd_technical_specification")
            self.assertEqual(reference_manifest_payload["reference_count"], 8)

    def test_write_regulation_corpus_entry_emits_coverage_report_for_ectd_technical_spec(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            draft_rules_root = output_root / "rules"
            outputs = write_regulation_corpus_entry(
                self.ectd_technical_spec_source_path,
                output_root=output_root,
                draft_rules_root=draft_rules_root,
            )

            self.assertIn("coverage_report", outputs)
            coverage_report_payload = json.loads(
                outputs["coverage_report"].read_text(encoding="utf-8")
            )
            self.assertEqual(coverage_report_payload["regulation_id"], "cn_ectd_technical_specification")
            self.assertEqual(coverage_report_payload["scope"], "chapter_1_to_4")

    def test_write_regulation_corpus_entry_emits_document_clause_and_candidate_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            draft_rules_root = output_root / "rules"
            outputs = write_regulation_corpus_entry(
                self.source_path,
                output_root=output_root,
                draft_rules_root=draft_rules_root,
            )

            self.assertEqual(
                sorted(outputs.keys()),
                ["clauses", "direct_rule_drafts", "document", "rule_candidates"],
            )
            for path in outputs.values():
                self.assertTrue(path.exists())

            document_payload = json.loads(outputs["document"].read_text(encoding="utf-8"))
            clauses_payload = json.loads(outputs["clauses"].read_text(encoding="utf-8"))
            candidates_payload = json.loads(outputs["rule_candidates"].read_text(encoding="utf-8"))
            direct_rules_payload = json.loads(outputs["direct_rule_drafts"].read_text(encoding="utf-8"))

            self.assertEqual(document_payload["regulation"]["article_count"], 89)
            self.assertEqual(clauses_payload["clause_count"], 89)
            self.assertEqual(candidates_payload["rule_candidate_count"], 89)
            self.assertEqual(direct_rules_payload["draft_rule_count"], 9)


if __name__ == "__main__":
    unittest.main()
