from __future__ import annotations

import importlib
import unittest

from core.material_assessment import build_compliance_result_payload
from tests.rule_tests.test_material_assessment import _build_parsed_document


class WorkbenchRuleProjectionTests(unittest.TestCase):
    def test_workbench_projects_scope_transition_guidance_for_missing_relative_paths(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        sequence_metadata_document = _build_parsed_document(
            filename="cn-regional.xml",
            source_path="D:\\submission\\cn-regional.xml",
            extra_metadata={
                "page_count": 1,
                "ectd_application_number": "x202112345",
                "ectd_sequence_number": "0000",
                "ectd_related_sequence_number": "0000",
                "ectd_previous_sequence_number": "0000",
                "ectd_envelope_attributes": {
                    "application-number": "x202112345",
                    "application-type": "clinical-trial-application",
                    "product-type": "chemical",
                    "regulatory-activity-type": "initial-application",
                    "sequence-type": "initial-submission",
                    "sequence-number": "0000",
                    "related-sequence": "0000",
                },
                "ectd_envelope_count": 1,
            },
            content_evidence_count=1,
            content_unit_count=1,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=0,
            toc_sequence_count=0,
        )
        compliance_result = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[sequence_metadata_document],
            consistency_rows=[],
            final_status="completed",
        )

        workbench = api_main._build_workbench(
            parsed_documents=[{**sequence_metadata_document, "file_id": "file_ectd_single"}],
            file_records=[
                {"file_id": "file_ectd_index", "filename": "index.xml"},
                {"file_id": "file_ectd_regional", "filename": "cn-regional.xml"},
            ],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result=compliance_result,
        )

        transition = workbench["rule_checks"]["summary"]["scope_transition_overview"]
        self.assertEqual(transition["transition_status"], "narrowed")
        self.assertEqual(transition["recommended_action_code"], "preserve_relative_paths")
        self.assertEqual(transition["guidance_priority"], "priority")
        self.assertGreaterEqual(len(transition["guidance_steps"]), 3)
        self.assertIn("relative paths", transition["guidance_targets"])

    def test_workbench_projects_submission_scope_overview_for_single_document_upload(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        sequence_metadata_document = _build_parsed_document(
            filename="cn-regional.xml",
            source_path="D:\\submission\\ectd\\x202112345\\0000\\m1\\cn\\cn-regional.xml",
            extra_metadata={
                "page_count": 1,
                "ectd_application_number": "x202112345",
                "ectd_sequence_number": "0000",
                "ectd_related_sequence_number": "0000",
                "ectd_previous_sequence_number": "0000",
                "ectd_sequence_directory_number": "0000",
                "ectd_envelope_attributes": {
                    "application-number": "x202112345",
                    "application-type": "clinical-trial-application",
                    "product-type": "chemical",
                    "regulatory-activity-type": "initial-application",
                    "sequence-type": "initial-submission",
                    "sequence-number": "0000",
                    "related-sequence": "0000",
                    "sequence-description": "首次提交",
                },
                "ectd_envelope_count": 1,
            },
            content_evidence_count=1,
            content_unit_count=1,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=0,
            toc_sequence_count=0,
        )
        compliance_result = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[sequence_metadata_document],
            consistency_rows=[],
            final_status="completed",
        )

        workbench = api_main._build_workbench(
            parsed_documents=[{**sequence_metadata_document, "file_id": "file_ectd_single"}],
            file_records=[{"file_id": "file_ectd_single", "filename": "cn-regional.xml"}],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result=compliance_result,
        )

        overview = workbench["rule_checks"]["summary"]["submission_scope_overview"]
        self.assertEqual(overview["upload_mode"], "single_document")
        self.assertEqual(overview["upload_mode_label"], "单文件上传")
        self.assertEqual(overview["available_scopes"], ["document"])
        self.assertEqual(overview["blocked_scopes"], ["sequence", "activity", "application"])
        self.assertIn("仅执行 document 级规则", overview["scope_notice"])
        self.assertEqual(overview["sequence_package_count"], 1)
        self.assertEqual(overview["regulatory_activity_count"], 1)
        self.assertEqual(overview["application_project_count"], 1)

    def test_workbench_projects_ectd_rule_group_summary_and_regulation_basis(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        ectd_document = _build_parsed_document(
            filename="index.xml",
            source_path="D:\\submission\\0000\\index.xml",
            content_evidence_count=0,
            content_unit_count=0,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=0,
            toc_sequence_count=0,
        )
        compliance_result = {
            "rules": [
                {
                    "rule_id": "HR-ECTD-001",
                    "category": "hard",
                    "status": "fail",
                    "message": "Sequence number must start at 0000 and progress by one.",
                    "citation": "cn_ectd_technical_specification#sec_2_3_1",
                    "details": {
                        "requirement_id": "cn_ectd_technical_specification:req_sequence_number_progression",
                        "citation_anchor": "cn_ectd_technical_specification#sec_2_3_1",
                    },
                },
                {
                    "rule_id": "SR-ECTD-002",
                    "category": "soft",
                    "status": "warn",
                    "message": "Sequence description is longer than 120 Chinese characters.",
                    "citation": "cn_ectd_technical_specification#sec_2_3_3",
                    "details": {
                        "requirement_id": "cn_ectd_technical_specification:req_sequence_description_length",
                        "citation_anchor": "cn_ectd_technical_specification#sec_2_3_3",
                    },
                },
                {
                    "rule_id": "HR-ECTD-011",
                    "category": "hard",
                    "status": "pass",
                    "message": "Envelope controlled vocabulary values are valid.",
                    "citation": "cn_ectd_technical_specification#sec_4_3",
                    "details": {
                        "requirement_id": "cn_ectd_technical_specification:req_envelope_controlled_vocabulary_validity",
                        "citation_anchor": "cn_ectd_technical_specification#sec_4_3",
                    },
                },
                {
                    "rule_id": "HR-PARSE-001",
                    "category": "hard",
                    "status": "pass",
                    "message": "Parsed material exposes review-ready evidence units.",
                    "citation": "material-review-contract-v1#documents.summary",
                },
            ],
            "risks": [],
            "summary": {
                "hard_failures": 1,
                "soft_risks": 1,
                "pass_rules": 2,
                "warn_rules": 1,
                "na_rules": 0,
            },
        }

        workbench = api_main._build_workbench(
            parsed_documents=[
                {
                    **ectd_document,
                    "file_id": "file_ectd_index",
                }
            ],
            file_records=[{"file_id": "file_ectd_index", "filename": "index.xml"}],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result=compliance_result,
        )

        rules_by_id = {item["rule_id"]: item for item in workbench["rule_checks"]["items"]}
        self.assertEqual(rules_by_id["HR-ECTD-001"]["basis"]["basis_kind"], "regulation")
        self.assertIn("eCTD", rules_by_id["HR-ECTD-001"]["basis"]["basis_label"])
        self.assertIn("2.3.1", rules_by_id["HR-ECTD-001"]["basis"]["basis_label"])
        self.assertIn("4 位数字", rules_by_id["HR-ECTD-001"]["basis"]["basis_detail"])

        group_summaries = workbench["rule_checks"]["group_summaries"]
        self.assertEqual(len(group_summaries), 1)
        self.assertEqual(group_summaries[0]["group_id"], "ectd_package_integrity")
        self.assertEqual(group_summaries[0]["overall_status"], "fail")
        self.assertEqual(group_summaries[0]["fail_count"], 1)
        self.assertEqual(group_summaries[0]["warn_count"], 1)
        self.assertEqual(group_summaries[0]["pass_count"], 1)
        self.assertEqual(group_summaries[0]["na_count"], 0)
        self.assertEqual(group_summaries[0]["focus_rule_ids"], ["HR-ECTD-001", "SR-ECTD-002"])
        self.assertEqual(
            group_summaries[0]["basis_refs"],
            [
                {
                    "citation": "cn_ectd_technical_specification#sec_2_3_1",
                    "basis_kind": "regulation",
                    "basis_label": rules_by_id["HR-ECTD-001"]["basis"]["basis_label"],
                },
                {
                    "citation": "cn_ectd_technical_specification#sec_2_3_3",
                    "basis_kind": "regulation",
                    "basis_label": rules_by_id["SR-ECTD-002"]["basis"]["basis_label"],
                },
                {
                    "citation": "cn_ectd_technical_specification#sec_4_3",
                    "basis_kind": "regulation",
                    "basis_label": rules_by_id["HR-ECTD-011"]["basis"]["basis_label"],
                },
            ],
        )

    def test_workbench_projects_regulation_severity_separately_from_rule_status(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        ectd_document = _build_parsed_document(
            filename="sample.pdf",
            source_path="D:\\submission\\sample.pdf",
            content_evidence_count=1,
            content_unit_count=1,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=0,
            toc_sequence_count=0,
        )
        compliance_result = {
            "rules": [
                {
                    "rule_id": "SR-ECTD-018",
                    "category": "soft",
                    "status": "warn",
                    "message": "PDF version is outside the allowed validation-standard set.",
                    "citation": "cn_ectd_validation_standard#sec_6_16",
                    "details": {
                        "requirement_id": "cn_ectd_validation_standard:req_pdf_version_allowed",
                        "citation_anchor": "cn_ectd_validation_standard#sec_6_16",
                    },
                },
                {
                    "rule_id": "HR-ECTD-001",
                    "category": "hard",
                    "status": "fail",
                    "message": "Sequence number must start at 0000 and progress by one.",
                    "citation": "cn_ectd_technical_specification#sec_2_3_1",
                    "details": {
                        "requirement_id": "cn_ectd_technical_specification:req_sequence_number_progression",
                        "citation_anchor": "cn_ectd_technical_specification#sec_2_3_1",
                    },
                },
            ],
            "risks": [],
            "summary": {
                "hard_failures": 1,
                "soft_risks": 1,
                "pass_rules": 0,
                "warn_rules": 1,
                "na_rules": 0,
            },
        }

        workbench = api_main._build_workbench(
            parsed_documents=[
                {
                    **ectd_document,
                    "file_id": "file_sample",
                }
            ],
            file_records=[{"file_id": "file_sample", "filename": "sample.pdf"}],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result=compliance_result,
        )

        rules_by_id = {item["rule_id"]: item for item in workbench["rule_checks"]["items"]}
        validation_rule = rules_by_id["SR-ECTD-018"]
        technical_rule = rules_by_id["HR-ECTD-001"]

        self.assertEqual(validation_rule["status"], "warn")
        self.assertEqual(validation_rule["details"]["regulation_severity"], "警告")
        self.assertEqual(
            validation_rule["details"]["citation_anchor"],
            "cn_ectd_validation_standard#sec_6_16",
        )
        self.assertIn("PDF", validation_rule["basis"]["basis_detail"])

        self.assertEqual(technical_rule["status"], "fail")
        self.assertIsNone(technical_rule["details"].get("regulation_severity"))
        self.assertEqual(
            technical_rule["details"]["citation_anchor"],
            "cn_ectd_technical_specification#sec_2_3_1",
        )

    def test_workbench_projects_structured_rule_details_into_rule_checks_items(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        module3_document = _build_parsed_document(
            filename="32-body-data.pdf",
            source_path="D:\\submission\\m3\\32-body-data.pdf",
            content_evidence_count=4,
            content_unit_count=1,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=0,
            toc_sequence_count=0,
        )
        compliance_result = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[module3_document],
            consistency_rows=[],
            final_status="completed",
        )

        workbench = api_main._build_workbench(
            parsed_documents=[
                {
                    **module3_document,
                    "file_id": "file_module3",
                }
            ],
            file_records=[{"file_id": "file_module3", "filename": "32-body-data.pdf"}],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result=compliance_result,
        )

        rule_checks = workbench["rule_checks"]
        self.assertTrue(rule_checks["enabled"])
        self.assertGreaterEqual(rule_checks["risk_count"], 1)
        self.assertGreaterEqual(len(rule_checks["navigation_audit_records"]), 1)
        rules_by_id = {item["rule_id"]: item for item in rule_checks["items"]}
        self.assertIn("SR-CTD-002", rules_by_id)

        ctd_rule = rules_by_id["SR-CTD-002"]
        self.assertEqual(ctd_rule["status"], "pass")
        self.assertIn("details", ctd_rule)
        self.assertEqual(ctd_rule["basis"]["basis_kind"], "regulation")
        self.assertIn("药品注册分类及申报资料要求", ctd_rule["basis"]["basis_label"])
        self.assertIn("CTD", ctd_rule["basis"]["basis_detail"])
        self.assertEqual(
            ctd_rule["details"]["requirement_id"],
            "cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission",
        )
        self.assertEqual(ctd_rule["details"]["match_strength"], "explicit_structure")
        self.assertEqual(ctd_rule["details"]["matched_documents"][0]["filename"], "32-body-data.pdf")
        self.assertEqual(ctd_rule["details"]["matched_documents"][0]["jump_page"], 1)
        self.assertEqual(ctd_rule["details"]["matched_documents"][0]["navigation_status"], "resolved_page")
        self.assertEqual(
            ctd_rule["details"]["matched_documents"][0]["navigation_reason"],
            "resolved_from_text_evidence",
        )
        self.assertEqual(ctd_rule["details"]["matched_documents"][0]["section_refs"][0]["jump_page"], 1)
        audit_record = next(
            item
            for item in rule_checks["navigation_audit_records"]
            if item["rule_id"] == "SR-CTD-002" and item["target_kind"] == "matched_document"
        )
        self.assertEqual(
            audit_record["requirement_id"],
            "cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission",
        )
        self.assertEqual(
            audit_record["citation_anchor"],
            "cn_drug_registration_classification_and_dossier_requirements#art_015",
        )
        self.assertEqual(audit_record["target_page"], 1)
        self.assertEqual(audit_record["backend_navigation_status"], "resolved_page")

        parse_rule = rules_by_id["HR-PARSE-001"]
        self.assertEqual(parse_rule["basis"]["basis_kind"], "system")
        self.assertIn("材料审阅契约 v1 / 文档摘要", parse_rule["basis"]["basis_label"])
        self.assertIn("内容证据", parse_rule["basis"]["basis_detail"])

    def test_workbench_projects_navigation_targets_for_weak_signal_snippets(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        weak_ctd_document = _build_parsed_document(
            filename="submission-summary.pdf",
            source_path="D:\\submission\\submission-summary.pdf",
            content_evidence_count=1,
            content_unit_count=1,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=0,
            toc_sequence_count=0,
            atomic_facts={},
            content_units_override=[
                {
                    "unit_id": "cu_text_weak_ctd_001",
                    "evidence_id": "ce_text_weak_ctd_001",
                    "source_type": "text",
                    "source_id": "txt_weak_ctd_001",
                    "page": 1,
                    "bbox": [10.0, 10.0, 250.0, 40.0],
                    "semantic_role": "text_block",
                    "unit_role": "body",
                    "unit_index": 1,
                    "text": "This submission package follows CTD format numbering and order for review.",
                    "attributes": {},
                    "fact_extraction_eligible": True,
                }
            ],
            content_evidence=[
                {
                    "evidence_id": "ce_text_weak_ctd_001",
                    "source_type": "text",
                    "source_id": "txt_weak_ctd_001",
                    "page": 1,
                    "bbox": [10.0, 10.0, 250.0, 40.0],
                    "semantic_role": "text_block",
                    "content_text": "This submission package follows CTD format numbering and order for review.",
                    "segments": [
                        {
                            "role": "body",
                            "text": "This submission package follows CTD format numbering and order for review.",
                        }
                    ],
                }
            ],
        )
        compliance_result = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[weak_ctd_document],
            consistency_rows=[],
            final_status="completed_with_warnings",
        )

        workbench = api_main._build_workbench(
            parsed_documents=[
                {
                    **weak_ctd_document,
                    "file_id": "file_weak_ctd",
                }
            ],
            file_records=[{"file_id": "file_weak_ctd", "filename": "submission-summary.pdf"}],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result=compliance_result,
        )

        rules_by_id = {item["rule_id"]: item for item in workbench["rule_checks"]["items"]}
        ctd_rule = rules_by_id["SR-CTD-002"]
        self.assertEqual(ctd_rule["status"], "warn")
        self.assertEqual(ctd_rule["details"]["weak_signal_snippets"][0]["jump_page"], 1)
        self.assertEqual(ctd_rule["details"]["weak_signal_snippets"][0]["navigation_status"], "resolved_page")
        self.assertEqual(
            ctd_rule["details"]["weak_signal_snippets"][0]["navigation_reason"],
            "resolved_from_text_evidence",
        )
        snippet_audit_record = next(
            item
            for item in workbench["rule_checks"]["navigation_audit_records"]
            if item["rule_id"] == "SR-CTD-002" and item["target_kind"] == "weak_signal_snippet"
        )
        self.assertEqual(snippet_audit_record["target_page"], 1)
        self.assertEqual(snippet_audit_record["backend_navigation_status"], "resolved_page")

    def test_workbench_projects_basis_and_jump_targets_for_pdf_hyperlink_rules(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        hyperlink_document = _build_parsed_document(
            filename="m2-hyperlink.pdf",
            source_path="D:\\submission\\m2-hyperlink.pdf",
            extra_metadata={
                "page_count": 4,
                "pdf_link_action_kinds": ["/GoTo", "/Named"],
                "pdf_link_action_pages": [2, 4],
            },
            content_evidence_count=1,
            content_unit_count=1,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=0,
            toc_sequence_count=0,
        )
        compliance_result = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[hyperlink_document],
            consistency_rows=[],
            final_status="completed_with_warnings",
        )

        workbench = api_main._build_workbench(
            parsed_documents=[
                {
                    **hyperlink_document,
                    "file_id": "file_hyperlink",
                }
            ],
            file_records=[{"file_id": "file_hyperlink", "filename": "m2-hyperlink.pdf"}],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result=compliance_result,
        )

        rules_by_id = {item["rule_id"]: item for item in workbench["rule_checks"]["items"]}
        hyperlink_rule = rules_by_id["HR-ECTD-045"]
        self.assertEqual(hyperlink_rule["basis"]["basis_kind"], "regulation")
        self.assertIn("eCTD", hyperlink_rule["basis"]["basis_label"])
        self.assertEqual(hyperlink_rule["details"]["matched_documents"][0]["jump_page"], 2)
        audit_record = next(
            item
            for item in workbench["rule_checks"]["navigation_audit_records"]
            if item["rule_id"] == "HR-ECTD-045" and item["target_kind"] == "matched_document"
        )
        self.assertEqual(audit_record["target_page"], 2)
        self.assertEqual(audit_record["backend_navigation_status"], "resolved_page")

    def test_workbench_projects_uri_hyperlink_rule_to_detected_uri_page(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        uri_document = _build_parsed_document(
            filename="m2-uri-link.pdf",
            source_path="D:\\submission\\m2-uri-link.pdf",
            extra_metadata={
                "page_count": 4,
                "external_uri_link_count": 1,
                "external_file_link_count": 0,
                "pdf_link_action_pages": [1, 3],
                "pdf_link_action_page_records": [
                    {
                        "page": 1,
                        "link_annotation_count": 1,
                        "link_annotation_xrefs": [11],
                        "link_action_kinds": ["/GoTo"],
                    },
                    {
                        "page": 3,
                        "link_annotation_count": 1,
                        "link_annotation_xrefs": [31],
                        "link_action_kinds": ["/URI"],
                    },
                ],
            },
            content_evidence_count=1,
            content_unit_count=1,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=0,
            toc_sequence_count=0,
        )
        compliance_result = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[uri_document],
            consistency_rows=[],
            final_status="completed_with_warnings",
        )

        workbench = api_main._build_workbench(
            parsed_documents=[{**uri_document, "file_id": "file_uri_link"}],
            file_records=[{"file_id": "file_uri_link", "filename": "m2-uri-link.pdf"}],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result=compliance_result,
        )

        rules_by_id = {item["rule_id"]: item for item in workbench["rule_checks"]["items"]}
        uri_rule = rules_by_id["SR-ECTD-022"]
        self.assertEqual(uri_rule["status"], "warn")
        self.assertEqual(uri_rule["details"]["matched_documents"][0]["jump_page"], 3)
        self.assertEqual(uri_rule["details"]["matched_documents"][0]["navigation_status"], "resolved_page")
        self.assertEqual(uri_rule["details"]["matched_documents"][0]["navigation_reason"], "resolved_from_rule_evidence")
        audit_record = next(
            item
            for item in workbench["rule_checks"]["navigation_audit_records"]
            if item["rule_id"] == "SR-ECTD-022" and item["target_kind"] == "matched_document"
        )
        self.assertEqual(audit_record["target_page"], 3)
        self.assertEqual(audit_record["backend_navigation_status"], "resolved_page")

    def test_workbench_projects_structural_navigation_targets_when_rule_evidence_points_to_a_table(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        table_document = _build_parsed_document(
            filename="table-material.pdf",
            source_path="D:\\submission\\m3\\table-material.pdf",
            content_evidence_count=2,
            content_unit_count=1,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=0,
            toc_sequence_count=0,
            table_asts_override=[
                {
                    "table_id": "tbl_001",
                    "page": 1,
                    "bbox": [10.0, 40.0, 200.0, 100.0],
                    "review_required": False,
                    "review_reasons": [],
                    "diagnostics": {},
                }
            ],
        )
        compliance_result = {
            "rules": [
                {
                    "rule_id": "SR-NAV-TEST",
                    "category": "soft",
                    "status": "pass",
                    "message": "Synthetic structural navigation test.",
                    "details": {
                        "requirement_id": "req_structural_nav",
                        "matched_documents": [
                            {
                                "filename": "table-material.pdf",
                                "evidence_refs": ["ce_table_tbl_001"],
                                "section_refs": [],
                            }
                        ],
                        "weak_signal_snippets": [],
                    },
                }
            ],
            "risks": [],
            "summary": {},
        }

        workbench = api_main._build_workbench(
            parsed_documents=[
                {
                    **table_document,
                    "file_id": "file_table_nav",
                }
            ],
            file_records=[{"file_id": "file_table_nav", "filename": "table-material.pdf"}],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result=compliance_result,
        )

        rules_by_id = {item["rule_id"]: item for item in workbench["rule_checks"]["items"]}
        nav_rule = rules_by_id["SR-NAV-TEST"]
        self.assertEqual(nav_rule["details"]["matched_documents"][0]["jump_page"], 1)
        self.assertEqual(nav_rule["details"]["matched_documents"][0]["structural_id"], "tbl_001")
        self.assertEqual(nav_rule["details"]["matched_documents"][0]["navigation_status"], "resolved_structural")
        self.assertEqual(
            nav_rule["details"]["matched_documents"][0]["navigation_reason"],
            "resolved_from_structural_evidence",
        )

    def test_workbench_marks_navigation_as_unresolved_when_no_evidence_or_section_target_exists(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        plain_document = _build_parsed_document(
            filename="plain-material.pdf",
            source_path="D:\\submission\\plain-material.pdf",
            content_evidence_count=1,
            content_unit_count=1,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=0,
            toc_sequence_count=0,
        )
        compliance_result = {
            "rules": [
                {
                    "rule_id": "SR-NAV-UNRESOLVED",
                    "category": "soft",
                    "status": "warn",
                    "message": "Synthetic unresolved navigation test.",
                    "details": {
                        "requirement_id": "req_unresolved_nav",
                        "matched_documents": [
                            {
                                "filename": "plain-material.pdf",
                                "evidence_refs": [],
                                "section_refs": [],
                            }
                        ],
                        "weak_signal_snippets": [
                            {
                                "filename": "plain-material.pdf",
                                "evidence_id": "missing_evidence_id",
                            }
                        ],
                    },
                }
            ],
            "risks": [],
            "summary": {},
        }

        workbench = api_main._build_workbench(
            parsed_documents=[
                {
                    **plain_document,
                    "file_id": "file_plain_nav",
                }
            ],
            file_records=[{"file_id": "file_plain_nav", "filename": "plain-material.pdf"}],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result=compliance_result,
        )

        rules_by_id = {item["rule_id"]: item for item in workbench["rule_checks"]["items"]}
        nav_rule = rules_by_id["SR-NAV-UNRESOLVED"]
        self.assertEqual(nav_rule["details"]["matched_documents"][0]["navigation_status"], "unresolved")
        self.assertEqual(
            nav_rule["details"]["matched_documents"][0]["navigation_reason"],
            "no_navigation_target",
        )
        self.assertEqual(nav_rule["details"]["weak_signal_snippets"][0]["navigation_status"], "unresolved")
        self.assertEqual(
            nav_rule["details"]["weak_signal_snippets"][0]["navigation_reason"],
            "no_navigation_target",
        )
        unresolved_audit_record = next(
            item
            for item in workbench["rule_checks"]["navigation_audit_records"]
            if item["rule_id"] == "SR-NAV-UNRESOLVED" and item["target_kind"] == "matched_document"
        )
        self.assertEqual(unresolved_audit_record["backend_navigation_status"], "unresolved")
        self.assertEqual(unresolved_audit_record["backend_navigation_reason"], "no_navigation_target")

    def test_workbench_projects_toc_structure_audit_records_from_rule_details(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        partial_subtree_document = _build_parsed_document(
            filename="toc-subtree-depth3-partial.pdf",
            source_path="D:\\submission\\toc-subtree-depth3-partial.pdf",
            content_evidence_count=5,
            content_unit_count=5,
            review_required_table_count=0,
            review_required_toc_count=0,
            toc_count=1,
            toc_sequence_count=1,
            atomic_facts={"drug_name": "ExampleDrug"},
            navigation_entries=[
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
                },
                {
                    "sequence_entry_index": 2,
                    "toc_id": "toc_001",
                    "page": 2,
                    "outline_index": "1.1",
                    "text": "Scope",
                    "page_locator": "4",
                    "page_locator_kind": "arabic",
                    "page_locator_value": 4,
                    "level": 2,
                    "parent_sequence_entry_index": 1,
                    "parent_toc_id": "toc_001",
                    "section_anchor_sequence_entry_index": 2,
                    "review_required": False,
                    "audit_flags": [],
                },
                {
                    "sequence_entry_index": 3,
                    "toc_id": "toc_001",
                    "page": 2,
                    "outline_index": "1.1.1",
                    "text": "Dosage",
                    "page_locator": "5",
                    "page_locator_kind": "arabic",
                    "page_locator_value": 5,
                    "level": 3,
                    "parent_sequence_entry_index": 2,
                    "parent_toc_id": "toc_001",
                    "section_anchor_sequence_entry_index": 3,
                    "review_required": False,
                    "audit_flags": [],
                },
                {
                    "sequence_entry_index": 4,
                    "toc_id": "toc_001",
                    "page": 2,
                    "outline_index": "1.1.1.1",
                    "text": "Dose Form",
                    "page_locator": "6",
                    "page_locator_kind": "arabic",
                    "page_locator_value": 6,
                    "level": 4,
                    "parent_sequence_entry_index": 3,
                    "parent_toc_id": "toc_001",
                    "section_anchor_sequence_entry_index": 4,
                    "review_required": False,
                    "audit_flags": [],
                },
                {
                    "sequence_entry_index": 5,
                    "toc_id": "toc_001",
                    "page": 2,
                    "outline_index": "1.1.1.2",
                    "text": "Administration",
                    "page_locator": "7",
                    "page_locator_kind": "arabic",
                    "page_locator_value": 7,
                    "level": 4,
                    "parent_sequence_entry_index": 3,
                    "parent_toc_id": "toc_001",
                    "section_anchor_sequence_entry_index": 5,
                    "review_required": False,
                    "audit_flags": [],
                },
            ],
            content_units_override=[
                {
                    "unit_id": "cu_text_001",
                    "evidence_id": "ce_text_001",
                    "source_type": "text",
                    "source_id": "txt_001",
                    "page": 3,
                    "bbox": [10.0, 10.0, 90.0, 20.0],
                    "semantic_role": "text_block",
                    "unit_role": "section_heading",
                    "unit_index": 1,
                    "text": "Overview",
                    "attributes": {},
                    "section_context": {
                        "module_label": "DOC-1",
                        "outline_index": "1",
                        "section_title": "Overview",
                        "anchor_source": "heading",
                        "anchor_confidence": 0.93,
                    },
                    "fact_extraction_eligible": False,
                },
                {
                    "unit_id": "cu_text_002",
                    "evidence_id": "ce_text_002",
                    "source_type": "text",
                    "source_id": "txt_002",
                    "page": 4,
                    "bbox": [10.0, 10.0, 90.0, 20.0],
                    "semantic_role": "text_block",
                    "unit_role": "section_heading",
                    "unit_index": 2,
                    "text": "Scope",
                    "attributes": {},
                    "section_context": {
                        "module_label": "DOC-1",
                        "outline_index": "1.1",
                        "section_title": "Scope",
                        "anchor_source": "heading",
                        "anchor_confidence": 0.93,
                    },
                    "fact_extraction_eligible": False,
                },
                {
                    "unit_id": "cu_text_003",
                    "evidence_id": "ce_text_003",
                    "source_type": "text",
                    "source_id": "txt_003",
                    "page": 5,
                    "bbox": [10.0, 10.0, 90.0, 20.0],
                    "semantic_role": "text_block",
                    "unit_role": "section_heading",
                    "unit_index": 3,
                    "text": "Dosage",
                    "attributes": {},
                    "section_context": {
                        "module_label": "DOC-1",
                        "outline_index": "1.1.1",
                        "section_title": "Dosage",
                        "anchor_source": "heading",
                        "anchor_confidence": 0.93,
                    },
                    "fact_extraction_eligible": False,
                },
                {
                    "unit_id": "cu_text_004",
                    "evidence_id": "ce_text_004",
                    "source_type": "text",
                    "source_id": "txt_004",
                    "page": 6,
                    "bbox": [10.0, 10.0, 90.0, 20.0],
                    "semantic_role": "text_block",
                    "unit_role": "section_heading",
                    "unit_index": 4,
                    "text": "Dose Form",
                    "attributes": {},
                    "section_context": {
                        "module_label": "DOC-1",
                        "outline_index": "1.1.1.1",
                        "section_title": "Dose Form",
                        "anchor_source": "heading",
                        "anchor_confidence": 0.93,
                    },
                    "fact_extraction_eligible": False,
                },
                {
                    "unit_id": "cu_text_005",
                    "evidence_id": "ce_text_005",
                    "source_type": "text",
                    "source_id": "txt_005",
                    "page": 6,
                    "bbox": [10.0, 30.0, 120.0, 40.0],
                    "semantic_role": "text_block",
                    "unit_role": "body",
                    "unit_index": 5,
                    "text": "Drug Name: ExampleDrug",
                    "attributes": {},
                    "section_context": {
                        "module_label": "DOC-1",
                        "outline_index": "1.1.1.1",
                        "section_title": "Dose Form",
                        "anchor_source": "heading",
                        "anchor_confidence": 0.93,
                    },
                    "fact_extraction_eligible": True,
                },
            ],
            content_evidence=[
                {
                    "evidence_id": "ce_text_001",
                    "source_type": "text",
                    "source_id": "txt_001",
                    "page": 3,
                    "bbox": [10.0, 10.0, 90.0, 20.0],
                    "semantic_role": "text_block",
                    "content_text": "Overview",
                    "section_context": {
                        "module_label": "DOC-1",
                        "outline_index": "1",
                        "section_title": "Overview",
                        "anchor_source": "heading",
                        "anchor_confidence": 0.93,
                    },
                    "segments": [{"role": "heading", "text": "Overview"}],
                },
                {
                    "evidence_id": "ce_text_002",
                    "source_type": "text",
                    "source_id": "txt_002",
                    "page": 4,
                    "bbox": [10.0, 10.0, 90.0, 20.0],
                    "semantic_role": "text_block",
                    "content_text": "Scope",
                    "section_context": {
                        "module_label": "DOC-1",
                        "outline_index": "1.1",
                        "section_title": "Scope",
                        "anchor_source": "heading",
                        "anchor_confidence": 0.93,
                    },
                    "segments": [{"role": "heading", "text": "Scope"}],
                },
                {
                    "evidence_id": "ce_text_003",
                    "source_type": "text",
                    "source_id": "txt_003",
                    "page": 5,
                    "bbox": [10.0, 10.0, 90.0, 20.0],
                    "semantic_role": "text_block",
                    "content_text": "Dosage",
                    "section_context": {
                        "module_label": "DOC-1",
                        "outline_index": "1.1.1",
                        "section_title": "Dosage",
                        "anchor_source": "heading",
                        "anchor_confidence": 0.93,
                    },
                    "segments": [{"role": "heading", "text": "Dosage"}],
                },
                {
                    "evidence_id": "ce_text_004",
                    "source_type": "text",
                    "source_id": "txt_004",
                    "page": 6,
                    "bbox": [10.0, 10.0, 90.0, 20.0],
                    "semantic_role": "text_block",
                    "content_text": "Dose Form",
                    "section_context": {
                        "module_label": "DOC-1",
                        "outline_index": "1.1.1.1",
                        "section_title": "Dose Form",
                        "anchor_source": "heading",
                        "anchor_confidence": 0.93,
                    },
                    "segments": [{"role": "heading", "text": "Dose Form"}],
                },
                {
                    "evidence_id": "ce_text_005",
                    "source_type": "text",
                    "source_id": "txt_005",
                    "page": 6,
                    "bbox": [10.0, 30.0, 120.0, 40.0],
                    "semantic_role": "text_block",
                    "content_text": "Drug Name: ExampleDrug",
                    "section_context": {
                        "module_label": "DOC-1",
                        "outline_index": "1.1.1.1",
                        "section_title": "Dose Form",
                        "anchor_source": "heading",
                        "anchor_confidence": 0.93,
                    },
                    "segments": [{"role": "body", "text": "Drug Name: ExampleDrug"}],
                },
            ],
        )
        compliance_result = build_compliance_result_payload(
            submission_profile="FIH",
            parsed_documents=[partial_subtree_document],
            consistency_rows=[],
            final_status="completed",
        )

        workbench = api_main._build_workbench(
            parsed_documents=[
                {
                    **partial_subtree_document,
                    "file_id": "file_toc_subtree",
                }
            ],
            file_records=[{"file_id": "file_toc_subtree", "filename": "toc-subtree-depth3-partial.pdf"}],
            consistency_rows=[],
            markdown_download_url=None,
            structure_audit_download_url="/api/v1/jobs/job_toc_audit/structure-audit/download",
            compliance_result=compliance_result,
        )

        rules_by_id = {item["rule_id"]: item for item in workbench["rule_checks"]["items"]}
        toc_rule = rules_by_id["SR-TOC-001"]
        self.assertEqual(toc_rule["status"], "warn")
        self.assertIn("details", toc_rule)
        structure_audit_rows = toc_rule["details"]["structure_audit_rows"]
        self.assertEqual(len(structure_audit_rows), 1)
        self.assertEqual(structure_audit_rows[0]["bounded_subtree_coverage_ratio"], 0.75)
        structure_audit_records = workbench["rule_checks"]["structure_audit_records"]
        self.assertEqual(len(structure_audit_records), 1)
        self.assertEqual(workbench["structure_audit_download_url"], "/api/v1/jobs/job_toc_audit/structure-audit/download")
        self.assertEqual(structure_audit_records[0]["rule_id"], "SR-TOC-001")
        self.assertEqual(structure_audit_records[0]["filename"], "toc-subtree-depth3-partial.pdf")
        self.assertEqual(structure_audit_records[0]["bounded_subtree_coverage_ratio"], 0.75)
        self.assertEqual(structure_audit_records[0]["missing_body_bounded_subtree_outline_indices"], ["1.1.1.2"])
        alignment_path_rows = structure_audit_records[0]["toc_body_alignment_path_rows"]
        self.assertEqual(
            [row["normalized_outline_index"] for row in alignment_path_rows],
            ["1", "1.1", "1.1.1", "1.1.1.1", "1.1.1.2"],
        )
        missing_path_row = next(row for row in alignment_path_rows if row["normalized_outline_index"] == "1.1.1.2")
        self.assertEqual(missing_path_row["alignment_status"], "nearest_parent")
        self.assertEqual(missing_path_row["body_anchor_page"], 5)
        self.assertEqual(missing_path_row["body_anchor_outline_index"], "1.1.1")
        self.assertEqual(
            structure_audit_records[0]["missing_body_bounded_subtree_path_rows"],
            [
                {
                    "root_outline_index": "1.0",
                    "root_normalized_outline_index": "1",
                    "parent_outline_index": "1.1.1",
                    "parent_normalized_outline_index": "1.1.1",
                    "outline_index": "1.1.1.2",
                    "normalized_outline_index": "1.1.1.2",
                    "outline_path": "1.0 > 1.1 > 1.1.1 > 1.1.1.2",
                    "text_path": "Overview > Scope > Dosage > Administration",
                    "page_locator_value": 7,
                    "navigation_page": 2,
                    "level": 4,
                    "nearest_body_anchor_outline_index": "1.1.1",
                    "nearest_body_anchor_page": 5,
                }
            ],
        )
        self.assertEqual(
            structure_audit_records[0]["root_page_alignment_rows"],
            [
                {
                    "outline_index": "1.0",
                    "normalized_outline_index": "1",
                    "toc_navigation_page": 2,
                    "toc_page_locator_value": 3,
                    "body_page_start": 3,
                    "body_page_end": 6,
                    "body_anchor_page_start": 3,
                    "projected_page": 3,
                    "offset": 0,
                    "toc_order_index": 1,
                    "body_order_index": 1,
                    "order_conflict": False,
                    "offset_conflict": False,
                    "span_conflict": False,
                }
            ],
        )
        self.assertEqual(
            structure_audit_records[0]["navigation_targets"],
            [
                {
                    "target_kind": "toc",
                    "label": "目录起始页",
                    "page": 2,
                    "toc_sequence_id": "tocseq_001",
                    "outline_indices": ["1.1.1.2"],
                },
                {
                    "target_kind": "body_root",
                    "label": "正文根章节起始页",
                    "page": 3,
                    "toc_sequence_id": None,
                    "outline_indices": ["1.1.1.2"],
                },
            ],
        )
        self.assertEqual(workbench["structure_audit_download_url"], "/api/v1/jobs/job_toc_audit/structure-audit/download")

    def test_workbench_projects_regulatory_readiness_for_phase_a_demo(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result={"rules": [], "risks": [], "summary": {}},
        )

        readiness = workbench["regulatory_readiness"]
        self.assertEqual(readiness["phase"], "phase_a_demo_workbench")
        self.assertEqual(readiness["phase_status"], "in_progress")
        self.assertEqual(
            readiness["recommended_next_action_code"],
            "build_source_readiness_matrix_and_triage_projection",
        )
        self.assertEqual(
            readiness["rule_triage_categories"],
            ["deterministic", "prerequisite_required", "human_review", "out_of_scope_low_roi"],
        )

        sources_by_id = {item["regulation_id"]: item for item in readiness["sources"]}
        self.assertEqual(set(sources_by_id), {
            "cn_ectd_validation_standard",
            "cn_ectd_technical_specification",
            "reg_3454a11dabae",
            "cn_drug_administration_law_implementation_regulation",
            "cn_drug_registration_classification_and_dossier_requirements",
        })
        self.assertEqual(sources_by_id["cn_ectd_validation_standard"]["coverage_status"], "closed")
        self.assertEqual(sources_by_id["cn_ectd_validation_standard"]["clause_count"], 149)
        self.assertEqual(sources_by_id["cn_ectd_validation_standard"]["covered_count"], 146)
        self.assertEqual(
            sources_by_id["cn_ectd_validation_standard"]["recommended_product_role"],
            "deterministic_validation_backbone",
        )
        self.assertEqual(sources_by_id["cn_ectd_technical_specification"]["traceability_gap_count"], 0)
        self.assertEqual(sources_by_id["cn_ectd_technical_specification"]["partial_count"], 12)
        self.assertEqual(
            sources_by_id["cn_drug_registration_classification_and_dossier_requirements"]["recommended_product_role"],
            "dossier_checklist_and_applicability_backbone",
        )
        self.assertEqual(
            sources_by_id["cn_drug_registration_classification_and_dossier_requirements"]["requirement_count"],
            9,
        )
        self.assertEqual(
            sources_by_id["cn_drug_administration_law_implementation_regulation"]["default_triage"],
            "human_review",
        )
        self.assertIn(
            "Do not pursue exhaustive rule automation",
            sources_by_id["cn_drug_administration_law_implementation_regulation"]["automation_boundary"],
        )
        self.assertEqual(readiness["summary"]["source_count"], 5)
        self.assertEqual(readiness["summary"]["closed_source_count"], 2)
        self.assertIn(
            "dossier checklist",
            readiness["summary"]["demo_value_statement"],
        )

    def test_workbench_projects_dossier_checklist_with_prerequisite_boundaries(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result={"rules": [], "risks": [], "summary": {}},
        )

        checklist = workbench["dossier_checklist"]
        self.assertEqual(checklist["schema_version"], "dossier-checklist-v1")
        self.assertEqual(
            checklist["regulation_id"],
            "cn_drug_registration_classification_and_dossier_requirements",
        )
        self.assertEqual(checklist["phase"], "phase_a_demo_workbench")
        self.assertEqual(checklist["applicability_mode"], "prerequisite_required")
        self.assertEqual(checklist["requirement_count"], 9)
        self.assertEqual(checklist["summary"]["requirement_count"], 9)
        self.assertEqual(checklist["summary"]["deterministic_decision_count"], 0)
        self.assertEqual(checklist["summary"]["prerequisite_required_count"], 9)
        self.assertIn("registration_class", checklist["missing_prerequisite_fact_keys"])
        self.assertIn("application_type", checklist["missing_prerequisite_fact_keys"])
        self.assertIn("submission_stage", checklist["missing_prerequisite_fact_keys"])
        self.assertIn("package_scope", checklist["missing_prerequisite_fact_keys"])

        prerequisite_facts = {item["fact_key"]: item for item in checklist["prerequisite_facts"]}
        self.assertEqual(prerequisite_facts["registration_class"]["status"], "missing")
        self.assertEqual(prerequisite_facts["application_type"]["status"], "missing")
        self.assertEqual(prerequisite_facts["product_type"]["status"], "missing")

        items_by_id = {item["requirement_id"]: item for item in checklist["items"]}
        ctd_requirement = items_by_id[
            "cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission"
        ]
        self.assertEqual(ctd_requirement["applicability_status"], "prerequisite_required")
        self.assertEqual(ctd_requirement["default_triage"], "prerequisite_required")
        self.assertEqual(ctd_requirement["source_article_no"], 15)
        self.assertEqual(
            ctd_requirement["citation_anchor"],
            "cn_drug_registration_classification_and_dossier_requirements#art_015",
        )
        self.assertIn("general_registration_dossier", ctd_requirement["expected_material_evidence"])
        self.assertIn("api_supporting_materials", ctd_requirement["expected_material_evidence"])
        self.assertIn("application_type", ctd_requirement["blocking_prerequisite_fact_keys"])
        self.assertIn("package_scope", ctd_requirement["blocking_prerequisite_fact_keys"])
        self.assertIn("Do not hard judge", ctd_requirement["automation_boundary"])
        self.assertTrue(ctd_requirement["requirement_text_preview"])

        clinical_database_requirement = items_by_id[
            "cn_drug_registration_classification_and_dossier_requirements:req_electronic_clinical_trial_database"
        ]
        self.assertIn(
            "clinical_trial_completion_status",
            clinical_database_requirement["blocking_prerequisite_fact_keys"],
        )
        self.assertIn(
            "clinical_trial_materials",
            clinical_database_requirement["expected_material_evidence"],
        )

    def test_dossier_checklist_uses_reliable_sequence_package_prerequisite_facts(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        submission_scope = {
            "upload_mode": "ectd_sequence_package",
            "available_scopes": ["document", "sequence", "activity", "application"],
            "ectd_project_context": {
                "sequence_package_count": 1,
                "regulatory_activity_count": 1,
                "application_project_count": 1,
                "sequence_packages": [
                    {
                        "sequence_package_id": "seqpkg:x202112345:0000",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                        "sequence_number": "0000",
                    }
                ],
            },
        }

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result={
                "rules": [],
                "risks": [],
                "summary": {},
                "submission_scope": submission_scope,
            },
        )

        checklist = workbench["dossier_checklist"]
        prerequisite_facts = {item["fact_key"]: item for item in checklist["prerequisite_facts"]}
        self.assertEqual(prerequisite_facts["application_type"]["status"], "present")
        self.assertEqual(prerequisite_facts["application_type"]["value"], "clinical-trial-application")
        self.assertEqual(prerequisite_facts["application_type"]["source"], "ectd_sequence_package")
        self.assertEqual(prerequisite_facts["application_type"]["source_label"], "eCTD sequence package")
        self.assertEqual(prerequisite_facts["application_type"]["evidence_status"], "local_evidence_present")
        self.assertEqual(prerequisite_facts["application_type"]["confidence"], "high")
        self.assertEqual(prerequisite_facts["application_type"]["review_action"], "confirm_if_business_context_disagrees")
        self.assertEqual(prerequisite_facts["product_type"]["status"], "present")
        self.assertEqual(prerequisite_facts["product_type"]["value"], "chemical")
        self.assertEqual(prerequisite_facts["submission_stage"]["status"], "present")
        self.assertEqual(prerequisite_facts["submission_stage"]["value"], "initial-submission")
        self.assertEqual(prerequisite_facts["package_scope"]["status"], "present")
        self.assertEqual(prerequisite_facts["package_scope"]["value"], "ectd_sequence_package")
        self.assertEqual(prerequisite_facts["registration_class"]["evidence_status"], "missing_required_prerequisite")
        self.assertEqual(prerequisite_facts["registration_class"]["confidence"], "none")
        self.assertEqual(prerequisite_facts["registration_class"]["source_label"], "not available")
        self.assertEqual(prerequisite_facts["registration_class"]["review_action"], "provide_before_applicability_review")
        self.assertIn(
            "registration_class",
            checklist["summary"]["missing_prerequisite_fact_keys"],
        )
        self.assertIn(
            "clinical_trial_completion_status",
            checklist["summary"]["missing_prerequisite_fact_keys"],
        )
        self.assertEqual(checklist["summary"]["present_prerequisite_fact_count"], 4)
        self.assertEqual(checklist["summary"]["missing_prerequisite_fact_count"], 2)

        self.assertIn("registration_class", checklist["missing_prerequisite_fact_keys"])
        self.assertIn("clinical_trial_completion_status", checklist["missing_prerequisite_fact_keys"])
        self.assertNotIn("application_type", checklist["missing_prerequisite_fact_keys"])
        self.assertNotIn("product_type", checklist["missing_prerequisite_fact_keys"])
        self.assertNotIn("submission_stage", checklist["missing_prerequisite_fact_keys"])
        self.assertNotIn("package_scope", checklist["missing_prerequisite_fact_keys"])

        items_by_id = {item["requirement_id"]: item for item in checklist["items"]}
        ctd_requirement = items_by_id[
            "cn_drug_registration_classification_and_dossier_requirements:req_ctd_base_submission"
        ]
        self.assertEqual(
            ctd_requirement["blocking_prerequisite_fact_keys"],
            ["registration_class"],
        )
        self.assertEqual(ctd_requirement["applicability_status"], "prerequisite_required")
        self.assertEqual(checklist["summary"]["deterministic_decision_count"], 0)

    def test_workbench_projects_phase_a_demo_summary_without_new_verdicts(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        submission_scope = {
            "upload_mode": "ectd_sequence_package",
            "available_scopes": ["document", "sequence", "activity", "application"],
            "ectd_project_context": {
                "sequence_package_count": 1,
                "regulatory_activity_count": 1,
                "application_project_count": 1,
                "sequence_packages": [
                    {
                        "sequence_package_id": "seqpkg:x202112345:0000",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                        "sequence_number": "0000",
                    }
                ],
            },
        }

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result={
                "rules": [
                    {
                        "rule_id": "HR-ECTD-001",
                        "category": "eCTD",
                        "status": "pass",
                        "message": "index.xml present",
                    },
                    {
                        "rule_id": "SR-ECTD-001",
                        "category": "eCTD",
                        "status": "warn",
                        "message": "review recommended",
                    },
                ],
                "risks": [{"risk_id": "risk_manual_review"}],
                "summary": {
                    "pass_rules": 1,
                    "warn_rules": 1,
                    "na_rules": 0,
                    "rule_count": 2,
                },
                "submission_scope": submission_scope,
            },
        )

        demo_summary = workbench["demo_summary"]
        self.assertEqual(demo_summary["schema_version"], "demo-summary-v1")
        self.assertEqual(demo_summary["phase"], "phase_a_demo_workbench")
        self.assertEqual(demo_summary["phase_status"], "demo_ready_in_progress")
        self.assertEqual(demo_summary["summary"]["closed_source_count"], 2)
        self.assertEqual(demo_summary["summary"]["source_count"], 5)
        self.assertEqual(demo_summary["summary"]["dossier_requirement_count"], 9)
        self.assertEqual(demo_summary["summary"]["deterministic_dossier_decision_count"], 0)
        self.assertEqual(
            demo_summary["summary"]["missing_prerequisite_fact_keys"],
            ["clinical_trial_completion_status", "registration_class"],
        )
        self.assertEqual(demo_summary["summary"]["rule_check_count"], 2)
        self.assertEqual(demo_summary["summary"]["risk_count"], 1)
        self.assertIn("deterministic_ectd_validation", demo_summary["demo_capability_codes"])
        self.assertIn("dossier_checklist_prerequisite_prompting", demo_summary["demo_capability_codes"])
        self.assertIn("content_consistency_checks", demo_summary["remaining_work_codes"])
        self.assertEqual(
            demo_summary["recommended_next_action_code"],
            "build_demo_report_summary_then_high_signal_consistency_checks",
        )
        self.assertIn("not a hard applicability", demo_summary["evidence_boundary"].lower())
        self.assertEqual(demo_summary["verdict_policy"], "no_new_verdicts_projection_only")

    def test_workbench_projects_high_signal_identity_consistency_without_rule_verdicts(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        submission_scope = {
            "upload_mode": "ectd_sequence_batch",
            "available_scopes": ["document", "sequence", "activity", "application"],
            "ectd_project_context": {
                "sequence_package_count": 2,
                "regulatory_activity_count": 1,
                "application_project_count": 1,
                "sequence_packages": [
                    {
                        "sequence_package_id": "seqpkg:x202112345:0000",
                        "sequence_root": "D:\\submission\\x202112345\\0000",
                        "sequence_name": "0000",
                        "application_root": "D:\\submission\\x202112345",
                        "application_root_name": "x202112345",
                        "application_number": "x202112345",
                        "sequence_number": "0000",
                        "filenames": ["index.xml", "cn-regional.xml"],
                    },
                    {
                        "sequence_package_id": "seqpkg:x202112999:0001",
                        "sequence_root": "D:\\submission\\x202112345\\0001",
                        "sequence_name": "0001",
                        "application_root": "D:\\submission\\x202112345",
                        "application_root_name": "x202112345",
                        "application_number": "x202112999",
                        "sequence_number": "0002",
                        "filenames": ["cn-regional.xml"],
                    },
                ],
            },
        }

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url=None,
            compliance_result={
                "rules": [],
                "risks": [],
                "summary": {},
                "submission_scope": submission_scope,
            },
        )

        consistency = workbench["content_consistency"]
        self.assertEqual(consistency["schema_version"], "content-consistency-v1")
        self.assertEqual(consistency["phase"], "phase_a_demo_workbench")
        self.assertEqual(consistency["verdict_policy"], "review_projection_only")
        self.assertEqual(consistency["summary"]["check_count"], 2)
        self.assertEqual(consistency["summary"]["issue_count"], 2)
        self.assertEqual(consistency["summary"]["deterministic_rule_verdict_count"], 0)
        checks_by_id = {item["check_id"]: item for item in consistency["checks"]}
        self.assertEqual(checks_by_id["ectd_application_identity"]["status"], "review_required")
        self.assertEqual(checks_by_id["ectd_sequence_identity"]["status"], "review_required")
        self.assertEqual(checks_by_id["ectd_application_identity"]["issue_count"], 1)
        self.assertEqual(checks_by_id["ectd_sequence_identity"]["issue_count"], 1)
        self.assertEqual(
            checks_by_id["ectd_application_identity"]["issues"][0]["field_name"],
            "application_number",
        )
        self.assertEqual(
            checks_by_id["ectd_sequence_identity"]["issues"][0]["field_name"],
            "sequence_number",
        )
        self.assertIn("manual review", consistency["evidence_boundary"].lower())
        self.assertIn("content_consistency_checks", workbench["demo_summary"]["demo_capability_codes"])
        self.assertEqual(workbench["demo_summary"]["summary"]["content_consistency_issue_count"], 2)
        self.assertNotIn("content_consistency_checks", workbench["demo_summary"]["remaining_work_codes"])

    def test_workbench_projects_demo_report_markdown_from_existing_projections(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        submission_scope = {
            "upload_mode": "ectd_sequence_batch",
            "available_scopes": ["document", "sequence", "activity", "application"],
            "ectd_project_context": {
                "sequence_package_count": 2,
                "regulatory_activity_count": 1,
                "application_project_count": 1,
                "sequence_packages": [
                    {
                        "sequence_package_id": "seqpkg:x202112345:0000",
                        "sequence_root": "D:\\submission\\x202112345\\0000",
                        "sequence_name": "0000",
                        "application_root": "D:\\submission\\x202112345",
                        "application_root_name": "x202112345",
                        "application_number": "x202112345",
                        "sequence_number": "0000",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                    },
                    {
                        "sequence_package_id": "seqpkg:x202112999:0001",
                        "sequence_root": "D:\\submission\\x202112345\\0001",
                        "sequence_name": "0001",
                        "application_root": "D:\\submission\\x202112345",
                        "application_root_name": "x202112345",
                        "application_number": "x202112999",
                        "sequence_number": "0002",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                    },
                ],
            },
        }

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url=None,
            demo_report_markdown_download_url="/api/v1/jobs/job_demo/demo-report/markdown/download",
            compliance_result={
                "rules": [],
                "risks": [],
                "summary": {},
                "submission_scope": submission_scope,
            },
        )

        report_markdown = workbench["demo_report_markdown"]
        self.assertEqual(
            workbench["demo_report_markdown_download_url"],
            "/api/v1/jobs/job_demo/demo-report/markdown/download",
        )
        self.assertIn("# AutoIND-Pro Phase A Demo Report", report_markdown)
        self.assertIn("demo-summary-v1", report_markdown)
        self.assertIn("content-consistency-v1", report_markdown)
        self.assertIn("Closed regulatory sources: 2/5", report_markdown)
        self.assertIn("Dossier requirements: 9", report_markdown)
        self.assertIn("Missing prerequisite facts: clinical_trial_completion_status, registration_class", report_markdown)
        self.assertIn("Content consistency issues: 2", report_markdown)
        self.assertIn("ectd_application_identity", report_markdown)
        self.assertIn("seqpkg:x202112999:0001", report_markdown)
        self.assertIn("review_projection_only", report_markdown)
        self.assertIn("No hard regulatory pass/fail", report_markdown)

    def test_workbench_projects_demo_flow_for_customer_walkthrough_without_new_verdicts(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        submission_scope = {
            "upload_mode": "ectd_sequence_batch",
            "available_scopes": ["document", "sequence", "activity", "application"],
            "ectd_project_context": {
                "sequence_package_count": 2,
                "regulatory_activity_count": 1,
                "application_project_count": 1,
                "sequence_packages": [
                    {
                        "sequence_package_id": "seqpkg:x202112345:0000",
                        "sequence_name": "0000",
                        "application_root_name": "x202112345",
                        "application_number": "x202112345",
                        "sequence_number": "0000",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                    },
                    {
                        "sequence_package_id": "seqpkg:x202112999:0001",
                        "sequence_name": "0001",
                        "application_root_name": "x202112345",
                        "application_number": "x202112999",
                        "sequence_number": "0002",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                    },
                ],
            },
        }

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url=None,
            demo_report_markdown_download_url="/api/v1/jobs/job_demo/demo-report/markdown/download",
            compliance_result={
                "rules": [],
                "risks": [],
                "summary": {},
                "submission_scope": submission_scope,
            },
        )

        demo_flow = workbench["demo_flow"]
        self.assertEqual(demo_flow["schema_version"], "demo-flow-v1")
        self.assertEqual(demo_flow["phase"], "phase_a_demo_workbench")
        self.assertEqual(demo_flow["verdict_policy"], "no_new_verdicts_walkthrough_only")
        self.assertEqual(demo_flow["summary"]["walkthrough_step_count"], 5)
        self.assertEqual(demo_flow["summary"]["issue_count"], 2)
        self.assertEqual(demo_flow["summary"]["missing_prerequisite_count"], 2)
        self.assertEqual(demo_flow["summary"]["deterministic_rule_verdict_count"], 0)
        self.assertEqual(
            demo_flow["summary"]["recommended_demo_mode"],
            "customer_readiness_walkthrough",
        )
        step_ids = [item["step_id"] for item in demo_flow["walkthrough_steps"]]
        self.assertEqual(
            step_ids,
            [
                "source_readiness",
                "dossier_prerequisites",
                "content_consistency_review",
                "report_download",
                "evidence_boundary_closeout",
            ],
        )
        self.assertEqual(
            demo_flow["walkthrough_steps"][1]["focus_items"],
            ["clinical_trial_completion_status", "registration_class"],
        )
        self.assertIn("ectd_application_identity", demo_flow["walkthrough_steps"][2]["focus_items"])
        self.assertIn("seqpkg:x202112999:0001", demo_flow["walkthrough_steps"][2]["focus_items"])
        self.assertEqual(
            demo_flow["walkthrough_steps"][3]["asset_url"],
            "/api/v1/jobs/job_demo/demo-report/markdown/download",
        )
        self.assertIn("No hard regulatory pass/fail", demo_flow["evidence_boundary"])
        self.assertIn("human review", demo_flow["walkthrough_steps"][4]["talk_track"].lower())

    def test_workbench_projects_stable_demo_scenario_without_new_verdicts(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        submission_scope = {
            "upload_mode": "ectd_sequence_batch",
            "available_scopes": ["document", "sequence", "activity", "application"],
            "ectd_project_context": {
                "sequence_package_count": 2,
                "regulatory_activity_count": 1,
                "application_project_count": 1,
                "sequence_packages": [
                    {
                        "sequence_package_id": "seqpkg:x202112345:0000",
                        "sequence_name": "0000",
                        "application_root_name": "x202112345",
                        "application_number": "x202112345",
                        "sequence_number": "0000",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                    },
                    {
                        "sequence_package_id": "seqpkg:x202112999:0001",
                        "sequence_name": "0001",
                        "application_root_name": "x202112345",
                        "application_number": "x202112999",
                        "sequence_number": "0002",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                    },
                ],
            },
        }

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url=None,
            demo_report_markdown_download_url="/api/v1/jobs/job_demo/demo-report/markdown/download",
            compliance_result={
                "rules": [],
                "risks": [],
                "summary": {},
                "submission_scope": submission_scope,
            },
        )

        scenario = workbench["demo_scenario"]
        self.assertEqual(scenario["schema_version"], "demo-scenario-v1")
        self.assertEqual(scenario["phase"], "phase_a_demo_workbench")
        self.assertEqual(scenario["scenario_id"], "phase_a_ectd_sequence_batch_readiness_demo")
        self.assertEqual(scenario["scenario_status"], "ready_for_controlled_demo")
        self.assertEqual(scenario["verdict_policy"], "no_new_verdicts_scenario_only")
        self.assertEqual(scenario["recommended_upload_mode"], "ectd_sequence_batch")
        self.assertEqual(scenario["summary"]["walkthrough_step_count"], 5)
        self.assertEqual(scenario["summary"]["expected_issue_count"], 2)
        self.assertEqual(scenario["summary"]["expected_missing_prerequisite_count"], 2)
        self.assertEqual(scenario["summary"]["deterministic_rule_verdict_count"], 0)
        self.assertIn("demo report download", scenario["demo_goal"].lower())
        self.assertEqual(
            scenario["recommended_sample_profile"]["must_have_local_evidence"],
            [
                "ectd_project_context.sequence_packages",
                "application_root_name",
                "application_number",
                "sequence_name",
                "sequence_number",
                "application_type",
                "product_type",
                "sequence_type",
            ],
        )
        self.assertIn("registration_class", scenario["recommended_sample_profile"]["intentionally_missing_prerequisites"])
        self.assertIn("clinical_trial_completion_status", scenario["recommended_sample_profile"]["intentionally_missing_prerequisites"])
        self.assertIn("content_consistency_review", scenario["demo_success_criteria"])
        self.assertIn("No hard regulatory pass/fail", scenario["do_not_claim"])
        self.assertEqual(
            scenario["primary_asset_urls"]["demo_report_markdown"],
            "/api/v1/jobs/job_demo/demo-report/markdown/download",
        )

    def test_workbench_projects_demo_script_markdown_from_scenario_and_flow(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        submission_scope = {
            "upload_mode": "ectd_sequence_batch",
            "available_scopes": ["document", "sequence", "activity", "application"],
            "ectd_project_context": {
                "sequence_package_count": 2,
                "regulatory_activity_count": 1,
                "application_project_count": 1,
                "sequence_packages": [
                    {
                        "sequence_package_id": "seqpkg:x202112345:0000",
                        "sequence_name": "0000",
                        "application_root_name": "x202112345",
                        "application_number": "x202112345",
                        "sequence_number": "0000",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                    },
                    {
                        "sequence_package_id": "seqpkg:x202112999:0001",
                        "sequence_name": "0001",
                        "application_root_name": "x202112345",
                        "application_number": "x202112999",
                        "sequence_number": "0002",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                    },
                ],
            },
        }

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url=None,
            demo_report_markdown_download_url="/api/v1/jobs/job_demo/demo-report/markdown/download",
            demo_script_markdown_download_url="/api/v1/jobs/job_demo/demo-script/markdown/download",
            compliance_result={
                "rules": [],
                "risks": [],
                "summary": {},
                "submission_scope": submission_scope,
            },
        )

        self.assertEqual(
            workbench["demo_script_markdown_download_url"],
            "/api/v1/jobs/job_demo/demo-script/markdown/download",
        )
        demo_script = workbench["demo_script_markdown"]
        self.assertIn("# AutoIND-Pro Customer Demo Script", demo_script)
        self.assertIn("demo-script-v1", demo_script)
        self.assertIn("phase_a_ectd_sequence_batch_readiness_demo", demo_script)
        self.assertIn("customer_readiness_walkthrough", demo_script)
        self.assertIn("source_readiness", demo_script)
        self.assertIn("dossier_prerequisites", demo_script)
        self.assertIn("content_consistency_review", demo_script)
        self.assertIn("report_download", demo_script)
        self.assertIn("evidence_boundary_closeout", demo_script)
        self.assertIn("No hard regulatory pass/fail", demo_script)
        self.assertIn("registration_class", demo_script)
        self.assertIn("clinical_trial_completion_status", demo_script)
        self.assertIn("/api/v1/jobs/job_demo/demo-report/markdown/download", demo_script)

    def test_workbench_projects_demo_run_checklist_without_new_verdicts(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        submission_scope = {
            "upload_mode": "ectd_sequence_batch",
            "available_scopes": ["document", "sequence", "activity", "application"],
            "ectd_project_context": {
                "sequence_package_count": 2,
                "regulatory_activity_count": 1,
                "application_project_count": 1,
                "sequence_packages": [
                    {
                        "sequence_package_id": "seqpkg:x202112345:0000",
                        "sequence_name": "0000",
                        "application_root_name": "x202112345",
                        "application_number": "x202112345",
                        "sequence_number": "0000",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                    },
                    {
                        "sequence_package_id": "seqpkg:x202112999:0001",
                        "sequence_name": "0001",
                        "application_root_name": "x202112345",
                        "application_number": "x202112999",
                        "sequence_number": "0002",
                        "application_type": "clinical-trial-application",
                        "product_type": "chemical",
                        "sequence_type": "initial-submission",
                    },
                ],
            },
        }

        workbench = api_main._build_workbench(
            parsed_documents=[],
            file_records=[],
            consistency_rows=[],
            markdown_download_url=None,
            demo_report_markdown_download_url="/api/v1/jobs/job_demo/demo-report/markdown/download",
            demo_script_markdown_download_url="/api/v1/jobs/job_demo/demo-script/markdown/download",
            compliance_result={
                "rules": [],
                "risks": [],
                "summary": {},
                "submission_scope": submission_scope,
            },
        )

        demo_run = workbench["demo_run"]
        self.assertEqual(demo_run["schema_version"], "demo-run-v1")
        self.assertEqual(demo_run["phase"], "phase_a_demo_workbench")
        self.assertEqual(demo_run["run_mode"], "controlled_customer_demo")
        self.assertEqual(demo_run["run_status"], "ready_with_review_items")
        self.assertEqual(demo_run["verdict_policy"], "no_new_verdicts_run_checklist_only")
        self.assertEqual(demo_run["summary"]["run_step_count"], 6)
        self.assertEqual(demo_run["summary"]["ready_step_count"], 4)
        self.assertEqual(demo_run["summary"]["review_required_step_count"], 1)
        self.assertEqual(demo_run["summary"]["prerequisite_prompt_step_count"], 1)
        self.assertEqual(demo_run["summary"]["deterministic_rule_verdict_count"], 0)
        self.assertEqual(
            [item["step_id"] for item in demo_run["run_steps"]],
            [
                "load_controlled_sample",
                "review_source_readiness",
                "explain_prerequisite_facts",
                "review_content_consistency",
                "open_demo_assets",
                "close_evidence_boundary",
            ],
        )
        self.assertIn("registration_class", demo_run["run_steps"][2]["focus_items"])
        self.assertIn("clinical_trial_completion_status", demo_run["run_steps"][2]["focus_items"])
        self.assertIn("ectd_application_identity", demo_run["run_steps"][3]["focus_items"])
        self.assertEqual(
            demo_run["asset_urls"]["demo_report_markdown"],
            "/api/v1/jobs/job_demo/demo-report/markdown/download",
        )
        self.assertEqual(
            demo_run["asset_urls"]["demo_script_markdown"],
            "/api/v1/jobs/job_demo/demo-script/markdown/download",
        )
        self.assertIn("No hard regulatory pass/fail", demo_run["evidence_boundary"])

    def test_controlled_demo_sample_job_returns_completed_workbench_without_upload(self) -> None:
        try:
            api_main = importlib.import_module("api.main")
            from fastapi.testclient import TestClient
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"api.main unavailable in this environment: {exc}") from exc

        client = TestClient(api_main.create_app())
        response = client.post("/api/v1/demo/controlled-sample/job")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["progress"], 100)
        self.assertEqual(payload["demo_sample"]["sample_id"], "phase_a_ectd_sequence_batch_readiness_demo")
        self.assertIs(payload["demo_sample"]["synthetic"], True)
        self.assertIn("not an uploaded regulatory submission", payload["demo_sample"]["evidence_boundary"])

        status_payload = client.get(f"/api/v1/jobs/{payload['job_id']}").json()
        self.assertEqual(status_payload["status"], "completed")
        self.assertEqual(status_payload["progress"], 100)
        self.assertEqual(status_payload["demo_sample"]["synthetic"], True)

        workbench_response = client.get(f"/api/v1/jobs/{payload['job_id']}/workbench")
        self.assertEqual(workbench_response.status_code, 200)
        workbench = workbench_response.json()
        self.assertEqual(workbench["demo_run"]["schema_version"], "demo-run-v1")
        self.assertEqual(workbench["demo_run"]["run_status"], "ready_with_review_items")
        self.assertEqual(workbench["demo_scenario"]["scenario_id"], "phase_a_ectd_sequence_batch_readiness_demo")
        self.assertIn("registration_class", workbench["dossier_checklist"]["missing_prerequisite_fact_keys"])
        self.assertIn(
            "clinical_trial_completion_status",
            workbench["dossier_checklist"]["missing_prerequisite_fact_keys"],
        )
        self.assertEqual(workbench["content_consistency"]["summary"]["issue_count"], 2)
        self.assertEqual(workbench["demo_run"]["summary"]["deterministic_rule_verdict_count"], 0)
        self.assertIn("demo_report_markdown", workbench["demo_run"]["asset_urls"])
        self.assertIn("demo_script_markdown", workbench["demo_run"]["asset_urls"])

        report_response = client.get(f"/api/v1/jobs/{payload['job_id']}/demo-report/markdown/download")
        script_response = client.get(f"/api/v1/jobs/{payload['job_id']}/demo-script/markdown/download")
        self.assertEqual(report_response.status_code, 200)
        self.assertEqual(script_response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
