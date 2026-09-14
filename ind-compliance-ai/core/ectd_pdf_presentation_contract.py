"""Traceable PDF presentation requirements for Chinese eCTD submissions."""

from __future__ import annotations

from typing import Any


PDF_PRESENTATION_CONTRACT_VERSION = "ectd-pdf-presentation-contract-v1"


def build_ectd_pdf_presentation_contract() -> dict[str, Any]:
    """Return the review contract without making a verdict for missing evidence.

    Existing parsers and validators remain authoritative for deterministic checks. This
    contract makes the regional requirements, ICH recommendations, source locations,
    and manual-review boundary available beside every related finding.
    """

    return {
        "schema_version": PDF_PRESENTATION_CONTRACT_VERSION,
        "scope": "pdf_material_presentation_and_navigation",
        "source_references": {
            "cn_technical_specification": {
                "source_filename": "eCTD技术规范.pdf",
                "section": "3.4",
                "heading": "PDF 电子提交标准",
                "pdf_page": 23,
                "continuation_pdf_page": 24,
                "citation_anchor": "cn_ectd_technical_specification#sec_3_4",
            },
            "ich_submission_formats": {
                "source_filename": "Specification_for_Submission_Formats_for_eCTD_v1_2.pdf",
                "sections": [f"2.{index}" for index in range(1, 19)],
                "pdf_page_start": 4,
                "pdf_page_end": 9,
                "citation_anchor": "ich_submission_formats#sec_2",
            },
            "cn_validation_standard": {
                "source_filename": "eCTD验证标准.pdf",
                "sections": ["6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9", "6.10", "6.11", "6.12", "6.13", "6.14", "6.15", "6.19", "6.22", "6.23", "6.24", "6.25", "6.26"],
                "citation_anchor": "cn_ectd_validation_standard#sec_6",
            },
        },
        "deterministic_checks": {
            "file_size": {
                "max_bytes": 500 * 1024 * 1024,
                "requirement_id": "cn_ectd_validation_standard:req_file_size_limit",
                "citation_anchor": "cn_ectd_validation_standard#sec_2_2",
            },
            "navigation": {
                "long_document_page_threshold_exclusive": 5,
                "bookmark_or_toc_required": True,
                "hyperlink_fallback_allowed": True,
                "requirement_ids": [
                    "cn_ectd_technical_specification:req_long_pdf_navigation_aids",
                    "cn_ectd_technical_specification:req_hyperlink_navigation_support",
                ],
            },
            "hyperlink_integrity": {
                "target_file_presence_required": True,
                "target_resolvability_required": True,
                "cross_application_links_prohibited": True,
                "requirement_ids": [
                    "cn_ectd_technical_specification:req_pdf_hyperlink_target_file_presence",
                    "cn_ectd_technical_specification:req_pdf_hyperlink_target_resolvability",
                    "cn_ectd_technical_specification:req_no_cross_application_pdf_hyperlinks",
                ],
            },
            "validation_standard": {
                "content_searchable": True,
                "fast_web_access_enabled": True,
                "security_settings_absent": True,
                "non_standard_fonts_embedded": True,
                "bookmark_and_hyperlink_actions_valid": True,
            },
        },
        "chinese_regional_requirements": {
            "long_document_navigation": {
                "page_threshold_exclusive": 5,
                "exceptions": ["foreign_reference_material", "literature_reference", "application_form"],
                "required_evidence": ["table_of_contents", "table_of_tables", "table_of_figures", "bookmarks"],
                "hyperlink_fallback_when_navigation_unavailable": True,
            },
            "font_family": {
                "required_for_chinese_submission": "宋体",
                "automation_boundary": "manual_review_when_role_or_font_mapping_is_unavailable",
            },
            "font_sizes_pt": {
                "narrative_min": 12,
                "table_min": 10.5,
                "toc_min": 12,
                "footnote_min": 10.5,
                "figure_or_table_text_recommended_min": 8,
                "automation_boundary": "manual_review_when_semantic_role_or_font_evidence_is_unavailable",
            },
            "font_colors": {
                "narrative": "black",
                "hyperlink_allowed": ["blue", "black_with_blue_border"],
                "automation_boundary": "manual_review_when_text_role_or_color_evidence_is_unavailable",
            },
            "cross_application_hyperlinks": {
                "prohibited": True,
                "manual_review_when_application_boundary_is_ambiguous": True,
            },
        },
        "ich_recommendations": {
            "pdf_version": True,
            "embedded_fonts": True,
            "opentype_or_truetype": True,
            "page_orientation_and_size": True,
            "binding_margin_cm_min": 2.5,
            "other_margin_cm_min": 1.0,
            "unique_header_footer_identifier": True,
            "searchable_source_or_ocr": True,
            "scan_resolution_dpi": {"default_recommended": 300, "image_recommended": 600},
            "image_compression_and_color_matching": True,
            "icc_profile_review": True,
            "page_numbering": True,
            "initial_view": True,
            "optimization": True,
            "security": True,
            "acrobat_plugin_dependency_review": True,
        },
        "automation_boundary": {
            "deterministic": [
                "file_size",
                "page_count_and_navigation_evidence",
                "hyperlink_target_presence_and_resolvability",
                "cross_application_link_detection_when_application_roots_are_known",
                "font_embedding_and_pdf_security_metadata",
            ],
            "manual_review": [
                "Chinese font family by semantic text role",
                "Chinese font size by semantic text role",
                "narrative_and_hyperlink color semantics",
                "scan quality, ICC profile suitability, image fidelity",
                "header/footer uniqueness and regulatory meaning",
                "Acrobat plug-in dependence",
            ],
            "manual_review_status": "review_required",
        },
    }

