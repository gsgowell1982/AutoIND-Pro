from __future__ import annotations

from typing import Any


def build_ectd_file_reuse_contract() -> dict[str, Any]:
    """Return the executable policy boundary for CN eCTD file reuse.

    The Chinese regional rule is stricter than the generic ICH Appendix 6
    guidance for cross-application references, so the regional prohibition is
    explicit and takes precedence for this product.
    """
    return {
        "schema_version": "ectd-file-reuse-contract-v1",
        "policy_precedence": "cn_regional_prohibition_over_ich_advisory",
        "china_regional_policy": {
            "regulation_id": "cn_ectd_technical_specification",
            "source_filename": "eCTD技术规范.pdf",
            "section": "3.3.4",
            "citation_anchor": "cn_ectd_technical_specification#sec_3_3_4",
            "cross_application_reference": {
                "prohibited": True,
                "rule_id": "HR-ECTD-027",
                "requirement_id": "cn_ectd_technical_specification:req_no_cross_application_leaf_reference",
            },
            "same_application_reuse": {
                "same_sequence": True,
                "prior_sequence": True,
                "physical_file_submitted_once": True,
                "leaf_may_reference_existing_file": True,
            },
        },
        "reuse_modes": {
            "same_sequence": {
                "allowed": True,
                "multiple_leaf_references_allowed": True,
                "caution": "Multiple pointers increase lifecycle-management complexity and should be visibly identified for reviewers.",
                "rule_id": "SR-ECTD-011",
            },
            "prior_sequence_same_application": {
                "allowed": True,
                "requires_resolvable_target": True,
                "requires_same_application_root": True,
                "requires_prior_sequence": True,
                "rule_id": "SR-ECTD-011",
            },
            "cross_application": {
                "allowed": False,
                "rule_id": "HR-ECTD-027",
            },
        },
        "lifecycle_operations": ["new", "replace", "append", "delete"],
        "operation_constraints": {
            "new": {
                "modified_file_required": False,
                "checksum_required": True,
            },
            "replace": {
                "modified_file_required": True,
                "target_must_be_current_leaf": True,
                "target_must_be_single_leaf": True,
                "checksum_required": True,
            },
            "append": {
                "modified_file_required": True,
                "target_must_be_current_leaf": True,
                "target_must_be_single_leaf": True,
                "checksum_required": True,
                "same_sequence_append_requires_authority_consultation": True,
            },
            "delete": {
                "modified_file_required": True,
                "target_must_be_current_leaf": True,
                "target_must_be_single_leaf": True,
                "checksum_must_be_empty": True,
                "entity_file_submitted": False,
            },
        },
        "source_references": {
            "cn_section_3_3_4": {
                "source_filename": "eCTD技术规范.pdf",
                "source_path": "data/regulations/eCTD技术规范.pdf",
                "section": "3.3.4",
                "pdf_page": 22,
                "description": "同一序列或同一申请前序序列可以通过骨架叶元素复用文件，不支持跨申请引用。",
            },
            "ich_appendix_6_file_reuse": {
                "source_filename": "eCTD_Specification_v3_2_2_0.pdf",
                "source_path": "data/regulations/eCTD_Specification_v3_2_2_0.pdf",
                "section": "Appendix 6 - File Reuse",
                "pdf_page": 103,
                "logical_page": "6-6",
                "description": "一个实体文件可以由多个 leaf 元素引用；跨序列或跨申请引用必须准确给出 xlink:href，并应事先咨询监管机构。",
            },
            "ich_appendix_6_operation_attribute": {
                "source_filename": "eCTD_Specification_v3_2_2_0.pdf",
                "source_path": "data/regulations/eCTD_Specification_v3_2_2_0.pdf",
                "section": "Appendix 6 - Operation Attribute",
                "pdf_page": 100,
                "logical_page": "6-3",
            },
        },
    }
