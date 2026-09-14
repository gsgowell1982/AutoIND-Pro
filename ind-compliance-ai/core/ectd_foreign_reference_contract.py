"""Traceable foreign reference material and language classification requirements for Chinese eCTD submissions."""

from __future__ import annotations

from typing import Any


FOREIGN_REFERENCE_CONTRACT_VERSION = "ectd-foreign-reference-contract-v1"

# ISO 639-1 two-letter language codes
# Reference: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
ISO639_1_CODES = [
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az",
    "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs",
    "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy",
    "da", "de", "dv", "dz",
    "ee", "el", "en", "eo", "es", "et", "eu",
    "fa", "ff", "fi", "fj", "fo", "fr", "fy",
    "ga", "gd", "gl", "gn", "gu", "gv",
    "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz",
    "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu",
    "ja", "jv",
    "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku", "kv", "kw", "ky",
    "la", "lb", "lg", "li", "ln", "lo", "lt", "lu", "lv",
    "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
    "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny",
    "oc", "oj", "om", "or", "os",
    "pa", "pi", "pl", "ps", "pt",
    "qu",
    "rm", "rn", "ro", "ru", "rw",
    "sa", "sc", "sd", "se", "sg", "si", "sk", "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw",
    "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt", "tw", "ty",
    "ug", "uk", "ur", "uz",
    "ve", "vi", "vo",
    "wa", "wo",
    "xh",
    "yi", "yo",
    "za", "zh", "zu"
]


def build_ectd_foreign_reference_contract() -> dict[str, Any]:
    """Return the foreign reference and language classification contract.

    This contract defines the boundary between Chinese dossier material and foreign
    reference material, based on the xml:lang attribute classification rules in
    eCTD技术规范.pdf section 3.5.

    Existing rule evaluators remain authoritative for pass/fail verdicts. This contract
    makes the classification boundary, ISO 639-1 language codes, structure requirements,
    and lifecycle consistency rules available beside every related finding.
    """

    return {
        "schema_version": FOREIGN_REFERENCE_CONTRACT_VERSION,
        "scope": "foreign_reference_material_language_classification_and_structure",
        "source_references": {
            "cn_technical_specification": {
                "source_filename": "eCTD技术规范.pdf",
                "section": "3.5",
                "heading": "外文参考资料的要求",
                "subsections": {
                    "3.5.1": {
                        "heading": "语言属性的设置",
                        "page_range": "PDF pages covering language attribute classification",
                        "implemented_rule_ids": ["HR-ECTD-023"],
                        "requirement_ids": ["cn_ectd_technical_specification:req_leaf_xml_lang_attribute_classification"],
                    },
                    "3.5.2": {
                        "heading": "语言属性的生命周期管理",
                        "page_range": "PDF pages covering language lifecycle consistency",
                        "implemented_rule_ids": ["HR-ECTD-026"],
                        "requirement_ids": ["cn_ectd_technical_specification:req_replace_leaf_language_class_consistency"],
                    },
                },
                "structure_requirements": {
                    "heading": "外文参考资料同级结构要求",
                    "implemented_rule_ids": ["HR-ECTD-024"],
                    "requirement_ids": ["cn_ectd_technical_specification:req_foreign_reference_leaf_sibling_structure"],
                },
            },
        },
        "language_classification": {
            "classification_boundary": "eCTD leaf elements are classified as Chinese dossier material or foreign reference material based on the xml:lang attribute value following 3.5.1 rules.",
            "chinese_dossier_indicators": {
                "xml_lang_values": ["zh", "", None],
                "description": "Leafs with xml:lang='zh', empty xml:lang, or missing xml:lang are classified as Chinese dossier material.",
                "note": "Empty string and missing attribute are treated identically as Chinese dossier indicators.",
            },
            "foreign_reference_indicators": {
                "xml_lang_pattern": "non-zh ISO 639-1 language codes",
                "description": "Leafs with xml:lang values other than 'zh' are classified as foreign reference material if the value is a valid ISO 639-1 code.",
                "validation": "Non-'zh' xml:lang values must be valid two-letter ISO 639-1 language codes.",
            },
            "iso639_1_requirement": {
                "standard": "ISO 639-1",
                "validation_scope": "All non-'zh' xml:lang attribute values",
                "valid_codes": ISO639_1_CODES,
                "invalid_code_handling": "Surface as deterministic finding; invalid codes prevent proper language classification.",
            },
        },
        "structure_requirements": {
            "sibling_colocation": {
                "requirement": "Foreign reference leafs should be colocated with a same-parent Chinese dossier leaf.",
                "scope": "Applied when local eCTD leaf structure metadata is available.",
                "rationale": "Ensures foreign reference materials are properly associated with their corresponding Chinese dossier content.",
                "failure_condition": "Foreign reference leaf exists without a Chinese dossier sibling under the same parent element.",
            },
            "ordering_constraint": {
                "requirement": "Foreign reference leafs should be ordered after their corresponding Chinese dossier leaf.",
                "relative_position": "foreign_after_chinese",
                "scope": "Applied when sibling colocation is satisfied.",
                "rationale": "Maintains consistent document organization with primary Chinese content followed by reference materials.",
                "failure_condition": "Foreign reference leaf appears before its Chinese dossier sibling in the leaf sequence.",
            },
        },
        "lifecycle_consistency": {
            "replace_operation_rule": "A replace-operation eCTD leaf should preserve the Chinese-dossier or foreign-reference language class of the uniquely matched prior-sequence leaf in the same application.",
            "matching_scope": "same_application_unique_prior_match",
            "consistency_requirement": "Language class (Chinese dossier vs. foreign reference) must remain stable across replace operations.",
            "rationale": "Prevents unintended language classification changes during document lifecycle updates.",
            "applicable_operation": "replace",
            "prerequisite": "Unique prior-sequence leaf match exists in the same application.",
            "out_of_scope_conditions": [
                "Missing prior sequence history",
                "Ambiguous leaf matches (multiple candidates)",
                "Unavailable leaf metadata",
                "Operations other than 'replace' (new, append, delete)",
            ],
            "failure_condition": "Replace operation changes language class (Chinese↔Foreign) compared to matched prior leaf.",
        },
        "automation_boundary": {
            "deterministic": [
                "xml:lang attribute presence and value extraction",
                "ISO 639-1 language code validation for non-zh values",
                "Language classification (Chinese dossier vs. foreign reference) by xml:lang value",
                "Same-parent sibling structure detection when leaf metadata available",
                "Ordering validation (foreign after Chinese) when sibling structure detected",
                "Replace operation language class comparison when unique prior match exists",
            ],
            "manual_review": [
                "Substantive content-language adequacy assessment",
                "Translation quality evaluation",
                "Appropriateness of foreign reference material selection",
                "Regulatory acceptability of foreign language choices",
            ],
            "out_of_scope": [
                "Language inference from filename, directory path, or document text content",
                "Content-language consistency checking (whether document text matches declared xml:lang)",
                "Translation completeness or accuracy validation",
                "Regulatory policy decisions on acceptable foreign languages for specific submission types",
            ],
            "metadata_prerequisite_handling": "When leaf structure metadata or prior-sequence history is unavailable, affected checks return 'na' status rather than 'fail', with explicit prerequisite guidance.",
        },
    }
