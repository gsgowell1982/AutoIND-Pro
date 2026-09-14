from __future__ import annotations

from typing import Any

from core.ectd_controlled_vocabulary_rules import build_ectd_vocabulary_rule_contract


_CLINICAL_TRIAL_ROWS = [
    {"sequence_number": "0000", "related_sequence": "0000", "regulatory_activity_type_code": "cnrat1", "sequence_type_code": "cnsqt1", "description_intent": "clinical_trial_application_indication"},
    {"sequence_number": "0001", "related_sequence": "0000", "regulatory_activity_type_code": "cnrat1", "sequence_type_code": "cnsqt2", "description_intent": "response_to_sequence", "description_target_sequence": "0000"},
    {"sequence_number": "0002", "related_sequence": "0002", "regulatory_activity_type_code": "cnrat2", "sequence_type_code": "cnsqt1", "description_intent": "supplement_submission"},
    {"sequence_number": "0003", "related_sequence": "0002", "regulatory_activity_type_code": "cnrat2", "sequence_type_code": "cnsqt2", "description_intent": "response_to_sequence", "description_target_sequence": "0002"},
    {"sequence_number": "0004", "related_sequence": "0004", "regulatory_activity_type_code": "cnrat5", "sequence_type_code": "cnsqt1", "description_intent": "new_indication_and_drug_combination"},
    {"sequence_number": "0005", "related_sequence": "0005", "regulatory_activity_type_code": "cnrat2", "sequence_type_code": "cnsqt1", "description_intent": "supplement_submission"},
    {"sequence_number": "0006", "related_sequence": "0005", "regulatory_activity_type_code": "cnrat2", "sequence_type_code": "cnsqt2", "description_intent": "response_to_sequence", "description_target_sequence": "0005"},
    {"sequence_number": "0007", "related_sequence": "0004", "regulatory_activity_type_code": "cnrat5", "sequence_type_code": "cnsqt2", "description_intent": "response_to_sequence", "description_target_sequence": "0004"},
    {"sequence_number": "0008", "related_sequence": "0008", "regulatory_activity_type_code": "cnrat7", "sequence_type_code": "cnsqt1", "description_intent": "development_safety_update"},
    {"sequence_number": "0009", "related_sequence": "0009", "regulatory_activity_type_code": "cnrat7", "sequence_type_code": "cnsqt1", "description_intent": "potential_serious_safety_risk"},
]

_NEW_DRUG_ROWS = [
    {"sequence_number": "0000", "related_sequence": "0000", "regulatory_activity_type_code": "cnrat1", "sequence_type_code": "cnsqt1", "description_intent": "new_drug_application_indication"},
    {"sequence_number": "0001", "related_sequence": "0000", "regulatory_activity_type_code": "cnrat1", "sequence_type_code": "cnsqt2", "description_intent": "response_to_sequence", "description_target_sequence": "0000"},
    {"sequence_number": "0002", "related_sequence": "0000", "regulatory_activity_type_code": "cnrat1", "sequence_type_code": "cnsqt2", "description_intent": "response_to_sequence", "description_target_sequence": "0000"},
    {"sequence_number": "0003", "related_sequence": "0003", "regulatory_activity_type_code": "cnrat2", "sequence_type_code": "cnsqt1", "description_intent": "manufacturing_process_change"},
    {"sequence_number": "0004", "related_sequence": "0004", "regulatory_activity_type_code": "cnrat2", "sequence_type_code": "cnsqt1", "description_intent": "analytical_method_change"},
    {"sequence_number": "0005", "related_sequence": "0003", "regulatory_activity_type_code": "cnrat2", "sequence_type_code": "cnsqt2", "description_intent": "response_to_sequence", "description_target_sequence": "0003"},
    {"sequence_number": "0006", "related_sequence": "0006", "regulatory_activity_type_code": "cnrat6", "sequence_type_code": "cnsqt1", "description_intent": "new_indication"},
    {"sequence_number": "0007", "related_sequence": "0004", "regulatory_activity_type_code": "cnrat2", "sequence_type_code": "cnsqt2", "description_intent": "response_to_sequence", "description_target_sequence": "0004"},
    {"sequence_number": "0008", "related_sequence": "0008", "regulatory_activity_type_code": "cnrat8", "sequence_type_code": "cnsqt1", "description_intent": "renewal"},
]

_CLINICAL_TRIAL_APPLICATION_TYPE_ALIASES = {
    "cnapt1",
    "clinical-trial-application",
    "clinical_trial_application",
    "clinical trial application",
}

_DESCRIPTION_INTENT_TERMS = {
    "clinical_trial_application_indication": ("clinical trial", "临床试验", "indication", "适应症"),
    "response_to_sequence": ("response", "回复", "发补", "补充回复"),
    "supplement_submission": ("supplement", "补充资料", "补充申请"),
    "new_indication_and_drug_combination": ("new indication", "新适应症", "drug combination", "联合用药"),
    "development_safety_update": ("development safety", "安全性更新", "研发期间安全"),
    "potential_serious_safety_risk": ("potential serious", "严重安全性风险", "潜在严重"),
}


def _description_intent_evidence(description: str, expected_intent: str) -> dict[str, Any]:
    normalized = str(description or "").strip().lower()
    terms = _DESCRIPTION_INTENT_TERMS.get(expected_intent, ())
    matched_terms = [term for term in terms if term.lower() in normalized]
    return {
        "expected_intent": expected_intent,
        "matched_terms": matched_terms,
        "confirmed": bool(matched_terms),
    }


def build_ectd_sequence_semantic_contract(*, vocabulary_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    vocabulary = vocabulary_contract or build_ectd_vocabulary_rule_contract()
    return {
        "schema_version": "ectd-sequence-semantic-contract-v1",
        "source": {
            "regulation_id": "cn_ectd_technical_specification",
            "source_clause_id": "cn_ectd_technical_specification:sec_2_2_2",
            "table": "table_1_clinical_trial_application_related_sequences",
            "source_locator": "eCTD技术规范.pdf#page=13",
        },
        "sequence_number_policy": {
            "format": "^\\d{4}$",
            "starts_at": "0000",
            "requires_contiguous_history": True,
            "example_range": ["0000", "0009"],
            "example_range_not_exhaustive": True,
        },
        "scenarios": {
            "clinical_trial_application": {
                "application_type_code": "cnapt1",
                "application_category": "clinical_trial_application",
                "source_table": "table_1_clinical_trial_application_related_sequences",
                "rows": list(_CLINICAL_TRIAL_ROWS),
                "description_policy": {
                    "max_characters": 120,
                    "semantic_evidence_required_for_automatic_confirmation": False,
                    "note": "Descriptions are purpose evidence; free text must not be compared as an exact literal.",
                },
            },
            "new_drug_application": {
                "application_type_code": "cnapt2",
                "application_category": "new_drug_application",
                "source_table": "table_2_new_drug_application_related_sequences",
                "rows": list(_NEW_DRUG_ROWS),
                "description_policy": {
                    "max_characters": 120,
                    "semantic_evidence_required_for_automatic_confirmation": False,
                    "note": "Descriptions are purpose evidence; free text must not be compared as an exact literal.",
                },
            },
        },
        "relationship_policy": {
            "source_clause_id": "cn_ectd_technical_specification:sec_2_4",
            "source_table": "table_3_application_activity_sequence_relationship_examples",
            "authoritative_matrix": "depend-apt-rat-sqt",
            "examples_are_non_exhaustive": True,
            "unlisted_but_matrix_compatible_is_allowed": True,
            "incompatible_triplet_requires_manual_review": True,
        },
        "sequence_quality_policy": {
            "description_max_characters": 120,
            "required_contact_fields": ["name", "phone", "email"],
            "prohibited_description_substitution_review": True,
            "prohibited_description_terms": ["监管机构问题", "说明函", "向监管机构提问", "question to regulator", "response letter"],
        },
        "controlled_vocabulary_refs": {
            "application_type": "cv-application-type",
            "regulatory_activity_type": "cv-regulatory-activity-type",
            "sequence_type": "cv-sequence-type",
        },
        "compatibility_matrix": dict(vocabulary.get("type_compatibility") or {}),
    }


def _application_scenario(application_type: str, contract: dict[str, Any]) -> dict[str, Any] | None:
    normalized = str(application_type or "").strip().lower()
    for scenario in contract.get("scenarios", {}).values():
        aliases = {
            str(scenario.get("application_type_code") or "").strip().lower(),
            *(_CLINICAL_TRIAL_APPLICATION_TYPE_ALIASES if scenario.get("application_category") == "clinical_trial_application" else set()),
        }
        if normalized in aliases:
            return dict(scenario)
    return None


def validate_ectd_sequence_semantics(
    application_type: str,
    sequences: list[dict[str, Any]],
    *,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_contract = contract or build_ectd_sequence_semantic_contract()
    scenario = _application_scenario(application_type, active_contract)
    rows_by_number = {
        str(row.get("sequence_number") or "").strip(): dict(row)
        for row in (scenario or {}).get("rows", []) or []
    }
    findings: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    normalized_sequences: list[dict[str, Any]] = []
    observed_numbers: list[int] = []
    for raw_sequence in sequences or []:
        sequence = dict(raw_sequence or {})
        number = str(sequence.get("sequence_number") or sequence.get("name") or "").strip()
        normalized = {**sequence, "sequence_number": number}
        normalized_sequences.append(normalized)
        if len(number) == 4 and number.isdigit():
            observed_numbers.append(int(number))
        else:
            findings.append({"issue_code": "invalid_sequence_number_format", "sequence_number": number})

    unique_numbers = sorted(set(observed_numbers))
    if unique_numbers:
        if unique_numbers[0] != 0:
            findings.append({"issue_code": "sequence_history_missing_0000", "observed_sequence_numbers": [f"{n:04d}" for n in unique_numbers]})
        expected = list(range(unique_numbers[0], unique_numbers[-1] + 1))
        if unique_numbers != expected:
            findings.append({"issue_code": "sequence_history_gap", "observed_sequence_numbers": [f"{n:04d}" for n in unique_numbers]})

    normalized_application_type = str(application_type or "").strip().lower()
    compatibility = dict(active_contract.get("compatibility_matrix") or {})
    compatibility_rows = {
        (
            str(row.get("application_type") or "").strip().lower(),
            str(row.get("regulatory_activity_type") or "").strip().lower(),
            str(row.get("sequence_type") or "").strip().lower(),
        )
        for row in compatibility.get("rows", []) or []
        if isinstance(row, dict)
    }
    if scenario is None and any(item[0] == normalized_application_type for item in compatibility_rows):
        for sequence in normalized_sequences:
            regulatory_activity_type = str(sequence.get("regulatory_activity_type") or "").strip().lower()
            sequence_type = str(sequence.get("sequence_type") or "").strip().lower()
            if regulatory_activity_type and sequence_type and (
                normalized_application_type,
                regulatory_activity_type,
                sequence_type,
            ) not in compatibility_rows:
                findings.append(
                    {
                        "issue_code": "incompatible_type_triplet",
                        "sequence_number": sequence["sequence_number"],
                        "application_type": normalized_application_type,
                        "regulatory_activity_type": regulatory_activity_type,
                        "sequence_type": sequence_type,
                    }
                )
        return {
            "status": "fail" if findings else "pass",
            "application_type": str(application_type or "").strip(),
            "scenario": "table3_relationship_examples",
            "findings": findings,
            "review_required": bool(review_items),
            "review_items": review_items,
            "sequence_count": len(normalized_sequences),
            "example_range_not_exhaustive": bool(active_contract.get("sequence_number_policy", {}).get("example_range_not_exhaustive")),
        }
    if scenario is None:
        return {
            "status": "not_applicable",
            "application_type": str(application_type or "").strip(),
            "findings": findings,
            "review_required": bool(review_items),
            "review_items": review_items,
            "sequence_count": len(normalized_sequences),
            "example_range_not_exhaustive": bool(active_contract.get("sequence_number_policy", {}).get("example_range_not_exhaustive")),
        }

    example_numbers = [int(value) for value in rows_by_number if value.isdigit()]
    example_maximum = max(example_numbers) if example_numbers else None
    if example_maximum is not None:
        for sequence in normalized_sequences:
            number = sequence["sequence_number"]
            if number.isdigit() and len(number) == 4 and int(number) > example_maximum:
                review_items.append(
                    {
                        "issue_code": "sequence_outside_table_example_range",
                        "sequence_number": number,
                        "example_range": [f"{min(example_numbers):04d}", f"{example_maximum:04d}"],
                        "message": "Sequence is beyond the illustrative Table 1/Table 2 range; review against current procedure and controlled vocabulary.",
                    }
                )

    for sequence in normalized_sequences:
        number = sequence["sequence_number"]
        expected = rows_by_number.get(number)
        if expected is not None:
            checks = (
                ("related_sequence", "related_sequence_mismatch"),
                ("regulatory_activity_type", "regulatory_activity_type_mismatch"),
                ("sequence_type", "sequence_type_mismatch"),
            )
            expected_values = {
                "related_sequence": expected.get("related_sequence"),
                "regulatory_activity_type": expected.get("regulatory_activity_type_code"),
                "sequence_type": expected.get("sequence_type_code"),
            }
            for field_name, issue_code in checks:
                observed = str(sequence.get(field_name) or "").strip()
                if observed and observed != str(expected_values[field_name] or "").strip():
                    findings.append({
                        "issue_code": issue_code,
                        "sequence_number": number,
                        "field": field_name,
                        "observed": observed,
                        "expected": expected_values[field_name],
                    })
        description = str(sequence.get("sequence_description") or "").strip()
        quality_policy = dict(active_contract.get("sequence_quality_policy") or {})
        max_description_characters = int(quality_policy.get("description_max_characters", 120) or 120)
        if len(description) > max_description_characters:
            findings.append(
                {
                    "issue_code": "sequence_description_too_long",
                    "sequence_number": number,
                    "description_length": len(description),
                    "max_characters": max_description_characters,
                }
            )
        expected_intent = str((expected or {}).get("description_intent") or "").strip()
        if description and expected_intent:
            evidence = _description_intent_evidence(description, expected_intent)
            if not evidence["confirmed"]:
                review_items.append(
                    {
                        "issue_code": "sequence_description_intent_unconfirmed",
                        "sequence_number": number,
                        "description_intent": expected_intent,
                        "description": description,
                        "evidence": evidence,
                    }
                )
        prohibited_terms = tuple(quality_policy.get("prohibited_description_terms") or ())
        matched_prohibited_terms = [term for term in prohibited_terms if term.lower() in description.lower()]
        if matched_prohibited_terms:
            review_items.append(
                {
                    "issue_code": "sequence_description_prohibited_use_suspected",
                    "sequence_number": number,
                    "matched_terms": matched_prohibited_terms,
                    "description": description,
                }
            )
        contact_fields = {
            "name": str(sequence.get("sequence_contact_name") or "").strip(),
            "phone": str(sequence.get("sequence_contact_phone") or "").strip(),
            "email": str(sequence.get("sequence_contact_email") or "").strip(),
        }
        observed_contact_fields = [value for value in contact_fields.values() if value]
        if observed_contact_fields:
            missing_contact_fields = [field for field, value in contact_fields.items() if not value]
            if missing_contact_fields:
                findings.append(
                    {
                        "issue_code": "sequence_contact_incomplete",
                        "sequence_number": number,
                        "missing_fields": missing_contact_fields,
                    }
                )
            email = contact_fields["email"]
            if email and ("@" not in email or "." not in email.rsplit("@", 1)[-1]):
                findings.append(
                    {
                        "issue_code": "sequence_contact_email_invalid",
                        "sequence_number": number,
                        "field": "email",
                        "observed": email,
                    }
                )

    return {
        "status": "fail" if findings else "pass",
        "application_type": str(application_type or "").strip(),
        "scenario": "table2_new_drug_application" if scenario.get("application_category") == "new_drug_application" else "table1_clinical_trial_application",
        "findings": findings,
        "review_required": bool(review_items),
        "review_items": review_items,
        "sequence_count": len(normalized_sequences),
        "covered_example_sequences": sorted(rows_by_number),
        "example_range_not_exhaustive": bool(active_contract.get("sequence_number_policy", {}).get("example_range_not_exhaustive")),
    }


__all__ = ["build_ectd_sequence_semantic_contract", "validate_ectd_sequence_semantics"]
