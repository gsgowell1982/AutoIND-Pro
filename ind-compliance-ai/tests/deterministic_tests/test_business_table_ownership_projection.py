from __future__ import annotations

import unittest

from parsers.pdf import postprocess


class BusinessTableOwnershipProjectionTests(unittest.TestCase):
    def test_strong_business_semantic_projection_gate_is_centralized(self) -> None:
        strong_projection_keys = [
            "dose_response_result_panel_projection",
            "genotoxicity_assay_matrix_projection",
            "toxicology_summary_schema_projection",
        ]
        for projection_key in strong_projection_keys:
            with self.subTest(projection_key=projection_key):
                table = {
                    "semantic_role": "business_table",
                    "semantic_projection_v2": {
                        projection_key: {"semantic_profile": projection_key},
                    },
                }
                self.assertTrue(
                    postprocess._table_has_strong_business_semantic_projection(table),
                    msg=table,
                )

        self.assertTrue(
            postprocess._table_has_strong_business_semantic_projection(
                {
                    "semantic_role": "business_table",
                    "semantic_projection_v2": {
                        "dose_response_result_panel_projection": {
                            "semantic_profile": "dose_response_result_panel",
                            "has_study_context": True,
                        },
                    },
                }
            )
        )
        self.assertFalse(
            postprocess._table_has_strong_business_semantic_projection(
                {
                    "semantic_role": "business_table",
                    "semantic_projection_v2": {
                        "single_column_keyed_list_projection": {
                            "semantic_profile": "keyed_long_list",
                        },
                    },
                }
            )
        )
        self.assertFalse(
            postprocess._table_has_strong_business_semantic_projection(
                {
                    "semantic_role": "figure",
                    "semantic_projection_v2": {
                        "toxicology_summary_schema_projection": {
                            "semantic_profile": "toxicology_summary_schema_table",
                        },
                    },
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
