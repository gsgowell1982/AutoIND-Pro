# Stable Table Row Lineage and Context Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve source-backed table rows across semantic and Markdown projection, and prevent absorbed study-context facts from being rendered twice.

**Architecture:** Attach stable references to display rows and propagate those references through semantic and presentation projections. Structural rows, presentation spans, and continuation composition resolve ownership by reference; normalized signatures remain a unique-only legacy fallback. Treat absorbed study context as typed label/value facts and transfer only uncovered facts.

**Tech Stack:** Python 3.11, `unittest`, PDF table postprocessing, typed semantic projection, IND-review Markdown.

---

## File Map

- Modify `parsers/pdf/table_modules/postprocess.py`: create stable display-row provenance and attach it to `merged_rows`.
- Modify `parsers/pdf/postprocess.py`: align semantic rows to display provenance; extract and audit study-context facts.
- Modify `api/main.py`: propagate presentation-row provenance and resolve structural rows and spans by source reference.
- Modify `tests/deterministic_tests/test_parse_markdown_export.py`: deterministic Markdown RED/GREEN coverage.
- Modify `tests/parser_tests/test_r2_regression.py`: parser contracts, page-106 end-to-end coverage, and context ownership coverage.
- Modify `D:/ind-session/ENGINEERING_DECISIONS.md`: record the cross-grid provenance and fact-coverage decisions.
- Create `docs/superpowers/plans/2026-07-22-stable-table-row-lineage-and-context-ownership.md`: this plan.

The authoritative worktree already contains the active uncommitted parser program described by the session state. Execute inline there and preserve every unrelated modification. Do not create or switch worktrees during this plan.

### Task 1: Establish Display-Row Identity

**Files:**
- Modify: `parsers/pdf/table_modules/postprocess.py:1057-1095`
- Modify: `parsers/pdf/table_modules/postprocess.py:3000-3060`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Write the failing test**

Import the table module and add this focused class before `R2RegressionTests`:

```python
from parsers.pdf.table_modules import postprocess as table_postprocess


class DisplayRowProvenanceTests(unittest.TestCase):
    def test_merged_row_records_its_display_grid_source_reference(self) -> None:
        table = {
            "table_id": "tbl_lineage",
            "display_grid": [
                ["Dose", "0", "25"],
                ["Sex", "M", "F"],
                ["Section:", "", ""],
                ["AUC", "10", "12"],
            ],
            "merged_rows": [
                {"row": 3, "kind": "table_note_title", "text": "Section:", "colspan": 3}
            ],
        }

        attach = getattr(table_postprocess, "_attach_display_row_provenance", None)
        self.assertIsNotNone(attach)
        attach(table)

        self.assertEqual(
            [item["row_ref"] for item in table["display_row_provenance"]],
            [
                "tbl_lineage:display_row:1",
                "tbl_lineage:display_row:2",
                "tbl_lineage:display_row:3",
                "tbl_lineage:display_row:4",
            ],
        )
        self.assertEqual(table["merged_rows"][0]["source_grid"], "display_grid")
        self.assertEqual(
            table["merged_rows"][0]["source_row_ref"],
            "tbl_lineage:display_row:3",
        )
```

- [ ] **Step 2: Run the test and verify RED**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.parser_tests.test_r2_regression.DisplayRowProvenanceTests.test_merged_row_records_its_display_grid_source_reference
```

Expected: FAIL because `_attach_display_row_provenance` is absent.

- [ ] **Step 3: Implement stable display-row references**

Add next to `_build_merged_row_metadata`:

```python
def _display_row_ref(table_id: str, row_number: int) -> str:
    return f"{table_id}:display_row:{row_number}"


def _attach_display_row_provenance(table: dict[str, Any]) -> None:
    rows = [row for row in table.get("display_grid", []) or [] if isinstance(row, list)]
    table_id = str(table.get("table_id") or table.get("block_id") or "table").strip()
    provenance = [
        {
            "row_ref": _display_row_ref(table_id, row_number),
            "source_grid": "display_grid",
            "source_row_number": row_number,
            "signature": _row_signature(row),
        }
        for row_number, row in enumerate(rows, start=1)
    ]
    table["display_row_provenance"] = provenance
    by_number = {item["source_row_number"]: item for item in provenance}
    for merged_row in table.get("merged_rows", []) or []:
        if not isinstance(merged_row, dict):
            continue
        try:
            row_number = int(merged_row.get("row", 0) or 0)
        except (TypeError, ValueError):
            continue
        source = by_number.get(row_number)
        if source is not None:
            merged_row["source_grid"] = "display_grid"
            merged_row["source_row_ref"] = source["row_ref"]
```

Call it after `_refresh_row_texts_from_grid` rebuilds `merged_rows`, and after any surviving helper in this module replaces `display_grid`. Rebuilding must replace stale aligned provenance.

- [ ] **Step 4: Verify GREEN and safety**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.parser_tests.test_r2_regression.DisplayRowProvenanceTests tests.parser_tests.test_semantic_row_filtering tests.parser_tests.test_nested_structure_diagnostics
```

Expected: PASS with unchanged grid values.

- [ ] **Step 5: Record the focused checkpoint without committing**

```powershell
git diff -- parsers/pdf/table_modules/postprocess.py tests/parser_tests/test_r2_regression.py
git diff --check -- parsers/pdf/table_modules/postprocess.py tests/parser_tests/test_r2_regression.py
```

These files already contain active uncommitted work. Do not stage or commit them; preserve unrelated hunks and record the passing command as the checkpoint.

### Task 2: Propagate Provenance Into Semantic Rows

**Files:**
- Modify: `parsers/pdf/postprocess.py:36541-36635`
- Modify: `parsers/pdf/postprocess.py:37842-37875`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Write failing semantic-lineage tests**

Add to `DisplayRowProvenanceTests`:

```python
    def test_dose_response_body_rows_keep_distinct_display_sources(self) -> None:
        table = {
            "table_id": "tbl_lineage",
            "display_grid": [
                ["Dose", "0", "", "25", ""],
                ["Sex", "M", "F", "M", "F"],
                ["Section:", "", "", "", ""],
                ["Day 28", "AUC", "10", "12", "14"],
            ],
            "merged_rows": [
                {"row": 3, "kind": "table_note_title", "text": "Section:", "colspan": 5}
            ],
        }
        table_postprocess._attach_display_row_provenance(table)
        semantic_grid = [
            ["Dose", "0 M", "0 F", "25 M", "25 F"],
            ["Section:", "", "", "", ""],
            ["Day 28 AUC", "10", "12", "14", ""],
        ]

        postprocess._apply_dose_response_result_panel_projection(
            table,
            semantic_grid=semantic_grid,
            continuation_schema_inherited=False,
            panel_metadata={"source_has_explicit_sex_header_row": True},
        )

        lineage = table["semantic_row_provenance"]
        self.assertEqual(len(lineage), len(semantic_grid))
        self.assertEqual(
            lineage[0]["source_row_refs"],
            ["tbl_lineage:display_row:1", "tbl_lineage:display_row:2"],
        )
        self.assertEqual(lineage[1]["source_row_refs"], ["tbl_lineage:display_row:3"])
        self.assertEqual(lineage[2]["source_row_refs"], ["tbl_lineage:display_row:4"])

    def test_duplicate_semantic_signatures_consume_display_rows_in_order(self) -> None:
        table = {
            "table_id": "tbl_duplicates",
            "display_grid": [["Header"], ["Same"], ["Same"]],
        }
        table_postprocess._attach_display_row_provenance(table)

        lineage = postprocess._semantic_row_provenance_from_display_grid(
            table,
            [["Header"], ["Same"], ["Same"]],
            source="fixture_projection",
        )

        self.assertEqual(lineage[1]["source_row_refs"], ["tbl_duplicates:display_row:2"])
        self.assertEqual(lineage[2]["source_row_refs"], ["tbl_duplicates:display_row:3"])
```

- [ ] **Step 2: Run both tests and verify RED**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.parser_tests.test_r2_regression.DisplayRowProvenanceTests.test_dose_response_body_rows_keep_distinct_display_sources tests.parser_tests.test_r2_regression.DisplayRowProvenanceTests.test_duplicate_semantic_signatures_consume_display_rows_in_order
```

Expected: FAIL because semantic provenance is absent.

- [ ] **Step 3: Implement semantic-row alignment**

Import `_attach_display_row_provenance` into `parsers/pdf/postprocess.py`. Add near `_apply_semantic_grid_projection_common`:

```python
def _semantic_row_provenance_from_display_grid(
    table: dict[str, Any],
    semantic_grid: list[list[str]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    _attach_display_row_provenance(table)
    display_rows = [row for row in table.get("display_grid", []) or [] if isinstance(row, list)]
    display_lineage = list(table.get("display_row_provenance", []) or [])
    refs_by_signature: dict[str, list[str]] = {}
    for row, lineage in zip(display_rows, display_lineage):
        signature = _compact_text(_grid_row_text(row))
        row_ref = str(lineage.get("row_ref") or "") if isinstance(lineage, dict) else ""
        if signature and row_ref:
            refs_by_signature.setdefault(signature, []).append(row_ref)
    consumed: Counter[str] = Counter()
    result: list[dict[str, Any]] = []
    for row_index, row in enumerate(semantic_grid):
        signature = _compact_text(_grid_row_text(row))
        candidates = refs_by_signature.get(signature, [])
        position = consumed[signature]
        row_refs = [candidates[position]] if position < len(candidates) else []
        if row_refs:
            consumed[signature] += 1
        result.append(
            {
                "semantic_row": row_index,
                "source_row_refs": row_refs,
                "derivation": "display_row_signature" if row_refs else source,
            }
        )
    return result
```

Set aligned `semantic_row_provenance` in `_apply_semantic_grid_projection_common`. In `_apply_dose_response_result_panel_projection`, augment the folded header entry with validated dose and optional sex display header references. Add a projection audit containing semantic, mapped, and unmapped row counts.

- [ ] **Step 4: Add invariant validation**

Reject cross-grid resolution if provenance length differs from grid length or display `row_ref` values are duplicated. Unmapped semantic rows require a derivation and remain renderable; they can never authorize row consumption.

- [ ] **Step 5: Verify GREEN**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.parser_tests.test_r2_regression.DisplayRowProvenanceTests
```

Expected: PASS.

- [ ] **Step 6: Record the semantic-provenance checkpoint**

```powershell
git diff -- parsers/pdf/postprocess.py parsers/pdf/table_modules/postprocess.py tests/parser_tests/test_r2_regression.py
git diff --check -- parsers/pdf/postprocess.py parsers/pdf/table_modules/postprocess.py tests/parser_tests/test_r2_regression.py
```

Do not commit overlapping dirty files.

### Task 3: Make Markdown Consume Row Lineage

**Files:**
- Modify: `api/main.py:902-920`
- Modify: `api/main.py:2196-2247`
- Modify: `api/main.py:2880-3038`
- Modify: `api/main.py:3186-3335`
- Modify: `tests/deterministic_tests/test_parse_markdown_export.py:266-405`
- Modify: `tests/parser_tests/test_r2_regression.py:10111-10160`

- [ ] **Step 1: Write a deterministic failing Markdown test**

Add beside the dose-response header tests:

```python
    def test_source_backed_sex_header_keeps_structural_and_first_data_rows(self) -> None:
        api_main = importlib.import_module("api.main")
        table = {
            "block_type": "table",
            "block_id": "tbl_lineage",
            "table_id": "tbl_lineage",
            "semantic_role": "business_table",
            "page": 1,
            "display_grid": [
                ["Dose", "0", "", "25", ""],
                ["Sex", "M", "F", "M", "F"],
                ["Toxicokinetics:", "", "", "", ""],
                ["Day 28", "AUC", "10", "12", "14"],
                ["Day 180 Css", "0.4", "0.5", "1.7", "0.3"],
            ],
            "semantic_grid": [
                ["Dose", "0 M", "0 F", "25 M", "25 F"],
                ["Toxicokinetics:", "", "", "", ""],
                ["Day 28 AUC", "10", "12", "14", ""],
                ["Day 180 Css", "0.4", "0.5", "1.7", "0.3"],
            ],
            "semantic_row_provenance": [
                {"source_row_refs": ["tbl_lineage:display_row:1", "tbl_lineage:display_row:2"]},
                {"source_row_refs": ["tbl_lineage:display_row:3"]},
                {"source_row_refs": ["tbl_lineage:display_row:4"]},
                {"source_row_refs": ["tbl_lineage:display_row:5"]},
            ],
            "merged_rows": [
                {
                    "row": 3,
                    "source_grid": "display_grid",
                    "source_row_ref": "tbl_lineage:display_row:3",
                    "kind": "table_note_title",
                    "text": "Toxicokinetics:",
                    "colspan": 5,
                }
            ],
            "semantic_projection_v2": {
                "dose_response_result_panel_projection": {
                    "semantic_profile": "dose_response_result_panel",
                    "has_sex_leaf_columns": True,
                    "source_has_explicit_sex_header_row": True,
                }
            },
        }
        document = {
            "filename": "row-lineage.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertEqual(markdown.count("Toxicokinetics:"), 1)
        self.assertIn("| Day 28 AUC | 10 | 12 | 14 |  |", markdown)
        self.assertIn("| Day 180 Css | 0.4 | 0.5 | 1.7 | 0.3 |", markdown)

    def test_ambiguous_legacy_structural_signature_preserves_all_rows(self) -> None:
        api_main = importlib.import_module("api.main")
        table = {
            "block_type": "table",
            "block_id": "tbl_legacy",
            "table_id": "tbl_legacy",
            "semantic_role": "business_table",
            "page": 1,
            "semantic_grid": [
                ["Label", "Value"],
                ["Repeated:", ""],
                ["Repeated:", ""],
                ["Result", "10"],
            ],
            "merged_rows": [
                {"row": 2, "kind": "table_note_title", "text": "Repeated:", "colspan": 2}
            ],
        }
        document = {
            "filename": "legacy-ambiguous.pdf",
            "source_type": "pdf",
            "metadata": {"page_count": 1, "parser_hint": "pdf"},
            "document_ast": {"pages": [{"page": 1, "blocks": [dict(table)]}]},
            "table_asts": [dict(table)],
        }

        markdown = api_main._build_full_markdown([document], markdown_profile="ind-review")

        self.assertEqual(markdown.count("| Repeated: |  |"), 2)
        self.assertIn("| Result | 10 |", markdown)
```

- [ ] **Step 2: Strengthen page-106 regression before production edits**

Extend `test_page106_carcinogenicity_dose_response_renders_sex_as_multilevel_header`:

```python
        auc_row = "| 第28 天AUC(µg-h/mla) | - | - | 10 | 12 | 40 | 48 | 815 | 570 |"
        self.assertEqual(region.count(auc_row), 1, msg=region)
        self.assertEqual(region.count("**毒代动力学：**"), 1, msg=region)
        self.assertNotIn("| 毒代动力学： |", region, msg=region)
```

Also assert aligned display and semantic provenance, and distinct structural/AUC source references.

- [ ] **Step 3: Run both tests and verify RED**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.deterministic_tests.test_parse_markdown_export.ParseMarkdownExportTests.test_source_backed_sex_header_keeps_structural_and_first_data_rows tests.deterministic_tests.test_parse_markdown_export.ParseMarkdownExportTests.test_ambiguous_legacy_structural_signature_preserves_all_rows tests.parser_tests.test_r2_regression.R2RegressionTests.test_page106_carcinogenicity_dose_response_renders_sex_as_multilevel_header
```

Expected: both FAIL because the current adapter maps the structural row to the first data row.

- [ ] **Step 4: Add aligned presentation provenance**

Keep `_project_markdown_visible_semantic_grid` as a compatibility wrapper. Add an internal presentation surface with `rows` plus aligned `row_provenance`. Map presentation rows to semantic rows by normalized signature in source order, then copy semantic `source_row_refs`. Synthesized headers get empty refs and a derivation. Validate equal list lengths.

- [ ] **Step 5: Resolve structural rows by reference**

Change `_merged_rows_by_projected_grid_row` to index presentation rows by `source_row_ref`. Resolve each structural reference to exactly one row. For legacy metadata without a reference, match the merged-row text only when one presentation row has that normalized signature. Unresolved or ambiguous metadata records a diagnostic and consumes no row. Never compare a display numeric row directly with a semantic or presentation numeric row.

Keep the existing `test_dose_response_header_projection_preserves_merged_row_source_alignment` as the positive unique-signature legacy fallback. The new ambiguous fixture is the negative fallback contract.

- [ ] **Step 6: Migrate spans and continuation composition**

Update `_markdown_projected_presentation_spans` to map semantic span rows through aligned lineage. Merge provenance arrays in `_merge_continued_table_chain` using the same skipped-header and boundary-title offsets as rows. Boundary-title rows have no source refs and an explicit derivation.

- [ ] **Step 7: Verify GREEN and cross-type safety**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.deterministic_tests.test_parse_markdown_export
& ".venv\Scripts\python.exe" -m unittest tests.parser_tests.test_r2_regression.SourceBackedBodySpanProjectionTests tests.parser_tests.test_r2_regression.R2RegressionTests.test_page106_carcinogenicity_dose_response_renders_sex_as_multilevel_header tests.parser_tests.test_r2_regression.R2RegressionTests.test_pages106_to_116_dose_response_panels_project_logical_schema_and_inherit_continuations
```

Expected: PASS; page 106 contains one AUC row and one structural label.

- [ ] **Step 8: Record the Markdown-lineage checkpoint**

```powershell
git diff -- api/main.py parsers/pdf/postprocess.py parsers/pdf/table_modules/postprocess.py tests/deterministic_tests/test_parse_markdown_export.py tests/parser_tests/test_r2_regression.py
git diff --check -- api/main.py parsers/pdf/postprocess.py parsers/pdf/table_modules/postprocess.py tests/deterministic_tests/test_parse_markdown_export.py tests/parser_tests/test_r2_regression.py
```

Do not commit overlapping dirty files.

### Task 4: Transfer Absorbed Context by Fact Coverage

**Files:**
- Modify: `parsers/pdf/postprocess.py:10728-10740`
- Modify: `parsers/pdf/postprocess.py:34690-34830`
- Test: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Write failing generic context tests**

Add `StudyContextFactCoverageTests`:

```python
class StudyContextFactCoverageTests(unittest.TestCase):
    def _transfer(self, table: dict, row: str) -> None:
        postprocess._transfer_absorbed_template_study_context_to_business_table(
            template={
                "structure_template_id": "template_context",
                "page": 1,
                "bbox": [0, 0, 100, 100],
            },
            table=table,
            original_rows=["Study title", row],
            source_template_id="template_context",
            require_title_anchor=True,
        )

    def test_fully_covered_combined_row_is_not_transferred(self) -> None:
        table = {
            "table_id": "tbl_context",
            "page": 1,
            "title": "Study title",
            "study_context_blocks": [
                {"text": "Start: 2020-01-01 Vehicle: feed", "source": "panel"},
                {"text": "Control: plain-feed GLP: yes", "source": "panel"},
            ],
        }
        self._transfer(
            table,
            "Start: 2020-01-01 Vehicle: feed Control: plain-feed GLP: yes",
        )

        self.assertEqual(len(table["study_context_blocks"]), 2)
        audit = table["study_context_transfer_audits"][-1]
        self.assertEqual(audit["transferred_fact_count"], 0)
        self.assertEqual(audit["already_covered_fact_count"], 4)

    def test_partially_covered_row_transfers_only_missing_fact(self) -> None:
        table = {
            "table_id": "tbl_context",
            "page": 1,
            "title": "Study title",
            "study_context_blocks": [
                {"text": "Start: 2020-01-01", "source": "panel"},
            ],
        }
        self._transfer(table, "Start: 2020-01-01 Vehicle: feed")

        visible = "\n".join(item["text"] for item in table["study_context_blocks"])
        self.assertEqual(visible.count("Start:"), 1)
        self.assertEqual(visible.count("Vehicle: feed"), 1)
        audit = table["study_context_transfer_audits"][-1]
        self.assertEqual(audit["transferred_fact_count"], 1)
        self.assertEqual(audit["already_covered_fact_count"], 1)

    def test_same_label_with_different_value_remains_distinct(self) -> None:
        table = {
            "table_id": "tbl_context",
            "page": 1,
            "title": "Study title",
            "study_context_blocks": [
                {"text": "Dose: 10", "source": "panel"},
            ],
        }
        self._transfer(table, "Dose: 20")

        visible = "\n".join(item["text"] for item in table["study_context_blocks"])
        self.assertEqual(visible.count("Dose:"), 2)
        self.assertIn("Dose: 10", visible)
        self.assertIn("Dose: 20", visible)
        audit = table["study_context_transfer_audits"][-1]
        self.assertEqual(audit["transferred_fact_count"], 1)
        self.assertEqual(audit["already_covered_fact_count"], 0)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.parser_tests.test_r2_regression.StudyContextFactCoverageTests
```

Expected: FAIL because whole-row deduplication appends combined rows and no audit exists.

- [ ] **Step 3: Implement typed context facts**

Add near `_study_metadata_key_value`:

```python
def _study_context_fact_key(label: str, value: str) -> str:
    return f"{_compact_text(label)}={_compact_text(value)}"


def _study_context_label_spans(text: str) -> list[tuple[int, int, str]]:
    clean = _clean_text(text)
    colon_positions = [match.start() for match in re.finditer(r"[:：]", clean)]
    spans: list[tuple[int, int, str]] = []
    lower_bound = 0
    for colon in colon_positions:
        starts = [lower_bound]
        starts.extend(
            lower_bound + match.end()
            for match in re.finditer(r"\s+", clean[lower_bound:colon])
        )
        candidates: list[tuple[int, int, str]] = []
        for start in sorted(set(starts)):
            label = _clean_text(clean[start:colon])
            if not label or len(label) > 42 or re.search(r"\d", label):
                continue
            score = max(
                int(_tabular_form_template_field_label_score(label) or 0),
                int(_sparse_tabular_form_field_score(label) or 0),
            )
            if _study_summary_template_field_text(label):
                score += 4
            if re.fullmatch(r"[A-Za-z][A-Za-z /_-]{0,31}", label):
                score += 1
            if score > 0:
                candidates.append((score, start, label))
        if not candidates:
            continue
        _, start, label = max(candidates, key=lambda item: (item[0], item[1]))
        spans.append((start, colon, label))
        lower_bound = colon + 1
    return spans


def _study_context_fact_records(
    text: str,
    *,
    source_ref: str,
    source_owner: str,
    page: int,
) -> list[dict[str, Any]]:
    clean = _clean_text(text)
    spans = _study_context_label_spans(clean)
    records: list[dict[str, Any]] = []
    for index, (start, colon, label) in enumerate(spans):
        value_end = spans[index + 1][0] if index + 1 < len(spans) else len(clean)
        value = _clean_text(clean[colon + 1 : value_end])
        if not value:
            continue
        records.append(
            {
                "label": label,
                "value": value,
                "fact_key": _study_context_fact_key(label, value),
                "source_ref": source_ref,
                "source_owner": source_owner,
                "page": page,
                "display_text": _clean_text(clean[start:value_end]),
            }
        )
    return records


def _ensure_study_context_fact_records(table: dict[str, Any]) -> None:
    table_id = str(table.get("table_id") or "table").strip()
    page = int(table.get("page", 0) or 0)
    for index, block in enumerate(table.get("study_context_blocks", []) or [], start=1):
        if not isinstance(block, dict):
            continue
        source_ref = str(block.get("source_ref") or f"{table_id}:study_context:{index}")
        block["source_ref"] = source_ref
        block["study_context_facts"] = _study_context_fact_records(
            str(block.get("text") or ""),
            source_ref=source_ref,
            source_owner=table_id,
            page=page,
        )
```

If a colon row yields no facts, record it as ambiguous in the transfer audit and do not create a visible duplicate. The label-span helper deliberately prefers established field-label scores; its generic ASCII fallback exists for deterministic fixtures and ordinary English metadata, not payload-specific labels.

- [ ] **Step 4: Replace whole-row transfer with fact coverage**

Extract candidate facts, compare them with existing owner fact keys, and transfer only missing facts grouped by source candidate row. Render partial transfer from the missing facts original fragments. Preserve ambiguous evidence without visible duplication. Append `study_context_transfer_audits` with candidate, transferred, covered, ambiguous, and visible block refs. Report actual transferred count in `semantic_repairs`.

- [ ] **Step 5: Verify generic GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 6: Add page-106 context assertions and run them**

Add to the focused page-106 regression:

```python
        context_text = "\n".join(
            block["text"]
            for block in page106_carcinogenicity.get("study_context_blocks", [])
            if isinstance(block, dict)
        )
        self.assertEqual(context_text.count("首次给药日期：1995 年9 月20 日"), 1)
        self.assertEqual(context_text.count("高剂量选择依据：根据毒性终点"), 1)
        audits = page106_carcinogenicity.get("study_context_transfer_audits", [])
        self.assertTrue(audits)
        self.assertGreater(audits[-1]["already_covered_fact_count"], 0)
```

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.parser_tests.test_r2_regression.R2RegressionTests.test_page106_carcinogenicity_dose_response_renders_sex_as_multilevel_header
```

Expected: PASS.

- [ ] **Step 7: Record the context-ownership checkpoint**

```powershell
git diff -- parsers/pdf/postprocess.py tests/parser_tests/test_r2_regression.py
git diff --check -- parsers/pdf/postprocess.py tests/parser_tests/test_r2_regression.py
```

Do not commit overlapping dirty files.

### Task 5: Verify Conservation, Record Decision, and Synchronize

**Files:**
- Modify: `D:/ind-session/ENGINEERING_DECISIONS.md`
- Verify: files changed in Tasks 1-4
- Synchronize: matching files under `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`

- [ ] **Step 1: Run syntax and whitespace checks**

```powershell
& ".venv\Scripts\python.exe" -m py_compile api/main.py parsers/pdf/postprocess.py parsers/pdf/table_modules/postprocess.py tests/deterministic_tests/test_parse_markdown_export.py tests/parser_tests/test_r2_regression.py
git diff --check
```

Expected: exit 0 with no whitespace errors.

- [ ] **Step 2: Run focused verification**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.deterministic_tests.test_parse_markdown_export tests.parser_tests.test_r2_regression.DisplayRowProvenanceTests tests.parser_tests.test_r2_regression.StudyContextFactCoverageTests tests.parser_tests.test_r2_regression.R2RegressionTests.test_page106_carcinogenicity_dose_response_renders_sex_as_multilevel_header tests.parser_tests.test_r2_regression.R2RegressionTests.test_pages106_to_116_dose_response_panels_project_logical_schema_and_inherit_continuations
```

Expected: PASS with zero failures.

- [ ] **Step 3: Run cross-type regression groups**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.parser_tests.test_r2_regression.SourceBackedBodySpanProjectionTests tests.parser_tests.test_r2_regression.R2RegressionTests.test_pages107_to_113_table_owned_statistical_notes_do_not_leak_to_body tests.parser_tests.test_ectd_validation_standard_regression
```

Expected: PASS.

- [ ] **Step 4: Run full relevant suites**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.deterministic_tests.test_parse_markdown_export tests.parser_tests.test_r2_regression
```

Expected: PASS. Record count, duration, skips, and non-failing parser warnings.

- [ ] **Step 5: Inspect generated page-106 Markdown**

Generate IND-review Markdown and inspect `2.6.7.10` through `2.6.7.11`. Confirm structural title count 1, AUC row count 1, first-dose-date fact count 1, and high-dose-rationale fact count 1.

- [ ] **Step 6: Record engineering decision**

Append a dated entry to `D:/ind-session/ENGINEERING_DECISIONS.md`: `merged_rows.row` is a display-grid compatibility coordinate; `source_row_ref` is cross-grid identity; ambiguous lineage preserves rows; context transfers by typed fact coverage; signatures are unique-only fallback; literal page/title/value/coordinate rules are prohibited.

- [ ] **Step 7: Synchronize touched files to mirror**

Copy the final authoritative versions of `api/main.py`, both parser postprocess files, both test files, and the 2026-07-22 spec and plan to matching relative paths under `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`. Compare SHA-256 hashes for every copied file.

- [ ] **Step 8: Run mirror verification**

```powershell
& ".venv\Scripts\python.exe" -m unittest tests.deterministic_tests.test_parse_markdown_export tests.parser_tests.test_r2_regression.DisplayRowProvenanceTests tests.parser_tests.test_r2_regression.StudyContextFactCoverageTests tests.parser_tests.test_r2_regression.R2RegressionTests.test_page106_carcinogenicity_dose_response_renders_sex_as_multilevel_header
```

Expected: PASS from the mirror tree.

- [ ] **Step 9: Final scope audit**

```powershell
git status --short
git diff --stat
git diff --check
```

Confirm no unrelated user change was reverted, no generated artifact was committed, and every design success criterion has a passing assertion.
