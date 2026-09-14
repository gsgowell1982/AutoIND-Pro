# Source-Typed Study Context Facts Implementation Plan

> **Required skill:** Use `executing-plans` to implement this plan task by task, with `test-driven-development` for each behavior change and `verification-before-completion` before reporting success.

**Goal:** Prevent study-context facts from being duplicated, swallowed, or moved when a rendered row contains several visually separated label/value fields, including the page 114 `首次给药日期 / 剔除/未剔除的仔鼠 / GLP 依从性` row.

**Architecture:** Promote producer-owned `study_context_facts` to the canonical ownership unit. The dose-response producer emits facts from positioned word clusters; absorbed templates emit facts from their structured `fields`. Existing text parsing remains a compatibility fallback only when a producer supplied no valid facts. Transfer and coverage continue to compare normalized `(label, value)` fact keys and preserve the existing audit contract.

**Tech stack:** Python 3, existing PDF post-processing pipeline, `unittest`, project `r2.pdf` regression fixture.

**Execution constraints:** Work inline in the authoritative dirty worktree because repository governance requires it. Preserve all unrelated/user edits. Do not commit changes to overlapping code files. Synchronize only the completed scoped files to the configured mirror after verification.

---

### Task 1: Lock the canonical fact contract with failing tests

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py` (`StudyContextFactCoverageTests`)

**Steps:**
1. Add a unit test whose positioned row contains three large-gap field clusters and assert that the facts are exactly `首次给药日期`, `剔除/未剔除的仔鼠`, and `GLP 依从性`, each with its own value and source metadata.
2. Add a unit test proving `_ensure_study_context_fact_records` is fill-only when valid producer facts already exist.
3. Add a unit test proving structured absorbed-template `fields` supply a fact even when the generic text label vocabulary does not know the label.
4. Add/retain transfer tests for fully covered candidates and for equal labels with different values.
5. Run the new unit tests and confirm they fail for the missing behavior before production edits.

**Command:**
`D:\AutoIND-Pro\ind-compliance-ai\.venv\Scripts\python.exe -m unittest tests.parser_tests.test_r2_regression.StudyContextFactCoverageTests -v`

### Task 2: Emit producer-owned facts from positioned dose-response rows

**Files:**
- Modify: `parsers/pdf/postprocess.py` near `_study_context_label_spans`, `_ensure_study_context_fact_records`, and `_dose_response_result_panel_from_page_words`

**Steps:**
1. Add a canonical fact-record builder containing `label`, `value`, `fact_key`, `display_text`, `source_ref`, `source_owner`, `source_kind`, `page`, and `bbox`.
2. Add a geometry helper that sorts positioned word fragments, estimates ordinary within-field spacing, and splits only on materially larger horizontal gaps whose resulting clusters can be parsed as populated label/value fields.
3. Parse each accepted cluster independently at the first full-width or ASCII colon, without a fixed label allowlist.
4. Attach those facts to each dose-response `study_context_block` before its display text is joined. If geometry is insufficient, leave facts absent so compatibility parsing can handle the block.
5. Re-run the focused fact tests.

### Task 3: Make enrichment fill-only and transfer template-owned facts

**Files:**
- Modify: `parsers/pdf/postprocess.py` near `_ensure_study_context_fact_records`, `_absorbed_template_study_context_rows_for_table`, and absorbed-template transfer logic
- Modify: `tests/parser_tests/test_r2_regression.py`

**Steps:**
1. Validate existing fact records and preserve them unchanged when they already carry a usable key, label, and value.
2. Invoke text-based fact parsing only for blocks without valid producer facts and mark those records `source_kind=text_fallback`.
3. Associate selected absorbed-template rows with `template.fields` by normalized rendered text plus occurrence order, and construct candidate facts directly from nonblank structured label/value pairs.
4. Fall back to existing row text parsing only where no structured field matches.
5. Compare destination and candidate facts by exact normalized `(label, value)` keys, transferring only missing facts.
6. Preserve existing transfer-audit keys and add counts grouped by `source_kind`.
7. Run all fact-coverage unit tests and confirm green.

### Task 4: Add the real page 114 regression and protect neighboring fixes

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py` near the existing section 2.6.7.14 regressions

**Steps:**
1. Assert the page 114 study-context row remains visible with all three fields.
2. Assert `首次给药日期：1995 年10 月8 日` occurs exactly once in the relevant table output.
3. Assert no transferred date bullet appears after `F1 雌性：75 mg/kg`.
4. Run the page 114 test, page 106 partial-transfer/AUC tests, page 109 wrapped method/vehicle tests, and pages 103-105 ownership regressions.

### Task 5: Verify, document, and synchronize

**Files:**
- Modify: `D:/ind-session/ENGINEERING_DECISIONS.md`
- Copy after verification: scoped code/test/spec/plan files to `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai`

**Steps:**
1. Run Python syntax compilation and inspect the scoped diff for accidental page-, title-, value-, or fixed-coordinate rules.
2. Run the complete relevant regression suite and compare failures with the recorded five-test baseline.
3. Generate/inspect the page 114 Markdown evidence and verify the date count and F1 ordering manually.
4. Record the source-typed fact ownership decision, compatibility behavior, and regression evidence as the next engineering decision entry.
5. Copy only `parsers/pdf/postprocess.py`, `tests/parser_tests/test_r2_regression.py`, the approved design spec, and this plan to the mirror; compare SHA-256 hashes and run the focused mirror tests with the authoritative virtual environment.

