# Structure Template Note Markdown Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve literal leading legend markers in structure-template notes without allowing CommonMark to reinterpret those notes as lists, headings, or quotations.

**Architecture:** Keep parser ownership and AST source text unchanged. Add one renderer-boundary formatter in `api/main.py` that first applies existing inline escaping and then neutralizes only a leading CommonMark block marker; route every normal and deferred structure-template note rendering branch through it.

**Tech Stack:** Python, `unittest`, existing PDF AST and IND-review Markdown renderer.

---

### Task 1: Lock The Rendering Contract

**Files:**
- Modify: `tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Add formatter contract tests**

Import `_markdown_structure_template_note_text` and assert that ordinary note text remains unchanged while `- `, `+ `, `* `, ordered-list, quote, and heading prefixes are escaped at the beginning of a note.

- [ ] **Step 2: Strengthen the real-document regression**

Assert that page 56 source block `txt_p56_017` remains a `structure_template_note` with its source bbox and literal AST text, while IND-review Markdown contains `\\- 无显著异常` and does not contain an unescaped `\n- 无显著异常`. Assert the same Markdown safety for the matching page 55 and 67 legends.

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.parser_tests.test_r2_regression.StructureTemplateNoteMarkdownSafetyTests -v
```

Expected: import or assertion failure because the dedicated formatter does not exist yet.

### Task 2: Add The Central Note-Safe Formatter

**Files:**
- Modify: `api/main.py`

- [ ] **Step 1: Implement the minimal formatter**

Add `_markdown_structure_template_note_text(value)` beside `_markdown_escape_inline_text`. It must apply existing inline escaping and then escape a leading CommonMark block marker only. It must not mutate the AST or add a visible label.

- [ ] **Step 2: Route all structure-template note exits through it**

Replace direct `_markdown_escape_inline_text(note.get("text") or "")` calls in the structure-template renderer and deferred-note renderer. Do not change entry rendering or the global inline formatter.

- [ ] **Step 3: Run focused tests and verify GREEN**

Run the formatter contract test and the real r2 legend regression. Expected: all selected tests pass.

### Task 3: Verify The Affected Surface And Mirror

**Files:**
- Verify: `api/main.py`
- Verify: `tests/parser_tests/test_r2_regression.py`
- Mirror: `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai/api/main.py`
- Mirror: `D:/d/funding/nation/new code/AutoIND-Pro/ind-compliance-ai/tests/parser_tests/test_r2_regression.py`

- [ ] **Step 1: Run impact-based regression**

Run the formatter contract, page 54-56 template regressions, page 64-68 legend/order regressions, and Markdown rendering tests that cover structure-template notes. Do not run unrelated parser suites because parser ownership and geometry are unchanged.

- [ ] **Step 2: Compile and inspect the diff**

Run `py_compile`, `git diff --check`, and inspect only the scoped diff. Confirm true form-entry list rendering is unchanged.

- [ ] **Step 3: Synchronize and verify hashes**

Apply the same scoped changes to the mirror, run the same focused tests there, and compare SHA-256 hashes for the two modified code/test files.
