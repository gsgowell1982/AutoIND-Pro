# Region Ownership Graph Design

Date: 2026-05-25

Status: draft for review

## Purpose

AutoIND-Pro's PDF parser must prioritize China IND application materials over benchmark presentation formats. The parser should first understand what regions exist on each page, then classify and structure content inside those regions. Table, figure, TOC, header/footer, note, and body-text changes must not fight each other through local suppression rules.

This design introduces an additive Region Ownership Graph layer. Existing extraction modules can continue to produce text blocks, image blocks, raw table candidates, TOC nodes, figure candidates, and table ASTs, but they should submit typed region candidates into one ownership arbitration layer before public AST and Markdown projection.

The immediate goal is region correctness and stable ownership. Figure content analysis is explicitly deferred: figures should be correctly bounded and linked to title/legend evidence now, while chart/XML/image semantic interpretation remains an interface for later work.

## Non-Goals

- Do not rewrite the whole PDF pipeline.
- Do not optimize production Markdown for OpenDataLoader benchmark formatting.
- Do not perform deep figure/chart/XML parsing in this phase.
- Do not replace protected A-tst, 2-column-tst, r2, eCTD, or test-ind behavior with benchmark-specific assumptions.
- Do not add production branches by PDF filename, page number, benchmark sample id, or exact text.

## Core Architecture

The parser should converge on five layers:

1. Evidence Layer
   - Records source facts without destructive decisions.
   - Sources include text spans, blocks, drawings, lines, images, OCR blocks, page geometry, fonts, repeated header/footer signatures, link/outline metadata, and table/figure candidates.
   - Every downstream object should keep provenance back to evidence ids or source bboxes.

2. Region Ownership Layer
   - Converts evidence into typed page regions.
   - Resolves conflicts through one candidate graph instead of local table/text/figure suppression.
   - Produces auditable ownership decisions: what evidence belongs to which region, what competed, why one owner won, and what confidence was assigned.

3. Type Structure Layer
   - Parses structure inside accepted regions.
   - Tables produce visual/logical/semantic grids.
   - Figures produce bounded figure nodes with title/legend ownership and a deferred content-analysis interface.
   - Body text produces paragraphs, headings, section markers, lists, formulas, algorithms, references, and outline/template structures.

4. Semantic Layer
   - Adds IND-facing meaning such as table subject, row/column header semantics, marker-note links, section/TOC alignment, continuation links, and figure/legend relation.
   - Keeps visual facts and semantic interpretation separate.

5. Projection Layer
   - Produces IND Markdown, frontend views, debug AST, downstream rule input, and optional benchmark adapter output from the same AST.
   - Benchmark-specific HTML/Markdown projection is an external adapter and must not mutate production AST or IND presentation.

## Region Types

The first implementation should define stable region roles even if not every role is fully classified on day one:

- `body_text`
- `section_heading`
- `subsection_heading`
- `list_item`
- `toc`
- `toc_continuation`
- `outline_template`
- `table`
- `table_title`
- `table_note`
- `table_footnote`
- `figure`
- `figure_title`
- `figure_legend`
- `formula_display`
- `formula_inline`
- `algorithm_pseudocode`
- `page_header`
- `page_footer`
- `page_number`
- `footnote`
- `logo`
- `watermark`
- `stamp_or_seal`
- `sidebar`
- `unknown_visual`
- `noise_or_artifact`

These roles are document-structure roles, not presentation classes. A region can later be projected differently for IND Markdown, debug output, or benchmark evaluation.

## Data Contract

The additive contract should be lightweight enough to add to `parsers/pdf/types.py` without disrupting existing AST consumers.

### RegionCandidate

Represents a possible region proposed by a module.

Required fields:

- `candidate_id`
- `page`
- `region_type`
- `bbox`
- `source`
- `evidence_refs`
- `text`
- `confidence`
- `signals`
- `metadata`

Candidate sources may include `text_blocks`, `table_raw_candidate`, `figure_candidate`, `toc_detector`, `header_footer_detector`, `formula_detector`, `algorithm_detector`, and `outline_template_detector`.

### OwnershipDecision

Represents an arbitration result for one accepted region.

Required fields:

- `region_id`
- `accepted_candidate_id`
- `region_type`
- `page`
- `bbox`
- `owned_evidence_refs`
- `owned_text_block_ids`
- `competing_candidate_ids`
- `decision_factors`
- `confidence`
- `warnings`

This is the audit boundary. Downstream suppression or projection should read ownership decisions, not independently delete text blocks.

### RegionNode

Represents the accepted page-region AST node.

Required fields:

- `region_id`
- `region_type`
- `page`
- `bbox`
- `text`
- `children`
- `links`
- `provenance`
- `structure_ref`
- `semantic_ref`

Examples:

- a `table` node links to `table_title`, `table_note`, and `table_footnote` children.
- a `figure` node links to `figure_title` and `figure_legend` children.
- a `toc_continuation` node links back to the parent `toc` sequence.
- a `body_text` node can link to formula-inline children without losing paragraph text.

## Ownership Arbitration Rules

The first graph should target high-impact cross-category conflicts:

- table vs body text
- table title/note/footnote vs body text
- figure title/legend vs body text
- figure/image/logo/watermark vs body text
- TOC/outline template vs body text
- page header/footer/page number vs body text
- algorithm pseudocode vs body/list/table
- formula display vs body/table

The arbitration should combine evidence rather than use one brittle threshold:

- geometry: bbox overlap, adjacency, vertical gaps, horizontal alignment, column/lane position
- source strength: structured table, visual rule evidence, OCR evidence, image evidence, repeated margin signature
- typographic signals: font size, weight, baseline, indentation, line spacing
- lexical/semantic cues: Table/Fig/Note/Source/References/目录/图/表/注/算法 and equivalents
- continuity: cross-page table, continued TOC, repeated headers/footers, section context
- role compatibility: table title can belong to table, but body paragraph should not be swallowed by visual grid unless table evidence owns it
- conflict history: rejected candidates and warnings must remain inspectable

The rule is not "table wins over text" or "caption wins by nearest object". It is "the best-supported region owner wins, and all losing candidates are retained as audit evidence".

## Table Design

Tables should be structured after table-region ownership is accepted.

Each table should preserve:

- `region_bbox`
- `title_region_id`
- `note_region_ids`
- `footnote_region_ids`
- `raw_evidence`
- `visual_grid`
- `logical_grid`
- `semantic_grid`
- `cell_provenance`
- `header_graph`
- `row_header_graph`
- `column_header_graph`
- `note_links`
- `continuation_links`
- `warnings`

Visual and semantic facts must be separate. If a cell is visually empty but semantically inherited from a row group above, the AST should record both facts:

- `visual_text=""`
- `semantic_text="..."` when justified
- `visual_empty=true`
- `inherited_from`
- `row_headers`
- `column_headers`
- `provenance`

The production IND presentation can choose how much semantic filling to display. Benchmark projection can use a separate adapter to output semantic HTML if it improves TEDS, but this must not become the source table AST.

## Figure Design

Figures should be region-correct first.

The accepted figure node should preserve:

- figure bbox
- title ownership
- legend ownership
- image payload reference
- nearby source/reference notes
- image/text/OCR evidence refs
- coarse figure kind when available: `chart`, `xml_or_code_image`, `flow_diagram`, `photo`, `logo_like`, `unknown`
- deferred content-analysis status

Deep parsing of axes, curves, XML/code, and visual summaries is out of scope for the first ownership phase. The interface should exist so later work can attach chart-axis extraction, OCR text, embedded-code parsing, or image semantic descriptions without changing region ownership.

## Body, TOC, and Section Design

Body reconstruction should consume only evidence owned by body-like regions. Table/figure/title/note/header/footer ownership should not silently remove body evidence without an auditable ownership decision.

Heading and section logic should distinguish:

- true section headings with document hierarchy role
- numbered list items under a section
- TOC entries
- no-page outline/template structures
- algorithm pseudocode labels
- references and bibliography items

For IND materials, a numbered marker such as `4.1` or `2.6.2.1` is a heading only when corroborated by typography, placement, context, TOC/outline relation, or heading-like line shape. Numbered points under a section remain body/list regions and should not be promoted to headings by marker alone.

## Benchmark Isolation

OpenDataLoader benchmark output should live behind an adapter:

`AutoIND AST -> benchmark projection adapter -> benchmark Markdown/HTML`

Production parser and IND presentation must not depend on:

- benchmark sample ids
- benchmark ground truth
- benchmark Markdown style
- benchmark-specific HTML table requirements

Benchmark metrics remain useful diagnostics. A benchmark-oriented projection can optimize TEDS/MHS/NID formatting, but it must not mutate evidence, region ownership, table AST, or IND-facing Markdown.

## Regression Gates

Every implementation step should report:

- protected A-tst parser/Markdown behavior
- protected 2-column-tst behavior
- protected r2 behavior
- protected eCTD/test-ind behavior
- region ownership targeted tests
- table-region targeted tests
- figure-region targeted tests
- body-text non-regression tests
- OpenDataLoader AutoIND-goal metrics as diagnostics

Stage movement should not accept a table-quality gain that creates broad core-text or reading-order regressions. For table-focused changes, the gate should include:

- no increase in `framework_reading_order_or_ownership` gap count without explanation
- no major `core_text_mean` drop
- table region AST tests for title/note/body boundaries
- production IND Markdown unchanged where AST is unchanged

## Migration Plan

1. Add region candidate/decision/node dataclasses and serialization helpers.
2. Add a non-invasive candidate collection pass that wraps existing text blocks, table ASTs/candidates, figure blocks, TOC nodes, and header/footer signals.
3. Add an ownership arbitration pass in observe-only mode. It writes debug/AST metadata but does not change public Markdown yet.
4. Add targeted tests for known conflicts: table-note/body, figure-legend/body, header-footer/body, TOC/body, outline-template/body.
5. Switch table text suppression to consume ownership decisions for a narrow protected subset, with fallback to current behavior.
6. Add figure region nodes with title/legend links and deferred content-analysis interface.
7. Add table `visual_grid/logical_grid/semantic_grid` contract while preserving current public table fields.
8. Move benchmark-specific semantic HTML choices into the benchmark adapter or a clearly gated projection layer.
9. Expand gates so table improvements cannot regress body ownership or protected IND samples.

## Acceptance Criteria

- The parser exposes region ownership metadata in AST/debug output.
- Table, figure, TOC, header/footer, and body text ownership decisions are auditable.
- Figure regions can be correctly represented without deep image content parsing.
- Table AST separates region ownership, raw evidence, visual grid, logical grid, semantic grid, and presentation.
- Production IND Markdown remains IND-oriented.
- Benchmark projection is isolated from production AST and presentation.
- Protected parser regressions pass before any completion claim.
- New OpenDataLoader table improvements are not accepted if they cause unexplained broad core-text or reading-order regressions.
