from __future__ import annotations

from pathlib import Path
import unittest

from parsers.pdf_parser import parse_pdf


BENCH_PDF_DIR = Path(
    r"C:\Users\gsgow\.config\superpowers\worktrees\AutoIND-Pro\opendataloader-bench-eval"
    r"\external_benchmarks\opendataloader-bench\autoind-full-output\pdfs-smoke"
)


def _bench_pdf(doc_id: str) -> Path:
    return BENCH_PDF_DIR / f"{doc_id}.pdf"


def _require_benchmark_pdf(doc_id: str) -> Path:
    path = _bench_pdf(doc_id)
    if not path.exists():
        raise unittest.SkipTest(f"OpenDataLoader benchmark fixture not found: {path}")
    return path


class OpenDataLoaderBenchmarkRegressionTests(unittest.TestCase):
    maxDiff = None

    def test_chart_numeric_figure_page_is_not_promoted_to_toc(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000078"))

        self.assertEqual(result.get("toc_blocks", []), [])
        self.assertEqual(result.get("toc_sequences", []), [])

    def test_chart_axis_numeric_visual_structure_is_not_projected_as_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000076"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        visual_structure_tables = [
            table
            for table in page_tables
            if str(table.get("detection_source") or table.get("detection_method") or "")
            == "visual_structure_grid"
        ]
        self.assertEqual(
            visual_structure_tables,
            [],
            msg="chart axis/tick numeric layout must be owned as chart/figure evidence, not projected as a data table",
        )

    def test_text_layer_chart_page_records_figure_region_semantics_without_table_projection(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000076"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(
            page_tables,
            [],
            msg="text-layer chart axis/tick rows must not be recovered through the table AST path",
        )

        figure_nodes = [
            node
            for node in (result.get("region_ownership") or {}).get("nodes", []) or []
            if node.get("region_type") == "figure"
        ]
        self.assertTrue(
            figure_nodes,
            msg="captioned text-layer charts must be represented as figure/chart region evidence",
        )

        figure_text = "\n".join(str(node.get("text") or "") for node in figure_nodes)
        self.assertIn("Figure 1.6", figure_text)
        self.assertIn("Source:", figure_text)

        semantics = [
            (node.get("metadata") or {}).get("figure_semantics") or {}
            for node in figure_nodes
        ]
        self.assertTrue(any(item.get("semantic_type") == "chart_figure" for item in semantics))
        self.assertTrue(any(item.get("content_kind") == "text_layer_chart" for item in semantics))

    def test_image_backed_chart_records_chart_semantics_without_promoting_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000038"))

        images = result.get("image_blocks", []) or []
        chart_images = [
            image
            for image in images
            if ((image.get("figure_semantics") or {}).get("semantic_type") == "chart_figure")
        ]
        self.assertTrue(
            chart_images,
            msg="image-backed charts must expose semantic chart evidence for axes/ticks/series/source text",
        )
        semantics = chart_images[0].get("figure_semantics") or {}
        self.assertEqual(semantics.get("content_kind"), "embedded_chart_text")
        self.assertTrue(semantics.get("axis_or_tick_text"))

    def test_chart_with_body_text_is_not_projected_as_table_from_any_candidate_source(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000039"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(
            page_tables,
            [],
            msg=(
                "chart axes, legends, and surrounding narrative body text must remain figure/body evidence; "
                "candidate ownership arbitration must not let any source project them as data tables"
            ),
        )

    def test_table_of_contents_page_is_owned_as_toc_not_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000044"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        page_tocs = [
            toc
            for toc in result.get("toc_blocks", []) or []
            if int(toc.get("page", 0) or 0) == 1
        ]
        page_blocks = [
            block
            for page in result.get("document_ast", {}).get("pages", []) or []
            if int(page.get("page", 0) or 0) == 1
            for block in page.get("blocks", []) or []
        ]

        self.assertEqual(
            page_tables,
            [],
            msg="a title-led table-of-contents region must not be emitted as business table evidence",
        )
        self.assertEqual(len(page_tocs), 1)
        toc_text = "\n".join(str(entry.get("text") or "") for entry in page_tocs[0].get("entries", []) or [])
        self.assertIn("Executive Summary", toc_text)
        self.assertIn("Recommendations", toc_text)
        self.assertTrue(any(block.get("block_type") == "toc" for block in page_blocks))

    def test_image_backed_chart_legend_and_body_text_are_not_projected_as_tables(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000038"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        body_text = _page_body_text(result, page_number=1)
        figure_nodes = [
            node
            for node in (result.get("region_ownership") or {}).get("nodes", []) or []
            if node.get("region_type") == "figure"
        ]

        self.assertEqual(
            page_tables,
            [],
            msg=(
                "chart legends and the following section/body lines must be owned by figure/body regions, "
                "not by a structured-text table candidate"
            ),
        )
        self.assertTrue(figure_nodes)
        self.assertIn("6.2. Expectations for Re-Hiring Employees", body_text)

    def test_image_ocr_bar_chart_profile_stays_chart_evidence_not_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000136"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        chart_images = [
            image
            for image in result.get("image_blocks", []) or []
            if ((image.get("figure_semantics") or {}).get("semantic_type") == "chart_figure")
        ]

        self.assertEqual(
            page_tables,
            [],
            msg=(
                "OCR rows from a bar chart may look like label/value pairs, but when the image is already "
                "owned as chart evidence they must not be promoted into a data-table AST"
            ),
        )
        self.assertTrue(chart_images)
        chart_text = "\n".join(str(image.get("content_text") or "") for image in chart_images)
        self.assertIn("usual demands", chart_text)
        self.assertIn("34%", chart_text)

    def test_multicolumn_body_and_chart_legend_are_not_projected_as_text_aligned_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000042"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(
            page_tables,
            [],
            msg=(
                "two-column prose around a chart legend may expose many word-left anchors, but without a local "
                "table header/caption boundary it must remain body/figure evidence instead of becoming a table"
            ),
        )

    def test_text_aligned_mitosis_meiosis_comparison_is_recovered_as_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000119"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        table_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for table in page_tables
            for row in (table.get("display_grid") or table.get("raw_grid") or [])
        )
        self.assertTrue(page_tables, msg="expected a table AST on the mitosis/meiosis comparison page")
        self.assertIn("Mitosis", table_text)
        self.assertIn("Meiosis", table_text)
        self.assertIn("# chromosomes", table_text)

    def test_visual_intro_plus_table_bridge_does_not_emit_duplicate_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000119"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(
            len(page_tables),
            1,
            msg=(
                "a captionless visual candidate that combines introductory prose with the top of a stronger "
                "structured table must release ownership instead of emitting a duplicate table"
            ),
        )

        table_text = _page_table_text(result, page_number=1)
        self.assertIn("Mitosis", table_text)
        self.assertIn("Meiosis", table_text)
        self.assertIn("# chromosomes", table_text)
        self.assertNotIn(
            "chromosome. Meiosis and | mitosis are both nuclear divisions",
            table_text,
        )

        body_text = _page_body_text(result, page_number=1)
        self.assertIn("Meiosis and mitosis are both nuclear divisions", body_text)

    def test_text_aligned_ai_pack_matrix_is_recovered_as_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000182"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        table_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for table in page_tables
            for row in (table.get("display_grid") or table.get("raw_grid") or [])
        )
        self.assertTrue(page_tables, msg="expected a table AST on the AI Pack comparison matrix page")
        self.assertIn("OCR", table_text)
        self.assertIn("Recommendation", table_text)
        self.assertIn("Product semantic search", table_text)

    def test_text_aligned_ai_pack_matrix_records_projected_row_header_semantics(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000182"))

        matrices = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
            and str(table.get("table_family") or "") in {"comparison_matrix", "projected_stub_matrix"}
        ]
        self.assertTrue(matrices, msg="matrix tables must preserve projected row-header/stub semantics")
        semantic_header = matrices[0].get("semantic_header") or []
        self.assertTrue(
            any(item.get("role") == "projected_row_header" for item in semantic_header)
            or any(row and row[0] is None for row in matrices[0].get("semantic_grid", []) or []),
            msg="the left stub/projected row-header column must be explicit semantic metadata",
        )

    def test_projected_stub_matrix_keeps_low_tail_row_inside_table_region(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000182"))

        matrices = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
            and str(table.get("table_family") or "") in {"comparison_matrix", "projected_stub_matrix"}
        ]
        self.assertEqual(len(matrices), 1)
        grid = matrices[0].get("semantic_grid") or matrices[0].get("display_grid") or matrices[0].get("raw_grid") or []
        table_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in grid)

        self.assertIn("Highlight", table_text)
        self.assertIn("Achieved 1st place in the OCR World Competition", table_text)
        self.assertIn("Kaggle", table_text)
        self.assertIn("KLUE", table_text)

    def test_multicolumn_body_flowchart_text_is_not_promoted_to_visual_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000120"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        table_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for table in page_tables
            for row in (table.get("display_grid") or table.get("raw_grid") or [])
        )

        self.assertNotIn("Sickle cell hemoglobin and | normal hemoglobin differ", table_text)
        visual_tables = [
            table
            for table in page_tables
            if str(table.get("detection_source") or table.get("detection_method") or "") == "visual_structure_grid"
        ]
        self.assertTrue(visual_tables, msg="the internal flowchart grid should remain recoverable as table evidence")
        self.assertTrue(
            all("Sickle cell hemoglobin and" not in str(table.get("display_grid") or table.get("raw_grid") or "") for table in visual_tables),
            msg="continuous body prose must be released from the visual table candidate before AST projection",
        )
        table_text_compact = " ".join(table_text.split())
        self.assertIn("Genes in DNA", table_text_compact)
        self.assertIn("Protein", table_text_compact)
        self.assertIn("Characteristics", table_text_compact)
        self.assertIn("normal hemoglobin", table_text_compact)

    def test_flowchart_matrix_does_not_emit_tail_fragment_as_independent_semantic_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000120"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        semantic_tables = [table for table in page_tables if table.get("semantic_projection_v2")]
        self.assertEqual(
            len([table for table in semantic_tables if table.get("table_family") == "flowchart_matrix"]),
            1,
        )
        tail_fragments = [
            table
            for table in semantic_tables
            if str(table.get("table_family") or "") == "single_column_tail_fragment"
        ]
        self.assertEqual(
            tail_fragments,
            [],
            msg="a one-column continuation like 'in long rods' should be merged into flowchart semantic evidence, not kept as its own semantic table",
        )
        semantic_text = " ".join(
            str(cell or "")
            for table in semantic_tables
            for row in (table.get("semantic_grid") or table.get("display_grid") or [])
            if isinstance(row, list)
            for cell in row
        )
        self.assertIn("sickle-shaped red blood cells", semantic_text)
        self.assertIn("clogged small blood vessels", semantic_text)

    def test_flowchart_matrix_compacts_wrapped_cell_fragments_into_logical_rows(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000120"))

        flowcharts = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
            and str(table.get("table_family") or "") == "flowchart_matrix"
        ]
        self.assertEqual(len(flowcharts), 1)
        table = flowcharts[0]
        semantic_grid = table.get("semantic_grid") or []
        semantic_rows = [
            " | ".join(str(cell or "") for cell in row)
            for row in semantic_grid
            if isinstance(row, list)
        ]
        semantic_text = "\n".join(semantic_rows)

        self.assertLessEqual(
            len(semantic_grid),
            4,
            msg="wrapped phrases inside a flowchart row should not remain as separate semantic rows",
        )
        self.assertIn("2 copies of the allele that codes for normal hemoglobin (SS)", semantic_text)
        self.assertIn("Normal hemoglobin dissolves in the cytosol of red blood cells", semantic_text)
        self.assertIn("Disk-shaped red blood cells", semantic_text)
        self.assertNotIn("\nthat codes for", semantic_text)
        self.assertNotIn("\nnormal hemoglobin", semantic_text)

    def test_multicolumn_body_flowchart_keeps_introductory_prose_before_table(self) -> None:
        from api.main import _build_document_body_markdown_sections

        path = _require_benchmark_pdf("01030000000120")
        result = parse_pdf(path)
        result.pop("source_path", None)
        markdown = "\n".join(
            _build_document_body_markdown_sections(
                {**result, "file_id": path.stem, "filename": path.name}
            )
        )

        intro = "Sickle cell hemoglobin and normal hemoglobin differ in only a single amino acid"
        self.assertIn(intro, markdown)
        self.assertLess(markdown.index(intro), markdown.index("Genes in DNA"))
        self.assertNotIn("Sickle cell hemoglobin and | normal hemoglobin differ", markdown)

    def test_embedded_image_table_without_caption_is_recovered_as_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000110"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        table_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for table in page_tables
            for row in (table.get("display_grid") or table.get("raw_grid") or [])
        )
        self.assertTrue(
            page_tables,
            msg=(
                "an embedded raster table with repeated column anchors must enter the unified table AST "
                "even when it has no explicit caption or text-layer table object"
            ),
        )
        self.assertIn("Temperature", table_text)
        self.assertIn("Kinematic viscosity", table_text)
        self.assertIn("1.793E-06", table_text)
        self.assertIn("8.930E-07", table_text)

    def test_textual_image_table_content_is_not_rendered_twice_when_table_owns_it(self) -> None:
        from api.main import _build_document_body_markdown_sections

        path = _require_benchmark_pdf("01030000000200")
        result = parse_pdf(path)
        result.pop("source_path", None)
        markdown = "\n".join(
            _build_document_body_markdown_sections(
                {**result, "file_id": path.stem, "filename": path.name},
                table_export_mode="auto_semantic",
            )
        )

        self.assertIn("| Service Stage | Function Name | Explanation | Expected Benefit |", markdown)
        self.assertLessEqual(
            markdown.count("Service Stage Function Name Explanation Expected Benefit"),
            1,
            msg="textual screenshot OCR should not be rendered repeatedly once a structured table owns the content",
        )
        self.assertLessEqual(
            markdown.count("Key Functions by Main Service Flow Introduction of product services"),
            1,
            msg="covered image OCR text should not duplicate the same table content before and after the structured table",
        )

    def test_textual_image_service_matrix_keeps_late_rows_inside_table_region(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000200"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(len(page_tables), 1)
        table = page_tables[0]
        grid = table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or []
        evidence_grid = table.get("display_grid") or table.get("raw_grid") or grid
        table_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in grid)

        self.assertIn("Service Stage", table_text)
        self.assertIn("Function Name", table_text)
        self.assertIn("Pipeline configuration", table_text)
        self.assertIn("Model training", table_text)
        self.assertIn("Full Pack Monitoring", table_text)
        self.assertIn("Guide and help", table_text)
        self.assertGreaterEqual(
            len([row for row in evidence_grid if any(str(cell or "").strip() for cell in row)]),
            12,
            msg="aligned text below a structured raster/text matrix should remain owned by the table region",
        )

    def test_textual_image_service_matrix_keeps_complete_schema_columns(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000200"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(len(page_tables), 1)
        grid = page_tables[0].get("display_grid") or page_tables[0].get("raw_grid") or []

        self.assertGreaterEqual(max((len(row) for row in grid if isinstance(row, list)), default=0), 4)
        schema_rows = [
            row
            for row in grid
            if isinstance(row, list)
            and sum(1 for cell in row if str(cell or "").strip()) >= 4
        ]
        self.assertTrue(
            schema_rows,
            msg="schema-complete wide table candidates must not be downgraded by lane-local split ownership",
        )

    def test_borderless_table_region_stops_before_following_body_prompts(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000117"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(len(page_tables), 1)
        grid = page_tables[0].get("display_grid") or page_tables[0].get("raw_grid") or []
        table_text = " ".join(
            str(cell or "")
            for row in grid
            if isinstance(row, list)
            for cell in row
        )

        self.assertIn("Saccharometer", table_text)
        self.assertIn("Yeast Suspension", table_text)
        self.assertNotIn("Employing Steps in the Scientific Method", table_text)
        self.assertLessEqual(
            len([row for row in grid if isinstance(row, list) and any(str(cell or "").strip() for cell in row)]),
            5,
            msg="following worksheet/body prompts must remain outside the table evidence region",
        )

    def test_sparse_text_layer_visual_text_page_uses_full_page_ocr_fallback(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000141"))

        page_blocks = [
            block
            for page in result.get("document_ast", {}).get("pages", []) or []
            if int(page.get("page", 0) or 0) == 1
            for block in page.get("blocks", []) or []
        ]
        recovered_text = "\n".join(
            str(block.get("text") or "")
            for block in page_blocks
            if str(block.get("block_type") or "") in {"text", "paragraph"}
        )
        self.assertIn("10 THINGS YOU SHOULD KNOW ABOUT", recovered_text)
        self.assertIn("We're all both consumers and creators of creative", recovered_text)
        self.assertIn("COPYRIGHT PROTECTS CREATIVE WORK", recovered_text)
        self.assertIn("Copyright protects creative work, so people", recovered_text)
        self.assertIn("Copyright gives a lot of protection", recovered_text)
        self.assertIn("Creative Commons", recovered_text)
        self.assertNotIn(
            "creative Copyright gives",
            recovered_text,
            msg="full-page OCR must split visual columns before row grouping instead of fusing same-y rows",
        )
        self.assertEqual(
            [table for table in result.get("table_asts", []) or [] if int(table.get("page", 0) or 0) == 1],
            [],
            msg="full-page OCR fallback must recover body text without promoting the visual page to a data table",
        )

    def test_two_column_reagent_supply_matrix_is_recovered_as_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000121"))

        table_text = _page_table_text(result, page_number=1)
        self.assertIn("Reagents", table_text)
        self.assertIn("Supplies and Equipment", table_text)
        self.assertIn("Restriction Buffer", table_text)
        self.assertIn("Microcentrifuge tube rack", table_text)

    def test_two_column_reagent_supply_matrix_is_one_semantic_inventory_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000121"))

        semantic_inventory_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
            and str(table.get("table_family") or "") == "two_column_inventory"
        ]
        self.assertEqual(
            len(semantic_inventory_tables),
            1,
            msg=(
                "a two-column inventory table with list-in-cell rows should be owned as one semantic table; "
                "repeated boundary rows must not create a second independent semantic table"
            ),
        )
        table = semantic_inventory_tables[0]
        semantic_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for row in table.get("semantic_grid", []) or []
        )
        self.assertIn("To be shared by all groups:", semantic_text)
        self.assertIn("Evidence A", semantic_text)
        self.assertEqual(semantic_text.count("Micropipet tips"), 1)

    def test_embedded_reagent_grid_image_is_recovered_as_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000122"))

        table_text = _page_table_text(result, page_number=1)
        self.assertIn("Tube", table_text)
        self.assertIn("BamHI", table_text)
        self.assertIn("Suspect 1", table_text)
        self.assertIn("DNA", table_text)
        self.assertIn("EA or EB", table_text)

    def test_caption_anchored_rule_table_stops_before_following_prose_and_figure(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000130"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertTrue(page_tables, msg="expected the caption-anchored returns table to be recovered")
        table_text = _page_table_text(result, page_number=1)
        self.assertIn("Time t", table_text)
        self.assertIn("2016", table_text)
        self.assertIn("5%", table_text)
        self.assertNotIn(
            "Another way to represent",
            table_text,
            msg="body prose after a ruled table must be released from table ownership",
        )
        self.assertNotIn(
            "Figure 15.3",
            table_text,
            msg="following figure captions must not be absorbed as rows of the preceding table",
        )

        body_text = "\n".join(
            str(block.get("text") or block.get("title") or "")
            for page in result.get("document_ast", {}).get("pages", []) or []
            if int(page.get("page", 0) or 0) == 1
            for block in page.get("blocks", []) or []
            if str(block.get("block_type") or "") in {"text", "paragraph", "image"}
        )
        self.assertIn("Another way to represent", body_text)
        self.assertIn("Figure 15.3", body_text)

    def test_caption_anchored_rule_table_projects_wrapped_headers_to_body_columns(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000130"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(len(page_tables), 1)
        semantic_grid = page_tables[0].get("semantic_grid") or []
        self.assertGreaterEqual(len(semantic_grid), 2)
        self.assertEqual(len(semantic_grid[0]), 3)
        header_text = " | ".join(str(cell or "") for cell in semantic_grid[0])
        self.assertIn("Time t", header_text)
        self.assertIn("Observed returns on the firm", header_text)
        self.assertIn("potential new investment", header_text)
        self.assertTrue(any(row[:3] == ["2012", "10%", "7%"] for row in semantic_grid), msg=semantic_grid)

    def test_caption_anchored_rule_table_exports_projected_semantic_grid_to_markdown(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000130"))

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                table_export_mode="markdown",
            )
        )

        self.assertIn(
            "| Time t | Observed returns on the firm鈥檚 portfolio over time rtp | Observed returns on a potential new investment for the firm鈥檚 rtj |",
            markdown,
        )
        self.assertIn("| 2012 | 10% | 7% |", markdown)
        self.assertNotIn(
            "| Time t | Observed | returns on the firm鈥檚 | Observed | returns on | a potential new investment |",
            markdown,
        )

    def test_stacked_header_body_fragments_with_same_caption_are_one_logical_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000078"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(
            len(page_tables),
            1,
            msg="a captioned table split into header and body bands should be merged before semantic export",
        )
        table_text = _page_semantic_table_text(result, page_number=1)
        self.assertIn("AMS", table_text)
        self.assertIn("Average Annual Growth", table_text)
        self.assertIn("Indonesia", table_text)
        self.assertIn("Remittance inflows", table_text)

    def test_two_column_species_list_with_bottom_caption_is_recovered_as_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000132"))

        table_text = _page_table_text(result, page_number=1)
        self.assertIn("Fish species", table_text)
        self.assertIn("on IUCN Red List", table_text)
        self.assertIn("Potosi Pupfish", table_text)
        self.assertIn("Cyprinodon alvarezi", table_text)
        self.assertIn("Golden Skiffia", table_text)

    def test_single_column_framework_list_inside_box_is_recovered_as_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000149"))

        table_text = _page_table_text(result, page_number=1)
        self.assertIn("Eco-Circle Competence Framework", table_text)
        self.assertIn("#1: The 3 Rs", table_text)
        self.assertIn("#7: Supporting Local Eco-friendly", table_text)

    def test_appendix_tables_with_bottom_captions_are_recovered_as_tables(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000197"))

        table_text = _page_table_text(result, page_number=1)
        self.assertIn("Filtered Task Name", table_text)
        self.assertIn("task228_arc_answer_generation_easy", table_text)
        self.assertIn("ARC", table_text)
        self.assertIn("GSM8K", table_text)
        self.assertIn("0.70", table_text)

    def test_dense_borderless_leaderboard_table_is_recovered_as_one_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000188"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        grids = [table.get("display_grid") or table.get("raw_grid") or [] for table in page_tables]
        dense_grids = [
            grid
            for grid in grids
            if len(grid) >= 10 and max((len(row) for row in grid), default=0) >= 10
        ]
        self.assertTrue(
            dense_grids,
            msg=(
                "a dense borderless leaderboard with a stable header row and repeated numeric columns "
                "should be reconstructed as one table, not fragmented into single-row PyMuPDF shards plus body text"
            ),
        )
        table_text = "\n".join(
            " | ".join(str(cell or "") for cell in row)
            for grid in dense_grids
            for row in grid
        )
        self.assertIn("Model", table_text)
        self.assertIn("HellaSwag", table_text)
        self.assertIn("SOLAR 10.7B-Instruct", table_text)
        self.assertIn("Qwen 72B", table_text)
        self.assertIn("Mistral 7B", table_text)
        self.assertIn("37.83", table_text)

    def test_bottom_caption_text_grid_must_not_start_from_middle_of_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000187"))

        table_text = _page_semantic_table_text(result, page_number=1)
        self.assertIn("Training Datasets", table_text)
        self.assertIn("Properties", table_text)
        self.assertIn("Instruction", table_text)
        self.assertIn("Alignment", table_text)
        self.assertIn("Alpaca-GPT4", table_text)
        self.assertIn("Open Source", table_text)

    def test_bottom_note_is_not_promoted_to_table_title(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000170"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertGreaterEqual(len(page_tables), 1)
        first_table = page_tables[0]
        first_title = str(first_table.get("title") or first_table.get("caption_text") or "")
        self.assertNotRegex(
            first_title,
            r"(?i)^table\s+adapted\s+from",
            msg="source/adapted/note text below a table is table note evidence, not the table title",
        )
        self.assertIn("Table adapted from Jones", _page_body_text(result, page_number=1))

    def test_same_row_wide_table_fragments_are_merged_by_shared_row_band(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000190"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        wide_tables = [
            table
            for table in page_tables
            if max((len(row) for row in (table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or [])), default=0) >= 8
        ]
        self.assertTrue(
            wide_tables,
            msg=(
                "left and right fragments with the same row band and complementary headers "
                "must be owned as one wide table instead of two side-by-side tables"
            ),
        )
        table = next(
            (
                candidate
                for candidate in wide_tables
                if "Merge v1" in str(candidate.get("semantic_grid") or candidate.get("display_grid") or candidate.get("raw_grid") or "")
            ),
            wide_tables[0],
        )
        grid = table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or []
        table_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in grid)
        table_text_compact = " ".join(table_text.replace("|", " ").split())

        for expected in (
            "Model Merge Method",
            "H6 (Avg.)",
            "ARC",
            "HellaSwag",
            "MMLU",
            "TruthfulQA",
            "Winogrande",
            "GSM8K",
            "Merge v1 Average",
            "88.01",
            "64.90",
        ):
            self.assertIn(expected, table_text_compact)
        self.assertLessEqual(
            sum(
                1
                for table in page_tables
                if "Table 6: Performance comparison" in str(table.get("title") or table.get("caption_text") or "")
            ),
            1,
            msg="the shared caption owner should not be duplicated across horizontally split fragments",
        )

    def test_bottom_caption_owns_preceding_short_borderless_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000190"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        table_texts = [
            "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or [])
            )
            for table in page_tables
        ]

        self.assertTrue(
            any(
                "Cand. 1" in text
                and "73.73" in text
                and "Cand. 2" in text
                and "59.14" in text
                for text in table_texts
            ),
            msg="a compact borderless table immediately above its caption must be owned as table evidence, not body text",
        )
        self.assertTrue(
            any(
                "Table 6: Performance comparison" in str(table.get("title") or table.get("caption_text") or "")
                and "Cand. 1" in table_text
                for table, table_text in zip(page_tables, table_texts)
            ),
            msg="bottom captions should attach to the preceding compact table instead of the following table",
        )

    def test_footnote_below_anchor_recovers_following_small_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000116"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        table_texts = [
            "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or [])
            )
            for table in page_tables
        ]

        self.assertGreaterEqual(
            len([text for text in table_texts if "Saccharometer" in text and "DI Water" in text]),
            2,
            msg="a footnote that explicitly points to a following compact table should not leave that table as body text",
        )
        self.assertTrue(
            any("16 ml" in text and "12 ml" in text and "0 ml" in text for text in table_texts),
            msg="the small continuation table after a 'see table below' anchor must enter the table AST",
        )

    def test_prose_after_captioned_table_is_not_reconstructed_as_second_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000166"))

        table_text = _page_table_text(result, page_number=1)
        body_text = _page_body_text(result, page_number=1)

        self.assertIn("Mineral or colloid type", table_text)
        self.assertIn("kaolinite", table_text)
        self.assertNotIn(
            "As an example of",
            table_text,
            msg=(
                "body prose immediately after a completed table must remain body flow; "
                "late borderless reconstruction must not reuse the prior caption and project prose as a new table"
            ),
        )
        self.assertIn("As an example of this mineralogy approach", body_text)

    def test_source_and_next_caption_bridge_is_not_reconstructed_as_extra_table(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000081"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(
            len(page_tables),
            2,
            msg=(
                "source notes and the following table caption form an inter-table boundary, "
                "not a third bridge table"
            ),
        )

        bridge_rows = []
        for table in page_tables:
            grid = table.get("display_grid") or table.get("raw_grid") or []
            grid_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in grid)
            if "Source: TeamLease Regtech" in grid_text and "TABLE 23" in grid_text:
                bridge_rows.append(grid_text)
        self.assertEqual(bridge_rows, [])

    def test_figure_internal_ocr_text_stays_figure_evidence_not_body_flow(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000014"))

        page_blocks = [
            block
            for page in result.get("document_ast", {}).get("pages", []) or []
            if int(page.get("page", 0) or 0) == 1
            for block in page.get("blocks", []) or []
        ]
        body_text = "\n".join(
            str(block.get("text") or "")
            for block in page_blocks
            if str(block.get("block_type") or "") in {"text", "paragraph"}
        )
        figure_text = "\n".join(
            str(block.get("content_text") or "")
            for block in page_blocks
            if str(block.get("block_type") or "") == "image"
        )

        self.assertNotIn(
            "Typical three-poled Bedeuin tent",
            body_text,
            msg="OCR text from an image region must not be inserted as body text when the region remains a figure",
        )
        self.assertIn("Figure 8.15 Typical black-and-white Bedouin tent.", figure_text)

    def test_borderless_long_description_table_keeps_header_and_body_together(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000090"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        table_texts = [
            "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or [])
            )
            for table in page_tables
        ]
        recovered = "\n".join(table_texts)

        self.assertTrue(
            any("Jurisdiction" in text and "Foreign Ownership Permitted" in text for text in table_texts),
            msg="the multi-row header of a borderless long-description table must remain table evidence",
        )
        self.assertTrue(
            any("Finland" in text and "Prior approval for a foreigner" in text for text in table_texts),
            msg="body rows with compact key columns and long description columns must be owned by the same table schema",
        )
        self.assertIn("India", recovered)
        self.assertIn("Prohibition on acquisition of", recovered)
        self.assertEqual(
            sum(1 for text in table_texts if "India" in text or "Finland" in text or "Jurisdiction" in text),
            1,
            msg="one logical borderless table should not be split into a header table plus body fragments",
        )

    def test_sparse_anchor_projection_does_not_discard_trailing_description_columns(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000088"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(len(page_tables), 1)
        semantic_grid = page_tables[0].get("semantic_grid") or []
        self.assertTrue(semantic_grid)
        self.assertEqual(
            len(semantic_grid[0]),
            5,
            msg=(
                "sparse-anchor projection may collapse header-wrap columns only when the projected anchors "
                "partition the whole source width; trailing long-description columns are real table evidence"
            ),
        )
        table_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in semantic_grid)
        self.assertIn("Restrictions on Foreign Ownership", table_text)
        self.assertIn("Foreign Ownership Reporting Requirements", table_text)
        self.assertIn("Prohibition on ownership of property", table_text)

    def test_stacked_numeric_atoms_in_schedule_rows_expand_to_logical_rows(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000127"))

        schedules = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
            and "3-Year" in str(table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or "")
        ]
        self.assertTrue(schedules)
        grid = schedules[0].get("semantic_grid") or schedules[0].get("display_grid") or schedules[0].get("raw_grid") or []

        self.assertTrue(any(row[:4] == ["6", None, "5.76%", "8.93%"] for row in grid), msg=grid)
        self.assertTrue(any(row[:4] == ["7", None, None, "8.93%"] for row in grid), msg=grid)
        self.assertTrue(any(row[:4] == ["8", None, None, "4.46%"] for row in grid), msg=grid)
        self.assertFalse(any(str(row[0] if row else "") == "6 7 8" for row in grid), msg=grid)

    def test_repeated_captionless_numeric_matrix_is_recovered_after_intro_prose(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000127"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        table_texts = [
            "\n".join(
                " | ".join(str(cell or "") for cell in row)
                for row in (table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or [])
            )
            for table in page_tables
        ]
        macrs_tables = [
            text
            for text in table_texts
            if "Recovery Rate" in text
            and "Unadjusted Basis" in text
            and "Depreciation Expense" in text
            and "Accumulated Depreciation" in text
            and ".4445" in text
            and "$77,780" in text
            and "$100,000" in text
        ]
        body_text = _page_body_text(result, page_number=1)

        self.assertGreaterEqual(
            len(page_tables),
            3,
            msg="sequential captionless text-layer numeric matrices should be independent table regions",
        )
        self.assertTrue(
            macrs_tables,
            msg="the MACRS numeric matrix must keep its header and value rows as table evidence",
        )
        self.assertFalse(
            "Year Recovery Rate Unadjusted Basis" in body_text and "$44,450" in body_text,
            msg="a recovered numeric matrix must not remain duplicated as body-flow text",
        )

    def test_caption_anchored_blank_form_keeps_short_key_rows_and_excludes_following_prose(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000165"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(len(page_tables), 1)
        table = page_tables[0]
        grid = table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or []
        table_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in grid)

        self.assertIn("Added cation", table_text)
        self.assertIn("Size & Settling Rates of Floccules", table_text)
        for label in ("K+", "Na+", "Ca2+", "Al3+", "Check"):
            self.assertIn(label, table_text)
        self.assertNotIn("Activity 4. Determining", table_text)
        self.assertNotIn("Phenolphthalein changes", table_text)

    def test_caption_anchored_numeric_value_rows_are_not_flattened_as_blank_form_keys(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000166"))

        page_tables = [
            table
            for table in result.get("table_asts", []) or []
            if int(table.get("page", 0) or 0) == 1
        ]
        self.assertEqual(len(page_tables), 1)
        grid = page_tables[0].get("semantic_grid") or page_tables[0].get("display_grid") or page_tables[0].get("raw_grid") or []
        table_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in grid)

        self.assertNotIn("kaolinite 10 |", table_text)
        self.assertTrue(any(row[:2] == ["kaolinite", "10"] for row in grid), msg=table_text)
        self.assertTrue(any(row[:2] == ["humus", "200"] for row in grid), msg=table_text)

    def test_contents_entries_are_not_promoted_to_body_headings(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000198"))
        page_text_blocks = [
            block
            for page in result.get("document_ast", {}).get("pages", []) or []
            if int(page.get("page", 0) or 0) == 1
            for block in page.get("blocks", []) or []
            if str(block.get("block_type") or "") == "text"
        ]
        contents_entries = [
            block
            for block in page_text_blocks
            if str(block.get("text") or "").strip() in {
                "1. Overview of OCR Pack",
                "2. Introduction of Product Services and Key Features",
                "3. Product - Detail Specification",
                "4. Integration Policy",
                "5. FAQ",
            }
        ]

        self.assertTrue(contents_entries)
        self.assertFalse(
            any(str(block.get("semantic_role") or "") == "section_heading" for block in contents_entries),
            msg="TOC/list entries on a contents page must not be owned as body section headings",
        )

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
            )
        )
        self.assertNotIn("### 1. Overview of OCR Pack", markdown)
        self.assertNotIn("### 2. Introduction of Product Services and Key Features", markdown)
        self.assertIn("1. Overview of OCR Pack", markdown)

    def test_toc_blocks_can_render_as_benchmark_neutral_body_without_promoting_entries(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000016"))

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
                render_toc_blocks=True,
            )
        )

        self.assertIn("# Table of contents", markdown)
        self.assertIn("Introduction 7", markdown)
        self.assertIn("1. Changing Practices, Shifting Sites 7", markdown)
        self.assertNotIn("### 1. Changing Practices, Shifting Sites", markdown)

    def test_wrapped_toc_entries_render_once_with_page_locator(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000044"))

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
                render_toc_blocks=True,
            )
        )

        self.assertIn("# Table of Contents", markdown)
        self.assertIn("Executive Summary 4", markdown)
        self.assertIn("Political Parties, Candidates Registration and Election Campaign 18", markdown)
        self.assertNotIn("### Executive Summary", markdown)

    def test_numbered_procedure_steps_remain_body_list_items_not_headings(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000115"))
        page_text_blocks = [
            block
            for page in result.get("document_ast", {}).get("pages", []) or []
            if int(page.get("page", 0) or 0) == 1
            for block in page.get("blocks", []) or []
            if str(block.get("block_type") or "") == "text"
        ]
        procedure_steps = [
            block
            for block in page_text_blocks
            if str(block.get("text") or "").strip().startswith(
                (
                    "5. Rotate the nosepiece",
                    "6. Refocus using",
                    "7. Move the slide",
                    "8. Now use the fine adjustment",
                    "9. Your slide MUST",
                )
            )
        ]

        self.assertGreaterEqual(len(procedure_steps), 4)
        self.assertTrue(
            all(str(block.get("semantic_role") or "") == "body_list_item" for block in procedure_steps),
            msg="single-level numbered procedure steps must keep body-list ownership across the sequence",
        )

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
            )
        )
        self.assertNotIn("### 5. Rotate the nosepiece", markdown)
        self.assertNotIn("### 6. Refocus using", markdown)
        self.assertIn("5. Rotate the nosepiece", markdown)

    def test_numeric_chart_atoms_are_not_promoted_to_headings(self) -> None:
        result = parse_pdf(_require_benchmark_pdf("01030000000199"))

        page_text_blocks = [
            block
            for page in result.get("document_ast", {}).get("pages", []) or []
            if int(page.get("page", 0) or 0) == 1
            for block in page.get("blocks", []) or []
            if str(block.get("block_type") or "") == "text"
        ]
        numeric_heading_false_positives = [
            block
            for block in page_text_blocks
            if str(block.get("text") or "").strip() in {"94.1 4"}
        ]

        self.assertTrue(numeric_heading_false_positives)
        self.assertFalse(
            any(str(block.get("semantic_role") or "") == "section_heading" for block in numeric_heading_false_positives),
            msg="numeric chart/table atoms without lexical heading text must not become section headings",
        )

    def test_benchmark_neutral_markdown_skips_decorative_image_placeholders(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000198"))

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
                embed_images=False,
                image_text_mode="caption_only",
                skip_uncaptioned_image_placeholders=True,
                visual_heading_projection=True,
            )
        )

        self.assertNotIn("![Figure 1]", markdown)
        self.assertNotIn("![Figure 2]", markdown)
        self.assertIn("1. Overview of OCR Pack", markdown)

    def test_visual_standalone_titles_remain_markdown_headings(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000115"))

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
                embed_images=False,
                image_text_mode="caption_only",
                skip_uncaptioned_image_placeholders=True,
                visual_heading_projection=True,
            )
        )

        self.assertIn("# Changing objectives:", markdown)
        self.assertIn("# Steps for Using the Microscope:", markdown)
        self.assertNotIn("### 5. Rotate the nosepiece", markdown)

    def test_short_page_title_remains_markdown_heading_without_promoting_body(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000157"))

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
                embed_images=False,
                image_text_mode="caption_only",
                skip_uncaptioned_image_placeholders=True,
                visual_heading_projection=True,
            )
        )

        self.assertIn("# Stop", markdown)
        self.assertNotIn("# Check your emotions", markdown)

    def test_visual_heading_projection_rejects_page_footer_band(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000107"))

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
                embed_images=False,
                image_text_mode="caption_only",
                skip_uncaptioned_image_placeholders=True,
                visual_heading_projection=True,
            )
        )

        self.assertIn("# Print vs. Digital", markdown)
        self.assertNotIn("# Online Survey | 39", markdown)

    def test_figure_title_sentence_can_project_as_visual_heading(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000175"))

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
                embed_images=False,
                image_text_mode="caption_only",
                skip_uncaptioned_image_placeholders=True,
                visual_heading_projection=True,
            )
        )

        self.assertIn("# Orion Region at Different Wavelengths.", markdown)

    def test_activity_and_numbered_section_titles_project_as_visual_headings(self) -> None:
        from api.main import _build_document_body_markdown_sections

        activity_result = parse_pdf(_require_benchmark_pdf("01030000000168"))
        numbered_result = parse_pdf(_require_benchmark_pdf("01030000000186"))

        activity_markdown = "\n".join(
            _build_document_body_markdown_sections(
                activity_result,
                body_heading=None,
                embed_images=False,
                image_text_mode="caption_only",
                skip_uncaptioned_image_placeholders=True,
                visual_heading_projection=True,
            )
        )
        numbered_markdown = "\n".join(
            _build_document_body_markdown_sections(
                numbered_result,
                body_heading=None,
                embed_images=False,
                image_text_mode="caption_only",
                skip_uncaptioned_image_placeholders=True,
                visual_heading_projection=True,
            )
        )

        self.assertIn("# Activity 1: Determining pH With Indicator Strips (Field Method)", activity_markdown)
        self.assertIn("# Activity 2: Determining Soil pH with a pH Meter", activity_markdown)
        self.assertIn("# 2 Depth Up-Scaling", numbered_markdown)

    def test_sparse_document_command_titles_project_without_promoting_following_list_items(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000069"))

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
                embed_images=False,
                image_text_mode="caption_only",
                skip_uncaptioned_image_placeholders=True,
                visual_heading_projection=True,
            )
        )

        self.assertIn("# Replace", markdown)
        self.assertIn("# Trash", markdown)
        self.assertNotIn("# l. Replace Plastics", markdown)

    def test_landscape_card_page_titles_project_as_visual_headings(self) -> None:
        from api.main import _build_document_body_markdown_sections

        result = parse_pdf(_require_benchmark_pdf("01030000000184"))

        markdown = "\n".join(
            _build_document_body_markdown_sections(
                result,
                body_heading=None,
                embed_images=False,
                image_text_mode="caption_only",
                skip_uncaptioned_image_placeholders=True,
                visual_heading_projection=True,
            )
        )

        self.assertIn("# SS Pack allows businesses to access further data more rapidly", markdown)
        self.assertIn("# Higher Return of Information", markdown)
        self.assertIn("# Optimal Attempt", markdown)
        self.assertIn("# Reduced Information Acquisition Time", markdown)
        self.assertIn("# SOTA", markdown)
        self.assertIn("# Cutting-Edge Technology", markdown)


def _page_table_text(result: dict, *, page_number: int) -> str:
    page_tables = [
        table
        for table in result.get("table_asts", []) or []
        if int(table.get("page", 0) or 0) == page_number
    ]
    if not page_tables:
        return ""
    return "\n".join(
        " | ".join(str(cell or "") for cell in row)
        for table in page_tables
        for row in (table.get("display_grid") or table.get("raw_grid") or [])
    )


def _page_semantic_table_text(result: dict, *, page_number: int) -> str:
    page_tables = [
        table
        for table in result.get("table_asts", []) or []
        if int(table.get("page", 0) or 0) == page_number
    ]
    if not page_tables:
        return ""
    return "\n".join(
        " | ".join(str(cell or "") for cell in row)
        for table in page_tables
        for row in (table.get("semantic_grid") or table.get("display_grid") or table.get("raw_grid") or [])
    )


def _page_body_text(result: dict, *, page_number: int) -> str:
    return "\n".join(
        str(block.get("text") or block.get("title") or "")
        for page in result.get("document_ast", {}).get("pages", []) or []
        if int(page.get("page", 0) or 0) == page_number
        for block in page.get("blocks", []) or []
        if str(block.get("block_type") or "") in {"text", "paragraph", "image"}
    )


