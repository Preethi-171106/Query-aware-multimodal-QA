"""Comprehensive Unit and Integration Tests for Phase 3 Multimodal Extraction.

Verifies:
1. Extraction of structured tables into Markdown and grid representations.
2. Extraction of embedded image/figure assets with linked captions.
3. Handling documents with no tables or images without crashing or false positives.
4. Correct page-number (1-based) and document-ID preservation across modalities.
5. Unique, collision-safe image filenames and on-disk file integrity.
6. Backward compatibility with Phase 2 text extraction and chunking.
"""

import io
import shutil
import tempfile
import unittest
from pathlib import Path

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.table_extractor import TableExtractor
from src.ingestion.visual_extractor import VisualExtractor
from src.ingestion.text_extractor import TextExtractor
from src.indexing.schema import ModalityType, EvidenceChunk, ExtractedDocument
from tests.test_pdf_ingestion import VALID_PDF_BYTES


# Synthetic PDF containing a 2x2 structured table with lines and text
TABLE_PDF_BYTES = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 500 500] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 300 >> stream
0 0 0 RG
1 w
100 300 200 100 re S
100 350 200 0 re S
200 300 0 100 re S
BT
/F1 12 Tf
110 365 Td (Model) Tj
100 0 Td (F1_Score) Tj
-100 -40 Td (Baseline) Tj
100 0 Td (0.84) Tj
ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000229 00000 n 
0000000580 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
650
%%EOF"""


# Synthetic PDF containing an embedded raster image XObject and a caption
IMAGE_PDF_BYTES = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] /Contents 4 0 R /Resources << /XObject << /Im1 5 0 R >> /Font << /F1 6 0 R >> >> >> endobj
4 0 obj << /Length 120 >> stream
q
100 0 0 100 50 150 cm
/Im1 Do
Q
BT
/F1 12 Tf
50 100 Td
(Figure 1: Multimodal architecture diagram) Tj
ET
endstream endobj
5 0 obj << /Type /XObject /Subtype /Image /Width 20 /Height 20 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 1200 >> stream
""" + (b"\xff\x00\x00" * 400) + b"""
endstream endobj
6 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 7
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000249 00000 n 
0000000420 00000 n 
0000001700 00000 n 
trailer << /Size 7 /Root 1 0 R >>
startxref
1780
%%EOF"""


class TestMultimodalExtractionPipeline(unittest.TestCase):
    """Test suite for Table, Figure, and Multimodal Evidence extraction."""

    def setUp(self):
        """Create isolated test directory for uploads and extracted figure assets."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_phase3_multimodal_"))
        self.figures_dir = self.test_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = PDFLoader(upload_dir=self.test_dir)
        self.table_extractor = TableExtractor()
        self.visual_extractor = VisualExtractor(output_dir=self.figures_dir)
        self.text_extractor = TextExtractor(chunk_size=300, chunk_overlap=50)

    def tearDown(self):
        """Clean up all temporary files and figures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. PDF containing a simple table
    # --------------------------------------------------------------------------
    def test_01_simple_table_extraction(self):
        """Test that structured tables are extracted with rows, columns, and Markdown."""
        doc_info = self.loader.process_and_save(TABLE_PDF_BYTES, "results_table.pdf")
        tables = self.table_extractor.extract_document_tables(
            file_path=doc_info.file_path,
            document_id=doc_info.document_id,
            document_name=doc_info.file_name
        )

        self.assertEqual(len(tables), 1)
        table_chunk = tables[0]
        self.assertEqual(table_chunk.modality, ModalityType.TABLE)
        self.assertEqual(table_chunk.page_number, 1)
        self.assertEqual(table_chunk.document_id, doc_info.document_id)
        self.assertTrue(table_chunk.chunk_id.endswith("_t1"))

        # Verify structured metadata
        table_data = table_chunk.table_data
        self.assertIsNotNone(table_data)
        self.assertEqual(table_data["row_count"], 2)
        self.assertEqual(table_data["col_count"], 2)
        self.assertEqual(table_data["headers"], ["Model", "F1_Score"])
        self.assertEqual(table_data["rows"], [["Baseline", "0.84"]])

        # Verify Markdown representation
        self.assertIn("| Model | F1_Score |", table_chunk.content)
        self.assertIn("| --- | --- |", table_chunk.content)
        self.assertIn("| Baseline | 0.84 |", table_chunk.content)
        self.assertEqual(table_chunk.citation_label, "[Page 1, Table 1]")

    # --------------------------------------------------------------------------
    # 2. PDF containing an embedded image & caption
    # --------------------------------------------------------------------------
    def test_02_embedded_image_extraction(self):
        """Test that embedded figures are extracted, saved to disk, and linked to captions."""
        doc_info = self.loader.process_and_save(IMAGE_PDF_BYTES, "system_figure.pdf")
        figures = self.visual_extractor.extract_document_figures(
            file_path=doc_info.file_path,
            document_id=doc_info.document_id,
            document_name=doc_info.file_name
        )

        self.assertEqual(len(figures), 1)
        fig_chunk = figures[0]
        self.assertEqual(fig_chunk.modality, ModalityType.FIGURE)
        self.assertEqual(fig_chunk.page_number, 1)
        self.assertEqual(fig_chunk.document_id, doc_info.document_id)
        self.assertTrue(fig_chunk.chunk_id.endswith("_fig1"))

        # Verify image file was written and exists on disk
        self.assertIsNotNone(fig_chunk.image_path)
        saved_img = Path(fig_chunk.image_path)
        self.assertTrue(saved_img.exists())
        self.assertGreater(saved_img.stat().st_size, 0)

        # Verify metadata & caption linking
        meta = fig_chunk.metadata
        self.assertEqual(meta["width"], 20)
        self.assertEqual(meta["height"], 20)
        self.assertEqual(meta["caption"], "Figure 1: Multimodal architecture diagram")
        self.assertTrue(meta["has_caption"])
        self.assertEqual(fig_chunk.citation_label, "[Page 1, Figure 1]")

    # --------------------------------------------------------------------------
    # 3. PDF with no tables or images
    # --------------------------------------------------------------------------
    def test_03_no_tables_or_images(self):
        """Test that a pure text document produces 0 tables and 0 figures without crashing."""
        doc_info = self.loader.process_and_save(VALID_PDF_BYTES, "pure_text.pdf")
        
        tables = self.table_extractor.extract_document_tables(
            file_path=doc_info.file_path,
            document_id=doc_info.document_id,
            document_name=doc_info.file_name
        )
        figures = self.visual_extractor.extract_document_figures(
            file_path=doc_info.file_path,
            document_id=doc_info.document_id,
            document_name=doc_info.file_name
        )

        self.assertEqual(len(tables), 0)
        self.assertEqual(len(figures), 0)

        # In full extraction, only text chunks are present
        extracted = self.text_extractor.extract_document(doc_info)
        self.assertEqual(extracted.total_tables, 0)
        self.assertEqual(extracted.total_figures, 0)
        self.assertGreater(extracted.total_text_chunks, 0)

    # --------------------------------------------------------------------------
    # 4. Correct page-number and document-ID preservation
    # --------------------------------------------------------------------------
    def test_04_provenance_and_citations(self):
        """Verify that all evidence chunks maintain strict 1-indexed page and doc provenance."""
        doc_info = self.loader.process_and_save(TABLE_PDF_BYTES, "audit_tables.pdf")
        extracted = self.text_extractor.extract_document(doc_info)

        for chunk in extracted.chunks:
            self.assertEqual(chunk.document_id, doc_info.document_id)
            self.assertEqual(chunk.document_name, doc_info.file_name)
            self.assertGreaterEqual(chunk.page_number, 1)
            self.assertTrue(chunk.citation_label.startswith(f"[Page {chunk.page_number}"))

    # --------------------------------------------------------------------------
    # 5. Unique image filenames and safe file handling
    # --------------------------------------------------------------------------
    def test_05_safe_unique_image_filenames(self):
        """Verify that extracted images use collision-safe filenames with content hashes."""
        doc_info = self.loader.process_and_save(IMAGE_PDF_BYTES, "figure_collision_test.pdf")
        figures = self.visual_extractor.extract_document_figures(
            file_path=doc_info.file_path,
            document_id=doc_info.document_id,
            document_name=doc_info.file_name
        )

        self.assertEqual(len(figures), 1)
        filename = Path(figures[0].image_path).name
        # Filename format: {doc_id}_p{page}_fig{index}_{hash}.png
        self.assertTrue(filename.startswith(f"{doc_info.document_id}_p1_fig1_"))
        self.assertTrue(filename.endswith(".png"))

    # --------------------------------------------------------------------------
    # 6. Backward compatibility with Phase 2 text extraction
    # --------------------------------------------------------------------------
    def test_06_backward_compatibility_with_phase2(self):
        """Verify that text extraction from Phase 2 continues to work unaltered."""
        doc_info = self.loader.process_and_save(VALID_PDF_BYTES, "phase2_compat.pdf")
        extracted = self.text_extractor.extract_document(doc_info)

        # Text chunks must be intact
        self.assertGreater(len(extracted.text_chunks), 0)
        first_text_chunk = extracted.text_chunks[0]
        self.assertEqual(first_text_chunk.modality, ModalityType.TEXT)
        self.assertIn("Query-Aware Multimodal Evidence Fusion", first_text_chunk.content)
        self.assertEqual(extracted.total_pages, 1)


if __name__ == "__main__":
    unittest.main()
