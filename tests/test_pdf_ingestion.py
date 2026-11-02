"""Comprehensive Unit and Integration Tests for PDF Ingestion & Text Extraction.

Tests all 6 Phase 2 requirements:
1. Valid text-based PDF extraction.
2. Empty and invalid / corrupted PDF handling.
3. Multi-page PDF processing with 1-based page citations.
4. Handling pages with no extractable text without crashing.
5. Chunking behavior, overlap, and metadata preservation.
6. Safe file handling and duplicate filename resolution.
"""

import io
import shutil
import tempfile
import unittest
from pathlib import Path
import pypdf

from src.ingestion.pdf_loader import (
    PDFLoader,
    EmptyPDFError,
    InvalidPDFFormatError,
    CorruptedPDFError,
)
from src.ingestion.text_extractor import TextExtractor
from src.indexing.schema import ModalityType, EvidenceChunk


def generate_clean_pdf_bytes(custom_text: str = "Query-Aware Multimodal Evidence Fusion for Grounded Document QA") -> bytes:
    """Generate a clean, structurally valid PDF byte stream with zero xref offset warnings."""
    text_bytes = custom_text.encode("latin1")
    text_len = len(text_bytes) + 26  # Length of stream content
    
    body = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(text_len).encode("latin1") + b" >>\nstream\n"
        b"BT\n/F1 14 Tf\n30 150 Td\n(" + text_bytes + b") Tj\nET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    offsets = [0]
    for obj_id in [b"1 0 obj", b"2 0 obj", b"3 0 obj", b"4 0 obj", b"5 0 obj"]:
        offsets.append(body.find(obj_id))

    xref_offset = len(body)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n".encode("latin1")

    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_offset).encode("latin1") + b"\n%%EOF\n"
    )
    return body + xref + trailer


VALID_PDF_BYTES = generate_clean_pdf_bytes()


def build_synthetic_multipage_pdf(num_text_pages: int = 2, include_blank_page: bool = True) -> bytes:
    """Helper to build a multi-page synthetic PDF in-memory using pypdf."""
    reader = pypdf.PdfReader(io.BytesIO(VALID_PDF_BYTES))
    writer = pypdf.PdfWriter()
    
    # Page 1: Has valid text
    writer.add_page(reader.pages[0])
    
    # Optional Page 2: Blank page (no text stream)
    if include_blank_page:
        writer.add_blank_page(width=400, height=200)

    # Remaining text pages
    for _ in range(num_text_pages - 1):
        writer.add_page(reader.pages[0])
        
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestPDFIngestionPipeline(unittest.TestCase):
    """Test suite for PDF loading, validation, text extraction, and chunking."""

    def setUp(self):
        """Create an isolated temporary directory for file upload operations."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_pdf_ingestion_"))
        self.loader = PDFLoader(upload_dir=self.test_dir)
        self.extractor = TextExtractor(chunk_size=300, chunk_overlap=50)

    def tearDown(self):
        """Clean up temporary test artifacts."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Valid text-based PDF
    # --------------------------------------------------------------------------
    def test_01_valid_text_pdf(self):
        """Test that a valid PDF is accepted, saved, and its text correctly extracted."""
        doc_info = self.loader.process_and_save(VALID_PDF_BYTES, "research_paper.pdf")
        
        self.assertTrue(doc_info.is_valid)
        self.assertEqual(doc_info.page_count, 1)
        self.assertTrue(doc_info.file_path.exists())
        self.assertTrue(doc_info.document_id.startswith("doc_"))

        extracted_doc = self.extractor.extract_document(doc_info)
        self.assertEqual(extracted_doc.total_pages, 1)
        self.assertEqual(extracted_doc.pages_with_text_count, 1)
        self.assertTrue(len(extracted_doc.chunks) >= 1)
        
        chunk = extracted_doc.chunks[0]
        self.assertEqual(chunk.modality, ModalityType.TEXT)
        self.assertEqual(chunk.page_number, 1)
        self.assertIn("Query-Aware Multimodal Evidence Fusion", chunk.content)

    # --------------------------------------------------------------------------
    # 2. Empty and invalid / corrupted PDF handling
    # --------------------------------------------------------------------------
    def test_02_empty_pdf_raises_error(self):
        """Test that a 0-byte file raises an EmptyPDFError."""
        with self.assertRaises(EmptyPDFError):
            self.loader.process_and_save(b"", "empty.pdf")

    def test_03_invalid_extension_raises_error(self):
        """Test that non-PDF extensions are rejected."""
        with self.assertRaises(InvalidPDFFormatError):
            self.loader.process_and_save(VALID_PDF_BYTES, "notes.docx")

    def test_04_missing_magic_bytes_raises_error(self):
        """Test that a file without PDF header magic bytes is rejected."""
        invalid_bytes = b"This is a plain text file pretending to be a PDF."
        with self.assertRaises(InvalidPDFFormatError):
            self.loader.process_and_save(invalid_bytes, "fake.pdf")

    def test_05_corrupted_pdf_raises_error(self):
        """Test that a corrupted PDF structure is rejected with CorruptedPDFError."""
        corrupted_bytes = b"%PDF-1.4\nGarbage data that cannot be parsed by any PDF parser\x00\xff\xfe"
        with self.assertRaises(CorruptedPDFError):
            self.loader.process_and_save(corrupted_bytes, "corrupt.pdf")

    # --------------------------------------------------------------------------
    # 3. Multi-page PDF processing with 1-based page citations
    # --------------------------------------------------------------------------
    def test_06_multipage_pdf_citations(self):
        """Test that multi-page documents maintain correct 1-indexed page numbering."""
        multipage_bytes = build_synthetic_multipage_pdf(num_text_pages=2, include_blank_page=False)
        doc_info = self.loader.process_and_save(multipage_bytes, "multi_page.pdf")
        
        self.assertEqual(doc_info.page_count, 2)
        extracted = self.extractor.extract_document(doc_info)
        self.assertEqual(extracted.total_pages, 2)

        page_nums = [p.page_number for p in extracted.pages]
        self.assertEqual(page_nums, [1, 2])

        # Verify chunk page attribution
        for chunk in extracted.chunks:
            self.assertIn(chunk.page_number, [1, 2])
            self.assertTrue(chunk.chunk_id.startswith(f"{doc_info.document_id}_p"))

    # --------------------------------------------------------------------------
    # 4. Handling pages with no extractable text without crashing
    # --------------------------------------------------------------------------
    def test_07_page_with_no_extractable_text(self):
        """Test that documents with blank or image-only non-text pages do not crash."""
        multipage_with_blank = build_synthetic_multipage_pdf(num_text_pages=2, include_blank_page=True)
        doc_info = self.loader.process_and_save(multipage_with_blank, "has_blank_page.pdf")
        
        # 2 text pages + 1 blank page = 3 pages total
        self.assertEqual(doc_info.page_count, 3)
        extracted = self.extractor.extract_document(doc_info)
        
        self.assertEqual(extracted.total_pages, 3)
        self.assertEqual(len(extracted.pages), 3)

        # Page 1: has text
        self.assertTrue(extracted.pages[0].has_text)
        self.assertGreater(extracted.pages[0].word_count, 0)

        # Page 2: blank (no text)
        self.assertFalse(extracted.pages[1].has_text)
        self.assertEqual(extracted.pages[1].word_count, 0)
        self.assertEqual(extracted.pages[1].raw_text, "")

        # Page 3: has text
        self.assertTrue(extracted.pages[2].has_text)

        # Ensure no empty chunks were generated for page 2
        page_2_chunks = [c for c in extracted.chunks if c.page_number == 2]
        self.assertEqual(len(page_2_chunks), 0)

    # --------------------------------------------------------------------------
    # 5. Chunking behavior, overlap, and metadata preservation
    # --------------------------------------------------------------------------
    def test_08_chunking_and_metadata(self):
        """Test that long text is split into overlapping chunks with complete metadata."""
        long_paragraph = (
            "Evidence extraction is the foundational stage of multimodal question answering. "
            "In real-world enterprise documents, critical facts are frequently embedded inside "
            "dense balance sheet tables, line plots, and annotated engineering diagrams. "
            "A query-aware system dynamically evaluates the user's intent to identify whether "
            "the question requires numerical tabular reasoning, visual trend comparison, or "
            "traditional conceptual text matching. Reciprocal rank fusion subsequently unites "
            "these heterogeneous modalities into a coherent, cited evidence context bundle."
        ) * 4  # ~1600 characters

        chunks = self.extractor.chunk_page_text(
            text=long_paragraph,
            document_id="doc_test123",
            document_name="manual.pdf",
            page_number=5
        )

        self.assertGreater(len(chunks), 1)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.document_id, "doc_test123")
            self.assertEqual(chunk.document_name, "manual.pdf")
            self.assertEqual(chunk.page_number, 5)
            self.assertEqual(chunk.chunk_id, f"doc_test123_p5_c{i + 1}")
            self.assertEqual(chunk.modality, ModalityType.TEXT)
            self.assertIn("char_count", chunk.metadata)
            self.assertIn("word_count", chunk.metadata)
            self.assertIn("start_char", chunk.metadata)
            self.assertIn("end_char", chunk.metadata)
            self.assertTrue(chunk.citation_label.startswith("[Page 5, Chunk"))

    # --------------------------------------------------------------------------
    # 6. Safe file handling and duplicate filename resolution
    # --------------------------------------------------------------------------
    def test_09_identical_duplicate_detection(self):
        """Test uploading identical file content under the same name detects duplicate."""
        doc1 = self.loader.process_and_save(VALID_PDF_BYTES, "report.pdf")
        self.assertFalse(doc1.is_duplicate)

        doc2 = self.loader.process_and_save(VALID_PDF_BYTES, "report.pdf")
        self.assertTrue(doc2.is_duplicate)
        self.assertEqual(doc1.file_path, doc2.file_path)

    def test_10_different_content_same_name_does_not_overwrite(self):
        """Test uploading different content with the same filename prevents silent overwrite."""
        content_a = VALID_PDF_BYTES
        content_b = generate_clean_pdf_bytes("Alternate document text for financial audit")

        doc_a = self.loader.process_and_save(content_a, "quarterly_results.pdf")
        doc_b = self.loader.process_and_save(content_b, "quarterly_results.pdf")

        self.assertNotEqual(doc_a.file_path, doc_b.file_path)
        self.assertTrue(doc_a.file_path.exists())
        self.assertTrue(doc_b.file_path.exists())
        # Verify content was preserved
        self.assertEqual(doc_a.file_path.read_bytes(), content_a)
        self.assertEqual(doc_b.file_path.read_bytes(), content_b)


if __name__ == "__main__":
    unittest.main()
