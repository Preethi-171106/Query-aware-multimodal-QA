"""Page-Level Text Extraction and Evidence Chunking Pipeline.

Extracts text page-by-page using pdfplumber (with pypdf fallback),
preserves 1-indexed page citations, handles blank/image pages gracefully,
and splits text into overlapping EvidenceChunk representations.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pdfplumber
import pypdf
from src.config import AppConfig, _RAW_CONFIG
from src.indexing.schema import EvidenceChunk, ExtractedDocument, ModalityType, PageText
from src.ingestion.pdf_loader import DocumentInfo
from src.ingestion.table_extractor import TableExtractor
from src.ingestion.visual_extractor import VisualExtractor


class TextExtractor:
    """Extracts text page-by-page from PDFs and splits content into EvidenceChunks."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        # Read defaults from config or fallback
        idx_cfg = _RAW_CONFIG.get("indexing", {})
        self.chunk_size = chunk_size or idx_cfg.get("text_chunk_size", 500)
        self.chunk_overlap = chunk_overlap or idx_cfg.get("text_chunk_overlap", 50)
        
        # Ensure overlap is strictly less than chunk_size
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 4)

    def extract_raw_pages(self, file_path: Path) -> List[PageText]:
        """Extract raw text from each page separately using pdfplumber with pypdf fallback.

        Preserves 1-based page numbers and flags pages with no extractable text.
        """
        pages: List[PageText] = []
        path_obj = Path(file_path)

        # Primary extractor: pdfplumber with per-page pypdf fallback
        try:
            with pdfplumber.open(path_obj) as pdf:
                # Prepare pypdf reader for per-page fallback if needed
                pypdf_reader = None
                try:
                    pypdf_reader = pypdf.PdfReader(str(path_obj))
                except Exception:
                    pass

                for idx, page in enumerate(pdf.pages):
                    page_num = idx + 1
                    raw_text = page.extract_text() or ""
                    extractor_used = "pdfplumber"

                    # Per-page fallback if pdfplumber returned empty text
                    if not raw_text.strip() and pypdf_reader and idx < len(pypdf_reader.pages):
                        try:
                            pypdf_text = pypdf_reader.pages[idx].extract_text() or ""
                            if pypdf_text.strip():
                                raw_text = pypdf_text
                                extractor_used = "pypdf_fallback"
                        except Exception:
                            pass

                    cleaned_text = raw_text.strip()
                    has_text = len(cleaned_text) > 0
                    word_count = len(cleaned_text.split()) if has_text else 0
                    char_count = len(cleaned_text)

                    pages.append(
                        PageText(
                            page_number=page_num,
                            raw_text=cleaned_text,
                            char_count=char_count,
                            word_count=word_count,
                            has_text=has_text,
                            metadata={
                                "extractor": extractor_used,
                                "page_width": getattr(page, "width", None),
                                "page_height": getattr(page, "height", None),
                                "ocr_applied": False,
                                "note": "Direct digital text stream. OCR not active."
                            }
                        )
                    )
                return pages
        except Exception:
            # Complete document fallback to pypdf
            return self._extract_raw_pages_pypdf(path_obj)

    def _extract_raw_pages_pypdf(self, file_path: Path) -> List[PageText]:
        """Fallback extraction using pypdf if pdfplumber encounters layout/syntax issues."""
        pages: List[PageText] = []
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                try:
                    raw_text = page.extract_text() or ""
                except Exception:
                    raw_text = ""
                cleaned_text = raw_text.strip()
                has_text = len(cleaned_text) > 0
                word_count = len(cleaned_text.split()) if has_text else 0

                pages.append(
                    PageText(
                        page_number=page_num,
                        raw_text=cleaned_text,
                        char_count=len(cleaned_text),
                        word_count=word_count,
                        has_text=has_text,
                        metadata={
                            "extractor": "pypdf_fallback",
                            "ocr_applied": False,
                            "note": "Extracted via pypdf fallback. OCR not active."
                        }
                    )
                )
        return pages

    def chunk_page_text(
        self,
        text: str,
        document_id: str,
        document_name: str,
        page_number: int
    ) -> List[EvidenceChunk]:
        """Split a single page's text into overlapping EvidenceChunks.

        Attempts to break chunks on paragraph or sentence boundaries
        to preserve coherent semantic meaning.
        """
        if not text or not text.strip():
            return []

        cleaned_text = text.strip()
        
        # If the page text is already smaller than or equal to chunk_size, return 1 chunk
        if len(cleaned_text) <= self.chunk_size:
            chunk_id = f"{document_id}_p{page_number}_c1"
            return [
                EvidenceChunk(
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    modality=ModalityType.TEXT,
                    content=cleaned_text,
                    metadata={
                        "char_count": len(cleaned_text),
                        "word_count": len(cleaned_text.split()),
                        "start_char": 0,
                        "end_char": len(cleaned_text),
                        "is_complete_page": True
                    }
                )
            ]

        # Sentence/punctuation-aware sliding window
        chunks: List[EvidenceChunk] = []
        step = self.chunk_size - self.chunk_overlap
        chunk_idx = 1
        start_char = 0
        total_len = len(cleaned_text)

        while start_char < total_len:
            end_char = min(start_char + self.chunk_size, total_len)

            # Try to snap to the nearest sentence or word boundary if not at the very end
            if end_char < total_len:
                # Look for sentence boundary (.!?) within the last 40 chars
                boundary_match = None
                window = cleaned_text[start_char:end_char]
                for match in re.finditer(r'([.!?]\s+)', window):
                    boundary_match = match
                
                if boundary_match and boundary_match.end() > (self.chunk_size * 0.6):
                    end_char = start_char + boundary_match.end()
                else:
                    # Look for whitespace boundary
                    space_pos = cleaned_text.rfind(" ", start_char + int(self.chunk_size * 0.7), end_char)
                    if space_pos != -1:
                        end_char = space_pos + 1

            chunk_content = cleaned_text[start_char:end_char].strip()
            if chunk_content:
                chunk_id = f"{document_id}_p{page_number}_c{chunk_idx}"
                chunks.append(
                    EvidenceChunk(
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_number,
                        chunk_id=chunk_id,
                        modality=ModalityType.TEXT,
                        content=chunk_content,
                        metadata={
                            "char_count": len(chunk_content),
                            "word_count": len(chunk_content.split()),
                            "start_char": start_char,
                            "end_char": end_char,
                            "is_complete_page": False
                        }
                    )
                )
                chunk_idx += 1

            if end_char >= total_len:
                break
            start_char += step

        return chunks

    def extract_document(
        self,
        doc_info: DocumentInfo,
        extract_tables: bool = True,
        extract_figures: bool = True
    ) -> ExtractedDocument:
        """Execute end-to-end multimodal page extraction (Text, Tables, Figures)."""
        # 1. Page extraction (Text)
        pages = self.extract_raw_pages(doc_info.file_path)

        # 2. Text chunk generation across all pages
        all_chunks: List[EvidenceChunk] = []
        for page in pages:
            if page.has_text:
                page_chunks = self.chunk_page_text(
                    text=page.raw_text,
                    document_id=doc_info.document_id,
                    document_name=doc_info.file_name,
                    page_number=page.page_number
                )
                all_chunks.extend(page_chunks)

        # 3. Table extraction (ModalityType.TABLE)
        if extract_tables:
            table_extractor = TableExtractor()
            table_chunks = table_extractor.extract_document_tables(
                file_path=doc_info.file_path,
                document_id=doc_info.document_id,
                document_name=doc_info.file_name
            )
            all_chunks.extend(table_chunks)

        # 4. Figure & visual image extraction (ModalityType.FIGURE)
        if extract_figures:
            visual_extractor = VisualExtractor()
            figure_chunks = visual_extractor.extract_document_figures(
                file_path=doc_info.file_path,
                document_id=doc_info.document_id,
                document_name=doc_info.file_name,
                page_texts=pages
            )
            all_chunks.extend(figure_chunks)

        return ExtractedDocument(
            document_id=doc_info.document_id,
            document_name=doc_info.file_name,
            total_pages=len(pages),
            pages=pages,
            chunks=all_chunks,
            metadata={
                **doc_info.metadata,
                "file_path": str(doc_info.file_path),
                "file_size_bytes": doc_info.file_size_bytes,
                "content_hash": doc_info.content_hash,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "extract_tables": extract_tables,
                "extract_figures": extract_figures
            }
        )
