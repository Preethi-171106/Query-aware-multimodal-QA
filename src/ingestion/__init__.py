"""Document Ingestion and Multi-Modality Extraction Module.

Responsibilities:
- PDF upload, validation, and layout parsing.
- Extraction of raw text segments and headings.
- Extraction and serialization of structured tables (Markdown / JSON).
- Extraction and cropping of visual elements (charts, plots, figures) with page numbers.
"""

from src.ingestion.pdf_loader import (
    PDFLoader,
    DocumentInfo,
    PDFLoaderError,
    InvalidPDFFormatError,
    EmptyPDFError,
    CorruptedPDFError,
)
from src.ingestion.text_extractor import TextExtractor
from src.ingestion.table_extractor import TableExtractor
from src.ingestion.visual_extractor import VisualExtractor

__all__ = [
    "PDFLoader",
    "DocumentInfo",
    "PDFLoaderError",
    "InvalidPDFFormatError",
    "EmptyPDFError",
    "CorruptedPDFError",
    "TextExtractor",
    "TableExtractor",
    "VisualExtractor",
]
