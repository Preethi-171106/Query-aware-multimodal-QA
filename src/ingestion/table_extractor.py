"""Structured Table Extraction Pipeline.

Extracts tables from PDFs page-by-page using pdfplumber, normalizes cell content,
converts tables to Markdown and structured grid formats, and creates EvidenceChunk
objects with modality=TABLE.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pdfplumber
from src.indexing.schema import EvidenceChunk, ModalityType


class TableExtractor:
    """Detects, parses, and serializes structured tabular data from PDF documents."""

    def __init__(self, min_rows: int = 2, min_cols: int = 2):
        """
        Args:
            min_rows: Minimum rows required to consider a structure a valid table.
            min_cols: Minimum columns required to prevent single-column paragraph misclassification.
        """
        self.min_rows = min_rows
        self.min_cols = min_cols

    @staticmethod
    def normalize_cell_value(val: Optional[Any]) -> str:
        """Normalize cell whitespace and sanitize pipe characters without modifying data values."""
        if val is None:
            return ""
        # Convert to string and replace newlines/tabs with space
        cleaned = str(val).replace("\r", " ").replace("\n", " ").strip()
        # Collapse multiple spaces into single space
        cleaned = " ".join(cleaned.split())
        # Escape pipe characters so markdown table rendering remains intact
        cleaned = cleaned.replace("|", "\\|")
        return cleaned

    def table_to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert headers and rows into a clean, GitHub-flavored Markdown table string."""
        if not headers and not rows:
            return ""

        # Determine column count
        col_count = len(headers) if headers else (len(rows[0]) if rows else 0)
        if col_count == 0:
            return ""

        # Ensure headers exist
        if not headers or all(h == "" for h in headers):
            headers = [f"Col {i + 1}" for i in range(col_count)]

        # Pad headers if needed
        if len(headers) < col_count:
            headers = headers + [f"Col {i + 1}" for i in range(len(headers), col_count)]

        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * col_count) + " |"
        
        row_lines = []
        for row in rows:
            padded_row = row + [""] * (col_count - len(row))
            row_lines.append("| " + " | ".join(padded_row[:col_count]) + " |")

        return "\n".join([header_line, separator_line] + row_lines)

    def extract_tables_from_page(
        self,
        page: Any,
        page_number: int,
        document_id: str,
        document_name: str
    ) -> List[EvidenceChunk]:
        """Extract all valid tables from a single pdfplumber page object."""
        evidence_chunks: List[EvidenceChunk] = []

        try:
            # pdfplumber find_tables provides bounding boxes and table objects
            table_objects = page.find_tables()
        except Exception:
            table_objects = []

        table_idx = 1
        for t_obj in table_objects:
            try:
                raw_table = t_obj.extract()
                if not raw_table or len(raw_table) < self.min_rows:
                    continue

                # Normalize rows
                normalized_rows: List[List[str]] = []
                for raw_row in raw_table:
                    if raw_row is None:
                        continue
                    normalized_rows.append([self.normalize_cell_value(cell) for cell in raw_row])

                if len(normalized_rows) < self.min_rows:
                    continue

                # Determine maximum column width
                max_cols = max(len(r) for r in normalized_rows)
                if max_cols < self.min_cols:
                    # Reject single-column structures (often ordinary bullet points or paragraphs)
                    continue

                # Check if entire table is empty
                total_text_chars = sum(len(cell) for row in normalized_rows for cell in row)
                if total_text_chars == 0:
                    continue

                # Assume first row is header, remaining are data rows
                headers = normalized_rows[0]
                data_rows = normalized_rows[1:]

                # Generate Markdown representation
                markdown_content = self.table_to_markdown(headers, data_rows)

                chunk_id = f"{document_id}_p{page_number}_t{table_idx}"
                bbox = getattr(t_obj, "bbox", None)

                chunk = EvidenceChunk(
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    modality=ModalityType.TABLE,
                    content=markdown_content,
                    table_data={
                        "headers": headers,
                        "rows": data_rows,
                        "row_count": len(normalized_rows),
                        "col_count": max_cols,
                        "bbox": list(bbox) if bbox else None,
                    },
                    metadata={
                        "table_index": table_idx,
                        "row_count": len(normalized_rows),
                        "col_count": max_cols,
                        "headers": headers,
                        "has_content": True
                    }
                )
                evidence_chunks.append(chunk)
                table_idx += 1

            except Exception:
                # Skip malformed table safely without crashing document extraction
                continue

        return evidence_chunks

    def extract_document_tables(
        self,
        file_path: Path,
        document_id: str,
        document_name: str
    ) -> List[EvidenceChunk]:
        """Extract all tables across all pages of a PDF document."""
        table_chunks: List[EvidenceChunk] = []
        path_obj = Path(file_path)

        try:
            with pdfplumber.open(path_obj) as pdf:
                for idx, page in enumerate(pdf.pages):
                    page_num = idx + 1
                    page_tables = self.extract_tables_from_page(
                        page=page,
                        page_number=page_num,
                        document_id=document_id,
                        document_name=document_name
                    )
                    table_chunks.extend(page_tables)
        except Exception:
            # If pdfplumber fails to open or parse tables, return empty list safely
            pass

        return table_chunks
