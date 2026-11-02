"""PDF Loading, Validation, and Storage Pipeline.

Responsible for safely accepting, validating, hashing, and storing
uploaded PDF documents while preventing silent file overwrites.
"""

import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union
import pypdf
from src.config import AppConfig


class PDFLoaderError(Exception):
    """Base exception for PDF loading errors."""
    pass


class InvalidPDFFormatError(PDFLoaderError):
    """Raised when the uploaded file is not a valid PDF."""
    pass


class EmptyPDFError(PDFLoaderError):
    """Raised when the uploaded file is empty (0 bytes or 0 pages)."""
    pass


class CorruptedPDFError(PDFLoaderError):
    """Raised when the PDF file structure is corrupted or unreadable."""
    pass


@dataclass
class DocumentInfo:
    """Metadata describing a validated and stored PDF document."""
    document_id: str
    file_name: str
    file_path: Path
    file_size_bytes: int
    page_count: int
    content_hash: str
    is_valid: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    is_duplicate: bool = False

    @property
    def file_size_kb(self) -> float:
        return round(self.file_size_bytes / 1024, 2)


class PDFLoader:
    """Validates, hashes, stores, and extracts metadata from PDF files."""

    PDF_MAGIC_BYTES = b"%PDF-"
    HEADER_CHECK_LIMIT = 1024  # Look for magic bytes within the first 1KB

    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = Path(upload_dir) if upload_dir else AppConfig.UPLOADS_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_bytes(self, file_bytes: bytes, file_name: str = "document.pdf") -> None:
        """Validate raw bytes to ensure they represent a non-empty, genuine PDF.

        Raises:
            EmptyPDFError: If bytes are empty.
            InvalidPDFFormatError: If file lacks PDF extension or magic bytes.
            CorruptedPDFError: If PDF structure cannot be parsed.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise EmptyPDFError(f"The file '{file_name}' is empty (0 bytes).")

        # Check extension
        if not file_name.lower().endswith(".pdf"):
            raise InvalidPDFFormatError(
                f"Invalid file extension for '{file_name}'. Only '.pdf' files are supported."
            )

        # Check header magic bytes
        header_sample = file_bytes[:self.HEADER_CHECK_LIMIT]
        if self.PDF_MAGIC_BYTES not in header_sample:
            raise InvalidPDFFormatError(
                f"The file '{file_name}' does not contain a valid PDF file header."
            )

        # Validate readability with pypdf
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            if len(reader.pages) == 0:
                raise EmptyPDFError(f"The PDF '{file_name}' contains 0 pages.")
        except (EmptyPDFError, InvalidPDFFormatError):
            raise
        except Exception as e:
            raise CorruptedPDFError(
                f"The PDF '{file_name}' appears to be corrupted or encrypted: {str(e)}"
            )

    def compute_hash(self, file_bytes: bytes) -> str:
        """Compute SHA-256 hash of file contents for deterministic identification."""
        return hashlib.sha256(file_bytes).hexdigest()

    def resolve_destination_path(self, target_name: str, content_hash: str) -> Tuple[Path, bool]:
        """Resolve a safe destination path to prevent silent overwrites.

        Returns:
            Tuple of (destination_path, is_identical_duplicate)
        """
        clean_name = Path(target_name).name
        target_path = self.upload_dir / clean_name

        if not target_path.exists():
            return target_path, False

        # If file exists, check whether content is identical
        try:
            with open(target_path, "rb") as existing_file:
                existing_bytes = existing_file.read()
                existing_hash = self.compute_hash(existing_bytes)
                if existing_hash == content_hash:
                    return target_path, True  # Exact duplicate already stored
        except Exception:
            pass

        # If different content with identical name, create collision-free name
        stem = target_path.stem
        suffix = target_path.suffix
        short_hash = content_hash[:8]
        new_path = self.upload_dir / f"{stem}_{short_hash}{suffix}"
        
        counter = 1
        while new_path.exists():
            new_path = self.upload_dir / f"{stem}_{short_hash}_{counter}{suffix}"
            counter += 1

        return new_path, False

    def extract_pdf_metadata(self, file_path_or_bytes: Union[Path, bytes]) -> Tuple[int, Dict[str, Any]]:
        """Extract page count and standard PDF metadata dictionary."""
        if isinstance(file_path_or_bytes, bytes):
            stream = io.BytesIO(file_path_or_bytes)
        else:
            stream = open(file_path_or_bytes, "rb")

        try:
            reader = pypdf.PdfReader(stream)
            page_count = len(reader.pages)
            raw_meta = reader.metadata or {}
            cleaned_meta = {}
            for k, v in raw_meta.items():
                clean_key = k.lstrip("/").lower()
                cleaned_meta[clean_key] = str(v)
            return page_count, cleaned_meta
        finally:
            if not isinstance(file_path_or_bytes, bytes):
                stream.close()

    def process_and_save(
        self,
        file_input: Union[bytes, BinaryIO, Path],
        file_name: Optional[str] = None
    ) -> DocumentInfo:
        """Validate, hash, save, and extract info for an uploaded PDF.

        Args:
            file_input: Raw bytes, file-like stream (e.g. Streamlit UploadedFile), or Path.
            file_name: Optional explicit file name.

        Returns:
            DocumentInfo describing the validated document.
        """
        # Resolve raw bytes and filename
        if isinstance(file_input, Path):
            path_obj = Path(file_input)
            name = file_name or path_obj.name
            with open(path_obj, "rb") as f:
                file_bytes = f.read()
        elif isinstance(file_input, bytes):
            name = file_name or "document.pdf"
            file_bytes = file_input
        elif hasattr(file_input, "read"):
            name = file_name or getattr(file_input, "name", "document.pdf")
            if hasattr(file_input, "seek"):
                file_input.seek(0)
            file_bytes = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(0)
        else:
            raise PDFValidationError(f"Unsupported file input type: {type(file_input)}")

        # 1. Validate
        self.validate_bytes(file_bytes, name)

        # 2. Compute Content Hash and Document ID
        content_hash = self.compute_hash(file_bytes)
        document_id = f"doc_{content_hash[:12]}"

        # 3. Resolve Safe File Path (prevent silent overwrite)
        save_path, is_duplicate = self.resolve_destination_path(name, content_hash)
        if not is_duplicate or not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(file_bytes)

        # 4. Extract Page Count & Metadata
        page_count, metadata = self.extract_pdf_metadata(file_bytes)

        return DocumentInfo(
            document_id=document_id,
            file_name=save_path.name,
            file_path=save_path,
            file_size_bytes=len(file_bytes),
            page_count=page_count,
            content_hash=content_hash,
            is_valid=True,
            metadata=metadata,
            is_duplicate=is_duplicate
        )
