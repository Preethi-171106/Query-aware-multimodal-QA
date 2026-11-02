"""Visual Figure and Image Extraction Pipeline.

Extracts embedded image assets page-by-page from PDFs, saves cropped/extracted
figures under data/extracted/figures/, associates nearby captions (e.g., 'Figure 1'),
and generates EvidenceChunk objects with modality=FIGURE.
"""

import hashlib
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import pypdf
import pdfplumber

from src.config import AppConfig
from src.indexing.schema import EvidenceChunk, ModalityType, PageText


class VisualExtractor:
    """Extracts embedded figures and images from PDF pages and links nearby captions."""

    # Regex to detect figure/chart caption mentions on the source page
    CAPTION_PATTERN = re.compile(
        r'((?:Figure|Fig\.?|Chart|Diagram|Graph)\s*\d+[\s:.-][^\n\r]{3,120})',
        re.IGNORECASE
    )

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else AppConfig.FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_page_captions(self, page_text: str) -> List[str]:
        """Find potential figure or chart captions in page text without guessing."""
        if not page_text or not page_text.strip():
            return []

        matches = self.CAPTION_PATTERN.findall(page_text)
        cleaned_captions = []
        for m in matches:
            cleaned = " ".join(m.split()).strip()
            if len(cleaned) >= 5 and cleaned not in cleaned_captions:
                cleaned_captions.append(cleaned)
        return cleaned_captions

    def extract_page_figures(
        self,
        page_obj: Any,
        page_number: int,
        document_id: str,
        document_name: str,
        page_text: str = ""
    ) -> List[EvidenceChunk]:
        """Extract all embedded images from a single pypdf page object.

        Saves images to disk and generates EvidenceChunk instances.
        """
        evidence_chunks: List[EvidenceChunk] = []

        # Find potential captions on this page
        candidate_captions = self.find_page_captions(page_text)

        images = getattr(page_obj, "images", [])
        if not images:
            return []

        fig_idx = 1
        for img_item in images:
            try:
                # Extract image bytes and PIL Image
                pil_image: Optional[Image.Image] = getattr(img_item, "image", None)
                if pil_image is None and hasattr(img_item, "data"):
                    pil_image = Image.open(io.BytesIO(img_item.data))

                if pil_image is None:
                    continue

                # Filter out tiny icon artifacts (e.g. < 10x10 px)
                if pil_image.width < 10 or pil_image.height < 10:
                    continue

                # Compute content hash for collision-free file storage
                img_bytes_io = io.BytesIO()
                # Convert RGBA/P to RGB if saving to JPEG or ensure PNG format
                save_format = "PNG"
                pil_image.save(img_bytes_io, format=save_format)
                img_bytes = img_bytes_io.getvalue()
                short_hash = hashlib.sha256(img_bytes).hexdigest()[:8]

                # Safe file name: {doc_id}_p{page}_fig{index}_{hash}.png
                img_filename = f"{document_id}_p{page_number}_fig{fig_idx}_{short_hash}.png"
                img_path = self.output_dir / img_filename

                if not img_path.exists():
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                # Link caption when confident (e.g., 1 figure & 1 caption, or index match)
                associated_caption: Optional[str] = None
                if len(candidate_captions) == 1 and len(images) == 1:
                    associated_caption = candidate_captions[0]
                elif fig_idx <= len(candidate_captions):
                    # Check if candidate mentions this figure index
                    for cap in candidate_captions:
                        if re.search(rf'\b(?:Fig(?:ure)?\.?|Chart)\s*{fig_idx}\b', cap, re.IGNORECASE):
                            associated_caption = cap
                            break

                chunk_id = f"{document_id}_p{page_number}_fig{fig_idx}"
                content_desc = (
                    f"Figure on Page {page_number}: {associated_caption}"
                    if associated_caption
                    else f"Figure on Page {page_number} ({pil_image.width}x{pil_image.height}px)"
                )

                chunk = EvidenceChunk(
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    modality=ModalityType.FIGURE,
                    content=content_desc,
                    image_path=str(img_path.resolve()),
                    metadata={
                        "figure_index": fig_idx,
                        "image_filename": img_filename,
                        "image_path": str(img_path.resolve()),
                        "relative_image_path": f"data/extracted/figures/{img_filename}",
                        "width": pil_image.width,
                        "height": pil_image.height,
                        "format": save_format,
                        "file_size_bytes": len(img_bytes),
                        "caption": associated_caption,
                        "has_caption": bool(associated_caption),
                        "is_vector_chart": False,
                        "note": "Visual raster figure extracted from embedded PDF image stream."
                    }
                )
                evidence_chunks.append(chunk)
                fig_idx += 1

            except Exception:
                # Skip corrupt or unsupported image streams gracefully
                continue

        return evidence_chunks

    def extract_document_figures(
        self,
        file_path: Path,
        document_id: str,
        document_name: str,
        page_texts: Optional[List[PageText]] = None
    ) -> List[EvidenceChunk]:
        """Extract all embedded figures across all pages of a PDF document, including rasterizing pages with figure captions but no embedded image streams."""
        figure_chunks: List[EvidenceChunk] = []
        path_obj = Path(file_path)

        # Build a map of page number to raw text if provided
        page_text_map: Dict[int, str] = {}
        if page_texts:
            for pt in page_texts:
                page_text_map[pt.page_number] = pt.raw_text

        try:
            # Open both pypdf (for image streams) and pdfplumber (for raster rendering)
            with open(path_obj, "rb") as f_reader, pdfplumber.open(path_obj) as pdf_plumber:
                pypdf_reader = pypdf.PdfReader(f_reader)
                plumber_pages = pdf_plumber.pages
                for idx, page in enumerate(pypdf_reader.pages):
                    page_num = idx + 1
                    raw_text = page_text_map.get(page_num, "")
                    if not raw_text:
                        try:
                            raw_text = page.extract_text() or ""
                        except Exception:
                            raw_text = ""

                    # First, extract embedded image figures as before
                    page_figs = self.extract_page_figures(
                        page_obj=page,
                        page_number=page_num,
                        document_id=document_id,
                        document_name=document_name,
                        page_text=raw_text
                    )
                    figure_chunks.extend(page_figs)

                    # If no embedded images were found but the page text contains a figure caption, rasterize the whole page
                    if not page_figs and any(self.CAPTION_PATTERN.search(raw_text) for _ in [1]):
                        plumber_page = plumber_pages[idx] if idx < len(plumber_pages) else None
                        if plumber_page:
                            raster_chunks = self._rasterize_page_figure(
                                plumber_page=plumber_page,
                                page_number=page_num,
                                document_id=document_id,
                                document_name=document_name,
                                page_text=raw_text
                            )
                            figure_chunks.extend(raster_chunks)
        except Exception:
            # If PDF reading fails, return empty list safely
            pass

        return figure_chunks

    def _rasterize_page_figure(
        self,
        plumber_page: "pdfplumber.page.Page",
        page_number: int,
        document_id: str,
        document_name: str,
        page_text: str
    ) -> List[EvidenceChunk]:
        """Render the entire PDF page to a raster image when a figure caption is present but no embedded image exists.

        The rendered PNG is saved using the same naming scheme as embedded figures, and an EvidenceChunk is created
        with the detected caption (if any) and image metadata.
        """
        # Identify candidate captions on this page
        candidate_captions = self.find_page_captions(page_text)
        if not candidate_captions:
            return []

        # Render page to a high‑resolution PNG
        img = plumber_page.to_image(resolution=150)
        pil_image = img.original
        # Compute hash for stable filename
        img_bytes_io = io.BytesIO()
        pil_image.save(img_bytes_io, format="PNG")
        img_bytes = img_bytes_io.getvalue()
        short_hash = hashlib.sha256(img_bytes).hexdigest()[:8]
        # Use figure index 1 for rasterized pages (they are unique per page)
        img_filename = f"{document_id}_p{page_number}_fig1_{short_hash}.png"
        img_path = self.output_dir / img_filename
        if not img_path.exists():
            with open(img_path, "wb") as f:
                f.write(img_bytes)

        # Use the first detected caption as the associated caption
        associated_caption = candidate_captions[0] if candidate_captions else None
        chunk_id = f"{document_id}_p{page_number}_fig1"
        content_desc = (
            f"Figure on Page {page_number}: {associated_caption}" if associated_caption
            else f"Figure on Page {page_number} ({pil_image.width}x{pil_image.height}px)"
        )
        chunk = EvidenceChunk(
            document_id=document_id,
            document_name=document_name,
            page_number=page_number,
            chunk_id=chunk_id,
            modality=ModalityType.FIGURE,
            content=content_desc,
            image_path=str(img_path.resolve()),
            metadata={
                "figure_index": 1,
                "image_filename": img_filename,
                "image_path": str(img_path.resolve()),
                "relative_image_path": f"data/extracted/figures/{img_filename}",
                "width": pil_image.width,
                "height": pil_image.height,
                "format": "PNG",
                "file_size_bytes": len(img_bytes),
                "caption": associated_caption,
                "has_caption": bool(associated_caption),
                "is_vector_chart": True,
                "note": "Rasterized page captured because no embedded image stream was found.",
            },
        )
        return [chunk]

