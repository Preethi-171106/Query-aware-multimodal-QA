"""Modular Figure Representation Strategy.

Handles visual feature extraction for extracted document figures and charts.
Implements image embedding via CLIP where available, with a clearly documented,
robust caption/text fallback strategy for missing images or offline environments.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from src.indexing.schema import EvidenceChunk, ModalityType


class FigureRepresentationStrategy:
    """Strategy for generating vector representations of figure/chart evidence."""

    def __init__(
        self,
        use_clip: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu"
    ):
        self.use_clip = use_clip
        self.clip_model_name = clip_model_name
        self.device = device
        self._clip_model = None
        self._clip_processor = None
        self._clip_initialized = False

    def is_clip_available(self) -> bool:
        """Check if CLIP dependencies and model weights are accessible."""
        if not self.use_clip:
            return False
        try:
            from transformers import CLIPModel, CLIPProcessor
            return True
        except ImportError:
            return False

    def _init_clip(self) -> bool:
        """Lazy initialization of CLIP model and processor."""
        if self._clip_initialized:
            return self._clip_model is not None
        
        self._clip_initialized = True
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self._clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
            self._clip_model.eval()
            return True
        except Exception:
            # Fall back to caption strategy if download fails or model cannot be loaded
            self._clip_model = None
            self._clip_processor = None
            return False

    def get_figure_search_text(self, chunk: EvidenceChunk) -> Tuple[str, str]:
        """Extract clean caption or metadata text for textual/fallback embedding.

        Returns:
            Tuple of (searchable_text, representation_type)
        """
        # Ensure we never embed raw file system paths
        caption = chunk.metadata.get("caption")
        if caption and caption.strip():
            return f"Figure: {caption.strip()}", "caption_text"

        content = (chunk.content or "").strip()
        if content and not content.startswith("data/") and not content.startswith("/") and "\\" not in content:
            return content, "caption_description"

        return f"Figure on Page {chunk.page_number}", "page_metadata_fallback"

    def embed_figure(
        self,
        chunk: EvidenceChunk,
        text_embed_fn: Optional[Any] = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Generate a vector representation for a figure chunk.

        If CLIP is enabled and image is readable, generates a visual embedding.
        Otherwise, falls back to caption text embedding via text_embed_fn.

        Returns:
            Tuple of (embedding_vector_or_none, provenance_metadata)
        """
        image_path_str = chunk.image_path or chunk.metadata.get("image_path")
        image_path = Path(image_path_str) if image_path_str else None

        # Check if CLIP visual embedding is possible
        if self.use_clip and self._init_clip() and image_path and image_path.exists():
            try:
                import torch
                with Image.open(image_path) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    inputs = self._clip_processor(images=rgb_img, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        image_features = self._clip_model.get_image_features(**inputs)
                        vec = image_features.cpu().numpy().flatten().astype(np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 1e-6:
                            vec = vec / norm
                        return vec, {
                            "representation_type": "clip_visual",
                            "model": self.clip_model_name,
                            "note": "Visual embedding computed from figure pixels via CLIP."
                        }
            except Exception as e:
                # Fall back to caption embedding if image read/process fails
                pass

        # Robust, clearly documented Caption / Text Metadata Fallback
        search_text, rep_type = self.get_figure_search_text(chunk)
        vec = None
        if text_embed_fn is not None and search_text:
            vec = text_embed_fn(search_text)

        metadata_provenance = {
            "representation_type": "caption_fallback",
            "search_text_used": search_text,
            "has_image_file": bool(image_path and image_path.exists()),
            "note": (
                "Caption/Text fallback embedding. Represents descriptive textual metadata only; "
                "does NOT understand chart pixels, geometric lines, or raster visual data."
            )
        }
        return vec, metadata_provenance
