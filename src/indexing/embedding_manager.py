"""Embedding Manager for Multimodal Evidence Representation.

Generates dense normalized vector representations for text, tables,
and figure captions using sentence-transformers (all-MiniLM-L6-v2)
and integrates modular figure representation strategies.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.config import AppConfig
from src.indexing.figure_representation import FigureRepresentationStrategy
from src.indexing.schema import EvidenceChunk, ModalityType


class EmbeddingManager:
    """Manages dense vector generation, normalization, and chunk-to-embedding mappings."""

    DEFAULT_DIMENSION = 384

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        figure_strategy: Optional[FigureRepresentationStrategy] = None,
        mock_dim: Optional[int] = None,
        custom_embed_fn: Optional[Callable[[Union[str, List[str]]], np.ndarray]] = None
    ):
        """
        Args:
            model_name: HuggingFace model name (defaults to AppConfig.TEXT_EMBEDDING_MODEL).
            device: 'cpu' or 'cuda' (defaults to AppConfig.DEVICE).
            figure_strategy: Strategy for handling figure representations.
            mock_dim: If set, operates in deterministic offline mock mode (for fast unit testing).
            custom_embed_fn: Optional custom callable to compute embeddings.
        """
        self.model_name = model_name or AppConfig.TEXT_EMBEDDING_MODEL
        self.device = device or AppConfig.DEVICE
        self.mock_dim = mock_dim
        self.custom_embed_fn = custom_embed_fn
        self.figure_strategy = figure_strategy or FigureRepresentationStrategy()
        
        self._model = None
        self._dimension = mock_dim or self.DEFAULT_DIMENSION

    @property
    def dimension(self) -> int:
        """Vector dimensionality of the embedding space."""
        return self._dimension

    def _get_model(self) -> Any:
        """Lazy loader for sentence-transformers model."""
        if self.custom_embed_fn or self.mock_dim:
            return None

        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize vector or matrix for exact cosine similarity in inner product indices."""
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim == 1:
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 1e-9 else vec
        elif vec.ndim == 2:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            return vec / norms
        return vec

    def _mock_encode(self, text: str) -> np.ndarray:
        """Generate a deterministic pseudo-random normalized vector for unit testing."""
        if not text or not text.strip():
            return np.zeros(self._dimension, dtype=np.float32)
        
        # Seed generator deterministically based on text content
        import hashlib
        h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(h)
        vec = rng.randn(self._dimension).astype(np.float32)
        return self._normalize(vec)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a single normalized 1D float32 vector for a text string."""
        if not text or not text.strip():
            return np.zeros(self._dimension, dtype=np.float32)

        if self.custom_embed_fn:
            raw = self.custom_embed_fn(text)
            return self._normalize(np.asarray(raw, dtype=np.float32).flatten())

        if self.mock_dim:
            return self._mock_encode(text)

        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate normalized 2D float32 embeddings for a batch of strings."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        if self.custom_embed_fn:
            raw = self.custom_embed_fn(texts)
            return self._normalize(np.asarray(raw, dtype=np.float32))

        if self.mock_dim:
            return np.vstack([self._mock_encode(t) for t in texts])

        # Filter empty strings safely
        cleaned = [t if t and t.strip() else " " for t in texts]
        model = self._get_model()
        vecs = model.encode(cleaned, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def embed_chunks(
        self,
        chunks: List[EvidenceChunk]
    ) -> Tuple[np.ndarray, List[EvidenceChunk]]:
        """Embed a list of evidence chunks with exact 1-to-1 alignment.

        Handles text, table markdown structures, and visual figures/captions.
        Never embeds image file paths as visual content.

        Returns:
            Tuple of (embeddings_matrix, valid_chunks_list)
        """
        valid_chunks: List[EvidenceChunk] = []
        vectors: List[np.ndarray] = []

        for chunk in chunks:
            try:
                if chunk.modality in (ModalityType.FIGURE, ModalityType.CHART):
                    # Route through modular figure strategy
                    vec, prov = self.figure_strategy.embed_figure(chunk, text_embed_fn=self.embed_text)
                    if vec is not None:
                        chunk.metadata["representation_provenance"] = prov
                        vectors.append(vec)
                        valid_chunks.append(chunk)
                else:
                    # Text and Table modalities
                    search_text = chunk.get_searchable_text()
                    if search_text:
                        vec = self.embed_text(search_text)
                        vectors.append(vec)
                        valid_chunks.append(chunk)
            except Exception:
                # Safely skip un-embeddable or corrupted chunk without crashing
                continue

        if not vectors:
            return np.empty((0, self._dimension), dtype=np.float32), []

        embeddings_matrix = np.vstack(vectors).astype(np.float32)
        return embeddings_matrix, valid_chunks
