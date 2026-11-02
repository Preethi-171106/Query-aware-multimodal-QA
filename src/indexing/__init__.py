"""Evidence Representation and Multimodal Indexing Module.

Responsibilities:
- Unified multimodal evidence schema (ModalityType, EvidenceChunk).
- Dense text embeddings (e.g., Sentence-Transformers).
- Visual embeddings (e.g., CLIP / OpenCLIP) and caption representations.
- Sparse keyword indexing (BM25) for lexical precision.
- Vector store integration (FAISS partitioned or tagged by modality).
"""

from src.indexing.schema import (
    ModalityType,
    EvidenceChunk,
    PageText,
    ExtractedDocument,
    RetrievalResult,
)
from src.indexing.figure_representation import FigureRepresentationStrategy
from src.indexing.embedding_manager import EmbeddingManager
from src.indexing.vector_store import FAISSVectorStore

__all__ = [
    "ModalityType",
    "EvidenceChunk",
    "PageText",
    "ExtractedDocument",
    "RetrievalResult",
    "FigureRepresentationStrategy",
    "EmbeddingManager",
    "FAISSVectorStore",
]
