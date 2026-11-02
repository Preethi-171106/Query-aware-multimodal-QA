"""Multimodal Retrieval Module.

Responsibilities:
- Hybrid dense (vector similarity) and sparse (BM25 keyword) search.
- Modality-aware filtering and candidate candidate retrieval across text, tables, and figures.
"""

from src.retrieval.lexical_retriever import LexicalRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    "LexicalRetriever",
    "HybridRetriever",
]
