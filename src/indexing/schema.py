"""Unified Evidence Representation Schema for Multimodal Document QA.

Defines the core data structures used across ingestion, indexing,
retrieval, fusion, and grounded answer generation.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ModalityType(str, Enum):
    """Supported evidence modality types."""
    TEXT = "TEXT"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    CHART = "CHART"


@dataclass
class EvidenceChunk:
    """Represents an atomic piece of multimodal evidence extracted from a document.

    Designed for seamless use across Phase 2 (Text), Phase 3 (Multimodal Indexing),
    Phase 4 (Query-Aware Fusion), and Phase 5 (Grounded Citations).
    """
    document_id: str
    document_name: str
    page_number: int  # 1-indexed for verifiable human and citation alignment
    chunk_id: str
    modality: ModalityType = ModalityType.TEXT
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Future-proofing for Phase 3: Tables and Figures
    image_path: Optional[str] = None
    table_data: Optional[Dict[str, Any]] = None

    @property
    def citation_label(self) -> str:
        """Generates a human-readable citation label for this evidence item."""
        if self.modality == ModalityType.TEXT:
            return f"[Page {self.page_number}, Chunk {self.chunk_id.split('_')[-1]}]"
        elif self.modality == ModalityType.TABLE:
            table_idx = self.metadata.get("table_index", 1)
            return f"[Page {self.page_number}, Table {table_idx}]"
        elif self.modality in (ModalityType.FIGURE, ModalityType.CHART):
            fig_idx = self.metadata.get("figure_index", 1)
            return f"[Page {self.page_number}, Figure {fig_idx}]"
        return f"[Page {self.page_number}]"

    def get_searchable_text(self) -> str:
        """Extract a clean textual representation for semantic embedding and lexical search.

        Ensures image paths are never embedded as content.
        """
        if self.modality == ModalityType.TEXT:
            return (self.content or "").strip()

        elif self.modality == ModalityType.TABLE:
            # Combine Markdown structure with linearized header/cell context
            parts = []
            if self.table_data and "headers" in self.table_data:
                headers = [h for h in self.table_data["headers"] if h]
                if headers:
                    parts.append(f"Table columns: {', '.join(headers)}")
            if self.content:
                parts.append(self.content.strip())
            return "\n".join(parts) if parts else (self.content or "").strip()

        elif self.modality in (ModalityType.FIGURE, ModalityType.CHART):
            # Figures use verified caption or descriptive text; never image file paths
            caption = self.metadata.get("caption")
            if caption and caption.strip():
                return f"Figure: {caption.strip()}"
            if self.content and not self.content.startswith("data/"):
                return self.content.strip()
            return f"Figure on Page {self.page_number}"

        return (self.content or "").strip()

    def has_searchable_content(self) -> bool:
        """Check whether this chunk has non-empty text, table, or caption content."""
        return len(self.get_searchable_text().strip()) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence chunk to a serializable dictionary."""
        d = asdict(self)
        d["modality"] = self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceChunk":
        """Create an EvidenceChunk from a dictionary."""
        data_copy = dict(data)
        if "modality" in data_copy and isinstance(data_copy["modality"], str):
            data_copy["modality"] = ModalityType(data_copy["modality"])
        return cls(**data_copy)


@dataclass
class RetrievalResult:
    """Represents a scored and ranked evidence chunk from a retrieval operation."""
    chunk: EvidenceChunk
    score: float
    rank: int
    retrieval_method: str = "hybrid_rrf"  # "dense", "lexical", "hybrid_rrf"
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    lexical_rank: Optional[int] = None
    lexical_score: Optional[float] = None
    rrf_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieval result to serializable dictionary with rounded scores."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": round(float(self.score), 6),
            "rank": self.rank,
            "retrieval_method": self.retrieval_method,
            "dense_rank": self.dense_rank,
            "dense_score": round(float(self.dense_score), 4) if self.dense_score is not None else None,
            "lexical_rank": self.lexical_rank,
            "lexical_score": round(float(self.lexical_score), 4) if self.lexical_score is not None else None,
            "rrf_score": round(float(self.rrf_score), 6) if self.rrf_score is not None else None,
        }


@dataclass
class PageText:
    """Represents the raw, unchunked extracted text of a single document page."""
    page_number: int  # 1-indexed
    raw_text: str
    char_count: int
    word_count: int
    has_text: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert page text object to a serializable dictionary."""
        return asdict(self)


@dataclass
class ExtractedDocument:
    """Encapsulates the complete extracted results for an uploaded document."""
    document_id: str
    document_name: str
    total_pages: int
    pages: List[PageText] = field(default_factory=list)
    chunks: List[EvidenceChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def text_chunks(self) -> List[EvidenceChunk]:
        """Return all text modality chunks."""
        return [c for c in self.chunks if c.modality == ModalityType.TEXT]

    @property
    def table_chunks(self) -> List[EvidenceChunk]:
        """Return all table modality chunks."""
        return [c for c in self.chunks if c.modality == ModalityType.TABLE]

    @property
    def figure_chunks(self) -> List[EvidenceChunk]:
        """Return all figure and chart modality chunks."""
        return [c for c in self.chunks if c.modality in (ModalityType.FIGURE, ModalityType.CHART)]

    @property
    def total_text_chunks(self) -> int:
        return len(self.text_chunks)

    @property
    def total_tables(self) -> int:
        return len(self.table_chunks)

    @property
    def total_figures(self) -> int:
        return len(self.figure_chunks)

    @property
    def pages_with_text_count(self) -> int:
        return sum(1 for p in self.pages if p.has_text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert extracted document to a serializable dictionary."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "total_text_chunks": self.total_text_chunks,
            "total_tables": self.total_tables,
            "total_figures": self.total_figures,
            "pages_with_text": self.pages_with_text_count,
            "metadata": self.metadata,
            "pages": [p.to_dict() for p in self.pages],
            "chunks": [c.to_dict() for c in self.chunks],
        }
