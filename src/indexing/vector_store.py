"""FAISS Dense Vector Store for Multimodal Evidence Retrieval.

Stores normalized dense vector embeddings with stable EvidenceChunk ID mappings,
supports cosine similarity search with modality filtering, and provides safe
JSON-based index persistence without pickle vulnerabilities.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import faiss
import numpy as np

from src.config import AppConfig
from src.indexing.schema import EvidenceChunk, ModalityType, RetrievalResult


class FAISSVectorStore:
    """In-memory and disk-persistent FAISS vector index with stable evidence mapping."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Use IndexFlatIP: on L2-normalized vectors, Inner Product == Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[EvidenceChunk] = []
        self._chunk_id_to_pos: Dict[str, int] = {}

    @property
    def total_vectors(self) -> int:
        """Total number of vectors indexed."""
        return self.index.ntotal if self.index else 0

    def clear(self) -> None:
        """Reset the vector index and mapping."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks.clear()
        self._chunk_id_to_pos.clear()

    def add_chunks(
        self,
        chunks: List[EvidenceChunk],
        embeddings: np.ndarray
    ) -> int:
        """Add evidence chunks and their corresponding normalized embeddings.

        Args:
            chunks: List of EvidenceChunks.
            embeddings: Float32 matrix of shape (len(chunks), dimension).

        Returns:
            Number of newly added chunks.
        """
        if not chunks or embeddings is None or len(embeddings) == 0:
            return 0

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Count mismatch: received {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimensionality mismatch: expected {self.dimension}, got {embeddings.shape[1]}."
            )

        # Filter out duplicates by chunk_id
        new_chunks = []
        new_vectors = []
        for i, chunk in enumerate(chunks):
            if chunk.chunk_id in self._chunk_id_to_pos:
                # Update existing chunk in metadata
                pos = self._chunk_id_to_pos[chunk.chunk_id]
                self.chunks[pos] = chunk
            else:
                pos = len(self.chunks)
                self.chunks.append(chunk)
                self._chunk_id_to_pos[chunk.chunk_id] = pos
                new_chunks.append(chunk)
                new_vectors.append(embeddings[i])

        if new_vectors:
            add_mat = np.vstack(new_vectors).astype(np.float32)
            self.index.add(add_mat)

        return len(new_chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        modality_filter: Optional[ModalityType] = None
    ) -> List[RetrievalResult]:
        """Perform cosine similarity search on the index.

        Args:
            query_vector: 1D normalized query embedding of shape (dimension,).
            top_k: Number of nearest items to retrieve.
            modality_filter: Optional filter to restrict search by modality.

        Returns:
            List of RetrievalResult objects ranked by similarity score.
        """
        if self.total_vectors == 0 or query_vector is None or len(query_vector) == 0:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if q_vec.shape[1] != self.dimension:
            return []

        # If modality filter is applied, request larger candidate pool to filter
        search_k = min(self.total_vectors, top_k * 4 if modality_filter else top_k)
        if search_k <= 0:
            return []

        scores, indices = self.index.search(q_vec, search_k)
        
        results: List[RetrievalResult] = []
        rank = 1
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            if modality_filter and chunk.modality != modality_filter:
                continue

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                    retrieval_method="dense",
                    dense_rank=rank,
                    dense_score=float(score)
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def save(self, storage_dir: Union[str, Path], prefix: str = "faiss_index") -> Tuple[Path, Path]:
        """Safely save FAISS binary index and JSON chunk metadata without pickle."""
        dir_path = Path(storage_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / f"{prefix}.bin"
        meta_file = dir_path / f"{prefix}_metadata.json"

        # 1. Write FAISS binary
        faiss.write_index(self.index, str(index_file.resolve()))

        # 2. Write Metadata safely via JSON
        serialized_chunks = [c.to_dict() for c in self.chunks]
        payload = {
            "dimension": self.dimension,
            "total_vectors": self.total_vectors,
            "chunks": serialized_chunks
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return index_file, meta_file

    def load(self, storage_dir: Union[str, Path], prefix: str = "faiss_index") -> bool:
        """Load index from storage safely."""
        dir_path = Path(storage_dir)
        index_file = dir_path / f"{prefix}.bin"
        meta_file = dir_path / f"{prefix}_metadata.json"

        if not index_file.exists() or not meta_file.exists():
            return False

        try:
            # 1. Read FAISS index
            self.index = faiss.read_index(str(index_file.resolve()))
            self.dimension = self.index.d

            # 2. Read Metadata JSON
            with open(meta_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.chunks = [EvidenceChunk.from_dict(d) for d in payload.get("chunks", [])]
            self._chunk_id_to_pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}
            return True
        except Exception:
            return False
