"""Query-Weighted Multimodal Evidence Fusion.

Combines dense and lexical retrieval results from multiple modalities using
query-modality affinity weights computed by the QueryAnalyzer.

Strategy: Modality-Weighted Reciprocal Rank Fusion (MW-RRF)
  - Run hybrid retrieval independently per active modality.
  - Scale each RRF contribution by the query's affinity weight for that modality.
  - Merge, deduplicate, and re-rank globally.

This is the core research contribution: evidence retrieved from text, tables,
and figures is reweighted by the query's own modality preference rather than
treating all modalities equally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.indexing.schema import EvidenceChunk, ModalityType, RetrievalResult
from src.indexing.vector_store import FAISSVectorStore
from src.indexing.embedding_manager import EmbeddingManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.lexical_retriever import LexicalRetriever
from src.query_understanding.query_analyzer import QueryAnalysisResult


# ──────────────────────────────────────────────────────────────────────────────
# Fusion result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FusedEvidenceResult:
    """Holds a ranked evidence chunk produced by the modality-weighted fusion pass."""
    chunk: EvidenceChunk
    fused_score: float          # Final weighted score after MW-RRF
    rank: int                   # 1-indexed global rank
    modality_weight: float      # Query-affinity weight for this chunk's modality
    source_results: List[RetrievalResult] = field(default_factory=list)

    @property
    def citation_label(self) -> str:
        return self.chunk.citation_label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "rank": self.rank,
            "fused_score": round(self.fused_score, 6),
            "modality": self.chunk.modality.value,
            "modality_weight": round(self.modality_weight, 4),
            "page_number": self.chunk.page_number,
            "citation_label": self.citation_label,
            "content_preview": self.chunk.content[:200] if self.chunk.content else "",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main Fuser
# ──────────────────────────────────────────────────────────────────────────────

class EvidenceFuser:
    """Performs query-modality-weighted evidence fusion across text, tables, and figures.

    Algorithm (MW-RRF):
    1. For each modality with weight > threshold, retrieve candidates using
       the shared HybridRetriever with a modality filter.
    2. Compute each chunk's contribution as:
           mw_rrf_score += modality_weight * (1 / (RRF_K + rank))
    3. Merge all modality-level results, aggregate by chunk_id, and sort globally.

    Args:
        vector_store:           Populated FAISS vector store.
        lexical_retriever:      Populated BM25 lexical retriever.
        embedding_manager:      Shared embedding model wrapper.
        rrf_k:                  RRF smoothing constant (default 60).
        modality_threshold:     Minimum query-affinity weight to activate per-modality
                                retrieval (avoids wasteful low-weight modality lookups).
        candidates_per_modality: Candidates retrieved per modality before global fusion.
    """

    DEFAULT_RRF_K = 60
    DEFAULT_MODALITY_THRESHOLD = 0.15
    DEFAULT_CANDIDATES_PER_MODALITY = 8

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        lexical_retriever: LexicalRetriever,
        embedding_manager: EmbeddingManager,
        rrf_k: int = DEFAULT_RRF_K,
        modality_threshold: float = DEFAULT_MODALITY_THRESHOLD,
        candidates_per_modality: int = DEFAULT_CANDIDATES_PER_MODALITY,
    ):
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            lexical_retriever=lexical_retriever,
            embedding_manager=embedding_manager,
            rrf_k=rrf_k,
        )
        self.rrf_k = rrf_k
        self.modality_threshold = modality_threshold
        self.candidates_per_modality = candidates_per_modality

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fuse(
        self,
        query: str,
        analysis: QueryAnalysisResult,
        top_k: int = 5,
    ) -> List[FusedEvidenceResult]:
        """Execute modality-weighted fusion and return the top-k fused results.

        Args:
            query:    Original user question.
            analysis: QueryAnalysisResult with modality weights and intent.
            top_k:    Final number of evidence items to return.

        Returns:
            Ranked list of FusedEvidenceResult objects.
        """
        if not query or not query.strip():
            return []

        # --- Step 1: Build per-modality retrieval candidates ---
        chunk_map: Dict[str, EvidenceChunk] = {}
        mw_rrf_scores: Dict[str, float] = {}
        modality_weight_map: Dict[str, float] = {}
        source_results_map: Dict[str, List[RetrievalResult]] = {}

        modality_to_filter = {
            ModalityType.TEXT: ModalityType.TEXT,
            ModalityType.TABLE: ModalityType.TABLE,
            ModalityType.FIGURE: ModalityType.FIGURE,
        }

        weights = analysis.modality_weights  # Dict[ModalityType, float]

        for modality, weight in weights.items():
            if weight < self.modality_threshold:
                continue

            mod_filter = modality_to_filter.get(modality)
            per_modality_results: List[RetrievalResult] = self.hybrid_retriever.search(
                query=query,
                top_k=self.candidates_per_modality,
                modality_filter=mod_filter,
            )

            for result in per_modality_results:
                cid = result.chunk.chunk_id
                chunk_map[cid] = result.chunk
                contribution = weight * (1.0 / (self.rrf_k + result.rank))
                mw_rrf_scores[cid] = mw_rrf_scores.get(cid, 0.0) + contribution

                current_w = modality_weight_map.get(cid, 0.0)
                modality_weight_map[cid] = max(current_w, weight)

                if cid not in source_results_map:
                    source_results_map[cid] = []
                source_results_map[cid].append(result)

        if not mw_rrf_scores:
            # Fallback: unfiltered hybrid retrieval when all weights below threshold
            fallback = self.hybrid_retriever.search(query=query, top_k=top_k)
            return self._wrap_as_fused(fallback, weights)

        # --- Step 2: Sort and truncate ---
        sorted_cids = sorted(mw_rrf_scores.keys(), key=lambda c: mw_rrf_scores[c], reverse=True)

        fused: List[FusedEvidenceResult] = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            fused.append(
                FusedEvidenceResult(
                    chunk=chunk_map[cid],
                    fused_score=mw_rrf_scores[cid],
                    rank=rank,
                    modality_weight=modality_weight_map.get(cid, 0.0),
                    source_results=source_results_map.get(cid, []),
                )
            )

        return fused

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _wrap_as_fused(
        self,
        results: List[RetrievalResult],
        weights: Dict[ModalityType, float],
    ) -> List[FusedEvidenceResult]:
        """Convert plain RetrievalResult list to FusedEvidenceResult (fallback path)."""
        fused = []
        for r in results:
            mod_weight = weights.get(r.chunk.modality, 0.33)
            fused.append(
                FusedEvidenceResult(
                    chunk=r.chunk,
                    fused_score=r.score,
                    rank=r.rank,
                    modality_weight=mod_weight,
                    source_results=[r],
                )
            )
        return fused
