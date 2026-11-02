"""Hybrid Retrieval combining Dense FAISS Vectors and Sparse BM25 Lexical Search.

Implements Reciprocal Rank Fusion (RRF) to merge complementary semantic
and exact keyword matches while preserving transparent provenance and component ranks.
"""

from typing import Dict, List, Optional
from src.indexing.embedding_manager import EmbeddingManager
from src.indexing.schema import EvidenceChunk, ModalityType, RetrievalResult
from src.indexing.vector_store import FAISSVectorStore
from src.retrieval.lexical_retriever import LexicalRetriever


class HybridRetriever:
    """Orchestrates hybrid dense + lexical search using Reciprocal Rank Fusion (RRF)."""

    DEFAULT_RRF_K = 60

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        lexical_retriever: LexicalRetriever,
        embedding_manager: EmbeddingManager,
        rrf_k: int = DEFAULT_RRF_K
    ):
        self.vector_store = vector_store
        self.lexical_retriever = lexical_retriever
        self.embedding_manager = embedding_manager
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        modality_filter: Optional[ModalityType] = None,
        candidate_multiplier: int = 3
    ) -> List[RetrievalResult]:
        """Execute hybrid search using Reciprocal Rank Fusion.

        Args:
            query: User search query.
            top_k: Final number of ranked results to return.
            modality_filter: Optional filter to restrict by modality.
            candidate_multiplier: Multiplier for candidate pool per channel.

        Returns:
            List of deduplicated RetrievalResult objects ranked by RRF score.
        """
        if not query or not query.strip():
            return []

        search_k = max(top_k * candidate_multiplier, 10)

        # 1. Dense Semantic Retrieval
        dense_results: List[RetrievalResult] = []
        if self.vector_store.total_vectors > 0:
            query_vector = self.embedding_manager.embed_text(query)
            dense_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=search_k,
                modality_filter=modality_filter
            )

        # 2. Lexical BM25 Retrieval
        lexical_results: List[RetrievalResult] = []
        if self.lexical_retriever.total_documents > 0:
            lexical_results = self.lexical_retriever.search(
                query=query,
                top_k=search_k,
                modality_filter=modality_filter
            )

        # If both channels returned empty, return empty
        if not dense_results and not lexical_results:
            return []

        # 3. Reciprocal Rank Fusion (RRF)
        # Tracks accumulated RRF score and component metrics per unique chunk_id
        chunk_map: Dict[str, EvidenceChunk] = {}
        rrf_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        lexical_ranks: Dict[str, int] = {}
        lexical_scores: Dict[str, float] = {}

        # Accumulate Dense ranks
        for res in dense_results:
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            dense_ranks[cid] = res.rank
            dense_scores[cid] = res.score
            contribution = 1.0 / (self.rrf_k + res.rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contribution

        # Accumulate Lexical ranks
        for res in lexical_results:
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            lexical_ranks[cid] = res.rank
            lexical_scores[cid] = res.score
            contribution = 1.0 / (self.rrf_k + res.rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contribution

        # Sort combined results by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)

        final_results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            final_results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=rrf_scores[cid],
                    rank=rank,
                    retrieval_method="hybrid_rrf",
                    dense_rank=dense_ranks.get(cid),
                    dense_score=dense_scores.get(cid),
                    lexical_rank=lexical_ranks.get(cid),
                    lexical_score=lexical_scores.get(cid),
                    rrf_score=rrf_scores[cid]
                )
            )

        return final_results
