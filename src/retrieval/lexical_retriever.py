"""Lexical Keyword Retrieval using BM25Okapi.

Indexes text, tables, and figure captions preserving technical terms,
acronyms, and numerical quantities, providing exact lexical search.
"""

import re
from typing import Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi

from src.indexing.schema import EvidenceChunk, ModalityType, RetrievalResult


class LexicalRetriever:
    """BM25-based keyword retriever for exact matching of technical terms and numbers."""

    # Tokenizer pattern preserving acronyms, decimals, hyphens, and alphanumeric tokens
    TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*')

    def __init__(self):
        self.chunks: List[EvidenceChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_id_to_idx: Dict[str, int] = {}

    @property
    def total_documents(self) -> int:
        """Total number of documents currently indexed."""
        return len(self.chunks)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text while preserving acronyms, numbers, and technical terms."""
        if not text:
            return []
        matches = self.TOKEN_PATTERN.findall(text.lower())
        return [m for m in matches if len(m) > 0]

    def clear(self) -> None:
        """Clear the lexical index."""
        self.chunks.clear()
        self.tokenized_corpus.clear()
        self._bm25 = None
        self._chunk_id_to_idx.clear()

    def index_chunks(self, chunks: List[EvidenceChunk]) -> int:
        """Index a collection of evidence chunks for lexical search.

        Extracts searchable text from text chunks, markdown tables,
        and figure captions.
        """
        if not chunks:
            return 0

        self.clear()
        for chunk in chunks:
            searchable_text = chunk.get_searchable_text()
            tokens = self.tokenize(searchable_text)
            # Ensure even sparse chunks have a minimal token representation
            if not tokens:
                tokens = [chunk.modality.value.lower()]

            pos = len(self.chunks)
            self.chunks.append(chunk)
            self.tokenized_corpus.append(tokens)
            self._chunk_id_to_idx[chunk.chunk_id] = pos

        if self.tokenized_corpus:
            self._bm25 = BM25Okapi(self.tokenized_corpus)

        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        modality_filter: Optional[ModalityType] = None
    ) -> List[RetrievalResult]:
        """Perform BM25 lexical search for a user query.

        Args:
            query: User search query.
            top_k: Number of top results to return.
            modality_filter: Optional filter to restrict by evidence modality.

        Returns:
            List of RetrievalResult objects ranked by BM25 score.
        """
        if not query or not query.strip() or self._bm25 is None or self.total_documents == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Compute raw BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # Sort indices by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[RetrievalResult] = []
        rank = 1
        for idx in ranked_indices:
            score = float(scores[idx])
            chunk = self.chunks[idx]

            # Modality filter
            if modality_filter and chunk.modality != modality_filter:
                continue

            # Only return chunks that have at least some lexical overlap with the query
            token_set = set(self.tokenized_corpus[idx])
            has_overlap = any(q_tok in token_set for q_tok in query_tokens)
            if not has_overlap and score <= 0:
                continue

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retrieval_method="lexical",
                    lexical_rank=rank,
                    lexical_score=score
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results
