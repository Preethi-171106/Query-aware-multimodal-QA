"""Focused Unit and Integration Tests for Phase 4 Hybrid Retrieval.

Tests:
1. Text and table embedding mapping alignment (exact 1-to-1 correspondence).
2. Missing, empty, or unreadable figure files handling.
3. Stable evidence-ID mapping in FAISS vector store.
4. Empty index and empty query safety across dense, lexical, and hybrid search.
5. BM25 exact matching for technical acronyms and numbers (EBITDA, Q3, 0.95, F1-Score).
6. Reciprocal Rank Fusion (RRF) scoring, ranking, and deduplication.
7. Metadata and 1-indexed page-citation preservation.
8. Backward compatibility with Phase 2 and Phase 3 extraction pipelines.

Uses deterministic mock embedding mode (mock_dim=64) to execute offline
without requiring large model weights or external network access.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.indexing.embedding_manager import EmbeddingManager
from src.indexing.figure_representation import FigureRepresentationStrategy
from src.indexing.schema import EvidenceChunk, ModalityType, RetrievalResult
from src.indexing.vector_store import FAISSVectorStore
from src.retrieval.lexical_retriever import LexicalRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


class TestHybridRetrievalPipeline(unittest.TestCase):
    """Test suite for embedding management, FAISS vector store, BM25, and RRF."""

    def setUp(self):
        """Set up deterministic test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_phase4_retrieval_"))
        self.mock_dim = 64
        self.embed_mgr = EmbeddingManager(mock_dim=self.mock_dim)
        self.vector_store = FAISSVectorStore(dimension=self.mock_dim)
        self.lexical_retriever = LexicalRetriever()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            lexical_retriever=self.lexical_retriever,
            embedding_manager=self.embed_mgr,
            rrf_k=60
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Text and table embedding mapping alignment
    # --------------------------------------------------------------------------
    def test_01_embedding_mapping_alignment(self):
        """Verify that embedding matrix rows align 1-to-1 with evidence chunks."""
        chunks = [
            EvidenceChunk(
                document_id="doc_1",
                document_name="report.pdf",
                page_number=1,
                chunk_id="doc_1_p1_c1",
                modality=ModalityType.TEXT,
                content="Deep neural networks achieve high generalization."
            ),
            EvidenceChunk(
                document_id="doc_1",
                document_name="report.pdf",
                page_number=2,
                chunk_id="doc_1_p2_t1",
                modality=ModalityType.TABLE,
                content="| Model | Accuracy |\n| --- | --- |\n| ResNet | 94.2% |",
                table_data={"headers": ["Model", "Accuracy"], "rows": [["ResNet", "94.2%"]]}
            ),
            EvidenceChunk(
                document_id="doc_1",
                document_name="report.pdf",
                page_number=3,
                chunk_id="doc_1_p3_fig1",
                modality=ModalityType.FIGURE,
                content="Training loss curve over 100 epochs",
                metadata={"caption": "Figure 1: Training convergence curve"}
            ),
        ]

        embeddings, valid_chunks = self.embed_mgr.embed_chunks(chunks)

        self.assertEqual(len(valid_chunks), 3)
        self.assertEqual(embeddings.shape, (3, self.mock_dim))
        
        # Verify L2 normalization
        for i in range(3):
            norm = np.linalg.norm(embeddings[i])
            self.assertAlmostEqual(norm, 1.0, places=5)
            self.assertEqual(valid_chunks[i].chunk_id, chunks[i].chunk_id)

    # --------------------------------------------------------------------------
    # 2. Missing or empty figure files handling
    # --------------------------------------------------------------------------
    def test_02_missing_or_empty_figure_file(self):
        """Verify missing image file gracefully falls back to caption text embedding."""
        figure_chunk = EvidenceChunk(
            document_id="doc_1",
            document_name="report.pdf",
            page_number=4,
            chunk_id="doc_1_p4_fig1",
            modality=ModalityType.FIGURE,
            content="Architecture diagram",
            image_path=str(self.test_dir / "non_existent_figure.png"),
            metadata={"caption": "Figure 2: System block diagram"}
        )

        embeddings, valid_chunks = self.embed_mgr.embed_chunks([figure_chunk])

        self.assertEqual(len(valid_chunks), 1)
        self.assertEqual(embeddings.shape, (1, self.mock_dim))
        prov = valid_chunks[0].metadata.get("representation_provenance", {})
        self.assertEqual(prov.get("representation_type"), "caption_fallback")
        self.assertIn("Figure 2: System block diagram", prov.get("search_text_used", ""))

    # --------------------------------------------------------------------------
    # 3. Stable evidence-ID mapping in FAISS
    # --------------------------------------------------------------------------
    def test_03_stable_evidence_id_mapping_faiss(self):
        """Verify vector index maintains stable EvidenceChunk ID mappings and search."""
        chunk_a = EvidenceChunk(
            document_id="doc_1", document_name="doc.pdf", page_number=1,
            chunk_id="chunk_alpha", modality=ModalityType.TEXT, content="Alpha content"
        )
        chunk_b = EvidenceChunk(
            document_id="doc_1", document_name="doc.pdf", page_number=2,
            chunk_id="chunk_beta", modality=ModalityType.TEXT, content="Beta content"
        )

        vec_a = self.embed_mgr.embed_text("Alpha query")
        vec_b = self.embed_mgr.embed_text("Beta query")
        add_mat = np.vstack([vec_a, vec_b])

        added = self.vector_store.add_chunks([chunk_a, chunk_b], add_mat)
        self.assertEqual(added, 2)
        self.assertEqual(self.vector_store.total_vectors, 2)

        # Search nearest to vec_a
        results = self.vector_store.search(vec_a, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "chunk_alpha")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)

        # Test index persistence roundtrip
        index_file, meta_file = self.vector_store.save(self.test_dir, "test_idx")
        self.assertTrue(index_file.exists())
        self.assertTrue(meta_file.exists())

        new_store = FAISSVectorStore(dimension=self.mock_dim)
        success = new_store.load(self.test_dir, "test_idx")
        self.assertTrue(success)
        self.assertEqual(new_store.total_vectors, 2)
        loaded_res = new_store.search(vec_a, top_k=1)
        self.assertEqual(loaded_res[0].chunk.chunk_id, "chunk_alpha")

    # --------------------------------------------------------------------------
    # 4. Empty index and empty query safety
    # --------------------------------------------------------------------------
    def test_04_empty_index_and_empty_query(self):
        """Verify that empty indices and blank queries return empty lists without crashing."""
        # Empty FAISS store
        self.assertEqual(self.vector_store.search(np.zeros(self.mock_dim)), [])
        # Empty Lexical retriever
        self.assertEqual(self.lexical_retriever.search("test"), [])
        # Empty Hybrid retriever
        self.assertEqual(self.hybrid_retriever.search(""), [])
        self.assertEqual(self.hybrid_retriever.search("   "), [])
        self.assertEqual(self.hybrid_retriever.search("sample query"), [])

    # --------------------------------------------------------------------------
    # 5. BM25 acronym and numeric retrieval
    # --------------------------------------------------------------------------
    def test_05_bm25_acronym_and_numeric_retrieval(self):
        """Verify BM25 accurately matches exact financial acronyms and numerical values."""
        chunks = [
            EvidenceChunk(
                document_id="doc_fin", document_name="annual_report.pdf", page_number=5,
                chunk_id="c_ebitda", modality=ModalityType.TEXT,
                content="Operating revenue increased while EBITDA rose by 18.5% in Q3 2024."
            ),
            EvidenceChunk(
                document_id="doc_fin", document_name="annual_report.pdf", page_number=6,
                chunk_id="c_ai", modality=ModalityType.TEXT,
                content="Our cloud infrastructure migrated to enterprise Kubernetes clusters."
            ),
            EvidenceChunk(
                document_id="doc_fin", document_name="annual_report.pdf", page_number=7,
                chunk_id="c_table", modality=ModalityType.TABLE,
                content="| Year | F1-Score |\n| --- | --- |\n| 2024 | 0.95 |",
                table_data={"headers": ["Year", "F1-Score"], "rows": [["2024", "0.95"]]}
            )
        ]

        self.lexical_retriever.index_chunks(chunks)
        self.assertEqual(self.lexical_retriever.total_documents, 3)

        # Test acronym and numerical match
        res_ebitda = self.lexical_retriever.search("EBITDA Q3 2024", top_k=1)
        self.assertEqual(len(res_ebitda), 1)
        self.assertEqual(res_ebitda[0].chunk.chunk_id, "c_ebitda")

        # Test decimal and hyphenated metric match
        res_table = self.lexical_retriever.search("F1-Score 0.95", top_k=1)
        self.assertEqual(len(res_table), 1)
        self.assertEqual(res_table[0].chunk.chunk_id, "c_table")

    # --------------------------------------------------------------------------
    # 6. Reciprocal Rank Fusion (RRF) ranking and deduplication
    # --------------------------------------------------------------------------
    def test_06_rrf_ranking_and_deduplication(self):
        """Verify that RRF combines ranks from both dense and lexical channels without duplicates."""
        chunk_1 = EvidenceChunk(
            document_id="doc_1", document_name="r.pdf", page_number=1,
            chunk_id="c_both", modality=ModalityType.TEXT,
            content="Multimodal evidence fusion with cross-modal retrieval."
        )
        chunk_2 = EvidenceChunk(
            document_id="doc_1", document_name="r.pdf", page_number=2,
            chunk_id="c_dense_only", modality=ModalityType.TEXT,
            content="Comprehensive literature review on deep representations."
        )
        chunk_3 = EvidenceChunk(
            document_id="doc_1", document_name="r.pdf", page_number=3,
            chunk_id="c_lexical_only", modality=ModalityType.TEXT,
            content="Exact keyword match for specific identifier XYZ-9988."
        )

        all_chunks = [chunk_1, chunk_2, chunk_3]
        embs, valid = self.embed_mgr.embed_chunks(all_chunks)
        self.vector_store.add_chunks(valid, embs)
        self.lexical_retriever.index_chunks(all_chunks)

        results = self.hybrid_retriever.search("Multimodal evidence fusion XYZ-9988", top_k=3)
        self.assertGreater(len(results), 0)

        # Verify no duplicate chunk IDs
        retrieved_ids = [r.chunk.chunk_id for r in results]
        self.assertEqual(len(retrieved_ids), len(set(retrieved_ids)))

        # Verify component ranks are populated
        for res in results:
            self.assertIsNotNone(res.rrf_score)
            self.assertEqual(res.retrieval_method, "hybrid_rrf")

    # --------------------------------------------------------------------------
    # 7. Metadata and page-citation preservation
    # --------------------------------------------------------------------------
    def test_07_page_citation_preservation(self):
        """Verify that 1-indexed page citations and metadata are intact after retrieval."""
        table_chunk = EvidenceChunk(
            document_id="doc_fin", document_name="audit.pdf", page_number=12,
            chunk_id="doc_fin_p12_t2", modality=ModalityType.TABLE,
            content="| Col | Val |\n| --- | --- |\n| Tax | 12% |",
            metadata={"table_index": 2}
        )

        self.lexical_retriever.index_chunks([table_chunk])
        results = self.lexical_retriever.search("Tax 12%", top_k=1)

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res.chunk.citation_label, "[Page 12, Table 2]")
        self.assertEqual(res.chunk.page_number, 12)
        self.assertEqual(res.chunk.document_id, "doc_fin")

    # --------------------------------------------------------------------------
    # 8. Backward compatibility with Phase 2 and Phase 3
    # --------------------------------------------------------------------------
    def test_08_backward_compatibility(self):
        """Verify that ExtractedDocument from previous phases indexes cleanly into Phase 4."""
        from src.indexing.schema import ExtractedDocument, PageText

        page = PageText(page_number=1, raw_text="Digital text", char_count=12, word_count=2, has_text=True)
        chunk = EvidenceChunk(
            document_id="doc_compat", document_name="compat.pdf", page_number=1,
            chunk_id="doc_compat_p1_c1", modality=ModalityType.TEXT, content="Digital text chunk"
        )
        doc = ExtractedDocument(
            document_id="doc_compat", document_name="compat.pdf", total_pages=1,
            pages=[page], chunks=[chunk]
        )

        embs, valid = self.embed_mgr.embed_chunks(doc.chunks)
        self.assertEqual(len(valid), 1)
        self.vector_store.add_chunks(valid, embs)
        self.lexical_retriever.index_chunks(doc.chunks)

        hybrid_res = self.hybrid_retriever.search("Digital text chunk", top_k=1)
        self.assertEqual(len(hybrid_res), 1)
        self.assertEqual(hybrid_res[0].chunk.chunk_id, "doc_compat_p1_c1")


if __name__ == "__main__":
    unittest.main()
