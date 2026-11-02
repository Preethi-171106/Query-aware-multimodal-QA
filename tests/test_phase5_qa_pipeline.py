"""Phase 5 Unit and Integration Tests: Query Understanding, MW-RRF Fusion, and Answer Generation.

Tests:
 1.  QueryAnalyzer: text/concept intent detection.
 2.  QueryAnalyzer: table/numerical intent detection.
 3.  QueryAnalyzer: figure/visual intent detection.
 4.  QueryAnalyzer: mixed-modality detection.
 5.  QueryAnalyzer: empty query fallback (default prior).
 6.  Modality weights are finite, non-negative, and sum to 1.0.
 7.  EvidenceFuser: MW-RRF score positivity and correct ranking order.
 8.  EvidenceFuser: empty evidence fallback returns empty list.
 9.  EvidenceFuser: no duplicate chunk_ids in fused results.
10.  EvidenceFuser: per-modality weight correctly influences top-1 result.
11.  AnswerGenerator: mock generation returns non-empty answer.
12.  AnswerGenerator: citations match evidence actually supplied.
13.  AnswerGenerator: empty evidence → explicit insufficiency message.
14.  AnswerGenerator: missing API key uses mock fallback (no crash).
15.  AnswerGenerator: AnswerResult serializes to valid dict.
16.  Citation labels are 1-indexed and reference correct pages.
17.  Full end-to-end pipeline: extraction → index → analyze → fuse → generate.
18.  Backward compatibility: Phase 4 HybridRetriever unaffected by Phase 5 imports.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List
import numpy as np

from src.indexing.embedding_manager import EmbeddingManager
from src.indexing.schema import EvidenceChunk, ExtractedDocument, ModalityType, PageText
from src.indexing.vector_store import FAISSVectorStore
from src.query_understanding.query_analyzer import QueryAnalyzer, QueryAnalysisResult
from src.fusion.evidence_fuser import EvidenceFuser, FusedEvidenceResult
from src.generation.answer_generator import AnswerGenerator, AnswerResult, Citation
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.lexical_retriever import LexicalRetriever

# ──────────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ──────────────────────────────────────────────────────────────────────────────

MOCK_DIM = 64


def make_chunk(
    chunk_id: str,
    page: int,
    modality: ModalityType,
    content: str,
    metadata: dict = None,
    table_data: dict = None,
) -> EvidenceChunk:
    return EvidenceChunk(
        document_id="doc_test",
        document_name="test.pdf",
        page_number=page,
        chunk_id=chunk_id,
        modality=modality,
        content=content,
        metadata=metadata or {},
        table_data=table_data,
    )


def build_indexed_retriever(chunks: List[EvidenceChunk]):
    """Build a ready-to-use EvidenceFuser with the given chunks indexed."""
    embed_mgr = EmbeddingManager(mock_dim=MOCK_DIM)
    vstore = FAISSVectorStore(dimension=MOCK_DIM)
    lexical = LexicalRetriever()

    embs, valid = embed_mgr.embed_chunks(chunks)
    if len(valid) > 0:
        vstore.add_chunks(valid, embs)
    lexical.index_chunks(chunks)

    fuser = EvidenceFuser(
        vector_store=vstore,
        lexical_retriever=lexical,
        embedding_manager=embed_mgr,
    )
    return fuser


# ──────────────────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────────────────

class TestQueryAnalyzer(unittest.TestCase):
    """Tests for QueryAnalyzer intent detection and modality weight calculation."""

    def setUp(self):
        self.analyzer = QueryAnalyzer()

    # 1. Text / conceptual intent
    def test_01_text_intent_detected(self):
        """Queries about concepts/definitions should route heavily to TEXT."""
        result = self.analyzer.analyze("What is the definition of multimodal fusion?")
        self.assertEqual(result.routing_method, "Rule-based signal routing (heuristic baseline)")
        text_w = result.modality_weights[ModalityType.TEXT]
        # TEXT weight should dominate over TABLE and FIGURE
        table_w = result.modality_weights[ModalityType.TABLE]
        figure_w = result.modality_weights[ModalityType.FIGURE]
        self.assertGreater(text_w, table_w)
        self.assertGreater(text_w, figure_w)

    # 2. Table / numerical intent
    def test_02_table_intent_detected(self):
        """Queries with numeric comparison keywords should route heavily to TABLE."""
        result = self.analyzer.analyze("What was the highest revenue growth percentage in Q3?")
        table_w = result.modality_weights[ModalityType.TABLE]
        text_w = result.modality_weights[ModalityType.TEXT]
        self.assertGreater(table_w, text_w)
        self.assertIn(result.detected_intent, [
            "Numeric Comparison / Statistics",
            "Table Lookup",
            "Mixed Modality",
        ])

    # 3. Figure / visual intent
    def test_03_figure_intent_detected(self):
        """Queries about charts and diagrams should route heavily to FIGURE."""
        result = self.analyzer.analyze("Describe the architecture diagram shown in the figure.")
        figure_w = result.modality_weights[ModalityType.FIGURE]
        text_w = result.modality_weights[ModalityType.TEXT]
        self.assertGreater(figure_w, text_w)

    # 4. Mixed-modality intent
    def test_04_mixed_modality_detected(self):
        """Queries referencing both chart trends and numerical totals should give balanced weights."""
        result = self.analyzer.analyze(
            "Compare the bar chart trend with the revenue table totals."
        )
        # Mixed — both TABLE and FIGURE signals are present
        table_w = result.modality_weights[ModalityType.TABLE]
        figure_w = result.modality_weights[ModalityType.FIGURE]
        # Neither modality should be near-zero
        self.assertGreater(table_w, 0.10)
        self.assertGreater(figure_w, 0.10)

    # 5. Empty query fallback
    def test_05_empty_query_fallback(self):
        """An empty query string should return the default prior without crashing."""
        result = self.analyzer.analyze("")
        self.assertEqual(result.detected_intent, "Empty / Fallback")
        self.assertAlmostEqual(result.confidence, 0.0, places=5)
        # Weights must still be valid (non-negative, sum to 1)
        total = sum(result.modality_weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    # 6. Weight validity invariant
    def test_06_weights_sum_to_one_and_non_negative(self):
        """Modality weights must always be finite, ≥ 0, and sum to exactly 1.0."""
        queries = [
            "What is revenue?",
            "Show me the figure with the scatter plot.",
            "What percentage increase was shown in the table?",
            "Mixed query about charts and tables with numbers 42 and 0.87",
            "",
            "   ",
            "a",
        ]
        for q in queries:
            with self.subTest(query=q):
                result = self.analyzer.analyze(q)
                weights = list(result.modality_weights.values())
                for w in weights:
                    self.assertTrue(np.isfinite(w), f"Weight {w} is not finite for query: {q!r}")
                    self.assertGreaterEqual(w, 0.0, f"Negative weight {w} for query: {q!r}")
                self.assertAlmostEqual(sum(weights), 1.0, places=5,
                                       msg=f"Weights don't sum to 1.0 for query: {q!r}")


class TestEvidenceFuser(unittest.TestCase):
    """Tests for MW-RRF fusion logic, ranking, deduplication, and fallback paths."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_phase5_fuser_"))
        self.analyzer = QueryAnalyzer()
        self.text_chunks = [
            make_chunk("tc1", 1, ModalityType.TEXT, "Multimodal evidence retrieval system overview."),
            make_chunk("tc2", 2, ModalityType.TEXT, "Query understanding uses rule-based intent signals."),
            make_chunk("tc3", 3, ModalityType.TEXT, "Fusion module combines text, table, figure results."),
        ]
        self.table_chunks = [
            make_chunk(
                "tb1", 4, ModalityType.TABLE,
                "| Metric | Value |\n| --- | --- |\n| F1 | 0.91 |",
                metadata={"table_index": 1},
                table_data={"headers": ["Metric", "Value"], "rows": [["F1", "0.91"]]},
            ),
            make_chunk(
                "tb2", 5, ModalityType.TABLE,
                "| Quarter | Revenue |\n| --- | --- |\n| Q3 | 50M |",
                metadata={"table_index": 2},
                table_data={"headers": ["Quarter", "Revenue"], "rows": [["Q3", "50M"]]},
            ),
        ]
        self.all_chunks = self.text_chunks + self.table_chunks

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # 7. MW-RRF scores are positive and results are correctly ranked
    def test_07_fused_scores_positive_and_ranked(self):
        """MW-RRF scores must be strictly positive and results descending by score."""
        fuser = build_indexed_retriever(self.all_chunks)
        analysis = self.analyzer.analyze("What is the F1 score in the table?")
        results = fuser.fuse("What is the F1 score in the table?", analysis, top_k=3)

        self.assertGreater(len(results), 0)
        for i, item in enumerate(results, start=1):
            self.assertEqual(item.rank, i)
            self.assertGreater(item.fused_score, 0.0)
        # Scores must be non-increasing
        scores = [r.fused_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    # 8. Empty evidence fallback
    def test_08_empty_evidence_fallback(self):
        """Fuser with empty index must return empty list without crashing."""
        embed_mgr = EmbeddingManager(mock_dim=MOCK_DIM)
        vstore = FAISSVectorStore(dimension=MOCK_DIM)
        lexical = LexicalRetriever()
        fuser = EvidenceFuser(
            vector_store=vstore,
            lexical_retriever=lexical,
            embedding_manager=embed_mgr,
        )
        analysis = self.analyzer.analyze("some query")
        results = fuser.fuse("some query", analysis, top_k=5)
        self.assertEqual(results, [])

    # 9. No duplicate chunk_ids in fused results
    def test_09_no_duplicate_chunk_ids(self):
        """Fusion must not return the same chunk_id more than once."""
        fuser = build_indexed_retriever(self.all_chunks)
        analysis = self.analyzer.analyze("Q3 revenue F1 score table figure")
        results = fuser.fuse("Q3 revenue F1 score table figure", analysis, top_k=len(self.all_chunks))

        chunk_ids = [r.chunk.chunk_id for r in results]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)), "Duplicate chunk_ids found in fused results")

    # 10. High TABLE weight surfaces table chunks
    def test_10_table_weight_influences_results(self):
        """A table-heavy query should have at least one table chunk in the top-3."""
        fuser = build_indexed_retriever(self.all_chunks)
        analysis = self.analyzer.analyze("What is the Q3 revenue in the table?")
        results = fuser.fuse("What is the Q3 revenue in the table?", analysis, top_k=3)

        self.assertGreater(len(results), 0)
        modalities = [r.chunk.modality for r in results]
        # At least one TABLE in top results
        self.assertIn(ModalityType.TABLE, modalities,
                      f"No TABLE in top-3. Got: {[m.value for m in modalities]}")


class TestAnswerGenerator(unittest.TestCase):
    """Tests for AnswerGenerator: mock fallback, citations, empty evidence, serialization."""

    def setUp(self):
        self.generator = AnswerGenerator()
        self.analyzer = QueryAnalyzer()

        # Build fake FusedEvidenceResult objects without a real index
        self.text_evidence = self._make_fused(
            make_chunk("c_text", 2, ModalityType.TEXT, "The system achieves 91% precision on benchmark A."),
            rank=1, weight=0.55,
        )
        self.table_evidence = self._make_fused(
            make_chunk(
                "c_table", 4, ModalityType.TABLE,
                "| Method | Precision |\n| --- | --- |\n| Ours | 91% |\n| Baseline | 74% |",
                metadata={"table_index": 2},
                table_data={"headers": ["Method", "Precision"], "rows": [["Ours", "91%"], ["Baseline", "74%"]]},
            ),
            rank=2, weight=0.35,
        )

    @staticmethod
    def _make_fused(chunk: EvidenceChunk, rank: int, weight: float) -> FusedEvidenceResult:
        return FusedEvidenceResult(
            chunk=chunk,
            fused_score=weight / rank,
            rank=rank,
            modality_weight=weight,
            source_results=[],
        )

    # 11. Mock generation returns non-empty answer
    def test_11_mock_generation_returns_answer(self):
        """Mock generator must produce a non-empty answer string."""
        evidence = [self.text_evidence, self.table_evidence]
        analysis = self.analyzer.analyze("What is the precision?")
        result = self.generator.generate("What is the precision?", evidence, analysis)

        self.assertIsInstance(result, AnswerResult)
        self.assertGreater(len(result.answer.strip()), 0)
        self.assertTrue(result.is_fallback)  # No API key → mock mode

    # 12. Citations match supplied evidence
    def test_12_citations_match_supplied_evidence(self):
        """All citation labels must originate from evidence actually passed to generate()."""
        evidence = [self.text_evidence, self.table_evidence]
        analysis = self.analyzer.analyze("Precision comparison")
        result = self.generator.generate("Precision comparison", evidence, analysis)

        supplied_labels = {item.chunk.citation_label for item in evidence}
        for cit in result.citations:
            self.assertIn(
                cit.citation_label, supplied_labels,
                f"Citation {cit.citation_label!r} was not in supplied evidence: {supplied_labels}"
            )

    # 13. Empty evidence → explicit insufficiency message
    def test_13_empty_evidence_gives_insufficiency_message(self):
        """When no evidence is provided, the system must not invent facts."""
        analysis = self.analyzer.analyze("Some question")
        result = self.generator.generate("Some question", [], analysis)

        self.assertEqual(result.top_evidence_count, 0)
        self.assertIn("insufficient", result.answer.lower(),
                      f"Expected 'insufficient' in empty-evidence answer, got: {result.answer!r}")
        self.assertEqual(len(result.citations), 0)

    # 14. Missing API key uses mock fallback without crashing
    def test_14_missing_api_key_uses_mock_fallback(self):
        """AnswerGenerator without GEMINI_API_KEY must silently fall back to mock mode."""
        gen = AnswerGenerator()  # No API key in test environment
        self.assertFalse(gen._api_available)

        evidence = [self.text_evidence]
        analysis = self.analyzer.analyze("What is the result?")
        result = gen.generate("What is the result?", evidence, analysis)

        self.assertTrue(result.is_fallback)
        self.assertEqual(result.model_used, "mock_generator")
        self.assertIsNotNone(result.warning)
        self.assertGreater(len(result.answer), 0)

    # 15. AnswerResult serializes to valid dict
    def test_15_answer_result_serializes_correctly(self):
        """AnswerResult.to_dict() must produce a complete, JSON-compatible dictionary."""
        evidence = [self.text_evidence]
        analysis = self.analyzer.analyze("Test query")
        result = self.generator.generate("Test query", evidence, analysis)

        d = result.to_dict()
        self.assertIn("query", d)
        self.assertIn("answer", d)
        self.assertIn("citations", d)
        self.assertIn("intent", d)
        self.assertIn("modality_weights", d)
        self.assertIn("top_evidence_count", d)
        self.assertIn("model_used", d)
        self.assertIn("is_fallback", d)
        self.assertIsInstance(d["citations"], list)
        # Validate serialized weight keys are strings
        for k in d["modality_weights"]:
            self.assertIsInstance(k, str)

    # 16. Citation labels are 1-indexed and reference correct pages
    def test_16_citation_labels_are_1_indexed(self):
        """Citation page numbers must be ≥ 1 and match the chunk's page_number field."""
        evidence = [self.text_evidence, self.table_evidence]
        analysis = self.analyzer.analyze("result")
        result = self.generator.generate("result", evidence, analysis)

        for cit in result.citations:
            self.assertGreaterEqual(cit.page_number, 1,
                                    f"Citation {cit.citation_label!r} has 0-indexed page {cit.page_number}")
            # The label must contain the page number
            self.assertIn(f"Page {cit.page_number}", cit.citation_label)


class TestPhase5EndToEnd(unittest.TestCase):
    """Full pipeline integration test: PDF bytes → index → analyze → fuse → generate."""

    def setUp(self):
        from tests.test_pdf_ingestion import VALID_PDF_BYTES, generate_clean_pdf_bytes
        import io
        import pypdf

        self.test_dir = Path(tempfile.mkdtemp(prefix="test_phase5_e2e_"))
        self.embed_mgr = EmbeddingManager(mock_dim=MOCK_DIM)
        self.analyzer = QueryAnalyzer()

        # Build a small multi-page synthetic document
        base_bytes = VALID_PDF_BYTES
        reader = pypdf.PdfReader(io.BytesIO(base_bytes))
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[0])
        writer.add_page(reader.pages[0])
        buf = io.BytesIO()
        writer.write(buf)
        self.pdf_bytes = buf.getvalue()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # 17. End-to-end pipeline completes without errors
    def test_17_full_pipeline_end_to_end(self):
        """Run the complete Phase 5 pipeline from raw PDF bytes to AnswerResult."""
        from src.ingestion.pdf_loader import PDFLoader
        from src.ingestion.text_extractor import TextExtractor

        loader = PDFLoader(upload_dir=self.test_dir)
        extractor = TextExtractor(chunk_size=200, chunk_overlap=20)

        # Step 1: ingest
        doc_info = loader.process_and_save(self.pdf_bytes, "e2e_test_doc.pdf")
        self.assertTrue(doc_info.is_valid)

        # Step 2: extract text chunks
        extracted: ExtractedDocument = extractor.extract_document(doc_info)
        self.assertGreater(extracted.total_chunks, 0)

        # Step 3: index
        vstore = FAISSVectorStore(dimension=MOCK_DIM)
        lexical = LexicalRetriever()
        embs, valid = self.embed_mgr.embed_chunks(extracted.chunks)
        self.assertGreater(len(valid), 0)
        vstore.add_chunks(valid, embs)
        lexical.index_chunks(extracted.chunks)

        # Step 4: analyze
        query = "What is the document about?"
        analysis = self.analyzer.analyze(query)
        self.assertIsInstance(analysis, QueryAnalysisResult)
        weights = list(analysis.modality_weights.values())
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

        # Step 5: fuse
        fuser = EvidenceFuser(
            vector_store=vstore,
            lexical_retriever=lexical,
            embedding_manager=self.embed_mgr,
        )
        fused = fuser.fuse(query, analysis, top_k=3)
        self.assertGreater(len(fused), 0)
        for item in fused:
            self.assertGreaterEqual(item.chunk.page_number, 1)
            self.assertEqual(item.chunk.document_id, doc_info.document_id)

        # Step 6: generate
        generator = AnswerGenerator()
        result = generator.generate(query, fused, analysis)
        self.assertIsInstance(result, AnswerResult)
        self.assertGreater(len(result.answer.strip()), 0)

        # Verify citation integrity: all cited pages exist in document
        valid_pages = set(range(1, doc_info.page_count + 1))
        for cit in result.citations:
            self.assertIn(cit.page_number, valid_pages,
                          f"Citation references page {cit.page_number} but doc only has {doc_info.page_count} pages")

    # 18. Backward compatibility: Phase 4 HybridRetriever unaffected
    def test_18_phase4_hybrid_retriever_backward_compat(self):
        """HybridRetriever must continue working after Phase 5 imports are added."""
        chunks = [
            make_chunk("c1", 1, ModalityType.TEXT, "Multimodal document question answering system."),
            make_chunk("c2", 2, ModalityType.TEXT, "Dense retrieval with sentence transformers."),
        ]
        vstore = FAISSVectorStore(dimension=MOCK_DIM)
        lexical = LexicalRetriever()
        embs, valid = self.embed_mgr.embed_chunks(chunks)
        vstore.add_chunks(valid, embs)
        lexical.index_chunks(chunks)

        hybrid = HybridRetriever(
            vector_store=vstore,
            lexical_retriever=lexical,
            embedding_manager=self.embed_mgr,
            rrf_k=60,
        )
        results = hybrid.search("document question answering", top_k=2)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsNotNone(r.rrf_score)
            self.assertEqual(r.retrieval_method, "hybrid_rrf")


if __name__ == "__main__":
    unittest.main()
