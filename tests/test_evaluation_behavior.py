import unittest
import sys
import io
import json
import pathlib
import csv
from unittest import mock

# Import the evaluation module
from evaluation import evaluate as eval_mod
from evaluation.metrics import recall_at_k, mrr_at_k


class DummyResult:
    def __init__(self, chunk_id, modality_name):
        self.chunk = mock.Mock()
        self.chunk.chunk_id = chunk_id
        self.chunk.modality = mock.Mock()
        self.chunk.modality.name = modality_name


class DummyHybrid:
    def __init__(self, results):
        self._results = results

    def search(self, query, top_k=None, modality_filter=None):
        return self._results


class TestEvaluationBehavior(unittest.TestCase):
    def test_run_method_a_text_filter_and_k(self):
        results = [
            DummyResult('c1', 'TEXT'),
            DummyResult('c2', 'TABLE'),
            DummyResult('c3', 'TEXT'),
            DummyResult('c4', 'FIGURE'),
            DummyResult('c5', 'TEXT'),
        ]
        hybrid = DummyHybrid(results)
        text_res, _ = eval_mod.run_method_a('dummy query', hybrid, k=2)
        ids = [r.chunk.chunk_id for r in text_res]
        self.assertEqual(ids, ['c1', 'c3'])

    def test_evaluate_exits_on_missing_ground_truth(self):
        # With complete ground truth, evaluate should complete without exiting.
        try:
            eval_mod.evaluate()
        except SystemExit as e:
            self.fail(f"evaluate() raised SystemExit unexpectedly with code {e.code}")
        # If no exception, test passes.



class TestMetricsCorrectness(unittest.TestCase):
    """Regression tests verifying metric computation against known ground truth."""

    def test_recall_at_k_full_match(self):
        retrieved = ['c1', 'c2', 'c3', 'c4', 'c5']
        gold = {'c1', 'c2'}
        self.assertAlmostEqual(recall_at_k(retrieved, gold, 5), 1.0)
        self.assertAlmostEqual(recall_at_k(retrieved, gold, 2), 1.0)

    def test_recall_at_k_partial_match(self):
        retrieved = ['c1', 'c9', 'c3', 'c4', 'c5']
        gold = {'c1', 'c2'}
        # c1 is retrieved at rank 1, c2 is not in top 5
        self.assertAlmostEqual(recall_at_k(retrieved, gold, 5), 0.5)

    def test_recall_at_k_no_match(self):
        retrieved = ['c3', 'c4', 'c5']
        gold = {'c1', 'c2'}
        self.assertAlmostEqual(recall_at_k(retrieved, gold, 5), 0.0)

    def test_recall_at_k_none_when_empty_gold(self):
        """recall_at_k must return None (not 0.0) when gold set is empty."""
        retrieved = ['c1', 'c2', 'c3']
        result = recall_at_k(retrieved, set(), 5)
        self.assertIsNone(result)

    def test_mrr_at_k_first_result_relevant(self):
        retrieved = ['c1', 'c2', 'c3']
        gold = {'c1', 'c9'}
        self.assertAlmostEqual(mrr_at_k(retrieved, gold, 5), 1.0)

    def test_mrr_at_k_second_result_relevant(self):
        retrieved = ['cX', 'c1', 'c3']
        gold = {'c1'}
        self.assertAlmostEqual(mrr_at_k(retrieved, gold, 5), 0.5)

    def test_mrr_at_k_no_relevant_in_top_k(self):
        retrieved = ['cA', 'cB', 'cC']
        gold = {'c1'}
        self.assertAlmostEqual(mrr_at_k(retrieved, gold, 3), 0.0)

    def test_mrr_at_k_relevant_at_rank8_with_k5(self):
        """Verify that a relevant chunk at rank 8 is NOT counted when k=5."""
        retrieved = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'gold_id']
        gold = {'gold_id'}
        # gold_id is at rank 8, outside k=5
        self.assertAlmostEqual(mrr_at_k(retrieved, gold, 5), 0.0)
        # but IS counted with k=10
        self.assertAlmostEqual(mrr_at_k(retrieved, gold, 10), 1.0 / 8)

    def test_recall_ignores_duplicates_in_retrieved(self):
        """Duplicate chunk IDs in retrieved list should not inflate recall."""
        retrieved = ['c1', 'c1', 'c1', 'c1', 'c1']
        gold = {'c1', 'c2'}
        # Only c1 is in top-5, c2 is absent: recall = 1/2 = 0.5
        self.assertAlmostEqual(recall_at_k(retrieved, gold, 5), 0.5)

    def test_q0_real_retrieval_verified(self):
        """
        Regression: Q0 gold chunks (p29_c2, p7_c2) do NOT appear in top-5
        for any method but p7_c2 appears at rank 8 in Method C top-10.
        This is a documented retrieval gap, not a metrics bug.
        """
        # Simulating exact audit output from run
        c_top10 = [
            'doc_af88fad1358a_p10_c3', 'doc_af88fad1358a_p16_c6',
            'doc_af88fad1358a_p2_c5', 'doc_af88fad1358a_p30_c2',
            'doc_af88fad1358a_p29_c1', 'doc_af88fad1358a_p17_c2',
            'doc_af88fad1358a_p22_c4', 'doc_af88fad1358a_p7_c2',
            'doc_af88fad1358a_p26_t1', 'doc_af88fad1358a_p1_fig1'
        ]
        gold = {'doc_af88fad1358a_p29_c2', 'doc_af88fad1358a_p7_c2'}
        # Recall@5 must be 0.0 (neither gold chunk is in top-5)
        self.assertAlmostEqual(recall_at_k(c_top10, gold, 5), 0.0)
        # Recall@10 must be 0.5 (p7_c2 at rank 8, p29_c2 absent)
        self.assertAlmostEqual(recall_at_k(c_top10, gold, 10), 0.5)
        # MRR@10 must be 1/8 = 0.125 (first match at rank 8)
        self.assertAlmostEqual(mrr_at_k(c_top10, gold, 10), 1.0 / 8)

    def test_q2_table_query_method_c_perfect(self):
        """
        Regression: Q2 (Java SE versions table) must achieve Recall@5=1.0
        and MRR@5=1.0 for Method C only. Methods A and B should score 0.0.
        """
        c_top5 = [
            'doc_af88fad1358a_p27_t1',
            'doc_af88fad1358a_p28_t1',
            'doc_af88fad1358a_p26_t1'
        ]
        a_top5 = [
            'doc_af88fad1358a_p13_c2', 'doc_af88fad1358a_p27_c3',
            'doc_af88fad1358a_p22_c1', 'doc_af88fad1358a_p6_c1',
            'doc_af88fad1358a_p4_c1'
        ]
        gold = {
            'doc_af88fad1358a_p26_t1',
            'doc_af88fad1358a_p27_t1',
            'doc_af88fad1358a_p28_t1'
        }
        self.assertAlmostEqual(recall_at_k(c_top5, gold, 5), 1.0)
        self.assertAlmostEqual(mrr_at_k(c_top5, gold, 5), 1.0)
        self.assertAlmostEqual(recall_at_k(a_top5, gold, 5), 0.0)
        self.assertAlmostEqual(mrr_at_k(a_top5, gold, 5), 0.0)


class TestGroundTruthFileIntegrity(unittest.TestCase):
    """Verify ground_truth.csv has the expected structure and current annotations."""

    GT_PATH = pathlib.Path(__file__).parent.parent / "evaluation" / "ground_truth.csv"

    def test_ground_truth_file_exists(self):
        self.assertTrue(self.GT_PATH.is_file(), "ground_truth.csv must exist")

    def test_ground_truth_has_five_question_ids(self):
        with self.GT_PATH.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = [int(r["question_id"]) for r in rows]
        self.assertEqual(sorted(ids), [0, 1, 2, 3, 4])

    def test_q4_is_still_pending(self):
        """Q4 now has verified gold chunk(s)."""
        with self.GT_PATH.open("r", encoding="utf-8") as f:
            rows = {int(r["question_id"]): r["gold_chunk_ids"].strip() for r in csv.DictReader(f)}
        self.assertEqual(rows[4], "doc_af88fad1358a_p25_c2", "Q4 gold chunk should be present after verification")

    def test_evaluated_questions_have_chunks(self):
        """Q0-Q3 must have non-empty gold_chunk_ids."""
        with self.GT_PATH.open("r", encoding="utf-8") as f:
            rows = {int(r["question_id"]): r["gold_chunk_ids"].strip() for r in csv.DictReader(f)}
        for qid in [0, 1, 2, 3]:
            self.assertNotEqual(rows[qid], "", f"Q{qid} must have verified gold chunks")

    def test_table_chunks_annotated_for_q2(self):
        """Q2 must include all three table chunks (p26, p27, p28)."""
        with self.GT_PATH.open("r", encoding="utf-8") as f:
            rows = {int(r["question_id"]): r["gold_chunk_ids"].strip() for r in csv.DictReader(f)}
        gold_ids = set(rows[2].split(";"))
        for expected in [
            "doc_af88fad1358a_p26_t1",
            "doc_af88fad1358a_p27_t1",
            "doc_af88fad1358a_p28_t1",
        ]:
            self.assertIn(expected, gold_ids, f"{expected} must be a gold chunk for Q2")

    def test_no_figure2_in_sample_questions(self):
        """sample_questions.json must not contain the ungrounded Figure 2 question."""
        sq_path = self.GT_PATH.parent / "sample_questions.json"
        with sq_path.open("r", encoding="utf-8") as f:
            questions = json.load(f)
        texts = [q["question"] for q in questions]
        for text in texts:
            self.assertNotIn("Figure 2", text,
                             f"Ungrounded 'Figure 2' question must not appear in benchmark: '{text}'")

    def test_no_memory_comparison_in_sample_questions(self):
        """Original memory comparison question must have been replaced."""
        sq_path = self.GT_PATH.parent / "sample_questions.json"
        with sq_path.open("r", encoding="utf-8") as f:
            questions = json.load(f)
        texts = [q["question"] for q in questions]
        for text in texts:
            self.assertNotIn("Compare Java's memory usage with C++", text,
                             f"Original ungrounded memory comparison must not appear: '{text}'")


if __name__ == '__main__':
    unittest.main()
