import pathlib
import unittest

from evaluation import evaluate as eval_module
from evaluation.evaluate import run_method_a
from evaluation.metrics import recall_at_k, mrr_at_k

class DummyModality:
    def __init__(self, name):
        self.name = name

class DummyChunk:
    def __init__(self, chunk_id, modality_name):
        self.chunk_id = chunk_id
        self.modality = DummyModality(modality_name)

class DummyResult:
    def __init__(self, chunk):
        self.chunk = chunk

class DummyHybridRetriever:
    def __init__(self, results):
        self._results = results
    def search(self, query, top_k=None, modality_filter=None):
        return self._results[:top_k] if top_k else self._results

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.eval_dir = pathlib.Path(__file__).parent.parent / "evaluation"
        self.gt_path = self.eval_dir / "ground_truth.csv"
        self.gt_backup = None
        if self.gt_path.is_file():
            self.gt_backup = self.gt_path.read_text(encoding="utf-8")
            self.gt_path.unlink()
        for fn in ["results.json", "summary.csv", "evaluation_report.md"]:
            p = self.eval_dir / fn
            if p.is_file():
                p.unlink()

    def tearDown(self):
        if self.gt_backup is not None:
            self.gt_path.write_text(self.gt_backup, encoding="utf-8")
        else:
            if self.gt_path.is_file():
                self.gt_path.unlink()
        for fn in ["results.json", "summary.csv", "evaluation_report.md"]:
            p = self.eval_dir / fn
            if p.is_file():
                p.unlink()

    def test_missing_ground_truth_metrics_unavailable(self):
        with self.assertRaises(SystemExit) as cm:
            eval_module.evaluate()
        self.assertNotEqual(cm.exception.code, 0)

    def test_loading_valid_ground_truth(self):
        # Provide ground truth for all questions to avoid exiting
        csv_content = (
            "question_id,gold_chunk_ids\n"
            "0,doc_af88fad1358a_p2_c1;doc_af88fad1358a_p2_c2\n"
            "1,doc_af88fad1358a_p3_c1\n"
            "2,doc_af88fad1358a_p4_c1\n"
            "3,doc_af88fad1358a_p5_c1\n"
            "4,doc_af88fad1358a_p6_c1\n"
        )
        self.gt_path.write_text(csv_content, encoding="utf-8")
        results = eval_module.evaluate()
        # Verify that ground truth was loaded for the first entry
        first_entry = results[0]
        self.assertIn("doc_af88fad1358a_p2_c1", first_entry["gold_chunk_ids"])
        self.assertIn("doc_af88fad1358a_p2_c2", first_entry["gold_chunk_ids"])
        # Metrics may be None or a float depending on retrieval results; ensure type
        self.assertTrue(isinstance(first_entry["k_5"]["recall"]["A"], (float, type(None))))

    def test_recall_and_mrr_calculation(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        gold = {"c2", "c4"}
        self.assertAlmostEqual(recall_at_k(retrieved, gold, 5), 1.0)
        self.assertAlmostEqual(mrr_at_k(retrieved, gold, 5), 0.5)

    def test_text_only_filtering(self):
        results = [
            DummyResult(DummyChunk("c1", "TEXT")),
            DummyResult(DummyChunk("c2", "FIGURE")),
            DummyResult(DummyChunk("c3", "TEXT")),
        ]
        hybrid = DummyHybridRetriever(results)
        filtered, _ = run_method_a("dummy query", hybrid, k=3)
        self.assertEqual([r.chunk.chunk_id for r in filtered], ["c1", "c3"])

if __name__ == "__main__":
    unittest.main()
