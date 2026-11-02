import math
from typing import List, Set, Dict, Tuple, Optional


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> Optional[float]:
    """Recall@k = |retrieved∩relevant| / |relevant|.
    Returns None if there are no relevant items.
    """
    if not relevant:
        return None
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def mrr_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Mean Reciprocal Rank@k for a single query.
    Returns 0.0 if no relevant in top‑k.
    """
    for idx, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: List[str], relevance_scores: Dict[str, int], k: int) -> Optional[float]:
    """nDCG@k using integer relevance grades (higher = more relevant).
    If all relevance scores are zero, returns None.
    """
    def dcg(scores: List[int]) -> float:
        return sum((2 ** rel - 1) / math.log2(idx + 1) for idx, rel in enumerate(scores, start=1))

    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    if not any(ideal):
        return None
    retrieved_scores = [relevance_scores.get(cid, 0) for cid in retrieved[:k]]
    return dcg(retrieved_scores) / dcg(ideal)


def latency_ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def lexical_f1(prediction: str, gold: str) -> Tuple[float, float, float]:
    """Token‑level precision, recall, F1.
    Tokens are split on whitespace.
    """
    pred_tokens = set(prediction.split())
    gold_tokens = set(gold.split())
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens:
        return 0.0, 0.0, 0.0
    tp = len(pred_tokens & gold_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens) if gold_tokens else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def citation_correctness(fused_chunk_ids: List[str], gold_chunk_ids: Set[str]) -> Optional[float]:
    """Fraction of citations that are in the gold set.
    Returns None if gold set is empty.
    """
    if not gold_chunk_ids:
        return None
    if not fused_chunk_ids:
        return 0.0
    return len(set(fused_chunk_ids) & gold_chunk_ids) / len(gold_chunk_ids)
