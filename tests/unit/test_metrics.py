"""Unit tests for retrieval and answer evaluation metrics."""

from evaluation.metrics.retrieval_metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_hit_rate_at_k,
    compute_reciprocal_rank,
    compute_average_precision,
    evaluate_retrieval_batch,
)
from evaluation.metrics.answer_metrics import (
    compute_fact_coverage,
    compute_citation_correctness,
    compute_abstention_accuracy,
    compute_grounding_score,
)


def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3"}
    # 2 relevant out of top 5 -> 2/5 = 0.4
    assert compute_precision_at_k(retrieved, relevant, k=5) == 0.4
    # 1 relevant out of top 2 -> 1/2 = 0.5
    assert compute_precision_at_k(retrieved, relevant, k=2) == 0.5
    # Empty relevant: if retrieved is non-empty -> 0.0
    assert compute_precision_at_k(retrieved, set(), k=3) == 0.0
    # Empty relevant and empty retrieved -> 1.0 (perfect abstention)
    assert compute_precision_at_k([], set(), k=3) == 1.0


def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc3", "doc4", "doc5"}
    # 2 retrieved out of 4 relevant -> 2/4 = 0.5
    assert compute_recall_at_k(retrieved, relevant, k=3) == 0.5
    # Empty relevant with empty retrieved -> 1.0
    assert compute_recall_at_k([], set(), k=3) == 1.0


def test_hit_rate_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    assert compute_hit_rate_at_k(retrieved, {"doc2"}, k=3) == 1.0
    assert compute_hit_rate_at_k(retrieved, {"doc99"}, k=3) == 0.0


def test_reciprocal_rank():
    # 1st position -> 1/1 = 1.0
    assert compute_reciprocal_rank(["doc1", "doc2"], {"doc1"}) == 1.0
    # 2nd position -> 1/2 = 0.5
    assert compute_reciprocal_rank(["doc1", "doc2"], {"doc2"}) == 0.5
    # 3rd position -> 1/3 = 0.3333...
    assert round(compute_reciprocal_rank(["doc1", "doc2", "doc3"], {"doc3"}), 4) == 0.3333
    # Not found -> 0.0
    assert compute_reciprocal_rank(["doc1", "doc2"], {"doc99"}) == 0.0


def test_average_precision():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc3"}
    # rank 1: precision 1/1 = 1.0
    # rank 3: precision 2/3 = 0.6667
    # AP = (1.0 + 0.6667) / 2 = 0.8333
    ap = compute_average_precision(retrieved, relevant, k=4)
    assert round(ap, 4) == 0.8333


def test_evaluate_retrieval_batch():
    batch = [
        {"retrieved_doc_ids": ["doc1", "doc2"], "expected_doc_ids": ["doc1"]},
        {"retrieved_doc_ids": ["doc2", "doc1"], "expected_doc_ids": ["doc1"]},
    ]
    res = evaluate_retrieval_batch(batch, k_values=[1, 2])
    assert res["total_queries"] == 2
    assert "mrr" in res
    assert "precision@1" in res


def test_fact_coverage():
    answer = "Patient should start an ACE inhibitor and SGLT2 inhibitor to control proteinuria."
    expected = ["ACE inhibitor", "SGLT2 inhibitor", "proteinuria"]
    assert compute_fact_coverage(answer, expected) == 1.0

    missing_expected = ["ACE inhibitor", "dialysis schedule"]
    assert compute_fact_coverage(answer, missing_expected) == 0.5


def test_citation_correctness():
    citations = ["[1] Ref: chunk_ada_1", "[2] Ref: chunk_unknown_99"]
    valid = {"chunk_ada_1"}
    res = compute_citation_correctness(citations, valid)
    assert res["citation_count"] == 2
    assert res["valid_citations"] == 1
    assert res["citation_precision"] == 0.5


def test_abstention_accuracy():
    # Should abstain and did abstain -> 1.0
    assert compute_abstention_accuracy("out_of_scope", did_abstain=True, is_verified=False) == 1.0
    # Should abstain and did NOT abstain -> 0.0
    assert compute_abstention_accuracy("out_of_scope", did_abstain=False, is_verified=True) == 0.0
    # Answerable and verified -> 1.0
    assert compute_abstention_accuracy("answerable", did_abstain=False, is_verified=True) == 1.0
    # Answerable and abstained -> 0.0
    assert compute_abstention_accuracy("answerable", did_abstain=True, is_verified=False) == 0.0
