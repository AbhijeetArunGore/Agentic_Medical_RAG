"""Information Retrieval Evaluation Metrics for Medical RAG.

Architectural Mechanics:
- Precision@K: Fraction of top-K retrieved documents that are relevant.
- Recall@K: Fraction of total known relevant documents successfully retrieved in top-K.
- Hit Rate@K: Binary indicator (1.0 or 0.0) whether at least one relevant document was retrieved in top-K.
- Reciprocal Rank (RR): Reciprocal of the rank (1/rank) of the first relevant document retrieved.
- Average Precision (AP): Area under the precision-recall curve across relevant positions.
- Correct handling of zero-relevant documents (e.g., out-of-scope/unanswerable queries).
"""

from typing import List, Set, Dict, Any


def compute_precision_at_k(
    retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int
) -> float:
    """Calculate Precision@K: (relevant items in top K) / K.

    If relevant_doc_ids is empty:
        - If retrieved_doc_ids[:k] is empty (correct abstention / 0 docs returned): returns 1.0
        - If system retrieved non-empty docs when 0 were expected: returns 0.0
    """
    if k <= 0:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    if not relevant_doc_ids:
        # Correctly retrieved nothing when nothing is relevant
        return 1.0 if len(top_k) == 0 else 0.0

    relevant_retrieved = sum(1 for doc in top_k if doc in relevant_doc_ids)
    return relevant_retrieved / float(k)


def compute_recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Calculate Recall@K: (relevant items in top K) / (total relevant items).

    If relevant_doc_ids is empty:
        - If retrieved_doc_ids[:k] is empty: returns 1.0 (perfect recall of 0 needed docs)
        - If system retrieved docs: returns 0.0
    """
    if not relevant_doc_ids:
        return 1.0 if len(retrieved_doc_ids[:k]) == 0 else 0.0

    top_k = retrieved_doc_ids[:k]
    relevant_retrieved = sum(1 for doc in top_k if doc in relevant_doc_ids)
    return relevant_retrieved / float(len(relevant_doc_ids))


def compute_hit_rate_at_k(
    retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int
) -> float:
    """Calculate Hit Rate@K: 1.0 if at least one relevant document is in top K, else 0.0."""
    if not relevant_doc_ids:
        return 1.0 if len(retrieved_doc_ids[:k]) == 0 else 0.0

    top_k = retrieved_doc_ids[:k]
    for doc in top_k:
        if doc in relevant_doc_ids:
            return 1.0
    return 0.0


def compute_reciprocal_rank(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str]) -> float:
    """Calculate Reciprocal Rank (RR): 1 / rank of first relevant item (1-indexed).

    Returns 0.0 if no relevant items are retrieved.
    For zero-relevant ground truth: returns 1.0 if no items returned, else 0.0.
    """
    if not relevant_doc_ids:
        return 1.0 if len(retrieved_doc_ids) == 0 else 0.0

    for rank, doc in enumerate(retrieved_doc_ids, start=1):
        if doc in relevant_doc_ids:
            return 1.0 / float(rank)
    return 0.0


def compute_average_precision(
    retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 10
) -> float:
    """Calculate Average Precision (AP@K) across retrieved documents."""
    if not relevant_doc_ids:
        return 1.0 if len(retrieved_doc_ids[:k]) == 0 else 0.0

    top_k = retrieved_doc_ids[:k]
    num_relevant = 0
    precision_sum = 0.0

    for i, doc in enumerate(top_k, start=1):
        if doc in relevant_doc_ids:
            num_relevant += 1
            precision_sum += num_relevant / float(i)

    if num_relevant == 0:
        return 0.0
    return precision_sum / float(min(len(relevant_doc_ids), k))


def evaluate_retrieval_batch(
    predictions: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """Computes Mean Precision@K, Recall@K, Hit Rate@K, MRR, and MAP over a batch of queries."""
    if not predictions:
        return {}

    total_queries = len(predictions)
    mrr_sum = 0.0
    map_sum = 0.0
    precisions_sum = {k: 0.0 for k in k_values}
    recalls_sum = {k: 0.0 for k in k_values}
    hit_rates_sum = {k: 0.0 for k in k_values}

    for item in predictions:
        retrieved = item["retrieved_doc_ids"]
        relevant = set(item["expected_doc_ids"])

        mrr_sum += compute_reciprocal_rank(retrieved, relevant)
        map_sum += compute_average_precision(retrieved, relevant, k=max(k_values))

        for k in k_values:
            precisions_sum[k] += compute_precision_at_k(retrieved, relevant, k)
            recalls_sum[k] += compute_recall_at_k(retrieved, relevant, k)
            hit_rates_sum[k] += compute_hit_rate_at_k(retrieved, relevant, k)

    results: Dict[str, float] = {
        "mrr": round(mrr_sum / total_queries, 4),
        "map": round(map_sum / total_queries, 4),
        "total_queries": total_queries,
    }

    for k in k_values:
        results[f"precision@{k}"] = round(precisions_sum[k] / total_queries, 4)
        results[f"recall@{k}"] = round(recalls_sum[k] / total_queries, 4)
        results[f"hit_rate@{k}"] = round(hit_rates_sum[k] / total_queries, 4)

    return results
