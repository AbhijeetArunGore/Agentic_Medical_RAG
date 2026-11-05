"""Retrieval evaluation runner comparing Baseline Dense vs Improved Hybrid retrieval.

Usage:
    python -m evaluation.runners.evaluate_retrieval

Outputs:
    - evaluation/reports/retrieval_benchmark_report.json
    - evaluation/reports/retrieval_benchmark_summary.csv
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from app.rag.retriever import MedicalRetriever
from app.rag.vector_store import VectorStoreManager
from evaluation.metrics.retrieval_metrics import evaluate_retrieval_batch
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("EvalRetrieval")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
GOLDEN_SET_PATH = ROOT_DIR / "evaluation" / "datasets" / "medical_rag_golden_set.jsonl"
REPORTS_DIR = ROOT_DIR / "evaluation" / "reports"


def load_golden_set() -> List[Dict[str, Any]]:
    """Loads benchmark queries and relevant document IDs from golden set."""
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found at {GOLDEN_SET_PATH}")

    dataset = []
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset


def run_configuration_evaluation(
    retriever: MedicalRetriever,
    dataset: List[Dict[str, Any]],
    strategy_name: str,
    top_k: int = 5,
) -> tuple[Dict[str, float], List[Dict[str, Any]], float]:
    """Evaluates a single retrieval configuration across the golden dataset."""
    predictions = []
    t0 = time.time()

    for item in dataset:
        query = item["question"]
        expected_doc_ids = item["expected_doc_ids"]

        # Run retrieval
        res = retriever.retrieve_with_diagnostics(query=query, top_k=top_k, strategy=strategy_name)

        # Extract document IDs from retrieved chunks
        retrieved_doc_ids = [c.document_id for c in res.chunks]

        predictions.append(
            {
                "question_id": item["question_id"],
                "query": query,
                "category": item["category"],
                "expected_doc_ids": expected_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_chunk_ids": [c.chunk_id for c in res.chunks],
                "scores": [c.score for c in res.chunks],
            }
        )

    elapsed_ms = (time.time() - t0) * 1000
    metrics = evaluate_retrieval_batch(predictions, k_values=[1, 3, 5])
    return metrics, predictions, elapsed_ms


def main():
    print("\n" + "=" * 80)
    print("=== STARTING MEDICAL RAG RETRIEVAL BENCHMARK EVALUATION ===")
    print("=" * 80 + "\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_golden_set()
    print(f"Loaded {len(dataset)} evaluation records from {GOLDEN_SET_PATH.name}\n")

    # Initialize shared vector store & seed
    vm = VectorStoreManager()
    vm.seed_initial_knowledge_base()

    # 1. Baseline: Dense Vector Retrieval
    print("Evaluating Baseline: Dense Cosine Vector Retrieval (Top K=5)...")
    baseline_retriever = MedicalRetriever(vector_store=vm, retrieval_mode="dense")
    baseline_metrics, baseline_preds, baseline_time = run_configuration_evaluation(
        retriever=baseline_retriever,
        dataset=dataset,
        strategy_name="dense",
        top_k=5,
    )
    print(f"  Baseline completed in {baseline_time:.1f}ms")

    # 2. Improved: Hybrid Retrieval (Dense + BM25 Fusion)
    print("Evaluating Improved: Hybrid Retrieval (Dense + BM25 Lexical Fusion, Top K=5)...")
    hybrid_retriever = MedicalRetriever(vector_store=vm, retrieval_mode="hybrid")
    hybrid_metrics, hybrid_preds, hybrid_time = run_configuration_evaluation(
        retriever=hybrid_retriever,
        dataset=dataset,
        strategy_name="hybrid",
        top_k=5,
    )
    print(f"  Improved Hybrid completed in {hybrid_time:.1f}ms\n")

    # Print comparison table
    print("=" * 80)
    print(f"{'Metric':<25} | {'Baseline (Dense)':<20} | {'Improved (Hybrid)':<20} | {'Delta':<10}")
    print("-" * 80)
    for metric_key in ["precision@5", "recall@5", "hit_rate@5", "mrr", "map"]:
        base_val = baseline_metrics.get(metric_key, 0.0)
        hyb_val = hybrid_metrics.get(metric_key, 0.0)
        delta = hyb_val - base_val
        print(f"{metric_key:<25} | {base_val:<20.4f} | {hyb_val:<20.4f} | {delta:+.4f}")
    print("=" * 80 + "\n")

    # Build report artifact
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "1.0.0",
        "total_test_queries": len(dataset),
        "configurations": {
            "baseline_dense": {
                "strategy": "dense",
                "top_k": 5,
                "similarity_metric": "cosine",
                "duration_ms": round(baseline_time, 2),
                "metrics": baseline_metrics,
            },
            "improved_hybrid": {
                "strategy": "hybrid_dense_bm25",
                "top_k": 5,
                "weights": {"dense": settings.DENSE_WEIGHT, "bm25": settings.BM25_WEIGHT},
                "duration_ms": round(hybrid_time, 2),
                "metrics": hybrid_metrics,
            },
        },
    }

    # Save JSON report
    json_path = REPORTS_DIR / "retrieval_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved machine-readable JSON report to: {json_path}")

    # Save CSV summary
    csv_path = REPORTS_DIR / "retrieval_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline_Dense", "Improved_Hybrid", "Delta"])
        for metric_key in [
            "precision@1",
            "precision@3",
            "precision@5",
            "recall@3",
            "recall@5",
            "hit_rate@5",
            "mrr",
            "map",
        ]:
            base_val = baseline_metrics.get(metric_key, 0.0)
            hyb_val = hybrid_metrics.get(metric_key, 0.0)
            writer.writerow([metric_key, base_val, hyb_val, round(hyb_val - base_val, 4)])
    print(f"Saved CSV benchmark summary to: {csv_path}\n")


if __name__ == "__main__":
    main()
