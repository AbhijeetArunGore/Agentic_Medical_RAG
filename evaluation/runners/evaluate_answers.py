"""End-to-end multi-agent answer quality, grounding, and citation evaluation runner.

Usage:
    python -m evaluation.runners.evaluate_answers
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from app.agents.orchestrator import create_medical_graph
from app.rag.retriever import MedicalRetriever
from evaluation.metrics.answer_metrics import (
    compute_fact_coverage,
    compute_citation_correctness,
    compute_abstention_accuracy,
    compute_abstention_confusion_matrix,
    compute_grounding_score,
)
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("EvalAnswers")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
GOLDEN_SET_PATH = ROOT_DIR / "evaluation" / "datasets" / "medical_rag_golden_set.jsonl"
REPORTS_DIR = ROOT_DIR / "evaluation" / "reports"


def main():
    print("\n" + "=" * 80)
    print("=== STARTING AGENTIC MEDICAL RAG ANSWER EVALUATION ===")
    print("=" * 80 + "\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    print(f"Evaluating {len(dataset)} queries across multi-agent LangGraph workflow...\n")

    retriever = MedicalRetriever(retrieval_mode="hybrid")
    retriever.vector_store.seed_initial_knowledge_base(force_reseed=True)
    graph_app = create_medical_graph(retriever=retriever)

    fact_coverages = []
    citation_precisions = []
    claim_entailments = []
    abstention_accuracies = []
    grounding_scores = []
    results_detail = []

    t0 = time.time()

    for idx, item in enumerate(dataset, start=1):
        qid = item["question_id"]
        qtext = item["question"]
        expected_facts = item.get("expected_key_facts", [])
        expected_answerability = item.get("expected_answerability", "answerable")

        # Patient history context based on query category
        patient_hist = ""
        if "stenosis" in qtext.lower() or item.get("category") == "contradictory_evidence":
            patient_hist = "Patient has confirmed history of bilateral renal artery stenosis."

        initial_state = {
            "patient_id": f"PAT-EVAL-{qid}",
            "patient_history": patient_hist,
            "clinical_trends": [],
            "query": qtext,
            "retrieved_docs": [],
            "retrieved_chunks": [],
            "draft_briefing": "",
            "reasoning": "Starting evaluation run.",
            "verified": False,
            "abstained": False,
            "abstention_reason": None,
            "contradictions": [],
            "safety_flags": [],
            "final_report": "",
            "citations": [],
            "validated_citations": [],
            "iteration_count": 0,
            "max_iterations": settings.MAX_AGENT_ITERATIONS,
        }

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        for _ in graph_app.stream(initial_state, config):
            pass

        final_state = graph_app.get_state(config).values

        report = final_state.get("final_report", "")
        verified = final_state.get("verified", False)
        abstained = final_state.get("abstained", False)
        retrieved_chunks = final_state.get("retrieved_chunks", [])
        citations = final_state.get("citations", [])
        retrieved_texts = final_state.get("retrieved_docs", [])

        # Evaluate Metrics
        coverage = compute_fact_coverage(report, expected_facts)
        valid_cids = {c["chunk_id"] for c in retrieved_chunks}
        cite_res = compute_citation_correctness(
            citations=citations,
            valid_chunk_ids=valid_cids,
            answer_text=report,
            retrieved_chunks=retrieved_chunks,
        )
        abstain_acc = compute_abstention_accuracy(expected_answerability, abstained, verified)
        grounding = compute_grounding_score(report, retrieved_texts)

        fact_coverages.append(coverage)
        citation_precisions.append(cite_res["citation_precision"])
        claim_entailments.append(cite_res["claim_entailment_precision"])
        abstention_accuracies.append(abstain_acc)
        grounding_scores.append(grounding)

        results_detail.append(
            {
                "question_id": qid,
                "category": item["category"],
                "expected_answerability": expected_answerability,
                "verified": verified,
                "abstained": abstained,
                "fact_coverage": coverage,
                "citation_existence_precision": cite_res["citation_precision"],
                "claim_entailment_precision": cite_res["claim_entailment_precision"],
                "abstention_accuracy": abstain_acc,
                "grounding_score": grounding,
            }
        )

        if idx % 8 == 0 or idx == len(dataset):
            print(f"  Processed {idx}/{len(dataset)} evaluation records...")

    total_time_ms = (time.time() - t0) * 1000
    avg_coverage = round(sum(fact_coverages) / len(fact_coverages), 4)
    avg_citation_precision = round(sum(citation_precisions) / len(citation_precisions), 4)
    avg_claim_entailment = round(sum(claim_entailments) / len(claim_entailments), 4)
    avg_abstention_acc = round(sum(abstention_accuracies) / len(abstention_accuracies), 4)
    avg_grounding = round(sum(grounding_scores) / len(grounding_scores), 4)

    # Compute formal 2x2 confusion matrix
    cm = compute_abstention_confusion_matrix(results_detail)

    print("\n" + "=" * 80)
    print("=== END-TO-END ANSWER & SAFETY EVALUATION RESULTS ===")
    print("=" * 80)
    print(f"{'Metric':<40} | {'Measured Score':<15}")
    print("-" * 60)
    print(f"{'Fact Coverage Ratio':<40} | {avg_coverage:<15.4f}")
    print(f"{'Citation Existence Precision':<40} | {avg_citation_precision:<15.4f}")
    print(f"{'Claim Lexical Support Ratio (Overlap >= 0.35)':<40} | {avg_claim_entailment:<15.4f}")
    print(f"{'Abstention & Safety Accuracy':<40} | {avg_abstention_acc:<15.4f}")
    print(f"{'Evidence Grounding Score':<40} | {avg_grounding:<15.4f}")
    print(f"{'Total Evaluation Latency':<40} | {total_time_ms:<15.1f} ms")
    print("-" * 60)
    print("=== CLINICAL SAFETY & ABSTENTION CONFUSION MATRIX ===")
    print(
        f"  True Positives (TP, Safe Halts on Unsafe Queries):     {cm['tp']:>2} / 16 (Target: 16)"
    )
    print(
        f"  False Negatives (FN, Unsafe Output Breaches):         {cm['fn']:>2} / 16 (Target:  0)"
    )
    print(f"  True Negatives (TN, Verified Answers on Safe Cases):   {cm['tn']:>2} / 16")
    print(f"  False Positives (FP, Over-Abstentions on Safe Cases):  {cm['fp']:>2} / 16")
    print(
        f"  Safety Recall (Sensitivity):                          {cm['safety_recall'] * 100:>5.1f}%"
    )
    print(
        f"  Abstention Specificity:                               {cm['abstention_specificity'] * 100:>5.1f}%"
    )
    print(
        f"  Abstention Precision:                                 {cm['abstention_precision'] * 100:>5.1f}%"
    )
    print(f"  Abstention F1 Score:                                  {cm['abstention_f1']:>6.4f}")
    print(
        f"  Overall Decision Accuracy:                            {cm['overall_accuracy'] * 100:>5.1f}%"
    )
    print("=" * 80 + "\n")

    report_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(dataset),
        "overall_metrics": {
            "fact_coverage_ratio": avg_coverage,
            "citation_existence_precision": avg_citation_precision,
            "claim_lexical_support_precision": avg_claim_entailment,
            "claim_level_entailment_precision": avg_claim_entailment,
            "abstention_safety_accuracy": avg_abstention_acc,
            "evidence_grounding_score": avg_grounding,
            "total_latency_ms": round(total_time_ms, 2),
        },
        "confusion_matrix": cm,
        "per_question_results": results_detail,
    }

    out_file = REPORTS_DIR / "answer_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"Report saved to: {out_file}\n")


if __name__ == "__main__":
    main()
