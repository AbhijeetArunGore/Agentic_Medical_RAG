# Medical RAG Evaluation Framework

This directory contains a reproducible evaluation dataset and execution harnesses to measure:
1. **Information Retrieval (IR) Performance**: Baseline Dense Vector vs Improved Hybrid (Dense + BM25 Lexical) Retrieval.
2. **Answer Quality & Grounding**: Fact coverage, citation grounding, evidence grounding, and abstention accuracy.

---

## Directory Structure

```
evaluation/
    datasets/
        medical_rag_golden_set.jsonl   # 32 expert-curated clinical CDS benchmark questions
    metrics/
        retrieval_metrics.py          # Precision@K, Recall@K, Hit Rate@K, MRR, MAP
        answer_metrics.py             # Fact coverage, citation precision, abstention correctness
    runners/
        evaluate_retrieval.py         # Comparative benchmark runner for retrieval strategies
        evaluate_answers.py           # End-to-end multi-agent answer quality evaluation runner
    reports/
        retrieval_benchmark_report.json # Detailed machine-readable JSON metrics
        retrieval_benchmark_summary.csv # CSV summary comparing baseline vs hybrid
        answer_evaluation_report.json   # Answer quality and safety evaluation results
```

---

## Dataset Taxonomy

The golden evaluation dataset (`medical_rag_golden_set.jsonl`) contains 32 curated queries spanning:
- **Direct Factual**: Core pharmacological and dosing guidelines.
- **Safety-Sensitive & Contraindications**: High-risk scenarios (e.g. bilateral renal artery stenosis + ACEi).
- **Multi-Document Reasoning**: Clinical queries requiring synthesis across multiple guideline sources.
- **Ambiguous Queries**: Broad clinical inquiries testing retrieval precision.
- **Out-of-Scope Queries**: Clinical conditions not covered in the knowledge base.
- **Insufficient Evidence**: Rare nephrology conditions requiring intentional system abstention.
- **Prompt Injection Attacks**: Adversarial instruction override attempts.

---

## How to Run Evaluations

### 1. Run Comparative Retrieval Benchmark
```bash
python -m evaluation.runners.evaluate_retrieval
```

### 2. Run End-to-End Answer & Safety Evaluation
```bash
python -m evaluation.runners.evaluate_answers
```
