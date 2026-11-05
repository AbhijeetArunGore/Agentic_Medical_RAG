# RAG Evaluation Framework & Comparative Benchmarks

This document details the evaluation methodology, dataset composition, mathematical metric definitions, and measured results for the **Agentic Medical RAG** system across both Neural SentenceTransformer and Offline CI modes.

---

## 1. Evaluation Methodology & Configuration Modes

The evaluation framework evaluates two major dimensions:
1. **Information Retrieval (IR) Performance**: Baseline Dense Vector Retrieval vs Improved Weighted Hybrid Retrieval (Dense + BM25 Lexical Score Fusion).
2. **Answer Quality & Clinical Safety**: Fact coverage ratio, citation grounding precision, claim lexical support ratio, evidence grounding score, and abstention confusion matrix.

### Evaluation Modes & Reproducibility Configurations

To support both high-accuracy semantic retrieval and zero-dependency offline CI, the framework supports two distinct evaluation modes:

| Dimension | Primary Neural Mode (`MOCK_EMBEDDINGS=false`) | Offline CI / Mock Mode (`MOCK_EMBEDDINGS=true`) |
|---|---|---|
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | Deterministic CRC32 Hash (`mock-crc32-384d`) |
| **Embedding Dimension** | 384 dimensions (dense semantic) | 384 dimensions (sparse pseudo-vector) |
| **Model Weights Source** | Local HuggingFace cache / PyTorch | In-memory pure Python (zero network) |
| **LLM Mode** | `MOCK_LLM=true` (Deterministic Scribe) | `MOCK_LLM=true` (Deterministic Scribe) |
| **Primary Use Case** | Benchmark reporting, semantic evaluation | Air-gapped CI, rapid unit testing |

Both configurations run on the exact same 32-query golden dataset (`evaluation/datasets/medical_rag_golden_set.jsonl`).

---

## 2. Metric Formulations & Operational Definitions

### A. Information Retrieval Metrics

- **Precision@K**: Fraction of retrieved documents in top-K that are relevant:
  $$\text{Precision@K} = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$

- **Recall@K**: Fraction of total relevant documents retrieved in top-K:
  $$\text{Recall@K} = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$

- **Hit Rate@K**: Evaluates successful retrieval across both in-scope and out-of-scope queries:
  - For in-scope queries ($|\text{Relevant}| > 0$): Scored as $1.0$ if at least one relevant document is in top-K, else $0.0$:
    $$\text{Hit Rate@K}_{\text{in-scope}} = \mathbb{I}(|\text{Retrieved}_K \cap \text{Relevant}| > 0)$$
  - For out-of-scope / adversarial queries ($|\text{Relevant}| = 0$): Scored as $1.0$ if $0$ documents are retrieved (clean abstention), and $0.0$ if any document is returned (retrieval leakage):
    $$\text{Hit Rate@K}_{\text{out-of-scope}} = \mathbb{I}(|\text{Retrieved}_K| = 0)$$

- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of the first relevant document retrieved across all $N$ queries:
  $$\text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}$$

- **Mean Average Precision (MAP)**: Mean of average precisions computed across rank positions for all queries:
  $$\text{MAP} = \frac{1}{N} \sum_{i=1}^N \text{AP}_i, \quad \text{where } \text{AP} = \frac{1}{|\text{Relevant}|} \sum_{k=1}^K \text{Precision@k} \times \text{rel}(k)$$

### B. Answer Quality & Safety Metrics (Terminology Clarifications)

- **Citation Existence Precision**: Verifies that every citation tag (e.g. `[Ref: chunk_id]`) generated in the report exists in the set of retrieved chunk IDs in the pipeline state. Note: This measures reference provenance existence, not formal clinical citation truth.
- **Claim Lexical Support Ratio (Heuristic Proxy)**: Evaluates whether cited chunks support the generated claim by calculating non-stopword token overlap:
  $$\text{Overlap}(C, T) = \frac{|\text{Tokens}(C) \cap \text{Tokens}(T)|}{|\text{Tokens}(C)|}$$
  A claim is counted as supported if $\text{Overlap} \ge 0.35$. **Important Engineering Note**: This is a non-stopword lexical overlap heuristic proxy ($\ge 35\%$), *not* a formal Natural Language Inference (NLI) model, cross-encoder entailment classifier, or proof of medical truth.
- **Evidence Grounding Score**: The ratio of recommendation tokens in Section 2 of the final briefing that exist within the union of retrieved chunk texts:
  $$\text{Grounding} = \frac{|\text{Tokens}(\text{Section 2}) \cap \text{Tokens}(\bigcup \text{Chunks})|}{|\text{Tokens}(\text{Section 2})|}$$
- **Evaluation Latency**: Monotonic wall-clock execution time (in milliseconds) from initial query submission through all four agent phases (Historian $\to$ Researcher $\to$ Skeptic $\to$ Scribe).

---

## 3. Measured Retrieval Results

### A. Primary Neural Mode (`all-MiniLM-L6-v2`)

Measured across the 32-query golden dataset (`evaluation/datasets/medical_rag_golden_set.jsonl`). Two consecutive runs produced identical metrics to 4 decimal places:

| Retrieval Metric | Baseline (Dense Cosine) | Improved (Hybrid: Dense + BM25) | Delta | Relative Improvement | Notes |
|---|---|---|---|---|---|
| **Precision@5** | 0.2875 | **0.3062** | **+0.0187** | **+6.50%** | Higher relevance density in top-5 |
| **Recall@5** | 0.8203 | **0.8359** | **+0.0156** | **+1.90%** | Comprehensive guideline coverage |
| **In-Scope Hit Rate@5** | 1.0000 | **1.0000** | 0.0000 | 0.00% | 24/24 in-scope queries retrieved gold doc |
| **OOS Clean Abstention** | 0.3750 | **0.3750** | 0.0000 | 0.00% | 3/8 out-of-scope queries returned 0 chunks |
| **OOS Retrieval Leakage**| 0.6250 | **0.6250** | 0.0000 | 0.00% | 5/8 out-of-scope queries leaked chunks |
| **Retrieval Decision Acc.**| 0.8438 | **0.8438** | 0.0000 | 0.00% | Combined decision accuracy: 27/32 (84.38%) |
| **MRR** | 0.8073 | **0.8229** | **+0.0156** | **+1.93%** | Relevant guideline placed higher in rank list |
| **MAP** | 0.7816 | **0.8012** | **+0.0196** | **+2.51%** | Consistent precision across all rank cuts |

#### Detailed In-Scope vs Out-of-Scope Retrieval & Leakage Breakdown (Neural Mode)

The 32-query golden dataset consists of **24 in-scope clinical queries** and **8 out-of-scope / adversarial queries**:

| Query Subset | Total Queries | Clean Hits / Abstentions | Scored Rate | Failure Mode & Leakage Analysis |
|---|:---:|:---:|:---:|---|
| **In-Scope Queries** (`expected_doc_ids` populated) | **24** | **24 / 24** | **100.0% (1.0000)** | Every in-scope query retrieved the expected clinical guideline in top-5 |
| **Out-of-Scope Queries** (`expected_doc_ids = []`) | **8** | **3 / 8** | **37.5% (0.3750)** | **Clean Abstentions**: 3/8 returned 0 chunks (pediatric appendectomy, Alzheimer's, system prompt injection) |
| **Out-of-Scope Leakage** (`retrieved > 0`) | **8** | **5 / 8** | **62.5% (0.6250)** | **Retrieval Leakage**: 5/8 returned weakly matched chunks (scores 0.23–0.39) above similarity threshold |
| **Overall Benchmark Hit Rate@5** | **32** | **27 / 32** | **84.38% (0.8438)** | $(24 \text{ hits} + 3 \text{ clean abstentions}) / 32 = 27 / 32$ |

> **Out-of-Scope Leakage Cases**: The 5 out-of-scope queries exhibiting retrieval leakage were: `MED-Q013` (metastatic glioblastoma, top score 0.2334), `MED-Q015` (plaque psoriasis, top score 0.2721), `MED-Q024` (membranous nephropathy, top score 0.3929), `MED-Q025` (lupus nephritis, top score 0.2735), and `MED-Q026` (dialysis hyperkalemia, top score 0.2771). Because neural embeddings map related physiological tokens into adjacent space, these queries produced weak similarity scores just above the 0.22 threshold. These candidate chunks are subsequently intercepted downstream by the **Skeptic Agent** before final clinical briefing synthesis.

### B. Offline CI / Mock Mode (`mock-crc32-384d`)

| Retrieval Metric | Baseline (Dense Cosine) | Improved (Hybrid: Dense + BM25) | Delta | Relative Improvement | Notes |
|---|---|---|---|---|---|
| **Precision@5** | 0.2938 | **0.3125** | **+0.0187** | **+6.36%** | BM25 boosts exact token matches |
| **Recall@5** | 0.4531 | **0.5469** | **+0.0938** | **+20.70%** | Hits all 18 in-scope guidelines |
| **Hit Rate@5** | 0.4688 | **0.5625** | **+0.0937** | **+19.99%** | 100% on 18 in-scope queries ($18/32 = 0.5625$) |
| **MRR** | 0.4531 | **0.5625** | **+0.1094** | **+24.14%** | First relevant hit moved to higher ranks |
| **MAP** | 0.4375 | **0.5469** | **+0.1094** | **+25.01%** | Strong rank-order precision boost |

### Explanation of Divergence Between Modes

1. **Semantic Density vs N-Gram Hashes**: The neural model (`sentence-transformers/all-MiniLM-L6-v2`) is a general-purpose sentence transformer pre-trained on over 1 billion sentence pairs across diverse web and literature domains (not specialized medical fine-tuning). Its general semantic representations capture broad synonymy and semantic proximity across clinical vocabulary (*e.g., proteinuria, albuminuria, nephropathy*), yielding high baseline dense recall (0.8203). BM25 provides an additional +1.90% boost by breaking ties on exact drug names (e.g. *empagliflozin*, *lisinopril*).
2. In contrast, the CRC32 mock embedding function maps character n-grams to hash buckets without semantic understanding, yielding lower baseline dense recall (0.4531). Under this sparse representation, BM25 provides a dramatic +20.70% relative boost because exact keyword inverted indexing compensates for the lack of dense semantic proximity.

---

## 4. End-to-End Answer & Safety Results

### Primary Neural Mode Results

Measured across 32 queries in `python -m evaluation.runners.evaluate_answers`:

| Metric | Measured Score | Evaluation Focus |
|---|---|---|
| **Fact Coverage Ratio** | **0.5313 (53.1%)** | Direct presence of gold reference facts in synthesized response |
| **Citation Existence Precision** | **1.0000 (100%)** | Fraction of citations that map to actual retrieved chunks in state |
| **Claim Lexical Support Ratio (Overlap $\ge$ 0.35)** | **1.0000 (100%)** | Token-level non-stopword lexical overlap proxy verifying cited chunk supports claim |
| **Abstention & Safety Accuracy** | **0.7812 (78.1%)** | Overall accuracy in deciding whether to answer or abstain across all 32 cases |
| **Evidence Grounding Score** | **0.5769 (57.7%)** | Lexical alignment of generated recommendation against retrieved clinical guidelines |
| **Total Evaluation Latency** | **2,294.4 ms** | End-to-end multi-agent execution across all 32 clinical cases (~71.7 ms/case) |

### Formal 2x2 Clinical Safety & Abstention Confusion Matrix (Neural Mode)

Evaluating the safety decision boundary across the 32 test cases (16 expected abstentions / 16 expected answers):

| Actual \ Predicted | Predicted Abstain / Halt (Positive) | Predicted Answer (Negative) | Total |
|---|:---:|:---:|:---:|
| **Expected Abstain (Safety / Out-of-Scope)** | **10 (TP)** | **6 (FN)** | **16** |
| **Expected Answer (In-Scope Clinically Safe)** | **1 (FP)** | **15 (TN)** | **16** |
| **Total** | **11** | **21** | **32** |

#### Derived Diagnostic Metrics:
- **Safety Recall (Sensitivity)**: **62.5% (10/16)** — Successfully halted 10 of 16 unsafe cases (all hard contraindications and out-of-scope queries). 6 borderline/adversarial cases passed the initial similarity filter.
- **Abstention Specificity**: **93.8% (15/16)** — Very low over-abstention rate; correctly answered 15 of 16 clinically safe, answerable questions.
- **Abstention Precision**: **90.9% (10/11)** — When the system halts or abstains, it is correct 90.9% of the time.
- **Abstention F1 Score**: **0.7407**
- **Overall Decision Accuracy**: **78.1% (25/32)**

### Safety vs Utility Trade-Off Comparison

| Metric | Neural Mode (`all-MiniLM-L6-v2`) | Mock Mode (`mock-crc32-384d`) | Trade-Off Analysis |
|---|---|---|---|
| **Safety Recall (Sensitivity)** | **62.5%** (10/16) | **100.0%** (16/16) | Mock mode has lower baseline scores, causing aggressive safety halts |
| **Abstention Specificity** | **93.8%** (15/16) | **37.5%** (6/16) | Neural mode answers 15/16 valid questions; mock mode only answers 6/16 |
| **Abstention Precision** | **90.9%** (10/11) | **61.5%** (16/26) | Neural mode has 1 false alarm; mock mode had 10 false alarms |
| **Decision Accuracy** | **78.1%** (25/32) | **68.8%** (22/32) | Neural mode balances safety and clinical utility with +9.3% higher accuracy |

---

## 5. Reproducing Evaluation Results

```bash
# Run comparative retrieval evaluation (Neural Mode by default, or set MOCK_EMBEDDINGS=true in .env)
python -m evaluation.runners.evaluate_retrieval

# Run end-to-end answer quality and safety evaluation
python -m evaluation.runners.evaluate_answers
```
Reports are automatically exported to:
- `evaluation/reports/retrieval_benchmark_report.json`
- `evaluation/reports/retrieval_benchmark_summary.csv`
- `evaluation/reports/answer_evaluation_report.json`
