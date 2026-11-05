# Changelog

All notable changes to **Agentic Medical RAG** are documented in this file.

---

## [1.1.0] - 2026-11-05

### Added
- **Hybrid Retrieval Engine**: Implemented pure-Python Okapi BM25 index and Weighted Hybrid Score Fusion (0.60 * S_dense + 0.40 * S_bm25) combining dense SentenceTransformers embeddings with normalized BM25 lexical token coverage.
- **Configurable Chunking Pipeline**: Built `MedicalChunker` supporting `recursive`, `fixed`, and `structure_aware` clinical section chunking with deterministic SHA256 content hashing.
- **Document Ingestion Module**: Built `DocumentIngestor` supporting `.txt`, `.md`, `.json`, and `.pdf` (via `pypdf`) with file size validation, duplicate detection, and empty-file safeguards.
- **RAG Evaluation Suite**:
  - Golden dataset with 32 categorized clinical test cases (`evaluation/datasets/medical_rag_golden_set.jsonl`).
  - Retrieval metrics: Precision@K, Recall@K, Hit Rate@K, MRR, MAP (`evaluation/metrics/retrieval_metrics.py`).
  - Answer quality metrics: Fact coverage ratio, citation grounding precision, evidence grounding score, abstention accuracy (`evaluation/metrics/answer_metrics.py`).
  - Reproducible runners generating JSON and CSV benchmark reports.
- **API Performance Benchmark Harness**:
  - In-process benchmark runner (`benchmarks/run_benchmark.py`) measuring p50, p90, p95, p99 latency percentiles and RPS across all endpoints.
  - Locust scenario file (`benchmarks/locustfile.py`) for headless distributed load testing.
- **Automated Test Suite**: Expanded test coverage from 7 tests to 54 comprehensive tests covering unit, integration, API, agent behaviors, and safety guardrails.
- **Prompt Injection & Adversarial Defenses**: Added regular expression and semantic pattern checks in `SkepticAgent` to intercept instruction overrides, DAN jailbreaks, and policy evasion.
- **Clinical Negation Handling**: Enhanced contraindication checking with regex negation detection to eliminate false-positive contraindication flags on notes stating "No stenosis".
- **Observability & Health Probes**: Added `GET /ready` and `GET /diagnostics` endpoints with request ID tracking (`X-Request-ID`) and response timing headers (`X-Response-Time-MS`).
- **Hardened Docker Setup**: Created multi-stage Dockerfile with non-root user (`appuser`, UID 10001), healthcheck probes, and Docker Compose with service health dependency.
- **Security CI Workflow**: Added `.github/workflows/security.yml` for automated secret scanning and vulnerability checks.

### Fixed
- **Git Hygiene**: Untracked runtime SQLite and binary files (`vector_db/chroma.sqlite3`) from git tracking and updated `.gitignore`.
- **Vanity Metrics in Dashboard**: Replaced hardcoded static metrics (`98.5%`, `96.2%`, `0.01%`) in `app/dashboard.py` with live execution latency, strategy telemetry, and honest evaluation guidance.
- **Windows / Network Sandbox Blocking**: Fixed module-level eager initialization in `app.api.routes` by introducing lazy dependency singletons and offline mock mode defaults.
- **Citation Provenance**: Scribe now maps each claim to an actual retrieved chunk ID and document source rather than generic truncated strings.
