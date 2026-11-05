# API Performance Benchmarking Suite

This directory contains automated, reproducible benchmarking harnesses to measure microservice latency profiles, throughput, and error rates across all core operations:

1. `GET /health` (Service health status)
2. `GET /ready` (Vector store and collection readiness)
3. `POST /query` (Single-turn RAG retrieval and briefing generation)
4. `POST /evaluate-patient` (Full multi-agent LangGraph workflow execution)
5. `POST /documents/ingest` (Document chunking, hashing, and vector indexing)

---

## Benchmark Modes

- **MODE A: Offline Mock Mode (Default)**: Uses deterministic offline models and vector embeddings to measure core application and microservice routing overhead without cloud API costs or network jitter.
- **MODE B: Real Provider Mode**: Connects to the configured cloud LLM provider to measure end-to-end latency including remote model generation.

---

## Running Benchmarks

### 1. In-Process Python Benchmark Runner
```bash
python benchmarks/run_benchmark.py --iterations 50
```

### 2. Live Server TCP Loopback Benchmark (Automated Test of 1,325 Requests)
```bash
python benchmarks/run_live_server_benchmark.py
```

### 3. Distributed Load Testing with Locust
Start the server in one terminal:
```bash
python main.py --server
```
In another terminal, execute headless Locust load testing:
```bash
locust -f benchmarks/locustfile.py --headless -u 10 -r 2 -t 30s --host http://localhost:8000
```

---

## Output Reports

- `benchmarks/reports/api_benchmark_report.json`: Full latency percentiles (p50, p90, p95, p99), RPS, and error metrics.
- `benchmarks/reports/api_benchmark_summary.csv`: Tabular CSV summary.
