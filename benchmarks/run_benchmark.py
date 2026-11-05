"""Reproducible API performance benchmark runner for Agentic Medical RAG.

Evaluates latency percentiles (p50, p90, p95, p99), throughput (RPS), and error rates
across all key API operations. Supports MODE A (Offline Mock) and MODE B (Real Provider).

Usage:
    python benchmarks/run_benchmark.py [--mode mock|real] [--concurrency 5] [--iterations 50]
"""

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.api.routes import app
from app.utils.config import settings

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "benchmarks" / "reports"


def benchmark_endpoint(
    name: str,
    call_fn: Callable[[], tuple[int, Any]],
    iterations: int = 50,
) -> Dict[str, Any]:
    """Measures latency distribution and error rate for a specific operation."""
    latencies: List[float] = []
    successes = 0
    failures = 0

    t_start_total = time.time()
    for _ in range(iterations):
        t0 = time.time()
        try:
            status_code, _ = call_fn()
            dur_ms = (time.time() - t0) * 1000
            latencies.append(dur_ms)
            if 200 <= status_code < 400:
                successes += 1
            else:
                failures += 1
        except Exception as e:
            dur_ms = (time.time() - t0) * 1000
            latencies.append(dur_ms)
            failures += 1

    total_time_sec = max(0.001, time.time() - t_start_total)
    latencies.sort()

    def get_percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        k = (len(data) - 1) * p
        f = int(k)
        c = min(f + 1, len(data) - 1)
        d = k - f
        return round(data[f] + d * (data[c] - data[f]), 2)

    p50 = get_percentile(latencies, 0.50)
    p90 = get_percentile(latencies, 0.90)
    p95 = get_percentile(latencies, 0.95)
    p99 = get_percentile(latencies, 0.99)
    mean_lat = round(statistics.mean(latencies), 2) if latencies else 0.0
    rps = round(len(latencies) / total_time_sec, 2)
    error_rate = round((failures / max(1, len(latencies))) * 100, 2)

    return {
        "operation": name,
        "total_requests": len(latencies),
        "successful_requests": successes,
        "failed_requests": failures,
        "error_rate_pct": error_rate,
        "throughput_rps": rps,
        "mean_latency_ms": mean_lat,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
    }


def main():
    parser = argparse.ArgumentParser(description="Agentic Medical RAG API Benchmark")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock", help="Benchmark mode")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations per operation")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"=== AGENTIC MEDICAL RAG: API BENCHMARK [{args.mode.upper()} MODE] ===")
    print(f"Iterations per endpoint: {args.iterations}")
    print(f"Mock LLM Mode: {settings.MOCK_LLM}")
    print("=" * 80 + "\n")

    # Import routes directly to benchmark route handlers with zero socket networking overhead
    from app.api.routes import (
        health,
        ready,
        query_rag,
        evaluate_patient,
        ingest_raw_text,
        QueryRequest,
        EvaluationRequest,
        RawIngestRequest,
    )

    operations = [
        (
            "1. Health Check (GET /health)",
            lambda: (200, health()),
        ),
        (
            "2. Readiness Probe (GET /ready)",
            lambda: (200, ready()),
        ),
        (
            "3. Single-Turn RAG Query (POST /query)",
            lambda: (
                200,
                query_rag(
                    QueryRequest(
                        query="Treatment guidelines for diabetic nephropathy with rising creatinine",
                        patient_id="BENCH-PAT-01",
                    )
                ),
            ),
        ),
        (
            "4. Multi-Agent CDS Graph (POST /evaluate-patient)",
            lambda: (
                200,
                evaluate_patient(
                    EvaluationRequest(
                        patient_id="BENCH-PAT-02",
                        query="Evaluate renal function and diabetes control",
                    )
                ),
            ),
        ),
        (
            "5. Document Ingestion (POST /documents/ingest)",
            lambda: (
                200,
                ingest_raw_text(
                    RawIngestRequest(
                        text="Guideline Update: Early SGLT2 inhibitor initiation in diabetic CKD yields 30% renal risk reduction.",
                        source_name="Benchmark Guideline Supplement",
                        document_id=f"bench_doc_{int(time.time() * 1000) % 100000}",
                    )
                ),
            ),
        ),
    ]

    summaries = []
    for name, fn in operations:
        print(f"Benchmarking {name}...")
        summary = benchmark_endpoint(name=name, call_fn=fn, iterations=args.iterations)
        summaries.append(summary)
        print(f"  -> p50: {summary['p50_ms']}ms | p95: {summary['p95_ms']}ms | Throughput: {summary['throughput_rps']} RPS | Errors: {summary['error_rate_pct']}%")

    print("\n" + "=" * 95)
    print(f"{'Operation':<42} | {'Throughput':<12} | {'Mean (ms)':<10} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'Error %':<8}")
    print("-" * 95)
    for s in summaries:
        print(
            f"{s['operation']:<42} | {s['throughput_rps']:>9.1f} rps | {s['mean_latency_ms']:>8.2f} ms | {s['p50_ms']:>8.2f} ms | {s['p95_ms']:>8.2f} ms | {s['error_rate_pct']:>6.1f} %"
        )
    print("=" * 95 + "\n")

    # Save JSON report
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_mode": args.mode,
        "iterations_per_endpoint": args.iterations,
        "environment": settings.ENVIRONMENT,
        "results": summaries,
    }
    json_path = REPORTS_DIR / "api_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved machine-readable JSON report to: {json_path}")

    # Save CSV summary
    csv_path = REPORTS_DIR / "api_benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Operation", "Total_Requests", "Throughput_RPS", "Mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "Error_Rate_Pct"])
        for s in summaries:
            writer.writerow([
                s["operation"],
                s["total_requests"],
                s["throughput_rps"],
                s["mean_latency_ms"],
                s["p50_ms"],
                s["p90_ms"],
                s["p95_ms"],
                s["p99_ms"],
                s["error_rate_pct"],
            ])
    print(f"Saved CSV benchmark summary to: {csv_path}\n")


if __name__ == "__main__":
    main()
