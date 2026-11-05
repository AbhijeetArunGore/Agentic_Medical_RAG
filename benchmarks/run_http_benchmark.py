"""Production-grade HTTP load and latency benchmark for Agentic Medical RAG.

Measures real-world HTTP network performance across:
- GET /health (lightweight operational probe)
- GET /ready (vector store readiness probe)
- POST /query (single-turn hybrid retrieval + RAG synthesis)
- POST /evaluate-patient (multi-agent 4-agent LangGraph execution)

Uses real HTTP requests over TCP loopback with warm-up, configurable concurrency,
and >= 1,000 requests for robust statistical measurement.
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

REPORTS_DIR = ROOT_DIR / "benchmarks" / "reports"


def send_http_request(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> tuple[int, float]:
    """Sends an HTTP request and measures round-trip time in milliseconds."""
    headers = {
        "User-Agent": "AgenticMedicalRAG-Benchmark/1.0",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = resp.status
            _ = resp.read()
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return status_code, duration_ms
    except urllib.error.HTTPError as e:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return e.code, duration_ms
    except Exception:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return 500, duration_ms


def run_concurrent_benchmark(
    name: str,
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    total_requests: int = 1000,
    concurrency: int = 5,
    warmup_requests: int = 50,
) -> Dict[str, Any]:
    """Executes concurrent load test with warm-up and percentile tracking."""
    print(f"\n--- Running Benchmark: {name} ---")
    print(f"  Target: {method} {url}")
    print(
        f"  Total Requests: {total_requests} | Concurrency: {concurrency} | Warm-up: {warmup_requests}"
    )

    # 1. Warm-up phase
    if warmup_requests > 0:
        for _ in range(warmup_requests):
            send_http_request(url, method=method, payload=payload, timeout=15.0)

    # 2. Measurement phase
    latencies: List[float] = []
    status_codes: Dict[int, int] = {}
    successes = 0
    failures = 0

    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_http_request, url, method, payload, 15.0)
            for _ in range(total_requests)
        ]
        for f in concurrent.futures.as_completed(futures):
            code, lat = f.result()
            latencies.append(lat)
            status_codes[code] = status_codes.get(code, 0) + 1
            if 200 <= code < 400:
                successes += 1
            else:
                failures += 1

    total_time_sec = max(0.001, time.perf_counter() - t_start)
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
    max_lat = round(max(latencies), 2) if latencies else 0.0
    mean_lat = round(statistics.mean(latencies), 2) if latencies else 0.0
    rps = round(len(latencies) / total_time_sec, 2)
    error_rate = round((failures / max(1, len(latencies))) * 100, 2)

    result = {
        "operation": name,
        "endpoint": f"{method} {url}",
        "concurrency": concurrency,
        "total_requests": len(latencies),
        "successful_requests": successes,
        "failed_requests": failures,
        "error_rate_pct": error_rate,
        "throughput_rps": rps,
        "duration_seconds": round(total_time_sec, 2),
        "mean_latency_ms": mean_lat,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_latency_ms": max_lat,
        "status_code_distribution": status_codes,
    }

    print(
        f"  -> RPS: {rps} | p50: {p50}ms | p90: {p90}ms | p95: {p95}ms | p99: {p99}ms | Error: {error_rate}%"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Real HTTP Benchmark for Agentic Medical RAG")
    parser.add_argument("--host", default="127.0.0.1", help="Target host")
    parser.add_argument("--port", type=int, default=8000, help="Target port")
    parser.add_argument(
        "--health-requests", type=int, default=1000, help="Requests for /health benchmark"
    )
    parser.add_argument(
        "--query-requests", type=int, default=100, help="Requests for /query benchmark"
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent client threads")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("=== AGENTIC MEDICAL RAG: REAL HTTP API BENCHMARK ===")
    print(f"Target Server: {base_url}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Health Benchmark Volume: {args.health_requests} requests")
    print(f"Query Benchmark Volume: {args.query_requests} requests")
    print("=" * 80)

    # 1. Health Probe Verification
    try:
        code, _ = send_http_request(f"{base_url}/health")
        if code != 200:
            print(f"ERROR: Server at {base_url}/health returned status {code}. Aborting benchmark.")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot connect to {base_url}/health ({e}). Ensure server is running.")
        sys.exit(1)

    benchmarks = []

    # Benchmark 1: GET /health (1,000 requests)
    b1 = run_concurrent_benchmark(
        name="GET /health (Liveness Probe)",
        url=f"{base_url}/health",
        method="GET",
        total_requests=args.health_requests,
        concurrency=args.concurrency,
        warmup_requests=50,
    )
    benchmarks.append(b1)

    # Benchmark 2: GET /ready (200 requests)
    b2 = run_concurrent_benchmark(
        name="GET /ready (Readiness & DB Probe)",
        url=f"{base_url}/ready",
        method="GET",
        total_requests=min(200, args.health_requests),
        concurrency=args.concurrency,
        warmup_requests=10,
    )
    benchmarks.append(b2)

    # Benchmark 3: POST /query (100 requests)
    b3 = run_concurrent_benchmark(
        name="POST /query (Hybrid RAG Pipeline)",
        url=f"{base_url}/query",
        method="POST",
        payload={
            "query": "What is the recommended treatment for Type 2 Diabetes with renal impairment?",
            "patient_id": "BENCH-PAT-01",
        },
        total_requests=args.query_requests,
        concurrency=min(5, args.concurrency),
        warmup_requests=10,
    )
    benchmarks.append(b3)

    # Benchmark 4: POST /evaluate-patient (25 requests)
    b4 = run_concurrent_benchmark(
        name="POST /evaluate-patient (Multi-Agent LangGraph Graph)",
        url=f"{base_url}/evaluate-patient",
        method="POST",
        payload={
            "query": "Evaluate renal trends and verify guideline contraindications",
            "patient_id": "BENCH-PAT-02",
        },
        total_requests=min(25, args.query_requests),
        concurrency=2,
        warmup_requests=2,
    )
    benchmarks.append(b4)

    # Summary Report Table
    print("\n" + "=" * 105)
    print(
        f"{'Operation / Endpoint':<40} | {'Throughput':<12} | {'Mean (ms)':<10} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10} | {'Errors':<8}"
    )
    print("-" * 105)
    for b in benchmarks:
        print(
            f"{b['operation']:<40} | {b['throughput_rps']:>9.1f} rps | {b['mean_latency_ms']:>8.2f} ms | {b['p50_ms']:>8.2f} ms | {b['p95_ms']:>8.2f} ms | {b['p99_ms']:>8.2f} ms | {b['error_rate_pct']:>6.1f} %"
        )
    print("=" * 105 + "\n")

    report_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": sys.platform,
            "python_version": sys.version.split()[0],
            "server": "uvicorn",
            "concurrency": args.concurrency,
            "target_base_url": base_url,
        },
        "benchmarks": benchmarks,
    }

    out_file = REPORTS_DIR / "http_benchmark_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Benchmark results written to: {out_file}")


if __name__ == "__main__":
    main()
