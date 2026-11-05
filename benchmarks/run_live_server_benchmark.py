"""Live server benchmark orchestration script.

Starts an ephemeral Uvicorn server process on 127.0.0.1:8765, executes the HTTP load
benchmark (1,000+ requests across endpoints), and ensures guaranteed server process cleanup.
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    port = 8765
    server_url = f"http://127.0.0.1:{port}"

    print(f"Starting Uvicorn server subprocess on {server_url}...")
    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.api.routes:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]

    proc = subprocess.Popen(server_cmd, cwd=str(ROOT_DIR))

    try:
        # Wait for server to become responsive
        max_attempts = 30
        is_ready = False
        for attempt in range(max_attempts):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"{server_url}/health", timeout=1.0) as resp:
                    if resp.status == 200:
                        is_ready = True
                        print(
                            f"Uvicorn server ready on {server_url} after {(attempt + 1) * 0.5:.1f}s."
                        )
                        break
            except Exception:
                continue

        if not is_ready:
            print("ERROR: Uvicorn server failed to become responsive within 15 seconds.")
            sys.exit(1)

        # Run HTTP load benchmark
        bench_cmd = [
            sys.executable,
            "-m",
            "benchmarks.run_http_benchmark",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--health-requests",
            "1000",
            "--query-requests",
            "100",
            "--concurrency",
            "5",
        ]
        print("\nExecuting HTTP load benchmark...")
        ret = subprocess.run(bench_cmd, cwd=str(ROOT_DIR))
        if ret.returncode != 0:
            print(f"ERROR: Benchmark exited with code {ret.returncode}")
            sys.exit(ret.returncode)

    finally:
        print("\nTerminating Uvicorn server process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server process terminated cleanly.")


if __name__ == "__main__":
    main()
