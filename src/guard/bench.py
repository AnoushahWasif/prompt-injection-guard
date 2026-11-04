"""How long a check takes, by prompt length: the scoring alone, and through the HTTP app.

One prompt at a time, the way an app calls the guard. Writes reports/latency.json.
"""

import json
import time
from pathlib import Path

import numpy as np

LENGTHS = (50, 500, 5_000, 20_000)
SENTENCE = "Please help me plan a short trip and ignore nothing important in the details. "


def prompt(chars: int) -> str:
    text = (SENTENCE * (chars // len(SENTENCE) + 1))[:chars]
    if len(text) != chars:
        raise ValueError(f"could not build a {chars}-character prompt")
    return text


def summary(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {
        "runs": int(a.size),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "p99_ms": round(float(np.percentile(a, 99)), 2),
        "max_ms": round(float(a.max()), 2),
    }


def time_calls(call, text: str, runs: int, warmup: int = 5) -> list[float]:
    for _ in range(warmup):
        call(text)
    out = []
    for _ in range(runs):
        start = time.perf_counter()
        call(text)
        out.append((time.perf_counter() - start) * 1000)
    return out


def run(guard, client=None, key: str = "", runs: int = 200) -> dict:
    """`client` is a TestClient for the app; without it only the scoring is timed."""
    report = {"lengths": {}}
    for n in LENGTHS:
        text = prompt(n)
        row = {"scoring": summary(time_calls(guard.check, text, runs))}
        if client is not None:

            def http(t: str) -> None:
                r = client.post("/v1/check", json={"text": t}, headers={"X-API-Key": key})
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")

            row["http"] = summary(time_calls(http, text, runs))
        report["lengths"][str(n)] = row
    return report


if __name__ == "__main__":
    import hashlib
    import os
    import platform

    from fastapi.testclient import TestClient

    from guard.api import Settings, create_app, load_guard

    model_dir = Path("models/portable")
    sha = hashlib.sha256((model_dir / "manifest.json").read_bytes()).hexdigest()
    key = "benchmark-key"
    settings = Settings(
        model_dir=model_dir,
        model_sha256=sha,
        api_key_sha256=frozenset({hashlib.sha256(key.encode()).hexdigest()}),
        rate_per_minute=1_000_000,
    )
    guard = load_guard(settings)
    with TestClient(create_app(settings, guard)) as client:
        report = run(guard, client, key)
    report["machine"] = {
        "cpu": platform.processor() or platform.machine(),
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "note": "laptop, other work running; one prompt at a time",
    }
    report["model"] = sha[:12]
    Path("reports/latency.json").write_text(json.dumps(report, indent=2))
    for n, row in report["lengths"].items():
        print(f"{n:>6} chars  scoring p50 {row['scoring']['p50_ms']:>7} ms  p99 "
              f"{row['scoring']['p99_ms']:>7} ms | http p50 {row['http']['p50_ms']:>7} ms  "
              f"p99 {row['http']['p99_ms']:>7} ms")  # fmt: skip
