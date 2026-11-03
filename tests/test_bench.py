import hashlib

import pytest
from fastapi.testclient import TestClient

from guard import api, bench


@pytest.mark.parametrize("n", bench.LENGTHS)
def test_prompts_have_exactly_the_requested_length(n):
    assert len(bench.prompt(n)) == n


def test_summary_percentiles():
    s = bench.summary([float(x) for x in range(1, 101)])
    assert s["runs"] == 100 and s["max_ms"] == 100
    assert s["p50_ms"] == pytest.approx(50.5)


def test_run_times_scoring_and_http(monkeypatch):
    monkeypatch.setattr(bench, "LENGTHS", (50, 500))
    guard = api.Guard(lambda t: 0.1, block_at=0.8, review_at=0.4)
    settings = api.Settings(
        model_dir=None,
        model_sha256="0" * 64,
        api_key_sha256=frozenset({hashlib.sha256(b"k").hexdigest()}),
        rate_per_minute=10_000,
    )
    with TestClient(api.create_app(settings, guard)) as client:
        report = bench.run(guard, client, "k", runs=5)
    assert set(report["lengths"]) == {"50", "500"}
    assert report["lengths"]["50"]["http"]["runs"] == 5


def test_a_failed_http_call_stops_the_benchmark():
    guard = api.Guard(lambda t: 0.1, block_at=0.8, review_at=0.4)
    settings = api.Settings(
        model_dir=None, model_sha256="0" * 64, api_key_sha256=frozenset({"1" * 64})
    )
    with TestClient(api.create_app(settings, guard)) as client, pytest.raises(RuntimeError):
        bench.run(guard, client, "wrong-key", runs=1)
