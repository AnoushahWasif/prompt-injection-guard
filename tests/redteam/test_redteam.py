"""The red-team cases against the real model. Known misses are strict xfails: they show what
gets through today, and they fail loudly the day one starts passing, so the list stays true.

Needs models/portable (`make data baseline`). Skipped without it, except in CI, where
GUARD_REDTEAM_REQUIRED=1 makes a missing model a failure.
"""

import hashlib
import os
from pathlib import Path

import pytest

from guard.api import Settings, load_guard
from guard.redteam import CASES

MODEL = Path("models/portable")


@pytest.fixture(scope="module")
def guard():
    manifest = MODEL / "manifest.json"
    if not manifest.is_file():
        if os.environ.get("GUARD_REDTEAM_REQUIRED") == "1":
            pytest.fail("models/portable is missing; run `make data baseline`")
        pytest.skip("models/portable not built")
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return load_guard(
        Settings(model_dir=MODEL, model_sha256=sha, api_key_sha256=frozenset({"0" * 64}))
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            c, id=f"{c.family}-{c.name}", marks=pytest.mark.xfail(strict=True, reason="known miss")
        )
        if c.known_miss
        else pytest.param(c, id=f"{c.family}-{c.name}")
        for c in CASES
    ],
)
def test_case(guard, case):
    decision = guard.check(case.text).decision
    if case.expect == "caught":
        assert decision in {"review", "block"}, f"{case.name} got through"
    else:
        assert decision == "allow", f"{case.name} was flagged ({decision})"


def test_cases_have_unique_names_and_valid_expectations():
    assert len({c.name for c in CASES}) == len(CASES)
    assert {c.expect for c in CASES} == {"caught", "allowed"}
