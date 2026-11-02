"""Calling the guard from an app. Anything other than a clear answer counts as "block".

The guard sits in front of an LLM that may have tools. If it can't be reached, is slow, or
answers with something unexpected, letting the prompt through unchecked is the worse mistake
(decision 6), so the default fails closed. An app can choose to fail open, explicitly.
"""

from dataclasses import dataclass

import httpx

DECISIONS = {"allow", "review", "block"}


@dataclass(frozen=True)
class Verdict:
    decision: str
    reasons: tuple[str, ...]
    checked: bool  # False when the guard didn't give a usable answer


def check(
    text: str,
    url: str,
    api_key: str,
    timeout: float = 3.0,
    fail_open: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> Verdict:
    fallback = Verdict("allow" if fail_open else "block", ("guard_unavailable",), checked=False)
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(
                url.rstrip("/") + "/v1/check", json={"text": text}, headers={"X-API-Key": api_key}
            )
        if response.status_code != 200:
            return fallback
        body = response.json()
        decision, reasons = body.get("decision"), body.get("reasons", [])
        if decision not in DECISIONS or not isinstance(reasons, list):
            return fallback
        return Verdict(decision, tuple(str(r) for r in reasons), checked=True)
    except (httpx.HTTPError, ValueError, AttributeError):
        return fallback
