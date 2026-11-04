"""The guard as a small HTTP service: POST /v1/check -> allow, review or block.

What it protects against (ARCHITECTURE.md, STRIDE table):
- Spoofing: every call needs an API key. Only SHA-256 hashes of keys are configured, and they
  are compared in constant time.
- Information disclosure: the prompt text is never logged, the score is never returned (it
  would tell an attacker how close they are), and a malformed request is not echoed back.
- Denial of service: a size cap, a rate limit per key, a cap on how many prompts are scored at
  once, and a time limit. When the guard can't answer it says so with a 503, and the client
  blocks (decision 6). A slow prompt keeps its scoring slot until it really finishes, so
  timed-out work can't pile up behind the cap.
- Tampering with the model: it is loaded only if its manifest's hash matches the one pinned
  in config, and only from the portable format, so nothing is ever unpickled.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from guard.normalize import MAX_CHARS, normalize

log = logging.getLogger("guard.api")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    model_sha256: str
    api_key_sha256: frozenset[str]
    rate_per_minute: int = 120
    max_concurrent: int = 4
    timeout_seconds: float = 2.0
    max_body_bytes: int = 4 * MAX_CHARS + 1_000  # up to 4 bytes per character, plus JSON

    def __post_init__(self):
        if not HEX64.match(self.model_sha256):
            raise ValueError("model_sha256 must be 64 lower-case hex characters")
        if not self.api_key_sha256:
            raise ValueError("at least one API key hash is required; the guard never runs open")
        if not all(HEX64.match(h) for h in self.api_key_sha256):
            raise ValueError("API key hashes must be 64 lower-case hex characters")
        if self.rate_per_minute < 1 or self.max_concurrent < 1 or self.timeout_seconds <= 0:
            raise ValueError("limits must be positive")

    @classmethod
    def from_env(cls, env=os.environ) -> "Settings":
        try:
            return cls(
                model_dir=Path(env.get("GUARD_MODEL_DIR", "models/portable")),
                model_sha256=env["GUARD_MODEL_SHA256"].strip().lower(),
                api_key_sha256=frozenset(
                    h.strip().lower() for h in env["GUARD_API_KEY_SHA256"].split(",") if h.strip()
                ),
                rate_per_minute=int(env.get("GUARD_RATE_PER_MINUTE", 120)),
                max_concurrent=int(env.get("GUARD_MAX_CONCURRENT", 4)),
                timeout_seconds=float(env.get("GUARD_TIMEOUT_SECONDS", 2.0)),
            )
        except KeyError as e:
            raise ValueError(f"{e.args[0]} is not set") from e


# --- deciding


@dataclass(frozen=True)
class Decision:
    decision: str  # allow, review or block
    reasons: tuple[str, ...] = ()


@dataclass
class Guard:
    """Turns a score into allow / review / block. `score` takes normalised text."""

    score: Callable[[str], float]
    block_at: float
    review_at: float
    version: str = "unknown"

    def __post_init__(self):
        if not 0 < self.review_at <= self.block_at <= 1:
            raise ValueError("need 0 < review threshold <= block threshold <= 1")

    def check(self, text: str) -> Decision:
        found = normalize(text)
        reasons = []
        if found.tag_chars_decoded:
            reasons.append("hidden_instructions")  # Unicode tag smuggling: never normal
        elif found.suspicious:
            reasons.append("unusual_characters")
        if not found.text:
            return Decision("review" if reasons else "allow", tuple(reasons))
        s = self.score(found.text)
        if s >= self.block_at:
            return Decision("block", ("looks_like_injection", *reasons))
        if s >= self.review_at:
            return Decision("review", ("possible_injection", *reasons))
        if found.tag_chars_decoded:
            return Decision("review", tuple(reasons))
        return Decision("allow", tuple(reasons))


def load_guard(settings: Settings) -> Guard:
    """Only the portable format (guard.portable): serving never unpickles anything."""
    from guard.portable import load

    loaded = load(settings.model_dir, settings.model_sha256)
    model = loaded.model
    return Guard(
        score=lambda text: float(model.predict_proba([text])[0, 1]),
        block_at=loaded.threshold,
        review_at=loaded.review,
        version=settings.model_sha256[:12],
    )


# --- limits


class RateLimiter:
    """At most `per_minute` calls per key in any 60-second window. Keys are the configured ones,
    so memory is bounded by config, not by callers."""

    def __init__(self, per_minute: int, clock=time.monotonic):
        self.per_minute = per_minute
        self.clock = clock
        self.calls: dict[str, deque] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        now = self.clock()
        calls = self.calls.setdefault(key, deque())
        while calls and now - calls[0] >= 60:
            calls.popleft()
        if len(calls) >= self.per_minute:
            return False, max(1, int(60 - (now - calls[0])) + 1)
        calls.append(now)
        return True, 0


class Busy(Exception):
    pass


class Scorer:
    """Runs `Guard.check` in a thread, at most `limit` at once, each within `timeout`.

    A slot is freed by the worker thread when the scoring really ends, not when the request
    gives up waiting. Otherwise timed-out work would keep running beyond the cap.
    """

    def __init__(self, guard: Guard, limit: int, timeout: float):
        self.guard = guard
        self.timeout = timeout
        self.slots = threading.BoundedSemaphore(limit)

    async def check(self, text: str) -> Decision:
        if not self.slots.acquire(blocking=False):
            raise Busy  # refuse straight away; a queue would turn a burst into timeouts

        def work() -> Decision:
            try:
                return self.guard.check(text)
            finally:
                self.slots.release()

        # shield: a timeout must not cancel the job before it starts, or its slot would leak.
        job = asyncio.ensure_future(asyncio.to_thread(work))
        return await asyncio.wait_for(asyncio.shield(job), self.timeout)


# --- the app


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=MAX_CHARS)


def _error(status: int, code: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status, headers=headers)


class ApiError(Exception):
    def __init__(self, status: int, code: str, headers: dict | None = None):
        self.status, self.code, self.headers = status, code, headers


def create_app(settings: Settings, guard: Guard | None = None) -> FastAPI:
    guard = guard or load_guard(settings)
    app = FastAPI(title="prompt-injection-guard", docs_url=None, redoc_url=None, openapi_url=None)
    limiter = RateLimiter(settings.rate_per_minute)
    scorer = Scorer(guard, settings.max_concurrent, settings.timeout_seconds)

    @app.exception_handler(ApiError)
    async def api_error(_request, e: ApiError):
        return _error(e.status, e.code, e.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid(_request, _e):
        # FastAPI's default 422 body repeats the input. Here the input is the prompt.
        return _error(422, "invalid_request")

    @app.middleware("http")
    async def body_size(request: Request, call_next):
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is None:
                return _error(411, "length_required")
            if not length.isdigit() or int(length) > settings.max_body_bytes:
                return _error(413, "too_large")
        return await call_next(request)

    def caller(request: Request) -> str:
        key = request.headers.get("x-api-key", "")
        digest = hashlib.sha256(key.encode()).hexdigest()
        # Compare against every configured hash, so timing doesn't say how many there are.
        matched = [hmac.compare_digest(digest, h) for h in settings.api_key_sha256]
        if not key or not any(matched):
            raise ApiError(401, "unauthorized")
        ok, retry = limiter.allow(digest)
        if not ok:
            raise ApiError(429, "rate_limited", {"Retry-After": str(retry)})
        return digest[:8]

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/check")
    async def check(body: CheckRequest, key_id: str = Depends(caller)):
        request_id, started = uuid.uuid4().hex, time.perf_counter()
        outcome = "error"
        try:
            result = await scorer.check(body.text)
            outcome = result.decision
            return {
                "decision": result.decision,
                "reasons": list(result.reasons),
                "request_id": request_id,
                "model": guard.version,
            }
        except Busy:
            outcome = "busy"
            raise ApiError(503, "busy", {"Retry-After": "1"}) from None
        except TimeoutError:
            outcome = "timeout"
            raise ApiError(503, "timeout") from None
        finally:
            # Never the text: only its length and what happened (STRIDE, information disclosure).
            log.info(json.dumps({
                "request_id": request_id, "key": key_id, "chars": len(body.text),
                "outcome": outcome, "ms": round((time.perf_counter() - started) * 1000, 1),
            }))  # fmt: skip

    return app


def configure_logging(stream=None) -> None:
    """One JSON line per request to stderr. Uvicorn only sets up its own loggers, so without
    this the audit lines (STRIDE: repudiation) would be dropped."""
    if not log.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def app_from_env() -> FastAPI:
    """For `uvicorn --factory guard.api:app_from_env`."""
    configure_logging()
    return create_app(Settings.from_env())
