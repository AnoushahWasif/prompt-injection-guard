"""Save a trained guard with a manifest, and only load it if its hash still matches.

A .joblib file is a pickle, and unpickling runs code, so the hash is checked before loading
(OWASP LLM03). A hash passed in from deploy config beats the manifest's own copy, because
someone who can swap the model file can usually edit the manifest next to it too.
"""

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn

MODEL_FILE, MANIFEST_FILE = "model.joblib", "model.json"


class ModelIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedModel:
    model: object
    threshold: float
    manifest: dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(model, threshold: float, out_dir: Path, info: dict | None = None) -> Path:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MODEL_FILE
    joblib.dump(model, path)
    manifest = {
        "sha256": sha256(path),
        "threshold": threshold,
        "sklearn_version": sklearn.__version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **(info or {}),
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return path


def load_model(model_dir: Path, expected_sha256: str | None = None) -> LoadedModel:
    try:
        manifest = json.loads((model_dir / MANIFEST_FILE).read_text())
        expected = (expected_sha256 or manifest["sha256"]).lower()
        threshold = float(manifest["threshold"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ModelIntegrityError("model manifest missing or unreadable") from exc
    if not 0.0 < threshold <= 1.0:
        raise ModelIntegrityError("threshold in manifest is out of range")
    path = model_dir / MODEL_FILE
    if not path.is_file():
        raise ModelIntegrityError("model file missing")
    if not hmac.compare_digest(sha256(path), expected):
        raise ModelIntegrityError("model file hash does not match; refusing to load it")
    return LoadedModel(joblib.load(path), threshold, manifest)
