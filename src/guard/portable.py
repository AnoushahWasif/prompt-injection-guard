"""The TF-IDF model without a pickle: vocabularies as JSON, numbers as safetensors.

Unpickling runs code, so a pickle can't safely be shared or downloaded (OWASP LLM03). This
format is data only. Loading rebuilds the same scikit-learn objects from it and gives the
same scores as the pickled model, which `export` checks before writing anything.

Files in the folder, each listed with its SHA-256 in manifest.json:
    config.json          vectoriser settings, thresholds, versions
    vocab_words.json     term -> column for the word n-grams
    vocab_chars.json     term -> column for the character n-grams
    weights.safetensors  idf_words, idf_chars, coef, intercept
"""

import hashlib
import hmac
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from guard.model_store import ModelIntegrityError

FILES = ("config.json", "vocab_words.json", "vocab_chars.json", "weights.safetensors")
MANIFEST = "manifest.json"
FORMAT = "guard-tfidf-v1"
# The only vectoriser settings the export supports. Anything else means the training code
# changed and this format has to change with it.
EXPECTED = {
    "words": {"analyzer": "word", "ngram_range": [1, 2]},
    "chars": {"analyzer": "char_wb", "ngram_range": [3, 5]},
}
COMMON = {"sublinear_tf": True, "lowercase": True, "norm": "l2", "smooth_idf": True}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _settings(vec: TfidfVectorizer) -> dict:
    return {
        "analyzer": vec.analyzer,
        "ngram_range": list(vec.ngram_range),
        **{k: getattr(vec, k) for k in COMMON},
    }


def export(model: Pipeline, threshold: float, review: float, out_dir: Path, check: list[str]):
    """Write the portable files for a fitted pipeline. Refuses unless the rebuilt model gives
    the same scores on `check` as the original."""
    from safetensors.numpy import save_file

    features, clf = model.named_steps["features"], model.named_steps["clf"]
    parts = dict(features.transformer_list)
    if set(parts) != set(EXPECTED):
        raise ValueError(f"unexpected feature parts {sorted(parts)}")
    for name, vec in parts.items():
        if _settings(vec) != {**EXPECTED[name], **COMMON}:
            raise ValueError(f"{name}: vectoriser settings changed; update the portable format")
    if list(clf.classes_) != [0, 1] or not 0 < review <= threshold <= 1:
        raise ValueError("expected classes [0, 1] and 0 < review <= threshold <= 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    blobs = {
        "config.json": json.dumps(
            {
                "format": FORMAT,
                "threshold": threshold,
                "review_threshold": review,
                "features": {n: _settings(v) for n, v in parts.items()},
            },
            indent=2,
        ).encode(),
        "vocab_words.json": json.dumps(
            {k: int(v) for k, v in parts["words"].vocabulary_.items()}
        ).encode(),
        "vocab_chars.json": json.dumps(
            {k: int(v) for k, v in parts["chars"].vocabulary_.items()}
        ).encode(),
    }
    weights = out_dir / "weights.safetensors"
    save_file(
        {
            "idf_words": parts["words"].idf_.astype(np.float64),
            "idf_chars": parts["chars"].idf_.astype(np.float64),
            "coef": clf.coef_.astype(np.float64),
            "intercept": clf.intercept_.astype(np.float64),
        },
        str(weights),
    )
    for name, data in blobs.items():
        (out_dir / name).write_bytes(data)
    manifest = {name: _sha((out_dir / name).read_bytes()) for name in FILES}
    (out_dir / MANIFEST).write_text(json.dumps({"format": FORMAT, "files": manifest}, indent=2))

    rebuilt = load(out_dir, _sha((out_dir / MANIFEST).read_bytes()))
    if not np.allclose(rebuilt.model.predict_proba(check), model.predict_proba(check), atol=1e-12):
        raise RuntimeError("the rebuilt model scores differently; not exporting")
    return _sha((out_dir / MANIFEST).read_bytes())


class Loaded:
    def __init__(self, model: Pipeline, threshold: float, review: float, manifest: dict):
        self.model, self.threshold, self.review, self.manifest = model, threshold, review, manifest


def _vectoriser(settings: dict, vocab: dict, idf: np.ndarray) -> TfidfVectorizer:
    if len(vocab) != idf.shape[0] or sorted(vocab.values()) != list(range(len(vocab))):
        raise ModelIntegrityError("vocabulary and idf don't match")
    vec = TfidfVectorizer(
        analyzer=settings["analyzer"],
        ngram_range=tuple(settings["ngram_range"]),
        vocabulary=vocab,
        **{k: settings[k] for k in COMMON},
    )
    vec._validate_vocabulary()
    vec.idf_ = idf
    return vec


def load(model_dir: Path, expected_manifest_sha256: str) -> Loaded:
    """Rebuild the model, only if every file matches the manifest and the manifest matches the
    hash pinned in config. Every file is read once and checked as the bytes that get used."""
    from safetensors.numpy import load as load_tensors

    if not expected_manifest_sha256:
        raise ModelIntegrityError("a pinned manifest hash is required")
    try:
        raw_manifest = (model_dir / MANIFEST).read_bytes()
    except OSError as e:
        raise ModelIntegrityError("manifest missing") from e
    if not hmac.compare_digest(_sha(raw_manifest), expected_manifest_sha256.lower()):
        raise ModelIntegrityError("manifest hash does not match; refusing to load")
    manifest = json.loads(raw_manifest)
    if manifest.get("format") != FORMAT or set(manifest.get("files", {})) != set(FILES):
        raise ModelIntegrityError("unknown format or file list")
    data = {}
    for name in FILES:
        try:
            raw = (model_dir / name).read_bytes()
        except OSError as e:
            raise ModelIntegrityError(f"{name} missing") from e
        if not hmac.compare_digest(_sha(raw), manifest["files"][name]):
            raise ModelIntegrityError(f"{name} changed; refusing to load")
        data[name] = raw

    config = json.loads(data["config.json"])
    tensors = load_tensors(data["weights.safetensors"])
    threshold, review = float(config["threshold"]), float(config["review_threshold"])
    if not 0 < review <= threshold <= 1:
        raise ModelIntegrityError("thresholds out of range")
    features = config["features"]
    words = _vectoriser(
        features["words"], json.loads(data["vocab_words.json"]), tensors["idf_words"]
    )
    chars = _vectoriser(
        features["chars"], json.loads(data["vocab_chars.json"]), tensors["idf_chars"]
    )
    coef = tensors["coef"]
    if coef.shape != (1, len(words.vocabulary_) + len(chars.vocabulary_)):
        raise ModelIntegrityError("coefficients don't match the vocabularies")
    clf = LogisticRegression()
    clf.coef_, clf.intercept_, clf.classes_ = coef, tensors["intercept"], np.array([0, 1])
    clf.n_features_in_ = coef.shape[1]
    model = Pipeline(
        [("features", FeatureUnion([("words", words), ("chars", chars)])), ("clf", clf)]
    )
    return Loaded(model, threshold, review, manifest)


if __name__ == "__main__":
    import pandas as pd

    from guard.model_store import load_model
    from guard.normalize import normalize

    source = Path("models/baseline")
    pickled = load_model(source, json.loads((source / "model.json").read_text())["sha256"])
    sample = pd.read_parquet("data/processed/prompts.parquet")["text"].sample(500, random_state=0)
    check = [normalize(t).text for t in sample]
    review = float(pickled.manifest["review_threshold"])
    digest = export(pickled.model, pickled.threshold, review, Path("models/portable"), check)
    print(f"wrote models/portable; pin GUARD_MODEL_SHA256={digest}")
    print({p.name: p.stat().st_size for p in Path("models/portable").iterdir()})
