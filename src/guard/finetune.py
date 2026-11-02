"""Fine-tune a small multilingual transformer on the same split as the baseline, on CPU.

The base model is pinned to a commit and only its config, tokenizer and safetensors weights are
downloaded: a pickled .bin can run code when loaded, and so can a repo's custom Python
(OWASP LLM03). The fine-tuned model is saved the same way, with every file in a hash manifest.

Long prompts: training sees the first and last tokens of each prompt. Scoring slides a window
over the whole prompt and keeps the highest score, so harmless padding in front of an
injection can't push it out of view.
"""

import hmac
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from guard.baseline import SEED, TARGET_FPR, evaluate_scores, prepare, split, threshold_at_fpr
from guard.model_store import ModelIntegrityError, sha256
from guard.normalize import normalize

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BASE_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
# Everything a fine-tuned model needs; nothing that can execute.
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
CODE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".py", ".h5", ".msgpack")
MANIFEST = "manifest.json"

WINDOW = 128  # tokens per window, special tokens included
STRIDE = 96  # windows overlap by 32 tokens so a phrase split at a boundary is seen whole
HEAD = 64  # training keeps the first HEAD and last WINDOW - 2 - HEAD tokens of a long prompt


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    batch_size: int = 16
    lr: float = 4e-5
    weight_decay: float = 0.01
    warmup: float = 0.06
    seed: int = SEED


def head_tail(ids: list[int], keep: int = WINDOW - 2, head: int = HEAD) -> list[int]:
    if len(ids) <= keep:
        return list(ids)
    return list(ids[:head]) + list(ids[len(ids) - (keep - head) :])


def windows(ids: list[int], size: int = WINDOW - 2, stride: int = STRIDE) -> list[list[int]]:
    """Overlapping slices that cover every token. An empty prompt still gets one window."""
    if stride <= 0 or stride > size:
        raise ValueError("stride must be in (0, size]")
    if len(ids) <= size:
        return [list(ids)]
    starts = list(range(0, len(ids) - size, stride)) + [len(ids) - size]
    return [list(ids[s : s + size]) for s in starts]


def token_ids(tokenizer, texts: list[str]) -> list[list[int]]:
    clean = [normalize(t).text for t in texts]
    if not clean:  # the fast tokenizer raises IndexError on an empty batch
        return []
    return tokenizer(clean, add_special_tokens=False)["input_ids"]


def batches(seqs: list[list[int]], batch_size: int, order: list[int] | None = None):
    """Index batches of similar length, so little time goes on padding."""
    idx = order if order is not None else sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    return [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]


def collate(seqs: list[list[int]], tokenizer) -> dict[str, torch.Tensor]:
    wrapped = [[tokenizer.cls_token_id, *s, tokenizer.sep_token_id] for s in seqs]
    width = max(len(s) for s in wrapped)
    ids = torch.full((len(wrapped), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(wrapped), width), dtype=torch.long)
    for row, s in enumerate(wrapped):
        ids[row, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[row, : len(s)] = 1
    return {"input_ids": ids, "attention_mask": mask}


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Same idea as class_weight="balanced" in the baseline: each class counts equally."""
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("need both attacks and benign prompts to train")
    return torch.tensor(len(labels) / (2 * counts), dtype=torch.float)


def train(model, tokenizer, texts: list[str], labels: np.ndarray, config: TrainConfig, log=print):
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    seqs = [head_tail(ids) for ids in token_ids(tokenizer, texts)]
    y = torch.tensor(labels, dtype=torch.long)
    weights = class_weights(np.asarray(labels))
    groups = batches(seqs, config.batch_size)
    total = len(groups) * config.epochs
    warmup = max(1, int(total * config.warmup))
    # The 250k-word vocabulary table is 96M of the model's 118M weights. Freezing it makes each
    # step about three times faster on CPU and keeps the vocabulary of the 50+ languages intact.
    model.get_input_embeddings().weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup, max(0.0, (total - step) / max(1, total - warmup))),
    )
    model.train()
    step = 0
    for epoch in range(config.epochs):
        rng.shuffle(groups)  # batches keep similar lengths; their order is random
        started, running = time.perf_counter(), 0.0
        for group in groups:
            batch = collate([seqs[i] for i in group], tokenizer)
            logits = model(**batch).logits
            loss = F.cross_entropy(logits, y[group], weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            running += loss.item()
            step += 1
            if step % 100 == 0:
                log(f"epoch {epoch + 1} step {step}/{total} loss {running / 100:.4f} "
                    f"{time.perf_counter() - started:.0f}s")  # fmt: skip
                running = 0.0
    model.eval()
    return model


@torch.inference_mode()
def predict(model, tokenizer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """P(attack) per prompt: the highest score over its windows."""
    model.eval()
    owner, pieces = [], []
    for i, ids in enumerate(token_ids(tokenizer, texts)):
        for w in windows(ids):
            owner.append(i)
            pieces.append(w)
    window_scores = np.empty(len(pieces))
    for group in batches(pieces, batch_size):
        logits = model(**collate([pieces[i] for i in group], tokenizer)).logits
        window_scores[group] = torch.softmax(logits, dim=-1)[:, 1].numpy()
    out = np.zeros(len(texts))
    np.maximum.at(out, np.asarray(owner, dtype=int), window_scores)
    return out


def latency(model, tokenizer, texts: list[str]) -> dict:
    """One prompt at a time, the way the guard API will see them."""
    predict(model, tokenizer, texts[:3])  # warm-up
    times = []
    for t in texts:
        start = time.perf_counter()
        predict(model, tokenizer, [t])
        times.append((time.perf_counter() - start) * 1000)
    return {
        "prompts": len(times),
        "threads": torch.get_num_threads(),
        "p50_ms": round(float(np.percentile(times, 50)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
        "max_ms": round(float(np.max(times)), 1),
    }


# --- base model download and the saved model's manifest


def fetch_base(cache_dir: Path = Path(".hf_cache")) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            BASE_MODEL,
            revision=BASE_REVISION,
            cache_dir=cache_dir,
            allow_patterns=list(MODEL_FILES),
        )
    )
    check_no_code(path)
    return path


def check_no_code(path: Path) -> None:
    risky = sorted(p.name for p in path.iterdir() if p.suffix.lower() in CODE_SUFFIXES)
    if risky:
        raise ModelIntegrityError(f"refusing a model folder with loadable code: {risky}")


def load_base(path: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        num_labels=2,
        use_safetensors=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    return model, tokenizer


def save(model, tokenizer, out_dir: Path, threshold: float, info: dict) -> str:
    """Write the model and a manifest of file hashes. Returns the manifest's own SHA-256."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    # Extra files get deleted below, so only write into an empty folder or an earlier model.
    if out_dir.is_dir() and any(out_dir.iterdir()) and not (out_dir / MANIFEST).is_file():
        raise ValueError(f"{out_dir} has other files in it; refusing to save a model there")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    # save_pretrained can add helper files; keep only what loading needs.
    for p in out_dir.iterdir():
        if p.name not in (*MODEL_FILES, MANIFEST):
            p.unlink()
    manifest = {
        "files": {name: sha256(out_dir / name) for name in MODEL_FILES},
        "threshold": threshold,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        **info,
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return sha256(out_dir / MANIFEST)


def load(model_dir: Path, expected_manifest_sha256: str | None = None):
    """Load a saved guard model only if every file matches its manifest."""
    path = model_dir / MANIFEST
    try:
        if expected_manifest_sha256 is not None and not hmac.compare_digest(
            sha256(path), expected_manifest_sha256.lower()
        ):
            raise ModelIntegrityError("manifest hash does not match; refusing to load")
        manifest = json.loads(path.read_text())
        files = manifest["files"]
        threshold = float(manifest["threshold"])
    except ModelIntegrityError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ModelIntegrityError("model manifest missing or unreadable") from exc
    if not isinstance(files, dict) or set(files) != set(MODEL_FILES):
        raise ModelIntegrityError("manifest must list exactly the expected model files")
    if not 0.0 < threshold <= 1.0 or math.isnan(threshold):
        raise ModelIntegrityError("threshold in manifest is out of range")
    check_no_code(model_dir)
    for name, expected in files.items():
        file = model_dir / name
        if not file.is_file() or not hmac.compare_digest(sha256(file), str(expected).lower()):
            raise ModelIntegrityError(f"{name} missing or changed; refusing to load")
    model, tokenizer = load_base(model_dir)
    return model, tokenizer, threshold, manifest


def run(
    df: pd.DataFrame,
    model_dir: Path = Path("models/transformer"),
    reports: Path = Path("reports"),
    config: TrainConfig | None = None,
    data_sha256: str | None = None,
) -> dict:
    config = config or TrainConfig()
    df, _ = prepare(df)
    train_df, val, test = split(df)
    short_chars = int(train_df.loc[train_df["label"] == 0, "clean"].str.len().median())

    model, tokenizer = load_base(fetch_base())
    started = time.perf_counter()
    train(model, tokenizer, train_df["clean"].tolist(), train_df["label"].to_numpy(), config)
    train_minutes = (time.perf_counter() - started) / 60

    val_benign = predict(model, tokenizer, val.loc[val["label"] == 0, "clean"].tolist())
    threshold = threshold_at_fpr(val_benign, TARGET_FPR)
    test_scores = predict(model, tokenizer, test["clean"].tolist())
    report = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "config": config.__dict__,
        "window_tokens": WINDOW,
        "train_minutes": round(train_minutes, 1),
        "target_fpr": TARGET_FPR,
        "threshold": round(threshold, 4),
        "short_attack_max_chars": short_chars,
        "test": evaluate_scores(test_scores, threshold, test, short_chars),
        "latency": latency(model, tokenizer, test["clean"].sample(200, random_state=SEED).tolist()),
    }
    report["manifest_sha256"] = save(
        model,
        tokenizer,
        model_dir,
        threshold,
        {"data_sha256": data_sha256, "target_fpr": TARGET_FPR},
    )
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "transformer.json").write_text(json.dumps(report, indent=2))
    if (reports / "baseline.json").is_file():
        plot_comparison(report, json.loads((reports / "baseline.json").read_text()), reports)
    # Kept for Day 5's error analysis. It holds prompt text, so it lives under data/, not reports/.
    scores_path = Path("data/processed/transformer_test_scores.parquet")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    test.assign(score=test_scores)[["text", "source", "label", "score"]].to_parquet(scores_path)
    return report


def plot_comparison(transformer: dict, baseline: dict, reports: Path) -> None:
    from guard.plots import BASELINE, BENIGN, TRANSFORMER, plot_recall

    models = baseline["models"]
    caught_t = transformer["test"]["all"]["recall"]
    caught_b = models["tfidf"]["test"]["all"]["recall"]
    title = f"Fine-tuned MiniLM catches {caught_t:.0%} of attacks; TF-IDF, {caught_b:.0%}"
    series = [
        ("fine-tuned MiniLM", TRANSFORMER, transformer["test"]),
        ("TF-IDF + logistic regression", BENIGN, models["tfidf"]["test"]),
        ("length only", BASELINE, models["length"]["test"]),
    ]
    path = reports / "figures" / "models_recall.png"
    plot_recall(series, transformer["target_fpr"], title, path)


if __name__ == "__main__":
    path = Path("data/processed/prompts.parquet")
    if not path.is_file():
        raise SystemExit(f"{path} not found; run `python -m guard.data` first")
    result = run(pd.read_parquet(path), data_sha256=sha256(path))
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))
