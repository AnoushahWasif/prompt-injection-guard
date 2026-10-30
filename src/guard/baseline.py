"""First model: TF-IDF + logistic regression, next to a length-only model on the same split.

Length alone separates attacks from benign prompts with AUC 0.87 on this data (Day 2). The
length model is here so that number is on the table: if TF-IDF can't beat it, especially on
short attacks, it has learned length, not wording.

The threshold is set on a validation split so that about 1% of benign prompts get flagged.
A guard that blocks normal users gets switched off, so false alarms are the budget and the
number that matters is how many attacks are caught inside it.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from guard.model_store import save_model, sha256
from guard.normalize import normalize

SEED = 42
TARGET_FPR = 0.01
# Rows from a source with fewer prompts than this (per label) are stratified with the biggest
# source of the same label, so a tiny new source can't break the split.
MIN_STRATUM = 10


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add the normalised text the models see. Rows that normalise to nothing are dropped."""
    missing = {"text", "label", "source"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("labels must be 0 or 1")
    clean = df["text"].map(lambda t: normalize(t).text)
    keep = clean != ""
    out = df.assign(clean=clean)[keep].reset_index(drop=True)
    return out, int((~keep).sum())


def _strata(df: pd.DataFrame) -> pd.Series:
    strata = df["source"] + "/" + df["label"].astype(str)
    big = strata.map(strata.value_counts()) >= MIN_STRATUM
    for label in (0, 1):
        small = (df["label"] == label) & ~big
        if small.any():
            pool = strata[(df["label"] == label) & big]
            strata = strata.mask(small, pool.mode()[0] if len(pool) else f"all/{label}")
    return strata


def split(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70 / 15 / 15, stratified by source and label so every source is in every part."""
    per_label = df["label"].value_counts()
    if len(per_label) < 2 or per_label.min() < MIN_STRATUM:
        raise ValueError(f"need at least {MIN_STRATUM} attacks and {MIN_STRATUM} benign prompts")
    strata = _strata(df)
    train, rest = train_test_split(df, test_size=0.3, stratify=strata, random_state=seed)
    val, test = train_test_split(
        rest, test_size=0.5, stratify=strata.loc[rest.index], random_state=seed
    )
    return train, val, test


def _log_length(texts) -> np.ndarray:
    return np.log1p(np.array([len(t) for t in texts], dtype=float)).reshape(-1, 1)


def length_model() -> Pipeline:
    return make_pipeline(
        FunctionTransformer(_log_length), StandardScaler(), LogisticRegression(random_state=SEED)
    )


def tfidf_model() -> Pipeline:
    # Character n-grams within words catch spelling tricks the normaliser doesn't
    # ("ign0re", "i-g-n-o-r-e" partly) and work for German without a tokenizer.
    features = FeatureUnion(
        [
            (
                "words",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000
                ),
            ),
            (
                "chars",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                    max_features=200_000,
                ),
            ),
        ]
    )
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    return Pipeline([("features", features), ("clf", clf)])


def fit(model: Pipeline, train: pd.DataFrame) -> Pipeline:
    if train["label"].nunique() < 2:
        raise ValueError("need both attacks and benign prompts to train")
    return model.fit(train["clean"], train["label"])


def scores(model: Pipeline, texts: pd.Series) -> np.ndarray:
    return model.predict_proba(texts)[:, 1]


def threshold_at_fpr(benign_scores: np.ndarray, target_fpr: float = TARGET_FPR) -> float:
    """Lowest threshold that flags at most `target_fpr` of these benign prompts."""
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    s = np.sort(np.asarray(benign_scores, dtype=float))[::-1]
    if s.size == 0:
        raise ValueError("no benign scores to set a threshold on")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    allowed = math.floor(target_fpr * s.size)
    # Just above the first benign score that must not be flagged (score >= threshold flags).
    if s[allowed] >= 1.0:
        raise ValueError("benign prompts score 1.0; the target false-positive rate is unreachable")
    return float(np.nextafter(s[allowed], np.inf))


def _rate(hits: np.ndarray) -> float | None:
    return round(float(hits.mean()), 3) if hits.size else None


def _metrics(y: np.ndarray, s: np.ndarray, flagged: np.ndarray) -> dict:
    both = len(np.unique(y)) == 2
    return {
        "rows": int(y.size),
        "attacks": int(y.sum()),
        "benign": int((y == 0).sum()),
        "recall": _rate(flagged[y == 1]),
        "false_positive_rate": _rate(flagged[y == 0]),
        "roc_auc": round(float(roc_auc_score(y, s)), 3) if both else None,
        "pr_auc": round(float(average_precision_score(y, s)), 3) if both else None,
    }


def evaluate(model: Pipeline, threshold: float, df: pd.DataFrame, short_chars: int) -> dict:
    """Overall, per source, and on short attacks (no longer than a typical benign prompt)."""
    y = df["label"].to_numpy()
    s = scores(model, df["clean"])
    flagged = s >= threshold
    out = {"all": _metrics(y, s, flagged), "per_source": {}}
    for source in sorted(df["source"].unique()):
        mask = (df["source"] == source).to_numpy()
        out["per_source"][source] = _metrics(y[mask], s[mask], flagged[mask])
    short = (y == 1) & (df["clean"].str.len() <= short_chars).to_numpy()
    out["short_attacks"] = {"rows": int(short.sum()), "recall": _rate(flagged[short])}
    return out


def top_features(model: Pipeline, n: int = 15) -> dict:
    names = model.named_steps["features"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)

    def pick(idx):
        return [(str(names[i]), round(float(coef[i]), 3)) for i in idx]

    return {"attack": pick(order[::-1][:n]), "benign": pick(order[:n])}


def explain(model: Pipeline, text: str, n: int = 5) -> list[tuple[str, float]]:
    """The n-grams in this text that pushed its score up the most."""
    row = model.named_steps["features"].transform([normalize(text).text]).tocsr()
    coef = model.named_steps["clf"].coef_[0]
    names = model.named_steps["features"].get_feature_names_out()
    contrib = row.data * coef[row.indices]
    order = np.argsort(contrib)[::-1]
    return [
        (str(names[row.indices[i]]), round(float(contrib[i]), 3))
        for i in order[:n]
        if contrib[i] > 0
    ]


def run(
    df: pd.DataFrame,
    model_dir: Path = Path("models/baseline"),
    reports: Path = Path("reports"),
    data_sha256: str | None = None,
) -> dict:
    df, dropped = prepare(df)
    train, val, test = split(df)
    short_chars = int(train.loc[train["label"] == 0, "clean"].str.len().median())

    report: dict = {
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "dropped_empty_after_normalising": dropped,
        "target_fpr": TARGET_FPR,
        "short_attack_max_chars": short_chars,
        "models": {},
    }
    fitted = {}
    for name, build in (("length", length_model), ("tfidf", tfidf_model)):
        model = fit(build(), train)
        val_benign = scores(model, val.loc[val["label"] == 0, "clean"])
        threshold = threshold_at_fpr(val_benign)
        fitted[name] = (model, threshold)
        report["models"][name] = {
            "threshold": round(threshold, 4),
            "test": evaluate(model, threshold, test, short_chars),
        }

    model, threshold = fitted["tfidf"]
    report["top_features"] = top_features(model)
    save_model(
        model,
        threshold,
        model_dir,
        {"model": "tfidf_logreg", "target_fpr": TARGET_FPR, "data_sha256": data_sha256},
    )

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "baseline.json").write_text(json.dumps(report, indent=2))
    from guard.plots import plot_baseline

    plot_baseline(report, reports / "figures" / "baseline_recall.png")
    return report


if __name__ == "__main__":
    path = Path("data/processed/prompts.parquet")
    if not path.is_file():
        raise SystemExit(f"{path} not found; run `python -m guard.data` first")
    result = run(pd.read_parquet(path), data_sha256=sha256(path))
    print(json.dumps({k: v for k, v in result.items() if k != "top_features"}, indent=2))
