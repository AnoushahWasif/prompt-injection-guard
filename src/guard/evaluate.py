"""Does the guard catch what it hasn't seen? Held-out sources and attack styles.

A random split puts near-copies of every attack template in both train and test, so it mostly
measures memory (decision 5). Here each experiment removes one source, or every attack of one
style, from training and validation, then scores exactly what was removed. The threshold is
still set on the remaining validation data, as it would be in practice: the held-out data
never gets a say in where the line is.

SPML is 81% of the rows. It is capped for these experiments, for both models, so it doesn't
decide every result and so the transformer trains in minutes instead of over an hour. The
random split is rerun with the same cap, so held-out numbers are compared like for like.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from guard.baseline import (
    SEED,
    TARGET_FPR,
    evaluate_scores,
    fit,
    prepare,
    scores,
    split,
    tfidf_model,
    threshold_at_fpr,
)
from guard.data import attack_style

SPML_CAP = 3_000
STYLES = ("override", "roleplay", "extraction")
BOOTSTRAP_ROUNDS = 1_000
OUT = Path("data/processed/heldout")  # holds prompt text, so it stays out of reports/

Scorer = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class Experiment:
    name: str
    pool: pd.DataFrame  # split into train / validation / test as usual
    unseen: pd.DataFrame | None  # never trained or thresholded on


def cap_source(df: pd.DataFrame, source: str, n: int, seed: int = SEED) -> pd.DataFrame:
    """Keep at most n rows of one source, with its attack share unchanged."""
    if n < 1:
        raise ValueError("cap must be at least 1")
    part = df[df["source"] == source]
    if len(part) <= n:
        return df
    kept = [
        g.sample(round(n * len(g) / len(part)), random_state=seed).index
        for _, g in part.groupby("label")
    ]
    return df.drop(part.index.difference(np.concatenate(kept)))


def with_styles(df: pd.DataFrame) -> pd.DataFrame:
    """Add each attack's style; benign prompts get none."""
    style = df["clean"].map(attack_style).where(df["label"] == 1, "")
    return df.assign(style=style)


def experiments(df: pd.DataFrame) -> list[Experiment]:
    out = [Experiment("random", df, None)]
    for source in sorted(df["source"].unique()):
        held = df["source"] == source
        out.append(Experiment(f"source:{source}", df[~held], df[held]))
    for style in STYLES:
        held = df["style"] == style
        out.append(Experiment(f"style:{style}", df[~held], df[held]))
    return out


def rate_with_interval(hits: np.ndarray, seed: int = SEED) -> dict | None:
    """Share of True, with a 95% bootstrap interval. None if there is nothing to count."""
    hits = np.asarray(hits, dtype=bool)
    if hits.size == 0:
        return None
    rng = np.random.default_rng(seed)
    resampled = rng.choice(hits, size=(BOOTSTRAP_ROUNDS, hits.size)).mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {
        "rate": round(float(hits.mean()), 3),
        "low": round(float(low), 3),
        "high": round(float(high), 3),
        "n": int(hits.size),
    }


def summarise(s: np.ndarray, threshold: float, df: pd.DataFrame) -> dict:
    """Recall and false-alarm rate with intervals, plus recall per attack style."""
    y = df["label"].to_numpy()
    flagged = np.asarray(s) >= threshold
    attacks = y == 1
    return {
        "recall": rate_with_interval(flagged[attacks]),
        "false_positive_rate": rate_with_interval(flagged[~attacks]),
        "per_style": {
            style: rate_with_interval(flagged[attacks & (df["style"] == style).to_numpy()])
            for style in sorted(df.loc[attacks, "style"].unique())
        },
    }


def run_experiment(exp: Experiment, fit_scorer: Callable[[pd.DataFrame], Scorer]) -> dict:
    train, val, test = split(exp.pool)
    score = fit_scorer(train)
    threshold = threshold_at_fpr(score(val.loc[val["label"] == 0, "clean"].tolist()))
    short_chars = int(train.loc[train["label"] == 0, "clean"].str.len().median())
    test_scores = score(test["clean"].tolist())
    result = {
        "experiment": exp.name,
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "threshold": round(threshold, 4),
        "seen": {
            **summarise(test_scores, threshold, test),
            "short_attacks": evaluate_scores(test_scores, threshold, test, short_chars)[
                "short_attacks"
            ],
        },
        "unseen": None,
    }
    scored = [test.assign(score=test_scores, part="seen test")]
    if exp.unseen is not None:
        unseen_scores = score(exp.unseen["clean"].tolist())
        result["unseen"] = summarise(unseen_scores, threshold, exp.unseen)
        scored.append(exp.unseen.assign(score=unseen_scores, part="unseen"))
    return result, pd.concat(scored).assign(flagged=lambda d: d["score"] >= threshold)


# --- the two models


def fit_tfidf(train: pd.DataFrame) -> Scorer:
    model = fit(tfidf_model(), train)
    return lambda texts: scores(model, pd.Series(texts, dtype=str))


def fit_transformer(train: pd.DataFrame) -> Scorer:
    from guard import finetune as ft

    model, tokenizer = ft.load_base(ft.fetch_base())
    ft.train(model, tokenizer, train["clean"].tolist(), train["label"].to_numpy(), ft.TrainConfig())
    return lambda texts: ft.predict(model, tokenizer, texts)


MODELS = {"tfidf": fit_tfidf, "transformer": fit_transformer}


def run(df: pd.DataFrame, model: str, out: Path = OUT, only: list[str] | None = None) -> list:
    """Every experiment for one model. Finished ones are skipped, so a long run can resume."""
    df, _ = prepare(df)
    df = with_styles(cap_source(df, "spml", SPML_CAP))
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for exp in experiments(df):
        if only and exp.name not in only:
            continue
        stem = f"{model}_{exp.name.replace(':', '_')}"
        done = out / f"{stem}.json"
        if done.is_file():
            results.append(json.loads(done.read_text()))
            continue
        print(f"{model}: {exp.name}", flush=True)
        result, scored = run_experiment(exp, MODELS[model])
        scored[["text", "source", "label", "style", "part", "score", "flagged"]].to_parquet(
            out / f"{stem}.parquet"
        )
        done.write_text(json.dumps(result, indent=2))
        results.append(result)
    return results


def report(out: Path = OUT, reports: Path = Path("reports")) -> dict:
    """Collect the finished experiments into reports/heldout.json (no prompt text in it)."""
    summary = {"spml_cap": SPML_CAP, "target_fpr": TARGET_FPR, "models": {}}
    for path in sorted(out.glob("*.json")):
        model, _, name = path.stem.partition("_")
        summary["models"].setdefault(model, {})[name.replace("_", ":", 1)] = json.loads(
            path.read_text()
        )
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "heldout.json").write_text(json.dumps(summary, indent=2))
    if summary["models"]:
        from guard.plots import plot_heldout

        plot_heldout(summary, reports / "figures" / "heldout.png")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODELS, "both"], default="tfidf")
    parser.add_argument("--only", nargs="*", help="experiment names, e.g. source:deepset")
    args = parser.parse_args()
    path = Path("data/processed/prompts.parquet")
    if not path.is_file():
        raise SystemExit(f"{path} not found; run `python -m guard.data` first")
    data = pd.read_parquet(path)
    for name in MODELS if args.model == "both" else [args.model]:
        run(data, name, only=args.only)
    report()
