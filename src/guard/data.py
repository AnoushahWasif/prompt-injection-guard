"""Download the public datasets at pinned commits, map them to one schema, normalise, dedupe.

Every file is pinned to a Hugging Face commit and checked by SHA-256, so a dataset can't
change (or be swapped) under us. Output columns: text, label (1 = attack), source.
"""

import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from guard.normalize import MAX_CHARS, normalize


@dataclass(frozen=True)
class RemoteFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    license: str
    files: tuple[RemoteFile, ...]


SOURCES = {
    "deepset": Source(
        "deepset/prompt-injections",
        "4f61ecb038e9c3fb77e21034b22511b523772cdd",
        "apache-2.0",
        (
            RemoteFile(
                "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
                "2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04",
            ),
            RemoteFile(
                "data/test-00000-of-00001-701d16158af87368.parquet",
                "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e",
            ),
        ),
    ),
    "jackhhao": Source(
        "jackhhao/jailbreak-classification",
        "2f2ceeb39658696fd3f462403562b6eea5306287",
        "apache-2.0",
        (
            RemoteFile(
                "default/jailbreak_dataset_full.csv",
                "79a7b90b0abe00e3586cc5048353c3236543cca228a1ee261fe3b57a7cb7e29f",
            ),
        ),
    ),
    "gandalf": Source(
        "Lakera/gandalf_ignore_instructions",
        "04737b65e90a6794ec227012e4a255a7def6344b",
        "mit",
        (
            RemoteFile(
                "data/train-00000-of-00001-ded53be747ff55cd.parquet",
                "5b6acf3e5a5998d21f8e1222bb45bbdec25a14408747b1cd63bebef4a75fa439",
            ),
            RemoteFile(
                "data/validation-00000-of-00001-94481a2a09ff2fff.parquet",
                "f51ab3e3407a368845b0f57932cc745c09280429416d7507bd178a16326a79f6",
            ),
            RemoteFile(
                "data/test-00000-of-00001-bc92128b9288a6d1.parquet",
                "56b646d133335ebc535266bd55dbe1b5bee7caa4b95bf49d040684b9b5dd9972",
            ),
        ),
    ),
    "spml": Source(
        "reshabhs/SPML_Chatbot_Prompt_Injection",
        "02ce8084e979bc7d4c24ee35d22ecb7f2db96ff5",
        "mit",
        (
            RemoteFile(
                "spml_prompt_injection.csv",
                "4ca6590cd8ec10832fcebee5e243acb32be5c3e79b7a260f8638eed738c784f8",
            ),
        ),
    ),
}


def data_dir() -> Path:
    return Path(os.environ.get("GUARD_DATA_DIR", "data/raw"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, dest: Path | None = None) -> list[Path]:
    source = SOURCES[name]
    folder = (dest or data_dir()) / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for remote in source.files:
        path = folder / Path(remote.path).name
        if not path.exists():
            url = f"https://huggingface.co/datasets/{source.repo}/resolve/{source.revision}/{remote.path}"
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        actual = sha256_of(path)
        if actual != remote.sha256:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {remote.sha256}, got {actual}"
            )
        paths.append(path)
    return paths


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


# Each loader maps one dataset's own columns to (text, label). Label 1 = attack.
def _deepset(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([_read(p) for p in paths], ignore_index=True)
    return pd.DataFrame({"text": df["text"], "label": df["label"]})


def _jackhhao(paths: list[Path]) -> pd.DataFrame:
    df = _read(paths[0])
    labels = df["type"].map({"jailbreak": 1, "benign": 0})
    if labels.isna().any():
        raise ValueError(
            f"unknown jackhhao types: {sorted(df.loc[labels.isna(), 'type'].unique())}"
        )
    return pd.DataFrame({"text": df["prompt"], "label": labels})


def _gandalf(paths: list[Path]) -> pd.DataFrame:
    # Every row is an attempt to get Gandalf to ignore its instructions: attacks only.
    df = pd.concat([_read(p) for p in paths], ignore_index=True)
    return pd.DataFrame({"text": df["text"], "label": 1})


def _spml(paths: list[Path]) -> pd.DataFrame:
    # Only the user's message is kept: the guard sees what users send, not the system prompt.
    df = _read(paths[0])
    return pd.DataFrame({"text": df["User Prompt"], "label": df["Prompt injection"]})


LOADERS = {"deepset": _deepset, "jackhhao": _jackhhao, "gandalf": _gandalf, "spml": _spml}


def load_source(name: str, paths: list[Path]) -> pd.DataFrame:
    df = LOADERS[name](paths)
    if not set(df["label"].dropna().unique()) <= {0, 1}:
        raise ValueError(f"{name}: labels must be 0 or 1")
    return df.assign(source=name)


def dedupe_key(text: str) -> str:
    """Same prompt up to hidden characters, lookalikes, case and spacing."""
    return re.sub(r"\s+", " ", normalize(text).text).casefold()


def merge(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Combine sources, drop unusable rows and duplicates, and count what happened at each step.

    A prompt labelled attack by one source and benign by another is dropped entirely rather
    than guessed: a wrong label teaches the model the wrong thing (OWASP LLM04).
    """
    df = pd.concat(frames, ignore_index=True)
    stats: dict = {"rows_in": len(df)}

    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].map(lambda t: isinstance(t, str))]
    df = df.assign(text=df["text"].str.strip())
    unusable = (df["text"] == "") | (df["text"].str.len() > MAX_CHARS)
    stats["dropped_empty_or_too_long"] = int(unusable.sum()) + stats["rows_in"] - len(df)
    df = df[~unusable].astype({"label": int})

    df = df.assign(key=df["text"].map(dedupe_key))
    labels_per_key = df.groupby("key")["label"].nunique()
    conflicting = set(labels_per_key[labels_per_key > 1].index)
    stats["conflicting_prompts"] = len(conflicting)
    stats["dropped_conflict_rows"] = int(df["key"].isin(conflicting).sum())
    df = df[~df["key"].isin(conflicting)]

    sources_per_key = df.groupby("key")["source"].agg(lambda s: ",".join(sorted(set(s))))
    before = len(df)
    df = df.drop_duplicates("key", keep="first")
    stats["dropped_duplicates"] = before - len(df)
    stats["in_more_than_one_source"] = int(sources_per_key.str.contains(",").sum())
    df = df.assign(sources=df["key"].map(sources_per_key)).drop(columns="key")

    stats["rows_out"] = len(df)
    stats["attacks"] = int(df["label"].sum())
    stats["benign"] = int((df["label"] == 0).sum())
    return df.reset_index(drop=True), stats


# Rough attack styles from wording, for looking at the data and for held-out-style splits.
# Checked in order; a prompt gets the first style that matches.
STYLES = {
    "override": re.compile(
        r"\b(ignore|disregard|forget|overlook|skip)\b.{0,40}\b(instruction|previous|above|prior"
        r"|rule|direction|told)|\b(ignoriere|vergiss)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "roleplay": re.compile(
        r"\b(pretend|act as|you are now|roleplay|role-play|role play|DAN|developer mode"
        r"|jailbreak|character|persona|from now on)\b",
        re.IGNORECASE,
    ),
    "extraction": re.compile(
        r"\b(password|secret|system prompt|your instructions|reveal|confidential|passwort)\b",
        re.IGNORECASE,
    ),
}


def attack_style(text: str) -> str:
    text = normalize(text).text
    for name, pattern in STYLES.items():
        if pattern.search(text):
            return name
    return "other"


def length_auc(df: pd.DataFrame) -> float | None:
    """How well text length alone separates attacks from benign (0.5 = not at all).

    A shortcut check: if length nearly decides the label, a model will learn length.
    """
    if df["label"].nunique() < 2:
        return None
    ranks = df["text"].str.len().rank()
    attacks = df["label"] == 1
    n_pos, n_neg = int(attacks.sum()), int((~attacks).sum())
    # Mann-Whitney U: the chance a random attack is longer than a random benign prompt.
    u = ranks[attacks].sum() - n_pos * (n_pos + 1) / 2
    return round(float(u / (n_pos * n_neg)), 3)


def summarise(df: pd.DataFrame) -> dict:
    attacks = df[df["label"] == 1]
    per_source = {}
    for name, group in df.groupby("source"):
        group_attacks = group[group["label"] == 1]
        per_source[name] = {
            "rows": len(group),
            "attack_share": round(float(group["label"].mean()), 3),
            "median_chars_attack": float(group_attacks["text"].str.len().median())
            if len(group_attacks)
            else None,
            "median_chars_benign": float(group.loc[group["label"] == 0, "text"].str.len().median())
            if (group["label"] == 0).any()
            else None,
            "length_auc": length_auc(group),
            "styles": group_attacks["text"].map(attack_style).value_counts().to_dict(),
        }
    return {
        "length_auc_all": length_auc(df),
        "per_source": per_source,
        "styles_all_attacks": attacks["text"].map(attack_style).value_counts().to_dict(),
    }


def build(dest: Path | None = None) -> tuple[pd.DataFrame, dict]:
    frames = [load_source(name, download(name, dest)) for name in SOURCES]
    df, stats = merge(frames)
    stats["rows_in_per_source"] = {f["source"].iloc[0]: len(f) for f in frames}
    return df, stats


if __name__ == "__main__":
    merged, stats = build()
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out / "prompts.parquet", index=False)
    report = {"merge": stats, **summarise(merged)}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/datasets.json").write_text(json.dumps(report, indent=2))
    from guard.plots import plot_lengths

    plot_lengths(merged, Path("reports/figures/lengths.png"))
    print(json.dumps(report, indent=2))
