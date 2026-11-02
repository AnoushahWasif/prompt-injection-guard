from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BENIGN, ATTACK = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BASELINE = "#a3a29e"


def plot_lengths(df: pd.DataFrame, path: Path) -> None:
    """Prompt length per source, benign next to attack, on a log scale."""
    sources = sorted(df["source"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for i, source in enumerate(sources):
        for offset, label, color in ((-0.18, 0, BENIGN), (0.18, 1, ATTACK)):
            lengths = df.loc[(df["source"] == source) & (df["label"] == label), "text"].str.len()
            if lengths.empty:
                continue
            box = ax.boxplot(
                lengths,
                positions=[i + offset],
                widths=0.3,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": INK, "lw": 1.5},
                whiskerprops={"color": MUTED},
                capprops={"color": MUTED},
            )
            box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(sources)), sources)
    ax.set_ylabel("Characters (log scale)", color=MUTED)
    title = "Attacks are longer than benign prompts in every source that has both"
    ax.set_title(title, color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BENIGN, ATTACK)]
    ax.legend(handles, ["benign", "attack"], frameon=False, fontsize=8, loc="upper left")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


TRANSFORMER = "#1baf7a"


def plot_recall(series: list[tuple[str, str, dict]], fpr: float, title: str, path: Path) -> None:
    """Share of attacks caught per source at the same false-alarm budget, one bar per model.

    `series` is (label, color, test results) in the order the bars are drawn, top to bottom.
    """
    first = series[0][2]
    rows = [("all test prompts", "all")]
    rows += [(name, name) for name in first["per_source"]]
    rows.append(("short attacks", "short_attacks"))

    def recall(test: dict, key: str) -> float:
        part = test[key] if key in ("all", "short_attacks") else test["per_source"][key]
        return part["recall"] or 0.0

    n = len(series)
    height = 0.8 / n
    y = np.arange(len(rows))[::-1] * (n * 0.3 + 0.4)
    fig, ax = plt.subplots(figsize=(8, (0.3 * n + 0.25) * len(rows) + 1.4), dpi=150)
    for i, (label, color, test) in enumerate(series):
        offset = (n - 1) / 2 * height * 0.75 - i * height * 0.75
        values = [recall(test, key) for _, key in rows]
        ys = y + offset
        ax.barh(ys, values, height=height * 0.7, color=color, label=label)
        for yi, v in zip(ys, values, strict=True):
            ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [name for name, _ in rows])
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    xlabel = f"Attacks caught, threshold set for {fpr:.0%} of benign prompts flagged"
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_title(title, color=INK, loc="left", pad=24)
    ax.legend(frameon=False, fontsize=8, ncol=n, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_baseline(report: dict, path: Path) -> None:
    """Length-only vs TF-IDF (Day 3)."""
    models = report["models"]
    caught = {m: models[m]["test"]["all"]["recall"] for m in ("length", "tfidf")}
    title = f"TF-IDF catches {caught['tfidf']:.0%} of attacks; length alone, {caught['length']:.0%}"
    series = [
        ("TF-IDF + logistic regression", BENIGN, models["tfidf"]["test"]),
        ("length only", BASELINE, models["length"]["test"]),
    ]
    plot_recall(series, report["target_fpr"], title, path)
