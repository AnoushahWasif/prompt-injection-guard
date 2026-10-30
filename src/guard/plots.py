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


def plot_baseline(report: dict, path: Path) -> None:
    """Share of attacks caught at the same false-alarm budget: length-only vs TF-IDF."""
    rows = [("all test prompts", "all")]
    rows += [(name, name) for name in report["models"]["tfidf"]["test"]["per_source"]]
    rows.append(("short attacks", "short_attacks"))

    def recall(model: str, key: str) -> float:
        test = report["models"][model]["test"]
        part = test.get(key) if key in ("all", "short_attacks") else test["per_source"][key]
        return part["recall"] or 0.0

    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.4), dpi=150)
    for offset, model, label, color in (
        (0.18, "length", "length only", BASELINE),
        (-0.18, "tfidf", "TF-IDF + logistic regression", BENIGN),
    ):
        values = [recall(model, key) for _, key in rows]
        ax.barh(y + offset, values, height=0.34, color=color, label=label)
        for yi, v in zip(y + offset, values, strict=True):
            ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [name for name, _ in rows])
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    fpr = report["target_fpr"]
    xlabel = f"Attacks caught, threshold set for {fpr:.0%} of benign prompts flagged"
    ax.set_xlabel(xlabel, color=MUTED)
    caught = {m: report["models"][m]["test"]["all"]["recall"] for m in ("length", "tfidf")}
    title = f"TF-IDF catches {caught['tfidf']:.0%} of attacks; length alone, {caught['length']:.0%}"
    ax.set_title(title, color=INK, loc="left", pad=24)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
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
