from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BENIGN, ATTACK = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


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
    ax.set_title("Attacks are longer than benign prompts in every source that has both", color=INK, loc="left")
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
