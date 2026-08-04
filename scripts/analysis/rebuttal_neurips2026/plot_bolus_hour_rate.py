#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""Hour-of-day bolus event rate per dataset.

For each of the 3 datasets with logged bolus events (Replace-BG, DCLP3, IOBP2),
plot the fraction of the dataset's total bolus events that fall in each clock
hour (0-23). Adds a uniform-rate reference line at 1/24 and shades the 00:00
to 08:00 nocturnal forecast window.

Reads: scripts/analysis/rebuttal_neurips2026/outputs/a3_hour_histogram.csv
Writes: scripts/analysis/rebuttal_neurips2026/outputs/figures/bolus_hour_rate.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "outputs" / "a3_hour_histogram.csv"
OUT_DIR = HERE / "outputs" / "figures"

DATASET_LABEL = {
    "aleppo_2017": "Replace-BG",
    "brown_2019": "DCLP3",
    "lynch_2022": "IOBP2",
}
DATASET_ORDER = ["aleppo_2017", "brown_2019", "lynch_2022"]
DATASET_COLOR = {
    "aleppo_2017": "tab:blue",
    "brown_2019": "tab:orange",
    "lynch_2022": "tab:green",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    df = df[df["dataset"].isin(DATASET_ORDER)].copy()
    fracs = {}
    totals = {}
    for d in DATASET_ORDER:
        sub = df[df.dataset == d].sort_values("hour")
        tot = float(sub["n_bolus"].sum())
        totals[d] = tot
        fracs[d] = (
            sub.set_index("hour")["n_bolus"].reindex(range(24), fill_value=0).to_numpy()
            / tot
        )

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(24)
    width = 0.27
    for i, d in enumerate(DATASET_ORDER):
        ax.bar(
            x + (i - 1) * width,
            100 * fracs[d],
            width=width,
            color=DATASET_COLOR[d],
            label=f"{DATASET_LABEL[d]} (n={int(totals[d]):,})",
            edgecolor="none",
        )
    # nocturnal shading
    ax.axvspan(-0.5, 7.5, color="gray", alpha=0.08, zorder=0)
    ax.text(
        3.5,
        ax.get_ylim()[1] * 0.98,
        "00:00\u201308:00",
        ha="center",
        va="top",
        fontsize=8,
        color="dimgray",
    )
    # uniform-rate reference (1/24 = 4.17%)
    ax.axhline(
        100 / 24,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=f"uniform ({100/24:.2f}%)",
    )
    ax.set_xlabel("hour of day")
    ax.set_ylabel("bolus events (% of dataset total)")
    ax.set_xticks(x)
    ax.set_xlim(-0.6, 23.6)
    ax.set_title("Fraction of total logged bolus events by hour of day")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"bolus_hour_rate.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
