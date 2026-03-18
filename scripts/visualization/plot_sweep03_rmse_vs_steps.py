"""
Plot RMSE vs fine_tune_steps for sweep-03 across aleppo_2017, brown_2019, lynch_2022.
Reads overall_rmse from nocturnal_results.json in each experiment output dir.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_BASE = Path("experiments/nocturnal_forecasting/512ctx_96fh/chronos2")
RESULTS_FILE = "nocturnal_results.json"
OUTPUT_PATH = Path("results/sweep_03_rmse_vs_steps_all_datasets.png")

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022"]
COLORS = {"aleppo_2017": "#1f77b4", "brown_2019": "#ff7f0e", "lynch_2022": "#2ca02c"}
LABELS = {"aleppo_2017": "Aleppo 2017", "brown_2019": "Brown 2019", "lynch_2022": "Lynch 2022"}

# Map step count → glob pattern for the main (non-resumed) model.pt runs
# Pattern: *sweep03_{N}k_modelpt_{dataset}_finetuned  or  *recheck_03_modelpt_{dataset}_finetuned (50k)
STEP_PATTERNS = {
    5_000:  ("*sweep03_5k_modelpt_{dataset}_finetuned",),
    10_000: ("*sweep03_10k_modelpt_{dataset}_finetuned",),
    15_000: ("*sweep03_15k_modelpt_{dataset}_finetuned",),
    20_000: ("*sweep03_20k_modelpt_{dataset}_finetuned",),
    25_000: ("*sweep03_25k_modelpt_{dataset}_finetuned",),
    30_000: ("*sweep03_30k_modelpt_{dataset}_finetuned",),
    35_000: ("*sweep03_35k_modelpt_{dataset}_finetuned",),
    40_000: ("*sweep03_40k_modelpt_{dataset}_finetuned",),
    45_000: ("*sweep03_45k_modelpt_{dataset}_finetuned",),
    50_000: ("*recheck_03_modelpt_{dataset}_finetuned",),
}


def find_rmse(steps: int, dataset: str) -> float | None:
    for pattern_tmpl in STEP_PATTERNS[steps]:
        pattern = pattern_tmpl.format(dataset=dataset)
        matches = sorted(EXPERIMENT_BASE.glob(pattern))
        if not matches:
            continue
        results_file = matches[-1] / RESULTS_FILE  # take latest if multiple
        if not results_file.exists():
            continue
        with open(results_file) as f:
            data = json.load(f)
        return data["overall_rmse"]
    return None


def main():
    all_steps = sorted(STEP_PATTERNS.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset in DATASETS:
        xs, ys = [], []
        for steps in all_steps:
            rmse = find_rmse(steps, dataset)
            if rmse is not None:
                xs.append(steps)
                ys.append(rmse)
                print(f"{dataset:12s}  {steps//1000:>3}k  RMSE={rmse:.4f}")
            else:
                print(f"{dataset:12s}  {steps//1000:>3}k  (no data)")
        if xs:
            ax.plot(
                [x / 1000 for x in xs],
                ys,
                marker="o",
                linewidth=2,
                markersize=6,
                color=COLORS[dataset],
                label=LABELS[dataset],
            )

    ax.set_xlabel("Fine-tune Steps (k)", fontsize=13)
    ax.set_ylabel("RMSE", fontsize=13)
    ax.set_title("Sweep-03 RMSE vs Fine-tune Steps", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}k"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
