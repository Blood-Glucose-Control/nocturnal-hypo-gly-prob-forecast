"""
Plot WQL (or RMSE fallback) vs fine_tune_steps from a single-run checkpoint sweep.

Reads nocturnal_results.json from:
    experiments/nocturnal_forecasting/512ctx_96fh/chronos2/250k_checkpoints/
        step_<N>_<dataset>/nocturnal_results.json

Results are produced by running:
    bash scripts/experiments/run_sweep03_checkpoint_evals.sh <artifact_dir>

after training with 99_250k_checkpoints.yaml (or any config with
checkpoint_save_steps set).  The script discovers all available step × dataset
combinations dynamically — no hardcoded step counts needed.

NOTE: WQL is reported when eval was run with --probabilistic; results produced
without that flag will have overall_rmse only, which is used as a fallback.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_BASE = Path("experiments/nocturnal_forecasting/512ctx_96fh/chronos2")
CHECKPOINT_BASE = EXPERIMENT_BASE / "250k_checkpoints"
RESULTS_FILE = "nocturnal_results.json"
OUTPUT_PATH = Path("results/checkpoint_wql_vs_steps_all_datasets.png")

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022"]
COLORS = {"aleppo_2017": "#1f77b4", "brown_2019": "#ff7f0e", "lynch_2022": "#2ca02c"}
LABELS = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
}

# step_<N>_<dataset>
_DIR_RE = re.compile(r"^step_(\d+)_(.+)$")


def discover_step_results(dataset: str) -> list[tuple[int, float]]:
    """Return sorted (step, metric_value) pairs for all available checkpoints for dataset.

    Prefers overall_wql (requires --probabilistic eval); falls back to overall_rmse.
    """
    if not CHECKPOINT_BASE.exists():
        return []
    results = []
    for d in CHECKPOINT_BASE.iterdir():
        if not d.is_dir():
            continue
        m = _DIR_RE.match(d.name)
        if not m or m.group(2) != dataset:
            continue
        results_file = d / RESULTS_FILE
        if not results_file.exists():
            continue
        with open(results_file) as f:
            data = json.load(f)
        metric = data.get("overall_wql") or data.get("overall_rmse")
        if metric is not None:
            results.append((int(m.group(1)), metric))
    return sorted(results)


def main():
    fig, ax = plt.subplots(figsize=(10, 6))
    found_any = False

    for dataset in DATASETS:
        points = discover_step_results(dataset)
        if not points:
            print(f"{dataset}: no results found under {CHECKPOINT_BASE}")
            continue
        for steps, metric in points:
            print(f"{dataset:12s}  {steps // 1000:>4}k  WQL={metric:.4f}")
        xs = [p[0] / 1000 for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            markersize=6,
            color=COLORS[dataset],
            label=LABELS[dataset],
        )
        found_any = True

    if not found_any:
        print(f"No checkpoint eval results found under {CHECKPOINT_BASE}")
        return

    ax.set_xlabel("Fine-tune Steps (k)", fontsize=13)
    ax.set_ylabel("WQL (↓ better)", fontsize=13)
    ax.set_title("Checkpoint Sweep — WQL vs Fine-tune Steps", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}k"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
