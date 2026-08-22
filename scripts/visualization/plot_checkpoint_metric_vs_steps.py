"""Plot checkpoint-sweep metric vs fine-tune steps for one or more datasets.

What this visualization tells you
---------------------------------
- Whether additional fine-tuning steps are still improving quality or have
  plateaued/regressed for each dataset.
- Relative dataset sensitivity to overtraining.

What to look for
----------------
- Stable downward trend (good): more steps are improving the metric.
- U-turn/upward tail (warning): late checkpoints are degrading.
- Divergence between datasets: one dataset may need an earlier/later stop.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_CHECKPOINT_BASE = (
    "experiments/nocturnal_forecasting/512ctx_96fh/chronos2/250k_checkpoints"
)
DEFAULT_RESULTS_FILE = "nocturnal_results.json"
DEFAULT_OUTPUT_PATH = "results/checkpoint_wql_vs_steps_all_datasets.png"
DEFAULT_DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022"]
COLORS = {"aleppo_2017": "#1f77b4", "brown_2019": "#ff7f0e", "lynch_2022": "#2ca02c"}
LABELS = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
}

# step_<N>_<dataset>
_DIR_RE = re.compile(r"^step_(\d+)_(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot checkpoint sweep metric vs fine-tune steps from per-step eval results."
        )
    )
    parser.add_argument(
        "--checkpoint-base",
        default=DEFAULT_CHECKPOINT_BASE,
        help="Directory containing step_<N>_<dataset>/ subdirectories.",
    )
    parser.add_argument(
        "--results-file",
        default=DEFAULT_RESULTS_FILE,
        help="Per-run JSON file under each step_<N>_<dataset>/ directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset IDs to include in the plot.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="Checkpoint Sweep — WQL vs Fine-tune Steps",
        help="Plot title.",
    )
    return parser.parse_args()


def discover_step_results(
    dataset: str, checkpoint_base: Path, results_file_name: str
) -> tuple[list[tuple[int, float]], bool]:
    """Return sorted (step, metric_value) and whether WQL was used anywhere."""
    if not checkpoint_base.exists():
        return [], False
    results = []
    saw_wql = False
    for d in checkpoint_base.iterdir():
        if not d.is_dir():
            continue
        m = _DIR_RE.match(d.name)
        if not m or m.group(2) != dataset:
            continue
        results_file = d / results_file_name
        if not results_file.exists():
            continue
        with open(results_file) as f:
            data = json.load(f)
        wql = data.get("overall_wql")
        if wql is not None:
            saw_wql = True
        metric = wql if wql is not None else data.get("overall_rmse")
        if metric is not None:
            results.append((int(m.group(1)), metric))
    return sorted(results), saw_wql


def main():
    args = parse_args()
    checkpoint_base = Path(args.checkpoint_base)
    output_path = Path(args.output)
    fig, ax = plt.subplots(figsize=(10, 6))
    found_any = False
    saw_any_wql = False

    for dataset in args.datasets:
        points, saw_wql = discover_step_results(
            dataset=dataset,
            checkpoint_base=checkpoint_base,
            results_file_name=args.results_file,
        )
        saw_any_wql = saw_any_wql or saw_wql
        if not points:
            print(f"{dataset}: no results found under {checkpoint_base}")
            continue
        for steps, metric in points:
            print(f"{dataset:12s}  {steps // 1000:>4}k  metric={metric:.4f}")
        xs = [p[0] / 1000 for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            markersize=6,
            color=COLORS.get(dataset),
            label=LABELS.get(dataset, dataset),
        )
        found_any = True

    if not found_any:
        print(f"No checkpoint eval results found under {checkpoint_base}")
        return

    ax.set_xlabel("Fine-tune Steps (k)", fontsize=13)
    if saw_any_wql:
        ax.set_ylabel("WQL / RMSE fallback (↓ better)", fontsize=13)
    else:
        ax.set_ylabel("RMSE (↓ better)", fontsize=13)
    ax.set_title(args.title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}k"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
