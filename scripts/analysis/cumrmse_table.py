"""
Produce a cumulative RMSE table for best model runs across datasets.

For each checkpoint horizon (15, 30, 60, 120, 240, 360, 480 minutes) reports:
    - mean cumRMSE across episodes
    - ±std of per-episode cumRMSE

Per-episode cumRMSE at step k:
    cumRMSE_i(k) = sqrt( mean_{t=1..k}( (pred_i,t - act_i,t)^2 ) )

Mean and std are then taken across all N episodes.

Outputs:
    - Pretty-printed table to stdout
    - CSV to --output path

Usage:
    python scripts/analysis/cumrmse_table.py
    python scripts/analysis/cumrmse_table.py \
        --csv experiments/nocturnal_forecasting/best_by_model_dataset.csv \
        --output results/figures/cumrmse_table.csv
"""

import argparse
import csv
import io
from pathlib import Path

import numpy as np

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022"]
DATASET_LABELS = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
}

CHECKPOINTS_MIN = [15, 30, 60, 120, 240, 360, 480]
SAMPLING_INTERVAL_MIN = 5

# Newer NPZ-bearing runs for models whose best CSV entry predates the NPZ format
FALLBACK_PATHS: dict[tuple[str, str], str] = {
    ("ttm", "aleppo_2017"): (
        "experiments/nocturnal_forecasting/512ctx_96fh/ttm"
        "/2026-04-16_0922_aleppo_2017_finetuned"
    ),
    ("ttm", "brown_2019"): (
        "experiments/nocturnal_forecasting/512ctx_96fh/ttm"
        "/2026-04-16_0925_brown_2019_finetuned"
    ),
}


def read_best_paths(csv_path: Path) -> dict[tuple[str, str], str]:
    best: dict[tuple[str, str], tuple[float, str]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["dataset"] not in DATASETS:
                continue
            key = (row["model"], row["dataset"])
            rmse = float(row["rmse"])
            if key not in best or rmse < best[key][0]:
                best[key] = (rmse, row["run_path"])
    return {k: v[1] for k, v in best.items()}


def load_npz(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(run_dir / "forecasts.npz", allow_pickle=False)
    return data["predictions"], data["actuals"]


def cumrmse_at_checkpoints(
    preds: np.ndarray,
    actuals: np.ndarray,
    checkpoints_min: list[int],
) -> dict[int, tuple[float, float]]:
    """Return {horizon_min: (mean_cumRMSE, std_cumRMSE)} across episodes.

    Per-episode cumRMSE at step k:
        cumRMSE_i(k) = sqrt( (1/k) * sum_{t=1}^{k} (pred_i,t - act_i,t)^2 )

    Mean and std are computed across episodes.
    """
    sq_err = (preds - actuals) ** 2  # (n_episodes, n_steps)
    n_steps = preds.shape[1]
    results = {}
    for horizon_min in checkpoints_min:
        k = horizon_min // SAMPLING_INTERVAL_MIN
        if k > n_steps:
            results[horizon_min] = (float("nan"), float("nan"))
            continue
        # Running mean of squared errors per episode up to step k
        per_ep_cum_mse = np.mean(sq_err[:, :k], axis=1)  # (n_episodes,)
        per_ep_cum_rmse = np.sqrt(per_ep_cum_mse)
        results[horizon_min] = (
            float(np.mean(per_ep_cum_rmse)),
            float(np.std(per_ep_cum_rmse)),
        )
    return results


def format_table(rows: list[dict], checkpoints: list[int]) -> str:
    """Pretty-print a fixed-width table."""
    col_w = 12
    horizon_headers = [f"{m}min" for m in checkpoints]

    buf = io.StringIO()

    def write_header(dataset_label: str) -> None:
        buf.write(f"\n{'─' * (14 + col_w * len(checkpoints))}\n")
        buf.write(f"  {dataset_label}\n")
        buf.write(f"{'─' * (14 + col_w * len(checkpoints))}\n")
        buf.write(f"  {'Model':<12}")
        for h in horizon_headers:
            buf.write(f"  {h:>{col_w - 2}}")
        buf.write("\n")
        buf.write(f"  {'':12}")
        for _ in horizon_headers:
            buf.write(f"  {'mean ± std':>{col_w - 2}}")
        buf.write("\n")
        buf.write(f"{'─' * (14 + col_w * len(checkpoints))}\n")

    last_ds = None
    for row in rows:
        if row["dataset"] != last_ds:
            write_header(DATASET_LABELS[row["dataset"]])
            last_ds = row["dataset"]
        buf.write(f"  {row['model']:<12}")
        for m in checkpoints:
            mean, std = row["metrics"][m]
            if np.isnan(mean):
                cell = "  N/A"
            else:
                cell = f"{mean:.3f}±{std:.3f}"
            buf.write(f"  {cell:>{col_w - 2}}")
        buf.write("\n")

    buf.write(f"{'─' * (14 + col_w * len(checkpoints))}\n")
    return buf.getvalue()


def write_csv(rows: list[dict], checkpoints: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["dataset", "model"]
        + [f"{m}min_mean" for m in checkpoints]
        + [f"{m}min_std" for m in checkpoints]
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {"dataset": row["dataset"], "model": row["model"]}
            for m in checkpoints:
                mean, std = row["metrics"][m]
                out[f"{m}min_mean"] = f"{mean:.4f}" if not np.isnan(mean) else ""
                out[f"{m}min_std"] = f"{std:.4f}" if not np.isnan(std) else ""
            writer.writerow(out)
    print(f"CSV saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="experiments/nocturnal_forecasting/best_by_model_dataset.csv",
    )
    parser.add_argument(
        "--output",
        default="results/figures/cumrmse_table.csv",
    )
    args = parser.parse_args()

    best_paths = read_best_paths(Path(args.csv))
    for key, path in FALLBACK_PATHS.items():
        best_paths[key] = path

    # Collect models with full coverage
    all_models = sorted({m for m, _ in best_paths})
    complete_models = [
        m
        for m in all_models
        if all(
            (m, ds) in best_paths
            and (Path(best_paths[(m, ds)]) / "forecasts.npz").exists()
            for ds in DATASETS
        )
    ]

    # Build rows ordered by dataset then by mean cumRMSE at 480 min
    rows = []
    for ds in DATASETS:
        ds_rows = []
        for model in complete_models:
            preds, actuals = load_npz(Path(best_paths[(model, ds)]))
            metrics = cumrmse_at_checkpoints(preds, actuals, CHECKPOINTS_MIN)
            ds_rows.append({"dataset": ds, "model": model, "metrics": metrics})
        # Sort best → worst by 480-min cumRMSE mean
        ds_rows.sort(key=lambda r: r["metrics"][480][0])
        rows.extend(ds_rows)

    print(format_table(rows, CHECKPOINTS_MIN))
    write_csv(rows, CHECKPOINTS_MIN, Path(args.output))


if __name__ == "__main__":
    main()
