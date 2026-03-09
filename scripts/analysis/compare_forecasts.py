#!/usr/bin/env python3
"""Generic model forecast comparison on holdout episodes.

Compares any combination of models (zero-shot or fine-tuned) on midnight-
anchored nocturnal episodes from the holdout set. Generates a grid plot
with context + ground truth + all model forecasts overlaid.

Model specs use the format: type:checkpoint:label
  - type        required (toto, chronos2, ttm, sundial, ...)
  - checkpoint  optional; empty = zero-shot
  - label       optional; defaults to 'type' or 'type-ft'

Usage:
    python scripts/analysis/compare_forecasts.py \\
        --model toto::Zero-shot \\
        --model toto:path/to/ft/checkpoint:Fine-tuned \\
        --model chronos2:path/to/ckpt:Chronos2
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.versioning.dataset_registry import DatasetRegistry
from src.evaluation.episode_builders import (
    DEFAULT_HOLDOUT_CONFIG_DIR,
    build_patient_episodes,
    get_holdout_patients,
    select_episodes_stratified,
)
from src.models.factory import create_model_and_config

CONTEXT_LENGTH = 512
FORECAST_LENGTH = 72
INTERVAL_MIN = 5
DEFAULT_DATASET = "brown_2019"
DEFAULT_EPISODES_PER_PATIENT = 4
DEFAULT_SEED = 42

COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def parse_model_spec(spec):
    """Parse 'type:checkpoint:label' into (model_type, checkpoint, label)."""
    parts = spec.split(":")
    model_type = parts[0]
    checkpoint = parts[1] if len(parts) > 1 and parts[1] else None
    if len(parts) > 2 and parts[2]:
        label = parts[2]
    else:
        label = model_type if not checkpoint else f"{model_type}-ft"
    return model_type, checkpoint, label


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", action="append", required=True,
        help="type:checkpoint:label  (repeatable; empty checkpoint = zero-shot)",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config-dir", default=DEFAULT_HOLDOUT_CONFIG_DIR)
    parser.add_argument("--episodes-per-patient", type=int, default=DEFAULT_EPISODES_PER_PATIENT)
    parser.add_argument("--patients", nargs="+", default=None,
                        help="Holdout patient IDs to include (default: all holdout patients)")
    parser.add_argument("--covariate-cols", nargs="+", default=None,
                        help="Covariate columns to include in episodes (e.g. iob); "
                             "fine-tuned checkpoints are also auto-detected")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    model_specs = [parse_model_spec(s) for s in args.model]

    output_dir = args.output_dir or f"experiments/{args.dataset}_comparison"
    os.makedirs(output_dir, exist_ok=True)

    patients = (
        args.patients
        if args.patients is not None
        else get_holdout_patients(args.dataset, args.config_dir)
    )
    print(f"Using {len(patients)} patients")

    print("Loading holdout data...")
    holdout_data = DatasetRegistry(holdout_config_dir=args.config_dir).load_holdout_data_only(args.dataset)

    # Collect covariate cols needed by any fine-tuned model (auto-detected from config.json)
    covariate_cols = set(args.covariate_cols or [])
    for _, checkpoint, _ in model_specs:
        if checkpoint:
            config_path = os.path.join(checkpoint, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    for col in (json.load(f).get("covariate_cols") or []):
                        covariate_cols.add(col)
    covariate_cols = list(covariate_cols) or None

    print("Building episodes...")
    episodes_by_patient = build_patient_episodes(
        holdout_data, patients, CONTEXT_LENGTH, FORECAST_LENGTH, covariate_cols=covariate_cols,
    )
    selected = select_episodes_stratified(episodes_by_patient, args.episodes_per_patient, args.seed)
    print(f"Selected {len(selected)} episodes ({args.episodes_per_patient}/patient, {len(episodes_by_patient)} patients)")

    # Load models
    models = []
    for model_type, checkpoint, label in model_specs:
        print(f"Loading {label} ({model_type}{', zero-shot' if not checkpoint else ''})...")
        kwargs: dict = {"forecast_length": FORECAST_LENGTH}
        if covariate_cols and model_type == "toto":
            kwargs["covariate_cols"] = covariate_cols
        model, _ = create_model_and_config(model_type, checkpoint, **kwargs)
        models.append((model, label, model_type))

    label_colors = {label: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, (_, label, _) in enumerate(models)}

    # Generate forecasts
    print("Generating forecasts...")
    results = []
    for i, ep in enumerate(selected):
        ctx = ep["context_df"].copy().reset_index(names="datetime")
        ctx["p_num"] = ep["patient_id"]
        target = ep["target_bg"][:FORECAST_LENGTH]

        try:
            forecasts, rmses = {}, {}
            for model, label, _ in models:
                pred = model.predict(ctx)[:FORECAST_LENGTH]
                forecasts[label] = pred
                rmses[label] = np.sqrt(np.mean((pred[:len(target)] - target) ** 2))

            results.append({
                "patient_id": ep["patient_id"],
                "anchor": str(ep["anchor"]),
                "context_bg": ep["context_df"]["bg_mM"].values[-36:],
                "target_bg": target,
                "forecasts": forecasts,
                "rmses": rmses,
            })
            rmse_str = "  ".join(f"{label}={rmses[label]:.2f}" for _, label, _ in models)
            print(f"  [{i+1}/{len(selected)}] {ep['patient_id']} {ep['anchor']}: {rmse_str}")

        except Exception as e:
            print(f"  [{i+1}/{len(selected)}] {ep['patient_id']} {ep['anchor']}: FAILED - {e}")

    print(f"\nSuccessful forecasts: {len(results)}/{len(selected)}")
    if not results:
        return

    # Sort by first model RMSE (best to worst)
    results.sort(key=lambda r: r["rmses"].get(model_specs[0][2], float("inf")))

    # --- Plot ---
    n_plots = len(results)
    n_cols = 5
    n_rows = max(1, (n_plots + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    time_ctx = np.arange(-36, 0) * INTERVAL_MIN / 60  # 3h context tail (hours)
    time_fh = np.arange(FORECAST_LENGTH) * INTERVAL_MIN / 60  # forecast horizon (hours)

    for idx, r in enumerate(results):
        ax = axes[idx]
        ax.plot(time_ctx, r["context_bg"], color="gray", linewidth=0.8, alpha=0.6)
        ax.plot(time_fh, r["target_bg"], color="black", linewidth=1.5, label="Ground truth")
        for _, label, _ in models:
            ax.plot(time_fh, r["forecasts"][label], color=label_colors[label], linewidth=1.2, label=label)
        ax.axhline(y=3.9, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        rmse_parts = [f"{label}:{r['rmses'][label]:.1f}" for _, label, _ in models]
        ax.set_title(f"{r['patient_id']} | {' '.join(rmse_parts)}", fontsize=7)
        ax.set_xlim(time_ctx[0], time_fh[-1])
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=5, loc="upper right")

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Nocturnal Forecast Comparison — {', '.join(label for _, label, _ in models)}\n"
        f"Black: ground truth | Red dashed: 3.9 mmol/L",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()

    out_name = args.output_name or (
        "comparison_"
        + "_vs_".join(label.replace(" ", "-") for _, label, _ in models)
        + f"_{n_plots}ep.png"
    )
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.close()

    # Summary
    labels = [label for _, label, _ in models]
    print(f"\nSummary ({n_plots} episodes):")
    for label in labels:
        print(f"  {label}: mean RMSE = {np.mean([r['rmses'][label] for r in results]):.3f}")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            li, lj = labels[i], labels[j]
            wins = sum(r["rmses"][lj] < r["rmses"][li] for r in results)
            print(f"  {lj} wins over {li}: {wins}/{n_plots}")


if __name__ == "__main__":
    main()
