#!/usr/bin/env python3
"""Compare nocturnal forecast results from pre-computed JSON files.

Reads N nocturnal_results.json files (produced by nocturnal_hypo_eval.py),
matches episodes by (patient_id, anchor), and generates comparison plots +
summary statistics. No model imports — runs in any Python env with
numpy + matplotlib.

Usage:
    python scripts/analysis/compare_forecasts.py \
        --results experiments/.../chronos2/nocturnal_results.json "Chronos2-ft" \
                  experiments/.../toto/nocturnal_results.json "Toto-zs" \
        --output-dir experiments/comparisons/chronos2_vs_toto/

    # Subset to specific patients
    python scripts/analysis/compare_forecasts.py \
        --results path/a.json "Model A" path/b.json "Model B" \
        --patients bro_92 bro_57 \
        --max-episodes 20
"""

import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

INTERVAL_MIN = 5
CONTEXT_TAIL_STEPS = 36  # 3h of context shown in plots

COLOR_CYCLE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def parse_results_args(results_args: list) -> list:
    """Parse --results args as alternating path label pairs.

    Accepts: path1 label1 path2 label2 ...
    Labels are required — every path must be followed by a label string.
    """
    if len(results_args) % 2 != 0:
        print(
            "Error: --results requires pairs of <path> <label>",
            file=sys.stderr,
        )
        sys.exit(1)
    return [
        (results_args[i], results_args[i + 1]) for i in range(0, len(results_args), 2)
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Alternating pairs of: path/to/nocturnal_results.json label",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Filter to these patient IDs (default: all common patients)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Max episodes to plot (default: all matched episodes)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["rmse", "divergence", "patient"],
        default="rmse",
        help="Sort episodes by: rmse (first model), divergence (max pred diff), patient",
    )
    parser.add_argument("--output-dir", default="experiments/comparisons")
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    entries = parse_results_args(args.results)

    # Load and index all result files (keep only the index, not the full data)
    models = []
    for path, label in entries:
        with open(path) as f:
            data = json.load(f)
        n_episodes = len(data["per_episode"])
        index = {(ep["patient_id"], ep["anchor"]): ep for ep in data["per_episode"]}
        models.append({"label": label, "index": index})
        print(f"Loaded {label}: {n_episodes} episodes from {path}")

    # Find common episodes across all models
    common_keys = set(models[0]["index"].keys())
    for m in models[1:]:
        common_keys &= set(m["index"].keys())

    if args.patients:
        patient_set = set(args.patients)
        common_keys = {k for k in common_keys if k[0] in patient_set}

    print(f"Common episodes: {len(common_keys)}")
    if not common_keys:
        print(
            "No common episodes found. Check that result files cover the same patients/dataset."
        )
        sys.exit(1)

    # Build matched results
    labels = [m["label"] for m in models]
    matched = []
    for key in common_keys:
        patient_id, anchor = key
        ep0 = models[0]["index"][key]
        forecast_len = len(ep0["pred"])

        entry = {
            "patient_id": patient_id,
            "anchor": anchor,
            "target_bg": np.array(ep0["target_bg"][:forecast_len]),
            "context_bg": np.array(ep0["context_bg"][-CONTEXT_TAIL_STEPS:]),
            "forecasts": {},
            "rmses": {},
        }
        for m in models:
            ep = m["index"][key]
            entry["forecasts"][m["label"]] = np.array(ep["pred"][:forecast_len])
            entry["rmses"][m["label"]] = float(ep["rmse"])
        matched.append(entry)

    # Free index memory now that matching is done
    del models

    # Sort
    if args.sort_by == "rmse":
        matched.sort(key=lambda r: r["rmses"].get(labels[0], float("inf")))
    elif args.sort_by == "divergence" and len(labels) >= 2:
        # Pre-compute divergence once per episode to avoid repeated work during sort
        for r in matched:
            preds = list(r["forecasts"].values())
            r["_divergence"] = max(
                np.sqrt(np.mean((preds[i] - preds[j]) ** 2))
                for i in range(len(preds))
                for j in range(i + 1, len(preds))
            )
        matched.sort(key=lambda r: r["_divergence"], reverse=True)
    elif args.sort_by == "patient":
        matched.sort(key=lambda r: (r["patient_id"], r["anchor"]))

    if args.max_episodes:
        matched = matched[: args.max_episodes]

    n_plots = len(matched)
    print(f"Plotting {n_plots} episodes")

    # --- Output setup ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Plot grid ---
    label_colors = {
        label: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, label in enumerate(labels)
    }

    n_cols = min(5, n_plots)
    n_rows = max(1, (n_plots + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    forecast_len = len(matched[0]["target_bg"])
    time_ctx = np.arange(-CONTEXT_TAIL_STEPS, 0) * INTERVAL_MIN / 60
    time_fh = np.arange(forecast_len) * INTERVAL_MIN / 60

    for idx, r in enumerate(matched):
        ax = axes_flat[idx]
        ax.plot(time_ctx, r["context_bg"], color="gray", linewidth=0.8, alpha=0.6)
        ax.plot(
            time_fh, r["target_bg"], color="black", linewidth=1.5, label="Ground truth"
        )
        for label in labels:
            ax.plot(
                time_fh,
                r["forecasts"][label],
                color=label_colors[label],
                linewidth=1.2,
                label=label,
            )
        ax.axhline(y=3.9, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        rmse_parts = [f"{label}:{r['rmses'][label]:.1f}" for label in labels]
        ax.set_title(f"{r['patient_id']} | {' '.join(rmse_parts)}", fontsize=7)
        ax.set_xlim(time_ctx[0], time_fh[-1])
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=5, loc="upper right")

    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Nocturnal Forecast Comparison — {', '.join(labels)}\n"
        f"Black: ground truth | Red dashed: 3.9 mmol/L | {n_plots} episodes",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()

    out_name = args.output_name or (
        "comparison_"
        + "_vs_".join(lbl.replace(" ", "-") for lbl in labels)
        + f"_{n_plots}ep.png"
    )
    out_path = os.path.join(args.output_dir, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")
    plt.close()

    # --- RMSE summary CSV ---
    csv_path = os.path.join(args.output_dir, out_name.replace(".png", "_rmse.csv"))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "anchor"] + [f"rmse_{lbl}" for lbl in labels])
        for r in matched:
            writer.writerow(
                [r["patient_id"], r["anchor"]]
                + [f"{r['rmses'][lbl]:.4f}" for lbl in labels]
            )
    print(f"RMSE CSV saved to: {csv_path}")

    # --- Summary stats ---
    print(f"\nSummary ({n_plots} episodes):")
    for label in labels:
        rmses = [r["rmses"][label] for r in matched]
        print(
            f"  {label}: mean RMSE = {np.mean(rmses):.3f}, median = {np.median(rmses):.3f}"
        )

    if len(labels) >= 2:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                li, lj = labels[i], labels[j]
                wins_j = sum(r["rmses"][lj] < r["rmses"][li] for r in matched)
                wins_i = sum(r["rmses"][li] < r["rmses"][lj] for r in matched)
                ties = n_plots - wins_i - wins_j
                print(
                    f"  {lj} wins {wins_j}/{n_plots}, {li} wins {wins_i}/{n_plots}, ties {ties}"
                )


if __name__ == "__main__":
    main()
