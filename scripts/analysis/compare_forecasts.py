#!/usr/bin/env python3
"""Generic model forecast comparison on holdout episodes.

Two-stage workflow to handle models that require different Python environments:

  Stage 1 — Inference (run once per model, in its model-specific env):

      source scripts/setup_model_env.sh chronos2
      python scripts/analysis/compare_forecasts.py --model chronos2::Chronos2 --no-plot
      # → prints: Results cached at: results/forecast_comparisons/a3f82c1d.json

      source scripts/setup_model_env.sh ttm
      python scripts/analysis/compare_forecasts.py --model ttm::TTM --no-plot
      # → prints: Results cached at: results/forecast_comparisons/9b14e702.json

  Both runs must use the same --dataset, --seed, --n-patients (or --patients),
  and --forecast-length so the patient lists match for comparison.

  Stage 2 — Plot (run in any env, using the paths printed above):

      python scripts/analysis/compare_forecasts.py \\
          --from-results \\
              results/forecast_comparisons/a3f82c1d.json \\
              results/forecast_comparisons/9b14e702.json
      # → saves figure + sidecar to results/forecast_comparisons/
      # Grid: n_patients rows x 4 cols (P10/P30/P60/P90 RMSE per patient)
      # Override percentile columns: --percentiles 25 50 75

  To list all cached runs and find the files you want:

      python -c "
      import json, pathlib
      for f in sorted(pathlib.Path('results/forecast_comparisons').glob('*.json')):
          m = json.loads(f.read_text())['metadata']
          print(f, m['label'], m['dataset'], 'seed=' + str(m['seed']))
      "

  Single-env shortcut (when all models share an environment, e.g. two
  fine-tuned checkpoints of the same model type):

      python scripts/analysis/compare_forecasts.py \\
          --model ttm::TTM-v1 \\
          --model ttm:trained_models/artifacts/ttm-ft-iob:TTM-IOB

  After Stage 1 the script prints the result file path(s); pass those
  to --from-results in Stage 2. Inference results are cached by content
  hash and reused automatically on subsequent runs.

Model specs use the format: type:checkpoint:label
  - type        required (chronos2, ttm, sundial, timegrad, timesfm, tide, ...)
                (toto support coming in a future PR)
  - checkpoint  optional; empty = zero-shot
  - label       optional; defaults to 'type' or 'type-ft'
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.versioning.dataset_registry import DatasetRegistry
from src.evaluation.episode_builders import (
    DEFAULT_HOLDOUT_CONFIG_DIR,
    build_patient_episodes,
    get_holdout_patients,
)
from src.models.factory import create_model_and_config

DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_FORECAST_LENGTH = 96
INTERVAL_MIN = 5
DEFAULT_DATASET = "brown_2019"
DEFAULT_N_PATIENTS = 5
DEFAULT_PERCENTILES = [10, 30, 70, 90]
DEFAULT_SEED = 42

RESULTS_DIR = Path("results/forecast_comparisons")
FIGURES_DIR = Path("results/forecast_comparisons")

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

# Keys that must match across result files before they can be compared
_COMPATIBILITY_KEYS = ["dataset", "config_dir", "seed", "forecast_length", "patients"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_model_spec(spec: str) -> Tuple[str, Optional[str], str]:
    """Parse 'type:checkpoint:label' into (model_type, checkpoint, label)."""
    parts = spec.split(":")
    model_type = parts[0]
    checkpoint = parts[1] if len(parts) > 1 and parts[1] else None
    if len(parts) > 2 and parts[2]:
        label = parts[2]
    else:
        label = model_type if not checkpoint else f"{model_type}-ft"
    return model_type, checkpoint, label


def select_patients(
    all_patients: List[str],
    n_patients: Optional[int],
    explicit_patients: Optional[List[str]],
    seed: int,
) -> List[str]:
    """Return the resolved patient list for this run.

    If explicit_patients is given, use it (sorted for determinism). Otherwise
    randomly sample n_patients from all_patients using seed.
    """
    if explicit_patients:
        available = set(all_patients)
        missing = [p for p in explicit_patients if p not in available]
        if missing:
            print(f"WARNING: Requested patients not found in holdout data: {missing}")
        return sorted(p for p in explicit_patients if p in available)
    rng = np.random.RandomState(seed)
    n = min(n_patients or DEFAULT_N_PATIENTS, len(all_patients))
    return sorted(rng.choice(sorted(all_patients), n, replace=False).tolist())


def compute_run_hash(
    model_type: str,
    checkpoint: Optional[str],
    dataset: str,
    config_dir: str,
    seed: int,
    context_length: int,
    forecast_length: int,
    patients: List[str],
    covariate_cols: Optional[List[str]],
) -> str:
    """Return an 8-char SHA-256 prefix uniquely identifying this inference run.

    The hash covers all parameters that affect which episodes are selected and
    what predictions are made, so the same set of arguments always maps to the
    same cache file and different arguments never collide.  Display-only fields
    like ``label`` are excluded so that relabelling a run hits the cache.

    Note: patients must be the fully resolved sorted list (not None) so that
    runs with the same patients always hit the same cache entry regardless of
    whether they were specified explicitly or sampled.
    """
    params = {
        "model_type": model_type,
        "checkpoint": checkpoint,
        "dataset": dataset,
        "config_dir": str(config_dir),
        "seed": seed,
        "context_length": context_length,
        "forecast_length": forecast_length,
        "patients": sorted(patients),
        "covariate_cols": sorted(covariate_cols) if covariate_cols else None,
    }
    canonical = json.dumps(params, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def collect_covariate_cols(
    model_specs: List[Tuple],
    extra_cols: Optional[List[str]],
) -> Optional[List[str]]:
    """Union of explicitly requested cols and any cols declared in checkpoint configs."""
    cols: set = set(extra_cols or [])
    for _, checkpoint, _ in model_specs:
        if checkpoint:
            config_path = os.path.join(checkpoint, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    for col in json.load(f).get("covariate_cols") or []:
                        cols.add(col)
    return sorted(cols) if cols else None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model_type: str,
    checkpoint: Optional[str],
    label: str,
    context_length: int,
    forecast_length: int,
    selected_episodes: List[Dict],
) -> List[Dict]:
    """Load model and generate forecasts for all selected episodes.

    Returns a list of per-episode dicts containing forecast arrays and RMSE.
    Episodes that fail are skipped with a printed warning.
    """
    print(f"Loading {label} ({model_type}{', zero-shot' if not checkpoint else ''})...")
    model, _ = create_model_and_config(
        model_type,
        checkpoint,
        context_length=context_length,
        forecast_length=forecast_length,
    )

    results = []
    n = len(selected_episodes)
    for i, ep in enumerate(selected_episodes):
        ctx = ep["context_df"].copy().reset_index(names="datetime")
        ctx["p_num"] = ep["patient_id"]
        target = ep["target_bg"][:forecast_length]
        try:
            pred = model.predict(ctx)[:forecast_length]
            rmse = float(np.sqrt(np.mean((pred[: len(target)] - target) ** 2)))
            results.append(
                {
                    "patient_id": ep["patient_id"],
                    "anchor": str(ep["anchor"]),
                    "context_bg": ep["context_df"]["bg_mM"].values[-36:].tolist(),
                    "target_bg": target.tolist(),
                    "forecast": pred.tolist(),
                    "rmse": rmse,
                }
            )
            print(f"  [{i+1}/{n}] {ep['patient_id']} {ep['anchor']}: RMSE={rmse:.3f}")
        except Exception as exc:
            print(f"  [{i+1}/{n}] {ep['patient_id']} {ep['anchor']}: FAILED — {exc}")

    return results


def save_result_file(episodes: List[Dict], metadata: Dict, out_path: Path) -> None:
    """Write inference results to a JSON cache file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metadata": metadata, "episodes": episodes}, f, indent=2)
    print(f"Results cached at: {out_path}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def load_result_file(path) -> Dict:
    with open(path) as f:
        return json.load(f)


def validate_compatibility(result_files: List[Dict]) -> None:
    """Raise ValueError if result files cannot be safely compared."""
    ref = result_files[0]["metadata"]
    for i, rf in enumerate(result_files[1:], 1):
        for key in _COMPATIBILITY_KEYS:
            if rf["metadata"].get(key) != ref.get(key):
                raise ValueError(
                    f"Result file {i} has incompatible '{key}': "
                    f"{rf['metadata'].get(key)!r} vs {ref.get(key)!r}\n"
                    f"All files must be generated with the same dataset, seed, "
                    f"forecast_length, and patient list."
                )


def pick_percentile_episode(
    patient_keys: List[Tuple],
    ref_index: Dict,
    percentile: int,
) -> Tuple:
    """Return the episode key at the given RMSE percentile for one patient.

    Episodes are sorted ascending by reference-model RMSE, so:
      - 10th percentile → lowest RMSE (easiest / best-fit episode)
      - 90th percentile → highest RMSE (hardest / worst-fit episode)
    """
    sorted_keys = sorted(patient_keys, key=lambda k: ref_index[k]["rmse"])
    idx = int(round(percentile / 100 * (len(sorted_keys) - 1)))
    return sorted_keys[max(0, min(idx, len(sorted_keys) - 1))]


def plot_comparison(
    result_files: List[Dict],
    result_paths: List[str],
    output_dir: Path,
    output_name: Optional[str],
    percentiles: List[int],
) -> None:
    """Render and save the grid comparison plot plus a sidecar JSON.

    Grid layout: one row per patient, one column per percentile.
    Column order matches the percentiles list (default: 90, 60, 30, 10 —
    hardest to easiest by reference-model RMSE within each patient).
    """
    labels = [rf["metadata"]["label"] for rf in result_files]
    label_colors = {
        label: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, label in enumerate(labels)
    }

    # Index episodes by (patient_id, anchor) per model
    indices = [
        {(ep["patient_id"], ep["anchor"]): ep for ep in rf["episodes"]}
        for rf in result_files
    ]
    ref_index = indices[0]  # first model drives percentile ranking

    # Common keys across all models
    common_keys = set(ref_index.keys())
    for idx in indices[1:]:
        common_keys &= set(idx.keys())

    if not common_keys:
        raise ValueError(
            "No episodes in common across result files. "
            "Were they generated with the same --seed and --patients?"
        )

    patients = sorted({k[0] for k in common_keys})
    n_patients = len(patients)
    n_cols = len(percentiles)
    forecast_length = result_files[0]["metadata"]["forecast_length"]
    dataset = result_files[0]["metadata"]["dataset"]

    time_ctx = np.arange(-36, 0) * INTERVAL_MIN / 60
    time_fh = np.arange(forecast_length) * INTERVAL_MIN / 60

    fig, axes = plt.subplots(
        n_patients,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_patients),
        squeeze=False,
        layout="constrained",
    )

    for row, pid in enumerate(patients):
        patient_keys = [k for k in common_keys if k[0] == pid]
        if not patient_keys:
            for col in range(n_cols):
                axes[row][col].set_visible(False)
            continue

        for col, pct in enumerate(percentiles):
            key = pick_percentile_episode(patient_keys, ref_index, pct)
            ax = axes[row][col]

            ep0 = ref_index[key]
            ax.plot(time_ctx, ep0["context_bg"], color="gray", linewidth=0.8, alpha=0.6)
            ax.plot(
                time_fh,
                ep0["target_bg"],
                color="black",
                linewidth=1.5,
                label="Ground truth",
            )
            for label, idx in zip(labels, indices):
                if key in idx:
                    ax.plot(
                        time_fh,
                        idx[key]["forecast"],
                        color=label_colors[label],
                        linewidth=1.2,
                        label=label,
                    )
            ax.axhline(
                y=13.9,
                color="orange",
                linewidth=1.0,
                linestyle="-",
                alpha=0.75,
                label="Very high (13.9 mmol/L)",
            )
            ax.axhline(
                y=10.0,
                color="orange",
                linewidth=0.75,
                linestyle="--",
                alpha=0.75,
                label="High (10.0 mmol/L)",
            )
            ax.axhline(
                y=3.9,
                color="red",
                linewidth=0.75,
                linestyle="--",
                alpha=0.75,
                label="Low (3.9 mmol/L)",
            )
            ax.axhline(
                y=3.0,
                color="red",
                linewidth=1.0,
                linestyle="-",
                alpha=0.75,
                label="Very low (3.0 mmol/L)",
            )
            ax.axvline(x=0, color="gray", linewidth=0.75, linestyle=":", alpha=0.75)
            ax.set_xlim(time_ctx[0], time_fh[-1])
            ax.set_ylim(0, 22.5)
            ax.tick_params(labelsize=6)

            rmse_parts = [
                f"{label}:{indices[i][key]['rmse']:.3f}"
                for i, label in enumerate(labels)
                if key in indices[i]
            ]
            anchor_dt = str(key[1])[:16]  # "YYYY-MM-DD HH:MM" from anchor timestamp
            title = f"{pid} | {anchor_dt} | {' '.join(rmse_parts)}"
            # Top row: prepend column header showing percentile
            if row == 0:
                title = f"$\\mathbf{{Percentile\\ {pct}\\ RMSE}}$\n{title}"
            ax.set_title(title, fontsize=7)

            if col == 0:
                ax.set_ylabel("BG (mmol/L)", fontsize=6)
            if row == n_patients - 1:
                ax.set_xlabel("Time (hours)", fontsize=6)

            # Orientation + threshold annotations on top-left subplot only
            if row == 0 and col == 0:
                actual_ctx_hours = (
                    result_files[0]["metadata"].get("context_length", len(time_ctx))
                    * INTERVAL_MIN
                    / 60
                )
                fh_hours = forecast_length * INTERVAL_MIN / 60
                mid_ctx = (time_ctx[0] + 0) / 2
                mid_fh = (0 + time_fh[-1]) / 2
                ax.text(
                    mid_ctx,
                    21.8,
                    f"\u2190 CONTEXT WINDOW ({actual_ctx_hours:.0f}h)",
                    fontsize=4.5,
                    color="dimgray",
                    va="top",
                    ha="center",
                    style="italic",
                )
                ax.text(
                    mid_fh,
                    21.8,
                    f"FORECASTING HORIZON ({fh_hours:.0f}h) \u2192",
                    fontsize=4.5,
                    color="dimgray",
                    va="top",
                    ha="center",
                    style="italic",
                )
                for y_val, zone_label, clr, va, offset in [
                    (13.9, " VERY HIGH \u2191", "orange", "bottom", +0.25),
                    (10.0, " HIGH \u2191", "orange", "bottom", +0.25),
                    (3.9, " LOW \u2193", "red", "top", -0.25),
                    (3.0, " VERY LOW \u2193", "red", "top", -0.25),
                ]:
                    ax.text(
                        time_ctx[0],
                        y_val + offset,
                        zone_label,
                        fontsize=4.5,
                        color=clr,
                        va=va,
                        ha="left",
                        style="italic",
                        alpha=0.85,
                    )

    label_str = "_vs_".join(label.replace(" ", "-") for label in labels)
    fig.suptitle(
        f"Nocturnal Forecast Comparison — {', '.join(labels)}\n"
        f"Dataset: {dataset} | Columns: RMSE percentile per patient (ref: {labels[0]})",
        fontsize=9,
        fontweight="bold",
    )

    # Horizontal figure-level legend below the subplots; constrained_layout reserves space automatically
    handles, leg_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        leg_labels,
        loc="outside lower center",
        ncol=len(handles),
        fontsize=6,
        frameon=True,
        framealpha=0.9,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_plots = n_patients * n_cols
    out_name = output_name or f"{timestamp}_{label_str}_{n_patients}pat.png"
    sidecar_name = Path(out_name).stem + ".json"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / out_name, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to: {output_dir / out_name}")
    plt.close()

    # Sidecar: records exactly what produced this figure for reproducibility
    sidecar = {
        "created": datetime.now().isoformat(),
        "figure": out_name,
        "result_files": [str(p) for p in result_paths],
        "models": [rf["metadata"] for rf in result_files],
        "patients": patients,
        "percentiles": percentiles,
        "n_plots": n_plots,
        "dataset": dataset,
        "seed": result_files[0]["metadata"]["seed"],
        "forecast_length": forecast_length,
    }
    with open(output_dir / sidecar_name, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Sidecar saved to:  {output_dir / sidecar_name}")

    # Summary
    print(
        f"\nSummary ({len(common_keys)} common episodes across {n_patients} patients):"
    )
    for i, label in enumerate(labels):
        mean_rmse = np.mean(
            [indices[i][k]["rmse"] for k in common_keys if k in indices[i]]
        )
        print(f"  {label}: mean RMSE = {mean_rmse:.3f}")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            li, lj = labels[i], labels[j]
            shared = [k for k in common_keys if k in indices[i] and k in indices[j]]
            wins = sum(indices[j][k]["rmse"] < indices[i][k]["rmse"] for k in shared)
            print(f"  {lj} wins over {li}: {wins}/{len(shared)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="SPEC",
        help="type:checkpoint:label (repeatable). Runs inference and caches results.",
    )
    mode.add_argument(
        "--from-results",
        nargs="+",
        metavar="JSON",
        help="Paths to cached episode_forecasts JSON files. Skips inference.",
    )

    # Inference-mode options
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config-dir", default=DEFAULT_HOLDOUT_CONFIG_DIR)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument(
        "--forecast-length",
        type=int,
        default=DEFAULT_FORECAST_LENGTH,
        help="Forecast horizon in steps (default: 96 = 8 hours at 5-min intervals)",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=DEFAULT_N_PATIENTS,
        help="Number of holdout patients to randomly sample (default: 5); "
        "ignored if --patients is given",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Explicit holdout patient IDs to use (overrides --n-patients)",
    )
    parser.add_argument(
        "--covariate-cols",
        nargs="+",
        default=None,
        help="Covariate columns to include (e.g. iob); "
        "fine-tuned checkpoints are also auto-detected",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help=f"Directory for cached inference JSON files (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Save inference results only; skip plotting even if multiple models given",
    )

    # Plot-mode options
    parser.add_argument(
        "--percentiles",
        type=int,
        nargs="+",
        default=DEFAULT_PERCENTILES,
        help="RMSE percentiles to show as columns in the grid "
        f"(default: {DEFAULT_PERCENTILES}; 10=easiest, 90=hardest)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FIGURES_DIR),
        help=f"Directory for figure output (default: {FIGURES_DIR})",
    )
    parser.add_argument("--output-name", default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Plot mode: --from-results
    # ------------------------------------------------------------------
    if args.from_results:
        result_files = [load_result_file(p) for p in args.from_results]
        validate_compatibility(result_files)
        plot_comparison(
            result_files,
            args.from_results,
            args.output_dir,
            args.output_name,
            args.percentiles,
        )
        return

    # ------------------------------------------------------------------
    # Inference mode: --model
    # ------------------------------------------------------------------
    model_specs = [parse_model_spec(s) for s in args.models]
    covariate_cols = collect_covariate_cols(model_specs, args.covariate_cols)

    # Resolve patient list once — used for both hashing and episode building
    all_holdout = get_holdout_patients(args.dataset, args.config_dir)
    patients = select_patients(all_holdout, args.n_patients, args.patients, args.seed)
    print(f"Using {len(patients)} patients: {patients}")

    results_dir = Path(args.results_dir)

    # Check cache; skip data loading entirely if all models are already cached
    cached: Dict[str, Path] = {}
    to_run = []
    for model_type, checkpoint, label in model_specs:
        run_hash = compute_run_hash(
            model_type,
            checkpoint,
            args.dataset,
            args.config_dir,
            args.seed,
            args.context_length,
            args.forecast_length,
            patients,
            covariate_cols,
        )
        out_path = results_dir / f"{run_hash}.json"
        if out_path.exists():
            print(f"Cache hit [{run_hash}] {label}: {out_path}")
            cached[label] = out_path
        else:
            to_run.append((model_type, checkpoint, label, out_path))

    if to_run:
        print("\nLoading holdout data...")
        holdout_data = DatasetRegistry(
            holdout_config_dir=args.config_dir
        ).load_holdout_data_only(args.dataset)

        print("Building episodes (all midnight anchors for selected patients)...")
        episodes_by_patient = build_patient_episodes(
            holdout_data,
            patients,
            args.context_length,
            args.forecast_length,
            covariate_cols=covariate_cols,
        )
        # Flatten all episodes — percentile selection happens at plot time
        all_episodes = [ep for eps in episodes_by_patient.values() for ep in eps]
        total_patients = len(episodes_by_patient)
        print(f"Built {len(all_episodes)} episodes across {total_patients} patients\n")

        for model_type, checkpoint, label, out_path in to_run:
            episodes = run_inference(
                model_type,
                checkpoint,
                label,
                args.context_length,
                args.forecast_length,
                all_episodes,
            )
            print(f"  {len(episodes)}/{len(all_episodes)} episodes succeeded")
            if episodes:
                metadata = {
                    "model_type": model_type,
                    "checkpoint": checkpoint,
                    "label": label,
                    "dataset": args.dataset,
                    "config_dir": args.config_dir,
                    "seed": args.seed,
                    "context_length": args.context_length,
                    "forecast_length": args.forecast_length,
                    "patients": patients,
                    "covariate_cols": covariate_cols,
                    "created": datetime.now().isoformat(),
                }
                save_result_file(episodes, metadata, out_path)
                cached[label] = out_path
            else:
                print(f"  No successful episodes for {label}; skipping cache write.")

    print("\nCached result files:")
    for label, path in cached.items():
        print(f"  {label}: {path}")

    if args.no_plot or len(cached) < 2:
        if not args.no_plot and len(cached) < 2:
            print(
                "\nOnly one model cached. Run the other model(s) then use:\n"
                "  python scripts/analysis/compare_forecasts.py --from-results "
                + " ".join(str(p) for p in cached.values())
            )
        return

    result_files = [load_result_file(p) for p in cached.values()]
    try:
        validate_compatibility(result_files)
    except ValueError as exc:
        print(f"\nCannot plot: {exc}")
        return

    plot_comparison(
        result_files,
        [str(p) for p in cached.values()],
        args.output_dir,
        args.output_name,
        args.percentiles,
    )


if __name__ == "__main__":
    main()
