#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Model-agnostic episode forecast overlay plots from a JSON payload.

What this visualization tells you
---------------------------------
- Episode-level forecast behavior: tracking quality, bias, and interval width.
- Whether prediction intervals cover the realized trace near risky transitions
  (for example, nocturnal dips near the hypo threshold).

What to look for
----------------
- Mean forecast consistently above/below target (systematic bias).
- Intervals too narrow with frequent target escapes (under-dispersion).
- Intervals excessively wide with little shape fidelity (over-dispersion).

Required data format
--------------------
Input file passed via `--episodes-json` must be either:

1) Preferred format
{
  "episodes": [
    {
      "episode_id": "patient_x::ep012",
      "target_bg": [ ... ],
      "pred_mean": [ ... ],
      "pred_interval_low": [ ... ],
      "pred_interval_high": [ ... ],

      # Optional:
      "context_bg": [ ... ],
      "anchor": "2026-01-01 23:55:00",
      "score": 2.31,
      "context_aux": { "iob": [ ... ] },
      "forecast_aux": { "iob": [ ... ] }
    }
  ]
}

2) Legacy compatibility format
{
  "id_a": {
    "episode_id": "...",
    "target_bg": [ ... ],
    "mean": [ ... ],
    "q10": [ ... ],
    "q90": [ ... ],
    "context_bg": [ ... ],
    "anchor": "...",
    "rmse": 2.31,
    "context_iob": [ ... ],
    "forecast_iob": [ ... ]
  },
  "id_b": { ... }
}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_OUTPUT_DIR = "results/figures/forecast_episode_overlays"
DEFAULT_COMBINED_NAME = "forecast_episodes_combined.png"
DEFAULT_STEP_MINUTES = 5


@dataclass(frozen=True)
class EpisodeData:
    episode_id: str
    target_bg: np.ndarray
    pred_mean: np.ndarray
    pred_interval_low: np.ndarray
    pred_interval_high: np.ndarray
    context_bg: np.ndarray | None
    anchor: str
    score: float | None
    context_aux: dict[str, np.ndarray]
    forecast_aux: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--episodes-json",
        required=True,
        help="JSON file containing episode forecast payload.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write plots (default: %(default)s).",
    )
    parser.add_argument(
        "--combined-name",
        default=DEFAULT_COMBINED_NAME,
        help="Filename for combined stacked plot (default: %(default)s).",
    )
    parser.add_argument(
        "--context-hours",
        type=float,
        default=2.0,
        help="Hours of context to show before forecast start (default: %(default)s).",
    )
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=DEFAULT_STEP_MINUTES,
        help="Minutes per timestep for x-axis conversion (default: %(default)s).",
    )
    parser.add_argument(
        "--interval-label",
        default="Prediction interval",
        help="Legend label for interval shading.",
    )
    parser.add_argument(
        "--hypo-threshold",
        type=float,
        default=3.9,
        help="Optional horizontal threshold line value.",
    )
    parser.add_argument(
        "--aux-series",
        default=None,
        help="Optional auxiliary series key to plot on secondary axis (e.g. 'iob').",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Only write combined plot; skip per-episode files.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    return parser.parse_args()


def _parse_float_array(
    payload: dict, key: str, required: bool = True
) -> np.ndarray | None:
    value = payload.get(key)
    if value is None:
        if required:
            raise ValueError(f"Missing required key '{key}' in episode payload")
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"Episode key '{key}' must be a 1D array, got shape {arr.shape}"
        )
    return arr


def _parse_aux_map(value: object) -> dict[str, np.ndarray]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            "Auxiliary map must be an object mapping series names to arrays"
        )
    out: dict[str, np.ndarray] = {}
    for key, arr_value in value.items():
        arr = np.asarray(arr_value, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Aux series '{key}' must be 1D, got shape {arr.shape}")
        out[str(key)] = arr
    return out


def _episode_from_payload(payload: dict) -> EpisodeData:
    episode_id = str(payload.get("episode_id") or payload.get("id") or "episode")
    target_bg = _parse_float_array(payload, "target_bg", required=True)
    pred_mean = _parse_float_array(payload, "pred_mean", required=True)
    pred_interval_low = _parse_float_array(payload, "pred_interval_low", required=True)
    pred_interval_high = _parse_float_array(
        payload, "pred_interval_high", required=True
    )
    context_bg = _parse_float_array(payload, "context_bg", required=False)

    if (
        target_bg is None
        or pred_mean is None
        or pred_interval_low is None
        or pred_interval_high is None
    ):
        raise ValueError(f"Episode '{episode_id}' missing required forecast arrays")

    n_forecast = len(pred_mean)
    for key, arr in (
        ("target_bg", target_bg),
        ("pred_interval_low", pred_interval_low),
        ("pred_interval_high", pred_interval_high),
    ):
        if len(arr) != n_forecast:
            raise ValueError(
                f"Episode '{episode_id}' key '{key}' length {len(arr)} does not match "
                f"pred_mean length {n_forecast}"
            )

    if np.any(pred_interval_low > pred_interval_high):
        raise ValueError(f"Episode '{episode_id}' has interval_low > interval_high")

    score_raw = payload.get("score", payload.get("rmse"))
    score = float(score_raw) if score_raw is not None else None
    anchor = str(payload.get("anchor", ""))

    context_aux = _parse_aux_map(payload.get("context_aux"))
    forecast_aux = _parse_aux_map(payload.get("forecast_aux"))
    return EpisodeData(
        episode_id=episode_id,
        target_bg=target_bg,
        pred_mean=pred_mean,
        pred_interval_low=pred_interval_low,
        pred_interval_high=pred_interval_high,
        context_bg=context_bg,
        anchor=anchor,
        score=score,
        context_aux=context_aux,
        forecast_aux=forecast_aux,
    )


def _load_episodes(json_path: Path) -> list[EpisodeData]:
    with json_path.open() as file_obj:
        payload = json.load(file_obj)

    episodes_payload: list[dict]
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        episodes_payload = payload["episodes"]
    elif isinstance(payload, dict):
        episodes_payload = []
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            if {"target_bg", "mean", "q10", "q90"}.issubset(value.keys()):
                episodes_payload.append(
                    {
                        "episode_id": value.get("episode_id", key),
                        "anchor": value.get("anchor", ""),
                        "target_bg": value["target_bg"],
                        "pred_mean": value["mean"],
                        "pred_interval_low": value["q10"],
                        "pred_interval_high": value["q90"],
                        "context_bg": value.get("context_bg"),
                        "score": value.get("rmse"),
                        "context_aux": {"iob": value["context_iob"]}
                        if value.get("context_iob") is not None
                        else {},
                        "forecast_aux": {"iob": value["forecast_iob"]}
                        if value.get("forecast_iob") is not None
                        else {},
                    }
                )
    else:
        raise ValueError(
            "Unsupported payload format. Expected {'episodes': [...]} or legacy dict payload."
        )

    episodes = [_episode_from_payload(ep) for ep in episodes_payload]
    if not episodes:
        raise ValueError(f"{json_path} did not contain any parseable episodes")
    return episodes


def _time_axis(length: int, step_minutes: int, offset_hours: float = 0.0) -> np.ndarray:
    return np.arange(length) * step_minutes / 60.0 + offset_hours


def _plot_single_episode(
    episode: EpisodeData,
    output_path: Path,
    context_hours: float,
    step_minutes: int,
    interval_label: str,
    threshold: float,
    aux_series: str | None,
    dpi: int,
) -> None:
    fig, ax_bg = plt.subplots(figsize=(6, 5))

    if episode.context_bg is not None and len(episode.context_bg) > 0:
        ctx_show = int(context_hours * 60 / step_minutes)
        ctx_show = min(ctx_show, len(episode.context_bg))
        context_bg = episode.context_bg[-ctx_show:]
        t_ctx = _time_axis(ctx_show, step_minutes, offset_hours=-context_hours)
        ax_bg.plot(
            t_ctx,
            context_bg,
            color="black",
            linewidth=2.0,
            label="Actual BG (context)",
            zorder=10,
        )

    t_pred = _time_axis(len(episode.pred_mean), step_minutes, offset_hours=0.0)
    t_target = _time_axis(len(episode.target_bg), step_minutes, offset_hours=0.0)
    ax_bg.plot(
        t_target,
        episode.target_bg,
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="Actual BG (target)",
        zorder=10,
    )
    ax_bg.plot(
        t_pred,
        episode.pred_mean,
        color="steelblue",
        linewidth=2.0,
        label="Forecast mean",
        zorder=8,
    )
    ax_bg.fill_between(
        t_pred,
        episode.pred_interval_low,
        episode.pred_interval_high,
        color="steelblue",
        alpha=0.15,
        label=interval_label,
        zorder=3,
    )
    ax_bg.axhline(
        y=threshold,
        color="red",
        linestyle=":",
        linewidth=1.2,
        alpha=0.6,
        label=f"Threshold ({threshold:g})",
    )
    ax_bg.axvline(x=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    if aux_series:
        context_aux = episode.context_aux.get(aux_series)
        forecast_aux = episode.forecast_aux.get(aux_series)
        if forecast_aux is not None:
            ax_aux = ax_bg.twinx()
            if (
                context_aux is not None
                and len(context_aux) > 0
                and episode.context_bg is not None
            ):
                ctx_show = int(context_hours * 60 / step_minutes)
                ctx_show = min(ctx_show, len(context_aux))
                t_ctx = _time_axis(ctx_show, step_minutes, offset_hours=-context_hours)
                ax_aux.plot(
                    t_ctx,
                    context_aux[-ctx_show:],
                    color="gray",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.45,
                )
            ax_aux.plot(
                t_pred[: len(forecast_aux)],
                forecast_aux,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
                label=aux_series,
            )
            ax_aux.set_ylabel(aux_series, fontsize=10, color="gray")
            ax_aux.tick_params(axis="y", labelcolor="gray", labelsize=8)
            max_aux = float(np.nanmax(forecast_aux)) if len(forecast_aux) else 1.0
            ax_aux.set_ylim(0, max(max_aux * 1.6, 0.5))

    score_text = f"  |  score: {episode.score:.3f}" if episode.score is not None else ""
    anchor_text = f"\nAnchor: {episode.anchor}" if episode.anchor else ""
    ax_bg.set_title(
        f"{episode.episode_id}{anchor_text}{score_text}",
        fontsize=12,
        fontweight="bold",
    )
    ax_bg.set_xlabel("Hours relative to forecast start", fontsize=11)
    ax_bg.set_ylabel("Blood Glucose (mmol/L)", fontsize=11)
    ax_bg.grid(True, alpha=0.3)
    ax_bg.tick_params(axis="both", labelsize=9)

    handles, labels = ax_bg.get_legend_handles_labels()
    ax_bg.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_combined(
    episodes: list[EpisodeData],
    output_path: Path,
    context_hours: float,
    step_minutes: int,
    interval_label: str,
    threshold: float,
    aux_series: str | None,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        len(episodes), 1, figsize=(7, 4.2 * len(episodes)), sharey=True
    )
    if len(episodes) == 1:
        axes = [axes]

    for idx, episode in enumerate(episodes):
        ax_bg: plt.Axes = axes[idx]

        if episode.context_bg is not None and len(episode.context_bg) > 0:
            ctx_show = int(context_hours * 60 / step_minutes)
            ctx_show = min(ctx_show, len(episode.context_bg))
            t_ctx = _time_axis(ctx_show, step_minutes, offset_hours=-context_hours)
            ax_bg.plot(
                t_ctx,
                episode.context_bg[-ctx_show:],
                color="black",
                linewidth=2.0,
                label="Actual BG (context)",
                zorder=10,
            )

        t_pred = _time_axis(len(episode.pred_mean), step_minutes, offset_hours=0.0)
        t_target = _time_axis(len(episode.target_bg), step_minutes, offset_hours=0.0)
        ax_bg.plot(
            t_target,
            episode.target_bg,
            "k--",
            linewidth=2.0,
            label="Actual BG (target)",
            zorder=10,
        )
        ax_bg.plot(
            t_pred,
            episode.pred_mean,
            color="steelblue",
            linewidth=2.0,
            label="Forecast mean",
            zorder=8,
        )
        ax_bg.fill_between(
            t_pred,
            episode.pred_interval_low,
            episode.pred_interval_high,
            color="steelblue",
            alpha=0.15,
            label=interval_label,
            zorder=3,
        )
        ax_bg.axhline(y=threshold, color="red", linestyle=":", linewidth=1.2, alpha=0.6)
        ax_bg.axvline(x=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.4)

        if aux_series:
            forecast_aux = episode.forecast_aux.get(aux_series)
            context_aux = episode.context_aux.get(aux_series)
            if forecast_aux is not None:
                ax_aux = ax_bg.twinx()
                if context_aux is not None and episode.context_bg is not None:
                    ctx_show = int(context_hours * 60 / step_minutes)
                    ctx_show = min(ctx_show, len(context_aux))
                    t_ctx = _time_axis(
                        ctx_show, step_minutes, offset_hours=-context_hours
                    )
                    ax_aux.plot(
                        t_ctx,
                        context_aux[-ctx_show:],
                        color="gray",
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.4,
                    )
                ax_aux.plot(
                    t_pred[: len(forecast_aux)],
                    forecast_aux,
                    color="gray",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.4,
                    label=aux_series,
                )
                ax_aux.set_ylabel(aux_series, fontsize=9, color="gray")
                ax_aux.tick_params(axis="y", labelcolor="gray", labelsize=7)

        score_text = (
            f"  |  score: {episode.score:.3f}" if episode.score is not None else ""
        )
        ax_bg.set_title(
            f"{episode.episode_id}{score_text}", fontsize=11, fontweight="bold"
        )
        ax_bg.set_xlabel("Hours relative to forecast start", fontsize=10)
        ax_bg.set_ylabel("Blood Glucose (mmol/L)", fontsize=10)
        ax_bg.grid(True, alpha=0.3)
        ax_bg.tick_params(axis="both", labelsize=8)

        if idx == 0:
            handles, labels = ax_bg.get_legend_handles_labels()
            ax_bg.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.07, hspace=0.35)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.context_hours < 0:
        raise ValueError("--context-hours must be >= 0")
    if args.step_minutes <= 0:
        raise ValueError("--step-minutes must be > 0")

    episodes = _load_episodes(Path(args.episodes_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_individual:
        for episode in episodes:
            safe_episode = episode.episode_id.replace("::", "_").replace("/", "_")
            output_path = output_dir / f"forecast_episode_{safe_episode}.png"
            _plot_single_episode(
                episode=episode,
                output_path=output_path,
                context_hours=args.context_hours,
                step_minutes=args.step_minutes,
                interval_label=args.interval_label,
                threshold=args.hypo_threshold,
                aux_series=args.aux_series,
                dpi=args.dpi,
            )
            print(f"Saved: {output_path}")

    combined_path = output_dir / args.combined_name
    _plot_combined(
        episodes=episodes,
        output_path=combined_path,
        context_hours=args.context_hours,
        step_minutes=args.step_minutes,
        interval_label=args.interval_label,
        threshold=args.hypo_threshold,
        aux_series=args.aux_series,
        dpi=args.dpi,
    )
    print(f"Saved: {combined_path}")


if __name__ == "__main__":
    main()
