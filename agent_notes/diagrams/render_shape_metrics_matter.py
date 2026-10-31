"""Illustrative (synthetic) plot: why shape metrics matter.

A true nocturnal glucose curve with a hypo dip is compared against two forecasts:

  - Forecast A -- a flat / mean prediction that scores LOW pointwise error
    (RMSE / MAE) yet completely misses the hypo event.
  - Forecast B -- the right shape but shifted ~20 min in time; it scores WORSE
    on pointwise error (misalignment is penalized point-by-point) but BETTER on
    a shape metric (DTW), and it is the clinically useful forecast because it
    actually predicts the hypo.

The metric values printed on the figure are computed from the synthetic data
(RMSE / MAE directly; DTW via a small in-script dynamic-programming routine), so
the ranking shown is real, not hand-typed. Data are synthetic and for
illustration only.

Run to (re)generate the PNG and SVG next to this script:

    python agent_notes/diagrams/render_shape_metrics_matter.py

Only dependencies are numpy and matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_TRUTH = "#1A365D"
C_FLAT = "#C53030"
C_SHIFT = "#2B6CB0"
C_HYPO = "#9B2C2C"

HYPO_THRESHOLD = 3.9  # mmol/L, level-1 hypoglycemia


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic DTW with L1 local cost (lower = better shape match)."""
    n, m = len(a), len(b)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            acc[i, j] = cost + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    t = np.arange(0, 121, 5.0)  # 2 h forecast horizon, 5-min CGM cadence

    baseline, depth, t0, sigma = 6.6, 3.5, 50.0, 12.0
    truth = baseline - depth * np.exp(-((t - t0) ** 2) / (2 * sigma**2))

    # Forecast A: flat / mean (minimises squared error among constants)
    flat = np.full_like(t, truth.mean())

    # Forecast B: right shape, shifted ~25 min late
    shift = 25.0
    shifted = baseline - depth * np.exp(-((t - (t0 + shift)) ** 2) / (2 * sigma**2))

    m_flat = dict(
        rmse=rmse(flat, truth), mae=mae(flat, truth), dtw=dtw_distance(flat, truth)
    )
    m_shift = dict(
        rmse=rmse(shifted, truth),
        mae=mae(shifted, truth),
        dtw=dtw_distance(shifted, truth),
    )
    print("flat   :", {k: round(v, 2) for k, v in m_flat.items()})
    print("shifted:", {k: round(v, 2) for k, v in m_shift.items()})

    fig, ax = plt.subplots(figsize=(12.5, 7.0))

    # hypo region + threshold
    ax.axhspan(2.5, HYPO_THRESHOLD, color="#FFF5F5", zorder=0)
    ax.axhline(HYPO_THRESHOLD, color=C_HYPO, ls=":", lw=1.6, zorder=1)
    ax.text(
        2,
        HYPO_THRESHOLD + 0.08,
        "hypoglycemia threshold (3.9 mmol/L)",
        color=C_HYPO,
        fontsize=10,
        va="bottom",
    )

    ax.plot(
        t,
        truth,
        color=C_TRUTH,
        lw=3.0,
        marker="o",
        ms=4,
        label="Actual BG (true hypo event)",
        zorder=5,
    )
    ax.plot(
        t,
        flat,
        color=C_FLAT,
        lw=2.6,
        ls="--",
        label="Forecast A: flat / mean (misses the hypo)",
        zorder=4,
    )
    ax.plot(
        t,
        shifted,
        color=C_SHIFT,
        lw=2.6,
        ls="-.",
        label="Forecast B: right shape, ~25 min late",
        zorder=4,
    )

    # mark the true hypo nadir
    i_min = int(np.argmin(truth))
    ax.annotate(
        "true hypo nadir",
        xy=(t[i_min], truth[i_min]),
        xytext=(t[i_min] - 33, truth[i_min] - 0.5),
        fontsize=10.5,
        color=C_TRUTH,
        arrowprops=dict(arrowstyle="-|>", color=C_TRUTH, lw=1.5),
    )

    # metrics table (monospace; equal-width rows so columns align when right-set)
    _rows = [
        f"{'metric (lower = better)':<22}{'A: flat':>9}{'B: shift':>10}",
        "",  # separator filled after width is known
        f"{'RMSE (pointwise)':<22}{m_flat['rmse']:>9.2f}{m_shift['rmse']:>10.2f}",
        f"{'MAE  (pointwise)':<22}{m_flat['mae']:>9.2f}{m_shift['mae']:>10.2f}",
        f"{'DTW  (shape)':<22}{m_flat['dtw']:>9.2f}{m_shift['dtw']:>10.2f}",
        "",
        "pointwise (RMSE/MAE) -> picks A",
        "shape (DTW)          -> picks B",
    ]
    width = max(len(r) for r in _rows)
    _rows[1] = "-" * width
    rows = [r.ljust(width) for r in _rows]
    ax.text(
        0.985,
        0.96,
        "\n".join(rows),
        transform=ax.transAxes,
        ha="right",
        va="top",
        family="monospace",
        fontsize=10.0,
        color="#1A202C",
        bbox=dict(boxstyle="round,pad=0.5", fc="#F7FAFC", ec="#A0AEC0", lw=1.2),
        zorder=6,
    )

    ax.set_title(
        "Why shape metrics matter: pointwise error rewards a forecast that "
        "misses the hypo",
        fontsize=14.5,
        fontweight="bold",
        color="#1A202C",
        pad=12,
    )
    ax.set_xlabel("Forecast horizon (minutes)", fontsize=12)
    ax.set_ylabel("Blood glucose (mmol/L)", fontsize=12)
    ax.set_ylim(2.5, 7.6)
    ax.set_xlim(t.min(), t.max())
    ax.grid(True, ls=":", lw=0.7, color="#CBD5E0", alpha=0.7)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)
    ax.text(
        0.5,
        -0.14,
        "Synthetic illustration. RMSE/MAE rank the flat line as 'better' though "
        "it never predicts the hypo; DTW (and DILATE's shape term) correctly "
        "reward the shape-faithful forecast.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        style="italic",
        color="#4A5568",
    )

    fig.tight_layout()
    out = Path(__file__).resolve().parent
    png = out / "shape_metrics_matter.png"
    svg = out / "shape_metrics_matter.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    # Strip trailing whitespace so the SVG satisfies the repo pre-commit hook.
    cleaned = "\n".join(line.rstrip() for line in svg.read_text().splitlines())
    svg.write_text(cleaned + "\n")
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
