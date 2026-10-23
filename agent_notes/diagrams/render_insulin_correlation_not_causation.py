"""Illustrative (synthetic) plot: correlation vs causation around insulin.

Shows an insulin bolus followed by a model forecast that predicts a blood-glucose
*spike* -- i.e. the model has learned the correlational pattern (insulin co-occurs
with meals/carbs) rather than the true causal effect (insulin lowers BG). The data
are synthetic and for illustration only.

Run to (re)generate the PNG and SVG next to this script:

    python agent_notes/diagrams/render_insulin_correlation_not_causation.py

Only dependencies are numpy and matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- colors --------------------------------------------------------------
C_HIST = "#1A365D"  # observed history
C_FCST = "#C53030"  # model (correlational) forecast
C_CAUSAL = "#2F855A"  # true causal effect
C_INSULIN = "#6B46C1"  # insulin bolus
C_NOW = "#718096"  # now line


def main() -> None:
    rng = np.random.default_rng(7)
    now = 0
    t = np.arange(-120, 121, 2.5)

    # --- observed history (t <= now): BG drifting up around a meal -------
    hist_smooth = 112.0 + 0.5 * np.clip(t + 45, 0, None)
    bg_hist = hist_smooth + rng.normal(0, 1.8, size=t.shape)
    bg_now = 112.0 + 0.5 * 45  # ~134.5, continuity point at t = now

    # --- model forecast (t >= now): learned correlation -> BG spikes ----
    s_up = 1.0 - np.exp(-np.clip(t, 0, None) / 32.0)
    bg_fcst = bg_now + (206.0 - bg_now) * s_up
    band = 5.0 + 0.18 * np.clip(t, 0, None)

    # --- true causal effect (t >= now): insulin lowers BG ---------------
    s_dn = 1.0 - np.exp(-np.clip(t, 0, None) / 40.0)
    bg_causal = bg_now - (bg_now - 84.0) * s_dn

    hist_mask = t <= now
    fcst_mask = t >= now

    fig, ax = plt.subplots(figsize=(12.5, 6.8))

    # forecast-region shading + now line
    ax.axvspan(now, t.max(), color="#F7FAFC", zorder=0)
    ax.axvline(now, color=C_NOW, ls="--", lw=1.6, zorder=1)
    ax.text(
        now + 3,
        70,
        "forecast  \u2192",
        color=C_NOW,
        fontsize=11,
        style="italic",
        va="center",
    )
    ax.text(
        now - 3,
        70,
        "\u2190  observed",
        color=C_NOW,
        fontsize=11,
        style="italic",
        va="center",
        ha="right",
    )

    # observed history
    ax.plot(
        t[hist_mask],
        bg_hist[hist_mask],
        color=C_HIST,
        lw=2.6,
        label="Observed BG (history)",
        zorder=4,
    )

    # model (correlational) forecast + uncertainty band
    ax.fill_between(
        t[fcst_mask],
        (bg_fcst - band)[fcst_mask],
        (bg_fcst + band)[fcst_mask],
        color=C_FCST,
        alpha=0.13,
        zorder=2,
    )
    ax.plot(
        t[fcst_mask],
        bg_fcst[fcst_mask],
        color=C_FCST,
        lw=2.8,
        ls="--",
        label="Model forecast (learned correlation)",
        zorder=4,
    )

    # true causal effect
    ax.plot(
        t[fcst_mask],
        bg_causal[fcst_mask],
        color=C_CAUSAL,
        lw=2.6,
        ls=":",
        label="True causal effect of insulin (BG \u2193)",
        zorder=4,
    )

    # insulin bolus marker
    t_bolus = -12.0
    bg_at_bolus = 112.0 + 0.5 * (t_bolus + 45)
    ax.annotate(
        "Insulin bolus\n(given with meal carbs)",
        xy=(t_bolus, bg_at_bolus),
        xytext=(t_bolus - 46, 182),
        fontsize=11,
        color=C_INSULIN,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=C_INSULIN, lw=2.0),
        zorder=5,
    )
    ax.plot(
        [t_bolus], [bg_at_bolus], marker="v", color=C_INSULIN, markersize=11, zorder=6
    )

    # callouts
    ax.annotate(
        "Model predicts a BG spike after insulin\n"
        "(insulin co-occurs with meals \u2192 confounded correlation)",
        xy=(70, np.interp(70, t, bg_fcst)),
        xytext=(18, 150),
        fontsize=10.5,
        color=C_FCST,
        arrowprops=dict(arrowstyle="-|>", color=C_FCST, lw=1.6),
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF5F5", ec=C_FCST, lw=1.2),
        zorder=6,
    )
    ax.annotate(
        "Physiology: insulin actually\nlowers blood glucose",
        xy=(78, np.interp(78, t, bg_causal)),
        xytext=(40, 58),
        fontsize=10.5,
        color=C_CAUSAL,
        arrowprops=dict(arrowstyle="-|>", color=C_CAUSAL, lw=1.6),
        bbox=dict(boxstyle="round,pad=0.35", fc="#F0FFF4", ec=C_CAUSAL, lw=1.2),
        zorder=6,
    )

    # cosmetics
    ax.set_title(
        "Correlation \u2260 causation: the model reads an insulin dose as a "
        "BG-spike signal",
        fontsize=15,
        fontweight="bold",
        color="#1A202C",
        pad=14,
    )
    ax.set_xlabel("Time relative to forecast origin (minutes)", fontsize=12)
    ax.set_ylabel("Blood glucose (mg/dL)", fontsize=12)
    ax.set_ylim(50, 220)
    ax.set_xlim(t.min(), t.max())
    ax.grid(True, ls=":", lw=0.7, color="#CBD5E0", alpha=0.7)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)
    ax.text(
        0.5,
        -0.17,
        "Synthetic illustration. The confounder (a meal eaten with the bolus) "
        "drives the BG rise; a correlational model misattributes it to insulin.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        style="italic",
        color="#4A5568",
    )

    fig.tight_layout()
    out = Path(__file__).resolve().parent
    png = out / "insulin_correlation_not_causation.png"
    svg = out / "insulin_correlation_not_causation.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    # Strip trailing whitespace so the SVG satisfies the repo pre-commit hook.
    cleaned = "\n".join(line.rstrip() for line in svg.read_text().splitlines())
    svg.write_text(cleaned + "\n")
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
