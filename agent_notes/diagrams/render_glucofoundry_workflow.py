"""Render the high-level GlucoFoundry research workflow diagram.

General-audience flowchart highlighting the core pipeline and the toolkit's
extensibility. Run this to (re)generate the PNG and SVG next to this script:

    python agent_notes/diagrams/render_glucofoundry_workflow.py

Only dependency is matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- palette -------------------------------------------------------------
IO = dict(fc="#CFF3EE", ec="#2C7A7B", tc="#1D4044")  # data in/out
CORE = dict(fc="#D6E4FF", ec="#2B6CB0", tc="#1A365D")  # pipeline stages
MODEL = dict(fc="#D7F5DD", ec="#2F855A", tc="#1C4532")  # model inputs
EXT = dict(fc="#FFF4D6", ec="#D69E2E", tc="#744210")  # extension points

FIG_W, FIG_H = 16.0, 8.6


def box(ax, cx, cy, w, h, text, style, *, dashed=False, fontsize=12, bold=True):
    boxstyle = "round,pad=0.02,rounding_size=0.14"
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle=boxstyle,
        linewidth=1.8,
        facecolor=style["fc"],
        edgecolor=style["ec"],
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=style["tc"],
        fontweight="bold" if bold else "normal",
        zorder=3,
    )
    return (cx, cy, w, h)


def arrow(ax, p0, p1, *, color="#4A5568", dashed=False, rad=0.0, lw=2.0, label=None):
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(a)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(
            mx,
            my + 0.35,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
            color=color,
            zorder=3,
        )


def top(b):  # top-center of a box
    cx, cy, w, h = b
    return (cx, cy + h / 2)


def bottom(b):
    cx, cy, w, h = b
    return (cx, cy - h / 2)


def left(b):
    cx, cy, w, h = b
    return (cx - w / 2, cy)


def right(b):
    cx, cy, w, h = b
    return (cx + w / 2, cy)


def main() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # title
    ax.text(
        FIG_W / 2,
        8.2,
        "GlucoFoundry — Research Workflow",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#1A202C",
    )
    ax.text(
        FIG_W / 2,
        7.75,
        "A reusable, task-general toolkit for glucose time-series "
        "foundation-model research",
        ha="center",
        va="center",
        fontsize=12,
        color="#4A5568",
    )

    y = 5.4
    bw, bh = 2.3, 1.15
    # core single row: Datasets -> Preprocessing -> Training -> Evaluation -> Results
    datasets = box(ax, 1.6, y, bw, bh, "Datasets\n(CGM / glucose\ntime series)", IO)
    preproc = box(ax, 4.5, y, bw, bh, "Preprocessing &\nFeature\nEngineering", CORE)
    training = box(ax, 7.4, y, bw, bh, "Training /\nFine-tuning", CORE)
    evaluation = box(ax, 10.3, y, bw, bh, "Evaluation &\nMetrics", CORE)
    results = box(ax, 13.3, y, bw, bh, "Experiments &\nReproducible\nResults", IO)

    # core flow arrows
    arrow(ax, right(datasets), left(preproc))
    arrow(ax, right(preproc), left(training))
    arrow(ax, right(training), left(evaluation))
    arrow(ax, right(evaluation), left(results))

    # Models feed training from below
    models = box(
        ax,
        7.4,
        2.9,
        3.0,
        1.05,
        "Models\n(foundation models via the\nModelAdapter protocol)",
        MODEL,
        fontsize=11,
    )
    arrow(ax, top(models), bottom(training), color="#2F855A")

    # Hyperparameter optimization above training, with a feedback loop
    hpo = box(
        ax,
        9.85,
        7.0,
        3.3,
        1.0,
        "Hyperparameter Optimization\n(Optuna) + Tracking (MLflow)",
        CORE,
        fontsize=11,
    )
    arrow(ax, top(training), left(hpo), color="#6B46C1", rad=0.25, label="tune")
    arrow(
        ax, bottom(hpo), top(evaluation), color="#6B46C1", rad=0.15, label="best config"
    )

    # ---- extension points (dashed) -----------------------------------
    ext_style = EXT
    ext_data = box(
        ax,
        1.6,
        2.9,
        2.3,
        0.85,
        "+  Add a\ndataset",
        ext_style,
        dashed=True,
        fontsize=11,
    )
    arrow(ax, top(ext_data), bottom(datasets), color=EXT["ec"], dashed=True)

    ext_model = box(
        ax,
        7.4,
        0.95,
        3.0,
        0.8,
        "+  Add a model (plug in via ModelAdapter)",
        ext_style,
        dashed=True,
        fontsize=10,
    )
    arrow(ax, top(ext_model), bottom(models), color=EXT["ec"], dashed=True)

    ext_metric = box(
        ax,
        10.3,
        2.9,
        2.3,
        0.85,
        "+  Add a\nmetric",
        ext_style,
        dashed=True,
        fontsize=11,
    )
    arrow(ax, top(ext_metric), bottom(evaluation), color=EXT["ec"], dashed=True)

    ext_exp = box(
        ax,
        13.3,
        2.9,
        2.3,
        0.85,
        "+  Add an\nexperiment /\ntask",
        ext_style,
        dashed=True,
        fontsize=10,
    )
    arrow(ax, top(ext_exp), bottom(results), color=EXT["ec"], dashed=True)

    # footer: forks reuse the pipeline
    ax.text(
        FIG_W / 2,
        0.25,
        "Every experiment fork — Nocturnal · Metabonet · Meal-Image · your next "
        "project — reuses this same pipeline.",
        ha="center",
        va="center",
        fontsize=11.5,
        style="italic",
        color="#2D3748",
    )

    # small legend for extension points
    ax.text(
        0.35,
        1.75,
        "dashed = extension point",
        ha="left",
        va="center",
        fontsize=9.5,
        color=EXT["tc"],
        style="italic",
    )

    out = Path(__file__).resolve().parent
    png = out / "glucofoundry_workflow.png"
    svg = out / "glucofoundry_workflow.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    # Strip trailing whitespace so the SVG satisfies the repo pre-commit hook.
    cleaned = "\n".join(line.rstrip() for line in svg.read_text().splitlines())
    svg.write_text(cleaned + "\n")
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
