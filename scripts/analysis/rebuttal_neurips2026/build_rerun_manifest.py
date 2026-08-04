# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""Build the Workstream-2 GPU rerun MANIFEST (does NOT launch anything).

Fills the two per-episode-array gaps needed for CIs on conditions whose raw runs
were lost in the cleanup incident:
  - A4 bg-only-for-all fairness leaderboard: models whose surviving best run used
    covariates (chronos2/moirai/tft iob/iob_cob; ttm iob; toto zeroshot) need a
    fine-tuned BG-ONLY eval.  Checkpoint is BORROWED from the same model's
    existing bg-only eval run (the exact checkpoint behind the paper's G cell;
    these checkpoints are shared across datasets -- verified: TTM used one
    bg-only ckpt for both aleppo & tamborlane).
  - Zero-shot CIs: foundation models with no surviving zeroshot run
    (chronos2/moirai/moment/timesfm/ttm + toto on 3 datasets). No checkpoint.

All models EXCEPT TimeGrad (diffusion, too slow) per user's instruction.

SAFETY: reruns write to NEW, explicitly-named output dirs
(experiments/.../<model>/rebuttal_<dataset>_<mode>); they never overwrite the
surviving runs. Each job carries the PUBLISHED RMSE (Table 3 "Ov") as a
post-run validation target so we can confirm the rerun reproduces the paper.

Outputs (scripts/analysis/rebuttal_neurips2026/outputs/reruns/):
    manifest.csv        one row per job (model,dataset,kind,condition,mode,
                        checkpoint,venv,cuda_device,expected_rmse,output_dir,command)
    run_gpu0.sh / run_gpu1.sh   tmux-friendly sequential runners (one per GPU)
    launch_all.sh       convenience: starts both tmux sessions

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.build_rerun_manifest
    # then review manifest.csv; nothing runs until you execute launch_all.sh
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pandas as pd

from scripts.analysis.rebuttal_neurips2026 import common as C

OUT = Path(__file__).resolve().parent / "outputs" / "reruns"
REPO = C.REPO
EVAL = "scripts/experiments/nocturnal_hypo_eval.py"
CTX, FH = 512, 96

# Models the CURRENT branch's nocturnal_hypo_eval.py can run (its --model choices).
# tft/patchtst/deepar are NOT wired into this branch's factory/CLI (they were run
# on metabonet-integration); their jobs are listed but excluded from the runners.
SUPPORTED_MODELS = {
    "sundial",
    "ttm",
    "chronos2",
    "moirai",
    "moment",
    "timegrad",
    "timesfm",
    "tide",
    "toto",
    "tft",
}

# tft is AutoGluon-based (restored from feat/autogluon-baselines); it runs in the
# chronos2 venv (autogluon.timeseries 1.5.0), not a dedicated .venvs/tft.
VENV_OVERRIDE = {"tft": "chronos2"}

# Point-only models: no quantile output, so --probabilistic raises ValueError.
# (Matches the paper: TTM/MOMENT have no WQL/calibration rows.)
POINT_ONLY = {"ttm", "moment", "timegrad"}

# Foundation models that have a zero-shot row in the paper (Table 3).
ZS_MODELS = ["chronos2", "moirai", "moment", "timesfm", "ttm", "toto"]  # sundial done
# Models that must appear in the bg-only fairness leaderboard (A4). DL models are
# already bg-only; gaps are only the covariate-best foundation/attention models.
BGONLY_MODELS = ["chronos2", "moirai", "tft", "ttm", "toto", "moment", "timesfm"]

# Published RMSE ("Ov" pooled column, Table 3) for post-run validation.
PAPER_RMSE = {
    # bg_only (G)
    ("chronos2", "aleppo_2017", "bg_only"): 2.751,
    ("chronos2", "brown_2019", "bg_only"): 2.522,
    ("chronos2", "lynch_2022", "bg_only"): 2.429,
    ("chronos2", "tamborlane_2008", "bg_only"): 3.270,
    ("moirai", "aleppo_2017", "bg_only"): 2.780,
    ("moirai", "brown_2019", "bg_only"): 2.577,
    ("moirai", "lynch_2022", "bg_only"): 2.519,
    ("moirai", "tamborlane_2008", "bg_only"): 3.341,
    ("tft", "aleppo_2017", "bg_only"): 2.740,
    ("tft", "brown_2019", "bg_only"): 2.507,
    ("tft", "lynch_2022", "bg_only"): 2.435,
    ("tft", "tamborlane_2008", "bg_only"): 3.316,
    ("ttm", "aleppo_2017", "bg_only"): 2.769,
    ("ttm", "brown_2019", "bg_only"): 2.566,
    ("ttm", "lynch_2022", "bg_only"): 2.520,
    ("ttm", "tamborlane_2008", "bg_only"): 3.258,
    ("toto", "aleppo_2017", "bg_only"): 2.928,
    ("toto", "brown_2019", "bg_only"): 2.676,
    ("toto", "lynch_2022", "bg_only"): 2.570,
    ("toto", "tamborlane_2008", "bg_only"): 3.340,
    ("moment", "aleppo_2017", "bg_only"): 3.315,
    ("moment", "brown_2019", "bg_only"): 3.021,
    ("moment", "lynch_2022", "bg_only"): 2.920,
    ("moment", "tamborlane_2008", "bg_only"): 3.507,
    ("timesfm", "aleppo_2017", "bg_only"): 2.709,
    ("timesfm", "brown_2019", "bg_only"): 2.488,
    ("timesfm", "lynch_2022", "bg_only"): 2.423,
    ("timesfm", "tamborlane_2008", "bg_only"): 3.260,
    # zero_shot (ZS)
    ("chronos2", "aleppo_2017", "zeroshot"): 2.765,
    ("chronos2", "brown_2019", "zeroshot"): 2.650,
    ("chronos2", "lynch_2022", "zeroshot"): 2.672,
    ("chronos2", "tamborlane_2008", "zeroshot"): 3.354,
    ("moirai", "aleppo_2017", "zeroshot"): 3.143,
    ("moirai", "brown_2019", "zeroshot"): 3.158,
    ("moirai", "lynch_2022", "zeroshot"): 3.307,
    ("moirai", "tamborlane_2008", "zeroshot"): 3.664,
    ("moment", "aleppo_2017", "zeroshot"): 3.771,
    ("moment", "brown_2019", "zeroshot"): 3.556,
    ("moment", "lynch_2022", "zeroshot"): 3.483,
    ("moment", "tamborlane_2008", "zeroshot"): 3.955,
    ("timesfm", "aleppo_2017", "zeroshot"): 2.767,
    ("timesfm", "brown_2019", "zeroshot"): 2.661,
    ("timesfm", "lynch_2022", "zeroshot"): 2.666,
    ("timesfm", "tamborlane_2008", "zeroshot"): 3.354,
    ("ttm", "aleppo_2017", "zeroshot"): 2.921,
    ("ttm", "brown_2019", "zeroshot"): 2.843,
    ("ttm", "lynch_2022", "zeroshot"): 3.001,
    ("ttm", "tamborlane_2008", "zeroshot"): 3.424,
    ("toto", "aleppo_2017", "zeroshot"): 2.866,
    ("toto", "brown_2019", "zeroshot"): 2.907,
    ("toto", "lynch_2022", "zeroshot"): 2.970,
    ("toto", "tamborlane_2008", "zeroshot"): 3.473,
}


def _venv(model: str) -> str:
    if model in VENV_OVERRIDE:
        return f".venvs/{VENV_OVERRIDE[model]}"
    p = REPO / ".venvs" / model
    return f".venvs/{model}" if p.exists() else ".noctprob-venv"


def _present(
    df: pd.DataFrame, model: str, dataset: str, *, condition=None, mode=None
) -> bool:
    m = (df.model == model) & (df.dataset == dataset)
    if condition is not None:
        m &= df.condition == condition
    if mode is not None:
        m &= df["mode"] == mode
    return bool(m.any())


def _bgonly_checkpoint(model: str) -> str | None:
    """Borrow the model's bg-only checkpoint from an existing FINE-TUNED bg-only
    eval run (zero-shot runs have no checkpoint)."""
    rp = None
    for ds in ("aleppo_2017", "tamborlane_2008", "brown_2019", "lynch_2022"):
        rp = C.run_path_for_condition(model, ds, "bg_only", mode="finetuned")
        if rp is not None:
            break
    if rp is None:
        return None
    try:
        cli = json.loads((rp / "experiment_config.json").read_text()).get(
            "cli_args", {}
        )
        return cli.get("checkpoint")
    except Exception:
        return None


def build() -> pd.DataFrame:
    df = C.scan_runs()
    jobs = []

    # ---- Zero-shot gaps (no checkpoint) ----
    for model in ZS_MODELS:
        for ds in C.DATASETS:
            if _present(df, model, ds, mode="zeroshot"):
                continue
            jobs.append(
                dict(
                    kind="zeroshot_ci",
                    model=model,
                    dataset=ds,
                    condition="bg_only",
                    mode="zeroshot",
                    checkpoint=None,
                    expected_rmse=PAPER_RMSE.get((model, ds, "zeroshot")),
                )
            )

    # ---- bg-only fine-tuned gaps (borrowed checkpoint) ----
    for model in BGONLY_MODELS:
        ck = _bgonly_checkpoint(model)
        for ds in C.DATASETS:
            if _present(df, model, ds, condition="bg_only", mode="finetuned"):
                continue
            jobs.append(
                dict(
                    kind="a4_bgonly",
                    model=model,
                    dataset=ds,
                    condition="bg_only",
                    mode="finetuned",
                    checkpoint=ck,
                    expected_rmse=PAPER_RMSE.get((model, ds, "bg_only")),
                )
            )

    man = pd.DataFrame(jobs)
    if man.empty:
        return man
    # assign venv, gpu (round-robin), output dir, command
    man["venv"] = man["model"].map(_venv)
    man["supported"] = man["model"].isin(SUPPORTED_MODELS)
    man["cuda_device"] = [i % 2 for i in range(len(man))]
    man["missing_checkpoint"] = man.apply(
        lambda r: bool(r["mode"] == "finetuned" and not r["checkpoint"]), axis=1
    )
    man["output_dir"] = man.apply(
        lambda r: f"experiments/nocturnal_forecasting/{CTX}ctx_{FH}fh/"
        f"{r.model}/rebuttal_{r.dataset}_{r['mode']}",
        axis=1,
    )
    man["command"] = man.apply(_command, axis=1)
    man.insert(
        0, "job_id", [f"{r.kind[:2]}_{r.model}_{r.dataset}" for r in man.itertuples()]
    )
    return man


def _command(r: pd.Series) -> str:
    parts = [
        f"{r['venv']}/bin/python",
        EVAL,
        "--model",
        r["model"],
        "--dataset",
        r["dataset"],
        "--config-dir",
        "configs/data/holdout_10pct",
        "--context-length",
        str(CTX),
        "--forecast-length",
        str(FH),
        "--cuda-device",
        str(r["cuda_device"]),
        "--output-dir",
        r["output_dir"],
    ]
    if r["model"] not in POINT_ONLY:
        parts += ["--probabilistic"]  # point-only models reject this flag
    if r["mode"] == "finetuned" and r["checkpoint"]:
        parts += ["--checkpoint", r["checkpoint"]]
    # Force BG-ONLY. The eval only falls back to the model config's default
    # covariate_cols when args.covariate_cols IS None; an EMPTY --covariate-cols
    # gives [] (not None), overriding defaults like chronos2's ['iob']. Placed
    # LAST so nargs="*" consumes nothing at end-of-args -> [].
    parts += ["--covariate-cols"]
    return " ".join(parts)


def write(man: pd.DataFrame, tag: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    man.to_csv(OUT / f"manifest{sfx}.csv", index=False)

    for gpu in (0, 1):
        sub = man[(man.cuda_device == gpu) & man.supported]
        lines = [
            "#!/usr/bin/env bash",
            "# Auto-generated rerun runner (GPU %d). Review manifest.csv first." % gpu,
            "set -uo pipefail",
            f'cd "{REPO}"',
            "",
        ]
        for r in sub.itertuples():
            log = f"scripts/analysis/rebuttal_neurips2026/outputs/reruns/logs/{r.job_id}.log"
            skip = (
                "  # !!! MISSING CHECKPOINT -- resolve before running"
                if r.missing_checkpoint
                else ""
            )
            lines += [
                f'echo "[GPU{gpu}] {r.job_id} (expect RMSE~{r.expected_rmse})"{skip}',
                f"{r.command} > {log} 2>&1",
                f'echo "  -> exit $? : {r.job_id}"',
                "",
            ]
        lines.append('echo "GPU%d DONE"' % gpu)
        p = OUT / f"run_gpu{gpu}{sfx}.sh"
        p.write_text("\n".join(lines))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)

    launch = OUT / f"launch_all{sfx}.sh"
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "# Launch both GPU runners in detached tmux sessions (disconnect-safe).",
                f'cd "{REPO}"',
                f"tmux new-session -d -s rerun{sfx}_gpu0 'bash scripts/analysis/rebuttal_neurips2026/outputs/reruns/run_gpu0{sfx}.sh; exec bash'",
                f"tmux new-session -d -s rerun{sfx}_gpu1 'bash scripts/analysis/rebuttal_neurips2026/outputs/reruns/run_gpu1{sfx}.sh; exec bash'",
                "tmux ls",
            ]
        )
    )
    launch.chmod(launch.stat().st_mode | stat.S_IEXEC)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build GPU rerun manifest (launches nothing)"
    )
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="restrict runners to these models (e.g. --only ttm moment)",
    )
    args = ap.parse_args()
    man = build()
    if man.empty:
        print("No gap jobs — all conditions already present on disk.")
        return
    if args.only:
        man = man[man.model.isin(args.only)].reset_index(drop=True)
        # re-balance GPUs over the filtered set
        man["cuda_device"] = [i % 2 for i in range(len(man))]
        man["command"] = man.apply(_command, axis=1)
        write(man, tag="only")
    else:
        write(man)
    n_zs = int((man.kind == "zeroshot_ci").sum())
    n_bg = int((man.kind == "a4_bgonly").sum())
    n_missing = int(man.missing_checkpoint.sum())
    n_unsupported = int((~man.supported).sum())
    print("=" * 78)
    print(
        f"RERUN MANIFEST: {len(man)} jobs  ({n_zs} zero-shot, {n_bg} bg-only fine-tuned)"
    )
    print(
        f"  runnable on THIS branch: {int(man.supported.sum())}  |  GPU0: "
        f"{int(((man.cuda_device==0)&man.supported).sum())}  GPU1: {int(((man.cuda_device==1)&man.supported).sum())}"
    )
    if n_unsupported:
        us = sorted(man.loc[~man.supported, "model"].unique())
        print(
            f"  !!! {n_unsupported} jobs EXCLUDED from runners (models not on this branch: {us}) "
            f"-- run on metabonet-integration or use published point estimates."
        )
    if n_missing:
        print(
            f"  !!! {n_missing} bg-only jobs have NO checkpoint found — flagged in manifest."
        )
    print("=" * 78)
    cols = [
        "job_id",
        "model",
        "dataset",
        "kind",
        "mode",
        "cuda_device",
        "expected_rmse",
        "supported",
        "venv",
    ]
    print(man[cols].to_string(index=False))
    print(f"\nWrote: {OUT}/manifest.csv , run_gpu0.sh , run_gpu1.sh , launch_all.sh")
    print("NOTHING LAUNCHED. Review manifest.csv, then: bash " f"{OUT}/launch_all.sh")


if __name__ == "__main__":
    main()
