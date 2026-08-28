#!/usr/bin/env python3
"""Forecasting-task sweep evaluation adapter."""

from __future__ import annotations

import argparse
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .....config.schemas import load_forecasting_eval_sweep_spec_from_yaml


@dataclass(frozen=True)
class SweepEvalConfig:
    model_config_path: str
    context_length: int
    forecast_length: int
    covariate_cols: Tuple[str, ...]
    finetuned_datasets: Tuple[str, ...]
    zeroshot_datasets: Tuple[str, ...]
    output_dir_template: str | None
    probabilistic: bool
    no_dilate: bool


@dataclass(frozen=True)
class EvalJob:
    mode: str  # finetuned | zeroshot
    stem: str
    model_config_path: str
    dataset: str
    context_length: int
    forecast_length: int
    covariate_cols: Tuple[str, ...]
    checkpoint_path: str | None
    output_dir_template: str | None
    probabilistic: bool
    no_dilate: bool


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _detect_gpu_ids(gpus_override: str | None) -> List[str]:
    gpus_env = (gpus_override or os.environ.get("GPUS", "")).strip()
    if gpus_env:
        return gpus_env.split()

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return gpu_ids or ["0"]
    except Exception:
        return ["0"]


def _read_latest_manifest_entry(manifest: Path, stem: str, lock: threading.Lock) -> str:
    with lock:
        if not manifest.exists():
            return ""
        latest = ""
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            if parts[0] == stem:
                latest = parts[1]
        return latest


def _resolve_checkpoint_path(project_root: Path, out_dir: str) -> str | None:
    run_dir = Path(out_dir)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    if not run_dir.exists():
        return None

    for candidate in (run_dir / "model.pt", run_dir / "model.ckpt", run_dir):
        if candidate.exists():
            return candidate.as_posix()
    return None


def _load_eval_configs(
    *,
    sweep_spec: Path,
    probabilistic_override: bool | None,
    no_dilate_override: bool | None,
    forecast_length_override: int | None,
) -> List[SweepEvalConfig]:
    validated = load_forecasting_eval_sweep_spec_from_yaml(sweep_spec)

    default_probabilistic = validated.probabilistic
    default_no_dilate = validated.no_dilate
    default_forecast_length = validated.forecast_length
    default_output_dir_template = validated.output_dir_template

    configs: List[SweepEvalConfig] = []
    for item in validated.jobs:
        probabilistic = (
            probabilistic_override
            if probabilistic_override is not None
            else (
                item.probabilistic
                if item.probabilistic is not None
                else default_probabilistic
            )
        )
        no_dilate = (
            no_dilate_override
            if no_dilate_override is not None
            else item.no_dilate
            if item.no_dilate is not None
            else default_no_dilate
        )
        forecast_length = (
            forecast_length_override
            if forecast_length_override is not None
            else (
                item.forecast_length
                if item.forecast_length is not None
                else default_forecast_length
            )
        )
        output_dir_template = (
            item.output_dir_template
            if item.output_dir_template is not None
            else default_output_dir_template
        )

        configs.append(
            SweepEvalConfig(
                model_config_path=item.model_config_path,
                context_length=item.context_length,
                forecast_length=forecast_length,
                covariate_cols=tuple(item.covariate_cols),
                finetuned_datasets=tuple(item.finetuned_datasets),
                zeroshot_datasets=tuple(item.zeroshot_datasets),
                output_dir_template=output_dir_template,
                probabilistic=probabilistic,
                no_dilate=no_dilate,
            )
        )
    return configs


def _build_eval_jobs(
    *,
    configs: Sequence[SweepEvalConfig],
    manifest: Path,
    project_root: Path,
    manifest_lock: threading.Lock,
    dry_run: bool,
) -> List[EvalJob]:
    jobs: List[EvalJob] = []
    for item in configs:
        stem = Path(item.model_config_path).stem

        if item.finetuned_datasets:
            out_dir = _read_latest_manifest_entry(manifest, stem, manifest_lock)
            checkpoint = None
            if out_dir:
                checkpoint = _resolve_checkpoint_path(project_root, out_dir)

            for dataset in item.finetuned_datasets:
                jobs.append(
                    EvalJob(
                        mode="finetuned",
                        stem=stem,
                        model_config_path=item.model_config_path,
                        dataset=dataset,
                        context_length=item.context_length,
                        forecast_length=item.forecast_length,
                        covariate_cols=item.covariate_cols,
                        checkpoint_path=checkpoint,
                        output_dir_template=item.output_dir_template,
                        probabilistic=item.probabilistic,
                        no_dilate=item.no_dilate,
                    )
                )

        for dataset in item.zeroshot_datasets:
            jobs.append(
                EvalJob(
                    mode="zeroshot",
                    stem=stem,
                    model_config_path=item.model_config_path,
                    dataset=dataset,
                    context_length=item.context_length,
                    forecast_length=item.forecast_length,
                    covariate_cols=item.covariate_cols,
                    checkpoint_path=None,
                    output_dir_template=item.output_dir_template,
                    probabilistic=item.probabilistic,
                    no_dilate=item.no_dilate,
                )
            )

    if not jobs:
        raise ValueError("Sweep eval spec resolved zero jobs.")

    requires_manifest = any(cfg.finetuned_datasets for cfg in configs)
    if requires_manifest and not dry_run and not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found for finetuned evaluation: {manifest.as_posix()}"
        )

    return jobs


def _resolve_output_dir(
    project_root: Path, model_type: str, job: EvalJob
) -> str | None:
    template = job.output_dir_template
    if not template:
        return None
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    try:
        rendered = template.format(
            model_type=model_type,
            stem=job.stem,
            dataset=job.dataset,
            mode=job.mode,
            context_length=job.context_length,
            forecast_length=job.forecast_length,
            timestamp=timestamp,
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown output_dir_template placeholder '{exc.args[0]}' for {job.stem}"
        ) from exc

    out_dir = Path(rendered)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    return out_dir.as_posix()


def _run_worker(
    *,
    slot: int,
    gpu: str,
    model_type: str,
    jobs: Sequence[EvalJob],
    project_root: Path,
    python_executable: str,
    eval_script: Path,
    data_config_dir: str,
    done_file: Path,
    dry_run: bool,
    log_dir: Path,
    done_lock: threading.Lock,
) -> Tuple[int, int, List[str]]:
    label = f"GPU{gpu}w{slot}"
    pass_count = 0
    fail_count = 0
    failed: List[str] = []
    log_file = log_dir / f"{model_type}_sweep_eval_gpu{gpu}_w{slot}.log"

    with log_file.open("w", encoding="utf-8") as log:

        def log_line(msg: str) -> None:
            print(msg, flush=True)
            print(msg, file=log, flush=True)

        log_line(
            f"[{label}] Starting at {datetime.now().isoformat(timespec='seconds')}"
        )

        for job in jobs:
            done_key = f"{job.stem}|{job.dataset}|{job.mode}"
            with done_lock:
                if done_file.exists():
                    existing_done = set(
                        line.strip()
                        for line in done_file.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                else:
                    existing_done = set()
            if done_key in existing_done:
                log_line(f"[{label}] [SKIP] {done_key} — already in done log")
                pass_count += 1
                continue

            label_mode = "Fine-tuned" if job.mode == "finetuned" else "Zero-shot"
            job_label = f"{model_type} / {job.stem} / {job.dataset} ({job.mode})"

            if job.mode == "finetuned" and not job.checkpoint_path:
                if dry_run:
                    log_line(
                        f"[{label}] [DRY_RUN] {job_label} — checkpoint unresolved "
                        "(manifest missing or no matching entry)"
                    )
                    pass_count += 1
                    continue
                log_line(f"[{label}] [SKIP] {job_label} — checkpoint missing")
                fail_count += 1
                failed.append(f"{job_label} (missing checkpoint)")
                continue

            cmd = [
                python_executable,
                eval_script.as_posix(),
                "--model",
                model_type,
                "--model-config",
                job.model_config_path,
                "--dataset",
                job.dataset,
                "--config-dir",
                data_config_dir,
                "--context-length",
                str(job.context_length),
                "--forecast-length",
                str(job.forecast_length),
                "--cuda-device",
                gpu,
            ]
            if job.mode == "finetuned" and job.checkpoint_path:
                cmd.extend(["--checkpoint", job.checkpoint_path])
            if job.probabilistic:
                cmd.append("--probabilistic")
            if job.no_dilate:
                cmd.append("--no-dilate")
            if job.covariate_cols:
                cmd.append("--covariate-cols")
                cmd.extend(job.covariate_cols)
            output_dir = _resolve_output_dir(project_root, model_type, job)
            if output_dir:
                cmd.extend(["--output-dir", output_dir])

            log_line("")
            log_line(f"[{label}] ============================================")
            log_line(f"[{label}]  {label_mode} eval: {job_label}")
            if job.checkpoint_path:
                log_line(f"[{label}]  Checkpoint: {job.checkpoint_path}")
            if output_dir:
                log_line(f"[{label}]  Output: {output_dir}")
            log_line(f"[{label}] ============================================")
            log_line(f"[{label}]  CMD: {' '.join(cmd)}")

            if dry_run:
                log_line(f"[{label}] [DRY_RUN] {done_key}")
                pass_count += 1
                continue

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            result = subprocess.run(
                cmd,
                cwd=project_root.as_posix(),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if result.returncode == 0:
                with done_lock:
                    with done_file.open("a", encoding="utf-8") as fh:
                        fh.write(f"{done_key}\n")
                log_line(f"[{label}] [OK] {done_key}")
                pass_count += 1
            else:
                log_line(f"[{label}] [FAIL] {done_key}")
                fail_count += 1
                failed.append(done_key)

        log_line("")
        log_line(
            f"[{label}] Done at {datetime.now().isoformat(timespec='seconds')}  —  "
            f"passed: {pass_count} / {pass_count + fail_count}"
        )
        if failed:
            log_line(f"[{label}] Failed: {' '.join(failed)}")

    return pass_count, fail_count, failed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecasting sweep evaluation adapter (Python-first)."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type passed to nocturnal_hypo_eval.py (default: MODEL_TYPE env or chronos2).",
    )
    parser.add_argument(
        "--sweep-spec",
        type=str,
        default=None,
        help="YAML sweep spec with eval jobs and dataset routing.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Space-separated GPU IDs override (default: GPUS env or auto-detect).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional dataset-name filter applied after loading sweep jobs "
            "(default: DATASETS env if set; otherwise no filter)."
        ),
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=None,
        help="Workers per GPU override (default: JOBS_PER_GPU env or 4).",
    )
    parser.add_argument(
        "--data-config-dir",
        type=str,
        default=None,
        help="Holdout data config directory (default: CONFIG_DIR env or configs/data/holdout_10pct).",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Training manifest path override (default: trained_models/artifacts/<model>/sweep_manifest.txt).",
    )
    parser.add_argument(
        "--eval-script",
        type=str,
        default="scripts/evaluation/nocturnal_hypo_eval.py",
        help="Evaluation script path relative to repo root.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=None,
        help="Python executable used to run eval script (default: EVAL_PYTHON env or python).",
    )
    parser.add_argument(
        "--done-file",
        type=str,
        default=None,
        help="Done-log path override (default: logs/<model>_eval_done.log).",
    )
    parser.add_argument(
        "--forecast-length",
        type=int,
        default=None,
        help="Override forecast length for all jobs.",
    )
    parser.add_argument(
        "--probabilistic",
        action="store_true",
        default=None,
        help="Force probabilistic eval on all jobs.",
    )
    parser.add_argument(
        "--no-probabilistic",
        action="store_false",
        dest="probabilistic",
        help="Force deterministic eval on all jobs.",
    )
    parser.add_argument(
        "--no-dilate",
        action="store_true",
        default=None,
        help="Force --no-dilate on all jobs.",
    )
    parser.add_argument(
        "--with-dilate",
        action="store_false",
        dest="no_dilate",
        help="Force DILATE metrics on all jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print orchestration commands without executing evaluation.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_model_type: str = "chronos2",
    default_sweep_spec: str | None = None,
    default_python_executable: str = "python",
) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[5]

    model_type = args.model_type or os.environ.get("MODEL_TYPE", default_model_type)
    data_config_dir = args.data_config_dir or os.environ.get(
        "CONFIG_DIR", "configs/data/holdout_10pct"
    )
    jobs_per_gpu = args.jobs_per_gpu or int(os.environ.get("JOBS_PER_GPU", "4"))
    dry_run = args.dry_run or _is_truthy_env("DRY_RUN")
    python_executable = (
        args.python_executable
        or os.environ.get("EVAL_PYTHON")
        or default_python_executable
    )

    sweep_spec_arg = args.sweep_spec or default_sweep_spec
    if not sweep_spec_arg:
        raise ValueError(
            "Sweep eval spec is required (pass --sweep-spec or set default)."
        )
    sweep_spec = Path(sweep_spec_arg)
    if not sweep_spec.is_absolute():
        sweep_spec = project_root / sweep_spec
    if not sweep_spec.exists():
        raise FileNotFoundError(f"Sweep eval spec not found: {sweep_spec.as_posix()}")

    manifest = (
        Path(args.manifest_path)
        if args.manifest_path
        else project_root / f"trained_models/artifacts/{model_type}/sweep_manifest.txt"
    )
    done_file = (
        Path(args.done_file)
        if args.done_file
        else project_root / "logs" / f"{model_type}_eval_done.log"
    )
    eval_script = project_root / args.eval_script
    if not eval_script.exists():
        raise FileNotFoundError(f"Eval script not found: {eval_script.as_posix()}")
    if not Path(python_executable).is_absolute() and "/" in python_executable:
        python_executable = (project_root / python_executable).as_posix()

    configs = _load_eval_configs(
        sweep_spec=sweep_spec,
        probabilistic_override=args.probabilistic,
        no_dilate_override=args.no_dilate,
        forecast_length_override=args.forecast_length,
    )

    manifest_lock = threading.Lock()
    jobs = _build_eval_jobs(
        configs=configs,
        manifest=manifest,
        project_root=project_root,
        manifest_lock=manifest_lock,
        dry_run=dry_run,
    )

    dataset_filter = args.datasets
    if dataset_filter is None:
        datasets_env = os.environ.get("DATASETS", "").strip()
        if datasets_env:
            dataset_filter = datasets_env.split()
    if dataset_filter:
        allowed = {dataset.strip() for dataset in dataset_filter if dataset.strip()}
        jobs = [job for job in jobs if job.dataset in allowed]
        if not jobs:
            raise ValueError(
                "Dataset filter removed all eval jobs. Check --datasets / DATASETS values."
            )

    gpu_ids = _detect_gpu_ids(args.gpus)
    n_gpus = len(gpu_ids)
    n_slots = max(1, n_gpus * jobs_per_gpu)

    print(
        f"=== {model_type} sweep eval  {datetime.now().isoformat(timespec='seconds')} ==="
    )
    print(f"  Sweep spec: {sweep_spec.as_posix()}")
    print(f"  Jobs: {len(jobs)}")
    if dataset_filter:
        print(f"  Dataset filter: {' '.join(dataset_filter)}")
    print(f"  Python: {python_executable}")
    print(f"  GPUs: {' '.join(gpu_ids)}  ({n_gpus} total)")
    print(f"  Jobs per GPU: {jobs_per_gpu}  ({n_slots} total slots)")
    print(f"  DRY_RUN: {'1' if dry_run else '0'}")
    print("")

    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.touch(exist_ok=True)
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    slot_gpu: Dict[int, str] = {}
    slot_jobs: Dict[int, List[EvalJob]] = {slot: [] for slot in range(n_slots)}
    for slot in range(n_slots):
        slot_gpu[slot] = gpu_ids[slot % n_gpus]
    for idx, job in enumerate(jobs):
        slot_jobs[idx % n_slots].append(job)

    print("Job distribution:")
    for slot in range(n_slots):
        gpu = slot_gpu[slot]
        print(f"  GPU {gpu} worker {slot}: {len(slot_jobs[slot])} jobs")
    print("")

    print("Launching workers...")
    for slot in range(n_slots):
        gpu = slot_gpu[slot]
        log_file = log_dir / f"{model_type}_sweep_eval_gpu{gpu}_w{slot}.log"
        print(f"Launching GPU {gpu} worker {slot} -> {log_file.as_posix()}")
    print("")
    print("All workers launched. Waiting for completion...")
    print(f"(tail -f logs/{model_type}_sweep_eval_gpu<N>_w<slot>.log to monitor)")
    print("")

    done_lock = threading.Lock()
    overall_fail = 0
    with ThreadPoolExecutor(max_workers=n_slots) as executor:
        futures = {
            slot: executor.submit(
                _run_worker,
                slot=slot,
                gpu=slot_gpu[slot],
                model_type=model_type,
                jobs=slot_jobs[slot],
                project_root=project_root,
                python_executable=python_executable,
                eval_script=eval_script,
                data_config_dir=data_config_dir,
                done_file=done_file,
                dry_run=dry_run,
                log_dir=log_dir,
                done_lock=done_lock,
            )
            for slot in range(n_slots)
        }
        for slot, future in futures.items():
            gpu = slot_gpu[slot]
            _, fail_count, _ = future.result()
            if fail_count == 0:
                print(f"GPU {gpu} worker {slot}: SUCCESS")
            else:
                print(
                    f"GPU {gpu} worker {slot}: FAILED  "
                    f"(see logs/{model_type}_sweep_eval_gpu{gpu}_w{slot}.log)"
                )
                overall_fail += 1

    print("")
    print(
        f"=== {model_type} sweep eval complete  "
        f"{datetime.now().isoformat(timespec='seconds')} ==="
    )
    print(f"  Done file: {done_file.as_posix()}")
    if overall_fail > 0:
        print(f"  {overall_fail} GPU worker(s) reported failures — check logs above.")
        return 1
    print("  All eval jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
