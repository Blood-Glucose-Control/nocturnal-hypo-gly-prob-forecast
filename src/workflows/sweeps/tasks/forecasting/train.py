#!/usr/bin/env python3
"""Forecasting-task sweep training adapter."""

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

from src.utils.config_loader import load_yaml_config


@dataclass(frozen=True)
class SweepConfig:
    model_config_path: str
    datasets: Sequence[str]


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


def _append_manifest(
    manifest: Path, stem: str, out_dir: Path, lock: threading.Lock
) -> None:
    with lock:
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write(f"{stem}\t{out_dir.as_posix()}\n")


def _load_sweep_configs(
    *,
    sweep_spec: Path | None,
    model_config_dir: Path | None,
    model_config_glob: str,
    datasets: Sequence[str] | None,
) -> List[SweepConfig]:
    if sweep_spec is not None:
        raw = load_yaml_config(str(sweep_spec))
        if not isinstance(raw, dict):
            raise ValueError(f"Sweep spec must be a mapping: {sweep_spec.as_posix()}")
        jobs = raw.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError(
                f"Sweep spec must define a non-empty 'jobs' list: {sweep_spec.as_posix()}"
            )

        configs: List[SweepConfig] = []
        for idx, item in enumerate(jobs):
            if not isinstance(item, dict):
                raise ValueError(f"jobs[{idx}] must be a mapping")
            config_path = item.get("model_config")
            item_datasets = item.get("datasets")
            if not isinstance(config_path, str) or not config_path.strip():
                raise ValueError(f"jobs[{idx}].model_config must be a non-empty string")
            if not isinstance(item_datasets, list) or not item_datasets:
                raise ValueError(f"jobs[{idx}].datasets must be a non-empty list")
            if any(not isinstance(ds, str) or not ds.strip() for ds in item_datasets):
                raise ValueError(
                    f"jobs[{idx}].datasets entries must be non-empty strings"
                )
            configs.append(
                SweepConfig(
                    model_config_path=config_path.strip(),
                    datasets=tuple(ds.strip() for ds in item_datasets),
                )
            )
        return configs

    if model_config_dir is None:
        raise ValueError("Either --sweep-spec or --model-config-dir must be provided.")

    if not datasets:
        raise ValueError("--datasets is required when --model-config-dir is used.")

    files = sorted(model_config_dir.glob(model_config_glob))
    if not files:
        raise ValueError(
            f"No model config files matched '{model_config_glob}' in "
            f"{model_config_dir.as_posix()}"
        )

    return [
        SweepConfig(model_config_path=file.as_posix(), datasets=tuple(datasets))
        for file in files
    ]


def _apply_dataset_filter(
    configs: Sequence[SweepConfig], allowed_datasets: Sequence[str]
) -> Tuple[List[SweepConfig], int]:
    allowed = {dataset.strip() for dataset in allowed_datasets if dataset.strip()}
    filtered: List[SweepConfig] = []
    dropped = 0
    for item in configs:
        datasets = tuple(dataset for dataset in item.datasets if dataset in allowed)
        if datasets:
            filtered.append(
                SweepConfig(model_config_path=item.model_config_path, datasets=datasets)
            )
        else:
            dropped += 1
    return filtered, dropped


def _run_worker(
    *,
    slot: int,
    gpu: str,
    model_type: str,
    configs: Sequence[SweepConfig],
    project_root: Path,
    workflow_script: Path,
    data_config_dir: str,
    skip_steps: str,
    manifest: Path,
    artifacts_root: Path,
    dry_run: bool,
    log_dir: Path,
    manifest_lock: threading.Lock,
) -> Tuple[int, int, List[str]]:
    label = f"GPU{gpu}w{slot}"
    pass_count = 0
    fail_count = 0
    failed: List[str] = []
    log_file = log_dir / f"{model_type}_sweep_gpu{gpu}_w{slot}.log"

    with log_file.open("w", encoding="utf-8") as log:

        def log_line(msg: str) -> None:
            print(msg, flush=True)
            print(msg, file=log, flush=True)

        log_line(
            f"[{label}] Starting at {datetime.now().isoformat(timespec='seconds')}"
        )

        for item in configs:
            stem = Path(item.model_config_path).stem
            existing_dir = _read_latest_manifest_entry(manifest, stem, manifest_lock)
            if existing_dir:
                existing_path = Path(existing_dir)
                if not existing_path.is_absolute():
                    existing_path = project_root / existing_path
                if existing_path.exists():
                    log_line(
                        f"[{label}] [SKIP] {stem} — already in manifest: "
                        f"{existing_path.name}"
                    )
                    pass_count += 1
                    continue

            datasets = " ".join(item.datasets)
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{slot}"
            out_dir = (
                artifacts_root
                / f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_RID{run_id}_{stem}"
            )

            log_line("")
            log_line(f"[{label}] ============================================")
            log_line(f"[{label}]  Training: {model_type} / {stem}")
            log_line(f"[{label}]  Config:   {item.model_config_path}")
            log_line(f"[{label}]  Datasets: {datasets}")
            log_line(f"[{label}]  Output:   {out_dir.as_posix()}")
            log_line(f"[{label}] ============================================")

            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "MODEL_TYPE": model_type,
                    "MODEL_CONFIG": item.model_config_path,
                    "CONFIG_DIR": data_config_dir,
                    "DATASETS": datasets,
                    "SKIP_TRAINING": "false",
                    "SKIP_STEPS": skip_steps,
                    "OUTPUT_BASE_DIR": out_dir.as_posix(),
                    "RUN_ID": run_id,
                }
            )

            cmd = ["bash", workflow_script.as_posix()]
            if dry_run:
                log_line(f"[{label}] [DRY_RUN] {' '.join(cmd)}")
                pass_count += 1
                continue

            result = subprocess.run(
                cmd,
                cwd=project_root.as_posix(),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if result.returncode == 0:
                _append_manifest(manifest, stem, out_dir, manifest_lock)
                log_line(f"[{label}] [OK] {stem}")
                pass_count += 1
            else:
                log_line(f"[{label}] [FAIL] {stem}")
                fail_count += 1
                failed.append(stem)

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
        description="Forecasting sweep training adapter (Python-first)."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type passed to run_forecasting_workflow.sh (default: MODEL_TYPE env or chronos2).",
    )
    parser.add_argument(
        "--sweep-spec",
        type=str,
        default=None,
        help="YAML sweep spec containing per-config dataset lists.",
    )
    parser.add_argument(
        "--model-config-dir",
        type=str,
        default=None,
        help="Directory of model YAML configs for one-dataset-list sweep mode.",
    )
    parser.add_argument(
        "--model-config-glob",
        type=str,
        default="*.yaml",
        help="Glob pattern used with --model-config-dir (default: *.yaml).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Datasets list used for each config when --model-config-dir is used. "
            "With --sweep-spec, acts as a dataset-name filter."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Space-separated GPU IDs override (default: GPUS env or auto-detect).",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=None,
        help="Workers per GPU override (default: JOBS_PER_GPU env or 2).",
    )
    parser.add_argument(
        "--data-config-dir",
        type=str,
        default=None,
        help="Holdout data config directory (default: CONFIG_DIR env or configs/data/holdout_10pct).",
    )
    parser.add_argument(
        "--skip-steps",
        type=str,
        default=None,
        help="Space-separated forecasting workflow steps to skip (default: SKIP_STEPS env or '1 2 4 7').",
    )
    parser.add_argument(
        "--workflow-script",
        type=str,
        default="scripts/workflows/forecasting/run_forecasting_workflow.sh",
        help="Workflow launcher script path relative to repo root.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default=None,
        help="Artifacts root directory (default: trained_models/artifacts/<model_type>).",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Manifest path override (default: <artifacts_root>/sweep_manifest.txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print orchestration commands without executing training.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_model_type: str = "chronos2",
    default_sweep_spec: str | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[5]

    model_type = args.model_type or os.environ.get("MODEL_TYPE", default_model_type)
    workflow_script = project_root / args.workflow_script
    data_config_dir = args.data_config_dir or os.environ.get(
        "CONFIG_DIR", "configs/data/holdout_10pct"
    )
    skip_steps = args.skip_steps or os.environ.get("SKIP_STEPS", "1 2 4 7")
    jobs_per_gpu = args.jobs_per_gpu or int(os.environ.get("JOBS_PER_GPU", "2"))
    dry_run = args.dry_run or _is_truthy_env("DRY_RUN")

    artifacts_root = (
        Path(args.artifacts_root)
        if args.artifacts_root
        else project_root / f"trained_models/artifacts/{model_type}"
    )
    manifest = (
        Path(args.manifest_path)
        if args.manifest_path
        else artifacts_root / "sweep_manifest.txt"
    )
    log_dir = project_root / "logs"

    if not workflow_script.exists():
        raise FileNotFoundError(
            f"Workflow script not found: {workflow_script.as_posix()}"
        )

    sweep_spec_arg = args.sweep_spec or default_sweep_spec
    sweep_spec = Path(sweep_spec_arg) if sweep_spec_arg else None
    if sweep_spec is not None and not sweep_spec.is_absolute():
        sweep_spec = project_root / sweep_spec
    model_config_dir = Path(args.model_config_dir) if args.model_config_dir else None
    if model_config_dir is not None and not model_config_dir.is_absolute():
        model_config_dir = project_root / model_config_dir

    if sweep_spec is not None and model_config_dir is not None:
        raise ValueError("Provide either --sweep-spec or --model-config-dir, not both.")

    configs = _load_sweep_configs(
        sweep_spec=sweep_spec,
        model_config_dir=model_config_dir,
        model_config_glob=args.model_config_glob,
        datasets=args.datasets,
    )
    if not configs:
        raise ValueError("No sweep configs resolved.")

    dataset_filter: List[str] | None = None
    if sweep_spec is not None:
        if args.datasets:
            dataset_filter = args.datasets
        else:
            datasets_env = os.environ.get("DATASETS", "").strip()
            if datasets_env:
                dataset_filter = datasets_env.split()
    dropped_configs = 0
    if dataset_filter:
        configs, dropped_configs = _apply_dataset_filter(configs, dataset_filter)
        if not configs:
            raise ValueError(
                "Dataset filter removed all sweep configs. Check --datasets / DATASETS values."
            )

    gpu_ids = _detect_gpu_ids(args.gpus)
    n_gpus = len(gpu_ids)
    n_slots = max(1, n_gpus * jobs_per_gpu)

    print(
        f"=== {model_type} sweep training  {datetime.now().isoformat(timespec='seconds')} ==="
    )
    print(f"  Configs: {len(configs)}")
    if sweep_spec is not None:
        print(f"  Sweep spec: {sweep_spec.as_posix()}")
        if dataset_filter:
            print(f"  Dataset filter: {' '.join(dataset_filter)}")
            if dropped_configs > 0:
                print(f"  Configs dropped by filter: {dropped_configs}")
    elif model_config_dir is not None:
        print(f"  Model config dir: {model_config_dir.as_posix()}")
        print(f"  Datasets: {' '.join(args.datasets or [])}")
    print(f"  GPUs: {' '.join(gpu_ids)}  ({n_gpus} total)")
    print(f"  Jobs per GPU: {jobs_per_gpu}  ({n_slots} total slots)")
    print(f"  DRY_RUN: {'1' if dry_run else '0'}")
    print("")

    artifacts_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest.touch(exist_ok=True)

    slot_gpu: Dict[int, str] = {}
    slot_configs: Dict[int, List[SweepConfig]] = {slot: [] for slot in range(n_slots)}
    for slot in range(n_slots):
        slot_gpu[slot] = gpu_ids[slot % n_gpus]
    for idx, item in enumerate(configs):
        slot_configs[idx % n_slots].append(item)

    print("Config distribution:")
    for slot in range(n_slots):
        gpu = slot_gpu[slot]
        print(f"  GPU {gpu} worker {slot}: {len(slot_configs[slot])} configs")
    print("")

    print("Launching workers...")
    for slot in range(n_slots):
        gpu = slot_gpu[slot]
        log_file = log_dir / f"{model_type}_sweep_gpu{gpu}_w{slot}.log"
        print(f"Launching GPU {gpu} worker {slot} -> {log_file.as_posix()}")
    print("")
    print("All workers launched. Waiting for completion...")
    print(f"(tail -f logs/{model_type}_sweep_gpu<N>_w<slot>.log to monitor)")
    print("")

    manifest_lock = threading.Lock()
    overall_fail = 0
    with ThreadPoolExecutor(max_workers=n_slots) as executor:
        futures = {
            slot: executor.submit(
                _run_worker,
                slot=slot,
                gpu=slot_gpu[slot],
                model_type=model_type,
                configs=slot_configs[slot],
                project_root=project_root,
                workflow_script=workflow_script,
                data_config_dir=data_config_dir,
                skip_steps=skip_steps,
                manifest=manifest,
                artifacts_root=artifacts_root,
                dry_run=dry_run,
                log_dir=log_dir,
                manifest_lock=manifest_lock,
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
                    f"(see logs/{model_type}_sweep_gpu{gpu}_w{slot}.log)"
                )
                overall_fail += 1

    print("")
    print(
        f"=== {model_type} sweep training complete  "
        f"{datetime.now().isoformat(timespec='seconds')} ==="
    )
    print(f"  Manifest: {manifest.as_posix()}")
    if overall_fail > 0:
        print(f"  {overall_fail} GPU worker(s) reported failures — check logs above.")
        return 1
    print("  All configs trained successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
