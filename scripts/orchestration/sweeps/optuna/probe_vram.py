# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Peak-VRAM probe for a single Chronos-2 LoRA trial.

Runs one fine-tune + validation-forecast cycle at a chosen batch size and
context length while sampling ``nvidia-smi`` memory in the background, then
reports the peak VRAM used. Use it to decide how many concurrent trials fit on
a 98 GB GPU (the PoC caps at 2 per GPU).

Example::

    CUDA_VISIBLE_DEVICES=0 python scripts/orchestration/sweeps/optuna/probe_vram.py \\
        --batch-size 32 --context-length 288 --fine-tune-steps 200 --patient-limit 12
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))


class GpuMemorySampler(threading.Thread):
    """Background sampler of used VRAM (MiB) for the visible GPU."""

    def __init__(self, gpu_index: str, interval_s: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                used = int(out.stdout.strip().splitlines()[0])
                self.peak_mib = max(self.peak_mib, used)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=288)
    parser.add_argument("--fine-tune-steps", type=int, default=200)
    parser.add_argument("--patient-limit", type=int, default=12)
    parser.add_argument("--gpu-index", type=str, default="0")
    parser.add_argument(
        "--processed-path", type=str, default="cache/data/metabonet/processed"
    )
    args = parser.parse_args()

    import optuna

    from src.workflows.sweeps.optuna_chronos2.data import MetabonetData
    from src.workflows.sweeps.optuna_chronos2.objective import Chronos2Objective
    from src.workflows.sweeps.optuna_chronos2.search_space import SearchSpaceBounds

    data = MetabonetData(
        args.processed_path,
        patient_limit=args.patient_limit,
        max_val_windows_per_patient=3,
        max_val_windows_total=24,
    ).load()

    bounds = SearchSpaceBounds(
        context_min=args.context_length,
        context_max=args.context_length,
        context_step=1,
        steps_choices=[args.fine_tune_steps],
        batch_size_choices=[args.batch_size],
    )
    objective = Chronos2Objective(
        data, "/tmp/optuna_vram_probe", bounds=bounds, gpu_label=args.gpu_index
    )

    sampler = GpuMemorySampler(args.gpu_index)
    sampler.start()
    start = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1, catch=())
    elapsed = time.time() - start
    sampler.stop()
    sampler.join(timeout=2)

    trial = study.trials[0]
    peak_gib = sampler.peak_mib / 1024.0
    print("=" * 60)
    print("  Chronos-2 LoRA VRAM probe")
    print("=" * 60)
    print(f"  batch_size        : {args.batch_size}")
    print(f"  context_length    : {args.context_length}")
    print(f"  fine_tune_steps   : {args.fine_tune_steps}")
    print(f"  peak VRAM (GiB)   : {peak_gib:.1f}")
    print(f"  trial RMSE@120    : {trial.value}")
    print(f"  wall time (s)     : {elapsed:.1f}")
    if peak_gib > 0:
        fit_2 = 2 * peak_gib
        headroom = 98.0 - fit_2
        verdict = "OK (>=10 GiB headroom)" if headroom >= 10 else "RISKY"
        print(f"  2 trials/GPU est. : {fit_2:.1f} GiB used -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
