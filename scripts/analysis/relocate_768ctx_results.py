"""
relocate_768ctx_results.py

One-shot script that:
1. Walks experiments/nocturnal_forecasting/512ctx_96fh/{deepar,patchtst,tft}/
2. Reads experiment_config.json in each run dir
3. Moves 768-ctx runs to experiments/nocturnal_forecasting/768ctx_96fh/{model}/
4. Moves 05_bg_high_lr TFT runs to experiments/nocturnal_forecasting/_bad_runs_archive/tft/

Done-file entries are NOT modified — they are keyed by stem|dataset, not path.
"""

import json
import shutil
from pathlib import Path

EXPERIMENT_ROOT = Path("experiments/nocturnal_forecasting")
SRC_DIR = EXPERIMENT_ROOT / "512ctx_96fh"
DST_768_DIR = EXPERIMENT_ROOT / "768ctx_96fh"
BAD_RUNS_DIR = EXPERIMENT_ROOT / "_bad_runs_archive" / "tft"

# Config stems that indicate a 768-ctx run (matched as substring of model_config path)
LONG_CTX_STEMS = {
    "deepar": ["02_long_ctx", "11_big"],
    "patchtst": ["04_long_ctx"],
    "tft": ["02_bg_long_ctx", "08_iob_long_ctx"],
}

# Config stems that should be archived as bad runs
BAD_STEMS = {
    "tft": ["05_bg_high_lr"],
}


def get_model_config(run_dir: Path) -> str:
    config_file = run_dir / "experiment_config.json"
    if not config_file.exists():
        return ""
    with open(config_file) as f:
        data = json.load(f)
    cli_args = data.get("cli_args", {})
    if isinstance(cli_args, dict):
        return cli_args.get("model_config", "")
    return data.get("model_config", "")


def move_run(src: Path, dst_parent: Path) -> None:
    dst_parent.mkdir(parents=True, exist_ok=True)
    dst = dst_parent / src.name
    if dst.exists():
        print(f"  [SKIP] destination already exists: {dst}")
        return
    shutil.move(str(src), str(dst))
    print(f"  [MOVED] {src} -> {dst}")


def main():
    moved_768 = 0
    moved_bad = 0

    for model in ["deepar", "patchtst", "tft"]:
        model_src = SRC_DIR / model
        if not model_src.exists():
            print(f"[WARN] {model_src} does not exist, skipping.")
            continue

        long_ctx_stems = LONG_CTX_STEMS.get(model, [])
        bad_stems = BAD_STEMS.get(model, [])

        run_dirs = sorted(p for p in model_src.iterdir() if p.is_dir())
        print(f"\n=== {model}: {len(run_dirs)} run dirs ===")

        for run_dir in run_dirs:
            model_config = get_model_config(run_dir)
            if not model_config:
                print(f"  [WARN] No model_config found in {run_dir}, skipping.")
                continue

            # Check bad runs first
            is_bad = any(stem in model_config for stem in bad_stems)
            if is_bad:
                move_run(run_dir, BAD_RUNS_DIR)
                moved_bad += 1
                continue

            # Check 768-ctx runs
            is_long_ctx = any(stem in model_config for stem in long_ctx_stems)
            if is_long_ctx:
                move_run(run_dir, DST_768_DIR / model)
                moved_768 += 1

    print(
        f"\nDone. Moved {moved_768} runs to 768ctx_96fh, {moved_bad} runs to _bad_runs_archive."
    )


if __name__ == "__main__":
    import os

    # Run from repo root
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    main()
