#!/usr/bin/env python3
"""Generate a grid and submit it to Slurm (or run locally).

Usage:
    python scripts/run_sweep.py --model rf              # submit to Slurm
    python scripts/run_sweep.py --model rf --local      # run locally (sequential)
    python scripts/run_sweep.py --model rf --dry-run    # preview only
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_grid import generate_grid, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRID_YAML = ROOT / "configs/sweep/grid.yaml"
ARRAY_SCRIPT = ROOT / "scripts/slurm/run_array.sh"


def main():
    p = argparse.ArgumentParser(description="Submit a model sweep.")
    p.add_argument("--model", required=True, choices=["rf", "lstm", "patchtst"])
    p.add_argument("--local",   action="store_true", help="Run locally instead of Slurm.")
    p.add_argument("--dry-run", action="store_true", help="Show configs, don't run.")
    args = p.parse_args()

    cfg = load_config(GRID_YAML)
    cfg["models"] = {args.model: cfg["models"][args.model]}
    rows = generate_grid(cfg)

    grid_path = ROOT / f"scripts/grid_{args.model}.jsonl"
    with open(grid_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    (ROOT / "logs").mkdir(exist_ok=True)

    print(f"model={args.model}  experiments={len(rows)}  grid={grid_path.name}")

    if args.dry_run:
        for row in rows[:5]:
            print(" ", row)
        if len(rows) > 5:
            print(f"  ... and {len(rows) - 5} more")
        return

    if args.local:
        for i, row in enumerate(rows):
            print(f"[{i + 1}/{len(rows)}] {row}")
            subprocess.run(
                [sys.executable, "scripts/run_config.py", json.dumps(row)],
                cwd=ROOT, check=True,
            )
    else:
        subprocess.run(
            [
                "sbatch",
                f"--array=0-{len(rows) - 1}",
                f"--export=ALL,GRID_FILE=scripts/grid_{args.model}.jsonl",
                str(ARRAY_SCRIPT),
            ],
            cwd=ROOT, check=True,
        )


if __name__ == "__main__":
    main()
