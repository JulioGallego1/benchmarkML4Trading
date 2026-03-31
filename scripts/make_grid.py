#!/usr/bin/env python3
"""Generate grid.jsonl from configs/sweep/grid.yaml.

Usage:
    python scripts/make_grid.py
    python scripts/make_grid.py --config configs/sweep/grid.yaml --output scripts/grid.jsonl

Each line in grid.jsonl is a JSON object with all parameters for one run.
Submit with:
    sbatch --array=0-$(( $(wc -l < scripts/grid.jsonl) - 1 )) scripts/slurm/run_array.sh
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def hparam_combos(model_cfg: dict) -> list[dict]:
    """Generate all products of hyperparameter lists for one model."""
    if not model_cfg:
        return [{}]
    keys = list(model_cfg.keys())
    values = [model_cfg[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def generate_grid(cfg: dict) -> list[dict]:
    regimes = cfg.get("regimes", [])

    horizons = cfg.get("horizons", [])
    context_lengths = cfg.get("context_lengths", [])

    strategies = cfg.get("strategies", ["mimo"]) #default to mimo if not specified

    steps = cfg.get("steps", [16])

    models_cfg = cfg.get("models", {})
    seeds = cfg.get("seeds", [2024]) #default seed

    rows = []
    for regime, H, L, strategy in itertools.product(regimes, horizons, context_lengths, strategies):
        # For MIMO, step is ialways 0 -> Dircet prediction.  
        # For recursive, generate one run per configured step value.
        step_values = steps if strategy == "recursive" else [0]
        for step in step_values:
            for model_name, model_hparams in models_cfg.items():
                for hparams in hparam_combos(model_hparams or {}):
                    for seed in seeds:
                        row = {
                            "model": model_name,
                            "regime": regime,
                            "L": L,
                            "H": H,
                            "strategy": strategy,
                            "step": step,
                            "seed": seed,
                        }
                        row.update(hparams)
                        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate grid.jsonl for Slurm array jobs.")
    parser.add_argument(
        "--config",
        default="configs/sweep/grid.yaml",
        help="Path to sweep grid YAML config.",
    )
    parser.add_argument(
        "--output",
        default="scripts/grid.jsonl",
        help="Output path for grid.jsonl.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    output_path = project_root / args.output

    cfg = load_config(config_path)
    rows = generate_grid(cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Generated {len(rows)} configs to {output_path}")


if __name__ == "__main__":
    main()
