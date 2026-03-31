#!/usr/bin/env python3
"""Run one experiment from a JSON config string.

Usage:
    python scripts/run_config.py '<json>'

Example:
    python scripts/run_config.py '{"model":"rf","regime":"bear","L":96,"H":21,"strategy":"mimo","step":0,"seed":2024,"n_estimators":200,"max_depth":null}'
"""
import json
import subprocess
import sys

cfg = json.loads(sys.argv[1])
skip = {"model", "regime", "L", "H", "strategy", "training_mode", "step", "seed"}
hparams = {k: v for k, v in cfg.items() if k not in skip}

subprocess.run(
    [
        sys.executable, "-m", "tsforecast.cli.train",
        "--model",    cfg["model"],
        "--regime",   cfg["regime"],
        "--L",        str(cfg["L"]),
        "--H",        str(cfg["H"]),
        "--strategy", cfg.get("strategy", "mimo"),
        "--training-mode",  cfg.get("training_mode", "per_ticker"),
        "--step",     str(cfg.get("step", 0)),
        "--seed",     str(cfg.get("seed", 2024)),
        "--hparams",  json.dumps(hparams),
    ],
    check=True,
)
