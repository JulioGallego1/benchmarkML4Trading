# ml4trading

ML pipeline for stock price forecasting. Benchmarks **Random Forest**, **LSTM**, and **PatchTST** across market regimes, forecasting strategies, horizons, and context lengths. Designed to run on a Slurm cluster (VRAIN) or locally.

## Table of Contents

- [Project Overview](#project-overview)
- [Experiment Design](#experiment-design)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Run Outputs](#run-outputs)
- [HPC / Slurm](#hpc--slurm)
- [Tests](#tests)
- [Aggregating Results](#aggregating-results)

---

## Project Overview

Each experiment is an independent, reproducible run that:

1. Loads daily close prices from pre-downloaded parquet files.
2. Creates sliding windows (context window L → forecast horizon H) on raw prices.
3. Trains a model in **pooled** mode (one shared model across all tickers) or **per-ticker** mode.
4. Evaluates on a held-out test set and computes 5 metrics (MAE, RMSE, MAPE, SMAPE, directional accuracy).
5. Saves metrics, predictions, plots, model checkpoint, and config to `runs/<run_id>/`.

Sweeps are generated from `configs/sweep/grid.yaml` as a Cartesian product of regimes × horizons × context lengths × strategies × hyperparameters. Each Slurm task receives its parameters as a JSON config line.

---

## Experiment Design

| Dimension | Values |
|---|---|
| **Market regime** | `bear` (test from 2022-03-11), `bull` (test from 2023-01-06) |
| **Horizon H** | 10 (≈2w), 21 (≈1m), 63 (≈3m) |
| **Context length L** | 32, 48, 96 days |
| **Forecasting strategy** | `mimo` (direct multi-output), `recursive` (iterative with block step) |
| **Training mode** | `pooled` (one shared model), `per_ticker` (independent model per ticker) |
| **Model** | `rf`, `lstm`, `patchtst` |

**MIMO** — the model maps L context steps directly to an H-step forecast vector in one shot.

**Recursive** — the model is trained to predict `--step` steps at a time; at inference, predictions are fed back as context and repeated until H steps are covered.

### Run ID Format

| Strategy | Format |
|---|---|
| MIMO | `{MODEL}_mimo_{regime}_L{L}_H{H}_{training_mode}_{timestamp}` |
| Recursive | `{MODEL}_rec_step{step}_{regime}_L{L}_H{H}_{training_mode}_{timestamp}` |

PatchTST appends an additional `revin` or `norevin` tag before the timestamp.

The timestamp uses millisecond precision with a 2-digit random suffix to avoid collisions in Slurm array jobs.

---

## Project Structure

```
ml4trading/
├── configs/
│   ├── tickers.txt             # one ticker per line (20 European equities by default)
│   ├── splits.yaml             # regime definitions (test_start, val_days, test_days)
│   ├── train.yaml              # shared training settings (epochs, batch_size, patience, seed)
│   ├── model/
│   │   ├── rf.yaml
│   │   ├── lstm.yaml
│   │   └── patchtst.yaml
│   └── sweep/
│       └── grid.yaml           # sweep axes and per-model hyperparameter grids
├── scripts/
│   ├── make_grid.py            # generates scripts/grid.jsonl from grid.yaml
│   ├── run_sweep.py            # one-command launcher: generates grid + submits to Slurm or runs locally
│   ├── run_config.py           # runs one experiment from a JSON config string
│   └── slurm/
│       ├── run_one.sh          # submit or run a single experiment
│       └── run_array.sh        # Slurm array job: reads one line from grid file per task
├── src/tsforecast/
│   ├── cli/train.py            # main entrypoint
│   ├── data/                   # loaders, splits, windows, cache, downloader
│   ├── models/                 # base (ABC), rf, lstm, patchtst
│   ├── training/               # engine, callbacks, reproducibility
│   ├── evaluation/             # metrics, plots, reports
│   ├── tracking/               # RunTracker (saves all artifacts atomically)
│   └── utils/                  # paths, logging
├── tests/
├── data/
│   ├── raw/                    # downloaded .parquet files (not versioned)
│   └── processed/              # windowed cache .npz files (not versioned)
└── runs/                       # one directory per run (not versioned)
```

---

## Setup

### 1. Clone and create the environment

```bash
git clone <repo-url>
cd ml4trading
conda env create -f environment.yml
conda activate tsforecast-env
```

### 2. Install the package

```bash
pip install -e ".[dev]"
```

### 3. Verify

```bash
python -c "import tsforecast; print('OK')"
pytest tests/ -m "not slow"
```

---

## Usage

### Step 1 — Add tickers

Edit `configs/tickers.txt` (one ticker per line, `#` comments supported):

```
AIR.PA
ASML
BMW.DE
```

### Step 2 — Download price data

```bash
# Use tickers from configs/tickers.txt (default)
python src/tsforecast/data/download_close_prices.py

# Specify tickers and date range explicitly
python src/tsforecast/data/download_close_prices.py \
  --tickers ASML IBE.MC AIR.PA \
  --start 2016-01-01 --end 2024-12-31

# Force re-download
python src/tsforecast/data/download_close_prices.py --force
```

Files are saved as `data/raw/<ticker>.parquet`.

### Step 3 — Run a single experiment

```bash
# MIMO strategy, per-ticker mode (defaults)
python -m tsforecast.cli.train --model rf --regime bear --L 96 --H 21

# Pooled mode
python -m tsforecast.cli.train --model rf --regime bear --L 96 --H 21 \
  --training-mode pooled

# Recursive strategy
python -m tsforecast.cli.train --model lstm --regime bear --L 96 --H 63 \
  --strategy recursive --step 16

# Override hyperparameters
python -m tsforecast.cli.train --model rf --regime bear --L 96 --H 21 \
  --hparams '{"n_estimators": 200, "max_depth": 10}'
```

**Key CLI flags:**

| Flag | Required | Default | Description |
|---|---|---|---|
| `--model` | Yes | — | `rf`, `lstm`, or `patchtst` |
| `--regime` | Yes | — | `bear` or `bull` |
| `--L` | Yes | — | Context length in trading days |
| `--H` | Yes | — | Forecast horizon in trading days |
| `--strategy` | No | `mimo` | `mimo` or `recursive` |
| `--step` | No | `16` | Block size for recursive strategy |
| `--training-mode` | No | `per_ticker` | `pooled` or `per_ticker` |
| `--seed` | No | from yaml | Random seed override |
| `--hparams` | No | `None` | JSON string of hyperparameter overrides |
| `--data-dir` | No | `data/raw` | Directory with `.parquet` files |
| `--config-dir` | No | `configs` | Directory with YAML config files |
| `--base-dir` | No | `.` | Project root |
| `--no-cache` | No | off | Skip window cache and recompute |

### Step 4 — Inspect results

```
runs/RF_mimo_bear_L96_H21_per_ticker_<timestamp>/
├── config.yaml
├── metrics.json
├── metrics_global.csv
├── predictions_detailed.csv
├── training_curves.png         # LSTM and PatchTST only
├── model/                      # pooled mode only
└── tickers/<TICKER>/
    ├── plot.png
    ├── plot_returns.png
    ├── training_curves.png     # LSTM and PatchTST, per-ticker mode only
    ├── predictions.csv         # per-ticker mode only
    ├── metrics.csv             # per-ticker mode only
    └── model/                  # per-ticker mode only
```

### Step 5 — Aggregate results

```python
from tsforecast.evaluation.reports import build_summary_table
df = build_summary_table(".")
print(df.sort_values("mae"))
```

---

## Configuration

### `configs/train.yaml`

Shared training settings applied to all models:

```yaml
max_epochs: 100
batch_size: 32
patience: 10
seed: 2024
```

### `configs/splits.yaml`

Defines the date boundaries for each regime:

```yaml
regimes:
  bear:
    test_start: "2022-03-11"
    val_days: 126
    test_days: 126
  bull:
    test_start: "2023-01-06"
    val_days: 252
    test_days: 126
```

### `configs/sweep/grid.yaml`

Defines the full sweep axes and per-model hyperparameter grids:

```yaml
regimes: [bear, bull]
horizons: [10, 21, 63]
context_lengths: [32, 48, 96]
strategies: [mimo]
steps: [16]       # block sizes for recursive strategy only
seeds: [2024]
training_mode: [per_ticker, pooled]

models:
  rf:
    n_estimators: [200]
    max_depth: [null, 10]
    max_features: ["sqrt"]
    min_samples_leaf: [1, 2]
  lstm:
    hidden_size: [64, 128]
    num_layers: [1, 2]
    dropout: [0.1]
    lr: [0.001]
    batch_size: [32]
  patchtst:
    d_model: [64, 128]
    num_attention_heads: [4, 8]
    num_hidden_layers: [2, 3]
    patch_length: [16]
    patch_stride: [8]
    dropout: [0.2]
    lr: [0.0015, 0.0003]
    use_revin: [true]
```

`make_grid.py` generates all Cartesian product combinations. Each MIMO row sets `step=0`; recursive rows generate one entry per value in `steps`.

### `--hparams` overrides

Any key passed via `--hparams` takes precedence over the YAML configs. `null` values become `None` in Python:

```bash
python -m tsforecast.cli.train --model rf --regime bear --L 96 --H 21 \
  --hparams '{"n_estimators": 200, "max_depth": null}'
```

---

## Run Outputs

| File | Contents |
|---|---|
| `config.yaml` | Full merged config for the run |
| `metrics.json` | `{mae, rmse, mape, smape, directional_accuracy}` |
| `metrics_global.csv` | Same metrics in CSV format |
| `predictions_detailed.csv` | Per-window predictions (pooled mode) |
| `training_curves.png` | Train/val loss curves (LSTM and PatchTST only) |
| `tickers/<T>/plot.png` | Forecast plot for ticker T |
| `tickers/<T>/plot_returns.png` | Return plot for ticker T |
| `tickers/<T>/predictions.csv` | Per-ticker predictions (per-ticker mode) |
| `tickers/<T>/metrics.csv` | Per-ticker metrics (per-ticker mode) |
| `model/` | Model checkpoint (pooled mode) |
| `tickers/<T>/model/` | Model checkpoint (per-ticker mode) |
| `logs.txt` | Training log |

---

## HPC / Slurm

### Scripts overview

| Script | When to use |
|---|---|
| `run_sweep.py` | Run all experiments for one model (recommended) |
| `run_one.sh` | Submit or run a single hand-crafted experiment |
| `run_array.sh` | Called by Slurm automatically — do not invoke directly |
| `run_config.py` | Debug a single JSON config locally |
| `make_grid.py` | Regenerate the full `grid.jsonl` manually |

### Full model sweep (recommended)

```bash
# 1. Edit the sweep config
#    configs/sweep/grid.yaml

# 2. Preview what will run (optional)
python scripts/run_sweep.py --model rf --dry-run

# 3. Submit to the cluster
python scripts/run_sweep.py --model rf
python scripts/run_sweep.py --model lstm
python scripts/run_sweep.py --model patchtst
```

`run_sweep.py` generates `scripts/grid_{model}.jsonl` and submits a Slurm array job where each task runs one experiment. Each task reads its JSON config line by `SLURM_ARRAY_TASK_ID` and calls `run_config.py → train.py`.

### Single experiment

```bash
# Submit to Slurm
sbatch scripts/slurm/run_one.sh rf bear 96 21
sbatch scripts/slurm/run_one.sh lstm bull 48 63 --strategy recursive --step 16

# Run locally (without sbatch)
bash scripts/slurm/run_one.sh rf bear 96 21
```

### Rerun failed tasks

```bash
sbatch --array=3,7 --export=ALL,GRID_FILE=scripts/grid_rf.jsonl scripts/slurm/run_array.sh
```

### Run locally (no cluster)

```bash
# Sequential sweep
python scripts/run_sweep.py --model rf --local

# Single config debug
python scripts/run_config.py '{"model":"rf","regime":"bear","L":96,"H":21,"strategy":"mimo","step":0,"seed":2024,"n_estimators":200,"max_depth":null}'
```

### Monitor jobs

```bash
squeue -u $USER
tail -f logs/slurm_<jobid>_<task>.out
```

> **Prerequisite:** The conda environment must be named `tsforecast-env` and the package must be installed with `pip install -e .` inside it.

---

## Tests

```bash
# Fast tests (recommended)
pytest tests/ -m "not slow"

# All tests including slow PatchTST smoke test
pytest tests/
```

| Test file | What it covers |
|---|---|
| `test_windows.py` | Window shapes, no data leakage, anchor values, dtype |
| `test_splits.py` | Train/val/test non-overlap, context boundary preservation |
| `test_metrics.py` | MAE/RMSE/MAPE/SMAPE/directional accuracy against known inputs |
| `test_models_smoke.py` | Fit + predict + save + load for RF, LSTM, PatchTST |
| `test_recursive.py` | Block decomposition, prediction shape, no future leakage |
| `test_cache_key.py` | Cache key uniqueness and round-trip load |

---

## Aggregating Results

```python
from tsforecast.evaluation.reports import build_summary_table

df = build_summary_table(".")
print(df.sort_values("mae"))

# Filter by model, regime, or strategy
rf_bear = df[(df["model"] == "RF") & (df["regime"] == "bear")]
mimo_results = df[df["strategy"] == "mimo"]
```

The DataFrame has columns: `run_id`, `model`, `strategy`, `regime`, `L`, `H`, and one column per metric. Runs with missing or corrupted `metrics.json` are skipped with a warning.
