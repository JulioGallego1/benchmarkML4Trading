from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tsforecast.utils.paths import get_runs_dir

logger = logging.getLogger(__name__)


class RunTracker:
    def __init__(self, run_id: str, base_dir: str | Path = ".") -> None:
        self.run_id = run_id
        self.run_dir = Path(get_runs_dir(str(base_dir))) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_file(self) -> Path:
        return self.run_dir / "logs.txt"

    @property
    def config_file(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def metrics_file(self) -> Path:
        """Path to metrics.json (read by reports.py)."""
        return self.run_dir / "metrics.json"

    @property
    def predictions_file(self) -> Path:
        return self.run_dir / "predictions_detailed.csv"

    def ticker_dir(self, ticker: str) -> Path:
        """Return (and create) the per-ticker subdirectory: tickers/{ticker}/."""
        d = self.run_dir / "tickers" / ticker
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ticker_plot_path(self, ticker: str) -> Path:
        """Return the path where the forecast plot for *ticker* will be saved."""
        return self.ticker_dir(ticker) / "plot.png"

    def ticker_return_plot_path(self, ticker: str) -> Path:
        """Return the path where the return plot for *ticker* will be saved."""
        return self.ticker_dir(ticker) / "plot_returns.png"

    # Config
    def save_config(self, config: dict) -> None:
        """Write config dict to config.yaml (atomic write)."""
        tmp = Path(str(self.config_file) + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, default_flow_style=False, allow_unicode=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.config_file)
        except OSError as exc:
            logger.error(f"Failed to save config to {self.config_file}: {exc}")
            tmp.unlink(missing_ok=True)
            raise

    # Global metrics
    def save_global_metrics(self, metrics: dict) -> None:
        """Save global run metrics as metrics_global.csv and metrics.json.

        metrics_global.csv  — human-readable CSV summary.
        metrics.json        — read by reports.py to build summary tables.
        """
        rounded = {
            k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        }

        # --- metrics_global.csv ---
        csv_dest = self.run_dir / "metrics_global.csv"
        tmp_csv = Path(str(csv_dest) + ".tmp")
        try:
            pd.DataFrame([rounded]).to_csv(tmp_csv, index=False)
            os.replace(tmp_csv, csv_dest)
        except OSError as exc:
            logger.error(f"Failed to save global metrics CSV to {csv_dest}: {exc}")
            tmp_csv.unlink(missing_ok=True)
            raise

        # --- metrics.json (reports.py compatibility) ---
        json_dest = self.metrics_file
        tmp_json = Path(str(json_dest) + ".tmp")
        try:
            with open(tmp_json, "w", encoding="utf-8") as f:
                f.write(json.dumps(rounded, indent=2))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_json, json_dest)
        except OSError as exc:
            logger.error(f"Failed to save metrics.json to {json_dest}: {exc}")
            tmp_json.unlink(missing_ok=True)
            raise


    # Per-ticker artifacts
    def save_ticker_metrics(self, ticker: str, metrics: dict) -> None:
        """Write per-ticker metrics to tickers/{ticker}/metrics.csv (atomic write)."""
        dest = self.ticker_dir(ticker) / "metrics.csv"
        rounded = {
            k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
        }
        tmp = Path(str(dest) + ".tmp")
        try:
            pd.DataFrame([rounded]).to_csv(tmp, index=False)
            os.replace(tmp, dest)
        except OSError as exc:
            logger.error(f"Failed to save ticker metrics to {dest}: {exc}")
            tmp.unlink(missing_ok=True)
            raise

    def save_ticker_predictions(
        self,
        ticker: str,
        dates: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anchors: np.ndarray,
    ) -> None:
        """Save per-ticker predictions to tickers/{ticker}/predictions.csv (atomic write)."""
        H = y_true.shape[1]
        date_strings = pd.to_datetime(dates).strftime("%Y-%m-%d")
        data: dict = {"date": date_strings, "ticker": ticker, "anchor": anchors}
        for h in range(H):
            data[f"y_true_{h}"] = y_true[:, h]
        for h in range(H):
            data[f"y_pred_{h}"] = y_pred[:, h]
        dest = self.ticker_dir(ticker) / "predictions.csv"
        tmp = Path(str(dest) + ".tmp")
        try:
            pd.DataFrame(data).to_csv(tmp, index=False)
            os.replace(tmp, dest)
        except OSError as exc:
            logger.error(f"Failed to save predictions to {dest}: {exc}")
            tmp.unlink(missing_ok=True)
            raise

    def save_ticker_model(self, model, ticker: str) -> Path:
        """Call model.save(tickers/{ticker}/model) and return the path."""
        path = self.ticker_dir(ticker) / "model"
        model.save(path)
        return path

    def save_predictions(
        self,
        dates: np.ndarray,
        ticker: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anchors: np.ndarray,
    ) -> None:
        """Save predictions_detailed.csv.

        Columns: date, ticker, anchor, y_true_0..y_true_{H-1}, y_pred_0..y_pred_{H-1}
        """
        H = y_true.shape[1]

        date_strings = pd.to_datetime(dates).strftime("%Y-%m-%d")

        data: dict = {
            "date": date_strings,
            "ticker": ticker,
            "anchor": anchors,
        }

        for h in range(H):
            data[f"y_true_{h}"] = y_true[:, h]

        for h in range(H):
            data[f"y_pred_{h}"] = y_pred[:, h]

        df = pd.DataFrame(data)
        tmp = Path(str(self.predictions_file) + ".tmp")
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, self.predictions_file)
        except OSError as exc:
            logger.error(f"Failed to save predictions to {self.predictions_file}: {exc}")
            tmp.unlink(missing_ok=True)
            raise

    def save_model(self, model, subdir: str = "model") -> Path:
        """Call model.save(run_dir / subdir) and return the path."""
        path = self.run_dir / subdir
        model.save(path)
        return path
