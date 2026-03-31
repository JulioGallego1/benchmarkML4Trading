"""
Main CLI entrypoint for the tsforecast pipeline.

Usage:
    python -m tsforecast.cli.train --model rf --regime bear --L 96 --H 21 --training-mode pooled
    tsforecast-train --model lstm --regime bull --L 48 --H 63 --training-mode per_ticker
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import pandas as pd
from tsforecast.tracking.run_id import make_run_id
from tsforecast.tracking.filesystem import RunTracker
from tsforecast.utils.logging import get_logger
from tsforecast.training.reproducibility import set_seed

from tsforecast.evaluation.metrics import (
        directional_accuracy,
        mae,
        mape,
        rmse,
        smape,
    )

from tsforecast.evaluation.plots import plot_ticker_forecast, plot_ticker_returns, plot_training_curves

from tsforecast.data.cache import WindowCache
from tsforecast.data.loaders import load_price_data
from tsforecast.data.splits import make_time_splits
from tsforecast.data.windows import generate_windows_mimo

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a forecasting model for the tsforecast pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["rf", "lstm", "patchtst"],
        required=True,
        help="Model to train.",
    )
    parser.add_argument(
        "--regime",
        choices=["bear", "bull"],
        required=True,
        help="Market regime to use.",
    )
    parser.add_argument(
        "--L",
        type=int,
        required=True,
        help="Context length in trading days.",
    )
    parser.add_argument(
        "--H",
        type=int,
        required=True,
        help="Forecast horizon in trading days.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["pooled", "per_ticker"],
        default="per_ticker",
        help=(
            "Training mode. "
            "'pooled': concatenate all ticker windows and train one shared model. "
            "'per_ticker': train a separate model for each ticker independently. "
            "(default: per_ticker)"
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["mimo", "recursive"],
        default="mimo",
        help="Forecasting strategy: mimo (direct) or recursive (iterative in --step blocks).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=16,
        help="Block size for recursive strategy (ignored if strategy=mimo).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing parquet files.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing YAML config files.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project root for runs/ output.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache even if a cached result exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override. Overrides the seed value from YAML configs.",
    )
    parser.add_argument(
        "--hparams",
        default=None,
        help=(
            "JSON string of hyperparameter overrides. Keys override values from YAML "
            "configs (e.g., '{\"n_estimators\": 200, \"max_depth\": 10}'). "
            "Used by sweep/Slurm launchers to inject hyperparameters."
        ),
    )
    return parser.parse_args()


def load_configs(config_dir: Path, model: str, regime: str) -> dict:
    """Load and merge model, splits, and train YAML configs."""
    model_cfg_path = config_dir / "model" / f"{model}.yaml"
    splits_cfg_path = config_dir / "splits.yaml"
    train_cfg_path = config_dir / "train.yaml"

    config = {}

    if train_cfg_path.exists():
        with open(train_cfg_path) as f:
            train_cfg = yaml.safe_load(f) or {}
        config.update(train_cfg)

    if model_cfg_path.exists():
        with open(model_cfg_path) as f:
            model_cfg = yaml.safe_load(f) or {}
        config.update(model_cfg)

    if splits_cfg_path.exists():
        with open(splits_cfg_path) as f:
            splits_cfg = yaml.safe_load(f) or {}
        regimes = splits_cfg.get("regimes", {})
        regime_cfg = regimes.get(regime, {})
        config.update(regime_cfg)

    return config


def build_model(
    model_name: str,
    config: dict,
    L: int,
    H: int,
    run_dir: Path,
    strategy: str = "mimo",
    step: int = 16,
    num_tickers: int = 1,
):
    """Instantiate the requested model using config parameters.

    Parameters
    ----------
    num_tickers : int
        Number of unique tickers seen in pooled mode.  Pass ``1`` (default)
        for per-ticker mode — neural models will skip the embedding path
        entirely and RF will not append a ticker-ID column.
    """
    if model_name == "rf":
        from tsforecast.models.rf import RandomForestModel
        return RandomForestModel(
            n_estimators=config.get("n_estimators", 200),
            max_features=config.get("max_features", "sqrt"),
            max_depth=config.get("max_depth", None),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            random_state=config.get("seed", 2024),
        )

    elif model_name == "lstm":
        from tsforecast.models.lstm import LSTMModel

        return LSTMModel(
            context_length=L,
            horizon=H,
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            lr=config.get("lr", 1e-3),
            max_epochs=config.get("max_epochs", 100),
            batch_size=config.get("batch_size", 32),
            patience=config.get("patience", 10),
            random_state=config.get("seed", 2024),
            strategy=strategy,
            step=step,
            num_tickers=num_tickers,
            ticker_embedding_dim=config.get("ticker_embedding_dim", 4),
        )

    elif model_name == "patchtst":
        from tsforecast.models.patchtst import PatchTSTModel

        output_dir = str(run_dir / "patchtst_tmp")
        return PatchTSTModel(
            context_length=L,
            horizon=H,
            patch_length=config.get("patch_length", 16),
            patch_stride=config.get("patch_stride", 8),
            d_model=config.get("d_model", 128),
            num_attention_heads=config.get("num_attention_heads", 4),
            num_hidden_layers=config.get("num_hidden_layers", 3),
            ffn_dim=config.get("ffn_dim", 256),
            dropout=config.get("dropout", 0.2),
            lr=config.get("lr", 1e-4),
            max_epochs=config.get("max_epochs", 100),
            batch_size=config.get("batch_size", 32),
            patience=config.get("patience", 10),
            random_state=config.get("seed", 2024),
            output_dir=output_dir,
            strategy=strategy,
            step=step,
            use_revin=config.get("use_revin", True),
            num_tickers=num_tickers,
            ticker_embedding_dim=config.get("ticker_embedding_dim", 4),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def process_ticker(
    parquet_path: Path,
    config: dict,
    L: int,
    H: int,
    use_cache: bool,
    cache_dir: Path,
    regime: str,
    logger,
    strategy: str = "mimo",
    step: int = 16,
) -> dict | None:
    """
    Load, split, window, and cache data for a single ticker file.

    Returns a dict with keys:
        ticker,
        X_train, Y_train, anchors_train, dates_train,
        X_val,   Y_val,   anchors_val,   dates_val,
        X_test,  Y_test,  anchors_test,  dates_test,
    or None if the ticker cannot be processed.
    """
    ticker = parquet_path.stem
    test_start = config["test_start"]
    test_days = config.get("test_days", 252)
    val_days = config.get("val_days", 252)

    cache = WindowCache(cache_dir=str(cache_dir))

    # Try to load from cache
    if use_cache and cache.exists(ticker, regime, L, H, strategy, step):
        logger.info(f"  [{ticker}] Loading from cache.")
        cached = cache.load(ticker, regime, L, H, strategy, step)
        X_train = cached["X_train"]
        Y_train = cached["Y_train"]
        anchors_train = cached["anchors_train"]
        dates_train = cached["dates_train"]
        X_val = cached["X_val"]
        Y_val = cached["Y_val"]
        anchors_val = cached["anchors_val"]
        dates_val = cached["dates_val"]
        X_test = cached["X_test"]
        Y_test = cached["Y_test"]
        anchors_test = cached["anchors_test"]
        dates_test = cached["dates_test"]
    else:
        logger.info(f"  [{ticker}] Generating windows.")
        try:
            df = load_price_data(parquet_path)
        except Exception as exc:
            logger.warning(f"  [{ticker}] Failed to load: {exc}")
            return None

        values = df["Close"].values.astype(np.float32)
        dates_arr = df["Date"].values

        df_work = pd.DataFrame({"Date": dates_arr, "Close": values})
        df_work["Date"] = pd.to_datetime(df_work["Date"])

        # MAKE SPLITS
        try:
            splits = make_time_splits(
                df_work,
                test_start=test_start,
                test_days=test_days,
                val_days=val_days,
                context_length=L,
            )
        except Exception as exc:
            logger.warning(f"  [{ticker}] make_time_splits failed: {exc}")
            return None

        train_start, train_end = splits["train"]
        val_start, val_end = splits["val"]
        test_start_idx, test_end_idx = splits["test"]

        # GENERATE WINDOWS

        try:
            X_train, Y_train, anchors_train, dates_train = generate_windows_mimo(
                values, dates_arr, train_start, train_end, L, H
            )
            X_val, Y_val, anchors_val, dates_val = generate_windows_mimo(
                values, dates_arr, val_start, val_end, L, H
            )
            X_test, Y_test, anchors_test, dates_test = generate_windows_mimo(
                values, dates_arr, test_start_idx, test_end_idx, L, H
            )
        except Exception as exc:
            logger.warning(f"  [{ticker}] Window generation failed: {exc}")
            return None

        if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
            logger.warning(f"  [{ticker}] Empty split — skipping.")
            return None

        try:
            cache.save(
                ticker,
                regime,
                L,
                H,
                strategy,
                step,
                X_train,
                Y_train,
                anchors_train,
                dates_train,
                X_val,
                Y_val,
                anchors_val,
                dates_val,
                X_test,
                Y_test,
                anchors_test,
                dates_test,
            )
        except Exception as exc:
            logger.warning(f"  [{ticker}] Cache save failed: {exc}")

    return {
        "ticker": ticker,
        "X_train": X_train,
        "Y_train": Y_train,
        "anchors_train": anchors_train,
        "dates_train": dates_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "anchors_val": anchors_val,
        "dates_val": dates_val,
        "X_test": X_test,
        "Y_test": Y_test,
        "anchors_test": anchors_test,
        "dates_test": dates_test,
    }


def _aggregate_ticker_metrics(ticker_metrics: list[dict]) -> dict:
    """Compute arithmetic mean of numeric metrics across tickers."""
    if not ticker_metrics:
        return {}
    all_keys: set[str] = set()
    for m in ticker_metrics:
        all_keys.update(m.keys())
    result: dict = {}
    for key in sorted(all_keys):
        values = [
            m[key]
            for m in ticker_metrics
            if key in m and isinstance(m[key], (int, float))
        ]
        if values:
            result[key] = float(np.mean(values))
    return result


def _run_pooled(
    all_results: list[dict],
    tracker,
    model_name: str,
    config: dict,
    L: int,
    H: int,
    strategy: str,
    step: int,
    logger,
) -> tuple[dict, dict]:
    """Train one ticker-aware shared model on the concatenation of all ticker windows.

    Each sample carries its ticker's integer ID so the model can condition
    predictions on asset identity while still sharing the backbone weights
    across all tickers.

    Returns
    -------
    metrics : dict
        Global test-set metrics.
    ticker_to_id : dict[str, int]
        Mapping of ticker name → integer ID used during this run.  Saved in
        config.yaml so the mapping is reproducible at inference time.

    Saves:
      runs/<run_id>/model/                    — the shared (ticker-aware) model
      runs/<run_id>/predictions_detailed.csv  — all tickers concatenated
      runs/<run_id>/metrics.json              — pooled test-set metrics
      runs/<run_id>/metrics_global.csv        — same metrics in CSV form
      runs/<run_id>/tickers/<T>/plot.png      — per-ticker forecast plot
    """
    # Build a ticker-integer ID mapping

    tickers_sorted = sorted(r["ticker"] for r in all_results)
    ticker_to_id: dict[str, int] = {t: i for i, t in enumerate(tickers_sorted)}
    num_tickers = len(ticker_to_id)
    logger.info(f"Ticker-ID mapping ({num_tickers} tickers): {ticker_to_id}")

    # Concatenate windows and build matching ticker-ID arrays
    X_train_all = np.concatenate([r["X_train"] for r in all_results], axis=0)
    Y_train_all = np.concatenate([r["Y_train"] for r in all_results], axis=0)
    X_val_all   = np.concatenate([r["X_val"]   for r in all_results], axis=0)
    Y_val_all   = np.concatenate([r["Y_val"]   for r in all_results], axis=0)
    X_test_all  = np.concatenate([r["X_test"]  for r in all_results], axis=0)
    Y_test_all  = np.concatenate([r["Y_test"]  for r in all_results], axis=0)
    anchors_test_all = np.concatenate([r["anchors_test"] for r in all_results], axis=0)
    dates_test_all   = np.concatenate([r["dates_test"]   for r in all_results], axis=0)

    def _make_ids(split: str) -> np.ndarray:
        return np.concatenate(
            [
                np.full(r[f"X_{split}"].shape[0], ticker_to_id[r["ticker"]], dtype=np.int64)
                for r in all_results
            ]
        )

    T_train_all = _make_ids("train")
    T_val_all   = _make_ids("val")
    T_test_all  = _make_ids("test")

    logger.info(
        f"Pooled shapes — X_train: {X_train_all.shape}, "
        f"X_val: {X_val_all.shape}, X_test: {X_test_all.shape}"
    )

    # Train
    model = build_model(
        model_name, config, L, H, tracker.run_dir,
        strategy=strategy, step=step, num_tickers=num_tickers,
    )
    logger.info(
        f"Training {type(model).__name__} on pooled data "
        f"({num_tickers} tickers, ticker_embedding_dim="
        f"{config.get('ticker_embedding_dim', 4)})..."
    )
    model.fit(
        X_train_all, Y_train_all, X_val_all, Y_val_all,
        ticker_ids_train=T_train_all,
        ticker_ids_val=T_val_all,
    )
    logger.info("Training complete.")
    if getattr(model, "history", None):
        plot_training_curves(model.history, save_path=tracker.run_dir / "training_curves.png")
        logger.info("Training curves saved.")

    y_pred = model.predict(X_test_all, ticker_ids=T_test_all)

    # Per-ticker forecast plots
    offset = 0
    for r in all_results:
        n = r["X_test"].shape[0]
        ticker = r["ticker"]
        plot_ticker_forecast(
            dates=r["dates_test"],
            y_true=Y_test_all[offset : offset + n],
            y_pred=y_pred[offset : offset + n],
            ticker=ticker,
            save_path=tracker.ticker_plot_path(ticker),
        )
        plot_ticker_returns(
            dates=r["dates_test"],
            y_true=Y_test_all[offset : offset + n],
            y_pred=y_pred[offset : offset + n],
            anchors=r["anchors_test"],
            ticker=ticker,
            save_path=tracker.ticker_return_plot_path(ticker),
        )
        logger.info(f"  [{ticker}] Forecast and return plots saved.")
        offset += n

    metrics = {
        "mae": mae(Y_test_all, y_pred),
        "rmse": rmse(Y_test_all, y_pred),
        "mape": mape(Y_test_all, y_pred),
        "smape": smape(Y_test_all, y_pred),
        "directional_accuracy": directional_accuracy(
            Y_test_all, y_pred, anchors_test_all
        ),
        "n_tickers_ok": len(all_results),
        "n_tickers_failed": 0,
    }

    tracker.save_global_metrics(metrics)
    tracker.save_predictions(
        dates=dates_test_all,
        ticker="all",
        y_true=Y_test_all,
        y_pred=y_pred,
        anchors=anchors_test_all,
    )
    tracker.save_model(model)

    return metrics, ticker_to_id


def _run_per_ticker(
    all_results: list[dict],
    tracker,
    model_name: str,
    config: dict,
    L: int,
    H: int,
    strategy: str,
    step: int,
    logger,
) -> tuple[dict, list[str]]:
    """Train one independent model per ticker and aggregate metrics.

    Saves per ticker inside runs/<run_id>/tickers/<T>/:
      metrics.csv, predictions.csv, plot.png, model/

    Also saves:
      runs/<run_id>/metrics_global.csv  — mean across successful tickers
      runs/<run_id>/metrics.json        — same (reports.py compat)
    """
    ticker_metrics_list: list[dict] = []
    failed_tickers: list[str] = []

    for r in all_results:
        ticker = r["ticker"]
        try:
            ticker_dir = tracker.ticker_dir(ticker)
            model = build_model(model_name, config, L, H, ticker_dir, strategy=strategy, step=step)
            logger.info(f"  [{ticker}] Training {type(model).__name__}...")
            model.fit(r["X_train"], r["Y_train"], r["X_val"], r["Y_val"])
            logger.info(f"  [{ticker}] Training complete.")
            if getattr(model, "history", None):
                plot_training_curves(
                    model.history,
                    save_path=tracker.ticker_dir(ticker) / "training_curves.png",
                )
                logger.info(f"  [{ticker}] Training curves saved.")

            y_pred = model.predict(r["X_test"])

            ticker_metrics = {
                "mae": mae(r["Y_test"], y_pred),
                "rmse": rmse(r["Y_test"], y_pred),
                "mape": mape(r["Y_test"], y_pred),
                "smape": smape(r["Y_test"], y_pred),
                "directional_accuracy": directional_accuracy(
                    r["Y_test"], y_pred, r["anchors_test"]
                ),
            }

            tracker.save_ticker_metrics(ticker, ticker_metrics)
            tracker.save_ticker_predictions(
                ticker=ticker,
                dates=r["dates_test"],
                y_true=r["Y_test"],
                y_pred=y_pred,
                anchors=r["anchors_test"],
            )
            plot_ticker_forecast(
                dates=r["dates_test"],
                y_true=r["Y_test"],
                y_pred=y_pred,
                ticker=ticker,
                save_path=tracker.ticker_plot_path(ticker),
            )
            plot_ticker_returns(
                dates=r["dates_test"],
                y_true=r["Y_test"],
                y_pred=y_pred,
                anchors=r["anchors_test"],
                ticker=ticker,
                save_path=tracker.ticker_return_plot_path(ticker),
            )
            tracker.save_ticker_model(model, ticker)

            ticker_metrics_list.append(ticker_metrics)
            logger.info(
                f"  [{ticker}] MAE={ticker_metrics['mae']:.4f}  "
                f"RMSE={ticker_metrics['rmse']:.4f}  "
                f"MAPE={ticker_metrics['mape']:.4f}  "
                f"SMAPE={ticker_metrics['smape']:.4f}  "
                f"Dir={ticker_metrics['directional_accuracy']:.2f}%"
            )

        except Exception as exc:
            logger.error(
                f"  [{ticker}] Failed during training/evaluation: {exc}",
                exc_info=True,
            )
            failed_tickers.append(ticker)

    if not ticker_metrics_list:
        return {}, failed_tickers

    global_metrics = _aggregate_ticker_metrics(ticker_metrics_list)
    global_metrics["n_tickers_ok"] = len(ticker_metrics_list)
    global_metrics["n_tickers_failed"] = len(failed_tickers)

    tracker.save_global_metrics(global_metrics)

    return global_metrics, failed_tickers


def main():
    args = parse_args()

    model_name = args.model
    regime = args.regime
    L = args.L
    H = args.H
    strategy = args.strategy
    step = args.step
    training_mode = args.training_mode
    data_dir = Path(args.data_dir)
    config_dir = Path(args.config_dir)
    base_dir = Path(args.base_dir)
    use_cache = not args.no_cache

    if strategy == "recursive" and step <= 0:
        print("ERROR: --step must be a positive integer when --strategy=recursive", file=sys.stderr)
        sys.exit(1)
    if strategy == "mimo":
        step = 0

    config = load_configs(config_dir, model_name, regime)

    if args.hparams:
        import json as _json
        try:
            hparam_overrides = _json.loads(args.hparams)
        except Exception as exc:
            print(f"ERROR: --hparams is not valid JSON: {exc}", file=sys.stderr)
            sys.exit(1)
        config.update(hparam_overrides)

    if args.seed is not None:
        config["seed"] = args.seed

    extra_tags: list[str] = []
    if model_name == "patchtst":
        use_revin = config.get("use_revin", True)
        extra_tags.append("revin" if use_revin else "norevin")

    run_id = make_run_id(model_name, regime, L, H, strategy=strategy, step=step, training_mode=training_mode, extra_tags=extra_tags or None)

    tracker = RunTracker(run_id, base_dir=str(base_dir))
    logger = get_logger("tsforecast.train", log_file=tracker.log_file)

    set_seed(config.get("seed", 2024))

    logger.info(
        f"Starting run {run_id} | model={model_name} regime={regime} "
        f"L={L} H={H} strategy={strategy} step={step} training_mode={training_mode} (raw prices)"
    )

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in '{data_dir}'. Nothing to do — exiting.")
        sys.exit(0)

    logger.info(f"Found {len(parquet_files)} parquet file(s) in '{data_dir}'.")

    cache_dir = base_dir / "data" / "processed"

    # Preprocessing
    all_results: list[dict] = []
    failed_load: list[str] = []

    for pf in parquet_files:
        ticker = pf.stem
        logger.info(f"Preprocessing ticker: {ticker}")
        result = process_ticker(
            parquet_path=pf,
            config=config,
            L=L,
            H=H,
            use_cache=use_cache,
            cache_dir=cache_dir,
            regime=regime,
            logger=logger,
            strategy=strategy,
            step=step,
        )
        if result is None:
            failed_load.append(ticker)
        else:
            all_results.append(result)

    if not all_results:
        logger.warning("No valid tickers could be processed. Exiting.")
        sys.exit(0)

    logger.info(f"Preprocessed {len(all_results)} ticker(s) successfully.")

    # Training and evaluation
    if training_mode == "pooled":
        global_metrics, ticker_to_id = _run_pooled(
            all_results, tracker, model_name, config, L, H, strategy, step, logger
        )
        failed_tickers = failed_load

    else:  # per_ticker
        global_metrics, failed_train = _run_per_ticker(
            all_results, tracker, model_name, config, L, H, strategy, step, logger
        )
        ticker_to_id = {}
        failed_tickers = failed_load + failed_train

        if not global_metrics:
            logger.warning("No tickers completed training successfully. No global metrics to save.")
            sys.exit(1)

    if failed_tickers:
        logger.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

    #Save config
    full_config = {
        **config,
        "model": model_name,
        "regime": regime,
        "L": L,
        "H": H,
        "run_id": run_id,
        "strategy": strategy,
        "step": step,
        "training_mode": training_mode,
        "tickers_ok": global_metrics.get("n_tickers_ok", 0),
        "tickers_failed": global_metrics.get("n_tickers_failed", 0),
        "ticker_to_id": ticker_to_id,
    }
    tracker.save_config(full_config)

    logger.info(
        f"Run {run_id} complete — "
        f"{global_metrics.get('n_tickers_ok', 0)} tickers ok, "
        f"{global_metrics.get('n_tickers_failed', 0)} failed."
    )
    logger.info(f"  Global MAE:   {global_metrics['mae']:.4f}")
    logger.info(f"  Global RMSE:  {global_metrics['rmse']:.4f}")
    logger.info(f"  Global MAPE:  {global_metrics['mape']:.4f}")
    logger.info(f"  Global SMAPE: {global_metrics['smape']:.4f}")
    logger.info(f"  Global Dir:   {global_metrics['directional_accuracy']:.2f}%")
    logger.info(f"  Saved to: {tracker.run_dir}")


if __name__ == "__main__":
    main()
