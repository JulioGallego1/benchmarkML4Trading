#!/usr/bin/env python3
"""
Best-of-Family model analysis.

For each (regime, horizon), selects the single best run per model family
independently for MAPE (lower = better) and Directional Accuracy (higher = better).

Model families:
  - RF
  - LSTM_mimo
  - LSTM_recursive
  - PATCHTST_mimo
  - PATCHTST_recursive

No averages are used. Selection is strict: best individual run per family.

Output structure:
  analysis/
  ├── best_of_family/
  │   ├── bear_H21/
  │   │   ├── best_by_mape.csv
  │   │   ├── best_by_da.csv
  │   │   ├── mape_comparison.png
  │   │   ├── da_comparison.png
  │   │   ├── table_best_mape.png
  │   │   └── table_best_da.png
  │   └── overview/
  │       ├── heatmap_mape.png
  │       ├── heatmap_da.png
  │       ├── best_by_mape_all.csv
  │       └── best_by_da_all.csv
  ├── lines/
  │   ├── mape_bear.png  mape_bull.png  mape_dual.png
  │   └── da_bear.png    da_bull.png    da_dual.png
  ├── best_selected/
  │   ├── by_mape/  (symlinks to best run dirs, named regime_H_family)
  │   └── by_da/    (symlinks to best run dirs, named regime_H_family)
  ├── all_runs_detail.csv
  └── summary_report.txt

Usage:
    python analyze_results.py
    python analyze_results.py --runs-dir runs --out-dir analysis
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ── Family definitions ────────────────────────────────────────────────────────

FAMILY_ORDER = ["RF", "LSTM_mimo", "LSTM_recursive", "PATCHTST_mimo", "PATCHTST_recursive"]

FAMILY_LABELS = {
    "RF":                 "Random Forest",
    "LSTM_mimo":          "LSTM — MIMO",
    "LSTM_recursive":     "LSTM — Recursive",
    "PATCHTST_mimo":      "Transformer — MIMO",
    "PATCHTST_recursive": "Transformer — Recursive",
}

FAMILY_COLORS = {
    "RF":                 "#2E86AB",
    "LSTM_mimo":          "#F18F01",
    "LSTM_recursive":     "#44BBA4",
    "PATCHTST_mimo":      "#E94F37",
    "PATCHTST_recursive": "#8963BA",
}

FAMILY_MARKERS = {
    "RF":                 "o",
    "LSTM_mimo":          "s",
    "LSTM_recursive":     "^",
    "PATCHTST_mimo":      "D",
    "PATCHTST_recursive": "P",
}


# ── Visual config ─────────────────────────────────────────────────────────────

BG_COLOR    = "#FAFAFA"
GRID_COLOR  = "#E0E0E0"
TEXT_COLOR  = "#2D2D2D"
ACCENT_GOLD = "#FFB703"
ACCENT_GREEN = "#2A9D8F"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor":   "#FFFFFF",
        "axes.edgecolor":   "#CCCCCC",
        "axes.grid":        True,
        "grid.color":       GRID_COLOR,
        "grid.alpha":       0.5,
        "grid.linewidth":   0.5,
        "font.family":      "sans-serif",
        "font.sans-serif":  ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size":        11,
        "axes.titlesize":   14,
        "axes.titleweight": "bold",
        "axes.labelsize":   12,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  9,
        "figure.dpi":       150,
        "savefig.dpi":      150,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.3,
    })


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_runs(runs_dir: str = "runs") -> pd.DataFrame:
    runs_path = Path(runs_dir)
    records = []
    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        metrics_file = run_dir / "metrics.json"
        config_file  = run_dir / "config.yaml"
        if not metrics_file.exists():
            continue
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        config = {}
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f) or {}
            except (yaml.YAMLError, OSError):
                pass
        records.append({
            "run_id":        run_dir.name,
            "run_dir":       str(run_dir),
            "model":         config.get("model", "unknown").upper(),
            "regime":        config.get("regime", "unknown"),
            "L":             config.get("L", 0),
            "H":             config.get("H", 0),
            "strategy":      config.get("strategy", "mimo"),
            "patch_length":  config.get("patch_length", None),
            "rec_step":      config.get("step", 0) or 0,
            "use_revin":     config.get("use_revin", False),
            "mape":          metrics.get("mape", None),
            "smape":         metrics.get("smape", None),
            "mae":           metrics.get("mae", None),
            "rmse":          metrics.get("rmse", None),
            "directional_accuracy": metrics.get("directional_accuracy", None),
        })
    return pd.DataFrame(records) if records else pd.DataFrame()


def build_family_label(row) -> str:
    """5 families: RF | LSTM_mimo | LSTM_recursive | PATCHTST_mimo | PATCHTST_recursive."""
    model = row["model"]
    strategy = row["strategy"]
    if model == "RF":
        return "RF"
    return f"{model}_{strategy}"


def get_color(family: str) -> str:
    return FAMILY_COLORS.get(family, "#999999")


def short_family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family)


def variant_label(row) -> str:
    """Compact description of the hyperparameter variant within a family."""
    parts = [f"L={int(row['L'])}"]
    if row.get("patch_length") is not None and not (
        isinstance(row["patch_length"], float) and np.isnan(row["patch_length"])
    ):
        parts.append(f"pl={int(row['patch_length'])}")
    if row.get("rec_step", 0):
        parts.append(f"stp={int(row['rec_step'])}")
    if row.get("use_revin", False):
        parts.append("revin")
    return ", ".join(parts)


# ── Chart helpers ─────────────────────────────────────────────────────────────

def plot_bar_comparison(data, metric, title, ylabel, save_path, lower_is_better=True):
    if data.empty:
        return
    sorted_data = data.sort_values(metric, ascending=lower_is_better)
    families = sorted_data["family"].tolist()
    values   = sorted_data[metric].tolist()
    colors   = [get_color(f) for f in families]

    fig, ax = plt.subplots(figsize=(12, max(4, len(families) * 0.8)))
    bars = ax.barh(range(len(families)), values, color=colors, height=0.65,
                   edgecolor="white", linewidth=0.5)
    bars[0].set_edgecolor(ACCENT_GOLD)
    bars[0].set_linewidth(3)

    max_val = max(values) if values else 1
    for i, (bar, val) in enumerate(zip(bars, values)):
        offset = max_val * 0.02
        weight = "bold" if i == 0 else "normal"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=10,
                fontweight=weight, color=TEXT_COLOR)

    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([short_family_label(f) for f in families], fontsize=10)
    ax.set_xlabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=TEXT_COLOR)
    ax.text(0.98, 0.02, f"Best: {short_family_label(families[0])}\n({values[0]:.4f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=ACCENT_GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                      edgecolor=ACCENT_GREEN, alpha=0.8))
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_metrics_table(data, title, save_path):
    if data.empty:
        return
    cols     = ["family", "mape", "directional_accuracy", "mae", "rmse", "run_id"]
    headers  = ["Family", "MAPE", "Dir.Acc.", "MAE", "RMSE", "Run ID"]
    available = [c for c in cols if c in data.columns]

    display_data = []
    for _, row in data.iterrows():
        formatted = []
        for col in available:
            val = row[col]
            if col == "family":
                formatted.append(short_family_label(str(val)))
            elif col == "run_id":
                s = str(val)
                formatted.append(s[:42] + "…" if len(s) > 42 else s)
            elif isinstance(val, float):
                formatted.append(f"{val:.4f}")
            else:
                formatted.append(str(val))
        display_data.append(formatted)

    fig, ax = plt.subplots(figsize=(18, max(2, len(display_data) * 0.6 + 1.5)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20, color=TEXT_COLOR)

    hdrs  = [headers[cols.index(c)] for c in available]
    table = ax.table(cellText=display_data, colLabels=hdrs, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    for j in range(len(hdrs)):
        table[0, j].set_facecolor("#2E86AB")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(display_data)):
        color = "#F5F5F5" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(hdrs)):
            table[i + 1, j].set_facecolor(color)
    if display_data:
        for j in range(len(hdrs)):
            table[1, j].set_facecolor("#E8F5E9")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_heatmap(pivot, title, save_path, lower_is_better=True):
    if pivot.empty:
        return
    cmap = "RdYlGn_r" if lower_is_better else "RdYlGn"
    data = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2),
                                    max(5, len(pivot.index) * 0.7)))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([short_family_label(f) for f in pivot.index], fontsize=10)

    vmin, vmax = np.nanmin(data), np.nanmax(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999999")
            else:
                norm  = (val - vmin) / (vmax - vmin + 1e-10)
                color = "white" if (norm > 0.65 if lower_is_better else norm < 0.35) else TEXT_COLOR
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    for j in range(data.shape[1]):
        col   = data[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        best_i = int(np.nanargmin(col) if lower_is_better else np.nanargmax(col))
        ax.add_patch(plt.Rectangle((j - 0.5, best_i - 0.5), 1, 1,
                     fill=False, edgecolor=ACCENT_GOLD, linewidth=3))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15, color=TEXT_COLOR)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_lines_by_horizon(df, regime, metric, title, ylabel, save_path, lower_is_better=True):
    """Line chart: X=horizon, one line per family, best run per (family, H)."""
    subset = df[df["regime"] == regime].copy()
    if subset.empty:
        return

    if lower_is_better:
        best_idx = subset.groupby(["family", "H"])[metric].idxmin()
    else:
        best_idx = subset.groupby(["family", "H"])[metric].idxmax()
    best = subset.loc[best_idx]

    horizons = sorted(best["H"].unique())
    families = [f for f in FAMILY_ORDER if f in best["family"].unique()]
    families += [f for f in best["family"].unique() if f not in FAMILY_ORDER]

    fig, ax = plt.subplots(figsize=(10, 6))
    for family in families:
        fam_data = best[best["family"] == family].sort_values("H")
        if fam_data.empty:
            continue
        h_vals = fam_data["H"].tolist()
        m_vals = fam_data[metric].tolist()
        color  = get_color(family)
        marker = FAMILY_MARKERS.get(family, "o")
        label  = FAMILY_LABELS.get(family, family)
        ax.plot(h_vals, m_vals, color=color, marker=marker, markersize=8,
                linewidth=2.2, label=label, zorder=3)
        for h, m in zip(h_vals, m_vals):
            ax.annotate(f"{m:.1f}", (h, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7.5,
                        color=color, fontweight="bold")

    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("Prediction Horizon (H)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    regime_label = "Bear Market (2022)" if regime == "bear" else "Bull Market (2023)"
    ax.set_title(f"{title}\n{regime_label}", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR, pad=15)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9,
              edgecolor="#CCCCCC", fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_lines_dual_regime(df, metric, title, ylabel, save_path, lower_is_better=True):
    """Side-by-side line charts (bear | bull)."""
    regimes = sorted(df["regime"].unique())
    if len(regimes) < 2:
        plot_lines_by_horizon(df, regimes[0], metric, title, ylabel, save_path, lower_is_better)
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    families  = [f for f in FAMILY_ORDER if f in df["family"].unique()]
    families += [f for f in df["family"].unique() if f not in FAMILY_ORDER]

    for ax, regime in zip(axes, regimes):
        subset = df[df["regime"] == regime]
        if subset.empty:
            continue
        if lower_is_better:
            best_idx = subset.groupby(["family", "H"])[metric].idxmin()
        else:
            best_idx = subset.groupby(["family", "H"])[metric].idxmax()
        best     = subset.loc[best_idx]
        horizons = sorted(best["H"].unique())

        for family in families:
            fam_data = best[best["family"] == family].sort_values("H")
            if fam_data.empty:
                continue
            h_vals = fam_data["H"].tolist()
            m_vals = fam_data[metric].tolist()
            color  = get_color(family)
            marker = FAMILY_MARKERS.get(family, "o")
            label  = FAMILY_LABELS.get(family, family)
            ax.plot(h_vals, m_vals, color=color, marker=marker, markersize=8,
                    linewidth=2.2, label=label, zorder=3)

        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlabel("Prediction Horizon (H)", fontsize=11)
        regime_label = "Bear Market (2022)" if regime == "bear" else "Bull Market (2023)"
        ax.set_title(regime_label, fontsize=13, fontweight="bold", color=TEXT_COLOR)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(ylabel, fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               framealpha=0.95, edgecolor="#CCCCCC", fancybox=True,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_final_comparison(df, regime, save_path):
    """
    Final comparative chart: two panels (MAPE | Directional Accuracy),
    one line per family, showing the best single run at each horizon.
    Each data point is annotated with the winning variant's key params.
    """
    subset = df[df["regime"] == regime].copy()
    if subset.empty:
        return

    families = [f for f in FAMILY_ORDER if f in subset["family"].unique()]
    families += [f for f in subset["family"].unique() if f not in FAMILY_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    metrics_cfg = [
        ("mape",                 True,  "MAPE",               "lower = better"),
        ("directional_accuracy", False, "Directional Accuracy","higher = better"),
    ]

    for ax, (metric, lower_is_better, ylabel, hint) in zip(axes, metrics_cfg):
        valid = subset.dropna(subset=[metric])
        if valid.empty:
            ax.set_title(f"{ylabel} — no data")
            continue

        if lower_is_better:
            best_idx = valid.groupby(["family", "H"])[metric].idxmin()
        else:
            best_idx = valid.groupby(["family", "H"])[metric].idxmax()
        best     = valid.loc[best_idx]
        horizons = sorted(best["H"].unique())

        for family in families:
            fam_data = best[best["family"] == family].sort_values("H")
            if fam_data.empty:
                continue
            h_vals = fam_data["H"].tolist()
            m_vals = fam_data[metric].tolist()
            color  = get_color(family)
            marker = FAMILY_MARKERS.get(family, "o")
            label  = FAMILY_LABELS.get(family, family)

            ax.plot(h_vals, m_vals, color=color, marker=marker, markersize=9,
                    linewidth=2.4, label=label, zorder=3)
            for h, m, (_, row) in zip(h_vals, m_vals, fam_data.iterrows()):
                vl = variant_label(row)
                ax.annotate(
                    f"{m:.4f}\n({vl})",
                    (h, m),
                    textcoords="offset points",
                    xytext=(0, 12),
                    ha="center",
                    fontsize=6.5,
                    color=color,
                    fontweight="bold",
                )

        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlabel("Prediction Horizon (H)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel}\n({hint})", fontsize=13, fontweight="bold",
                     color=TEXT_COLOR)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.9,
                  edgecolor="#CCCCCC", fancybox=True)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    regime_label = "Bear Market (2022)" if regime == "bear" else "Bull Market (2023)"
    fig.suptitle(
        f"Final Comparison — Best Run per Family\n{regime_label}",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ── Best-run selection ────────────────────────────────────────────────────────

def select_best_per_family(subset: pd.DataFrame, metric: str, lower_is_better: bool) -> pd.DataFrame:
    """Return one row per family: the single best run for this metric."""
    if lower_is_better:
        idx = subset.groupby("family")[metric].idxmin()
    else:
        idx = subset.groupby("family")[metric].idxmax()
    return subset.loc[idx].reset_index(drop=True)


# ── Terminal output ───────────────────────────────────────────────────────────

def print_selection_table(regime: str, H: int, best_mape: pd.DataFrame, best_da: pd.DataFrame):
    sep = "─" * 72
    header = f"  {regime.upper()} | H={H}"
    print(f"\n{'═' * 72}")
    print(header)
    print(sep)

    col_w = max(len(short_family_label(f)) for f in FAMILY_ORDER) + 2

    # MAPE table
    print(f"  {'Best by MAPE':}")
    print(f"  {'Family':<{col_w}}  {'MAPE':>10}  {'Run ID'}")
    print(f"  {'-' * col_w}  {'----------'}  {'--------------------'}")
    for _, row in best_mape.sort_values("mape").iterrows():
        fam = short_family_label(row["family"])
        print(f"  {fam:<{col_w}}  {row['mape']:>10.4f}  {row['run_id']}")

    print()

    # DA table
    print(f"  {'Best by Directional Accuracy':}")
    print(f"  {'Family':<{col_w}}  {'Dir.Acc.':>10}  {'Run ID'}")
    print(f"  {'-' * col_w}  {'----------'}  {'--------------------'}")
    for _, row in best_da.sort_values("directional_accuracy", ascending=False).iterrows():
        fam = short_family_label(row["family"])
        print(f"  {fam:<{col_w}}  {row['directional_accuracy']:>10.4f}  {row['run_id']}")


# ── Collect best runs ─────────────────────────────────────────────────────────

def link_or_copy_run(src: str, dst: Path):
    """Create a symlink at dst pointing to src; fall back to copying key files."""
    src_path = Path(src).resolve()
    if dst.exists() or dst.is_symlink():
        dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)
    try:
        dst.symlink_to(src_path)
    except (OSError, NotImplementedError):
        # Fall back: copy just config.yaml and metrics.json
        dst.mkdir(parents=True, exist_ok=True)
        for fname in ("config.yaml", "metrics.json"):
            src_f = src_path / fname
            if src_f.exists():
                shutil.copy2(src_f, dst / fname)


def collect_best_runs(
    all_best_mape: list[dict],
    all_best_da: list[dict],
    selected_dir: Path,
):
    """
    Create selected_dir/by_mape/ and selected_dir/by_da/ with one entry per
    (regime, H, family), named  <regime>_H<H>__<family>.
    """
    mape_dir = selected_dir / "by_mape"
    da_dir   = selected_dir / "by_da"
    mape_dir.mkdir(parents=True, exist_ok=True)
    da_dir.mkdir(parents=True, exist_ok=True)

    for rec in all_best_mape:
        name = f"{rec['regime']}_H{rec['H']}__{rec['family']}"
        link_or_copy_run(rec["run_dir"], mape_dir / name)

    for rec in all_best_da:
        name = f"{rec['regime']}_H{rec['H']}__{rec['family']}"
        link_or_copy_run(rec["run_dir"], da_dir / name)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir",  default="analysis")
    args = parser.parse_args()

    setup_style()
    df = load_all_runs(args.runs_dir)
    if df.empty:
        print("No valid runs found.")
        return

    df["family"] = df.apply(build_family_label, axis=1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nLoaded {len(df)} runs")
    print(f"Families : {sorted(df['family'].unique())}")
    print(f"Regimes  : {sorted(df['regime'].unique())}")
    print(f"Horizons : {sorted(df['H'].unique())}")

    # ── Per-(regime, H) selection ─────────────────────────────────────────────
    best_dir = out / "best_of_family"
    best_dir.mkdir(exist_ok=True)

    all_best_mape: list[dict] = []
    all_best_da:   list[dict] = []

    report = ["=" * 72, "  RESULTS — ml4trading", "=" * 72, f"\nTotal runs: {len(df)}"]

    for regime in sorted(df["regime"].unique()):
        for H in sorted(df["H"].unique()):
            subset = df[(df["regime"] == regime) & (df["H"] == H)].copy()
            if subset.empty:
                continue

            label    = f"{regime}_H{H}"
            cell_dir = best_dir / label
            cell_dir.mkdir(exist_ok=True)

            best_mape = select_best_per_family(subset, "mape", lower_is_better=True)
            best_da   = select_best_per_family(subset, "directional_accuracy", lower_is_better=False)

            # Save CSVs
            best_mape.to_csv(cell_dir / "best_by_mape.csv", index=False, float_format="%.6f")
            best_da.to_csv(  cell_dir / "best_by_da.csv",   index=False, float_format="%.6f")

            # Charts
            plot_bar_comparison(
                best_mape.sort_values("mape"), "mape",
                f"MAPE — Best run per family\n{regime.upper()} | H={H}",
                "MAPE", cell_dir / "mape_comparison.png", True)

            plot_bar_comparison(
                best_da.sort_values("directional_accuracy", ascending=False),
                "directional_accuracy",
                f"Directional Accuracy — Best run per family\n{regime.upper()} | H={H}",
                "Directional Accuracy", cell_dir / "da_comparison.png", False)

            plot_metrics_table(
                best_mape.sort_values("mape"),
                f"Best MAPE per family — {regime.upper()} | H={H}",
                cell_dir / "table_best_mape.png")

            plot_metrics_table(
                best_da.sort_values("directional_accuracy", ascending=False),
                f"Best Dir. Accuracy per family — {regime.upper()} | H={H}",
                cell_dir / "table_best_da.png")

            # Collect for overview
            for _, row in best_mape.iterrows():
                all_best_mape.append({**row.to_dict(), "cell": label})
            for _, row in best_da.iterrows():
                all_best_da.append({**row.to_dict(), "cell": label})

            # Terminal output
            print_selection_table(regime, H, best_mape, best_da)

            # Report summary
            report.append(f"\n  {regime.upper()} H={H}:")
            top_mape = best_mape.sort_values("mape").iloc[0]
            top_da   = best_da.sort_values("directional_accuracy", ascending=False).iloc[0]
            report.append(
                f"    Best MAPE → {short_family_label(top_mape['family'])}"
                f"  ({top_mape['mape']:.4f})  [{top_mape['run_id']}]")
            report.append(
                f"    Best DA   → {short_family_label(top_da['family'])}"
                f"  ({top_da['directional_accuracy']:.4f})  [{top_da['run_id']}]")

    print(f"\n{'═' * 72}")

    # ── Overview heatmaps ─────────────────────────────────────────────────────
    overview_dir = best_dir / "overview"
    overview_dir.mkdir(exist_ok=True)

    def make_pivot(records, value_col):
        dff = pd.DataFrame(records)
        pivot = dff.pivot_table(index="family", columns="cell", values=value_col, aggfunc="first")
        try:
            col_order = sorted(pivot.columns, key=lambda x: (x.split("_")[0], int(x.split("H")[1])))
            pivot = pivot.reindex(columns=col_order)
        except Exception:
            pass
        row_order = [f for f in FAMILY_ORDER if f in pivot.index]
        extra     = [f for f in pivot.index if f not in FAMILY_ORDER]
        return pivot.reindex(row_order + extra)

    if all_best_mape:
        df_bm = pd.DataFrame(all_best_mape)
        df_bm.to_csv(overview_dir / "best_by_mape_all.csv", index=False, float_format="%.6f")
        pivot = make_pivot(all_best_mape, "mape")
        plot_heatmap(pivot,
            "MAPE — Best run per family per cell\n(lower = better | gold border = best in column)",
            overview_dir / "heatmap_mape.png", lower_is_better=True)

    if all_best_da:
        df_bd = pd.DataFrame(all_best_da)
        df_bd.to_csv(overview_dir / "best_by_da_all.csv", index=False, float_format="%.6f")
        pivot = make_pivot(all_best_da, "directional_accuracy")
        plot_heatmap(pivot,
            "Directional Accuracy — Best run per family per cell\n(higher = better | gold border = best in column)",
            overview_dir / "heatmap_da.png", lower_is_better=False)

    # ── Line charts ───────────────────────────────────────────────────────────
    lines_dir = out / "lines"
    lines_dir.mkdir(exist_ok=True)

    for regime in sorted(df["regime"].unique()):
        plot_lines_by_horizon(df, regime, "mape",
            "MAPE: Best run per family at each horizon",
            "MAPE", lines_dir / f"mape_{regime}.png", lower_is_better=True)
        plot_lines_by_horizon(df, regime, "directional_accuracy",
            "Dir. Accuracy: Best run per family at each horizon",
            "Directional Accuracy", lines_dir / f"da_{regime}.png", lower_is_better=False)

    plot_lines_dual_regime(df, "mape",
        "MAPE: Model Comparison across Regimes",
        "MAPE", lines_dir / "mape_dual.png", lower_is_better=True)
    plot_lines_dual_regime(df, "directional_accuracy",
        "Directional Accuracy: Model Comparison across Regimes",
        "Directional Accuracy", lines_dir / "da_dual.png", lower_is_better=False)

    # ── Final comparative: best run per family × horizon (MAPE + DA together) ─
    for regime in sorted(df["regime"].unique()):
        plot_final_comparison(
            df, regime, lines_dir / f"final_comparison_{regime}.png"
        )

    # ── Collect best run dirs ─────────────────────────────────────────────────
    selected_dir = out / "best_selected"
    collect_best_runs(all_best_mape, all_best_da, selected_dir)
    print(f"\nBest runs collected in: {selected_dir.resolve()}")
    print(f"  by_mape/ — one entry per (regime, H, family), best MAPE run")
    print(f"  by_da/   — one entry per (regime, H, family), best DA run")

    # ── Detail CSV + report ───────────────────────────────────────────────────
    df.to_csv(out / "all_runs_detail.csv", index=False, float_format="%.6f")

    report.append(f"\nOutput: {out.resolve()}")
    report_text = "\n".join(report)
    (out / "summary_report.txt").write_text(report_text, encoding="utf-8")

    print(f"\n>>> Analysis complete: {out.resolve()}")


if __name__ == "__main__":
    main()
