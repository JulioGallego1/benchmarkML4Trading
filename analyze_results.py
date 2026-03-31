#!/usr/bin/env python3
"""
Análisis comparativo de modelos: Best-of-Family y Medias.

Genera una estructura de carpetas con gráficos profesionales:

  analysis/
  ├── best_of_family/
  │   ├── bear_H21/
  │   │   ├── mape_comparison.png
  │   │   ├── directional_accuracy_comparison.png
  │   │   ├── all_metrics_table.png
  │   │   └── best_runs.csv
  │   ├── bear_H63/
  │   │   └── ...
  │   └── overview/
  │       ├── heatmap_mape.png
  │       ├── heatmap_da.png
  │       ├── global_ranking_mape.png
  │       ├── global_ranking_da.png
  │       └── best_of_family_all.csv
  ├── averages/
  │   ├── heatmap_mape_avg.png
  │   ├── heatmap_da_avg.png
  │   └── averages_all.csv
  └── summary_report.txt

Uso:
    python analyze_results.py
    python analyze_results.py --runs-dir runs --out-dir analysis
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# Visual config
FAMILY_COLORS = {
    "RF_pooled":                       "#2E86AB",
    "RF_per_ticker":                   "#A23B72",
    "LSTM_mimo_pooled":                "#F18F01",
    "LSTM_mimo_per_ticker":            "#C73E1D",
    "LSTM_recursive_pooled":           "#3B1F2B",
    "LSTM_recursive_per_ticker":       "#44BBA4",
    "PATCHTST_mimo_pooled":            "#E94F37",
    "PATCHTST_mimo_per_ticker":        "#393E41",
    "PATCHTST_recursive_pooled":       "#8963BA",
    "PATCHTST_recursive_per_ticker":   "#17BEBB",
}

FAMILY_MARKERS = {
    "RF_pooled":                       "o",
    "RF_per_ticker":                   "s",
    "LSTM_mimo_pooled":                "o",
    "LSTM_mimo_per_ticker":            "s",
    "LSTM_recursive_pooled":           "^",
    "LSTM_recursive_per_ticker":       "D",
    "PATCHTST_mimo_pooled":            "o",
    "PATCHTST_mimo_per_ticker":        "s",
    "PATCHTST_recursive_pooled":       "^",
    "PATCHTST_recursive_per_ticker":   "D",
}

FAMILY_LABELS = {
    "RF_pooled":                       "RF — MIMO (pooled)",
    "RF_per_ticker":                   "RF — MIMO (per-ticker)",
    "LSTM_mimo_pooled":                "LSTM — MIMO (pooled)",
    "LSTM_mimo_per_ticker":            "LSTM — MIMO (per-ticker)",
    "LSTM_recursive_pooled":           "LSTM — Recursive (pooled)",
    "LSTM_recursive_per_ticker":       "LSTM — Recursive (per-ticker)",
    "PATCHTST_mimo_pooled":            "Transformer — MIMO (pooled)",
    "PATCHTST_mimo_per_ticker":        "Transformer — MIMO (per-ticker)",
    "PATCHTST_recursive_pooled":       "Transformer — Recursive (pooled)",
    "PATCHTST_recursive_per_ticker":   "Transformer — Recursive (per-ticker)",
}

FAMILY_ORDER = [
    "RF_pooled", "RF_per_ticker",
    "LSTM_mimo_pooled", "LSTM_mimo_per_ticker",
    "LSTM_recursive_pooled", "LSTM_recursive_per_ticker",
    "PATCHTST_mimo_pooled", "PATCHTST_mimo_per_ticker",
    "PATCHTST_recursive_pooled", "PATCHTST_recursive_per_ticker",
]

BG_COLOR = "#FAFAFA"
GRID_COLOR = "#E0E0E0"
TEXT_COLOR = "#2D2D2D"
ACCENT_GOLD = "#FFB703"
ACCENT_GREEN = "#2A9D8F"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


# Data loading
def load_all_runs(runs_dir: str = "runs") -> pd.DataFrame:
    runs_path = Path(runs_dir)
    records = []
    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        metrics_file = run_dir / "metrics.json"
        config_file = run_dir / "config.yaml"
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
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "model": config.get("model", "unknown").upper(),
            "regime": config.get("regime", "unknown"),
            "L": config.get("L", 0),
            "H": config.get("H", 0),
            "strategy": config.get("strategy", "mimo"),
            "training_mode": config.get("training_mode", "pooled"),
            "mape": metrics.get("mape", None),
            "smape": metrics.get("smape", None),
            "mae": metrics.get("mae", None),
            "rmse": metrics.get("rmse", None),
            "directional_accuracy": metrics.get("directional_accuracy", None),
        })
    return pd.DataFrame(records) if records else pd.DataFrame()


def build_family_label(row) -> str:
    model = row["model"]
    strategy = row["strategy"]
    mode = row["training_mode"]
    if model == "RF":
        return f"RF_{mode}"
    return f"{model}_{strategy}_{mode}"


def short_family_label(family: str) -> str:
    return (family
        .replace("_pooled", "\n(pooled)")
        .replace("_per_ticker", "\n(per-ticker)")
        .replace("_mimo", "\nMIMO")
        .replace("_recursive", "\nRecursive"))


def get_color(family: str) -> str:
    return FAMILY_COLORS.get(family, "#999999")


# Chart generators
def plot_bar_comparison(data, metric, title, ylabel, save_path, lower_is_better=True):
    if data.empty:
        return
    sorted_data = data.sort_values(metric, ascending=lower_is_better)
    families = sorted_data["family"].tolist()
    values = sorted_data[metric].tolist()
    colors = [get_color(f) for f in families]

    fig, ax = plt.subplots(figsize=(12, max(4, len(families) * 0.6)))
    bars = ax.barh(range(len(families)), values, color=colors, height=0.65,
                   edgecolor="white", linewidth=0.5)
    bars[0].set_edgecolor(ACCENT_GOLD)
    bars[0].set_linewidth(3)

    max_val = max(values) if values else 1
    for i, (bar, val) in enumerate(zip(bars, values)):
        offset = max_val * 0.02
        weight = "bold" if i == 0 else "normal"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=10,
                fontweight=weight, color=TEXT_COLOR)

    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([short_family_label(f) for f in families], fontsize=9)
    ax.set_xlabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=TEXT_COLOR)
    ax.text(0.98, 0.02, f"Mejor: {families[0]}\n({values[0]:.2f})",
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
    cols = ["family", "mape", "directional_accuracy", "mae", "rmse", "L", "run_id"]
    headers = ["Familia", "MAPE (%)", "Dir. Acc. (%)", "MAE", "RMSE", "L", "Run ID"]
    available = [c for c in cols if c in data.columns]

    display_data = []
    for _, row in data.iterrows():
        formatted = []
        for col in available:
            val = row[col]
            if col == "run_id":
                s = str(val)
                formatted.append(s[:40] + "..." if len(s) > 40 else s)
            elif isinstance(val, float):
                formatted.append(f"{val:.4f}")
            else:
                formatted.append(str(val))
        display_data.append(formatted)

    fig, ax = plt.subplots(figsize=(16, max(2, len(display_data) * 0.5 + 1.5)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20, color=TEXT_COLOR)

    hdrs = [headers[cols.index(c)] for c in available]
    table = ax.table(cellText=display_data, colLabels=hdrs, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for j in range(len(hdrs)):
        table[0, j].set_facecolor("#2E86AB")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(display_data)):
        color = "#F5F5F5" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(hdrs)):
            table[i + 1, j].set_facecolor(color)
    if "mape" in available:
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
                                     max(5, len(pivot.index) * 0.55)))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([short_family_label(f).replace("\n", " ") for f in pivot.index], fontsize=9)

    vmin, vmax = np.nanmin(data), np.nanmax(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999999")
            else:
                norm = (val - vmin) / (vmax - vmin + 1e-10)
                color = "white" if (norm > 0.65 if lower_is_better else norm < 0.35) else TEXT_COLOR
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    for j in range(data.shape[1]):
        col = data[:, j]
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


def plot_global_ranking(data, metric, title, ylabel, save_path, lower_is_better=True):
    if data.empty:
        return
    sorted_data = data.sort_values(metric, ascending=lower_is_better)
    families = sorted_data["family"].tolist()
    values = sorted_data[metric].tolist()
    colors = [get_color(f) for f in families]

    fig, ax = plt.subplots(figsize=(max(10, len(families) * 1.2), 6))
    bars = ax.bar(range(len(families)), values, color=colors, width=0.7,
                  edgecolor="white", linewidth=0.5)
    bars[0].set_edgecolor(ACCENT_GOLD)
    bars[0].set_linewidth(3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([short_family_label(f) for f in families], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=TEXT_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_lines_by_horizon(
    df: pd.DataFrame,
    regime: str,
    metric: str,
    title: str,
    ylabel: str,
    save_path: Path,
    lower_is_better: bool = True,
):
    """Line chart: X=horizon, one line per family, best run per (family, H)."""
    subset = df[df["regime"] == regime].copy()
    if subset.empty:
        return

    # Best run per (family, H)
    if lower_is_better:
        best_idx = subset.groupby(["family", "H"])[metric].idxmin()
    else:
        best_idx = subset.groupby(["family", "H"])[metric].idxmax()
    best = subset.loc[best_idx]

    horizons = sorted(best["H"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Order families consistently
    families_present = [f for f in FAMILY_ORDER if f in best["family"].unique()]
    extra_fam = [f for f in best["family"].unique() if f not in FAMILY_ORDER]
    families = families_present + extra_fam

    for family in families:
        fam_data = best[best["family"] == family].sort_values("H")
        if fam_data.empty:
            continue

        h_vals = fam_data["H"].tolist()
        m_vals = fam_data[metric].tolist()
        color = get_color(family)
        marker = FAMILY_MARKERS.get(family, "o")
        label = FAMILY_LABELS.get(family, family)

        ax.plot(h_vals, m_vals,
                color=color, marker=marker, markersize=8,
                linewidth=2.2, label=label, zorder=3)

        # Value annotations
        for h, m in zip(h_vals, m_vals):
            ax.annotate(f"{m:.1f}", (h, m), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=7.5,
                       color=color, fontweight="bold")

    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("Horizonte de Predicción (H)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    regime_label = "Bear Market (Bajista 2022)" if regime == "bear" else "Bull Market (Alcista 2023)"
    ax.set_title(f"{title}\n{regime_label}", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=15)

    ax.legend(loc="upper left", fontsize=8, framealpha=0.9,
              edgecolor="#CCCCCC", fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_lines_dual_regime(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    save_path: Path,
    lower_is_better: bool = True,
):
    """Side-by-side line charts (bear | bull) like the reference image."""
    regimes = sorted(df["regime"].unique())
    if len(regimes) < 2:
        # Fallback to single plot
        plot_lines_by_horizon(df, regimes[0], metric, title, ylabel, save_path, lower_is_better)
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    families_present = [f for f in FAMILY_ORDER if f in df["family"].unique()]
    extra_fam = [f for f in df["family"].unique() if f not in FAMILY_ORDER]
    families = families_present + extra_fam

    for ax, regime in zip(axes, regimes):
        subset = df[df["regime"] == regime]
        if subset.empty:
            continue

        if lower_is_better:
            best_idx = subset.groupby(["family", "H"])[metric].idxmin()
        else:
            best_idx = subset.groupby(["family", "H"])[metric].idxmax()
        best = subset.loc[best_idx]

        horizons = sorted(best["H"].unique())

        for family in families:
            fam_data = best[best["family"] == family].sort_values("H")
            if fam_data.empty:
                continue

            h_vals = fam_data["H"].tolist()
            m_vals = fam_data[metric].tolist()
            color = get_color(family)
            marker = FAMILY_MARKERS.get(family, "o")
            label = FAMILY_LABELS.get(family, family)

            ax.plot(h_vals, m_vals,
                    color=color, marker=marker, markersize=8,
                    linewidth=2.2, label=label, zorder=3)

        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlabel("Horizonte de Predicción (H)", fontsize=11)

        regime_label = "Bear Market (Bajista 2022)" if regime == "bear" else "Bull Market (Alcista 2023)"
        ax.set_title(regime_label, fontsize=13, fontweight="bold", color=TEXT_COLOR)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(ylabel, fontsize=12)

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               framealpha=0.95, edgecolor="#CCCCCC", fancybox=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="analysis")
    args = parser.parse_args()

    setup_style()
    df = load_all_runs(args.runs_dir)
    if df.empty:
        print("No se encontraron runs válidos.")
        return

    df["family"] = df.apply(build_family_label, axis=1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Total runs: {len(df)}")
    print(f"Familias: {sorted(df['family'].unique())}")
    print(f"Regímenes: {sorted(df['regime'].unique())}")
    print(f"Horizontes: {sorted(df['H'].unique())}")

    report = []
    report.append("=" * 80)
    report.append("  INFORME DE RESULTADOS — ml4trading")
    report.append("=" * 80)
    report.append(f"\nTotal runs: {len(df)}")

    # BEST OF FAMILY
    best_dir = out / "best_of_family"
    best_dir.mkdir(exist_ok=True)
    all_best_by_mape = []
    all_best_by_da = []

    for regime in sorted(df["regime"].unique()):
        for H in sorted(df["H"].unique()):
            subset = df[(df["regime"] == regime) & (df["H"] == H)]
            if subset.empty:
                continue
            label = f"{regime}_H{H}"
            cell_dir = best_dir / label
            cell_dir.mkdir(exist_ok=True)

            # Best by MAPE
            idx_mape = subset.groupby("family")["mape"].idxmin()
            best_by_mape = subset.loc[idx_mape].reset_index(drop=True).sort_values("mape")

            # Best by DA
            idx_da = subset.groupby("family")["directional_accuracy"].idxmax()
            best_by_da = subset.loc[idx_da].reset_index(drop=True).sort_values("directional_accuracy", ascending=False)

            # Save CSVs
            best_by_mape.to_csv(cell_dir / "best_by_mape.csv", index=False, float_format="%.6f")
            best_by_da.to_csv(cell_dir / "best_by_da.csv", index=False, float_format="%.6f")

           
            plot_bar_comparison(best_by_mape, "mape",
                f"MAPE — Mejor run por familia (criterio: menor MAPE)\n{regime.upper()} | H={H}",
                "MAPE (%)", cell_dir / "mape_comparison.png", True)

            
            plot_bar_comparison(best_by_da, "directional_accuracy",
                f"Dir. Accuracy — Mejor run por familia (criterio: mayor DA)\n{regime.upper()} | H={H}",
                "Dir. Accuracy (%)", cell_dir / "directional_accuracy_comparison.png", False)

            # Tables
            plot_metrics_table(best_by_mape,
                f"Mejor MAPE por familia — {regime.upper()} | H={H}",
                cell_dir / "table_best_mape.png")
            plot_metrics_table(best_by_da,
                f"Mejor Dir. Accuracy por familia — {regime.upper()} | H={H}",
                cell_dir / "table_best_da.png")

            # Collect for overview heatmaps
            for _, row in best_by_mape.iterrows():
                rc = row.copy()
                rc["cell"] = label
                all_best_by_mape.append(rc)
            for _, row in best_by_da.iterrows():
                rc = row.copy()
                rc["cell"] = label
                all_best_by_da.append(rc)

            report.append(f"\n  {regime.upper()} H={H}:")
            report.append(f"    Mejor MAPE → {best_by_mape.iloc[0]['family']} ({best_by_mape.iloc[0]['mape']:.4f}%)")
            report.append(f"    Mejor DA   → {best_by_da.iloc[0]['family']} ({best_by_da.iloc[0]['directional_accuracy']:.2f}%)")
            print(f"  [{label}] Best MAPE: {best_by_mape.iloc[0]['family']} ({best_by_mape.iloc[0]['mape']:.4f}) | Best DA: {best_by_da.iloc[0]['family']} ({best_by_da.iloc[0]['directional_accuracy']:.2f})")


    overview_dir = best_dir / "overview"
    overview_dir.mkdir(exist_ok=True)


    if all_best_by_mape:
        df_bm = pd.DataFrame(all_best_by_mape)
        df_bm.to_csv(overview_dir / "best_by_mape_all.csv", index=False, float_format="%.6f")

        pivot = df_bm.pivot_table(index="family", columns="cell", values="mape")
        col_order = sorted(pivot.columns, key=lambda x: (x.split("_")[0], int(x.split("H")[1])))
        pivot = pivot.reindex(columns=col_order)
        row_order = [f for f in FAMILY_ORDER if f in pivot.index]
        extra = [f for f in pivot.index if f not in FAMILY_ORDER]
        pivot = pivot.reindex(row_order + extra)
        plot_heatmap(pivot,
            "MAPE (%) — Mejor run por MAPE de cada familia\nMenor = Mejor | Borde dorado = mejor por columna",
            overview_dir / "heatmap_mape.png", lower_is_better=True)


    if all_best_by_da:
        df_bd = pd.DataFrame(all_best_by_da)
        df_bd.to_csv(overview_dir / "best_by_da_all.csv", index=False, float_format="%.6f")

        pivot = df_bd.pivot_table(index="family", columns="cell", values="directional_accuracy")
        col_order = sorted(pivot.columns, key=lambda x: (x.split("_")[0], int(x.split("H")[1])))
        pivot = pivot.reindex(columns=col_order)
        row_order = [f for f in FAMILY_ORDER if f in pivot.index]
        extra = [f for f in pivot.index if f not in FAMILY_ORDER]
        pivot = pivot.reindex(row_order + extra)
        plot_heatmap(pivot,
            "Dir. Accuracy (%) — Mejor run por DA de cada familia\nMayor = Mejor | Borde dorado = mejor por columna",
            overview_dir / "heatmap_da.png", lower_is_better=False)


    if all_best_by_mape:
        df_bm = pd.DataFrame(all_best_by_mape)
        global_mape = df_bm.groupby("family").agg(
            mape=("mape", "mean"), n_cells=("cell", "count")).reset_index()
        plot_global_ranking(global_mape, "mape",
            "Ranking Global MAPE\n(media del mejor MAPE por celda — menor = mejor)",
            "MAPE (%)", overview_dir / "global_ranking_mape.png", True)

    if all_best_by_da:
        df_bd = pd.DataFrame(all_best_by_da)
        global_da = df_bd.groupby("family").agg(
            directional_accuracy=("directional_accuracy", "mean"), n_cells=("cell", "count")).reset_index()
        plot_global_ranking(global_da, "directional_accuracy",
            "Ranking Global Dir. Accuracy\n(media de la mejor DA por celda — mayor = mejor)",
            "Dir. Accuracy (%)", overview_dir / "global_ranking_da.png", False)

    # AVERAGES
    avg_dir = out / "averages"
    avg_dir.mkdir(exist_ok=True)

    grouped = df.groupby(["family", "regime", "H"]).agg(
        mape_mean=("mape", "mean"), da_mean=("directional_accuracy", "mean"),
        n_runs=("run_id", "count")).reset_index()
    grouped["cell"] = grouped["regime"] + "_H" + grouped["H"].astype(str)
    grouped.to_csv(avg_dir / "averages_all.csv", index=False, float_format="%.6f")

    for metric_col, val_col, is_lower, name in [
        ("mape_mean", "mape_mean", True, "MAPE"),
        ("da_mean", "da_mean", False, "Dir. Accuracy")
    ]:
        pivot = grouped.pivot_table(index="family", columns="cell", values=val_col)
        if pivot.empty:
            continue
        col_order = sorted(pivot.columns, key=lambda x: (x.split("_")[0], int(x.split("H")[1])))
        pivot = pivot.reindex(columns=col_order)
        row_order = [f for f in FAMILY_ORDER if f in pivot.index]
        extra = [f for f in pivot.index if f not in FAMILY_ORDER]
        pivot = pivot.reindex(row_order + extra)
        qualifier = "Menor = Mejor" if is_lower else "Mayor = Mejor"
        plot_heatmap(pivot,
            f"{name} (%) — Media de todos los runs\n(Complementario — {qualifier})",
            avg_dir / f"heatmap_{name.lower().replace(' ', '_').replace('.', '')}_avg.png",
            lower_is_better=is_lower)

    # LINE CHARTS
    lines_dir = out / "lines"
    lines_dir.mkdir(exist_ok=True)

    for regime in sorted(df["regime"].unique()):
        plot_lines_by_horizon(df, regime, "mape",
            "MAPE: Mejor modelo por familia en cada horizonte",
            "MAPE (%)", lines_dir / f"mape_{regime}.png", lower_is_better=True)

        plot_lines_by_horizon(df, regime, "directional_accuracy",
            "Dir. Accuracy: Mejor modelo por familia en cada horizonte",
            "Dir. Accuracy (%)", lines_dir / f"da_{regime}.png", lower_is_better=False)

    plot_lines_dual_regime(df, "mape",
        "MAPE: Comparativa de Modelos Competitivos",
        "MAPE (%)", lines_dir / "mape_dual.png", lower_is_better=True)

    plot_lines_dual_regime(df, "directional_accuracy",
        "Dir. Accuracy: Comparativa de Modelos Competitivos",
        "Dir. Accuracy (%)", lines_dir / "da_dual.png", lower_is_better=False)

    print(f"  Line charts generados en {lines_dir}")

    # Detail CSV
    df.to_csv(out / "all_runs_detail.csv", index=False, float_format="%.6f")

    # Report
    report.append(f"\nArchivos en: {out.resolve()}")
    report_text = "\n".join(report)
    (out / "summary_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n>>> Análisis completo en: {out.resolve()}")


if __name__ == "__main__":
    main()