"""Part 2 — Exploratory Data Analysis on the clean point-in-time panel.

Reads ``master_clean.parquet`` and writes EDA figures (``charts/eda_*.png``) plus
a summary-statistics CSV. The notebook narrates these; all computation lives here
so the notebook stays a thin display layer.

Run::

    uv run python -m qfr.eda
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from qfr.factors.transforms import cs_rank, cs_winsorize, cs_zscore
from qfr.utils.config import settings
from qfr.utils.io import read_parquet
from qfr.utils.logging import logger
from qfr.utils.viz import PALETTE, save_fig, set_plot_style

VALUE = ["priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
         "evToEBITDA", "earningsYield", "freeCashFlowYield"]
QUALITY = ["returnOnEquity", "returnOnInvestedCapital", "returnOnAssets",
           "grossProfitMargin", "operatingProfitMargin", "netProfitMargin"]
GROWTH = ["revenueGrowth", "epsgrowth", "netIncomeGrowth", "ebitdaGrowth"]
SIZE = ["marketCap"]
LEVERAGE = ["debtToEquityRatio", "interestCoverageRatio"]
INPUTS = VALUE + QUALITY + GROWTH + SIZE + LEVERAGE


def load_inputs() -> pd.DataFrame:
    m = read_parquet(settings.processed_dir / "master_clean.parquet")
    m = m[m["investable"]].copy()
    return m


def _clip(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    s = s.dropna()
    return s.clip(s.quantile(lo), s.quantile(hi))


def fig_distributions(m: pd.DataFrame):
    cols = [c for c in INPUTS if c in m.columns][:12]
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(_clip(m[c]), bins=60, color=PALETTE["primary"], alpha=0.85)
        ax.set_title(c, fontsize=11)
        ax.tick_params(labelsize=8)
    fig.suptitle("Cross-sectional distributions of raw inputs (1–99% clipped)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return save_fig(fig, "eda_distributions")


def summary_stats(m: pd.DataFrame):
    rows = []
    for c in INPUTS:
        if c not in m.columns:
            continue
        s = m[c]
        rows.append({
            "field": c,
            "missing_%": round(100 * s.isna().mean(), 1),
            "median": round(float(s.median()), 4),
            "skew": round(float(s.skew()), 2),
            "excess_kurtosis": round(float(s.kurt()), 1),
        })
    stats = pd.DataFrame(rows).set_index("field")
    settings.charts_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(settings.charts_dir / "eda_summary_stats.csv")

    fig, ax = plt.subplots(figsize=(11, 6))
    order = stats["skew"].abs().sort_values(ascending=True)
    ax.barh(order.index, stats.loc[order.index, "skew"], color=PALETTE["accent"], alpha=0.85)
    ax.axvline(0, color="#444", lw=1)
    ax.set_title("Skewness of raw inputs (why we winsorise & rank)")
    ax.set_xlabel("skewness")
    fig.tight_layout()
    return save_fig(fig, "eda_skewness"), stats


def fig_outliers_box(m: pd.DataFrame):
    cols = [c for c in INPUTS if c in m.columns]
    z = cs_zscore(m[["date", *cols]], cols)
    data = [z[c].dropna().clip(-6, 6) for c in cols]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.boxplot(data, vert=False, tick_labels=cols, showfliers=True, flierprops={"markersize": 2})
    for x in (-3, 3):
        ax.axvline(x, color=PALETTE["accent"], ls="--", lw=1)
    ax.set_title("Cross-sectional z-scores: fat tails beyond ±3σ (clipped at ±6)")
    ax.set_xlabel("z-score")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return save_fig(fig, "eda_outliers_box")


def fig_transformation(m: pd.DataFrame, factor: str = "earningsYield"):
    raw = _clip(m[factor])
    wz = cs_zscore(cs_winsorize(m[["date", factor]], [factor]), [factor])[factor].dropna()
    rk = cs_rank(m[["date", factor]], [factor])[factor].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(raw, bins=60, color=PALETTE["muted"]); axes[0].set_title(f"{factor}: raw (clipped)")
    axes[1].hist(wz, bins=60, color=PALETTE["primary"]); axes[1].set_title("winsorised z-score")
    axes[2].hist(rk, bins=60, color=PALETTE["green"]); axes[2].set_title("cross-sectional rank")
    fig.suptitle("Taming a fat-tailed factor: raw → winsorised z → rank",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, "eda_transformation")


def fig_correlation_cluster(m: pd.DataFrame):
    cols = [c for c in INPUTS if c in m.columns]
    w = cs_winsorize(m[["date", *cols]], cols)
    corr = w[cols].corr(method="spearman").fillna(0.0)
    g = sns.clustermap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True,
                       fmt=".2f", annot_kws={"size": 6}, figsize=(11, 11),
                       cbar_kws={"label": "Spearman ρ"})
    g.fig.suptitle("Input correlation & clustering", y=1.02, fontsize=14, fontweight="bold")
    path = settings.charts_dir / "eda_correlation_cluster.png"
    g.fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    return path


def fig_sector_over_time(m: pd.DataFrame):
    ct = m.groupby([m["date"], "sector"]).size().unstack(fill_value=0)
    share = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.stackplot(share.index, share.T.values, labels=share.columns, alpha=0.9)
    ax.set_title("Sector composition of the investable universe over time")
    ax.set_ylabel("share")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize=8)
    fig.tight_layout()
    return save_fig(fig, "eda_sector_over_time")


def fig_return_dispersion(m: pd.DataFrame):
    disp = m.groupby("date")["ret_fwd_1m"].std()
    fig, ax = plt.subplots()
    ax.plot(disp.index, disp.values, color=PALETTE["primary"], lw=1.4)
    ax.set_title("Cross-sectional dispersion of forward 1-month returns\n"
                 "(the opportunity set — wider = more to gain from ranking)")
    ax.set_ylabel("std of forward 1m return")
    fig.tight_layout()
    return save_fig(fig, "eda_return_dispersion")


def make_all() -> None:
    set_plot_style()
    m = load_inputs()
    logger.info(f"EDA on {len(m):,} investable member-months, {len(INPUTS)} inputs")
    fig_distributions(m)
    _, stats = summary_stats(m)
    fig_outliers_box(m)
    fig_transformation(m)
    fig_correlation_cluster(m)
    fig_sector_over_time(m)
    fig_return_dispersion(m)
    logger.info("EDA figures complete:")
    logger.info("\n" + stats.to_string())


def main() -> None:
    make_all()


if __name__ == "__main__":
    main()
