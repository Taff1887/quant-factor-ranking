"""Part 1 data-coverage analytics and publication figures.

Run::

    uv run python -m qfr.data.coverage

Reads ``data/processed/master_pit.parquet`` and writes figures to ``charts/``:
    01_universe_coverage        members vs investable names over time
    01_survivorship_bias        share of the universe no longer in today's index
    01_missing_data_heatmap     field availability by year
    01_filing_lag_hist          fundamental publication lag (look-ahead control)
    01_factor_correlation       redundancy among raw value/quality signals
    01_forward_return_dist      cross-sectional spread of forward 1m returns
plus ``charts/01_summary_stats.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from qfr.utils.config import settings
from qfr.utils.io import read_parquet
from qfr.utils.logging import logger
from qfr.utils.viz import PALETTE, save_fig, set_plot_style

COVERAGE_FIELDS = [
    "adjClose", "revenue", "netIncome", "eps", "ebitda", "totalAssets",
    "totalStockholdersEquity", "totalDebt", "operatingCashFlow", "freeCashFlow",
    "priceToEarningsRatio", "priceToBookRatio", "returnOnEquity",
    "returnOnInvestedCapital", "marketCap", "revenueGrowth",
]
CORR_FIELDS = [
    "priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio", "evToEBITDA",
    "earningsYield", "freeCashFlowYield", "returnOnEquity", "returnOnInvestedCapital",
    "returnOnAssets", "grossProfitMargin", "operatingProfitMargin", "netProfitMargin",
    "debtToEquityRatio", "revenueGrowth",
]
SUMMARY_FIELDS = [
    "ret_fwd_1m", "ret_fwd_3m", "ret_fwd_6m", "priceToEarningsRatio",
    "priceToBookRatio", "returnOnEquity", "marketCap", "revenueGrowth",
    "filing_lag_days",
]


def load_master() -> pd.DataFrame:
    return read_parquet(settings.processed_dir / "master_pit.parquet")


def fig_universe_coverage(master: pd.DataFrame):
    g = master.groupby("date")
    members = g.size()
    investable = g["investable"].sum()
    fig, ax = plt.subplots()
    ax.plot(members.index, members.values, color=PALETTE["primary"], lw=2,
            label="S&P 500 members (point-in-time)")
    ax.plot(investable.index, investable.values, color=PALETTE["accent"], lw=2,
            label="Investable (PIT price + fresh fundamentals)")
    ax.fill_between(members.index, investable.values, members.values,
                    color=PALETTE["muted"], alpha=0.18, label="Data gap")
    ax.set_title("Universe coverage over time (survivorship-bias-free)")
    ax.set_ylabel("Number of stocks")
    ax.set_ylim(0, 540)
    ax.legend(loc="lower right")
    return save_fig(fig, "01_universe_coverage")


def fig_survivorship(master: pd.DataFrame):
    current = set(read_parquet(settings.processed_dir / "members_now.parquet")["symbol"])
    share_gone = master.groupby("date").apply(
        lambda d: 1.0 - d["symbol"].isin(current).mean(), include_groups=False
    )
    fig, ax = plt.subplots()
    ax.plot(share_gone.index, 100 * share_gone.values, color=PALETTE["accent"], lw=2)
    ax.fill_between(share_gone.index, 0, 100 * share_gone.values,
                    color=PALETTE["accent"], alpha=0.15)
    ax.set_title("Survivorship bias avoided: share of each month's index\n"
                 "that is NOT in today's S&P 500")
    ax.set_ylabel("% of universe no longer a current member")
    ax.set_ylim(0, None)
    return save_fig(fig, "01_survivorship_bias")


def fig_missing_heatmap(master: pd.DataFrame):
    fields = [f for f in COVERAGE_FIELDS if f in master.columns]
    yr = master["date"].dt.year
    miss = {f: master.assign(_y=yr).groupby("_y")[f].apply(lambda s: float(s.isna().mean()))
            for f in fields}
    M = pd.DataFrame(miss).T  # fields x year
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(M, cmap="rocket_r", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "fraction MISSING"}, linewidths=0.3, linecolor="white")
    ax.set_title("Field availability by year (fraction of member-months missing)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return save_fig(fig, "01_missing_data_heatmap")


def fig_filing_lag(master: pd.DataFrame):
    lag = master.loc[master["investable"], "filing_lag_days"].dropna()
    med = float(lag.median())
    fig, ax = plt.subplots()
    ax.hist(lag.clip(0, 200), bins=60, color=PALETTE["primary"], alpha=0.85)
    ax.axvline(med, color=PALETTE["accent"], lw=2, ls="--", label=f"median = {med:.0f} days")
    ax.set_title("Fundamental publication lag (rebalance date − SEC acceptance date)\n"
                 "Positive everywhere = no look-ahead")
    ax.set_xlabel("days since filing was public")
    ax.set_ylabel("member-months")
    ax.legend()
    return save_fig(fig, "01_filing_lag_hist")


def fig_factor_correlation(master: pd.DataFrame):
    fields = [f for f in CORR_FIELDS if f in master.columns]
    data = master.loc[master["investable"], fields]
    corr = data.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, square=True, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Spearman ρ"}, ax=ax)
    ax.set_title("Raw signal correlations (Spearman, pooled)")
    return save_fig(fig, "01_factor_correlation")


def fig_forward_return_dist(master: pd.DataFrame):
    r = master.loc[master["investable"], "ret_fwd_1m"].dropna()
    fig, ax = plt.subplots()
    ax.hist(r.clip(-0.4, 0.4), bins=80, color=PALETTE["green"], alpha=0.85)
    ax.axvline(float(r.mean()), color=PALETTE["accent"], lw=2, ls="--",
               label=f"mean = {r.mean():.3f}")
    ax.axvline(float(r.median()), color=PALETTE["primary"], lw=2, ls=":",
               label=f"median = {r.median():.3f}")
    ax.set_title("Cross-sectional spread of forward 1-month returns\n"
                 "(the dispersion a ranking model tries to exploit)")
    ax.set_xlabel("forward 1-month total return")
    ax.set_ylabel("member-months")
    ax.legend()
    return save_fig(fig, "01_forward_return_dist")


def write_summary_stats(master: pd.DataFrame):
    fields = [f for f in SUMMARY_FIELDS if f in master.columns]
    desc = master.loc[master["investable"], fields].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).T
    path = settings.charts_dir / "01_summary_stats.csv"
    settings.charts_dir.mkdir(parents=True, exist_ok=True)
    desc.to_csv(path)
    return path


def main() -> None:
    set_plot_style()
    master = load_master()
    logger.info(f"master_pit: {len(master):,} rows, {master['investable'].sum():,} investable")
    builders = [
        fig_universe_coverage,
        fig_survivorship,
        fig_missing_heatmap,
        fig_filing_lag,
        fig_factor_correlation,
        fig_forward_return_dist,
    ]
    for fn in builders:
        path = fn(master)
        logger.info(f"saved {path.name}")
    stats = write_summary_stats(master)
    logger.info(f"saved {stats.name}")
    logger.info("Coverage figures complete.")


if __name__ == "__main__":
    main()
