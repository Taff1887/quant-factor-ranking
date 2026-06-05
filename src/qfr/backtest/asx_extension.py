"""ASX 200 extension — same 5-factor methodology, institutional-grade implementation.

Built on an INDEPENDENT data pipeline that pulls fresh data directly from:
  - FMP stable API: universe, dividend-adjusted prices, quarterly fundamentals
    with `acceptedDate` filing-date column for strict PIT lag.
  - Yahoo Finance: current free-float ratio (`floatShares` / `sharesOutstanding`)
    for cap-weighting that matches how the real S&P/ASX 200 index weights itself.

The Short King 2.0 panel is no longer used. To reproduce the data from scratch:

    uv run python -m qfr.backtest.asx_pull_data     # universe + prices + fundamentals + freefloat
    uv run python -m qfr.backtest.asx_assemble      # PIT-join, free-float-adjust, top-200 filter
    uv run python -m qfr.backtest.asx_extension     # IC + portfolios + diagnostics

Pipeline:
  1. Top-250 ASX ordinaries by market cap (current), filtered to PIT top-200 each month
  2. Monthly resample; ret_fwd_1m = next-month dividend-adjusted close / current close - 1
  3. PIT-lagged fundamentals via merge_asof on `acceptedDate`
  4. Free-float-adjusted market cap = sharesOut * float_ratio * (price_t / price_now)
  5. 5-factor equal-weight z-score composite (ROIC, ROE, FCF yield, rev growth, EPS growth)

Diagnostics produced (parity with the S&P 500 §8 analysis):
  - Per-factor rank IC + composite IC at multiple horizons
  - Long-only top-quintile / top-decile (EW + CW on free-float mcap)
  - Dollar-neutral LS Q1-Q5 EW + sector-neutral LS variant
  - Cost sensitivity sweep at 10, 25, 50 bps/side (25 bps = realistic ASX)
  - Turnover decomposition + holding-period analysis
  - Gross-vs-net cost-drag decomposition
  - Benchmark: IOZ.AX (iShares Core S&P/ASX 200 total return)

Run::  uv run python -m qfr.backtest.asx_extension
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.backtest.asx_assemble import FACTORS as ASX_FACTORS
from qfr.backtest.asx_assemble import build_panel
from qfr.backtest.portfolio import (
    ANN,
    COST_LEVELS_BPS,
    _spearman_ic_monthly,
    capm,
    composite_ic_horizons,
    cost_sensitivity_table,
    cs_z,
    gross_vs_net_table,
    longshort_monthly,
    perf_metrics,
    portfolio_monthly,
    sector_neutral_ls,
    summary_metrics_table,
    turnover_analysis,
)
from qfr.data.fmp_client import FMPClient
from qfr.utils.config import PROJECT_ROOT
from qfr.utils.logging import logger

# Realistic ASX cost assumptions
REALISTIC_COST_PER_SIDE = 0.0025   # 25 bps/side baseline (more honest than the US 10 bps)
COST_LEVELS_BPS_ASX = [10, 25, 50]

BENCHMARK_SYMBOL = "IOZ.AX"
BENCHMARK_START = "2010-01-01"
BENCHMARK_END = "2026-05-31"


# --------------------------------------------------------------------------
# Composite + IC helpers
# --------------------------------------------------------------------------
def make_composite(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    zs = pd.DataFrame({c: cs_z(panel, c) for c in ASX_FACTORS})
    arr = zs.to_numpy()
    mask = ~np.isnan(arr)
    counts = mask.sum(axis=1)
    sums = np.where(mask, arr, 0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        panel["composite"] = np.where(counts > 0, sums / counts, np.nan)
    return panel


def per_factor_ic_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in ASX_FACTORS:
        ic = _spearman_ic_monthly(panel, f, "ret_fwd_1m")
        ic2 = _spearman_ic_monthly(panel, f, "ret_fwd_3m")
        s1, s2 = ic.dropna(), ic2.dropna()
        rows.append({
            "factor": f,
            "IC_1m_%": s1.mean() * 100,
            "t_1m": (s1.mean() / s1.std() * np.sqrt(len(s1))) if s1.std() > 0 else np.nan,
            "hit_1m_%": (s1 > 0).mean() * 100,
            "IC_3m_%": s2.mean() * 100,
            "t_3m": (s2.mean() / s2.std() * np.sqrt(len(s2))) if s2.std() > 0 else np.nan,
            "n_months": len(s1),
        })
    return pd.DataFrame(rows)


def composite_ic_table(panel: pd.DataFrame) -> pd.DataFrame:
    df, _ = composite_ic_horizons(panel, "composite")
    return df


def fetch_benchmark_monthly() -> pd.Series:
    rows = FMPClient().historical_prices(BENCHMARK_SYMBOL,
                                         from_date=BENCHMARK_START,
                                         to_date=BENCHMARK_END,
                                         series="dividend-adjusted")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date").sort_index()["adjClose"].resample("ME").last()
    return (monthly.shift(-1) / monthly - 1).rename(BENCHMARK_SYMBOL)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def chart_cumulative_asx(long_only: list[dict], ls: dict, ls_sn: dict,
                         benchmark: pd.Series, cost_per_side: float) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import save_fig, set_plot_style
    set_plot_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    lo_colors = ["#1f3a5f", "#456b9a", "#0a7a3a", "#3aa56b"]
    for s, c in zip(long_only, lo_colors):
        net = (s["gross"] - s["total_traded"] * cost_per_side).fillna(0)
        cum = (1 + net).cumprod()
        ax.plot(cum.index, cum.values, lw=1.8, color=c, label=s["name"])
    for s, c in [(ls, "#bc4a3c"), (ls_sn, "#d9a02e")]:
        net = (s["gross"] - s["total_traded"] * cost_per_side).fillna(0)
        cum = (1 + net).cumprod()
        ax.plot(cum.index, cum.values, lw=2.0, color=c, label=s["name"])
    bm = benchmark.reindex(long_only[0]["gross"].index).fillna(0)
    cum_bm = (1 + bm).cumprod()
    ax.plot(cum_bm.index, cum_bm.values, lw=1.6, color="#222", ls="--",
            label=f"{BENCHMARK_SYMBOL} (S&P/ASX 200 TR proxy)")
    ax.set_yscale("log")
    ax.set_title(f"ASX 200 — growth of $1, long-only + dollar-neutral + sector-neutral LS  "
                 f"(net of {int(cost_per_side*10000)} bps/side, free-float-adjusted mcap)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "asx_cumulative")


def chart_ic_bars(per_factor_df: pd.DataFrame, comp_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import save_fig, set_plot_style
    set_plot_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    pretty = {"returnOnInvestedCapital": "ROIC", "returnOnEquity": "ROE",
              "freeCashFlowYield": "FCF yield", "revenueGrowth": "Rev growth",
              "epsgrowth": "EPS growth"}
    pf = per_factor_df.copy()
    pf["label"] = pf["factor"].map(pretty)
    pf = pf.sort_values("IC_1m_%")
    a1.barh(pf["label"], pf["IC_1m_%"],
            color=["#bc4a3c" if v < 0 else "#1f3a5f" for v in pf["IC_1m_%"]], alpha=0.85)
    for i, (v, t) in enumerate(zip(pf["IC_1m_%"], pf["t_1m"])):
        a1.text(v + (0.05 if v > 0 else -0.05), i, f"{v:.2f}% (t={t:.2f})",
                va="center", fontsize=9, ha="left" if v > 0 else "right")
    a1.axvline(0, color="#444", lw=0.7)
    a1.set_xlabel("Mean rank IC (1m forward returns, %)")
    a1.set_title("ASX 200 — per-factor rank IC (Spearman, 1m)", fontsize=11,
                 fontweight="bold", loc="left")
    horizons = comp_df["horizon_months"].tolist()
    ics = comp_df["mean_ic_%"].tolist()
    ts = comp_df["t_stat"].tolist()
    xs = np.arange(len(horizons))
    bars = a2.bar(xs, ics, color="#1f3a5f", alpha=0.85)
    for i, (b, v, t) in enumerate(zip(bars, ics, ts)):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}%\nt={t:.2f}",
                ha="center", fontsize=8.5)
    a2.set_xticks(xs)
    a2.set_xticklabels([f"{h}m" for h in horizons])
    a2.axhline(0, color="#444", lw=0.7)
    a2.set_xlabel("Forward-return horizon")
    a2.set_ylabel("Mean rank IC (%)")
    a2.set_title("ASX 200 — composite rank IC by horizon", fontsize=11,
                 fontweight="bold", loc="left")
    fig.tight_layout()
    save_fig(fig, "asx_ic_bars")


def chart_cost_sensitivity_asx(cs_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import save_fig, set_plot_style
    set_plot_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, sub in cs_df.groupby("strategy"):
        sub = sub.sort_values("cost_bps")
        a1.plot(sub["cost_bps"], sub["Sharpe"], marker="o", lw=1.7, label=name)
        a2.plot(sub["cost_bps"], sub["CAGR_%"], marker="o", lw=1.7, label=name)
    for ax, ylab in zip([a1, a2], ["Sharpe (net)", "CAGR (%, net)"]):
        ax.set_xlabel("Transaction cost (bps per side)")
        ax.set_ylabel(ylab)
        ax.axhline(0, color="#444", lw=0.6)
        ax.grid(alpha=0.3)
    a1.legend(fontsize=8, loc="upper right")
    fig.suptitle("ASX cost sensitivity — long-only and LS at 10/25/50 bps/side  (25 bps = realistic ASX)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "asx_cost_sensitivity")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main() -> None:
    panel = build_panel()
    panel = make_composite(panel)
    benchmark = fetch_benchmark_monthly()

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- IC tables ----
    per_factor = per_factor_ic_table(panel)
    comp_ic = composite_ic_table(panel)
    per_factor.to_csv(out_dir / "asx_per_factor_ic.csv", index=False)
    comp_ic.to_csv(out_dir / "asx_composite_ic.csv", index=False)

    # ---- Portfolios ----
    lo_dec_ew = portfolio_monthly(panel, "composite", n_buckets=10, weight="equal", name="Top decile EW")
    lo_dec_cw = portfolio_monthly(panel, "composite", n_buckets=10, weight="cap",   name="Top decile CW (ff-adj)")
    lo_qui_ew = portfolio_monthly(panel, "composite", n_buckets=5,  weight="equal", name="Top quintile EW")
    lo_qui_cw = portfolio_monthly(panel, "composite", n_buckets=5,  weight="cap",   name="Top quintile CW (ff-adj)")
    ls_q_ew   = longshort_monthly(panel, "composite", n_buckets=5,  weight="equal", name="LS Q1-Q5 EW")
    ls_q_sn   = sector_neutral_ls(panel, "composite", n_buckets=5,  weight="equal")
    ls_q_sn["name"] = "LS Q1-Q5 EW (sector-neutral)"

    long_only = [lo_dec_ew, lo_dec_cw, lo_qui_ew, lo_qui_cw]
    ls_set = [ls_q_ew, ls_q_sn]
    all_strats = long_only + ls_set

    # ---- Summary tables (at the REALISTIC cost level of 25 bps/side) ----
    summary_lo = summary_metrics_table(long_only, benchmark, cost_per_side=REALISTIC_COST_PER_SIDE)
    summary_ls = summary_metrics_table(ls_set, benchmark, cost_per_side=REALISTIC_COST_PER_SIDE)

    bm_aligned = benchmark.reindex(lo_qui_cw["gross"].index).dropna()
    m_bm = perf_metrics(bm_aligned)
    bm_row = pd.DataFrame([{
        "strategy": BENCHMARK_SYMBOL, "kind": "benchmark",
        "CAGR_%": m_bm["CAGR"] * 100, "ann_vol_%": m_bm["ann_vol"] * 100,
        "Sharpe": m_bm["Sharpe"], "max_drawdown_%": m_bm["max_drawdown"] * 100,
        "beta_vs_SPY": 1.0, "Jensen_alpha_%": 0.0,
        "ann_turnover_pa_%": np.nan, "cost_drag_pa_%": 0.0,
    }])
    summary_lo_full = pd.concat([summary_lo, bm_row], ignore_index=True)
    summary_lo_full.to_csv(out_dir / "asx_summary_long_only.csv", index=False)
    summary_ls.to_csv(out_dir / "asx_summary_long_short.csv", index=False)

    # ---- Diagnostics ----
    cs_table = cost_sensitivity_table(all_strats, benchmark, cost_levels_bps=COST_LEVELS_BPS_ASX)
    gn_table = gross_vs_net_table(all_strats, cost_per_side=REALISTIC_COST_PER_SIDE)
    to_table = turnover_analysis(all_strats)
    cs_table.to_csv(out_dir / "asx_cost_sensitivity.csv", index=False)
    gn_table.to_csv(out_dir / "asx_gross_vs_net.csv", index=False)
    to_table.to_csv(out_dir / "asx_turnover.csv", index=False)

    # ---- Charts ----
    chart_cumulative_asx(long_only, ls_q_ew, ls_q_sn, benchmark, REALISTIC_COST_PER_SIDE)
    chart_ic_bars(per_factor, comp_ic)
    chart_cost_sensitivity_asx(cs_table)

    # ---- Console summary ----
    logger.info(f"\n=== ASX per-factor rank IC ===\n{per_factor.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX composite rank IC by horizon ===\n{comp_ic.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX long-only summary (net of 25 bps/side, vs IOZ.AX) ===\n"
                f"{summary_lo_full.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX long-short summary (net of 25 bps/side) ===\n"
                f"{summary_ls.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX cost sensitivity ===\n{cs_table.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX gross vs net (at 25 bps/side) ===\n{gn_table.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX turnover ===\n{to_table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
