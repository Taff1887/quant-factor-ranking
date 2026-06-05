"""ASX 200 extension — same 5-factor methodology applied to Australian equities.

Re-uses the point-in-time fundamentals + price panel from the companion project
Short King 2.0 (`master_pit.parquet`). That panel already provides:

  - 500 ASX tickers, 2010-06 -> 2026-05, weekly cadence
  - SEC-equivalent filing-date-lagged fundamentals via FMP (AsOfDate /
    ReleaseDate columns) -> no look-ahead in factor inputs
  - Forward returns (`fwd_ret_1m`, `fwd_ret_3m`) pre-computed
  - All five factors we use in the S&P 500 composite:
      key_metrics_returnOnInvestedCapital   (ROIC)
      key_metrics_returnOnEquity            (ROE)
      cash_flow_freeCashFlow / mktCap       (FCF yield, computed here)
      financial_growth_revenueGrowth        (revenue growth)
      financial_growth_epsgrowth            (EPS growth)

What this module does:

  1. Resample the weekly panel to monthly (last weekly observation in month)
  2. Filter to top-200-by-marketCap at each date as a PIT proxy for the
     S&P/ASX 200 (FMP has no `historical-asx200-constituent` endpoint, so
     this is the cleanest stand-in; rebuilt month by month so it stays PIT)
  3. Build the SAME 5-factor equal-weight composite as the S&P 500 model
  4. Compute composite Spearman rank IC at 1m / 3m / 12m horizons
  5. Run top-quintile / top-decile long-only portfolios (EW and CW)
  6. Run LS Q1-Q5 EW (dollar-neutral) diagnostic
  7. Benchmark vs IOZ.AX (iShares Core S&P/ASX 200) total return

Outputs all artefacts under reports/ and charts/ with an `asx_` prefix so
they sit beside the S&P 500 results.

Run::  uv run python -m qfr.backtest.asx_extension
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qfr.backtest.portfolio import (
    ANN,
    DEFAULT_COST_PER_SIDE,
    _spearman_ic_monthly,
    capm,
    composite_ic_horizons,
    cs_z,
    longshort_monthly,
    perf_metrics,
    portfolio_monthly,
    summary_metrics_table,
)
from qfr.data.fmp_client import FMPClient
from qfr.utils.config import PROJECT_ROOT
from qfr.utils.logging import logger

# Source PIT panel (built and maintained in the Short King 2.0 project)
SHORTKING_PIT = Path(r"C:\Users\Taffy Jackson\short-king-2.0\data\processed\master_pit.parquet")

# Map Short King 2.0 column names -> qfr factor names
COL_MAP = {
    "key_metrics_returnOnInvestedCapital": "returnOnInvestedCapital",
    "key_metrics_returnOnEquity": "returnOnEquity",
    "financial_growth_revenueGrowth": "revenueGrowth",
    "financial_growth_epsgrowth": "epsgrowth",
    # FCF yield computed below
}

TOP_FACTORS = ["returnOnInvestedCapital", "returnOnEquity", "freeCashFlowYield",
               "revenueGrowth", "epsgrowth"]

# PIT proxy for ASX 200 = top N by marketCap at each month-end
UNIVERSE_TOP_N = 200

# Benchmark
BENCHMARK_SYMBOL = "IOZ.AX"
BENCHMARK_START = "2010-01-01"
BENCHMARK_END = "2026-05-31"


# --------------------------------------------------------------------------
# Load + transform the source panel
# --------------------------------------------------------------------------
def load_asx_panel(top_n: int | None = UNIVERSE_TOP_N,
                   extra_lag_months: int = 0) -> pd.DataFrame:
    """Load weekly PIT panel, resample monthly, optionally keep top-N by cap each month.

    Parameters
    ----------
    top_n : int or None
        If provided, filter to top-N by marketCap at each month-end (PIT-200 proxy
        for the S&P/ASX 200). If None, keep all names with valid market cap (full
        ~500-ticker universe, useful as a robustness check).
    extra_lag_months : int
        Additional lag (months) to apply to the fundamental factors on top of the
        Short King 2.0 panel's existing PIT carry-forward. Use to stress-test
        whether results depend on a tight filing-date convention. 0 = baseline,
        2 = paranoia control.
    """
    label = f"top-{top_n}" if top_n else "full"
    logger.info(f"Loading ASX PIT panel ({label}, extra_lag={extra_lag_months}m) from {SHORTKING_PIT}")
    raw = pd.read_parquet(SHORTKING_PIT)

    # Compute FCF yield (FCF / market cap) before resampling
    raw["freeCashFlowYield"] = raw["cash_flow_freeCashFlow"] / raw["mktCap"]

    keep = ["Date", "Ticker", "sector", "mktCap", "adjClose", "fwd_ret_1m",
            "fwd_ret_3m", "freeCashFlowYield"] + list(COL_MAP.keys())
    keep = [c for c in keep if c in raw.columns]
    df = raw[keep].copy().rename(columns=COL_MAP)
    df["Date"] = pd.to_datetime(df["Date"])

    # Resample weekly -> monthly: keep the last weekly observation in each month per ticker
    df["month_end"] = df["Date"] + pd.offsets.MonthEnd(0)
    df = (df.sort_values(["Ticker", "Date"])
            .groupby(["Ticker", "month_end"]).tail(1)
            .copy())
    df = df.rename(columns={"month_end": "date", "Ticker": "symbol"}).drop(columns="Date")

    # Optional extra lag on the fundamental factors only (price / returns / mktCap stay at t)
    if extra_lag_months > 0:
        df = df.sort_values(["symbol", "date"])
        for c in TOP_FACTORS:
            if c in df.columns:
                df[c] = df.groupby("symbol")[c].shift(extra_lag_months)

    # Universe filter
    df = df.dropna(subset=["mktCap"])
    if top_n is not None:
        df["mc_rank"] = df.groupby("date")["mktCap"].rank(ascending=False, method="first")
        panel = df[df["mc_rank"] <= top_n].drop(columns="mc_rank")
    else:
        panel = df

    # qfr portfolio funcs expect ret_fwd_1m + marketCap (lowercase)
    panel = panel.rename(columns={"fwd_ret_1m": "ret_fwd_1m",
                                  "fwd_ret_3m": "ret_fwd_3m",
                                  "mktCap": "marketCap"})

    # Keep only dates with enough names with non-null returns + factors for robust quintiles
    cov = panel.groupby("date").apply(
        lambda g: g[["ret_fwd_1m"] + TOP_FACTORS].notna().all(axis=1).sum(),
        include_groups=False)
    valid_dates = cov[cov >= 50].index
    panel = panel[panel["date"].isin(valid_dates)].copy()

    logger.info(f"  monthly {label} panel: {len(panel):,} rows, "
                f"{panel['symbol'].nunique()} unique tickers, "
                f"{panel['date'].nunique()} months "
                f"({panel['date'].min().date()} -> {panel['date'].max().date()})")
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Composite + portfolios + IC
# --------------------------------------------------------------------------
def make_composite(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    zs = pd.DataFrame({c: cs_z(panel, c) for c in TOP_FACTORS})
    arr = zs.to_numpy()
    mask = ~np.isnan(arr)
    counts = mask.sum(axis=1)
    sums = np.where(mask, arr, 0).sum(axis=1)
    panel["composite"] = np.where(counts > 0, sums / counts, np.nan)
    return panel


def fetch_benchmark_monthly() -> pd.Series:
    """Monthly total return of IOZ.AX (S&P/ASX 200 total return ETF)."""
    rows = FMPClient().historical_prices(BENCHMARK_SYMBOL,
                                         from_date=BENCHMARK_START,
                                         to_date=BENCHMARK_END,
                                         series="dividend-adjusted")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date").sort_index()["adjClose"].resample("ME").last()
    return (monthly.shift(-1) / monthly - 1).rename(BENCHMARK_SYMBOL)


# --------------------------------------------------------------------------
# Per-factor + composite IC summary
# --------------------------------------------------------------------------
def per_factor_ic_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in TOP_FACTORS:
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


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def chart_cumulative_asx(long_only: list[dict], ls: dict, benchmark: pd.Series,
                         cost_per_side: float, out_name: str) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import save_fig, set_plot_style
    set_plot_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    colors = ["#1f3a5f", "#456b9a", "#0a7a3a", "#3aa56b"]
    for s, c in zip(long_only, colors):
        net = (s["gross"] - s["total_traded"] * cost_per_side).fillna(0)
        cum = (1 + net).cumprod()
        ax.plot(cum.index, cum.values, lw=1.8, color=c, label=s["name"])
    # Long-short overlay
    net_ls = (ls["gross"] - ls["total_traded"] * cost_per_side).fillna(0)
    cum_ls = (1 + net_ls).cumprod()
    ax.plot(cum_ls.index, cum_ls.values, lw=2.0, color="#bc4a3c",
            label=f"{ls['name']} (dollar-neutral, levered $1 long − $1 short)")
    # Benchmark
    bm = benchmark.reindex(long_only[0]["gross"].index).fillna(0)
    cum_bm = (1 + bm).cumprod()
    ax.plot(cum_bm.index, cum_bm.values, lw=1.6, color="#222", ls="--",
            label=f"{BENCHMARK_SYMBOL} (S&P/ASX 200 TR proxy)")
    ax.set_yscale("log")
    ax.set_title("ASX 200 — growth of $1: long-only quintile/decile, LS Q1−Q5 dollar-neutral, vs benchmark  (net of 10 bps/side)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_name)


def chart_ic_bars(per_factor_df: pd.DataFrame, comp_df: pd.DataFrame,
                  out_name: str) -> None:
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
                va="center", fontsize=9,
                ha="left" if v > 0 else "right")
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
    save_fig(fig, out_name)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def run_universe(panel: pd.DataFrame, benchmark: pd.Series, label_suffix: str,
                 ) -> tuple[pd.DataFrame, dict[str, dict], pd.DataFrame, pd.DataFrame]:
    """Run the full set of portfolios + IC tables on a single universe panel."""
    lo_dec_ew = portfolio_monthly(panel, "composite", n_buckets=10, weight="equal",
                                  name=f"Top decile EW {label_suffix}")
    lo_dec_cw = portfolio_monthly(panel, "composite", n_buckets=10, weight="cap",
                                  name=f"Top decile CW {label_suffix}")
    lo_qui_ew = portfolio_monthly(panel, "composite", n_buckets=5, weight="equal",
                                  name=f"Top quintile EW {label_suffix}")
    lo_qui_cw = portfolio_monthly(panel, "composite", n_buckets=5, weight="cap",
                                  name=f"Top quintile CW {label_suffix}")
    ls_q_ew = longshort_monthly(panel, "composite", n_buckets=5, weight="equal",
                                name=f"LS Q1-Q5 EW {label_suffix}")

    long_only = [lo_dec_ew, lo_dec_cw, lo_qui_ew, lo_qui_cw]
    summary_lo = summary_metrics_table(long_only, benchmark)
    summary_ls = summary_metrics_table([ls_q_ew], benchmark)

    bm_aligned = benchmark.reindex(lo_qui_cw["gross"].index).dropna()
    m_bm = perf_metrics(bm_aligned)
    bm_row = pd.DataFrame([{
        "strategy": f"{BENCHMARK_SYMBOL} {label_suffix}", "kind": "benchmark",
        "CAGR_%": m_bm["CAGR"] * 100, "ann_vol_%": m_bm["ann_vol"] * 100,
        "Sharpe": m_bm["Sharpe"], "max_drawdown_%": m_bm["max_drawdown"] * 100,
        "beta_vs_SPY": 1.0, "Jensen_alpha_%": 0.0,
        "ann_turnover_pa_%": np.nan, "cost_drag_pa_%": 0.0,
    }])
    summary_lo_full = pd.concat([summary_lo, bm_row], ignore_index=True)
    return summary_lo_full, {"lo": long_only, "ls": ls_q_ew}, summary_ls, per_factor_ic_table(panel)


def lag_sensitivity_ic(extra_lags: tuple[int, ...] = (0, 1, 2, 3)) -> pd.DataFrame:
    """Re-compute the composite IC under additional fundamentals lag.

    The Short King 2.0 panel already lags fundamentals to the most recent filing
    available at each weekly snapshot. This adds an EXTRA lag on top. If the IC
    collapses under extra lag, that flags a potential PIT issue. If the IC is
    broadly stable, the panel is clean.
    """
    rows = []
    for lag in extra_lags:
        p = load_asx_panel(top_n=UNIVERSE_TOP_N, extra_lag_months=lag)
        p = make_composite(p)
        df, _ = composite_ic_horizons(p, "composite")
        for _, r in df.iterrows():
            rows.append({
                "extra_lag_months": lag,
                "horizon_months": int(r["horizon_months"]),
                "mean_ic_%": r["mean_ic_%"],
                "t_stat": r["t_stat"],
                "hit_rate_%": r["hit_rate_%"],
                "n_months": int(r["n_months"]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    benchmark = fetch_benchmark_monthly()
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Universe 1: PIT top-200 by market cap (the headline universe) ---
    panel_200 = make_composite(load_asx_panel(top_n=UNIVERSE_TOP_N))
    comp_ic_200 = composite_ic_table(panel_200)
    summary_lo_200, strats_200, summary_ls_200, per_factor_200 = run_universe(
        panel_200, benchmark, "(top200)")

    per_factor_200.to_csv(out_dir / "asx_per_factor_ic.csv", index=False)
    comp_ic_200.to_csv(out_dir / "asx_composite_ic.csv", index=False)
    summary_lo_200.to_csv(out_dir / "asx_summary_long_only.csv", index=False)
    summary_ls_200.to_csv(out_dir / "asx_summary_long_short.csv", index=False)

    chart_cumulative_asx(strats_200["lo"], strats_200["ls"], benchmark,
                         DEFAULT_COST_PER_SIDE, "asx_cumulative")
    chart_ic_bars(per_factor_200, comp_ic_200, "asx_ic_bars")

    # --- Universe 2: Full ~500-ticker robustness check ---
    panel_full = make_composite(load_asx_panel(top_n=None))
    comp_ic_full = composite_ic_table(panel_full)
    summary_lo_full, _, summary_ls_full, per_factor_full = run_universe(
        panel_full, benchmark, "(full)")

    per_factor_full.to_csv(out_dir / "asx_per_factor_ic_full_universe.csv", index=False)
    comp_ic_full.to_csv(out_dir / "asx_composite_ic_full_universe.csv", index=False)
    summary_lo_full.to_csv(out_dir / "asx_summary_long_only_full_universe.csv", index=False)
    summary_ls_full.to_csv(out_dir / "asx_summary_long_short_full_universe.csv", index=False)

    # --- Lag sensitivity (PIT paranoia control) ---
    lag_df = lag_sensitivity_ic(extra_lags=(0, 1, 2, 3))
    lag_df.to_csv(out_dir / "asx_lag_sensitivity.csv", index=False)

    # Console summary
    logger.info(f"\n=== ASX top-200 per-factor rank IC ===\n{per_factor_200.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX top-200 composite rank IC by horizon ===\n{comp_ic_200.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX top-200 long-only summary (net of 10 bps/side, vs IOZ.AX) ===\n"
                f"{summary_lo_200.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX top-200 long-short summary ===\n{summary_ls_200.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX full universe per-factor rank IC ===\n{per_factor_full.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX full universe composite rank IC by horizon ===\n{comp_ic_full.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX full universe long-only summary ===\n{summary_lo_full.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX full universe long-short summary ===\n{summary_ls_full.round(3).to_string(index=False)}")
    logger.info(f"\n=== ASX lag sensitivity (extra months of fundamentals lag) ===\n"
                f"{lag_df.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
