"""Portfolio backtest of the top-5-factor composite vs SPY.

The composite is an equal-weight cross-sectional z-score of the five strongest
individual factors from the screen (ROIC, ROE, FCF yield, revenue growth, EPS
growth). Monthly rebalanced.

We test four families of strategies:

  1. Long-only buckets (top decile / top quintile, equal- and cap-weight) -
     the main practical implementation.
  2. Dollar-neutral long-short (D1-D10, Q1-Q5, EW and CW) - a diagnostic for
     whether the composite contains cross-sectional information.
  3. Beta-neutral long-short - rescales the short leg by trailing 36-month
     beta ratio so the realised portfolio beta is ~ zero. Diagnostic only.
  4. Sector-neutral long-short - within-GICS-sector quintile spreads
     aggregated equal-weight across sectors. Strips sector bets.

Plus a battery of analytics:

  - Cost sensitivity at 0, 5, 10, 25, 50 bps/side
  - Gross vs net decomposition (raw alpha vs cost drag)
  - Turnover breakdown (per leg, annualised, holding period)
  - CAPM regression vs SPY (Jensen alpha + beta)

Run::  uv run python -m qfr.backtest.portfolio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.factors.build import build_factor_panel
from qfr.utils.config import PROJECT_ROOT, settings
from qfr.utils.logging import logger

ANN = 12
DEFAULT_COST_PER_SIDE = 0.001  # 10 bps per side
COST_LEVELS_BPS = [0, 5, 10, 25, 50]
BETA_WINDOW = 36  # months for trailing beta estimation
IC_HORIZONS = (1, 2, 3, 6, 12)
IC_DECAY_MAX_LAG = 12
IC_MIN_NAMES = 20

TOP_FACTORS = [
    "returnOnInvestedCapital",   # quality
    "returnOnEquity",             # quality
    "freeCashFlowYield",          # value
    "revenueGrowth",              # growth
    "epsgrowth",                  # growth
]


# --------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------
def cs_z(panel: pd.DataFrame, col: str) -> pd.Series:
    g = panel.groupby("date")[col]
    return (panel[col] - g.transform("mean")) / g.transform("std")


def composite_score(panel: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Equal-weight cross-sectional z-score composite (nanmean across factors)."""
    zs = pd.DataFrame({c: cs_z(panel, c) for c in cols})
    arr = zs.to_numpy()
    mask = ~np.isnan(arr)
    counts = mask.sum(axis=1)
    sums = np.where(mask, arr, 0).sum(axis=1)
    return pd.Series(np.where(counts > 0, sums / counts, np.nan), index=panel.index)


# --------------------------------------------------------------------------
# Composite rank IC (Spearman) - the signal-quality diagnostic
#
# Per individual factor we already report rank ICs in factor_screen.csv;
# here we report the IC of the composite that we actually trade.
# --------------------------------------------------------------------------
def _spearman_ic_monthly(panel: pd.DataFrame, signal_col: str, ret_col: str,
                         min_names: int = IC_MIN_NAMES) -> pd.Series:
    """Per-month cross-sectional Spearman corr between signal_t and ret_col_t."""
    out: dict = {}
    sub_all = panel[["date", signal_col, ret_col]].dropna()
    for d, sub in sub_all.groupby("date"):
        if len(sub) >= min_names:
            ic = sub[signal_col].corr(sub[ret_col], method="spearman")
            if pd.notna(ic):
                out[d] = ic
    return pd.Series(out, name=f"IC_{ret_col}").sort_index()


def _ic_summary(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    if n == 0:
        return {"mean_ic": np.nan, "std_ic": np.nan, "ic_ir": np.nan,
                "t_stat": np.nan, "hit_rate": np.nan, "n_months": 0}
    sd = float(s.std())
    return {
        "mean_ic": float(s.mean()),
        "std_ic": sd,
        "ic_ir": float(s.mean() / sd) if sd > 0 else np.nan,
        "t_stat": float(s.mean() / sd * np.sqrt(n)) if sd > 0 else np.nan,
        "hit_rate": float((s > 0).mean()),
        "n_months": n,
    }


def composite_ic_horizons(panel: pd.DataFrame, signal_col: str = "composite",
                          horizons: tuple[int, ...] = IC_HORIZONS,
                          ) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Composite rank IC at multiple horizons (cumulative h-month forward return).

    For h not already in the panel as ret_fwd_{h}m, we compound monthly returns
    using shifted ret_fwd_1m. Returns a (summary_df, ic_series_by_horizon) pair.
    """
    p = panel.sort_values(["symbol", "date"]).copy()
    g1 = p.groupby("symbol")["ret_fwd_1m"]
    series_by_h: dict[int, pd.Series] = {}
    rows = []
    for h in horizons:
        col = f"_cum_ret_fwd_{h}m"
        if h == 1:
            p[col] = p["ret_fwd_1m"]
        elif f"ret_fwd_{h}m" in p.columns:
            p[col] = p[f"ret_fwd_{h}m"]
        else:
            cum = (1.0 + p["ret_fwd_1m"])
            for k in range(1, h):
                cum = cum * (1.0 + g1.shift(-k))
            p[col] = cum - 1.0
        ic_s = _spearman_ic_monthly(p, signal_col, col)
        series_by_h[h] = ic_s
        st = _ic_summary(ic_s)
        rows.append({
            "horizon_months": h,
            "mean_ic_%": st["mean_ic"] * 100,
            "ic_ir": st["ic_ir"],
            "t_stat": st["t_stat"],
            "hit_rate_%": st["hit_rate"] * 100,
            "n_months": st["n_months"],
        })
    return pd.DataFrame(rows), series_by_h


def composite_ic_decay(panel: pd.DataFrame, signal_col: str = "composite",
                       max_lag: int = IC_DECAY_MAX_LAG) -> pd.DataFrame:
    """Average IC at lags 1..max_lag (corr of signal_t with return *in* month t+lag-1).

    Lag 1 = ret in [t, t+1] (the contemporaneous forward month, same as 1m IC).
    Lag k = ret in [t+k-1, t+k] (the single-month return that far ahead).
    Use this to read 'how fast does the signal decay'.
    """
    p = panel.sort_values(["symbol", "date"]).copy()
    g = p.groupby("symbol")["ret_fwd_1m"]
    rows = []
    for lag in range(1, max_lag + 1):
        p["_r"] = g.shift(-(lag - 1))
        ic_s = _spearman_ic_monthly(p, signal_col, "_r")
        st = _ic_summary(ic_s)
        rows.append({
            "lag": lag,
            "avg_ic_%": st["mean_ic"] * 100,
            "t_stat": st["t_stat"],
            "hit_rate_%": st["hit_rate"] * 100,
            "n_months": st["n_months"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Helpers: bucket + weight
# --------------------------------------------------------------------------
def _assign_bucket(d: pd.DataFrame, score_col: str, n: int, group_cols: list[str]) -> pd.DataFrame:
    d = d.copy()
    d["bucket"] = d.groupby(group_cols)[score_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n, labels=False, duplicates="drop")
        if len(s) >= n else np.nan
    )
    return d


def _weight_within(d: pd.DataFrame, group_cols: list[str], weight: str) -> pd.DataFrame:
    d = d.copy()
    if weight == "equal":
        d["w"] = 1.0 / d.groupby(group_cols)["symbol"].transform("size")
    else:
        d["w"] = d["marketCap"] / d.groupby(group_cols)["marketCap"].transform("sum")
    return d


def _traded(weights_pivot: pd.DataFrame) -> pd.Series:
    """Total notional traded each month (= sum of |delta_w| across stocks)."""
    t = weights_pivot.diff().abs().sum(axis=1)
    if len(t):
        t.iloc[0] = weights_pivot.iloc[0].abs().sum()
    return t


# --------------------------------------------------------------------------
# Strategy builders (all return dicts: name, gross, total_traded, + LS legs)
# --------------------------------------------------------------------------
def portfolio_monthly(panel: pd.DataFrame, score_col: str, *,
                      n_buckets: int = 10, weight: str = "equal",
                      name: str | None = None) -> dict:
    """Long-only top-bucket portfolio, monthly rebalanced."""
    d = panel.dropna(subset=[score_col, "ret_fwd_1m", "marketCap"])
    d = _assign_bucket(d, score_col, n_buckets, ["date"])
    top = _weight_within(d[d["bucket"] == n_buckets - 1], ["date"], weight)
    gross = top.groupby("date").apply(
        lambda g: (g["w"] * g["ret_fwd_1m"]).sum(), include_groups=False)
    wmat = top.pivot_table(index="date", columns="symbol", values="w", fill_value=0.0).sort_index()
    traded = _traded(wmat)
    return {
        "name": name or f"Top {'decile' if n_buckets == 10 else 'quintile'} {weight.upper()[:2]}",
        "kind": "long_only",
        "gross": gross,
        "total_traded": traded,
        "top_traded": traded,
        "bot_traded": pd.Series(0.0, index=traded.index),
    }


def longshort_monthly(panel: pd.DataFrame, score_col: str, *,
                      n_buckets: int = 10, weight: str = "equal",
                      name: str | None = None) -> dict:
    """Dollar-neutral long-short: $1 long top bucket - $1 short bottom bucket."""
    d = panel.dropna(subset=[score_col, "ret_fwd_1m", "marketCap"])
    d = _assign_bucket(d, score_col, n_buckets, ["date"])
    top = _weight_within(d[d["bucket"] == n_buckets - 1], ["date"], weight)
    bot = _weight_within(d[d["bucket"] == 0], ["date"], weight)
    long_r = top.groupby("date").apply(
        lambda g: (g["w"] * g["ret_fwd_1m"]).sum(), include_groups=False)
    short_r = bot.groupby("date").apply(
        lambda g: (g["w"] * g["ret_fwd_1m"]).sum(), include_groups=False)
    gross = long_r - short_r
    top_w = top.pivot_table(index="date", columns="symbol", values="w", fill_value=0.0).sort_index()
    bot_w = bot.pivot_table(index="date", columns="symbol", values="w", fill_value=0.0).sort_index()
    traded_top, traded_bot = _traded(top_w), _traded(bot_w)
    traded = traded_top.add(traded_bot, fill_value=0)
    label = "D1-D10" if n_buckets == 10 else "Q1-Q5"
    return {
        "name": name or f"LS {label} {weight.upper()[:2]}",
        "kind": "long_short",
        "gross": gross,
        "total_traded": traded,
        "top_traded": traded_top,
        "bot_traded": traded_bot,
        "long_leg": long_r,
        "short_leg": short_r,
    }


def beta_neutral_ls(panel: pd.DataFrame, score_col: str, *, n_buckets: int = 10,
                    weight: str = "equal", spy_monthly: pd.Series,
                    window: int = BETA_WINDOW) -> dict:
    """Long-short with the short leg scaled by trailing-window beta_long/beta_short
    so the realised portfolio beta is approximately zero.

    Caveat: rolling-beta estimates on 36-month windows are noisy. First `window`
    months have no scale and are dropped from the net series.
    """
    base = longshort_monthly(panel, score_col, n_buckets=n_buckets, weight=weight)
    long_r = base["long_leg"]
    short_r = base["short_leg"]
    spy = spy_monthly.reindex(long_r.index)

    df = pd.concat([long_r.rename("L"), short_r.rename("S"), spy.rename("M")], axis=1).dropna()
    var_m = df["M"].rolling(window).var()
    beta_L = df["L"].rolling(window).cov(df["M"]) / var_m
    beta_S = df["S"].rolling(window).cov(df["M"]) / var_m
    # Use the prior-period estimate (no look-ahead)
    scale = (beta_L / beta_S).shift(1).clip(lower=0.25, upper=4.0)   # guard against noise
    gross = (df["L"] - scale * df["S"]).rename("bn_gross")

    # Approximate traded: long leg + |scale| × short leg's traded
    top_traded = base["top_traded"].reindex(gross.index)
    bot_traded_scaled = base["bot_traded"].reindex(gross.index) * scale.abs().fillna(1.0)
    traded = (top_traded + bot_traded_scaled).fillna(0)

    label = "D1-D10" if n_buckets == 10 else "Q1-Q5"
    return {
        "name": f"Beta-neutral LS {label} {weight.upper()[:2]}",
        "kind": "long_short",
        "gross": gross,
        "total_traded": traded,
        "top_traded": top_traded,
        "bot_traded": bot_traded_scaled,
        "scale_factor": scale,
    }


def sector_neutral_ls(panel: pd.DataFrame, score_col: str, *,
                      n_buckets: int = 5, weight: str = "equal") -> dict:
    """Within-sector quintile spreads, aggregated equal-weight across sectors.

    Strips sector bets - the residual is genuinely cross-sectional within-sector
    selection. Uses quintiles (n=5) by default since sector cross-sections are
    smaller (~40 names per sector, so a top-quintile gives ~8 stocks/leg).
    """
    d = panel.dropna(subset=[score_col, "ret_fwd_1m", "marketCap", "sector"])
    # Drop (date, sector) cells with too few stocks to bucket
    sec_size = d.groupby(["date", "sector"]).size()
    keep_idx = sec_size[sec_size >= n_buckets].index
    d = d.set_index(["date", "sector"]).loc[keep_idx].reset_index()

    d = _assign_bucket(d, score_col, n_buckets, ["date", "sector"])
    top = _weight_within(d[d["bucket"] == n_buckets - 1], ["date", "sector"], weight)
    bot = _weight_within(d[d["bucket"] == 0], ["date", "sector"], weight)

    # Each sector contributes 1 / n_sectors to the overall LS (equal-weight across sectors)
    n_sec = d.groupby("date")["sector"].nunique().rename("n_sec")
    top = top.merge(n_sec, on="date")
    bot = bot.merge(n_sec, on="date")
    top["w_scaled"] = top["w"] / top["n_sec"]
    bot["w_scaled"] = -bot["w"] / bot["n_sec"]

    long_r = top.groupby(["date", "sector"]).apply(
        lambda g: (g["w"] * g["ret_fwd_1m"]).sum(), include_groups=False)
    short_r = bot.groupby(["date", "sector"]).apply(
        lambda g: (g["w"] * g["ret_fwd_1m"]).sum(), include_groups=False)
    long_agg = long_r.unstack("sector").mean(axis=1)      # average across sectors (= equal-weight)
    short_agg = short_r.unstack("sector").mean(axis=1)
    gross = (long_agg - short_agg).rename("sn_gross")

    # Turnover: aggregate stock-level signed weights and diff
    combined = pd.concat([
        top[["date", "symbol", "w_scaled"]],
        bot[["date", "symbol", "w_scaled"]],
    ], ignore_index=True)
    stock_w = combined.groupby(["date", "symbol"])["w_scaled"].sum().unstack("symbol").fillna(0.0).sort_index()
    traded = _traded(stock_w)

    label = "Q1-Q5" if n_buckets == 5 else f"D1-D{n_buckets}"
    return {
        "name": f"Sector-neutral LS {label} {weight.upper()[:2]}",
        "kind": "long_short",
        "gross": gross,
        "total_traded": traded,
        "top_traded": traded * 0.5,  # symmetric estimate
        "bot_traded": traded * 0.5,
    }


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------
def universe_return(panel: pd.DataFrame, *, weight: str = "cap") -> pd.Series:
    d = panel.dropna(subset=["ret_fwd_1m", "marketCap"]).copy()
    if weight == "cap":
        d["w"] = d["marketCap"] / d.groupby("date")["marketCap"].transform("sum")
    else:
        d["w"] = 1.0 / d.groupby("date")["symbol"].transform("size")
    return d.groupby("date").apply(
        lambda g: (g["w"] * g["ret_fwd_1m"]).sum(), include_groups=False
    )


def fetch_spy_monthly(start: str = "2009-12-01", end: str = "2026-05-31") -> pd.Series:
    from qfr.data.fmp_client import FMPClient
    rows = FMPClient().historical_prices("SPY", from_date=start, to_date=end,
                                         series="dividend-adjusted")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date").sort_index()["adjClose"].resample("ME").last()
    return (monthly.shift(-1) / monthly - 1).rename("SPY")


# --------------------------------------------------------------------------
# Metrics & analytics
# --------------------------------------------------------------------------
def perf_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {}
    cum = float((1 + r).prod())
    sd = float(r.std())
    sharpe = float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan
    cc = (1 + r).cumprod()
    dd = float((cc / cc.cummax() - 1).min())
    return dict(
        total_return=cum - 1, CAGR=cum ** (ANN / n) - 1,
        ann_vol=float(sd * np.sqrt(ANN)), Sharpe=sharpe,
        max_drawdown=dd, hit_rate=float((r > 0).mean()), n_months=int(n),
    )


def capm(r: pd.Series, mkt: pd.Series) -> tuple[float, float]:
    df = pd.concat([r, mkt], axis=1).dropna()
    if len(df) < 12:
        return np.nan, np.nan
    x, y = df.iloc[:, 1].to_numpy(), df.iloc[:, 0].to_numpy()
    beta, intercept = np.polyfit(x, y, 1)
    return float(beta), float(intercept * ANN)


def net_returns(strat: dict, cost_per_side: float) -> pd.Series:
    return (strat["gross"] - strat["total_traded"] * cost_per_side).rename(strat["name"])


# Cost sensitivity
def cost_sensitivity_table(strats: list[dict], spy: pd.Series,
                           cost_levels_bps: list[int] = COST_LEVELS_BPS) -> pd.DataFrame:
    """For each (strategy, cost level) report CAGR/vol/Sharpe/MaxDD/turnover/cost-drag/alpha."""
    rows = []
    for s in strats:
        # capm returns (beta, alpha_annualised_as_decimal). gross_alpha_pct is the
        # annualised Jensen alpha of the gross series, expressed as a percent.
        beta, gross_alpha_decimal = capm(s["gross"], spy)
        gross_alpha_pct = gross_alpha_decimal * 100 if pd.notna(gross_alpha_decimal) else np.nan
        for c_bps in cost_levels_bps:
            cps = c_bps / 10000
            net = s["gross"] - s["total_traded"] * cps
            cost_drag_ann = (s["total_traded"] * cps).mean() * ANN  # decimal
            cost_drag_pct = cost_drag_ann * 100
            m = perf_metrics(net.dropna())
            net_alpha_pct = (gross_alpha_pct - cost_drag_pct) if pd.notna(gross_alpha_pct) else np.nan
            rows.append({
                "strategy": s["name"], "cost_bps": c_bps,
                "CAGR_%": m.get("CAGR", np.nan) * 100,
                "ann_vol_%": m.get("ann_vol", np.nan) * 100,
                "Sharpe": m.get("Sharpe", np.nan),
                "max_drawdown_%": m.get("max_drawdown", np.nan) * 100,
                "ann_turnover_%": s["total_traded"].mean() * 0.5 * ANN * 100,  # one-way ann
                "cost_drag_%": cost_drag_pct,
                "Jensen_alpha_%": net_alpha_pct,
                "beta": beta,
            })
    return pd.DataFrame(rows)


def gross_vs_net_table(strats: list[dict], cost_per_side: float = DEFAULT_COST_PER_SIDE) -> pd.DataFrame:
    rows = []
    for s in strats:
        gross = s["gross"]
        net = gross - s["total_traded"] * cost_per_side
        cost_drag_ann = (s["total_traded"] * cost_per_side).mean() * ANN
        mg, mn = perf_metrics(gross), perf_metrics(net.dropna())
        rows.append({
            "strategy": s["name"],
            "gross_CAGR_%": mg.get("CAGR", np.nan) * 100,
            "cost_drag_pa_%": cost_drag_ann * 100,
            "net_CAGR_%": mn.get("CAGR", np.nan) * 100,
            "gross_Sharpe": mg.get("Sharpe", np.nan),
            "net_Sharpe": mn.get("Sharpe", np.nan),
            "ann_turnover_pa_%": s["total_traded"].mean() * 0.5 * ANN * 100,
        })
    return pd.DataFrame(rows)


def turnover_analysis(strats: list[dict]) -> pd.DataFrame:
    rows = []
    for s in strats:
        tt = s["total_traded"].mean()
        tt_long = s.get("top_traded", pd.Series([np.nan])).mean()
        tt_short = s.get("bot_traded", pd.Series([np.nan])).mean()
        one_way = 0.5 * tt
        # holding period in months ~ 1 / one_way_turnover  (clip to avoid div-by-zero)
        hp = float(1.0 / max(one_way, 1e-9))
        rows.append({
            "strategy": s["name"], "kind": s["kind"],
            "monthly_one_way_turnover_%": one_way * 100,
            "ann_turnover_pa_%": one_way * ANN * 100,
            "long_leg_monthly_turnover_%": tt_long * 0.5 * 100 if pd.notna(tt_long) else np.nan,
            "short_leg_monthly_turnover_%": tt_short * 0.5 * 100 if pd.notna(tt_short) else np.nan,
            "avg_holding_period_months": hp,
        })
    return pd.DataFrame(rows)


def summary_metrics_table(strats: list[dict], spy: pd.Series,
                          cost_per_side: float = DEFAULT_COST_PER_SIDE) -> pd.DataFrame:
    rows = []
    for s in strats:
        net = (s["gross"] - s["total_traded"] * cost_per_side).dropna()
        m = perf_metrics(net)
        beta, alpha_ann = capm(net, spy)
        rows.append({
            "strategy": s["name"], "kind": s["kind"],
            "CAGR_%": m["CAGR"] * 100, "ann_vol_%": m["ann_vol"] * 100,
            "Sharpe": m["Sharpe"], "max_drawdown_%": m["max_drawdown"] * 100,
            "beta_vs_SPY": beta, "Jensen_alpha_%": alpha_ann * 100,
            "ann_turnover_pa_%": s["total_traded"].mean() * 0.5 * ANN * 100,
            "cost_drag_pa_%": (s["total_traded"] * cost_per_side).mean() * ANN * 100,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def _save(fig, name: str) -> None:
    from qfr.utils.viz import save_fig
    save_fig(fig, name)


def chart_cumulative_long_only(strats: list[dict], spy: pd.Series, u_ew: pd.Series,
                               cost_per_side: float) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    colors = ["#1f3a5f", "#456b9a", "#0a7a3a", "#3aa56b"]
    for s, c in zip(strats, colors):
        net = (s["gross"] - s["total_traded"] * cost_per_side).fillna(0)
        cum = (1 + net).cumprod()
        ax.plot(cum.index, cum.values, lw=1.8, color=c, label=s["name"])
    ax.plot(spy.index, (1 + spy.fillna(0)).cumprod().values, lw=1.8, color="#222", ls="--", label="SPY (total return)")
    ax.plot(u_ew.index, (1 + u_ew.fillna(0)).cumprod().values, lw=1.4, color="#888", ls=":", label="Equal-weighted universe")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 — 5-factor composite long-only portfolios vs SPY (2010+, net of 10 bps/side)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(fontsize=8.5, loc="upper left")
    _save(fig, "backtest_cumulative")


def chart_longshort(strats: list[dict], cost_per_side: float) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    colors = ["#1f3a5f", "#456b9a", "#0a7a3a", "#3aa56b"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 2])
    for s, c in zip(strats, colors):
        net = (s["gross"] - s["total_traded"] * cost_per_side).fillna(0)
        cum = (1 + net).cumprod()
        a1.plot(cum.index, cum.values, lw=1.7, color=c, label=s["name"])
    a1.axhline(1.0, color="#444", lw=0.7, ls="--", label="dollar-neutral baseline")
    a1.set_title("Long-short diagnostics — D1 long − D10 short / Q1 long − Q5 short  (dollar-neutral, net of 10 bps/side)",
                 fontsize=12, fontweight="bold", loc="left")
    a1.set_ylabel("Growth of $1 (linear)")
    a1.legend(fontsize=8.5, loc="upper left")
    for s, c in zip(strats, colors):
        net = (s["gross"] - s["total_traded"] * cost_per_side).fillna(0)
        cum = (1 + net).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        a2.plot(dd.index, dd.values, lw=1.2, color=c, label=s["name"])
    a2.axhline(0, color="#444", lw=0.7)
    a2.set_title("Long-short drawdowns (net)", fontsize=11, fontweight="bold", loc="left")
    a2.set_ylabel("Drawdown (%)")
    fig.tight_layout()
    _save(fig, "backtest_longshort")


def chart_drawdowns(strats: list[dict], spy: pd.Series, cost_per_side: float) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ["#1f3a5f", "#456b9a", "#0a7a3a", "#3aa56b"]
    for s, c in zip(strats, colors):
        net = (s["gross"] - s["total_traded"] * cost_per_side).fillna(0)
        cum = (1 + net).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, lw=1.3, color=c, label=s["name"])
    cum_spy = (1 + spy.fillna(0)).cumprod()
    dd_spy = (cum_spy / cum_spy.cummax() - 1) * 100
    ax.plot(dd_spy.index, dd_spy.values, lw=1.5, color="#222", ls="--", label="SPY")
    ax.axhline(0, color="#444", lw=0.7)
    ax.set_title("Drawdowns (net of 10 bps/side)", fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8.5)
    _save(fig, "backtest_drawdowns")


def chart_cost_sensitivity(cs_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, sub in cs_df.groupby("strategy"):
        sub = sub.sort_values("cost_bps")
        a1.plot(sub["cost_bps"], sub["Sharpe"], marker="o", lw=1.7, label=name)
        a2.plot(sub["cost_bps"], sub["CAGR_%"], marker="o", lw=1.7, label=name)
    for ax, ylab in zip([a1, a2], ["Sharpe ratio (net)", "CAGR (%, net)"]):
        ax.set_xlabel("Transaction cost (bps per side)")
        ax.set_ylabel(ylab)
        ax.axhline(0, color="#444", lw=0.6)
        ax.grid(alpha=0.3)
    a1.legend(fontsize=8, loc="upper right")
    fig.suptitle("Long-short cost sensitivity — Sharpe and CAGR by transaction-cost assumption",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "backtest_cost_sensitivity")


def chart_gross_vs_net(gn_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    df = gn_df.set_index("strategy").sort_values("gross_CAGR_%")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(df) + 1.5)))
    y = np.arange(len(df))
    ax.barh(y, df["gross_CAGR_%"], color="#1f3a5f", alpha=0.85, label="Gross CAGR")
    ax.barh(y, df["net_CAGR_%"], color="#3aa56b", alpha=0.85, label="Net CAGR")
    for yi, (g, n) in enumerate(zip(df["gross_CAGR_%"], df["net_CAGR_%"])):
        drag = g - n
        ax.text(max(g, n) + 0.05, yi, f"  drag = {drag:.2f}%", va="center", fontsize=8, color="#555")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=9)
    ax.axvline(0, color="#444", lw=0.7)
    ax.set_xlabel("CAGR (%)")
    ax.set_title("Gross vs net CAGR — long-short portfolios at 10 bps/side", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, "backtest_gross_vs_net")


def chart_turnover(to_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    df = to_df.set_index("strategy")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(df) + 1.5)))
    y = np.arange(len(df))
    long = df["long_leg_monthly_turnover_%"].fillna(0)
    short = df["short_leg_monthly_turnover_%"].fillna(0)
    ax.barh(y, long, color="#1f3a5f", alpha=0.85, label="Long-leg monthly turnover")
    ax.barh(y, short, left=long, color="#bc4a3c", alpha=0.85, label="Short-leg monthly turnover")
    for yi, (l, s, hp) in enumerate(zip(long, short, df["avg_holding_period_months"])):
        ax.text(l + s + 0.5, yi, f"  ~{hp:.1f} mo holding", va="center", fontsize=8, color="#555")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlabel("Monthly one-way turnover (%)")
    ax.set_title("Turnover decomposition — monthly one-way per leg", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, "backtest_turnover")


def chart_ls_neutral_variants(ls_dollar: dict, ls_beta: dict, ls_sector: dict,
                              cost_per_side: float) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    variants = [
        ("Dollar-neutral", ls_dollar, "#1f3a5f"),
        ("Beta-neutral (trailing 36m)", ls_beta, "#bc4a3c"),
        ("Sector-neutral (within-GICS)", ls_sector, "#0a7a3a"),
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    for label, s, c in variants:
        net = (s["gross"] - s["total_traded"] * cost_per_side).dropna()
        cum = (1 + net.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, lw=1.7, color=c, label=f"{label} ({s['name']})")
    ax.axhline(1.0, color="#444", lw=0.7, ls="--")
    ax.set_title("Long-short neutrality variants — dollar-neutral vs beta-neutral vs sector-neutral  (net of 10 bps/side)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Growth of $1 (linear)")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    _save(fig, "backtest_ls_neutral_variants")


def chart_composite_ic_monthly(ic1m: pd.Series) -> None:
    """Monthly composite rank IC bars + 12m rolling mean + summary stats box."""
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    st = _ic_summary(ic1m)
    roll = ic1m.rolling(12, min_periods=6).mean()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(ic1m.index, ic1m.values * 100, width=22, color="#bcbcbc", alpha=0.9,
           label="Monthly rank IC")
    ax.plot(roll.index, roll.values * 100, lw=1.8, color="#1f3a5f",
            label="12m rolling avg")
    ax.axhline(0, color="#444", lw=0.7)
    ax.set_title("Composite rank IC — Spearman corr of 5-factor composite with 1m forward returns",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Rank IC (%)")
    ax.text(0.985, 0.97,
            f"mean IC   = {st['mean_ic'] * 100:.2f} %\n"
            f"IC IR     = {st['ic_ir']:.2f}\n"
            f"t-stat    = {st['t_stat']:.2f}\n"
            f"hit rate  = {st['hit_rate'] * 100:.1f} %\n"
            f"n months  = {st['n_months']}",
            transform=ax.transAxes, ha="right", va="top",
            family="monospace", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.93))
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, "composite_rank_ic")


def chart_composite_ic_decay(decay_df: pd.DataFrame) -> None:
    """Composite IC at lags 1..12 — how fast does the signal decay?"""
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#1f3a5f" if v >= 0 else "#bc4a3c" for v in decay_df["avg_ic_%"]]
    ax.bar(decay_df["lag"], decay_df["avg_ic_%"], color=colors, alpha=0.85)
    for x, y, t in zip(decay_df["lag"], decay_df["avg_ic_%"], decay_df["t_stat"]):
        offset = 0.04 if y >= 0 else -0.08
        ax.text(x, y + offset, f"{y:.2f}\nt={t:.2f}",
                ha="center", fontsize=8, color="#333")
    ax.axhline(0, color="#444", lw=0.7)
    ax.set_xticks(list(decay_df["lag"]))
    ax.set_xlabel("Lag (months ahead) — single-month return at that lag")
    ax.set_ylabel("Avg rank IC (%)")
    ax.set_title("Composite IC decay — avg single-month rank IC by lag",
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    _save(fig, "composite_ic_decay")


def chart_composite_ic_horizons(horizon_df: pd.DataFrame) -> None:
    """Composite IC vs cumulative forward-return horizons (1/2/3/6/12 months)."""
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(horizon_df))
    ax.bar(xs, horizon_df["mean_ic_%"], color="#1f3a5f", alpha=0.85)
    for i, (mic, t, hit) in enumerate(zip(horizon_df["mean_ic_%"],
                                          horizon_df["t_stat"],
                                          horizon_df["hit_rate_%"])):
        ax.text(i, mic + 0.05, f"{mic:.2f}%\nt={t:.2f}\nhit={hit:.0f}%",
                ha="center", fontsize=8.5)
    ax.axhline(0, color="#444", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{h}m" for h in horizon_df["horizon_months"]])
    ax.set_xlabel("Forward-return horizon (cumulative)")
    ax.set_ylabel("Mean rank IC (%)")
    ax.set_title("Composite rank IC by forward-return horizon",
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    _save(fig, "composite_ic_horizons")


def render_metrics_table(df: pd.DataFrame, title: str, out_name: str) -> None:
    import matplotlib.pyplot as plt
    keys = ["CAGR_%", "ann_vol_%", "Sharpe", "max_drawdown_%",
            "beta_vs_SPY", "Jensen_alpha_%", "ann_turnover_pa_%", "cost_drag_pa_%"]
    headers = ["CAGR", "Ann. Vol", "Sharpe", "Max DD",
               "β vs SPY", "Jensen α", "Ann. Turnover", "Cost Drag"]
    disp = df.set_index("strategy")[keys]

    def fmt(k, v):
        if pd.isna(v):
            return "—"
        if k in ("CAGR_%", "ann_vol_%", "max_drawdown_%",
                 "Jensen_alpha_%", "ann_turnover_pa_%", "cost_drag_pa_%"):
            return f"{v:.2f}%"
        return f"{v:.2f}"

    cell_text = [[fmt(k, r[k]) for k in keys] for _, r in disp.iterrows()]
    row_labels = disp.index.tolist()
    fig, ax = plt.subplots(figsize=(15, 0.45 * len(row_labels) + 1.7))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=14)
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=headers,
                   cellLoc="center", loc="center",
                   colWidths=[0.08, 0.07, 0.07, 0.07, 0.08, 0.09, 0.10, 0.09])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for j in range(len(headers)):
        c = tbl[0, j]
        c.set_facecolor("#1f3a5f")
        c.set_text_props(color="white", fontweight="bold")
    for i, (_, r) in enumerate(disp.iterrows()):
        rl = tbl[i + 1, -1]
        rl.set_text_props(ha="right", fontweight="bold")
        rl.set_facecolor("#e9ecf2")
        alpha = r["Jensen_alpha_%"]
        if pd.notna(alpha) and alpha > 1.0:
            for j in range(len(headers)):
                tbl[i + 1, j].set_facecolor("#cfe6cf")
        elif pd.notna(alpha) and alpha < -1.0:
            for j in range(len(headers)):
                tbl[i + 1, j].set_facecolor("#f4d6d2")
    fig.text(0.02, 0.02,
             "Composite = equal-weight cross-sectional z-score of ROIC, ROE, FCF yield, "
             "Revenue growth, EPS growth.  Monthly rebalance.  10 bps/side costs.",
             fontsize=8.5, style="italic", color="#555")
    _save(fig, out_name)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main() -> None:
    panel = build_factor_panel().sort_values(["date", "symbol"]).reset_index(drop=True)
    panel["composite"] = composite_score(panel, TOP_FACTORS)
    logger.info(f"Composite of top 5 factors: {TOP_FACTORS}")

    # Composite rank IC (Spearman) - signal-quality diagnostic for the blended signal
    horizon_df, _ic_series_by_h = composite_ic_horizons(panel, "composite")
    decay_df = composite_ic_decay(panel, "composite")
    ic_1m_series = _spearman_ic_monthly(panel, "composite", "ret_fwd_1m")

    # Long-only
    lo_dec_ew = portfolio_monthly(panel, "composite", n_buckets=10, weight="equal", name="Top decile EW")
    lo_dec_cw = portfolio_monthly(panel, "composite", n_buckets=10, weight="cap",   name="Top decile CW")
    lo_qui_ew = portfolio_monthly(panel, "composite", n_buckets=5,  weight="equal", name="Top quintile EW")
    lo_qui_cw = portfolio_monthly(panel, "composite", n_buckets=5,  weight="cap",   name="Top quintile CW")

    # Long-short (dollar-neutral)
    ls_d_ew = longshort_monthly(panel, "composite", n_buckets=10, weight="equal", name="LS D1-D10 EW")
    ls_d_cw = longshort_monthly(panel, "composite", n_buckets=10, weight="cap",   name="LS D1-D10 CW")
    ls_q_ew = longshort_monthly(panel, "composite", n_buckets=5,  weight="equal", name="LS Q1-Q5 EW")
    ls_q_cw = longshort_monthly(panel, "composite", n_buckets=5,  weight="cap",   name="LS Q1-Q5 CW")

    long_only_strats = [lo_dec_ew, lo_dec_cw, lo_qui_ew, lo_qui_cw]
    ls_strats = [ls_d_ew, ls_d_cw, ls_q_ew, ls_q_cw]

    spy = fetch_spy_monthly()
    u_ew = universe_return(panel, weight="equal")

    # Neutrality variants (use Q1-Q5 EW as the diagnostic base)
    ls_q_ew_beta = beta_neutral_ls(panel, "composite", n_buckets=5, weight="equal", spy_monthly=spy)
    ls_q_ew_sector = sector_neutral_ls(panel, "composite", n_buckets=5, weight="equal")

    # Tables
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    summary_long_only = summary_metrics_table(long_only_strats, spy)
    summary_ls = summary_metrics_table(ls_strats, spy)
    summary_neutral = summary_metrics_table([ls_q_ew, ls_q_ew_beta, ls_q_ew_sector], spy)
    cs_table = cost_sensitivity_table(ls_strats, spy)
    gn_table = gross_vs_net_table(ls_strats)
    to_table = turnover_analysis(long_only_strats + ls_strats)

    summary_long_only.to_csv(PROJECT_ROOT / "reports" / "backtest_summary_long_only.csv", index=False)
    summary_ls.to_csv(PROJECT_ROOT / "reports" / "backtest_summary_long_short.csv", index=False)
    summary_neutral.to_csv(PROJECT_ROOT / "reports" / "backtest_ls_neutral_variants.csv", index=False)
    cs_table.to_csv(PROJECT_ROOT / "reports" / "backtest_cost_sensitivity.csv", index=False)
    gn_table.to_csv(PROJECT_ROOT / "reports" / "backtest_gross_vs_net.csv", index=False)
    to_table.to_csv(PROJECT_ROOT / "reports" / "backtest_turnover.csv", index=False)

    horizon_df.to_csv(PROJECT_ROOT / "reports" / "composite_rank_ic_horizons.csv", index=False)
    decay_df.to_csv(PROJECT_ROOT / "reports" / "composite_ic_decay.csv", index=False)
    ic_1m_series.rename("composite_rank_ic").to_frame().to_csv(
        PROJECT_ROOT / "reports" / "composite_rank_ic_monthly.csv")

    # Charts
    chart_cumulative_long_only(long_only_strats, spy, u_ew, DEFAULT_COST_PER_SIDE)
    chart_longshort(ls_strats, DEFAULT_COST_PER_SIDE)
    chart_drawdowns(long_only_strats + [ls_q_ew], spy, DEFAULT_COST_PER_SIDE)
    chart_cost_sensitivity(cs_table)
    chart_gross_vs_net(gn_table)
    chart_turnover(to_table)
    chart_ls_neutral_variants(ls_q_ew, ls_q_ew_beta, ls_q_ew_sector, DEFAULT_COST_PER_SIDE)
    chart_composite_ic_monthly(ic_1m_series)
    chart_composite_ic_decay(decay_df)
    chart_composite_ic_horizons(horizon_df)

    render_metrics_table(summary_long_only, "Long-only portfolios — 5-factor composite vs SPY (net of 10 bps/side)", "backtest_metrics")
    render_metrics_table(summary_ls, "Long-short diagnostics — dollar-neutral spreads on the 5-factor composite", "backtest_metrics_longshort")
    render_metrics_table(summary_neutral, "Long-short neutrality variants — dollar-, beta- and sector-neutral", "backtest_metrics_neutral_variants")

    logger.info(f"\n=== Long-only summary ===\n{summary_long_only.round(3).to_string(index=False)}")
    logger.info(f"\n=== Long-short summary ===\n{summary_ls.round(3).to_string(index=False)}")
    logger.info(f"\n=== Neutrality variants ===\n{summary_neutral.round(3).to_string(index=False)}")
    logger.info(f"\n=== Cost sensitivity ===\n{cs_table.round(3).to_string(index=False)}")
    logger.info(f"\n=== Gross vs net ===\n{gn_table.round(3).to_string(index=False)}")
    logger.info(f"\n=== Turnover ===\n{to_table.round(3).to_string(index=False)}")
    logger.info(f"\n=== Composite rank IC by horizon ===\n{horizon_df.round(3).to_string(index=False)}")
    logger.info(f"\n=== Composite IC decay ===\n{decay_df.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
