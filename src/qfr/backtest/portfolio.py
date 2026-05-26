"""Portfolio backtest: top-factor composite vs the S&P 500.

Picks the 5 strongest individual factors from the screen (ROIC, ROE, FCF yield,
Revenue growth, EPS growth - the "Top-5" composite that achieved a 1m-IC t-stat
of 2.44 and a 2m-IC t-stat of 3.17), forms an equal-weight z-score composite,
and back-tests four long-only monthly-rebalanced portfolios vs SPY:

    Top decile, equal-weight within     ~47 names, concentrated
    Top decile, cap-weight within       ~47 names, large-cap tilt
    Top quintile, equal-weight within   ~94 names, more diversified
    Top quintile, cap-weight within     ~94 names, large-cap tilt

Each portfolio is charged 10 bps per side on traded notional (so a name turning
over once costs 20 bps round-trip). The benchmark is SPY total return; we also
report Jensen's alpha + beta vs SPY.

Run::  uv run python -m qfr.backtest.portfolio
Outputs: charts/backtest_cumulative.png, charts/backtest_drawdowns.png,
         charts/backtest_metrics.png, reports/backtest_metrics.csv,
         reports/backtest_monthly_returns.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.factors.build import build_factor_panel
from qfr.utils.config import PROJECT_ROOT, settings
from qfr.utils.logging import logger

ANN = 12
COST_PER_SIDE = 0.001  # 10 bps per side

# The 5 strongest individual factors per Table 5 (max |t| from 1m IC, 2m IC, pure)
TOP_FACTORS = [
    "returnOnInvestedCapital",   # quality
    "returnOnEquity",             # quality
    "freeCashFlowYield",          # value
    "revenueGrowth",              # growth
    "epsgrowth",                  # growth
]


# --------------------------------------------------------------------------
# Signal construction
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
# Portfolio construction
# --------------------------------------------------------------------------
def portfolio_monthly(panel: pd.DataFrame, score_col: str, *,
                      n_buckets: int = 10, weight: str = "equal"):
    """Long-only top-bucket portfolio, monthly rebalanced.

    Returns (gross_returns, net_returns, one_way_turnover) as monthly series.
    """
    d = panel.dropna(subset=[score_col, "ret_fwd_1m", "marketCap"]).copy()
    d["bucket"] = d.groupby("date")[score_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_buckets, labels=False)
    )
    top = d[d["bucket"] == n_buckets - 1].copy()
    if weight == "equal":
        top["w"] = 1.0 / top.groupby("date")["symbol"].transform("size")
    else:  # cap
        top["w"] = top["marketCap"] / top.groupby("date")["marketCap"].transform("sum")

    gross = top.groupby("date").apply(
        lambda g: (g["w"] * g["ret_fwd_1m"]).sum(), include_groups=False
    )
    wmat = top.pivot_table(index="date", columns="symbol", values="w", fill_value=0.0).sort_index()
    traded = wmat.diff().abs().sum(axis=1)   # total volume traded (buys + sells)
    if len(traded):
        traded.iloc[0] = wmat.iloc[0].abs().sum()  # initial build
    cost = traded * COST_PER_SIDE
    net = gross - cost
    one_way = 0.5 * traded                   # one-way turnover (= % portfolio replaced)
    return gross, net, one_way


def longshort_monthly(panel: pd.DataFrame, score_col: str, *,
                      n_buckets: int = 10, weight: str = "equal"):
    """Dollar-neutral long-short: $1 long top bucket - $1 short bottom bucket.

    Monthly rebalanced; transaction costs charged on BOTH legs (long + short)
    at 10 bps per side of traded notional. Returns (gross, net, one_way_turnover)
    where turnover is the combined fraction of the gross 2x exposure rebalanced.
    """
    d = panel.dropna(subset=[score_col, "ret_fwd_1m", "marketCap"]).copy()
    d["bucket"] = d.groupby("date")[score_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_buckets, labels=False)
    )
    top = d[d["bucket"] == n_buckets - 1].copy()
    bot = d[d["bucket"] == 0].copy()

    def wcol(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if weight == "equal":
            df["w"] = 1.0 / df.groupby("date")["symbol"].transform("size")
        else:
            df["w"] = df["marketCap"] / df.groupby("date")["marketCap"].transform("sum")
        return df

    top, bot = wcol(top), wcol(bot)
    long_r = top.groupby("date").apply(
        lambda g: (g["w"] * g["ret_fwd_1m"]).sum(), include_groups=False)
    short_r = bot.groupby("date").apply(
        lambda g: (g["w"] * g["ret_fwd_1m"]).sum(), include_groups=False)
    gross = long_r - short_r

    top_w = top.pivot_table(index="date", columns="symbol", values="w", fill_value=0.0).sort_index()
    bot_w = bot.pivot_table(index="date", columns="symbol", values="w", fill_value=0.0).sort_index()
    traded_top = top_w.diff().abs().sum(axis=1)
    traded_bot = bot_w.diff().abs().sum(axis=1)
    if len(traded_top):
        traded_top.iloc[0] = top_w.iloc[0].abs().sum()
    if len(traded_bot):
        traded_bot.iloc[0] = bot_w.iloc[0].abs().sum()
    traded = traded_top.add(traded_bot, fill_value=0)
    cost = traded * COST_PER_SIDE
    net = gross - cost
    one_way = 0.5 * traded
    return gross, net, one_way


def universe_return(panel: pd.DataFrame, *, weight: str = "cap") -> pd.Series:
    d = panel.dropna(subset=["ret_fwd_1m", "marketCap"]).copy()
    if weight == "cap":
        d["w"] = d["marketCap"] / d.groupby("date")["marketCap"].transform("sum")
    else:
        d["w"] = 1.0 / d.groupby("date")["symbol"].transform("size")
    return d.groupby("date").apply(
        lambda g: (g["w"] * g["ret_fwd_1m"]).sum(), include_groups=False
    )


# --------------------------------------------------------------------------
# SPY (true S&P 500 benchmark) - pull from FMP
# --------------------------------------------------------------------------
def fetch_spy_monthly(start: str = "2009-12-01", end: str = "2026-05-31") -> pd.Series:
    """SPY total return month-end series, aligned to our ret_fwd_1m convention."""
    from qfr.data.fmp_client import FMPClient
    rows = FMPClient().historical_prices("SPY", from_date=start, to_date=end,
                                         series="dividend-adjusted")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date").sort_index()["adjClose"].resample("ME").last()
    # ret_fwd_1m at month-end t = price(t+1)/price(t) - 1
    return (monthly.shift(-1) / monthly - 1).rename("SPY")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def perf_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {}
    cum = float((1 + r).prod())
    cagr = cum ** (ANN / n) - 1
    sd = float(r.std())
    sharpe = float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan
    cc = (1 + r).cumprod()
    dd = float((cc / cc.cummax() - 1).min())
    return dict(total_return=cum - 1, CAGR=cagr,
                ann_vol=float(sd * np.sqrt(ANN)), Sharpe=sharpe,
                max_drawdown=dd, hit_rate=float((r > 0).mean()), n_months=int(n))


def capm(r: pd.Series, mkt: pd.Series) -> tuple[float, float]:
    df = pd.concat([r, mkt], axis=1).dropna()
    if len(df) < 12:
        return np.nan, np.nan
    x, y = df.iloc[:, 1].to_numpy(), df.iloc[:, 0].to_numpy()
    beta, intercept = np.polyfit(x, y, 1)
    return float(beta), float(intercept * ANN)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main() -> None:
    panel = build_factor_panel().sort_values(["date", "symbol"]).reset_index(drop=True)
    panel["composite"] = composite_score(panel, TOP_FACTORS)
    logger.info(f"Composite of top 5 factors: {TOP_FACTORS}")

    # Long-only strategies
    g_d_ew, n_d_ew, t_d_ew = portfolio_monthly(panel, "composite", n_buckets=10, weight="equal")
    g_d_cw, n_d_cw, t_d_cw = portfolio_monthly(panel, "composite", n_buckets=10, weight="cap")
    g_q_ew, n_q_ew, t_q_ew = portfolio_monthly(panel, "composite", n_buckets=5, weight="equal")
    g_q_cw, n_q_cw, t_q_cw = portfolio_monthly(panel, "composite", n_buckets=5, weight="cap")

    # Long-short strategies (dollar-neutral, D1-D10 or Q1-Q5)
    lg_d_ew, ln_d_ew, lt_d_ew = longshort_monthly(panel, "composite", n_buckets=10, weight="equal")
    lg_d_cw, ln_d_cw, lt_d_cw = longshort_monthly(panel, "composite", n_buckets=10, weight="cap")
    lg_q_ew, ln_q_ew, lt_q_ew = longshort_monthly(panel, "composite", n_buckets=5, weight="equal")
    lg_q_cw, ln_q_cw, lt_q_cw = longshort_monthly(panel, "composite", n_buckets=5, weight="cap")

    # Benchmarks
    u_cw = universe_return(panel, weight="cap")
    u_ew = universe_return(panel, weight="equal")
    spy = fetch_spy_monthly()

    rets = pd.DataFrame({
        "Top decile EW (net)":       n_d_ew,
        "Top decile CW (net)":       n_d_cw,
        "Top quintile EW (net)":     n_q_ew,
        "Top quintile CW (net)":     n_q_cw,
        "LS D1-D10 EW (net)":        ln_d_ew,
        "LS D1-D10 CW (net)":        ln_d_cw,
        "LS Q1-Q5 EW (net)":         ln_q_ew,
        "LS Q1-Q5 CW (net)":         ln_q_cw,
        "SPY (total return)":        spy,
        "Cap-wtd universe":          u_cw,
        "Equal-wtd universe":        u_ew,
    }).sort_index()
    # only keep months where strategy + SPY exist
    rets = rets.dropna(subset=["Top decile EW (net)", "SPY (total return)"])

    metrics = pd.DataFrame({k: perf_metrics(rets[k]) for k in rets}).T
    bench = rets["SPY (total return)"]
    metrics["alpha_vs_SPY_simple_%"] = (metrics["CAGR"] - perf_metrics(bench)["CAGR"]) * 100
    betas, alphas = [], []
    for k in rets:
        b, a = capm(rets[k], bench)
        betas.append(b); alphas.append(a * 100)
    metrics["CAPM_beta_vs_SPY"] = betas
    metrics["CAPM_alpha_vs_SPY_%"] = alphas
    metrics["avg_turnover_%"] = np.nan
    for k, ts in [("Top decile EW (net)", t_d_ew), ("Top decile CW (net)", t_d_cw),
                  ("Top quintile EW (net)", t_q_ew), ("Top quintile CW (net)", t_q_cw),
                  ("LS D1-D10 EW (net)", lt_d_ew), ("LS D1-D10 CW (net)", lt_d_cw),
                  ("LS Q1-Q5 EW (net)", lt_q_ew), ("LS Q1-Q5 CW (net)", lt_q_cw)]:
        metrics.loc[k, "avg_turnover_%"] = ts.mean() * 100

    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(PROJECT_ROOT / "reports" / "backtest_metrics.csv")
    rets.to_csv(PROJECT_ROOT / "reports" / "backtest_monthly_returns.csv")

    make_figures(rets, metrics)
    logger.info(f"\nBacktest summary:\n{metrics.round(3).to_string()}")


def make_figures(rets: pd.DataFrame, metrics: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import save_fig, set_plot_style

    set_plot_style()
    show = ["Top decile EW (net)", "Top decile CW (net)",
            "Top quintile EW (net)", "Top quintile CW (net)",
            "SPY (total return)", "Equal-wtd universe"]
    colors = {
        "Top decile EW (net)": "#1f3a5f", "Top decile CW (net)": "#456b9a",
        "Top quintile EW (net)": "#0a7a3a", "Top quintile CW (net)": "#3aa56b",
        "SPY (total return)": "#222", "Equal-wtd universe": "#888",
    }
    styles = {
        "Top decile EW (net)": "-", "Top decile CW (net)": "-",
        "Top quintile EW (net)": "-", "Top quintile CW (net)": "-",
        "SPY (total return)": "--", "Equal-wtd universe": ":",
    }

    # 1. Cumulative
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for col in show:
        if col not in rets:
            continue
        cum = (1 + rets[col].fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, lw=1.8, color=colors[col],
                ls=styles[col], label=col)
    ax.set_yscale("log")
    ax.set_title("Growth of $1 — top-5-factor portfolio vs S&P 500 (2010+, net of 10 bps/side)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(fontsize=8.5, loc="upper left")
    save_fig(fig, "backtest_cumulative")

    # 2. Drawdowns
    fig, ax = plt.subplots(figsize=(13, 5))
    for col in show:
        if col not in rets:
            continue
        cum = (1 + rets[col].fillna(0)).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, lw=1.3, color=colors[col],
                ls=styles[col], label=col)
    ax.axhline(0, color="#444", lw=0.7)
    ax.set_title("Drawdowns (net)", fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8.5)
    save_fig(fig, "backtest_drawdowns")

    # 3. Long-short cumulative (separate chart — much smaller magnitude)
    ls_show = ["LS D1-D10 EW (net)", "LS D1-D10 CW (net)",
               "LS Q1-Q5 EW (net)", "LS Q1-Q5 CW (net)"]
    ls_colors = {
        "LS D1-D10 EW (net)": "#1f3a5f", "LS D1-D10 CW (net)": "#456b9a",
        "LS Q1-Q5 EW (net)": "#0a7a3a", "LS Q1-Q5 CW (net)": "#3aa56b",
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 2])
    for col in ls_show:
        if col not in rets:
            continue
        cum = (1 + rets[col].fillna(0)).cumprod()
        ax1.plot(cum.index, cum.values, lw=1.7, color=ls_colors[col], label=col)
    ax1.axhline(1.0, color="#444", lw=0.7, ls="--", label="zero (dollar-neutral)")
    ax1.set_title("Long-short (dollar-neutral) cumulative growth — D1 long − D10 short / Q1 long − Q5 short",
                  fontsize=12, fontweight="bold", loc="left")
    ax1.set_ylabel("Growth of $1 (linear)")
    ax1.legend(fontsize=8.5, loc="upper left")
    for col in ls_show:
        if col not in rets:
            continue
        cum = (1 + rets[col].fillna(0)).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        ax2.plot(dd.index, dd.values, lw=1.2, color=ls_colors[col], label=col)
    ax2.axhline(0, color="#444", lw=0.7)
    ax2.set_title("Long-short drawdowns (net)", fontsize=11, fontweight="bold", loc="left")
    ax2.set_ylabel("Drawdown (%)")
    fig.tight_layout()
    save_fig(fig, "backtest_longshort")

    # 4. Metrics table
    render_metrics_table(metrics)


def render_metrics_table(metrics: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import save_fig

    keys = ["CAGR", "ann_vol", "Sharpe", "max_drawdown",
            "CAPM_beta_vs_SPY", "CAPM_alpha_vs_SPY_%",
            "alpha_vs_SPY_simple_%", "hit_rate", "avg_turnover_%"]
    headers = ["CAGR", "Ann. Vol", "Sharpe", "Max DD",
               "Beta vs SPY", "Jensen α vs SPY",
               "ΔCAGR vs SPY", "Hit Rate", "Avg Turnover"]
    disp = metrics[keys].copy()

    def fmt(k, v):
        if pd.isna(v):
            return "—"
        if k in ("CAGR", "ann_vol", "max_drawdown", "hit_rate"):
            return f"{v * 100:.2f}%"
        if k in ("CAPM_alpha_vs_SPY_%", "alpha_vs_SPY_simple_%", "avg_turnover_%"):
            return f"{v:.2f}%"
        return f"{v:.2f}"

    cell_text = [[fmt(k, r[k]) for k in keys] for _, r in disp.iterrows()]
    row_labels = disp.index.tolist()

    fig, ax = plt.subplots(figsize=(15, 0.45 * len(row_labels) + 1.6))
    ax.axis("off")
    ax.set_title(
        "Portfolio backtest — 5-factor composite vs SPY  (2010+, monthly rebalance, 10 bps/side costs)",
        fontsize=12, fontweight="bold", loc="left", pad=14,
    )
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=headers,
                   cellLoc="center", loc="center",
                   colWidths=[0.08, 0.07, 0.07, 0.07, 0.08, 0.11, 0.10, 0.07, 0.10])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for j in range(len(headers)):
        c = tbl[0, j]
        c.set_facecolor("#1f3a5f")
        c.set_text_props(color="white", fontweight="bold")
    # Highlight rows where CAPM alpha is positive AND statistically meaningful (alpha > 1%)
    for i, (_, r) in enumerate(disp.iterrows()):
        rl = tbl[i + 1, -1]
        rl.set_text_props(ha="right", fontweight="bold")
        rl.set_facecolor("#e9ecf2")
        alpha = r["CAPM_alpha_vs_SPY_%"]
        if pd.notna(alpha) and alpha > 1.0:
            for j in range(len(headers)):
                tbl[i + 1, j].set_facecolor("#cfe6cf")
        elif pd.notna(alpha) and alpha < -1.0:
            for j in range(len(headers)):
                tbl[i + 1, j].set_facecolor("#f4d6d2")
    fig.text(0.02, 0.02,
             "Composite = equal-weight cross-sectional z-score of ROIC, ROE, FCF yield, "
             "Revenue growth, EPS growth.  Long-only, monthly rebalance, 10 bps/side costs. "
             "Green rows = positive Jensen α > 1%; red = negative α < −1%.",
             fontsize=8.5, style="italic", color="#555")
    save_fig(fig, "backtest_metrics")


if __name__ == "__main__":
    main()
