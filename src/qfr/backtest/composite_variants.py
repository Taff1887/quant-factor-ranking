"""Factor-level weighting variants of the 5-factor composite.

The baseline composite in `portfolio.py` equal-weights the 5 factors
(20 % each). This module tests three alternative weighting schemes that let
the strongest signals do more of the work in the composite, while applying
**strict out-of-sample discipline**:

  - Weights at month *t* use only IC data through month *t-1*.
  - Weights are recomputed quarterly (not every month) to limit turnover
    and estimation churn.
  - Non-EW schemes are shrunk 50 % toward equal-weight to control
    estimation noise -- standard institutional practice.

Schemes
-------
1. **EW (baseline)**         w_k = 1/5 for all k
2. **IC-weighted**           w_k proportional to trailing 36m mean IC_k
                              (negative-IC factors get floored at zero)
3. **IC-IR weighted**         w_k proportional to trailing IR_k = mean / std
                              of monthly IC, then shrunk 50 % to EW
4. **t-stat squared**         w_k proportional to t_k^2 (Stambaugh-style),
                              then shrunk 50 % to EW

Caveat
------
The 5 factors are already selected *because* they ranked highly in the
factor screen. Doing trailing-IR weighting on top of that is a form of
double-dipping vs a fully OOS design (where the factor *universe* is
chosen on theory, then weights are estimated from data). The honest
framing: this section shows the **methodology**; the EW baseline remains
the more robust point estimate. For a production system you would select
the full factor universe on theory and IR-weight from there.

Run::  uv run python -m qfr.backtest.composite_variants
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.backtest.portfolio import (
    ANN,
    DEFAULT_COST_PER_SIDE,
    TOP_FACTORS,
    _ic_summary,
    _spearman_ic_monthly,
    capm,
    composite_ic_horizons,
    cs_z,
    fetch_spy_monthly,
    longshort_monthly,
    perf_metrics,
    portfolio_monthly,
)
from qfr.factors.build import build_factor_panel
from qfr.utils.config import PROJECT_ROOT
from qfr.utils.logging import logger

# Estimation window for IC stats (months)
TRAILING_WINDOW = 36
# Gap between most recent IC observation and the month it's used to weight
WEIGHT_LAG_MONTHS = 1
# How often weights are re-estimated
REBAL_EVERY_MONTHS = 3
# Shrinkage toward equal-weight (1.0 = full EW, 0.0 = no shrink). 0.5 is institutional default.
SHRINKAGE = 0.5
# Min number of months in the trailing window to trust a stat-based weight
MIN_WINDOW_MONTHS = 24

SCHEMES: tuple[str, ...] = ("ew", "ic", "ic_ir", "t2")
SCHEME_LABELS = {
    "ew": "Equal-weight (baseline)",
    "ic": "IC-weighted",
    "ic_ir": "IC-IR weighted (50 % shrunk)",
    "t2": "t-stat squared (50 % shrunk)",
}


# --------------------------------------------------------------------------
# Per-factor IC panel
# --------------------------------------------------------------------------
def per_factor_ic_panel(panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Monthly Spearman rank IC for each factor. Index = date, columns = factor."""
    ic_cols = {}
    for f in factors:
        ic_cols[f] = _spearman_ic_monthly(panel, f, "ret_fwd_1m")
    return pd.DataFrame(ic_cols).sort_index()


# --------------------------------------------------------------------------
# Trailing IC stats at a given anchor date (uses only prior data)
# --------------------------------------------------------------------------
def trailing_ic_stats(ic_panel: pd.DataFrame, t_anchor: pd.Timestamp,
                      window: int = TRAILING_WINDOW,
                      lag: int = WEIGHT_LAG_MONTHS) -> pd.DataFrame | None:
    """At month *t_anchor*, compute IC stats per factor using ICs from months
    [t_anchor - lag - window, t_anchor - lag).

    Returns a DataFrame indexed by factor with columns: mean, std, ir, t_stat, n.
    Returns None if not enough history.
    """
    end = t_anchor - pd.DateOffset(months=lag)
    start = end - pd.DateOffset(months=window)
    window_ic = ic_panel[(ic_panel.index >= start) & (ic_panel.index < end)]
    n = window_ic.count()
    if int(n.min()) < MIN_WINDOW_MONTHS:
        return None
    mean = window_ic.mean()
    std = window_ic.std().replace(0, np.nan)
    ir = mean / std
    t_stat = ir * np.sqrt(n)
    return pd.DataFrame({"mean": mean, "std": std, "ir": ir,
                         "t_stat": t_stat, "n": n})


# --------------------------------------------------------------------------
# Weight rules
# --------------------------------------------------------------------------
def compute_weights(stats: pd.DataFrame | None, factors: list[str], scheme: str,
                    shrinkage: float = SHRINKAGE) -> pd.Series:
    """Convert IC stats to factor weights. Floor at 0, normalise, shrink to EW."""
    ew = pd.Series(1.0 / len(factors), index=factors)
    if scheme == "ew" or stats is None:
        return ew
    if scheme == "ic":
        raw = stats["mean"].clip(lower=0)
    elif scheme == "ic_ir":
        raw = stats["ir"].clip(lower=0)
    elif scheme == "t2":
        raw = (stats["t_stat"].clip(lower=0)) ** 2
    else:
        raise ValueError(f"Unknown scheme {scheme!r}")
    s = float(raw.sum())
    if not np.isfinite(s) or s <= 0:
        return ew
    norm = (raw / s).reindex(factors).fillna(0.0)
    # Shrink toward EW (only ic_ir and t2 are shrunk; "ic" is presented raw
    # so the contrast with shrinkage is visible)
    if scheme in ("ic_ir", "t2"):
        return shrinkage * ew + (1 - shrinkage) * norm
    return norm


# --------------------------------------------------------------------------
# Build a time-varying composite given a weighting scheme
# --------------------------------------------------------------------------
def weighted_composite_series(panel: pd.DataFrame, factors: list[str],
                              ic_panel: pd.DataFrame, scheme: str,
                              shrinkage: float = SHRINKAGE,
                              rebal_every: int = REBAL_EVERY_MONTHS,
                              ) -> tuple[pd.Series, pd.DataFrame]:
    """Compute the time-varying composite score using out-of-sample factor weights.

    Returns (composite_series, weights_log_df). Weights are rebalanced every
    `rebal_every` months. Months without enough trailing history fall back to EW.
    """
    # Per-factor cross-sectional z-scores for the whole panel (1:1 with panel index)
    z_panel = pd.DataFrame({f: cs_z(panel, f) for f in factors})

    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    composite = pd.Series(np.nan, index=panel.index, dtype=float)
    weight_log: list[dict] = []

    rebal_dates = dates[::rebal_every]
    rebal_dates_list = list(rebal_dates) + [dates[-1] + pd.DateOffset(months=1)]

    for i, t_reb in enumerate(rebal_dates_list[:-1]):
        t_end = rebal_dates_list[i + 1]
        stats = trailing_ic_stats(ic_panel, t_reb)
        w = compute_weights(stats, factors, scheme=scheme, shrinkage=shrinkage)
        weight_log.append({"date": t_reb, "scheme": scheme,
                           **{f: float(w[f]) for f in factors},
                           "_has_history": stats is not None})

        # Apply weights over [t_reb, t_end)
        mask = (panel["date"] >= t_reb) & (panel["date"] < t_end)
        idx = panel.index[mask]
        if len(idx) == 0:
            continue
        z_window = z_panel.loc[idx]
        w_arr = w.reindex(factors).to_numpy()
        z_arr = z_window.to_numpy()
        valid = ~np.isnan(z_arr)
        # Renormalise weights per-row over factors that have a non-NaN z-score
        w_eff = np.where(valid, w_arr[None, :], 0.0)
        w_sum = w_eff.sum(axis=1)
        z_eff = np.where(valid, z_arr, 0.0)
        weighted = (z_eff * w_eff).sum(axis=1)
        out = np.where(w_sum > 0, weighted / w_sum, np.nan)
        composite.loc[idx] = out

    return composite, pd.DataFrame(weight_log).set_index("date")


# --------------------------------------------------------------------------
# IC summary at a few horizons for any composite series
# --------------------------------------------------------------------------
def composite_ic_summary(panel: pd.DataFrame, comp_col: str = "comp") -> dict:
    h_df, _ = composite_ic_horizons(panel, comp_col)
    h_df = h_df.set_index("horizon_months")
    out: dict = {}
    for h in (1, 3, 12):
        out[f"IC_{h}m_%"] = float(h_df.loc[h, "mean_ic_%"])
        out[f"t_{h}m"] = float(h_df.loc[h, "t_stat"])
        out[f"hit_{h}m_%"] = float(h_df.loc[h, "hit_rate_%"])
    return out


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def chart_weights_over_time(weights_by_scheme: dict[str, pd.DataFrame],
                            factors: list[str]) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import save_fig, set_plot_style
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    colors = {"returnOnInvestedCapital": "#1f3a5f", "returnOnEquity": "#456b9a",
              "freeCashFlowYield": "#0a7a3a", "revenueGrowth": "#bc4a3c",
              "epsgrowth": "#d9a02e"}
    pretty = {"returnOnInvestedCapital": "ROIC", "returnOnEquity": "ROE",
              "freeCashFlowYield": "FCF yield", "revenueGrowth": "Rev growth",
              "epsgrowth": "EPS growth"}
    for ax, scheme in zip(axes.ravel(), SCHEMES):
        w_df = weights_by_scheme[scheme][factors]
        ax.stackplot(w_df.index, [w_df[f].values * 100 for f in factors],
                     labels=[pretty[f] for f in factors],
                     colors=[colors[f] for f in factors], alpha=0.9)
        ax.axhline(20, color="#222", lw=0.7, ls=":")
        ax.set_title(SCHEME_LABELS[scheme], fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("Factor weight (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="lower right", fontsize=8, ncol=5)
    fig.suptitle("Composite weighting variants — factor weights over time (quarterly rebalance)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "composite_variants_weights")


def chart_ic_compare(ic_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import save_fig, set_plot_style
    set_plot_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    horizons = [1, 3, 12]
    schemes = list(ic_df.index)
    x = np.arange(len(schemes))
    width = 0.25
    palette = ["#1f3a5f", "#0a7a3a", "#bc4a3c"]
    for i, h in enumerate(horizons):
        ic_vals = [ic_df.loc[s, f"IC_{h}m_%"] for s in schemes]
        t_vals = [ic_df.loc[s, f"t_{h}m"] for s in schemes]
        a1.bar(x + (i - 1) * width, ic_vals, width, label=f"{h}m", color=palette[i], alpha=0.85)
        a2.bar(x + (i - 1) * width, t_vals, width, label=f"{h}m", color=palette[i], alpha=0.85)
    for ax, ylab, title in zip([a1, a2],
                               ["Mean rank IC (%)", "t-stat of IC"],
                               ["Composite rank IC by weighting scheme",
                                "IC t-stat by weighting scheme"]):
        ax.set_xticks(x)
        ax.set_xticklabels([SCHEME_LABELS[s].split(" (")[0] for s in schemes],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.axhline(0, color="#444", lw=0.7)
        ax.legend(title="Horizon", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    a2.axhline(1.96, color="#bc4a3c", lw=0.7, ls="--", alpha=0.7)
    a2.axhline(-1.96, color="#bc4a3c", lw=0.7, ls="--", alpha=0.7)
    fig.tight_layout()
    save_fig(fig, "composite_variants_ic")


def chart_cumulative_variants(perf_series: dict[str, pd.Series],
                              spy: pd.Series) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import save_fig, set_plot_style
    set_plot_style()
    fig, ax = plt.subplots(figsize=(13, 6))
    palette = ["#1f3a5f", "#0a7a3a", "#bc4a3c", "#d9a02e"]
    for (s, r), c in zip(perf_series.items(), palette):
        cum = (1 + r.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, lw=1.8, color=c, label=SCHEME_LABELS[s])
    spy_aligned = spy.reindex(next(iter(perf_series.values())).index)
    ax.plot(spy_aligned.index, (1 + spy_aligned.fillna(0)).cumprod().values,
            lw=1.6, color="#222", ls="--", label="SPY (total return)")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 — top-quintile cap-weighted, by composite weighting scheme  (net of 10 bps/side)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "composite_variants_cumulative")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main() -> None:
    panel = build_factor_panel().sort_values(["date", "symbol"]).reset_index(drop=True)
    factors = TOP_FACTORS
    logger.info(f"Building per-factor IC panel for {len(factors)} factors...")
    ic_panel = per_factor_ic_panel(panel, factors)
    logger.info(f"IC panel: {ic_panel.shape[0]} months, "
                f"date range {ic_panel.index.min().date()} - {ic_panel.index.max().date()}")

    # Build 4 composite variants
    composites: dict[str, pd.Series] = {}
    weights_logs: dict[str, pd.DataFrame] = {}
    for scheme in SCHEMES:
        logger.info(f"  building composite under scheme: {scheme}")
        comp, wlog = weighted_composite_series(panel, factors, ic_panel,
                                               scheme=scheme)
        composites[scheme] = comp
        weights_logs[scheme] = wlog

    # IC summary per scheme (drop the warmup period where weights = EW fallback)
    # We compare on the post-warmup window only: from the first quarterly date
    # at which all schemes have a real weight estimate.
    valid_starts = []
    for s, wl in weights_logs.items():
        real = wl.loc[wl["_has_history"]]
        if len(real):
            valid_starts.append(real.index[0])
    if valid_starts:
        eval_start = max(valid_starts)
    else:
        eval_start = panel["date"].min()
    logger.info(f"OOS evaluation window starts: {eval_start.date()}")

    # IC table per scheme on the OOS window
    ic_rows = []
    for s, comp in composites.items():
        p_oos = panel.copy()
        p_oos["comp"] = comp
        p_oos = p_oos[p_oos["date"] >= eval_start]
        st = composite_ic_summary(p_oos, "comp")
        st["scheme"] = s
        st["label"] = SCHEME_LABELS[s]
        ic_rows.append(st)
    ic_df = pd.DataFrame(ic_rows).set_index("scheme")

    # Portfolio backtests per scheme (top-quintile CW long-only + LS Q1-Q5 EW),
    # again on the OOS window only.
    spy = fetch_spy_monthly()
    bt_rows = []
    perf_lo: dict[str, pd.Series] = {}
    for s, comp in composites.items():
        p_oos = panel.copy()
        p_oos["comp"] = comp
        p_oos = p_oos[p_oos["date"] >= eval_start].copy()
        lo = portfolio_monthly(p_oos, "comp", n_buckets=5, weight="cap",
                               name=f"TopQ CW [{s}]")
        ls = longshort_monthly(p_oos, "comp", n_buckets=5, weight="equal",
                               name=f"LS Q1-Q5 EW [{s}]")
        spy_eval = spy[spy.index >= eval_start]
        for strat in (lo, ls):
            net = (strat["gross"] - strat["total_traded"] * DEFAULT_COST_PER_SIDE).dropna()
            m = perf_metrics(net)
            beta, alpha = capm(net, spy_eval)
            bt_rows.append({
                "scheme": s,
                "label": SCHEME_LABELS[s],
                "kind": strat["kind"],
                "strategy": strat["name"],
                "CAGR_%": m["CAGR"] * 100,
                "Sharpe": m["Sharpe"],
                "ann_vol_%": m["ann_vol"] * 100,
                "max_drawdown_%": m["max_drawdown"] * 100,
                "beta": beta,
                "Jensen_alpha_%": alpha * 100,
                "ann_turnover_pa_%": strat["total_traded"].mean() * 0.5 * ANN * 100,
            })
            if strat["kind"] == "long_only":
                perf_lo[s] = net
    bt_df = pd.DataFrame(bt_rows)

    # Average factor weights (for the table)
    avg_w_rows = []
    for s, wl in weights_logs.items():
        real = wl.loc[wl["_has_history"], factors]
        if len(real):
            mean_w = real.mean()
        else:
            mean_w = pd.Series(1.0 / len(factors), index=factors)
        avg_w_rows.append({"scheme": s, "label": SCHEME_LABELS[s],
                           **{f: float(mean_w[f]) for f in factors}})
    avg_w_df = pd.DataFrame(avg_w_rows).set_index("scheme")

    # Save artefacts
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ic_df.reset_index().to_csv(out_dir / "composite_variants_ic.csv", index=False)
    bt_df.to_csv(out_dir / "composite_variants_summary.csv", index=False)
    avg_w_df.reset_index().to_csv(out_dir / "composite_variants_avg_weights.csv", index=False)
    for s, wl in weights_logs.items():
        wl.to_csv(out_dir / f"composite_variants_weights_{s}.csv")

    # Charts
    chart_weights_over_time(weights_logs, factors)
    chart_ic_compare(ic_df)
    chart_cumulative_variants(perf_lo, spy)

    # Console summary
    logger.info(f"\n=== Average factor weights over OOS window ===\n{(avg_w_df[factors] * 100).round(1).to_string()}")
    logger.info(f"\n=== Composite IC by weighting scheme (OOS, from {eval_start.date()}) ===\n"
                f"{ic_df.drop(columns=['label']).round(3).to_string()}")
    logger.info(f"\n=== Portfolio backtests by scheme (OOS, net of 10 bps/side) ===\n"
                f"{bt_df.drop(columns=['label']).round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
