"""Part 5 - Traditional multi-factor baseline (the benchmark the ML must beat).

Two non-ML benchmarks on the 2010+ factor panel:
  * Equal-weight composite - average of the four classic style factors
    (Value, Quality, Momentum, Growth), re-ranked cross-sectionally. The textbook
    multi-factor signal; deliberately does NOT peek at the Part-4 ICs.
  * Walk-forward Ridge - a cross-sectional linear regression of next-month return
    on all six factor ranks, fit on an expanding past-only window (no look-ahead).
    Also previews the walk-forward machinery used by the ML stage.

Each score is run through the decile backtest (long-only top decile + dollar-
neutral long-short) net of 10 bps/side costs, versus an equal-weight S&P 500.

Run::  uv run python -m qfr.models.baseline
Outputs: reports/baseline_performance.csv, data/processed/baseline_returns.parquet, charts/05_*.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from qfr.backtest.engine import (
    backtest_score,
    cumulative,
    decile_avg_returns,
    drawdown,
    perf_metrics,
)
from qfr.factors.transforms import cs_rank
from qfr.utils.config import PROJECT_ROOT, settings
from qfr.utils.io import read_parquet, write_parquet
from qfr.utils.logging import logger

ALPHA_FACTORS = ["value", "quality", "momentum", "growth"]  # classic composite
ALL_FACTORS = ["value", "quality", "momentum", "growth", "risk", "size"]


def composite_score(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["composite_raw"] = f[ALPHA_FACTORS].mean(axis=1)
    f["composite"] = cs_rank(f, ["composite_raw"])["composite_raw"]
    return f


def ridge_walkforward(f: pd.DataFrame, feats=ALL_FACTORS, target: str = "ret_fwd_1m",
                      min_train_months: int = 36, alpha: float = 10.0) -> pd.DataFrame:
    """Expanding-window cross-sectional Ridge; predicts each month from the past only."""
    dates = sorted(f["date"].unique())
    hist = f.dropna(subset=[*feats, target])
    preds = []
    for i, dt in enumerate(dates):
        if i < min_train_months:
            continue
        train = hist[hist["date"] < dt]
        test = f[f["date"] == dt].dropna(subset=feats)
        if len(train) < 500 or test.empty:
            continue
        mdl = Ridge(alpha=alpha).fit(train[feats], train[target])
        p = test[["date", "symbol"]].copy()
        p["ridge"] = mdl.predict(test[feats])
        preds.append(p)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["date", "symbol", "ridge"])


def make_figures(f: pd.DataFrame, series: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    from qfr.utils.viz import PALETTE, save_fig, set_plot_style

    set_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for col in series.columns:
        c = cumulative(series[col])
        ax.plot(c.index, c.values, lw=1.7, label=col)
    ax.set_yscale("log")
    ax.set_title("Growth of $1 - traditional baselines vs benchmark (2010+, net of 10 bps/side)")
    ax.set_ylabel("growth of $1 (log scale)")
    ax.legend(fontsize=8)
    save_fig(fig, "05_cumulative_returns")

    dec = decile_avg_returns(f, "composite")
    fig, ax = plt.subplots()
    ax.bar([str(i) for i in dec.index], dec.values, color=PALETTE["primary"])
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_title("Composite score: mean forward 1-month return by decile (10 = best)")
    ax.set_ylabel("mean fwd 1m return (%)")
    ax.set_xlabel("decile")
    save_fig(fig, "05_composite_deciles")

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in ["Composite long-only (top decile)", "Composite long-short", "EW S&P 500 benchmark"]:
        dd = drawdown(series[col]) * 100
        ax.plot(dd.index, dd.values, lw=1.3, label=col)
    ax.set_title("Drawdowns (net)")
    ax.set_ylabel("drawdown (%)")
    ax.legend(fontsize=8)
    save_fig(fig, "05_drawdowns")


def build() -> pd.DataFrame:
    f = read_parquet(settings.processed_dir / "factors.parquet")
    f = composite_score(f)
    f = f.merge(ridge_walkforward(f), on=["date", "symbol"], how="left")

    comp = backtest_score(f, "composite")
    ridge_bt = backtest_score(f.dropna(subset=["ridge"]), "ridge")

    series = pd.DataFrame({
        "Composite long-only (top decile)": comp["long_only_top"],
        "Composite long-short": comp["long_short"],
        "Ridge long-only (top decile)": ridge_bt["long_only_top"],
        "Ridge long-short": ridge_bt["long_short"],
        "EW S&P 500 benchmark": comp["benchmark_ew"],
    }).sort_index()

    metrics = pd.DataFrame({k: perf_metrics(series[k]) for k in series}).T
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(PROJECT_ROOT / "reports" / "baseline_performance.csv")
    write_parquet(series, settings.processed_dir / "baseline_returns.parquet")
    make_figures(f, series)

    logger.info("Baseline performance (2010+, net of 10 bps/side):\n" + metrics.round(3).to_string())
    return metrics


def main() -> None:
    build()


if __name__ == "__main__":
    main()
