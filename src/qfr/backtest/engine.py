"""Backtest engine: decile portfolios, transaction costs, performance metrics.

Monthly rebalance, equal-weight within bucket. Transaction costs are charged on
traded notional: cost_t = sum |w_{i,t} - w_{i,t-1}| * cost_per_side (a full
round-trip = 2 x per-side, since a name swap is a sell + a buy). Reused by Parts
5 (baseline) and 7 (portfolio analysis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _leg(x: pd.DataFrame, ret_col: str, cost: float) -> tuple[pd.Series, pd.Series]:
    """Gross monthly return and cost series for an equal-weight leg (col 'w')."""
    gross = x.groupby("date").apply(lambda r: float((r["w"] * r[ret_col]).sum()), include_groups=False)
    w = x.pivot_table(index="date", columns="symbol", values="w", fill_value=0.0).sort_index()
    turn = w.diff().abs().sum(axis=1)
    if len(turn):
        turn.iloc[0] = float(w.iloc[0].abs().sum())  # initial build
    return gross, turn * cost


def _equal_weight(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub["w"] = 1.0 / sub.groupby("date")["symbol"].transform("size")
    return sub


def backtest_score(df: pd.DataFrame, score_col: str, ret_col: str = "ret_fwd_1m",
                   n_buckets: int = 10, cost: float = 0.001) -> pd.DataFrame:
    """Net monthly returns for long-only top decile, dollar-neutral long-short,
    and an equal-weight universe benchmark, ranking by ``score_col``."""
    d = df.dropna(subset=[score_col, ret_col]).copy()
    d["bucket"] = d.groupby("date")[score_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_buckets, labels=False)
    )
    top = _equal_weight(d[d["bucket"] == n_buckets - 1])
    bot = _equal_weight(d[d["bucket"] == 0])
    allw = _equal_weight(d)
    tg, tc = _leg(top, ret_col, cost)
    bg, bc = _leg(bot, ret_col, cost)
    ag, ac = _leg(allw, ret_col, cost)
    return pd.DataFrame({
        "long_only_top": tg - tc,
        "long_short": (tg - bg) - (tc + bc),
        "benchmark_ew": ag - ac,
    })


def decile_avg_returns(df: pd.DataFrame, score_col: str, ret_col: str = "ret_fwd_1m",
                       n_buckets: int = 10) -> pd.Series:
    d = df.dropna(subset=[score_col, ret_col]).copy()
    d["bucket"] = d.groupby("date")[score_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_buckets, labels=False)
    ) + 1
    return d.groupby("bucket")[ret_col].mean() * 100.0


def perf_metrics(r: pd.Series, ann: int = 12) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    cum = float((1 + r).prod())
    n = len(r)
    dn = r[r < 0].std()
    return {
        "total_return": cum - 1,
        "CAGR": cum ** (ann / n) - 1,
        "ann_vol": r.std() * np.sqrt(ann),
        "Sharpe": (r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else np.nan,
        "Sortino": (r.mean() / dn * np.sqrt(ann)) if dn and dn > 0 else np.nan,
        "max_drawdown": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()),
        "hit_rate": float((r > 0).mean()),
        "n_months": n,
    }


def cumulative(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    c = (1 + r.fillna(0)).cumprod()
    return c / c.cummax() - 1.0
