"""Cross-sectional transforms (grouped by rebalance date).

These are the building blocks for both EDA previews and factor construction:
winsorisation to tame outliers, z-scoring to standardise scale, and rank
normalisation (the most robust to fat tails, and the basis for ranking models).
All operate *within each date* so we never mix information across time.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def cs_winsorize(
    df: pd.DataFrame, cols: Sequence[str], by: str = "date",
    lower: float = 0.01, upper: float = 0.99,
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(by)
    for c in cols:
        out[c] = g[c].transform(lambda s: winsorize(s, lower, upper))
    return out


def cs_zscore(df: pd.DataFrame, cols: Sequence[str], by: str = "date") -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(by)
    for c in cols:
        out[c] = g[c].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    return out


def cs_rank(
    df: pd.DataFrame, cols: Sequence[str], by: str = "date", pct: bool = True
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(by)
    for c in cols:
        out[c] = g[c].rank(pct=pct)
    return out
