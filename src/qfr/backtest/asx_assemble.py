"""Assemble the independent ASX PIT factor panel from the raw FMP + Yahoo pulls.

Builds a monthly panel with:
  - dividend-adjusted prices + forward returns (true forward, t -> t+1m)
  - point-in-time fundamentals (most recent filing with acceptedDate <= t)
  - free-float-adjusted market cap for cap-weighted portfolios
  - 5 factor inputs: ROIC, ROE, FCF yield, revenue growth, EPS growth

Output: data/processed/asx_panel.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.backtest.asx_pull_data import (
    FREEFLOAT_PARQUET,
    FUNDAMENTALS_PARQUET,
    MCAP_HISTORY_PARQUET,
    PRICES_PARQUET,
    UNIVERSE_PARQUET,
)
from qfr.utils.config import PROJECT_ROOT
from qfr.utils.logging import logger

PANEL_PARQUET = PROJECT_ROOT / "data" / "processed" / "asx_panel.parquet"
PANEL_PARQUET.parent.mkdir(parents=True, exist_ok=True)

# Universe for PIT proxy (top N by FREE-FLOAT-adjusted market cap each month)
UNIVERSE_TOP_N = 200

FACTORS = ["returnOnInvestedCapital", "returnOnEquity", "freeCashFlowYield",
           "revenueGrowth", "epsgrowth"]


# --------------------------------------------------------------------------
# Monthly price panel + forward returns
# --------------------------------------------------------------------------
def build_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end; compute next-month forward return per symbol.

    Forward returns are winsorised at +/- 100% / +/- 200% to handle unadjusted
    corporate actions, halt artefacts, and other data errors. This is standard
    institutional practice; anything beyond +/- 100% in a single month is
    overwhelmingly a data quirk rather than a real economic move.
    """
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["symbol", "date"])
    p["month_end"] = p["date"] + pd.offsets.MonthEnd(0)
    monthly = (p.groupby(["symbol", "month_end"])
                 .agg(adjClose=("adjClose", "last"))
                 .reset_index()
                 .rename(columns={"month_end": "date"}))
    monthly = monthly.sort_values(["symbol", "date"])
    monthly["ret_fwd_1m"] = monthly.groupby("symbol")["adjClose"].shift(-1) / monthly["adjClose"] - 1
    monthly["ret_fwd_3m"] = monthly.groupby("symbol")["adjClose"].shift(-3) / monthly["adjClose"] - 1
    # Winsorise extreme moves (data-error guard)
    monthly["ret_fwd_1m"] = monthly["ret_fwd_1m"].clip(lower=-1.0, upper=1.0)
    monthly["ret_fwd_3m"] = monthly["ret_fwd_3m"].clip(lower=-1.0, upper=2.0)
    return monthly


# --------------------------------------------------------------------------
# PIT fundamentals: most recent filing with acceptedDate <= month-end
# --------------------------------------------------------------------------
def _consolidate_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Fix the row-explosion from the data pull's outer merge.

    Each filing is split across 2-3 rows (one per source endpoint) because the
    pull merged on acceptedDate which is only present in cash-flow-statement.
    Collapse back to one row per (symbol, fiscal_date, period) by taking the
    first non-null value of each field.
    """
    f = fundamentals.copy()
    # Drop duplicated `calendarYear_*` columns (the pull added them as suffixes)
    drop = [c for c in f.columns if c.startswith("calendarYear")]
    f = f.drop(columns=drop, errors="ignore")
    # group key
    f["fiscal_date"] = pd.to_datetime(f["date"])
    grp_keys = ["symbol", "fiscal_date", "period"]
    value_cols = [c for c in f.columns if c not in grp_keys + ["date"]]
    # take first non-null per group per column
    consolidated = (f.sort_values(grp_keys)
                     .groupby(grp_keys, dropna=False)[value_cols]
                     .first()
                     .reset_index())
    return consolidated


def pit_join_fundamentals(monthly_prices: pd.DataFrame,
                          fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Merge_asof: at each (symbol, month-end), pick the most recent filing
    with acceptedDate <= month-end."""
    f = _consolidate_fundamentals(fundamentals)
    f["acceptedDate"] = pd.to_datetime(f["acceptedDate"], errors="coerce")
    # If acceptedDate missing, fall back to fiscal date + a safe 90d lag
    # (typical ASX filing delay is 60-90 days from period-end)
    f["pit_date"] = f["acceptedDate"]
    missing = f["pit_date"].isna()
    f.loc[missing, "pit_date"] = f.loc[missing, "fiscal_date"] + pd.Timedelta(days=90)

    # Compute FCF yield from FCF / marketCap if it's missing or zero
    if "freeCashFlowYield" not in f.columns or (f["freeCashFlowYield"].fillna(0) == 0).mean() > 0.5:
        f["freeCashFlowYield"] = f["freeCashFlow"] / f["marketCap"].replace(0, np.nan)

    keep_cols = ["symbol", "pit_date"] + FACTORS
    f = f[[c for c in keep_cols if c in f.columns]].dropna(subset=["pit_date"])
    f = f.sort_values(["symbol", "pit_date"])

    monthly_prices = monthly_prices.sort_values(["symbol", "date"])

    # asof merge per ticker
    out = pd.merge_asof(
        monthly_prices.sort_values("date"),
        f.sort_values("pit_date"),
        left_on="date", right_on="pit_date",
        by="symbol", direction="backward",
    )
    # Drop fundamentals older than 18 months (stale)
    out["stale_days"] = (out["date"] - out["pit_date"]).dt.days
    too_stale = out["stale_days"] > 18 * 30
    for c in FACTORS:
        if c in out.columns:
            out.loc[too_stale, c] = np.nan
    return out


# --------------------------------------------------------------------------
# Free-float-adjusted market cap (uses true historical mcap from FMP)
# --------------------------------------------------------------------------
def add_ff_adjusted_mcap(panel: pd.DataFrame, freefloat: pd.DataFrame,
                         mcap_history: pd.DataFrame) -> pd.DataFrame:
    """Apply free-float ratio to the historical market cap at each (symbol, date).

    Historical mcap comes from FMP's `historical-market-capitalization` endpoint
    (true shares outstanding × close price at that date). Free-float ratio is
    treated as constant per symbol (current Yahoo `floatShares / sharesOutstanding`)
    since we don't have a historical free-float series. This is the standard
    practical approximation when historical float data is unavailable.
    """
    mh = mcap_history.copy()
    mh["date"] = pd.to_datetime(mh["date"])
    # Resample daily mcap to month-end per ticker (last value in month)
    mh["month_end"] = mh["date"] + pd.offsets.MonthEnd(0)
    mh_monthly = (mh.sort_values(["symbol", "date"])
                    .groupby(["symbol", "month_end"])
                    .agg(true_mcap=("marketCap", "last"))
                    .reset_index()
                    .rename(columns={"month_end": "date"}))

    ff = freefloat[["symbol", "free_float_ratio"]].copy()
    ff["free_float_ratio"] = ff["free_float_ratio"].fillna(1.0).clip(lower=0.1, upper=1.0)

    p = panel.merge(mh_monthly, on=["symbol", "date"], how="left")
    p = p.merge(ff, on="symbol", how="left")
    p["marketCap"] = p["true_mcap"] * p["free_float_ratio"].fillna(1.0)
    return p


# --------------------------------------------------------------------------
# Apply PIT top-N filter + final cleanup
# --------------------------------------------------------------------------
def apply_universe_filter(panel: pd.DataFrame, top_n: int = UNIVERSE_TOP_N
                          ) -> pd.DataFrame:
    p = panel.dropna(subset=["marketCap", "adjClose"]).copy()
    p["mc_rank"] = p.groupby("date")["marketCap"].rank(ascending=False, method="first")
    p = p[p["mc_rank"] <= top_n].drop(columns=["mc_rank"])
    return p


def add_sector(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(universe[["symbol", "sector"]], on="symbol", how="left")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def build_panel(force: bool = False) -> pd.DataFrame:
    if PANEL_PARQUET.exists() and not force:
        logger.info(f"Loading cached panel from {PANEL_PARQUET}")
        return pd.read_parquet(PANEL_PARQUET)

    logger.info("Building fresh ASX PIT panel...")
    universe = pd.read_parquet(UNIVERSE_PARQUET)
    prices = pd.read_parquet(PRICES_PARQUET)
    fundamentals = pd.read_parquet(FUNDAMENTALS_PARQUET)
    freefloat = pd.read_parquet(FREEFLOAT_PARQUET)
    mcap_history = pd.read_parquet(MCAP_HISTORY_PARQUET)

    logger.info(f"  prices: {len(prices):,} rows, {prices['symbol'].nunique()} tickers")
    logger.info(f"  fundamentals: {len(fundamentals):,} rows, {fundamentals['symbol'].nunique()} tickers")
    logger.info(f"  freefloat: {freefloat['free_float_ratio'].notna().sum()}/{len(freefloat)} valid")
    logger.info(f"  mcap history: {len(mcap_history):,} rows, {mcap_history['symbol'].nunique()} tickers")

    monthly = build_monthly_prices(prices)
    logger.info(f"  monthly prices: {len(monthly):,} rows")

    joined = pit_join_fundamentals(monthly, fundamentals)
    joined = add_ff_adjusted_mcap(joined, freefloat, mcap_history)
    joined = add_sector(joined, universe)
    panel = apply_universe_filter(joined, top_n=UNIVERSE_TOP_N)

    # Keep only dates with enough names with all 5 factors + forward return
    cov = panel.groupby("date").apply(
        lambda g: g[["ret_fwd_1m"] + FACTORS].notna().all(axis=1).sum(),
        include_groups=False)
    valid_dates = cov[cov >= 50].index
    panel = panel[panel["date"].isin(valid_dates)].copy().sort_values(["date", "symbol"])

    panel.to_parquet(PANEL_PARQUET, index=False)
    logger.info(f"Panel saved: {len(panel):,} rows, {panel['symbol'].nunique()} tickers, "
                f"{panel['date'].nunique()} months "
                f"({panel['date'].min().date()} -> {panel['date'].max().date()})")

    # Diagnostics
    avg_ff = panel.drop_duplicates("symbol")["free_float_ratio"].mean()
    logger.info(f"  avg free-float ratio across universe: {avg_ff:.3f}")
    logger.info(f"  factor non-null coverage: "
                f"{ {c: f'{panel[c].notna().mean()*100:.0f}%' for c in FACTORS} }")

    return panel


if __name__ == "__main__":
    build_panel(force=True)
