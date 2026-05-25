"""Assemble the point-in-time analytics panel.

This is the methodological core of the data layer. It produces one tidy frame
keyed by ``[date, symbol]`` over the monthly rebalance grid, where every value is
knowable *as of* that date:

* **Prices / labels** — month-end split-&-dividend-adjusted close (as-of, using
  the last trade on/just-before the month-end), plus forward 1/3/6-month returns.
* **Fundamentals** — for each rebalance date we take the most recent quarterly
  filing whose ``acceptedDate`` (SEC acceptance timestamp) is on or before that
  date. This is the look-ahead control: a fundamental only becomes a usable
  signal once the filing was actually public.
* **Membership mask** — only ``[date, symbol]`` pairs that were genuinely in the
  S&P 500 on that date (point-in-time universe).
* **Recycling / staleness filter** — a row survives only if it has a PIT price
  *and* a fundamental filing that is not impossibly stale (guards against
  delisted companies whose ticker was later reused, e.g. ``CA``).

Curated field lists keep the panel transparent and auditable rather than
carrying ~300 raw vendor columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.utils.config import settings
from qfr.utils.dates import month_end_dates
from qfr.utils.io import read_parquet, write_parquet
from qfr.utils.logging import logger

REBALANCE_START = "2000-01-31"
REBALANCE_END = "2026-04-30"
PRICE_TOLERANCE_DAYS = 7  # month-end calendar date -> last trade within a week
MAX_FILING_STALENESS_DAYS = 400  # drop fundamentals older than ~4 quarters

# --- Curated fundamental fields (disjoint across datasets, keyed by symbol/FY/period)
FIELDS: dict[str, list[str]] = {
    "fund_income": [
        "revenue", "grossProfit", "operatingIncome", "ebitda", "netIncome",
        "eps", "epsDiluted", "interestExpense", "weightedAverageShsOut",
        "researchAndDevelopmentExpenses",
    ],
    "fund_balance": [
        "totalAssets", "totalStockholdersEquity", "totalDebt", "netDebt",
        "cashAndCashEquivalents", "totalCurrentAssets", "totalCurrentLiabilities",
        "inventory",
    ],
    "fund_cashflow": ["operatingCashFlow", "freeCashFlow", "capitalExpenditure"],
    "fund_ratios": [
        "priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
        "debtToEquityRatio", "interestCoverageRatio", "grossProfitMargin",
        "operatingProfitMargin", "netProfitMargin", "assetTurnover",
        "currentRatio", "dividendYield", "bookValuePerShare",
    ],
    "fund_key_metrics": [
        "marketCap", "enterpriseValue", "evToEBITDA", "returnOnEquity",
        "returnOnInvestedCapital", "returnOnAssets", "earningsYield",
        "freeCashFlowYield",
    ],
    "fund_growth": ["revenueGrowth", "epsgrowth", "netIncomeGrowth", "ebitdaGrowth"],
}
KEYS = ["symbol", "fiscalYear", "period"]


def _load(name: str) -> pd.DataFrame:
    return read_parquet(settings.processed_dir / f"{name}.parquet")


# --------------------------------------------------------------------------
# Fundamentals
# --------------------------------------------------------------------------
def build_combined_fundamentals() -> pd.DataFrame:
    """Merge curated quarterly fundamentals + a conservative availability date.

    Availability = max ``acceptedDate`` across income/balance/cash-flow for the
    period (a filing is fully public once its last component is accepted).
    """
    # Availability date per (symbol, fiscalYear, period).
    avail_parts = []
    for name in ("fund_income", "fund_balance", "fund_cashflow"):
        df = _load(name)
        if "acceptedDate" in df.columns:
            avail_parts.append(df[KEYS + ["acceptedDate"]])
    avail = pd.concat(avail_parts, ignore_index=True).dropna(subset=["acceptedDate"])
    avail["acceptedDate"] = pd.to_datetime(avail["acceptedDate"])
    avail = avail.groupby(KEYS, as_index=False)["acceptedDate"].max()

    # Period-end date (from income) for reference / staleness.
    inc = _load("fund_income")
    period_end = inc[KEYS + ["date"]].rename(columns={"date": "period_end"})
    period_end["period_end"] = pd.to_datetime(period_end["period_end"])

    combined = avail.merge(period_end, on=KEYS, how="left")
    for name, fields in FIELDS.items():
        df = _load(name)
        present = [c for c in fields if c in df.columns]
        sub = df[KEYS + present].drop_duplicates(subset=KEYS)
        combined = combined.merge(sub, on=KEYS, how="left")

    logger.info(
        f"combined fundamentals: {len(combined):,} period-rows, "
        f"{combined['symbol'].nunique()} symbols, {combined.shape[1]} cols"
    )
    return combined


def pit_fundamentals(grid: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    """As-of join: latest filing per [date, symbol] with acceptedDate <= date."""
    left = grid[["date", "symbol"]].sort_values("date").reset_index(drop=True)
    right = combined.sort_values("acceptedDate").reset_index(drop=True)
    out = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="acceptedDate",
        by="symbol",
        direction="backward",
    )
    out["filing_lag_days"] = (out["date"] - out["acceptedDate"]).dt.days
    return out


# --------------------------------------------------------------------------
# Prices and forward returns
# --------------------------------------------------------------------------
def month_end_price_panel(
    prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, *, tol_days: int = PRICE_TOLERANCE_DAYS
) -> pd.DataFrame:
    """As-of month-end adjusted close per symbol (last trade within ``tol_days``)."""
    px = prices.loc[:, ["symbol", "date", "adjClose"]].dropna(subset=["adjClose"])
    px = px[px["adjClose"] > 0]  # guard against zero/garbage prices -> inf returns
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date")
    reb = pd.DataFrame({"date": pd.DatetimeIndex(rebalance_dates)})
    tol = pd.Timedelta(days=tol_days)
    frames = []
    for sym, g in px.groupby("symbol", sort=False):
        m = pd.merge_asof(reb, g[["date", "adjClose"]], on="date", direction="backward", tolerance=tol)
        m["symbol"] = sym
        frames.append(m)
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def add_forward_returns(panel: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 6)) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "date"]).copy()
    g = panel.groupby("symbol")["adjClose"]
    for h in horizons:
        panel[f"ret_fwd_{h}m"] = g.shift(-h) / panel["adjClose"] - 1.0
    ret_cols = [f"ret_fwd_{h}m" for h in horizons]
    panel[ret_cols] = panel[ret_cols].replace([np.inf, -np.inf], np.nan)
    return panel


# --------------------------------------------------------------------------
# Sector map
# --------------------------------------------------------------------------
def sector_map() -> pd.Series:
    """symbol -> sector, preferring profiles (broad) then current members."""
    mapping: dict[str, str] = {}
    members = _load("members_now")
    if {"symbol", "sector"} <= set(members.columns):
        mapping.update(members.dropna(subset=["sector"]).set_index("symbol")["sector"].to_dict())
    profiles = _load("profiles")
    if {"symbol", "sector"} <= set(profiles.columns):
        mapping.update(profiles.dropna(subset=["sector"]).set_index("symbol")["sector"].to_dict())
    return pd.Series(mapping, name="sector")


# --------------------------------------------------------------------------
# Master assembly
# --------------------------------------------------------------------------
def assemble_master() -> pd.DataFrame:
    rebalance = month_end_dates(REBALANCE_START, REBALANCE_END)

    universe = _load("universe_panel")
    universe["date"] = pd.to_datetime(universe["date"])
    grid = universe[universe["date"].isin(rebalance)][["date", "symbol"]].copy()
    logger.info(f"membership grid (>= {REBALANCE_START}): {len(grid):,} rows")

    prices = _load("prices_long")
    px_panel = add_forward_returns(month_end_price_panel(prices, rebalance))

    combined = build_combined_fundamentals()
    pit = pit_fundamentals(grid, combined)

    master = grid.merge(px_panel, on=["date", "symbol"], how="left")
    master = master.merge(pit, on=["date", "symbol"], how="left")
    master["sector"] = master["symbol"].map(sector_map()).fillna("Unknown")

    # Coverage flags + recycling/staleness filter.
    master["has_price"] = master["adjClose"].notna()
    master["has_fundamentals"] = master["acceptedDate"].notna()
    master["fresh_filing"] = master["filing_lag_days"].le(MAX_FILING_STALENESS_DAYS)
    master["investable"] = master["has_price"] & master["has_fundamentals"] & master["fresh_filing"]

    n = len(master)
    logger.info(
        f"master: {n:,} member-months | price {master['has_price'].mean():.1%} | "
        f"fundamentals {master['has_fundamentals'].mean():.1%} | "
        f"investable {master['investable'].mean():.1%}"
    )
    return master.sort_values(["date", "symbol"]).reset_index(drop=True)


def main() -> None:
    settings.ensure_dirs()
    master = assemble_master()
    out = settings.processed_dir / "master_pit.parquet"
    write_parquet(master, out)
    logger.info(f"wrote {out} ({len(master):,} rows, {master.shape[1]} cols)")


if __name__ == "__main__":
    main()
