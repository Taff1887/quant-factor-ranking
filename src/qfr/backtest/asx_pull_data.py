"""Pull a fresh, independent ASX universe + prices + fundamentals + free-float.

This module replaces the dependency on the Short King 2.0 panel and pulls
all data directly from FMP (prices, fundamentals) + Yahoo Finance (free float).

Run sequentially - each stage caches to disk so re-runs are cheap.

Run::  uv run python -m qfr.backtest.asx_pull_data
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from qfr.data.fmp_client import FMPClient
from qfr.utils.config import PROJECT_ROOT
from qfr.utils.logging import logger

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "asx"
DATA_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PARQUET = DATA_DIR / "asx_universe.parquet"
PRICES_PARQUET = DATA_DIR / "asx_prices.parquet"
FUNDAMENTALS_PARQUET = DATA_DIR / "asx_fundamentals.parquet"
FREEFLOAT_PARQUET = DATA_DIR / "asx_freefloat.parquet"

PRICE_START = "2009-12-01"   # 1m buffer before our 2010-01 analysis start
PRICE_END = "2026-05-31"
UNIVERSE_SIZE = 250          # buffer over 200 so PIT-top-200 rebalance has room


# --------------------------------------------------------------------------
# Step 1: Universe
# --------------------------------------------------------------------------
def pull_universe() -> pd.DataFrame:
    """Top ASX ordinaries by market cap (clean of ETFs, funds, pref shares)."""
    if UNIVERSE_PARQUET.exists():
        logger.info(f"Universe cached at {UNIVERSE_PARQUET}")
        return pd.read_parquet(UNIVERSE_PARQUET)

    c = FMPClient()
    rows = c.get("company-screener", params={
        "country": "AU", "isActivelyTrading": "true",
        "isEtf": "false", "isFund": "false", "exchange": "ASX",
        "limit": 1000,
    })
    df = pd.DataFrame(rows)
    # Keep only ordinaries: alpha-only tickers of length 2-4 before .AX
    def _is_ordinary(s: str) -> bool:
        base = s.replace(".AX", "")
        return 2 <= len(base) <= 4 and base.isalpha()
    df = df[df["symbol"].apply(_is_ordinary)].copy()
    df = df.sort_values("marketCap", ascending=False).head(UNIVERSE_SIZE).reset_index(drop=True)
    df["rank"] = df.index + 1
    df.to_parquet(UNIVERSE_PARQUET, index=False)
    logger.info(f"Universe saved: {len(df)} tickers (rank-200 mktCap = "
                f"${df.iloc[min(199, len(df)-1)]['marketCap']/1e9:.2f}B AUD)")
    return df


# --------------------------------------------------------------------------
# Step 2: Prices (dividend-adjusted)
# --------------------------------------------------------------------------
def pull_prices(universe: pd.DataFrame, *, force_refresh: bool = False) -> pd.DataFrame:
    """Monthly dividend-adjusted close for every ticker."""
    if PRICES_PARQUET.exists() and not force_refresh:
        logger.info(f"Prices cached at {PRICES_PARQUET}")
        return pd.read_parquet(PRICES_PARQUET)

    c = FMPClient()
    rows: list[dict] = []
    n = len(universe)
    for i, sym in enumerate(universe["symbol"]):
        try:
            data = c.historical_prices(sym, from_date=PRICE_START, to_date=PRICE_END,
                                       series="dividend-adjusted")
            for r in data:
                rows.append({"symbol": sym, "date": r.get("date"),
                             "adjClose": r.get("adjClose")})
        except Exception as e:
            logger.warning(f"  prices fail for {sym}: {e}")
        if (i + 1) % 25 == 0:
            logger.info(f"  prices: {i + 1}/{n}")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df.to_parquet(PRICES_PARQUET, index=False)
    logger.info(f"Prices saved: {len(df):,} rows for {df['symbol'].nunique()} tickers")
    return df


# --------------------------------------------------------------------------
# Step 3: Fundamentals (key_metrics + financial_growth + cash_flow)
# --------------------------------------------------------------------------
KEY_METRICS_FIELDS = ["date", "calendarYear", "period", "acceptedDate",
                      "returnOnEquity", "returnOnInvestedCapital", "marketCap",
                      "freeCashFlowYield"]
GROWTH_FIELDS = ["date", "calendarYear", "period", "acceptedDate",
                 "revenueGrowth", "epsgrowth"]
CASHFLOW_FIELDS = ["date", "calendarYear", "period", "acceptedDate",
                   "freeCashFlow"]


def _fetch_one_endpoint(c: FMPClient, endpoint: str, sym: str,
                        keep_fields: list[str]) -> list[dict]:
    rows = c.get(endpoint, params={"symbol": sym, "period": "quarter", "limit": 80})
    out = []
    for r in rows:
        out.append({k: r.get(k) for k in keep_fields if k in r or True})
    return out


def pull_fundamentals(universe: pd.DataFrame, *, force_refresh: bool = False
                      ) -> pd.DataFrame:
    """Quarterly fundamentals from FMP for each ticker (with filing-date acceptedDate)."""
    if FUNDAMENTALS_PARQUET.exists() and not force_refresh:
        logger.info(f"Fundamentals cached at {FUNDAMENTALS_PARQUET}")
        return pd.read_parquet(FUNDAMENTALS_PARQUET)

    c = FMPClient()
    km_rows: list[dict] = []
    gr_rows: list[dict] = []
    cf_rows: list[dict] = []
    n = len(universe)
    for i, sym in enumerate(universe["symbol"]):
        for endpoint, keep, bucket in (
            ("key-metrics", KEY_METRICS_FIELDS, km_rows),
            ("financial-growth", GROWTH_FIELDS, gr_rows),
            ("cash-flow-statement", CASHFLOW_FIELDS, cf_rows),
        ):
            try:
                rs = _fetch_one_endpoint(c, endpoint, sym, keep)
                for r in rs:
                    r["symbol"] = sym
                    bucket.append(r)
            except Exception as e:
                logger.warning(f"  {endpoint} fail for {sym}: {e}")
        if (i + 1) % 25 == 0:
            logger.info(f"  fundamentals: {i + 1}/{n}")

    km = pd.DataFrame(km_rows)
    gr = pd.DataFrame(gr_rows)
    cf = pd.DataFrame(cf_rows)
    # All three share (symbol, date, period) so merge on those
    merge_on = ["symbol", "date", "period", "acceptedDate"]
    df = km.merge(gr, on=merge_on, how="outer", suffixes=("", "_g"))
    df = df.merge(cf, on=merge_on, how="outer", suffixes=("", "_c"))
    df["date"] = pd.to_datetime(df["date"])
    df["acceptedDate"] = pd.to_datetime(df["acceptedDate"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df.to_parquet(FUNDAMENTALS_PARQUET, index=False)
    logger.info(f"Fundamentals saved: {len(df):,} rows for {df['symbol'].nunique()} tickers")
    return df


# --------------------------------------------------------------------------
# Step 4: Free float from Yahoo Finance
# --------------------------------------------------------------------------
def pull_freefloat(universe: pd.DataFrame, *, force_refresh: bool = False
                   ) -> pd.DataFrame:
    """Current floatShares and sharesOutstanding per ticker (Yahoo)."""
    if FREEFLOAT_PARQUET.exists() and not force_refresh:
        logger.info(f"Free-float cached at {FREEFLOAT_PARQUET}")
        return pd.read_parquet(FREEFLOAT_PARQUET)

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance is not installed. Run `uv add yfinance`.")
        raise

    rows: list[dict] = []
    n = len(universe)
    for i, sym in enumerate(universe["symbol"]):
        try:
            info = yf.Ticker(sym).info
            float_shares = info.get("floatShares")
            shares_out = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            if float_shares and shares_out and shares_out > 0:
                ratio = float(float_shares) / float(shares_out)
            else:
                ratio = None
            rows.append({"symbol": sym, "float_shares": float_shares,
                         "shares_outstanding": shares_out,
                         "free_float_ratio": ratio})
        except Exception as e:
            logger.warning(f"  free-float fail for {sym}: {e}")
            rows.append({"symbol": sym, "float_shares": None,
                         "shares_outstanding": None, "free_float_ratio": None})
        time.sleep(0.2)
        if (i + 1) % 25 == 0:
            logger.info(f"  free-float: {i + 1}/{n}")
    df = pd.DataFrame(rows)
    df.to_parquet(FREEFLOAT_PARQUET, index=False)
    logger.info(f"Free-float saved: {df['free_float_ratio'].notna().sum()}/{len(df)} "
                f"with valid ratio; median ratio = {df['free_float_ratio'].median():.3f}")
    return df


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Step 1: pulling universe ===")
    universe = pull_universe()
    logger.info(f"Universe: {len(universe)} tickers")

    logger.info("\n=== Step 2: pulling prices ===")
    prices = pull_prices(universe)
    logger.info(f"Prices: {len(prices):,} rows, {prices['symbol'].nunique()} unique tickers, "
                f"{prices['date'].min().date()} -> {prices['date'].max().date()}")

    logger.info("\n=== Step 3: pulling fundamentals ===")
    fundamentals = pull_fundamentals(universe)
    logger.info(f"Fundamentals: {len(fundamentals):,} rows, {fundamentals['symbol'].nunique()} unique tickers")

    logger.info("\n=== Step 4: pulling free-float (Yahoo) ===")
    freefloat = pull_freefloat(universe)
    logger.info(f"Free-float: {freefloat['free_float_ratio'].notna().sum()}/{len(freefloat)} valid")


if __name__ == "__main__":
    main()
