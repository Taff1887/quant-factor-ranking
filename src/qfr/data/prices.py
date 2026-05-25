"""Adjusted daily price assembly from the FMP stable API.

We use the ``dividend-adjusted`` EOD series so that ``adjClose`` reflects both
splits and dividends — the correct basis for computing total **returns** (the
``adjClose`` is a total-return index, not a price level).

For value-factor *levels* we also pull the ``full`` series, whose ``close`` is
**split-adjusted (to today's basis) but NOT dividend-adjusted** — i.e. an actual
price level, comparable across splits. That lets us refresh value ratios with the
live rebalance-date price instead of the stale period-end price FMP bakes into
its ratios (see ``qfr.factors.build``). FMP caps these endpoints at ~5,000 daily
bars per request (~20 years), which fully covers the 2010+ analysis window.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from qfr.data.fmp_client import FMPClient
from qfr.utils.logging import logger

PRICE_COLS = ["symbol", "date", "adjOpen", "adjHigh", "adjLow", "adjClose", "volume"]


def _fetch_symbol_paginated(
    client: FMPClient,
    symbol: str,
    *,
    floor: str,
    to_date: str | None,
    series: str,
    start_needed: pd.Timestamp,
    max_calls: int = 5,
) -> list[dict]:
    """Walk the ``to`` cursor backwards to defeat the ~5,000-bar/request cap.

    Each request fetches up to ~5,000 bars ending at ``end``; we then move ``end``
    to the day before the earliest bar seen and repeat, until we reach
    ``start_needed`` or a short chunk signals the start of the symbol's history.
    """
    seen: dict[str, dict] = {}
    end = to_date
    for _ in range(max_calls):
        rows = client.historical_prices(symbol, from_date=floor, to_date=end, series=series)
        if not rows:
            break
        for r in rows:
            seen[r["date"]] = r
        earliest = min(r["date"] for r in rows)
        if pd.Timestamp(earliest) <= start_needed or len(rows) < 4999:
            break
        end = (pd.Timestamp(earliest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return list(seen.values())


def fetch_prices_long(
    symbols: Iterable[str],
    client: FMPClient | None = None,
    *,
    floor: str = "1990-01-01",
    to_date: str | None = None,
    start_needed: str = "1998-06-01",
    series: str = "dividend-adjusted",
    paginate: bool = True,
    log_every: int = 50,
) -> pd.DataFrame:
    """Pull adjusted daily prices for ``symbols`` into one long DataFrame.

    With ``paginate=True`` the ``to`` cursor is walked backwards so history before
    the ~5,000-bar cap (pre-2006) is retrieved down to ``start_needed``. Returns
    ``[symbol, date, adjOpen, adjHigh, adjLow, adjClose, volume]`` (whichever are
    present), sorted by ``[symbol, date]``.
    """
    client = client or FMPClient()
    symbols = list(symbols)
    start_ts = pd.Timestamp(start_needed)
    frames: list[pd.DataFrame] = []
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        try:
            if paginate:
                rows = _fetch_symbol_paginated(
                    client,
                    sym,
                    floor=floor,
                    to_date=to_date,
                    series=series,
                    start_needed=start_ts,
                )
            else:
                rows = client.historical_prices(
                    sym, from_date=floor, to_date=to_date, series=series
                )
        except Exception as e:  # noqa: BLE001 - log and continue the bulk pull
            logger.warning(f"prices {sym}: {e}")
            rows = []
        if rows:
            frames.append(pd.DataFrame(rows))
        if i % log_every == 0:
            logger.info(f"prices: {i}/{n} symbols")
    if not frames:
        return pd.DataFrame(columns=PRICE_COLS)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    keep = [c for c in PRICE_COLS if c in out.columns]
    return out[keep].sort_values(["symbol", "date"]).reset_index(drop=True)


def fetch_split_adjusted_close(
    symbols: Iterable[str],
    client: FMPClient | None = None,
    *,
    from_date: str = "2009-01-01",
    to_date: str | None = None,
    log_every: int = 100,
) -> pd.DataFrame:
    """Pull the ``full``-series ``close`` (split-adjusted, dividend-UNadjusted).

    Unlike ``adjClose`` (a total-return index), this ``close`` is an actual price
    *level* back-adjusted only for splits, so it is comparable to FMP's
    per-share fundamentals and period-end ``stockPrice`` (also split-adjusted to
    today). Used to refresh value ratios to the live rebalance-date price.

    One request per symbol (the 2009+ window is < 5,000 bars, under the cap).
    Returns ``[symbol, date, splitAdjClose]`` sorted by ``[symbol, date]``.
    """
    client = client or FMPClient()
    symbols = list(symbols)
    frames: list[pd.DataFrame] = []
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        try:
            rows = client.historical_prices(sym, from_date=from_date, to_date=to_date, series="full")
        except Exception as e:  # noqa: BLE001 - log and continue the bulk pull
            logger.warning(f"split-adj prices {sym}: {e}")
            rows = []
        if rows:
            df = pd.DataFrame(rows)
            if "close" in df.columns:
                df = df[["date", "close"]].copy()
                df["symbol"] = sym
                frames.append(df)
        if i % log_every == 0:
            logger.info(f"split-adj prices: {i}/{n} symbols")
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "splitAdjClose"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.rename(columns={"close": "splitAdjClose"})
    return out[["symbol", "date", "splitAdjClose"]].sort_values(["symbol", "date"]).reset_index(drop=True)
