"""End-to-end raw data collection for the point-in-time S&P 500 universe.

Run::

    uv run python -m qfr.data.collect

Writes Parquet files under ``data/processed/``. Idempotent: the FMP client
caches every response on disk, so re-running resumes cheaply (only un-cached
symbols hit the network).
"""

from __future__ import annotations

from qfr.data.fmp_client import FMPClient
from qfr.data.fundamentals import STATEMENT_METHODS, fetch_profiles, fetch_statement_long
from qfr.data.prices import fetch_prices_long, fetch_split_adjusted_close
from qfr.data.universe import all_symbols, build_universe
from qfr.utils.config import settings
from qfr.utils.dates import month_end_dates
from qfr.utils.io import write_parquet
from qfr.utils.logging import logger

# Universe discovery window. Starts a year before the 2000 backtest start so that
# 12-month momentum and fundamental lags are available from the first rebalance.
UNIVERSE_START = "1999-01-31"
UNIVERSE_END = "2026-04-30"
# Price pagination: the endpoint caps at ~5,000 daily bars per request, so the
# fetcher walks the cursor back from UNIVERSE_END to PRICE_START_NEEDED, never
# requesting older than PRICE_FLOOR.
PRICE_FLOOR = "1990-01-01"
PRICE_START_NEEDED = "1998-06-01"


def main() -> None:
    settings.ensure_dirs()
    out = settings.processed_dir
    client = FMPClient()

    # 1) Point-in-time universe -------------------------------------------
    dates = month_end_dates(UNIVERSE_START, UNIVERSE_END)
    panel, members_now, changes = build_universe(dates, client=client)
    write_parquet(panel, out / "universe_panel.parquet")
    write_parquet(members_now, out / "members_now.parquet")
    write_parquet(changes, out / "constituent_changes.parquet")
    symbols = all_symbols(panel)
    logger.info(f"Universe symbols to pull: {len(symbols)}")

    # 2) Adjusted prices (paginated back through the ~5,000-bar cap) -------
    prices = fetch_prices_long(
        symbols,
        client=client,
        floor=PRICE_FLOOR,
        to_date=UNIVERSE_END,
        start_needed=PRICE_START_NEEDED,
    )
    write_parquet(prices, out / "prices_long.parquet")
    logger.info(f"prices_long: {len(prices):,} rows -> {prices['symbol'].nunique() if len(prices) else 0} symbols")

    # 2b) Split-adjusted (dividend-unadjusted) close — actual price levels for
    #     refreshing value ratios to the live rebalance date (2010+ window).
    raw_close = fetch_split_adjusted_close(symbols, client=client, from_date="2009-01-01", to_date=UNIVERSE_END)
    write_parquet(raw_close, out / "prices_raw_long.parquet")
    logger.info(f"prices_raw_long: {len(raw_close):,} rows -> {raw_close['symbol'].nunique() if len(raw_close) else 0} symbols")

    # 3) Fundamentals (7 quarterly datasets) ------------------------------
    for kind in STATEMENT_METHODS:
        df = fetch_statement_long(kind, symbols, client=client, period="quarter", limit=400)
        write_parquet(df, out / f"fund_{kind}.parquet")
        logger.info(f"fund_{kind}: {len(df):,} rows")

    # 4) Company profiles (sector / beta / industry / IPO) ----------------
    profiles = fetch_profiles(symbols, client=client)
    write_parquet(profiles, out / "profiles.parquet")
    logger.info(f"profiles: {len(profiles):,} rows")

    logger.info("Data collection complete.")


if __name__ == "__main__":
    main()
