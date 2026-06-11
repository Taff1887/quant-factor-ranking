"""Pull earnings-call transcripts from FMP for a sample of S&P 500 tickers.

We're starting with a focused validation: ~30 large-cap S&P 500 stocks across
sectors, ~16 quarters (2020 Q1 -> 2023 Q4). That gives us ~480 transcripts to
test whether a transcript-derived signal has any IC vs forward returns. Only
if validation passes do we expand to the full universe.

Run::  uv run python -m qfr.transcripts.pull
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qfr.data.fmp_client import FMPClient
from qfr.utils.config import PROJECT_ROOT
from qfr.utils.logging import logger

# Where we cache the raw transcripts (one file per (symbol, year, quarter))
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "raw" / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Index of what we've pulled (symbol, year, quarter, date, file_path)
TRANSCRIPTS_INDEX = PROJECT_ROOT / "data" / "raw" / "transcripts" / "_index.parquet"

# 30 large-cap S&P 500 tickers across sectors -- enough sample for a clean IC test
SAMPLE_TICKERS: list[str] = [
    # Tech / comms
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "LLY",
    # Financials
    "JPM", "BAC", "V", "MA", "GS",
    # Energy
    "XOM", "CVX", "COP",
    # Consumer
    "WMT", "PG", "KO", "COST", "MCD",
    # Industrials
    "BA", "CAT", "GE",
    # Materials / Utilities / Real Estate
    "LIN", "NEE", "AMT",
]

# 16 quarters covering 2020 Q1 .. 2023 Q4 -- enough for ~30 transcripts/ticker
SAMPLE_PERIODS: list[tuple[int, int]] = [
    (y, q) for y in (2020, 2021, 2022, 2023) for q in (1, 2, 3, 4)
]


def _path_for(symbol: str, year: int, quarter: int) -> Path:
    return TRANSCRIPTS_DIR / f"{symbol}_{year}_Q{quarter}.json"


def pull_one(symbol: str, year: int, quarter: int,
             *, force: bool = False) -> dict | None:
    """Pull a single transcript and cache to disk. Returns the parsed dict or None
    if FMP doesn't have one for that period."""
    path = _path_for(symbol, year, quarter)
    if path.exists() and not force:
        with open(path) as f:
            data = json.load(f)
        return data if data else None

    c = FMPClient()
    try:
        rows = c.get("earning-call-transcript",
                     params={"symbol": symbol, "year": year, "quarter": quarter})
    except Exception as e:
        logger.warning(f"  {symbol} {year}Q{quarter}: FMP error: {str(e)[:80]}")
        path.write_text("{}")
        return None

    if not rows:
        path.write_text("{}")
        return None
    rec = rows[0]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    return rec


def pull_sample(*, force: bool = False) -> pd.DataFrame:
    """Pull the validation sample. Returns a DataFrame indexing what we got."""
    index_rows = []
    n_total = len(SAMPLE_TICKERS) * len(SAMPLE_PERIODS)
    n_done = 0
    n_ok = 0
    for sym in SAMPLE_TICKERS:
        for (year, q) in SAMPLE_PERIODS:
            n_done += 1
            rec = pull_one(sym, year, q, force=force)
            if rec and rec.get("content"):
                content = rec.get("content", "")
                index_rows.append({
                    "symbol": sym,
                    "year": year,
                    "quarter": q,
                    "date": rec.get("date"),
                    "content_chars": len(content),
                    "path": str(_path_for(sym, year, q).relative_to(PROJECT_ROOT)),
                })
                n_ok += 1
            if n_done % 50 == 0:
                logger.info(f"  pull progress: {n_done}/{n_total}  ({n_ok} with content)")

    df = pd.DataFrame(index_rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df.to_parquet(TRANSCRIPTS_INDEX, index=False)
    logger.info(f"Sample pulled: {n_ok}/{n_total} transcripts have content; "
                f"index at {TRANSCRIPTS_INDEX}")
    return df


def load_transcript(symbol: str, year: int, quarter: int) -> str | None:
    """Load a cached transcript's raw text content. Returns None if missing/empty."""
    path = _path_for(symbol, year, quarter)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("content") if data else None


def main() -> None:
    df = pull_sample()
    if not len(df):
        logger.warning("No transcripts pulled. Check FMP API key + subscription tier.")
        return
    logger.info(f"\n=== sample index ===")
    logger.info(f"  unique symbols with at least one transcript: {df['symbol'].nunique()}")
    logger.info(f"  total transcripts: {len(df)}")
    logger.info(f"  median chars: {df['content_chars'].median():.0f}")
    logger.info(f"  date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    logger.info(f"  by year: \n{df.groupby('year').size().to_string()}")


if __name__ == "__main__":
    main()
