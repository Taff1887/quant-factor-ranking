"""Fundamental statement assembly from the FMP stable API.

Pulls the seven quarterly datasets used for factor construction. The three core
statements (income / balance / cash-flow) carry ``filingDate`` and
``acceptedDate`` — the public-availability timestamps that drive point-in-time
lagging. The derived datasets (ratios, key-metrics, enterprise-values,
financial-growth) carry only the period-end ``date``; their availability is
mapped to the parent filing during panel assembly (join on symbol + fiscalYear +
period).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from qfr.data.fmp_client import FMPClient
from qfr.utils.logging import logger

# logical name -> FMPClient method
STATEMENT_METHODS = {
    "income": "income_statement",
    "balance": "balance_sheet",
    "cashflow": "cash_flow",
    "ratios": "ratios",
    "key_metrics": "key_metrics",
    "enterprise": "enterprise_values",
    "growth": "financial_growth",
}

_DATE_COLS = ("date", "filingDate", "acceptedDate")


def fetch_statement_long(
    kind: str,
    symbols: Iterable[str],
    client: FMPClient | None = None,
    *,
    period: str = "quarter",
    limit: int = 400,
    log_every: int = 50,
) -> pd.DataFrame:
    """Pull one statement type (see ``STATEMENT_METHODS``) for all ``symbols``."""
    if kind not in STATEMENT_METHODS:
        raise ValueError(f"unknown statement kind {kind!r}; expected {list(STATEMENT_METHODS)}")
    client = client or FMPClient()
    method = getattr(client, STATEMENT_METHODS[kind])
    symbols = list(symbols)
    frames: list[pd.DataFrame] = []
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        try:
            rows = method(sym, period=period, limit=limit)
        except Exception as e:  # noqa: BLE001 - log and continue the bulk pull
            logger.warning(f"{kind} {sym}: {e}")
            rows = []
        if rows:
            df = pd.DataFrame(rows)
            if "symbol" not in df.columns:
                df["symbol"] = sym
            frames.append(df)
        if i % log_every == 0:
            logger.info(f"{kind}: {i}/{n} symbols")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in _DATE_COLS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    sort_cols = [c for c in ("symbol", "date") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def fetch_profiles(
    symbols: Iterable[str],
    client: FMPClient | None = None,
    *,
    log_every: int = 100,
) -> pd.DataFrame:
    """Company profiles (sector, industry, beta, IPO date, ...) for all symbols."""
    client = client or FMPClient()
    symbols = list(symbols)
    rows: list[dict] = []
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        try:
            p = client.profile(sym)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"profile {sym}: {e}")
            p = {}
        if p:
            rows.append(p)
        if i % log_every == 0:
            logger.info(f"profiles: {i}/{n}")
    return pd.DataFrame(rows)
