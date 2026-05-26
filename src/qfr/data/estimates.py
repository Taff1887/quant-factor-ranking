"""Analyst-sentiment data (rating actions) from the FMP stable API.

Why only rating actions (and not EPS-estimate revisions): FMP's
``analyst-estimates`` endpoint returns a single *current* consensus value per
fiscal period (a forward snapshot), with no history of how that estimate evolved
- so it cannot produce a look-ahead-free estimate-revision factor. The ``grades``
endpoint, by contrast, is a **dated log of individual analyst rating actions**
(upgrade / downgrade / maintain / initiate) back to ~2012, which *is* point-in-
time safe and is the basis for a recommendation-revision / sentiment factor.

Run (via collect, or standalone):  ``fetch_grades_long(symbols)`` -> grades_long.parquet
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from qfr.data.fmp_client import FMPClient
from qfr.utils.logging import logger

KEEP = ["symbol", "date", "gradingCompany", "previousGrade", "newGrade", "action"]


def fetch_grades_long(
    symbols: Iterable[str], client: FMPClient | None = None, *, log_every: int = 100
) -> pd.DataFrame:
    """Pull the full dated analyst rating-action log for ``symbols`` (one call each)."""
    client = client or FMPClient()
    symbols = list(symbols)
    frames: list[pd.DataFrame] = []
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        try:
            rows = client.grades(sym)
        except Exception as e:  # noqa: BLE001 - log and continue the bulk pull
            logger.warning(f"grades {sym}: {e}")
            rows = []
        if rows:
            df = pd.DataFrame(rows)
            if "symbol" not in df.columns:
                df["symbol"] = sym
            frames.append(df)
        if i % log_every == 0:
            logger.info(f"grades: {i}/{n} symbols")
    if not frames:
        return pd.DataFrame(columns=KEEP)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    keep = [c for c in KEEP if c in out.columns]
    return out[keep].dropna(subset=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
