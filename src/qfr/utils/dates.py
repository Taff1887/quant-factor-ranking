"""Date helpers for rebalance scheduling."""

from __future__ import annotations

import pandas as pd


def month_end_dates(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Calendar month-end dates in ``[start, end]`` (pandas ``ME`` frequency)."""
    return pd.date_range(start=start, end=end, freq="ME")


def quarter_end_dates(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="QE")
