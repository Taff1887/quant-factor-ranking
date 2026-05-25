"""Point-in-time S&P 500 universe reconstruction.

FMP exposes the *current* index members plus a chronological log of *changes*
(additions / removals with effective dates). Membership as-of any past date is
recovered by starting from today's members and walking the change-log
**backwards in time**: for every change that happened *after* the target date we
undo it — discarding the security that was added and restoring the one that was
removed.

Processing strictly in reverse-chronological order is important: a name that was
added and later removed (both after the target date) must end up *excluded*,
which only holds if the later change is undone first.

The output is a survivorship-bias-free membership panel — on each rebalance date
we only ever see the names that were genuinely in the index at that time.

Caveat: reconstruction quality is bounded by the completeness of FMP's change
log; very old history (pre-~2014) should be validated empirically (Part 1).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from qfr.data.fmp_client import FMPClient
from qfr.utils.logging import logger


def parse_changes(raw: list[dict]) -> pd.DataFrame:
    """Normalise the FMP change-log into ``[date, added, removed]`` rows."""
    rows = []
    for rec in raw:
        date = rec.get("date")
        if not date:
            continue
        added = (rec.get("symbol") or "").strip() or None
        removed = (rec.get("removedTicker") or "").strip() or None
        if added is None and removed is None:
            continue
        rows.append({"date": pd.Timestamp(date), "added": added, "removed": removed})
    if not rows:
        return pd.DataFrame(columns=["date", "added", "removed"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def current_members(raw: list[dict]) -> pd.DataFrame:
    """Current constituents as a DataFrame (symbol, name, sector, ...)."""
    df = pd.DataFrame(raw)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip()
    return df


def reconstruct_membership(
    today: set[str], changes: pd.DataFrame, as_of: pd.Timestamp
) -> set[str]:
    """Set of index members as-of ``as_of`` (inclusive)."""
    members = set(today)
    future = changes[changes["date"] > as_of].sort_values("date", ascending=False)
    for _, ch in future.iterrows():
        added, removed = ch["added"], ch["removed"]
        # Guard against NaN/None: only act on genuine ticker strings. (float('nan')
        # is truthy, so a bare `if removed:` would otherwise inject NaN into the set.)
        if isinstance(added, str) and added:
            members.discard(added)  # wasn't a member before it was added
        if isinstance(removed, str) and removed:
            members.add(removed)  # was a member before it was removed
    return members


def membership_panel(
    today: set[str], changes: pd.DataFrame, dates: Iterable[pd.Timestamp]
) -> pd.DataFrame:
    """Long ``[date, symbol]`` panel of point-in-time membership."""
    records = []
    for d in dates:
        ts = pd.Timestamp(d)
        for sym in reconstruct_membership(today, changes, ts):
            records.append((ts, sym))
    return pd.DataFrame(records, columns=["date", "symbol"]).sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)


def build_universe(
    dates: Iterable[pd.Timestamp],
    client: FMPClient | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the point-in-time membership panel.

    Returns
    -------
    panel : DataFrame ``[date, symbol]``
        Point-in-time membership across the requested rebalance dates.
    members_now : DataFrame
        Current constituents with metadata (sector etc.).
    changes : DataFrame ``[date, added, removed]``
        Parsed historical change log.
    """
    client = client or FMPClient()
    raw_current = client.sp500_constituents(force_refresh=force_refresh)
    raw_changes = client.historical_sp500_constituents(force_refresh=force_refresh)

    members_now = current_members(raw_current)
    today = set(members_now["symbol"]) if "symbol" in members_now else set()
    changes = parse_changes(raw_changes)

    logger.info(
        f"Current members: {len(today)} | change records: {len(changes)} | "
        f"rebalance dates: {len(list(dates)) if hasattr(dates, '__len__') else '?'}"
    )
    panel = membership_panel(today, changes, dates)
    logger.info(
        f"Membership panel: {len(panel):,} rows | "
        f"{panel['symbol'].nunique()} unique symbols ever in-universe"
    )
    return panel, members_now, changes


def all_symbols(panel: pd.DataFrame) -> list[str]:
    """Every symbol that appears in the membership panel (for bulk data pulls)."""
    return sorted(panel["symbol"].unique())
