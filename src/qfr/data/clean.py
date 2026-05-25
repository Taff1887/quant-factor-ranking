"""Data-quality remediation informed by the Yahoo cross-validation.

Conservative policy (see notebooks/02_data_validation):

* **Exclude corrupted series** — symbols whose monthly returns barely correlate
  with Yahoo *and* carry a large persistent gap (e.g. CPWR, whose FMP adjusted
  prices are off by ~200x). These are genuine vendor errors.
* **Null bad/ambiguous months** — individual (symbol, month) cells where FMP and
  Yahoo monthly returns disagree by more than ``SUSPECT_DIFF`` (isolated bad
  ticks like FMC 2009-08, and spinoff-month distortions like MO 2008). The
  *symbol* is kept; only the unreliable month's price is dropped.
* Forward returns are then recomputed on the cleaned, balanced price panel so a
  nulled month correctly invalidates every label that spans it.

Reads ``master_pit.parquet`` (+ the validation outputs) and writes
``master_clean.parquet`` — the canonical analytics panel for Parts 2+.

Run::

    uv run python -m qfr.data.clean
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.data.assemble import (
    REBALANCE_END,
    REBALANCE_START,
    add_forward_returns,
    month_end_price_panel,
)
from qfr.utils.config import settings
from qfr.utils.dates import month_end_dates
from qfr.utils.io import read_parquet, write_parquet
from qfr.utils.logging import logger

CORRUPT_CORR = 0.50  # return correlation below this ...
CORRUPT_MED = 0.05  # ... AND median monthly |diff| above this => corrupted series
SUSPECT_DIFF = 0.50  # monthly return |FMP - Yahoo| above this => null that month
HORIZONS = (1, 3, 6)


def corrupted_symbols(rep: pd.DataFrame) -> set[str]:
    mask = (rep["ret_corr"] < CORRUPT_CORR) & (rep["median_abs_ret_diff"] > CORRUPT_MED)
    return set(rep.loc[mask, "symbol"])


def suspect_month_keys(obs: pd.DataFrame) -> set[tuple]:
    s = obs.loc[obs["abs_ret_diff"] > SUSPECT_DIFF, ["symbol", "ym"]].copy()
    # Normalise the month to a 'YYYY-MM' string (robust to Period/Timestamp round-trips).
    ym = pd.to_datetime(s["ym"].astype(str)).dt.strftime("%Y-%m")
    return set(zip(s["symbol"].to_numpy(), ym.to_numpy()))


def clean() -> pd.DataFrame:
    proc = settings.processed_dir
    master = read_parquet(proc / "master_pit.parquet")
    rep = read_parquet(proc / "yahoo_validation_by_symbol.parquet")
    obs = read_parquet(proc / "yahoo_validation_obs.parquet")

    corrupt = corrupted_symbols(rep)
    suspect = suspect_month_keys(obs)
    logger.info(f"corrupted symbols excluded : {len(corrupt)} -> {sorted(corrupt)}")
    logger.info(f"suspect (symbol, month) nulled : {len(suspect)}")

    # Rebuild the balanced month-end price panel, null the bad cells, recompute
    # forward returns there (so membership gaps and label windows stay correct).
    prices = read_parquet(proc / "prices_long.parquet")
    reb = month_end_dates(REBALANCE_START, REBALANCE_END)
    pxp = month_end_price_panel(prices, reb)
    pxp_ym = pxp["date"].dt.strftime("%Y-%m")
    susp_mask = np.array(
        [(s, y) in suspect for s, y in zip(pxp["symbol"].to_numpy(), pxp_ym.to_numpy())]
    )
    bad = pxp["symbol"].isin(corrupt).to_numpy() | susp_mask
    pxp.loc[bad, "adjClose"] = np.nan
    pxp = add_forward_returns(pxp)

    # Splice cleaned price + labels back onto the panel.
    price_cols = ["adjClose", "ret_fwd_1m", "ret_fwd_3m", "ret_fwd_6m"]
    base = master.drop(columns=price_cols + ["has_price", "investable"])
    out = base.merge(pxp[["date", "symbol", *price_cols]], on=["date", "symbol"], how="left")

    out["excluded_corrupted"] = out["symbol"].isin(corrupt)
    out_ym = out["date"].dt.strftime("%Y-%m")
    out["suspect_month"] = [
        (s, y) in suspect for s, y in zip(out["symbol"].to_numpy(), out_ym.to_numpy())
    ]
    out["has_price"] = out["adjClose"].notna()
    out["investable"] = (
        out["has_price"]
        & out["has_fundamentals"]
        & out["fresh_filing"]
        & ~out["excluded_corrupted"]
    )
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    before = int(master["investable"].sum())
    after = int(out["investable"].sum())
    logger.info(f"investable member-months: {before:,} -> {after:,} (removed {before - after:,})")
    write_parquet(out, proc / "master_clean.parquet")
    logger.info(f"wrote {proc / 'master_clean.parquet'} ({len(out):,} rows, {out.shape[1]} cols)")
    return out


def main() -> None:
    clean()


if __name__ == "__main__":
    main()
