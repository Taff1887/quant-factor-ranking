"""Cross-validate FMP prices/returns against Yahoo Finance (independent sense check).

Adjusted *price levels* differ between vendors (each normalises to its own latest
price), so the apples-to-apples comparison is **monthly returns**, which are
invariant to the adjustment base. Large return discrepancies surface the data
hazards that matter: missed splits/dividends, bad ticks, and ticker recycling
(a delisted ticker reused for a different company).

Run::

    uv run python -m qfr.data.validate_yahoo

Outputs (gitignored, under data/processed/):
    yahoo_validation_obs.parquet        per (symbol, month) FMP vs Yahoo return
    yahoo_validation_by_symbol.parquet  per-symbol agreement stats
Yahoo's raw monthly closes are cached under data/external/yahoo_monthly.parquet.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yfinance as yf

from qfr.utils.config import settings
from qfr.utils.io import read_parquet, write_parquet
from qfr.utils.logging import logger


def _yahoo_cache() -> Path:
    return settings.external_dir / "yahoo_monthly.parquet"


def _extract_close(df: pd.DataFrame, yahoo_tickers: list[str]) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            return df["Close"].copy()
        return None
    if "Close" in df.columns:  # single-ticker frame
        out = df[["Close"]].copy()
        out.columns = [yahoo_tickers[0]]
        return out
    return None


def fetch_yahoo_monthly(
    symbols: Iterable[str],
    *,
    start: str = "1999-12-01",
    end: str = "2026-05-01",
    chunk: int = 40,
    pause: float = 0.8,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Monthly adjusted closes from Yahoo, long format [date, symbol, y_adjclose].

    Symbols are mapped to Yahoo convention (``BRK.B`` -> ``BRK-B``). Results are
    cached; only un-cached symbols hit the network.
    """
    symbols = list(dict.fromkeys(symbols))
    cache = _yahoo_cache()
    frames: list[pd.DataFrame] = []
    have: set[str] = set()
    if use_cache and cache.exists():
        cached = read_parquet(cache)
        frames.append(cached)
        have = set(cached["symbol"].unique())
    todo = [s for s in symbols if s not in have]
    logger.info(f"Yahoo: {len(todo)} symbols to fetch ({len(have)} already cached)")

    for i in range(0, len(todo), chunk):
        batch = todo[i : i + chunk]
        ymap = {s: s.replace(".", "-") for s in batch}  # fmp -> yahoo
        rev = {v: k for k, v in ymap.items()}
        try:
            df = yf.download(
                list(ymap.values()), start=start, end=end, interval="1mo",
                auto_adjust=True, progress=False, threads=True,
            )
            close = _extract_close(df, list(ymap.values()))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"yahoo chunk @{i}: {type(e).__name__}: {e}")
            close = None
        if close is not None and not close.empty:
            close = close.rename(columns=rev)
            long = (
                close.reset_index(names="date")
                .melt(id_vars="date", var_name="symbol", value_name="y_adjclose")
                .dropna(subset=["y_adjclose"])
            )
            frames.append(long[["date", "symbol", "y_adjclose"]])
        logger.info(f"yahoo: {min(i + chunk, len(todo))}/{len(todo)} fetched")
        time.sleep(pause)

    out = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "date"])
    out["date"] = pd.to_datetime(out["date"])
    write_parquet(out, cache)
    return out


def compare(master: pd.DataFrame, yahoo: pd.DataFrame) -> pd.DataFrame:
    """Align FMP vs Yahoo by calendar month and compute return differences."""
    fmp = master.loc[master["has_price"], ["date", "symbol", "adjClose"]].copy()
    fmp["ym"] = fmp["date"].dt.to_period("M")
    y = yahoo.copy()
    y["ym"] = y["date"].dt.to_period("M")
    m = fmp.merge(y[["symbol", "ym", "y_adjclose"]], on=["symbol", "ym"], how="inner")
    m = m.sort_values(["symbol", "ym"])
    m["fmp_ret"] = m.groupby("symbol")["adjClose"].pct_change()
    m["y_ret"] = m.groupby("symbol")["y_adjclose"].pct_change()
    m["ret_diff"] = m["fmp_ret"] - m["y_ret"]
    m["abs_ret_diff"] = m["ret_diff"].abs()
    return m


def per_symbol_report(obs: pd.DataFrame) -> pd.DataFrame:
    d = obs.dropna(subset=["fmp_ret", "y_ret"])
    agg = d.groupby("symbol").agg(
        n_months=("ret_diff", "size"),
        median_abs_ret_diff=("abs_ret_diff", "median"),
        p95_abs_ret_diff=("abs_ret_diff", lambda s: s.quantile(0.95)),
        max_abs_ret_diff=("abs_ret_diff", "max"),
    )
    corr = d.groupby("symbol").apply(
        lambda g: g["fmp_ret"].corr(g["y_ret"]), include_groups=False
    ).rename("ret_corr")
    rep = agg.join(corr)
    # "Big" discrepancy = shape mismatch OR a large single-month gap. A small,
    # persistent median offset (~0.5%/m for high-yield names) is a dividend-
    # adjustment convention difference, not an error, so it does NOT flag.
    rep["flag"] = (rep["ret_corr"] < 0.95) | (rep["max_abs_ret_diff"] > 0.10)
    return rep.sort_values("max_abs_ret_diff", ascending=False)


def make_figures() -> list:
    """Validation figures -> charts/02_*.png (run after main())."""
    import matplotlib.pyplot as plt

    from qfr.utils.viz import PALETTE, save_fig, set_plot_style

    set_plot_style()
    obs = read_parquet(settings.processed_dir / "yahoo_validation_obs.parquet")
    rep = read_parquet(settings.processed_dir / "yahoo_validation_by_symbol.parquet")
    master = read_parquet(settings.processed_dir / "master_pit.parquet")
    paths = []

    fig, ax = plt.subplots()
    ax.hist(rep["ret_corr"].dropna().clip(-0.5, 1), bins=60, color=PALETTE["primary"], alpha=0.85)
    ax.axvline(0.99, color=PALETTE["accent"], ls="--", lw=2, label="0.99")
    ax.set_title("Per-symbol correlation of monthly returns (FMP vs Yahoo)")
    ax.set_xlabel("return correlation")
    ax.set_ylabel("number of symbols")
    ax.legend()
    paths.append(save_fig(fig, "02_return_corr_dist"))

    fig, ax = plt.subplots()
    ax.hist(obs["abs_ret_diff"].dropna().clip(0, 0.2), bins=80, color=PALETTE["green"], alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(0.005, color=PALETTE["accent"], ls="--", lw=2, label="0.5%/month")
    ax.set_title("Monthly |return difference| FMP vs Yahoo (log scale)")
    ax.set_xlabel("|FMP return − Yahoo return|  (clipped at 0.2)")
    ax.set_ylabel("observations (log)")
    ax.legend()
    paths.append(save_fig(fig, "02_return_diff_dist"))

    s = obs.dropna(subset=["fmp_ret", "y_ret"]).sample(min(30000, len(obs)), random_state=0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(s["y_ret"].clip(-0.5, 0.5), s["fmp_ret"].clip(-0.5, 0.5), s=3, alpha=0.15,
               color=PALETTE["primary"])
    ax.plot([-0.5, 0.5], [-0.5, 0.5], color=PALETTE["accent"], lw=1)
    ax.set_title("FMP vs Yahoo monthly returns (30k sample)")
    ax.set_xlabel("Yahoo monthly return")
    ax.set_ylabel("FMP monthly return")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    paths.append(save_fig(fig, "02_fmp_vs_yahoo_scatter"))

    priced = int(master.loc[master["has_price"], "symbol"].nunique())
    compared = int(rep["ret_corr"].notna().sum())
    fig, ax = plt.subplots()
    bars = ["FMP priced\nsymbols", "Validated\nby Yahoo", "Yahoo has\nNO data"]
    vals = [priced, compared, priced - compared]
    ax.bar(bars, vals, color=[PALETTE["primary"], PALETTE["green"], PALETTE["accent"]])
    for i, v in enumerate(vals):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    ax.set_title("Coverage: FMP keeps delisted names that Yahoo has dropped")
    ax.set_ylabel("number of symbols")
    paths.append(save_fig(fig, "02_coverage_comparison"))
    return paths


def main() -> None:
    settings.ensure_dirs()
    master = read_parquet(settings.processed_dir / "master_pit.parquet")
    symbols = sorted(master.loc[master["has_price"], "symbol"].unique())
    logger.info(f"Cross-validating {len(symbols)} symbols against Yahoo Finance")

    yahoo = fetch_yahoo_monthly(symbols)
    obs = compare(master, yahoo)
    rep = per_symbol_report(obs)

    write_parquet(obs, settings.processed_dir / "yahoo_validation_obs.parquet")
    write_parquet(rep.reset_index(), settings.processed_dir / "yahoo_validation_by_symbol.parquet")

    covered = set(yahoo["symbol"].unique())
    missing = [s for s in symbols if s not in covered]
    logger.info("=" * 60)
    logger.info(f"symbols compared        : {rep.shape[0]} / {len(symbols)}")
    logger.info(f"no Yahoo data           : {len(missing)}")
    logger.info(f"median return |diff|    : {obs['abs_ret_diff'].median():.5f}")
    logger.info(f"agree within 0.5%/month : {(obs['abs_ret_diff'] <= 0.005).mean():.1%}")
    logger.info(f"return corr >= 0.99     : {(rep['ret_corr'] >= 0.99).mean():.1%} of symbols")
    logger.info(f"FLAGGED symbols         : {int(rep['flag'].sum())}")
    logger.info("--- top 25 by largest single-month return gap ---")
    cols = ["n_months", "ret_corr", "median_abs_ret_diff", "max_abs_ret_diff"]
    for sym, row in rep.head(25)[cols].iterrows():
        logger.info(
            f"  {sym:6s} n={int(row.n_months):3d} corr={row.ret_corr:6.3f} "
            f"medDiff={row.median_abs_ret_diff:.4f} maxDiff={row.max_abs_ret_diff:.3f}"
        )
    logger.info("Yahoo cross-validation complete.")


if __name__ == "__main__":
    main()
