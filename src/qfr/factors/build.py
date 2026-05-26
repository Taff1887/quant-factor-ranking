"""Part 3 - Factor construction.

Builds five factor families (Value, Quality, Momentum, Growth, Risk) plus a Size
control from the clean point-in-time panel.

Construction: each raw component is winsorised (1/99) then converted to a
**cross-sectional percentile rank within each rebalance date**, oriented so that
*higher = better / more-exposed*. Family composites are the mean of their
component ranks, then re-ranked to a clean [0, 1]. Rank-based construction is the
deliberate choice because the EDA showed the raw inputs are extremely fat-tailed
(skew in the hundreds), which would let a few names dominate a z-score.

Price-based factors (momentum, volatility) are computed point-in-time from the
month-end price panel; fundamentals come from ``master_clean`` (already
filing-date-lagged, so no look-ahead).

Run::

    uv run python -m qfr.factors.build

Output: ``data/processed/factors.parquet``  (date x symbol x factor scores + labels)

NOTE: built on the FMP 2010+ window for now; re-run unchanged on the CRSP panel
once WRDS access lands (this module reads ``master_clean`` regardless of source).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.data.assemble import month_end_price_panel
from qfr.factors.transforms import cs_rank, cs_winsorize
from qfr.utils.config import settings
from qfr.utils.dates import month_end_dates
from qfr.utils.io import read_parquet, write_parquet
from qfr.utils.logging import logger

PRIMARY_START = "2010-01-31"  # FMP high-coverage window; CRSP will extend to 2000 later

# Each component is defined/derived so that HIGHER = better (more factor exposure).
FAMILIES: dict[str, list[str]] = {
    "value": ["earningsYield", "freeCashFlowYield", "bookToMarket", "salesYield", "ebitdaToEV"],
    "quality": ["returnOnEquity", "returnOnInvestedCapital", "returnOnAssets",
                "grossProfitMargin", "operatingProfitMargin", "netProfitMargin",
                "interestCoverageRatio", "lowLeverage"],
    "momentum": ["mom_12_1", "mom_6_1", "mom_3_1"],
    "growth": ["revenueGrowth", "epsgrowth", "netIncomeGrowth", "ebitdaGrowth"],
    "risk": ["lowVol", "lowLeverage"],
    "sentiment": ["rating_rev_3m", "rating_rev_6m", "rating_breadth_12m"],  # analyst rec. revisions
}


def _price_factors(rebalance: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time momentum and volatility from the month-end price panel."""
    prices = read_parquet(settings.processed_dir / "prices_long.parquet")
    px = month_end_price_panel(prices, rebalance).sort_values(["symbol", "date"])
    g = px.groupby("symbol")["adjClose"]
    px["ret_1m"] = g.pct_change()
    px["mom_12_1"] = g.shift(1) / g.shift(12) - 1.0  # 12-1 momentum (skip last month)
    px["mom_6_1"] = g.shift(1) / g.shift(6) - 1.0
    px["mom_3_1"] = g.shift(1) / g.shift(3) - 1.0
    px["vol_12m"] = (
        px.groupby("symbol")["ret_1m"].transform(lambda s: s.rolling(12, min_periods=6).std())
        * np.sqrt(12)
    )
    px["st_rev"] = -px["ret_1m"]  # short-term reversal: last month's loser = high score
    cols = ["mom_12_1", "mom_6_1", "mom_3_1", "vol_12m", "st_rev"]
    px[cols] = px[cols].replace([np.inf, -np.inf], np.nan)
    return px[["date", "symbol", *cols]]


def _refresh_value_prices(m: pd.DataFrame) -> pd.DataFrame:
    """Attach ``price_ratio`` = price(period-end) / price(rebalance) to the panel.

    FMP bakes the **period-end** price into its value ratios and holds it stale
    until the next filing, so the price inside our value signal is ~2-3 months
    old. Multiplying a yield (earnings/book/sales/FCF over price) by this ratio
    swaps that stale price for the **live rebalance-date** price, leaving the
    lagged fundamental untouched. Both legs use the split-adjusted,
    dividend-UNadjusted close (``prices_raw_long``), so the rescaling is
    split-safe and free of total-return (dividend) contamination. Rows with no
    matched price get ``price_ratio`` NaN -> caller falls back to the stale ratio.
    """
    sdf = read_parquet(settings.processed_dir / "prices_raw_long.parquet")
    sdf["date"] = pd.to_datetime(sdf["date"])
    sdf = sdf.dropna(subset=["splitAdjClose"]).sort_values("date")
    tol = pd.Timedelta(days=7)

    at_reb = pd.merge_asof(
        m[["date", "symbol"]].sort_values("date"), sdf,
        on="date", by="symbol", direction="backward", tolerance=tol,
    ).rename(columns={"splitAdjClose": "px_now"})

    fe = (m[["symbol", "period_end"]].dropna(subset=["period_end"]).drop_duplicates()
          .rename(columns={"period_end": "date"}).sort_values("date"))
    at_fil = pd.merge_asof(
        fe, sdf, on="date", by="symbol", direction="backward", tolerance=tol,
    ).rename(columns={"date": "period_end", "splitAdjClose": "px_filing"})

    m = m.merge(at_reb[["date", "symbol", "px_now"]], on=["date", "symbol"], how="left")
    m = m.merge(at_fil[["symbol", "period_end", "px_filing"]], on=["symbol", "period_end"], how="left")
    m["price_ratio"] = (
        (m["px_filing"] / m["px_now"]).replace([np.inf, -np.inf], np.nan).clip(lower=0.1, upper=10.0)
    )
    return m


def _rating_factors(rebalance: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time analyst recommendation-revision factors from the dated grade log.

    For each rebalance date, per symbol: upgrades minus downgrades over a trailing
    3m and 6m window, plus a 12-month breadth ratio (net / total actions) - using
    only actions on or before that date (look-ahead-free). Coverage begins ~2012,
    so values are NaN before a name's first logged action. Higher = more positive
    revisions = better.
    """
    path = settings.processed_dir / "grades_long.parquet"
    if not path.exists():
        logger.warning("grades_long.parquet missing - sentiment factors will be NaN")
        return pd.DataFrame(columns=["date", "symbol", "rating_rev_3m", "rating_rev_6m", "rating_breadth_12m"])
    g = read_parquet(path)
    g["date"] = pd.to_datetime(g["date"])
    act = g["action"].astype(str).str.lower()
    g["up"], g["down"] = (act == "upgrade").astype(int), (act == "downgrade").astype(int)
    g = g.sort_values(["symbol", "date"])

    reb = pd.DatetimeIndex(sorted(rebalance))
    rn = reb.values.astype("datetime64[ns]")
    r3 = (reb - pd.DateOffset(months=3)).values.astype("datetime64[ns]")
    r6 = (reb - pd.DateOffset(months=6)).values.astype("datetime64[ns]")
    r12 = (reb - pd.DateOffset(months=12)).values.astype("datetime64[ns]")

    out = []
    for sym, gg in g.groupby("symbol", sort=False):
        d = gg["date"].values.astype("datetime64[ns]")
        cu, cd = np.cumsum(gg["up"].values), np.cumsum(gg["down"].values)

        def cum_at(times, arr):
            idx = np.searchsorted(d, times, side="right")
            return np.where(idx > 0, arr[np.clip(idx - 1, 0, len(arr) - 1)], 0)

        cut, cdt = cum_at(rn, cu), cum_at(rn, cd)
        u3, dn3 = cut - cum_at(r3, cu), cdt - cum_at(r3, cd)
        u6, dn6 = cut - cum_at(r6, cu), cdt - cum_at(r6, cd)
        u12, dn12 = cut - cum_at(r12, cu), cdt - cum_at(r12, cd)
        tot12 = u12 + dn12
        breadth = np.where(tot12 > 0, (u12 - dn12) / np.where(tot12 > 0, tot12, 1), np.nan)
        cov = rn >= d[0]  # covered only from a name's first logged action
        out.append(pd.DataFrame({
            "date": reb, "symbol": sym,
            "rating_rev_3m": np.where(cov, u3 - dn3, np.nan),
            "rating_rev_6m": np.where(cov, u6 - dn6, np.nan),
            "rating_breadth_12m": np.where(cov, breadth, np.nan),
        }))
    return pd.concat(out, ignore_index=True)


def build_factor_panel() -> pd.DataFrame:
    """Construct the full factor panel: component ranks + family composites.

    Returns the 2010+ investable panel carrying each winsorised component rank
    (``<comp>_rk``), the six family composites + ``size`` (re-ranked to a clean
    [0,1] within date), plus labels and forward returns. ``build`` writes the
    family-level slice to ``factors.parquet``; ``qfr.validation.factor_screen``
    reuses the component ranks for the granular factor screen.
    """
    m = read_parquet(settings.processed_dir / "master_clean.parquet")
    rebalance = month_end_dates("2000-01-31", "2026-04-30")

    # Merge price-based factors (computed over full history so momentum has lookback).
    m = m.merge(_price_factors(rebalance), on=["date", "symbol"], how="left")
    # Merge analyst recommendation-revision (sentiment) factors (PIT, ~2012+).
    m = m.merge(_rating_factors(rebalance), on=["date", "symbol"], how="left")

    # Refresh the stale period-end price baked into FMP's value ratios to the live
    # rebalance-date price (see _refresh_value_prices). r = price(filing)/price(now):
    # a yield = fundamental/price scales by r; market cap scales by 1/r.
    m = _refresh_value_prices(m)
    r = m["price_ratio"].fillna(1.0)  # no live price -> keep the original stale ratio
    mktcap_fresh = m["marketCap"] / r
    net_debt = m["enterpriseValue"] - m["marketCap"]  # balance-sheet, price-insensitive
    ev_fresh = mktcap_fresh + net_debt

    # Derived value yields + oriented helpers (all higher = better), live price.
    m["bookToMarket"] = (1.0 / m["priceToBookRatio"].replace(0, np.nan)) * r
    m["salesYield"] = (1.0 / m["priceToSalesRatio"].replace(0, np.nan)) * r
    m["earningsYield"] = m["earningsYield"] * r
    m["freeCashFlowYield"] = m["freeCashFlowYield"] * r
    m["ebitdaToEV"] = (1.0 / m["evToEBITDA"].replace(0, np.nan)) * (m["enterpriseValue"] / ev_fresh)
    m["lowLeverage"] = -m["debtToEquityRatio"]
    m["lowVol"] = -m["vol_12m"]
    m["size_raw"] = -np.log(mktcap_fresh.replace(0, np.nan))  # small-cap tilt (fresh mkt cap)
    m["marketCap"] = mktcap_fresh  # carry the live market cap downstream (Part 7 weights)
    for c in ["bookToMarket", "salesYield", "ebitdaToEV", "earningsYield", "freeCashFlowYield"]:
        m[c] = m[c].replace([np.inf, -np.inf], np.nan)

    _cov = m.loc[m["date"] >= PRIMARY_START, "price_ratio"]
    logger.info(f"value-price refresh: {_cov.notna().mean():.1%} of 2010+ rows got a live price; "
                f"price_ratio median={_cov.median():.3f}, IQR=[{_cov.quantile(.25):.3f}, {_cov.quantile(.75):.3f}]")

    panel = m[m["investable"] & (m["date"] >= PRIMARY_START)].copy()
    logger.info(
        f"factor panel: {len(panel):,} rows, "
        f"{panel['date'].min().date()}..{panel['date'].max().date()}, "
        f"{panel['date'].nunique()} months"
    )

    comps = sorted({c for cols in FAMILIES.values() for c in cols})
    panel = cs_winsorize(panel, comps)              # tame outliers within date
    ranks = cs_rank(panel, comps)                   # -> [0,1] percentile within date
    for c in comps:
        panel[c + "_rk"] = ranks[c]

    fam_cols = list(FAMILIES) + ["size"]
    for fam, cols in FAMILIES.items():
        panel[fam] = panel[[c + "_rk" for c in cols]].mean(axis=1)  # skips NaN components
    panel["size_raw_rk"] = cs_rank(cs_winsorize(panel, ["size_raw"]), ["size_raw"])["size_raw"]
    panel["size"] = panel["size_raw_rk"]
    # Short-term reversal: standalone factor (not in any family composite).
    panel["st_rev_rk"] = cs_rank(cs_winsorize(panel, ["st_rev"]), ["st_rev"])["st_rev"]
    panel["reversal"] = panel["st_rev_rk"]

    # Re-rank composites to a clean uniform [0,1] within each date.
    rer = cs_rank(panel, fam_cols)
    for f in fam_cols:
        panel[f] = rer[f]
    return panel


def build() -> pd.DataFrame:
    panel = build_factor_panel()
    fam_cols = list(FAMILIES) + ["size"]
    keep = ["date", "symbol", "sector", "marketCap", "adjClose",
            "ret_fwd_1m", "ret_fwd_3m", "ret_fwd_6m", *fam_cols, "reversal",
            "mom_12_1", "vol_12m"]
    out = panel[keep].sort_values(["date", "symbol"]).reset_index(drop=True)
    write_parquet(out, settings.processed_dir / "factors.parquet")

    logger.info(f"factors.parquet: {len(out):,} rows | families = {fam_cols}")
    logger.info("composite coverage (non-null %):\n"
                + (out[fam_cols].notna().mean() * 100).round(1).to_string())
    logger.info("family rank correlations:\n" + out[fam_cols].corr().round(2).to_string())
    return out


def make_figures() -> None:
    """Part 3 figures -> charts/03_*.png."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    from qfr.utils.viz import PALETTE, save_fig, set_plot_style

    set_plot_style()
    f = read_parquet(settings.processed_dir / "factors.parquet")
    fams = list(FAMILIES) + ["size"]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(f[fams].corr(), annot=True, fmt=".2f", cmap="vlag", center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={"label": "rank corr"}, ax=ax)
    ax.set_title("Factor family correlations (2010+)")
    save_fig(fig, "03_factor_correlation")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, fam in zip(axes.ravel(), fams):
        d = f.dropna(subset=[fam, "ret_fwd_1m"]).copy()
        d["q"] = d.groupby("date")[fam].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        )
        sp = d.groupby("q", observed=True)["ret_fwd_1m"].mean() * 100
        ax.bar([str(i) for i in sp.index], sp.values, color=PALETTE["primary"])
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_title(f"{fam}")
        ax.set_ylabel("mean fwd 1m ret (%)")
        ax.set_xlabel("quintile (5 = high score)")
    fig.suptitle("Preview: mean forward 1-month return by factor quintile (full IC analysis in Part 4)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "03_factor_quintile_spread")

    cov = f.groupby("date").size()
    fig, ax = plt.subplots()
    ax.plot(cov.index, cov.values, color=PALETTE["primary"], lw=1.5)
    ax.set_ylim(0, 520)
    ax.set_title("Investable names with full factor scores per month (2010+)")
    ax.set_ylabel("number of names")
    save_fig(fig, "03_factor_coverage")


def main() -> None:
    build()
    make_figures()


if __name__ == "__main__":
    main()
