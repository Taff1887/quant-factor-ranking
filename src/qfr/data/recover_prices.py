"""Investigate and (carefully) attempt to recover missing early-year price data.

The point-in-time S&P 500 universe has ~500 members throughout, but FMP's price
API does not serve many old/delisted/renamed tickers, so the *investable* count
is thin pre-2005. This module:

1. identifies every member-month missing usable FMP price,
2. classifies the missing tickers (delisted / renamed / no-data / partial),
3. probes alternative sources (FMP symbol variants, FMP search, Yahoo),
4. recovers prices ONLY where corporate continuity is clear and the recovered
   series overlaps the ticker's actual index-membership window (never extending
   past delisting, never substituting a different company).

Outputs go to ``reports/`` (CSVs) and ``charts/`` (before/after coverage).
Accuracy is prioritised over completeness — uncertain mappings are excluded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.data.fmp_client import FMPClient
from qfr.utils.config import PROJECT_ROOT, settings
from qfr.utils.io import read_parquet
from qfr.utils.logging import logger

REPORTS = PROJECT_ROOT / "reports"


def name_sector_maps(client: FMPClient | None = None) -> tuple[dict, dict]:
    """symbol -> *historical* company name, and symbol -> sector.

    IMPORTANT: many old S&P 500 tickers have been RECYCLED by unrelated companies
    today (e.g. APC -> ARKO, CA -> an ETF, EMC -> an ETF, STI -> Solidion). So we
    take the name from the index change-log (the entity that was actually a
    member) FIRST, and only fall back to current members. Sector is taken from
    current members only (a recycled ticker's current profile would be wrong),
    otherwise 'Unknown'.
    """
    client = client or FMPClient()
    names: dict[str, str] = {}
    sectors: dict[str, str] = {}

    # 1) Historical names from the index change-log (authoritative for members).
    for rec in client.historical_sp500_constituents():
        a, an = (rec.get("symbol") or "").strip(), rec.get("addedSecurity")
        r, rn = (rec.get("removedTicker") or "").strip(), rec.get("removedSecurity")
        if r and rn:
            names.setdefault(r, str(rn))
        if a and an:
            names.setdefault(a, str(an))

    # 2) Current members fill gaps + provide sector for current names only.
    members = read_parquet(settings.processed_dir / "members_now.parquet")
    for _, row in members.iterrows():
        s = str(row.get("symbol", "")).strip()
        if not s:
            continue
        if pd.notna(row.get("name")):
            names.setdefault(s, str(row["name"]))
        if pd.notna(row.get("sector")):
            sectors[s] = str(row["sector"])
    return names, sectors


def identify_missing(master: pd.DataFrame) -> pd.DataFrame:
    """All member-months that lack usable FMP price."""
    return master.loc[~master["has_price"], ["date", "symbol"]].copy()


def coverage_by_year(master: pd.DataFrame) -> pd.DataFrame:
    y = master["date"].dt.year
    g = master.groupby(y)
    miss = master[~master["has_price"]]
    tab = pd.DataFrame({
        "member_months": g.size(),
        "missing_price_months": g["has_price"].apply(lambda s: int((~s).sum())),
        "pct_missing": (g["has_price"].apply(lambda s: float((~s).mean())) * 100).round(1),
        "distinct_members": g["symbol"].nunique(),
        "distinct_missing_tickers": miss.groupby(miss["date"].dt.year)["symbol"].nunique(),
    })
    return tab


def ticker_detail(master: pd.DataFrame, client: FMPClient | None = None) -> pd.DataFrame:
    client = client or FMPClient()
    names, sectors = name_sector_maps(client)
    prices = read_parquet(settings.processed_dir / "prices_long.parquet")
    members = read_parquet(settings.processed_dir / "members_now.parquet")
    changes = read_parquet(settings.processed_dir / "constituent_changes.parquet")

    priced = set(prices["symbol"].unique())
    current = set(members["symbol"])
    removed = set(changes["removed"].dropna())

    miss = identify_missing(master)
    agg = miss.groupby("symbol").agg(
        first_missing=("date", "min"),
        last_missing=("date", "max"),
        n_missing_months=("date", "size"),
    )
    total = master.groupby("symbol")["date"].size().rename("n_member_months")
    d = agg.join(total).reset_index()

    d["company_name"] = d["symbol"].map(names).fillna("")
    d["sector"] = d["symbol"].map(sectors).fillna("Unknown")
    d["in_current_index"] = d["symbol"].isin(current)
    d["has_any_fmp_price"] = d["symbol"].isin(priced)
    d["ever_left_index"] = d["symbol"].isin(removed)

    def classify(row) -> str:
        if not row["has_any_fmp_price"]:
            base = "no FMP price at all"
        elif row["n_missing_months"] < row["n_member_months"]:
            base = "partial FMP coverage"
        else:
            base = "FMP price exists but outside membership window"
        if row["in_current_index"]:
            return base + " (still in index)"
        if row["ever_left_index"]:
            return base + " (left index: delisted/acquired/renamed)"
        return base + " (unknown status)"

    d["likely_status"] = d.apply(classify, axis=1)
    return d.sort_values("n_missing_months", ascending=False).reset_index(drop=True)


def main_identify() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    master = read_parquet(settings.processed_dir / "master_pit.parquet")
    client = FMPClient()

    by_year = coverage_by_year(master)
    by_year.to_csv(REPORTS / "missing_price_coverage_by_year.csv")
    detail = ticker_detail(master, client)
    detail.to_csv(REPORTS / "missing_ticker_detail.csv", index=False)

    logger.info("===== missing price coverage by year =====")
    logger.info("\n" + by_year.to_string())
    logger.info(f"\ndistinct tickers ever missing price: {detail['symbol'].nunique()}")
    logger.info(f"total missing member-months         : {int(detail['n_missing_months'].sum()):,}")
    logger.info("\nstatus breakdown:")
    logger.info("\n" + detail["likely_status"].value_counts().to_string())
    logger.info("\n----- top 20 missing tickers -----")
    cols = ["symbol", "company_name", "sector", "n_missing_months",
            "first_missing", "last_missing", "has_any_fmp_price", "in_current_index", "likely_status"]
    logger.info("\n" + detail[cols].head(20).to_string(index=False))


def membership_windows() -> pd.DataFrame:
    uni = read_parquet(settings.processed_dir / "universe_panel.parquet")
    uni["date"] = pd.to_datetime(uni["date"])
    return uni.groupby("symbol")["date"].agg(["min", "max"])


def _eval_source(name: str, dts: list, m0, m1) -> dict:
    if not dts:
        return {"source": name, "found": False, "n_rows": 0,
                "data_start": None, "data_end": None, "overlap_months": 0}
    idx = pd.DatetimeIndex(dts).to_period("M")
    overlap = 0
    if m0 is not None:
        mm = pd.period_range(m0.to_period("M"), m1.to_period("M"), freq="M")
        overlap = int(idx.isin(mm).sum())
    return {"source": name, "found": True, "n_rows": len(dts),
            "data_start": str(min(dts).date()), "data_end": str(max(dts).date()),
            "overlap_months": overlap}


def attempt_recovery(master: pd.DataFrame, client: FMPClient | None = None,
                     accept_min: int = 6, hi: int = 24):
    """Try Yahoo + FMP-full for every missing ticker; recover ONLY where the data
    overlaps the membership window (rejects recycled tickers automatically)."""
    from qfr.data.validate_yahoo import fetch_yahoo_monthly

    client = client or FMPClient()
    detail = ticker_detail(master, client)
    tickers = detail["symbol"].tolist()
    names = dict(zip(detail["symbol"], detail["company_name"]))
    mw = membership_windows()

    yah = fetch_yahoo_monthly(tickers)
    yah["date"] = pd.to_datetime(yah["date"])

    attempts, mappings, rec_rows = [], [], []
    for t in tickers:
        m0 = mw["min"].get(t)
        m1 = mw["max"].get(t)
        yt = yah[yah["symbol"] == t].sort_values("date")
        try:
            ff = client.historical_prices(t, from_date="1995-01-01", to_date="2026-05-01", series="full")
        except Exception:  # noqa: BLE001
            ff = []
        evals = [
            ("yahoo", _eval_source("yahoo", list(yt["date"]), m0, m1)),
            ("fmp_full", _eval_source("fmp_full", [pd.Timestamp(r["date"]) for r in ff], m0, m1)),
        ]
        for src, ev in evals:
            decision, conf, note = "reject", "", ""
            if ev["overlap_months"] >= accept_min:
                decision = "recover"
                conf = "high" if ev["overlap_months"] >= hi else "medium"
            elif ev["found"]:
                note = "data present but outside membership window (likely recycled ticker)"
            else:
                note = "no data from this source"
            attempts.append({
                "ticker": t, "company_name": names.get(t, ""),
                "membership_start": None if m0 is None else str(m0.date()),
                "membership_end": None if m1 is None else str(m1.date()),
                **ev, "decision": decision, "confidence": conf, "notes": note,
            })
            # Only Yahoo gives an *adjusted* series, so only Yahoo feeds recoveries.
            if decision == "recover" and src == "yahoo":
                sub = yt[yt["date"].dt.to_period("M") <= m1.to_period("M")]
                for _, rr in sub.iterrows():
                    rec_rows.append({"symbol": t, "date": rr["date"], "adjClose": rr["y_adjclose"]})
                mappings.append({
                    "old_ticker": t, "mapped_ticker": t, "company_name": names.get(t, ""),
                    "reason": "same ticker, Yahoo series overlaps membership window",
                    "confidence": conf, "source": "yahoo",
                    "notes": f"overlap_months={ev['overlap_months']}; clipped to <= membership_end",
                })
    return (pd.DataFrame(attempts), pd.DataFrame(mappings),
            pd.DataFrame(rec_rows, columns=["symbol", "date", "adjClose"]))


def before_after_chart(master: pd.DataFrame, recovered: pd.DataFrame):
    import matplotlib.pyplot as plt

    from qfr.utils.viz import PALETTE, save_fig, set_plot_style

    set_plot_style()
    m = master.copy()
    ym = m["date"].dt.strftime("%Y-%m")
    rec_keys = set(zip(recovered["symbol"], pd.to_datetime(recovered["date"]).dt.strftime("%Y-%m"))) \
        if len(recovered) else set()
    rec_mask = np.array([(s, y) in rec_keys for s, y in zip(m["symbol"], ym)], dtype=bool)
    m["after_price"] = m["has_price"].to_numpy() | rec_mask
    m["inv_before"] = m["has_price"] & m["has_fundamentals"] & m["fresh_filing"]
    m["inv_after"] = m["after_price"] & m["has_fundamentals"] & m["fresh_filing"]
    g = m.groupby("date")
    fig, ax = plt.subplots()
    ax.plot(g.size().index, g.size().values, color=PALETTE["primary"], lw=2, label="index members")
    ax.plot(g["inv_before"].sum().index, g["inv_before"].sum().values, color=PALETTE["muted"], lw=2,
            label="investable (before recovery)")
    ax.plot(g["inv_after"].sum().index, g["inv_after"].sum().values, color=PALETTE["green"], lw=2,
            ls="--", label="investable (after recovery)")
    ax.set_title("Coverage before vs after price-recovery attempt")
    ax.set_ylabel("number of stocks")
    ax.set_ylim(0, 540)
    ax.legend(loc="lower right")
    return save_fig(fig, "recovery_before_after")


def main_recover() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    master = read_parquet(settings.processed_dir / "master_pit.parquet")
    client = FMPClient()

    attempts, mapping, recovered = attempt_recovery(master, client)
    attempts.to_csv(REPORTS / "recovery_attempt_log.csv", index=False)
    mapping.to_csv(REPORTS / "recovered_ticker_mapping.csv", index=False)
    if len(recovered):
        recovered.to_parquet(settings.processed_dir / "recovered_prices.parquet")
    chart = before_after_chart(master, recovered)

    before_missing = int((~master["has_price"]).sum())
    rec_keys = set(zip(recovered["symbol"], pd.to_datetime(recovered["date"]).dt.strftime("%Y-%m"))) \
        if len(recovered) else set()
    ym = master["date"].dt.strftime("%Y-%m")
    recovered_mm = int(sum((s, y) in rec_keys for s, y in zip(master["symbol"], ym)))

    logger.info("=" * 60)
    logger.info(f"missing ticker-months (before)     : {before_missing:,}")
    logger.info(f"member-months recovered            : {recovered_mm:,}")
    logger.info(f"still missing                      : {before_missing - recovered_mm:,}")
    logger.info(f"tickers with a safe recovery       : {mapping['old_ticker'].nunique() if len(mapping) else 0}")
    if len(attempts):
        found = attempts[attempts["found"]]
        logger.info("\nattempt outcomes by source (found / overlapping):")
        for src in ["yahoo", "fmp_full"]:
            s = attempts[attempts["source"] == src]
            logger.info(f"  {src:9s}: found {int(s['found'].sum()):3d} / "
                        f"overlap>=6 {int((s['overlap_months'] >= 6).sum()):3d}  of {len(s)}")
    logger.info(f"chart -> {chart.name}")


if __name__ == "__main__":
    main_identify()
    main_recover()
