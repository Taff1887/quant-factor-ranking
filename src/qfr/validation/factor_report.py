"""Per-factor analysis: each factor gets its own folder under its family.

For every individual factor (charts/factors/<family>/<factor>/):

    chart1_rank_ic.png            Monthly rank IC (bars) + 12m average (line) +
                                   t-stat(IC) annotation.
    chart3_ic_decay.png           Average IC at lags 1-12 months (bars) + success
                                   rate per lag (line, right axis).
    chart5_deciles.png            10 equal-weight deciles, cumulative growth of
                                   $1 vs the equal-weight universe (D1 = best).
    chart5_quintiles.png          5 equal-weight quintiles (Q1 = best), same.
    table1_quintile_stats.png     Quintile stats table (Quintile 1..5 + Q1-Q5
                                   spread + Market): Total return, Active return,
                                   Tracking error, Information ratio, t-stat(IR),
                                   Monthly success rate, Turnover, Volatility,
                                   Sharpe, CAPM beta/alpha.
    chart7_pure_factor_index.png  Cumulative index (base 100) of the 1-SD pure
                                   factor return (with size + sector + book-to-
                                   price stripped), annotated with annualised
                                   pure return, tracking error, information
                                   ratio, monthly success and t-stat(IR).
    chart8_raw_factor_index.png   Same, raw factor return (univariate).
    chart9_pure_factor_returns.png   Monthly pure factor returns (bars) + 12m
                                      rolling average (line).
    chart10_raw_factor_returns.png   Monthly raw factor returns (bars) + 12m
                                      rolling average (line).

A combined summary CSV (one row per factor) is written to
reports/factor_report_summary.csv, separate from the per-factor folders.

Run::  uv run python -m qfr.validation.factor_report
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from qfr.factors.build import FAMILIES, build_factor_panel
from qfr.utils.config import PROJECT_ROOT, settings
from qfr.utils.logging import logger

ANN = 12
MIN_NAMES = 20
MAX_LAG = 12
UNIVERSE = "S&P 500"

LABELS = {
    "earningsYield": "Earnings yield", "freeCashFlowYield": "FCF yield",
    "bookToMarket": "Book-to-market", "salesYield": "Sales yield", "ebitdaToEV": "EBITDA/EV",
    "returnOnEquity": "Return on equity", "returnOnInvestedCapital": "ROIC",
    "returnOnAssets": "Return on assets", "grossProfitMargin": "Gross margin",
    "operatingProfitMargin": "Operating margin", "netProfitMargin": "Net margin",
    "interestCoverageRatio": "Interest coverage", "lowLeverage": "Low leverage",
    "mom_12_1": "12-1m momentum", "mom_6_1": "6-1m momentum", "mom_3_1": "3-1m momentum",
    "revenueGrowth": "Revenue growth", "epsgrowth": "EPS growth",
    "netIncomeGrowth": "Net-income growth", "ebitdaGrowth": "EBITDA growth",
    "lowVol": "Low volatility", "size_raw": "Small size", "st_rev": "Short-term reversal",
    "rating_rev_3m": "Rating revision 3m", "rating_rev_6m": "Rating revision 6m",
    "rating_breadth_12m": "Rating breadth 12m",
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def factor_list() -> list[tuple[str, str, str, str]]:
    """[(family, label, slug, rank_col)] for every individual factor (deduped)."""
    items, seen = [], set()
    for fam, comps in FAMILIES.items():
        for c in comps:
            col = c + "_rk"
            if col in seen:
                continue
            seen.add(col)
            label = LABELS.get(c, c)
            items.append((fam, label, _slug(label), col))
    for fam, c in [("size", "size_raw"), ("reversal", "st_rev")]:
        col = c + "_rk"
        if col in seen:
            continue
        seen.add(col)
        label = LABELS.get(c, c)
        items.append((fam, label, _slug(label), col))
    return items


# --------------------------------------------------------------------------
# 1. Rank IC (monthly + rolling)
# --------------------------------------------------------------------------
def ic_monthly(panel: pd.DataFrame, col: str, ret: str = "ret_fwd_1m") -> pd.Series:
    out = {}
    for d, g in panel.groupby("date"):
        sub = g[[col, ret]].dropna()
        if len(sub) >= MIN_NAMES:
            ic = sub[col].corr(sub[ret], method="spearman")
            if pd.notna(ic):
                out[d] = ic
    return pd.Series(out).sort_index()


def _ic_ts_stats(s: pd.Series) -> dict:
    sd = s.std()
    return {"mean_ic": s.mean(), "ic_ir": (s.mean() / sd) if sd else np.nan,
            "t_stat": (s.mean() / sd * np.sqrt(len(s))) if sd else np.nan,
            "success": float((s > 0).mean()), "n": len(s)}


# --------------------------------------------------------------------------
# 2. IC decay (lags 1..12)
# --------------------------------------------------------------------------
def ic_decay(panel: pd.DataFrame, col: str, max_lag: int = MAX_LAG) -> pd.DataFrame:
    p = panel.sort_values(["symbol", "date"])
    rows = []
    for n in range(1, max_lag + 1):
        rn = p.groupby("symbol")["ret_fwd_1m"].shift(-(n - 1))
        tmp = pd.DataFrame({"date": p["date"].values, "f": p[col].values, "r": rn.values})
        ics = {}
        for d, g in tmp.groupby("date"):
            sub = g[["f", "r"]].dropna()
            if len(sub) >= MIN_NAMES:
                ic = sub["f"].corr(sub["r"], method="spearman")
                if pd.notna(ic):
                    ics[d] = ic
        s = pd.Series(ics)
        sd = s.std()
        rows.append({"lag": n, "avg_ic": s.mean(), "success": float((s > 0).mean()),
                     "t_stat": (s.mean() / sd * np.sqrt(len(s))) if sd else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Fractiles
# --------------------------------------------------------------------------
def _turnover(members: pd.DataFrame) -> float:
    if members.empty:
        return np.nan
    w = members.assign(w=1.0).pivot_table(index="date", columns="symbol", values="w", fill_value=0.0).sort_index()
    w = w.div(w.sum(axis=1), axis=0)
    return float((0.5 * w.diff().abs().sum(axis=1)).iloc[1:].mean())


def _capm(r: pd.Series, mkt: pd.Series) -> tuple[float, float]:
    df = pd.concat([r, mkt], axis=1).dropna()
    if len(df) < 12:
        return np.nan, np.nan
    x, y = df.iloc[:, 1].values, df.iloc[:, 0].values
    beta = np.cov(x, y)[0, 1] / np.var(x)
    alpha = (y.mean() - beta * x.mean()) * ANN
    return beta, alpha


def fractiles(panel: pd.DataFrame, col: str, ret: str = "ret_fwd_1m", n: int = 10,
              weighted: bool = False):
    """Cumulative monthly returns + stats per fractile. n=10 -> D1..D10, n=5 -> Q1..Q5.

    weighted=False -> equal-weight within fractile (each name 1/N).
    weighted=True  -> market-cap-weight within fractile (so the result reflects
    the larger-cap names; comparison vs equal-weight reveals if the factor
    works better in small caps or large caps).
    """
    needed = [col, ret] + (["marketCap"] if weighted else [])
    d = panel.dropna(subset=needed).copy()
    q = d.groupby("date")[col].transform(lambda s: pd.qcut(s.rank(method="first"), n, labels=False))
    d["B"] = (n - q).astype(int)  # B1 = top bucket (best factor score)
    prefix = "Q" if n == 5 else "D"

    if weighted:
        d["wret"] = d["marketCap"] * d[ret]
        qg = d.groupby(["date", "B"])
        qret = (qg["wret"].sum() / qg["marketCap"].sum()).unstack("B").sort_index()
        mg = d.groupby("date")
        mkt = (mg["wret"].sum() / mg["marketCap"].sum()).sort_index()
    else:
        qret = d.groupby(["date", "B"])[ret].mean().unstack("B").sort_index()
        mkt = d.groupby("date")[ret].mean().sort_index()
    qret.columns = [f"{prefix}{c}" for c in qret.columns]
    mkt_ann = (1 + mkt).prod() ** (ANN / len(mkt)) - 1
    years = len(mkt) / ANN
    spread_name = f"{prefix}1-{prefix}{n}"

    def stat_block(r: pd.Series, vs_mkt: bool) -> dict:
        r = r.dropna()
        ann_tot = (1 + r).prod() ** (ANN / len(r)) - 1
        act = (r - mkt).dropna() if vs_mkt else r
        te = act.std() * np.sqrt(ANN)
        ir = (act.mean() / act.std() * np.sqrt(ANN)) if act.std() > 0 else np.nan
        beta, alpha = _capm(r, mkt) if vs_mkt else (np.nan, np.nan)
        return {"total_return": ann_tot,
                "active_return": (ann_tot - mkt_ann) if vs_mkt else ann_tot,
                "tracking_error": te, "info_ratio": ir,
                "t_stat": (ir * np.sqrt(years)) if pd.notna(ir) else np.nan,
                "monthly_success": float((act > 0).mean()),
                "volatility": r.std() * np.sqrt(ANN),
                "sharpe": (r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else np.nan,
                "capm_beta": beta, "capm_alpha": alpha}

    rows = {}
    for fc in qret.columns:
        rows[fc] = stat_block(qret[fc], vs_mkt=True)
        rows[fc]["turnover"] = _turnover(d[d["B"] == int(fc[1:])][["date", "symbol"]])
    spread = (qret[f"{prefix}1"] - qret[f"{prefix}{n}"]).dropna()
    rows[spread_name] = stat_block(spread, vs_mkt=False)
    rows[spread_name]["active_return"] = rows[spread_name]["total_return"]
    rows[spread_name]["turnover"] = np.nan
    rows["Market"] = stat_block(mkt, vs_mkt=False)
    rows["Market"]["active_return"] = 0.0
    rows["Market"]["turnover"] = np.nan
    tbl = pd.DataFrame(rows).T
    return qret, mkt, tbl


# --------------------------------------------------------------------------
# 4. Pure factor returns (Fama-MacBeth, controlling for size + sector + B/P)
# --------------------------------------------------------------------------
def _z(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else np.zeros_like(x)


def pure_factor_return(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    drop_size = col == "size_raw_rk"
    drop_btp = col == "bookToMarket_rk"
    need = [col, "ret_fwd_1m", "marketCap", "sector"] + ([] if drop_btp else ["bookToMarket"])
    rows = []
    for d, g in panel.groupby("date"):
        sub = g.dropna(subset=need)
        if len(sub) < 30:
            continue
        y = sub["ret_fwd_1m"].to_numpy(float)
        fz = _z(sub[col].to_numpy(float))
        ctrl = []
        if not drop_size:
            ctrl.append(_z(np.log(sub["marketCap"].clip(lower=1).to_numpy(float))))
        if not drop_btp:
            ctrl.append(_z(sub["bookToMarket"].to_numpy(float)))
        sec = pd.get_dummies(sub["sector"], drop_first=True).to_numpy(float)
        parts = [np.ones(len(y)), fz, *ctrl] + ([sec] if sec.size else [])
        X = np.column_stack(parts)
        try:
            pure = np.linalg.lstsq(X, y, rcond=None)[0][1]
            raw = np.linalg.lstsq(np.column_stack([np.ones(len(y)), fz]), y, rcond=None)[0][1]
            rows.append((d, pure, raw))
        except np.linalg.LinAlgError:
            continue
    return pd.DataFrame(rows, columns=["date", "pure", "raw"]).set_index("date").sort_index()


def _series_stats(s: pd.Series) -> dict:
    s = s.dropna()
    sd = s.std()
    return {"ann_return": s.mean() * ANN, "tracking_error": sd * np.sqrt(ANN),
            "info_ratio": (s.mean() / sd * np.sqrt(ANN)) if sd else np.nan,
            "t_stat": (s.mean() / sd * np.sqrt(len(s))) if sd else np.nan,
            "success": float((s > 0).mean())}


# --------------------------------------------------------------------------
# Chart renderers
# --------------------------------------------------------------------------
def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _setup_style() -> None:
    from qfr.utils.viz import set_plot_style
    set_plot_style()


# Chart 1
def chart1_rank_ic(ic: pd.Series, label: str, outdir: Path) -> None:
    import matplotlib.pyplot as plt
    stats = _ic_ts_stats(ic)
    roll = ic.rolling(12, min_periods=6).mean()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(ic.index, ic.values * 100, width=22, color="#bcbcbc", alpha=0.9, label="Rank IC (monthly)")
    ax.plot(roll.index, roll.values * 100, color="#1f3a5f", lw=2.0, label="12m average")
    ax.axhline(0, color="#444", lw=0.7)
    ax.set_ylabel("Rank IC (%)")
    ax.set_title(f"Chart 1: {label} for {UNIVERSE} (Rank ICs)",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.text(0.985, 0.96,
            f"mean IC = {stats['mean_ic'] * 100:.2f}%\nt-stat(IC) = {stats['t_stat']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.92))
    ax.legend(fontsize=9, loc="lower left")
    _save(fig, outdir / "chart1_rank_ic.png")


# Chart 3
def chart3_ic_decay(dec: pd.DataFrame, label: str, outdir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(dec["lag"], dec["avg_ic"] * 100, color="#1f3a5f", alpha=0.85, label="Avg IC")
    ax.axhline(0, color="#444", lw=0.7)
    ax.set_xticks(list(dec["lag"]))
    ax.set_xlabel("lag (months ahead)")
    ax.set_ylabel("Avg rank IC (%)")
    axt = ax.twinx()
    axt.plot(dec["lag"], dec["success"] * 100, color="#0a7a3a", marker="o", lw=1.7, label="Success rate")
    axt.set_ylabel("Success rate (%)", color="#0a7a3a")
    axt.set_ylim(20, 80)
    axt.axhline(50, color="#0a7a3a", lw=0.6, ls=":")
    ax.set_title(f"Chart 3: {label} for {UNIVERSE} (IC decay profile)",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    axt.legend(loc="upper right", fontsize=9)
    _save(fig, outdir / "chart3_ic_decay.png")


# Chart 5 (deciles or quintiles)
def chart5_fractiles(qret: pd.DataFrame, mkt: pd.Series, label: str, n: int, outdir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    name = "Deciles" if n == 10 else "Quintiles"
    fname = "chart5_deciles.png" if n == 10 else "chart5_quintiles.png"
    fig, ax = plt.subplots(figsize=(11, 5.8))
    cmap = mpl.colormaps["RdYlGn"](np.linspace(0.88, 0.12, len(qret.columns)))
    for i, fc in enumerate(qret.columns):
        ax.plot(qret.index, (1 + qret[fc].fillna(0)).cumprod().values,
                lw=1.3, color=cmap[i], label=fc)
    ax.plot(mkt.index, (1 + mkt.fillna(0)).cumprod().values, color="#222", lw=1.9, ls="--", label="Market")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log)")
    ax.set_title(f"Chart 5: {label} for {UNIVERSE} ({name})",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, ncol=(4 if n == 10 else 3), loc="upper left")
    _save(fig, outdir / fname)


# Table 1 (quintile stats rendered as image)
TABLE_ROWS = [
    ("Total return", "total_return", "pct"),
    ("Active return", "active_return", "pct"),
    ("Tracking error", "tracking_error", "pct"),
    ("Information ratio", "info_ratio", "num"),
    ("t-stat (IR)", "t_stat", "num"),
    ("Monthly success rate", "monthly_success", "pct"),
    ("Turnover", "turnover", "pct"),
    ("Volatility", "volatility", "pct"),
    ("Sharpe ratio", "sharpe", "num"),
    ("CAPM Beta (vs benchmark)", "capm_beta", "num"),
    ("CAPM Alpha", "capm_alpha", "pct"),
]


def _fmt(v: float, kind: str) -> str:
    if pd.isna(v):
        return "—"
    return f"{v * 100:.2f}%" if kind == "pct" else f"{v:.2f}"


def _render_quintile_table(tbl: pd.DataFrame, title: str, outpath: Path) -> None:
    """Render one quintile-stats table as a PNG."""
    import matplotlib.pyplot as plt
    cols = [f"Quintile {i}" for i in range(1, 6)] + ["Q1 − Q5\n(long − short)", "Market"]
    src_cols = [f"Q{i}" for i in range(1, 6)] + ["Q1-Q5", "Market"]

    cell_text = []
    for _, key, kind in TABLE_ROWS:
        row = []
        for sc in src_cols:
            v = tbl.loc[sc, key] if sc in tbl.index else np.nan
            row.append(_fmt(v, kind))
        cell_text.append(row)
    row_labels = [r[0] for r in TABLE_ROWS]

    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=18)
    tbl_obj = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=cols,
                       cellLoc="center", loc="center",
                       colWidths=[0.085] * 5 + [0.11, 0.085])
    tbl_obj.auto_set_font_size(False)
    tbl_obj.set_fontsize(9)
    tbl_obj.scale(1, 1.55)
    ncol = len(cols)
    for j in range(ncol):
        c = tbl_obj[0, j]
        c.set_facecolor("#1f3a5f")
        c.set_text_props(color="white", fontweight="bold")
    nrow = len(row_labels)
    for i in range(nrow):
        rl = tbl_obj[i + 1, -1]
        rl.set_text_props(ha="right", fontweight="bold")
        rl.set_facecolor("#e9ecf2")
    fig.text(0.02, 0.02,
             "Q1 = top quintile (best factor exposure, held long). Q5 = bottom quintile. "
             "Q1−Q5 = long Q1 / short Q5.",
             fontsize=8.5, style="italic", color="#555")
    _save(fig, outpath)


def table1_quintile_stats(panel: pd.DataFrame, col: str, label: str, outdir: Path):
    """Render two quintile stats tables: equal-weighted and market-cap weighted."""
    _, _, tbl_ew = fractiles(panel, col, n=5, weighted=False)
    _, _, tbl_cw = fractiles(panel, col, n=5, weighted=True)
    _render_quintile_table(
        tbl_ew, f"Table 1: {label} for {UNIVERSE}  (equal-weighted within fractile)",
        outdir / "table1_quintile_stats_equal_weighted.png",
    )
    _render_quintile_table(
        tbl_cw, f"Table 1: {label} for {UNIVERSE}  (market-cap weighted within fractile)",
        outdir / "table1_quintile_stats_cap_weighted.png",
    )
    return tbl_ew, tbl_cw


# Chart 7 / 8 (cumulative pure/raw factor return index)
def chart_cumulative_index(series: pd.Series, label: str, outdir: Path, *, pure: bool) -> dict:
    import matplotlib.pyplot as plt
    s = series.dropna()
    cum = (1 + s).cumprod() * 100
    stats = _series_stats(s)
    chart_no = 7 if pure else 8
    kind = "pure" if pure else "raw"
    fname = f"chart{chart_no}_{kind}_factor_index.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(cum.index, cum.values, color="#1f3a5f" if pure else "#7f7f7f", lw=2.0)
    ax.axhline(100, color="#888", lw=0.7, ls="--")
    txt = (f"Annualised {kind} return : {stats['ann_return'] * 100:6.2f}%\n"
           f"Annualised tracking error: {stats['tracking_error'] * 100:6.2f}%\n"
           f"{kind.capitalize()} information ratio : {stats['info_ratio']:6.2f}\n"
           f"Monthly success rate     : {stats['success'] * 100:6.1f}%\n"
           f"t-stat (IR)              : {stats['t_stat']:6.2f}")
    ax.text(0.018, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.93))
    ax.set_ylabel("Index (base 100)")
    title = f"Chart {chart_no}: Index of {UNIVERSE} {kind} factor returns for {label}"
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
    _save(fig, outdir / fname)
    return stats


# Chart 9 / 10 (monthly + 12m rolling pure/raw factor return)
def chart_monthly_rolling(series: pd.Series, label: str, outdir: Path, *, pure: bool) -> None:
    import matplotlib.pyplot as plt
    s = series.dropna()
    roll = s.rolling(12, min_periods=6).mean()
    chart_no = 9 if pure else 10
    kind = "pure" if pure else "raw"
    fname = f"chart{chart_no}_{kind}_factor_returns.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(s.index, s.values * 100, width=22, color="#bcbcbc", alpha=0.9, label=f"Monthly {kind}")
    ax.plot(roll.index, roll.values * 100, color="#1f3a5f" if pure else "#5a5a5a",
            lw=2.0, label="12m average")
    ax.axhline(0, color="#444", lw=0.7)
    ax.set_ylabel(f"{kind.capitalize()} factor return (%)")
    title = f"Chart {chart_no}: {label} {kind} factor returns over time (monthly + 12m average)"
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, outdir / fname)


# --------------------------------------------------------------------------
# Per-factor driver
# --------------------------------------------------------------------------
def ic_2m_series(panel: pd.DataFrame, col: str) -> pd.Series:
    """Monthly Spearman IC of the factor vs the return realised TWO months ahead."""
    p = panel.sort_values(["symbol", "date"]).copy()
    p["r2"] = p.groupby("symbol")["ret_fwd_1m"].shift(-1)  # return over [t+1, t+2]
    out = {}
    for d, g in p.groupby("date"):
        sub = g[[col, "r2"]].dropna()
        if len(sub) >= MIN_NAMES:
            ic = sub[col].corr(sub["r2"], method="spearman")
            if pd.notna(ic):
                out[d] = ic
    return pd.Series(out).sort_index()


def factor_pack(panel: pd.DataFrame, family: str, label: str, slug: str, col: str) -> dict:
    outdir = settings.charts_dir / "factors" / family / slug
    ic = ic_monthly(panel, col)
    icst = _ic_ts_stats(ic)
    ic2 = ic_2m_series(panel, col)
    ic2st = _ic_ts_stats(ic2)
    dec = ic_decay(panel, col)
    qret10, mkt, _ = fractiles(panel, col, n=10)
    qret5, _, tbl5 = fractiles(panel, col, n=5)
    pfr = pure_factor_return(panel, col)

    chart1_rank_ic(ic, label, outdir)
    chart3_ic_decay(dec, label, outdir)
    chart5_fractiles(qret10, mkt, label, 10, outdir)
    chart5_fractiles(qret5, mkt, label, 5, outdir)
    table1_quintile_stats(panel, col, label, outdir)
    pure_stats = chart_cumulative_index(pfr["pure"], label, outdir, pure=True)
    raw_stats = chart_cumulative_index(pfr["raw"], label, outdir, pure=False)
    chart_monthly_rolling(pfr["pure"], label, outdir, pure=True)
    chart_monthly_rolling(pfr["raw"], label, outdir, pure=False)

    top = tbl5.loc["Q1"]
    ls = tbl5.loc["Q1-Q5"]
    return {
        "group": family, "factor": label, "slug": slug,
        "mean_ic_%": icst["mean_ic"] * 100, "ic_t": icst["t_stat"], "ic_hit_%": icst["success"] * 100,
        "mean_ic_2m_%": ic2st["mean_ic"] * 100, "ic_t_2m": ic2st["t_stat"],
        "Q1_active_%": top["active_return"] * 100, "Q1_IR": top["info_ratio"],
        "Q1Q5_ann_%": ls["total_return"] * 100, "Q1Q5_IR": ls["info_ratio"],
        "Q1_turnover_%": top["turnover"] * 100,
        "pure_ann_%": pure_stats["ann_return"] * 100, "pure_IR": pure_stats["info_ratio"], "pure_t": pure_stats["t_stat"],
        "raw_ann_%": raw_stats["ann_return"] * 100, "raw_IR": raw_stats["info_ratio"], "raw_t": raw_stats["t_stat"],
    }


# --------------------------------------------------------------------------
# Table 5: statistically significant factors
# --------------------------------------------------------------------------
def table5_significant_factors(rows: list[dict], threshold: float = 1.5, strict: float = 2.0) -> Path:
    """Render Table 5: factors with |t-stat| >= threshold on at least one of
    {IC 1m, IC 2m, pure factor return}. Cells where |t| >= strict are highlighted."""
    import matplotlib.pyplot as plt

    def max_t(r: dict) -> float:
        return max(abs(r["ic_t"]), abs(r["ic_t_2m"]), abs(r["pure_t"]))

    selected = [r for r in rows if max_t(r) >= threshold]
    selected.sort(key=max_t, reverse=True)
    n_strict = sum(1 for r in selected if max_t(r) >= strict)

    hdrs = ["Factor group", "Factor", "Avg IC (1m, %)", "t-stat (IC, 1m)",
            "t-stat (IC, 2m)", "t-stat (pure)"]
    cell_text = [
        [r["group"].capitalize(), r["factor"], f"{r['mean_ic_%']:.2f}%",
         f"{r['ic_t']:.2f}", f"{r['ic_t_2m']:.2f}", f"{r['pure_t']:.2f}"]
        for r in selected
    ]

    height = max(2.6, 0.42 * len(cell_text) + 1.6)
    fig, ax = plt.subplots(figsize=(13, height))
    ax.axis("off")
    title = (f"Table 5: Statistically significant factors  "
             f"(|t-stat| >= {threshold:.1f}; bold/green = strict |t-stat| >= {strict:.1f})")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=14)

    if not selected:
        ax.text(0.5, 0.5, "(no factors meet the t-stat criterion)", ha="center", va="center", fontsize=11)
        outpath = settings.charts_dir / "table5_significant_factors.png"
        _save(fig, outpath)
        return outpath

    tbl_obj = ax.table(cellText=cell_text, colLabels=hdrs, cellLoc="center", loc="center",
                       colWidths=[0.12, 0.30, 0.13, 0.13, 0.13, 0.12])
    tbl_obj.auto_set_font_size(False)
    tbl_obj.set_fontsize(9)
    tbl_obj.scale(1, 1.55)
    for j in range(len(hdrs)):
        c = tbl_obj[0, j]
        c.set_facecolor("#1f3a5f")
        c.set_text_props(color="white", fontweight="bold")
    # Bold + light-green any t-stat cell that itself crosses |t| >= strict
    for i, r in enumerate(selected):
        for j, tval in enumerate([r["ic_t"], r["ic_t_2m"], r["pure_t"]], start=3):
            c = tbl_obj[i + 1, j]
            if abs(tval) >= strict:
                c.set_facecolor("#cfe6cf")
                c.set_text_props(fontweight="bold")
        tbl_obj[i + 1, 1].set_text_props(ha="left")
    note = (f"Showing {len(selected)} factors at |t-stat| >= {threshold:.1f} on at least one test "
            f"(of 1m IC, 2m IC, or pure factor return). "
            f"{n_strict} cross strict significance (|t-stat| >= {strict:.1f}, highlighted). "
            f"Sorted by max |t-stat|.")
    fig.text(0.02, 0.02, note, fontsize=8.5, style="italic", color="#555")
    outpath = settings.charts_dir / "table5_significant_factors.png"
    _save(fig, outpath)
    return outpath


def main() -> None:
    _setup_style()
    panel = build_factor_panel().sort_values(["date", "symbol"]).reset_index(drop=True)
    facs = [(f, lab, slug, col) for f, lab, slug, col in factor_list() if col in panel.columns]
    logger.info(f"building per-factor packs for {len(facs)} factors -> charts/factors/<family>/<factor>/")
    rows = []
    for family, label, slug, col in facs:
        logger.info(f"  {family}/{slug} ({col})")
        rows.append(factor_pack(panel, family, label, slug, col))
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    sdf = pd.DataFrame(rows)
    sdf.round(3).to_csv(PROJECT_ROOT / "reports" / "factor_report_summary.csv", index=False)
    t5_path = table5_significant_factors(rows)
    logger.info(f"wrote Table 5 -> {t5_path}")
    logger.info(f"wrote {len(facs)} factor packs (9 files each) + reports/factor_report_summary.csv\n"
                + sdf.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
