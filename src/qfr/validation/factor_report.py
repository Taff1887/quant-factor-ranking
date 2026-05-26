"""Part 4c - Per-factor tearsheets: the full factor-testing battery.

For every individual factor we run the standard quant factor-validation battery:

  1. Rank IC          monthly cross-sectional Spearman IC vs next-month return;
                      mean IC, 12-month rolling IC, IC t-stat.
  2. IC decay         average IC at lags 1..12 months + per-lag success rate &
                      t-stat (how fast the market prices the signal away).
  3. Fractiles        10 equal-weight deciles (D1 = best), monthly rebalanced:
                      cumulative growth-of-$1 vs the universe + a stats table
                      (D1..D10 + D1-D10 spread + Market: total/active return,
                      tracking error, information ratio, t-stat(IR), monthly
                      success, turnover, volatility, Sharpe, CAPM alpha/beta).
  4. Pure factor ret  monthly cross-sectional (Fama-MacBeth) regression of the
                      forward return on the normalised factor PLUS risk controls
                      (size, sector, book-to-price); the factor coefficient is the
                      "pure" return to a 1-SD exposure. Reported pure vs raw,
                      cumulative, with annualised return / TE / IR / success / t.

Each factor gets a 4-panel tearsheet PNG (charts/factors/<factor>.png) and a row
in reports/factor_report_summary.csv (+ per-decile stats in
reports/factor_report_fractiles.csv).

Run::  uv run python -m qfr.validation.factor_report
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.factors.build import FAMILIES, build_factor_panel
from qfr.utils.config import PROJECT_ROOT, settings
from qfr.utils.logging import logger

ANN = 12
MIN_NAMES = 20
N_FRACTILES = 10        # deciles (D1 = top, D{n} = bottom)
MAX_LAG = 12

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


def factor_list() -> list[tuple[str, str, str]]:
    """(group, label, rank-column) for every individual factor (deduped)."""
    items, seen = [], set()
    for fam, comps in FAMILIES.items():
        for c in comps:
            col = c + "_rk"
            if col not in seen:
                seen.add(col)
                items.append((fam, LABELS.get(c, c), col))
    for fam, c in [("size", "size_raw"), ("reversal", "st_rev")]:
        col = c + "_rk"
        if col not in seen:
            items.append((fam, LABELS.get(c, c), col))
    return items


# --------------------------------------------------------------------------
# 1. Rank IC
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


def _ts(s: pd.Series) -> dict:
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
        rn = p.groupby("symbol")["ret_fwd_1m"].shift(-(n - 1))  # single-month return n months ahead
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
# 3. Fractiles (deciles, D1 = best)
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


def fractiles(panel: pd.DataFrame, col: str, ret: str = "ret_fwd_1m", n: int = N_FRACTILES):
    d = panel.dropna(subset=[col, ret]).copy()
    q = d.groupby("date")[col].transform(lambda s: pd.qcut(s.rank(method="first"), n, labels=False))
    d["D"] = (n - q).astype(int)  # D1 = top decile (best factor score)
    qret = d.groupby(["date", "D"])[ret].mean().unstack("D").sort_index()
    qret.columns = [f"D{c}" for c in qret.columns]
    mkt = d.groupby("date")[ret].mean().sort_index()
    years = len(mkt) / ANN
    spread_name = f"D1-D{n}"

    def stat_block(r: pd.Series, active_vs_mkt: bool) -> dict:
        ann_tot = (1 + r).prod() ** (ANN / len(r)) - 1
        act = (r - mkt).dropna() if active_vs_mkt else r
        te = act.std() * np.sqrt(ANN)
        ir = (act.mean() / act.std() * np.sqrt(ANN)) if act.std() > 0 else np.nan
        beta, alpha = _capm(r, mkt) if active_vs_mkt else (np.nan, np.nan)
        return {"total_return": ann_tot,
                "active_return": (ann_tot - ((1 + mkt).prod() ** (ANN / len(mkt)) - 1)) if active_vs_mkt else ann_tot,
                "tracking_error": te, "info_ratio": ir,
                "t_stat": (ir * np.sqrt(years)) if pd.notna(ir) else np.nan,
                "monthly_success": float((act > 0).mean()),
                "volatility": r.std() * np.sqrt(ANN),
                "sharpe": (r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else np.nan,
                "capm_beta": beta, "capm_alpha": alpha}

    rows = {}
    for dc in qret.columns:
        rows[dc] = stat_block(qret[dc].dropna(), active_vs_mkt=True)
        rows[dc]["turnover"] = _turnover(d[d["D"] == int(dc[1:])][["date", "symbol"]])
    spread = (qret["D1"] - qret[f"D{n}"]).dropna()
    rows[spread_name] = stat_block(spread, active_vs_mkt=False)
    rows[spread_name].update({"active_return": stat_block(spread, False)["total_return"], "turnover": np.nan})
    rows["Market"] = stat_block(mkt, active_vs_mkt=False)
    rows["Market"].update({"active_return": 0.0, "turnover": np.nan})
    tbl = pd.DataFrame(rows).T
    return qret, mkt, tbl


# --------------------------------------------------------------------------
# 4. Pure factor returns (monthly Fama-MacBeth cross-sectional regression)
# --------------------------------------------------------------------------
def _z(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else np.zeros_like(x)


def pure_factor_return(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Monthly factor coefficient controlling for size + sector + book-to-price (pure),
    and the univariate coefficient (raw). Units: return to a 1-SD exposure."""
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
    sd = s.std()
    return {"ann_return": s.mean() * ANN, "tracking_error": sd * np.sqrt(ANN),
            "info_ratio": (s.mean() / sd * np.sqrt(ANN)) if sd else np.nan,
            "t_stat": (s.mean() / sd * np.sqrt(len(s))) if sd else np.nan,
            "success": float((s > 0).mean())}


# --------------------------------------------------------------------------
# Tearsheet (4 panels) + run loop
# --------------------------------------------------------------------------
def tearsheet(panel: pd.DataFrame, group: str, label: str, col: str):
    import matplotlib.pyplot as plt

    from qfr.utils.viz import PALETTE, save_fig, set_plot_style

    set_plot_style()
    ic = ic_monthly(panel, col)
    icst = _ts(ic)
    dec = ic_decay(panel, col)
    qret, mkt, ftbl = fractiles(panel, col)
    pfr = pure_factor_return(panel, col)
    pst, rst = _series_stats(pfr["pure"]), _series_stats(pfr["raw"])

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(15, 10))

    roll = ic.rolling(12, min_periods=6).mean()
    a1.plot(ic.index, ic.values * 100, color=PALETTE["muted"], lw=0.6, alpha=0.5, label="monthly IC")
    a1.plot(roll.index, roll.values * 100, color=PALETTE["primary"], lw=1.9, label="12m rolling IC")
    a1.axhline(icst["mean_ic"] * 100, color=PALETTE["green"], lw=1.0, ls="--",
               label=f"mean {icst['mean_ic'] * 100:.2f}%")
    a1.axhline(0, color="#444", lw=0.7)
    a1.set_title(f"1. Rank IC — mean {icst['mean_ic'] * 100:.2f}%, t={icst['t_stat']:.2f}, hit {icst['success'] * 100:.0f}%")
    a1.set_ylabel("rank IC (%)")
    a1.legend(fontsize=7, loc="upper left")

    a2.bar(dec["lag"], dec["avg_ic"] * 100, color=PALETTE["primary"], alpha=0.85)
    a2.axhline(0, color="#444", lw=0.7)
    a2.set_xlabel("lag (months ahead)")
    a2.set_ylabel("avg rank IC (%)")
    a2.set_xticks(list(dec["lag"]))
    a2b = a2.twinx()
    a2b.plot(dec["lag"], dec["success"] * 100, color=PALETTE["green"], marker="o", lw=1.4)
    a2b.axhline(50, color=PALETTE["green"], lw=0.6, ls=":")
    a2b.set_ylabel("success rate (%)", color=PALETTE["green"])
    a2b.set_ylim(30, 75)
    a2.set_title("2. IC decay (bars = avg IC, line = success rate)")

    import matplotlib as mpl

    fcolors = mpl.colormaps["RdYlGn"](np.linspace(0.88, 0.12, len(qret.columns)))
    for i, dc in enumerate(qret.columns):
        a3.plot(qret.index, (1 + qret[dc].fillna(0)).cumprod().values,
                lw=1.2, color=fcolors[i], label=dc)
    a3.plot(mkt.index, (1 + mkt.fillna(0)).cumprod().values, color="#222", lw=1.8, ls="--", label="Market")
    a3.set_yscale("log")
    a3.set_title(f"3. Fractiles — growth of $1 (D1 = top decile, D{N_FRACTILES} = bottom)")
    a3.set_ylabel("growth of $1 (log)")
    a3.legend(fontsize=6.5, ncol=4, loc="upper left")

    a4.plot(pfr.index, (1 + pfr["pure"]).cumprod().values * 100, color=PALETTE["primary"], lw=1.9, label="pure")
    a4.plot(pfr.index, (1 + pfr["raw"]).cumprod().values * 100, color=PALETTE["muted"], lw=1.3, label="raw")
    a4.set_title("4. Pure vs raw factor return (1-SD exposure, index = 100)")
    a4.set_ylabel("index")
    a4.legend(fontsize=8, loc="upper left")
    txt = (f"pure: ann {pst['ann_return'] * 100:5.1f}%  IR {pst['info_ratio']:.2f}  "
           f"t {pst['t_stat']:.2f}  hit {pst['success'] * 100:.0f}%\n"
           f"raw : ann {rst['ann_return'] * 100:5.1f}%  IR {rst['info_ratio']:.2f}  t {rst['t_stat']:.2f}")
    a4.text(0.02, 0.97, txt, transform=a4.transAxes, va="top", fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.85))

    fig.suptitle(f"{label}  ·  {group.capitalize()} factor  ·  S&P 500, 2010-2026 (gross)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, col.replace("_rk", ""), subdir="factors")

    top = ftbl.loc["D1"]
    ls = ftbl.loc[f"D1-D{N_FRACTILES}"]
    summary = {"group": group, "factor": label, "mean_ic_%": icst["mean_ic"] * 100, "ic_t": icst["t_stat"],
               "ic_hit_%": icst["success"] * 100, "Top_active_%": top["active_return"] * 100, "Top_IR": top["info_ratio"],
               "TopBot_ann_%": ls["total_return"] * 100, "TopBot_IR": ls["info_ratio"], "Top_turnover_%": top["turnover"] * 100,
               "pure_ann_%": pst["ann_return"] * 100, "pure_IR": pst["info_ratio"], "pure_t": pst["t_stat"],
               "raw_ann_%": rst["ann_return"] * 100}
    return summary, ftbl.assign(group=group, factor=label).rename_axis("fractile").reset_index()


def main() -> None:
    panel = build_factor_panel().sort_values(["date", "symbol"]).reset_index(drop=True)
    facs = [(g, l, c) for g, l, c in factor_list() if c in panel.columns]
    logger.info(f"building {len(facs)} factor tearsheets ...")
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    summary, frac_all = [], []
    for group, label, col in facs:
        logger.info(f"  tearsheet: {label} ({col})")
        row, ftbl = tearsheet(panel, group, label, col)
        summary.append(row)
        frac_all.append(ftbl)
    sdf = pd.DataFrame(summary)
    sdf.round(3).to_csv(PROJECT_ROOT / "reports" / "factor_report_summary.csv", index=False)
    pd.concat(frac_all, ignore_index=True).round(4).to_csv(
        PROJECT_ROOT / "reports" / "factor_report_fractiles.csv", index=False)
    logger.info(f"wrote {len(facs)} tearsheets -> charts/factors/ + 2 CSVs\n"
                + sdf.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

