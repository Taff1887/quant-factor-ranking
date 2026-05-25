"""Part 4b - The "final factor screen" (Macquarie-style summary sheet).

For every candidate factor - the six **family composites** plus their underlying
**components** - we report, over the 2010+ window, the battery of statistics an
institutional quant desk uses to *select* factors for a multi-factor model (cf.
Macquarie Research, "A Practitioner's Guide to Factor Models"):

    Avg Rank IC (lag 1m / 2m)  cross-sectional Spearman corr of the factor with
                               the return 1 and 2 months ahead - signal strength
                               and how fast it decays.
    Hit Rate (lag 1m / 2m)     share of months with a positive IC (consistency).
    t-stat (lag 1m / 2m)       mean_IC / std_IC * sqrt(N) - is the IC real?
    Active Return top / bottom annualised return of the top / bottom quintile in
                               excess of the equal-weight universe (GROSS).
    Active Return top-bottom   annualised long-short (top minus bottom).
    Tracking Error top / bot   annualised vol of the quintile's active return.
    Information Ratio top / bot active return / tracking error.
    Monthly Success top / bot  share of months the quintile beats the universe.
    Avg Turnover top / bottom  average one-way monthly name turnover (cost proxy).

Macquarie's selection rule favours factors with **high & significant lag-1m/2m
rank ICs**, **well-separated top vs bottom** active returns / IRs, and **low
turnover**. Returns are gross so signal quality and trading cost (turnover) can
be judged separately.

Run::  uv run python -m qfr.validation.factor_screen
Outputs: reports/factor_screen.csv + charts/04b_factor_screen.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.factors.build import FAMILIES, build_factor_panel
from qfr.utils.config import PROJECT_ROOT
from qfr.utils.logging import logger

ANN = 12
N_FRACTILES = 5          # quintiles (Macquarie "fractiles")
MIN_NAMES = 20           # need a real cross-section for an IC
TURNOVER_SKIP_FIRST = True

# Pretty labels for the component rank columns.
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
    "lowVol": "Low volatility", "size_raw": "Small size", "st_rev": "Short-term reversal (-1m)",
}


def candidate_columns() -> list[tuple[str, str, str]]:
    """Ordered (group, label, column) list: each family composite then its parts."""
    items: list[tuple[str, str, str]] = []
    for fam, comps in FAMILIES.items():
        items.append((fam.capitalize(), f"{fam.capitalize()} (composite)", fam))
        for c in comps:
            items.append((fam.capitalize(), "   " + LABELS.get(c, c), c + "_rk"))
    items.append(("Size", "Size (composite)", "size"))
    items.append(("Size", "   " + LABELS["size_raw"], "size_raw_rk"))
    items.append(("Reversal", LABELS["st_rev"], "st_rev_rk"))
    return items


def _ic_stats(panel: pd.DataFrame, col: str, ret_col: str) -> dict:
    ics = []
    for _, g in panel.groupby("date"):
        sub = g[[col, ret_col]].dropna()
        if len(sub) >= MIN_NAMES:
            ic = sub[col].corr(sub[ret_col], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
    s = pd.Series(ics, dtype=float)
    if s.empty:
        return {"ic": np.nan, "t": np.nan, "hit": np.nan}
    sd = s.std()
    return {
        "ic": s.mean(),
        "t": (s.mean() / sd * np.sqrt(len(s))) if sd > 0 else np.nan,
        "hit": float((s > 0).mean()),
    }


def _turnover(frame: pd.DataFrame) -> float:
    """Average one-way monthly turnover of an equal-weight quintile."""
    if frame.empty:
        return np.nan
    w = frame.assign(w=1.0).pivot_table(index="date", columns="symbol", values="w",
                                        fill_value=0.0).sort_index()
    w = w.div(w.sum(axis=1), axis=0)
    to = 0.5 * w.diff().abs().sum(axis=1)
    to = to.iloc[1:] if TURNOVER_SKIP_FIRST else to
    return float(to.mean())


def _fractile_stats(panel: pd.DataFrame, col: str, ret_col: str = "ret_fwd_1m",
                    q: int = N_FRACTILES) -> dict:
    d = panel.dropna(subset=[col, ret_col]).copy()
    d["fr"] = d.groupby("date")[col].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False)
    )
    leg = lambda fr: fr.groupby("date")[ret_col].mean()
    uni = leg(d)
    top = leg(d[d["fr"] == q - 1]).reindex(uni.index)
    bot = leg(d[d["fr"] == 0]).reindex(uni.index)
    act_top, act_bot = (top - uni).dropna(), (bot - uni).dropna()
    ls = (top - bot).dropna()

    def ir(s: pd.Series) -> float:
        sd = s.std()
        return float(s.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan

    return {
        "act_top": act_top.mean() * ANN, "act_bot": act_bot.mean() * ANN, "act_ls": ls.mean() * ANN,
        "te_top": act_top.std() * np.sqrt(ANN), "te_bot": act_bot.std() * np.sqrt(ANN),
        "ir_top": ir(act_top), "ir_bot": ir(act_bot),
        "succ_top": float((act_top > 0).mean()), "succ_bot": float((act_bot > 0).mean()),
        "turn_top": _turnover(d[d["fr"] == q - 1]), "turn_bot": _turnover(d[d["fr"] == 0]),
    }


def build_screen(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    if panel is None:
        panel = build_factor_panel()
    p = panel.sort_values(["symbol", "date"]).copy()
    p["ret_2m_ahead"] = p.groupby("symbol")["ret_fwd_1m"].shift(-1)  # return over [t+1, t+2]

    rows = []
    for group, label, col in candidate_columns():
        if col not in p.columns:
            logger.warning(f"screen: column {col!r} missing, skipping")
            continue
        ic1, ic2 = _ic_stats(p, col, "ret_fwd_1m"), _ic_stats(p, col, "ret_2m_ahead")
        fr = _fractile_stats(p, col, "ret_fwd_1m")
        rows.append({
            "Group": group, "Factor": label,
            "IC_1m": ic1["ic"], "IC_2m": ic2["ic"],
            "Hit_1m": ic1["hit"], "Hit_2m": ic2["hit"],
            "t_1m": ic1["t"], "t_2m": ic2["t"],
            "Act_top": fr["act_top"], "Act_bot": fr["act_bot"], "Act_LS": fr["act_ls"],
            "TE_top": fr["te_top"], "TE_bot": fr["te_bot"],
            "IR_top": fr["ir_top"], "IR_bot": fr["ir_bot"],
            "Succ_top": fr["succ_top"], "Succ_bot": fr["succ_bot"],
            "Turn_top": fr["turn_top"], "Turn_bot": fr["turn_bot"],
        })
    return pd.DataFrame(rows)


# columns expressed as percentages for display
PCT_COLS = ["IC_1m", "IC_2m", "Hit_1m", "Hit_2m", "Act_top", "Act_bot", "Act_LS",
            "TE_top", "TE_bot", "Succ_top", "Succ_bot", "Turn_top", "Turn_bot"]
TWO_DP_PCT = ["IC_1m", "IC_2m", "Act_top", "Act_bot", "Act_LS"]  # small magnitudes -> 2 dp
NO_PCT = ["t_1m", "t_2m", "IR_top", "IR_bot"]                    # t-stat / IR: no % sign


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in PCT_COLS:
        out[c] = (out[c] * 100).round(2 if c in TWO_DP_PCT else 1)
    for c in NO_PCT:
        out[c] = out[c].round(2)
    return out


def _fmt_cell(col: str, v: float) -> str:
    """Match the Macquarie sheet: % on every column except t-stat and IR."""
    if pd.isna(v):
        return "-"
    if col in NO_PCT:
        return f"{v:.2f}"
    if col in TWO_DP_PCT:
        return f"{v:.2f}%"
    return f"{v:.1f}%"


def _selected(row: pd.Series) -> bool:
    """Macquarie-style highlight: significant 1m/2m rank IC and/or a strong top IR."""
    return bool((row["t_1m"] >= 1.5) or (row["t_2m"] >= 1.5) or (row["IR_top"] >= 0.30))


def _panel(ax, disp: pd.DataFrame, cols: list[str], subhdr: list[str],
           spans: list[tuple[str, int, int]], colw: list[float], title: str) -> None:
    """Render one Macquarie-style sub-table with a Group column + spanning headers."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=34)

    is_comp = disp["Factor"].str.contains("composite").to_numpy()
    sel = disp.apply(_selected, axis=1).to_numpy()
    seen: set[str] = set()
    text = []
    for _, row in disp.iterrows():
        g = row["Group"]
        glabel = "" if g in seen else g
        seen.add(g)
        vals = [_fmt_cell(c, row[c]) for c in cols]
        text.append([glabel, row["Factor"].strip(), *vals])

    collabels = ["Group", "Factor", *subhdr]
    tbl = ax.table(cellText=text, colLabels=collabels, colWidths=colw,
                   cellLoc="center", bbox=[0, 0, 1.0, 0.86])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    ncol = len(collabels)
    for j in range(ncol):  # sub-header row
        c = tbl[0, j]
        c.set_facecolor("#2b3a55")
        c.set_text_props(color="white", fontweight="bold")
    for i in range(len(text)):  # data rows
        bg = "#d9dee8" if is_comp[i] else ("#eef0f4" if sel[i] else "white")
        for j in range(ncol):
            c = tbl[i + 1, j]
            c.set_facecolor(bg)
            if j <= 1:
                c.set_text_props(ha="left", fontweight="bold" if (is_comp[i] or j == 0) else "normal")
            elif is_comp[i]:
                c.set_text_props(fontweight="bold")

    cum = np.concatenate([[0.0], np.cumsum(colw)])
    cum = cum / cum[-1]  # table spans the full axes width
    for label, a, b in spans:
        xL, xR = cum[a], cum[b + 1]
        ax.text((xL + xR) / 2, 0.95, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#2b3a55")
        ax.plot([xL + 0.004, xR - 0.004], [0.905, 0.905], color="#2b3a55", lw=1.1, clip_on=False)


def make_figures(disp: pd.DataFrame) -> None:
    """Two-panel Macquarie-style screen table -> charts/04b_factor_screen.png."""
    import matplotlib.pyplot as plt

    from qfr.utils.viz import save_fig, set_plot_style

    set_plot_style()
    a_cols = ["IC_1m", "IC_2m", "Hit_1m", "Hit_2m", "t_1m", "t_2m", "Act_top", "Act_bot", "Act_LS"]
    a_sub = ["Lag 1m", "Lag 2m", "Lag 1m", "Lag 2m", "Lag 1m", "Lag 2m", "Top", "Bottom", "Top-Bottom"]
    a_spans = [("Avg Rank IC", 2, 3), ("Hit Rate", 4, 5), ("t-stat", 6, 7), ("Active Return", 8, 10)]
    a_w = [0.11, 0.23] + [0.66 / 9] * 9

    b_cols = ["TE_top", "TE_bot", "IR_top", "IR_bot", "Succ_top", "Succ_bot", "Turn_top", "Turn_bot"]
    b_sub = ["Top", "Bottom", "Top", "Bottom", "Top", "Bottom", "Top", "Bottom"]
    b_spans = [("Tracking Error", 2, 3), ("Information Ratio", 4, 5),
               ("Monthly Success Rate", 6, 7), ("Avg Turnover", 8, 9)]
    b_w = [0.11, 0.23] + [0.66 / 8] * 8

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(16, 20))
    fig.suptitle("Summary sheet of factor performance statistics - S&P 500 (2010-2026)",
                 fontsize=15, fontweight="bold", y=0.995)
    _panel(axA, disp, a_cols, a_sub, a_spans, a_w,
           "Signal: average rank IC %, hit rate % (months IC>0) and t-stat; active return ann. % (gross)")
    _panel(axB, disp, b_cols, b_sub, b_spans, b_w,
           "Portfolio: top/bottom quintile vs equal-weight universe - tracking error %, information ratio, monthly success %, one-way turnover %")
    fig.text(0.5, 0.018,
             "Highlighted = factors that fit the criteria: high & significant 1m/2m rank ICs, well-distinguished top/bottom fractiles, lower turnover."
             "   Bold = family composite.   Source: qfr factor screen, 2010-2026 (gross, quintiles).",
             ha="center", fontsize=9, style="italic", color="#555")
    fig.subplots_adjust(top=0.94, bottom=0.05, hspace=0.16)
    save_fig(fig, "04b_factor_screen")


def main() -> None:
    panel = build_factor_panel()
    df = build_screen(panel)
    disp = to_display(df)
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    disp.to_csv(PROJECT_ROOT / "reports" / "factor_screen.csv", index=False)
    make_figures(disp)

    with pd.option_context("display.width", 240, "display.max_columns", 40,
                           "display.max_rows", 60):
        logger.info("Factor screen (2010+, gross, quintiles) - IC/hit/t in %, returns/TE/turnover ann.%:\n"
                    + disp.to_string(index=False))
    logger.info(f"\nwrote reports/factor_screen.csv ({len(disp)} factors)")


if __name__ == "__main__":
    main()
