"""Composite factor construction and significance testing.

Applies the textbook signal-construction methodology:

  - Cross-sectional **z-scores** (winsorise -> standardise within date) instead of
    raw ranks (z-scores preserve magnitude information rank-mean averages lose).
  - Combine factor z-scores into a composite z-score (equal-weight default).
  - Test the composite via the **Fundamental Law of Active Management** (Grinold):

        IR(composite)  ~  E[IC] x sqrt(N_independent)

    so combining several borderline-significant factors should diversify away
    the monthly IC noise and lift the composite t-stat above 2.

We test:
  - the seven existing family composites (already in the panel),
  - cross-family z-score composites (Quality-Value, Quality-Growth, etc.),
  - "top-N by t-stat" composites of the screen's strongest individual factors,
  - and a broad multi-style composite.

Each composite is evaluated on rank IC (1m + 2m), pure factor return
(Fama-MacBeth, controls = size + sector + book-to-price), and Q1 information
ratio. Cells where any |t-stat| >= 2 are highlighted - i.e. composites that
clear strict 95% statistical significance.

Run::  uv run python -m qfr.validation.composites
Outputs: charts/composites_significance.png + reports/composites_summary.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.factors.build import build_factor_panel
from qfr.utils.config import PROJECT_ROOT, settings
from qfr.utils.logging import logger
from qfr.validation.factor_report import (
    _ic_ts_stats,
    _series_stats,
    _setup_style,
    fractiles,
    ic_2m_series,
    ic_monthly,
    pure_factor_return,
)

# Cross-family composite specs. Each entry: (name, list-of-raw-factor-cols).
# These are the WINSORISED raw column names (no _rk suffix) - we z-score them.
CROSS_FAMILY = [
    ("Quality-Value (4)", ["returnOnEquity", "returnOnInvestedCapital",
                           "freeCashFlowYield", "earningsYield"]),
    ("Quality-Growth (4)", ["returnOnEquity", "returnOnInvestedCapital",
                            "revenueGrowth", "epsgrowth"]),
    ("Value-Growth (4)", ["freeCashFlowYield", "earningsYield",
                          "revenueGrowth", "epsgrowth"]),
    ("Top-3 by t-stat", ["returnOnInvestedCapital", "returnOnEquity", "freeCashFlowYield"]),
    ("Top-5 by t-stat", ["returnOnInvestedCapital", "returnOnEquity", "freeCashFlowYield",
                         "revenueGrowth", "epsgrowth"]),
    ("Top-7 by t-stat", ["returnOnInvestedCapital", "returnOnEquity", "freeCashFlowYield",
                         "revenueGrowth", "epsgrowth", "salesYield", "ebitdaGrowth"]),
    ("Top-9 by t-stat", ["returnOnInvestedCapital", "returnOnEquity", "freeCashFlowYield",
                         "revenueGrowth", "epsgrowth", "salesYield", "ebitdaGrowth",
                         "rating_rev_6m", "netIncomeGrowth"]),
    ("QVG broad (12)", ["returnOnEquity", "returnOnInvestedCapital", "returnOnAssets",
                        "interestCoverageRatio", "freeCashFlowYield", "earningsYield",
                        "salesYield", "ebitdaToEV", "revenueGrowth", "epsgrowth",
                        "netIncomeGrowth", "ebitdaGrowth"]),
    ("Multi-style (best of each, 6)", ["returnOnEquity", "freeCashFlowYield",
                                       "revenueGrowth", "mom_12_1", "rating_rev_6m",
                                       "interestCoverageRatio"]),
]

# Family composites already pre-built in the panel.
FAMILY_COMPOSITES = [
    ("Value family (5)", "value", 5),
    ("Quality family (8)", "quality", 8),
    ("Momentum family (3)", "momentum", 3),
    ("Growth family (4)", "growth", 4),
    ("Risk family (2)", "risk", 2),
    ("Sentiment family (3)", "sentiment", 3),
]


def cs_z(panel: pd.DataFrame, col: str) -> pd.Series:
    """Cross-sectional z-score per date (winsorisation is upstream in build)."""
    g = panel.groupby("date")[col]
    return (panel[col] - g.transform("mean")) / g.transform("std")


def composite_zscore(panel: pd.DataFrame, factor_cols: list[str],
                     weights: list[float] | None = None) -> pd.Series:
    """Equal-weight (default) or custom-weight z-score composite."""
    if weights is None:
        weights = [1.0 / len(factor_cols)] * len(factor_cols)
    weights = np.array(weights) / np.array(weights).sum()
    pieces = []
    for w, c in zip(weights, factor_cols):
        if c not in panel.columns:
            raise KeyError(f"factor column missing from panel: {c}")
        pieces.append(w * cs_z(panel, c))
    return sum(pieces)


def composite_stats(panel: pd.DataFrame, comp: pd.Series, name: str, n_factors: int) -> dict:
    """Evaluate one composite: IC (1m + 2m), pure factor return, top-quintile IR."""
    p = panel.copy()
    p["_comp"] = comp
    ic = ic_monthly(p, "_comp")
    icst = _ic_ts_stats(ic)
    ic2 = ic_2m_series(p, "_comp")
    ic2st = _ic_ts_stats(ic2)
    pfr = pure_factor_return(p, "_comp")
    pure = _series_stats(pfr["pure"])
    try:
        _, _, tbl5 = fractiles(p, "_comp", n=5)
        q1_ir = tbl5.loc["Q1", "info_ratio"]
        q1_t = tbl5.loc["Q1", "t_stat"]
    except Exception:
        q1_ir = np.nan
        q1_t = np.nan
    max_t = max(abs(icst["t_stat"]), abs(ic2st["t_stat"]), abs(pure["t_stat"]), abs(q1_t))
    return {
        "composite": name, "N": n_factors,
        "mean_ic_%": icst["mean_ic"] * 100, "t_1m": icst["t_stat"],
        "mean_ic_2m_%": ic2st["mean_ic"] * 100, "t_2m": ic2st["t_stat"],
        "Q1_IR": q1_ir, "Q1_t": q1_t,
        "pure_ann_%": pure["ann_return"] * 100, "pure_t": pure["t_stat"],
        "max_|t|": max_t,
    }


def render_table(df: pd.DataFrame, outpath) -> None:
    import matplotlib.pyplot as plt
    cols = ["Composite", "N", "Avg IC (1m, %)", "t (1m)", "Avg IC (2m, %)", "t (2m)",
            "Q1 IR (quintile)", "t (Q1 IR)", "Pure return (%)", "t (pure)", "Max |t|"]
    keys = ["composite", "N", "mean_ic_%", "t_1m", "mean_ic_2m_%", "t_2m",
            "Q1_IR", "Q1_t", "pure_ann_%", "pure_t", "max_|t|"]

    def fmt(v, key):
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return "—"
        if key in ("N",):
            return f"{int(v)}"
        if key in ("mean_ic_%", "mean_ic_2m_%", "pure_ann_%"):
            return f"{v:.2f}%"
        return f"{v:.2f}"

    cell_text = [[fmt(r[k], k) for k in keys] for _, r in df.iterrows()]

    fig, ax = plt.subplots(figsize=(15, max(3, 0.42 * len(cell_text) + 1.6)))
    ax.axis("off")
    ax.set_title(
        "Composite factors significance test  (z-score composites; bold/green = strict |t| >= 2)",
        fontsize=12, fontweight="bold", loc="left", pad=14,
    )
    tbl = ax.table(
        cellText=cell_text, colLabels=cols, cellLoc="center", loc="center",
        colWidths=[0.20, 0.04, 0.08, 0.06, 0.08, 0.06, 0.09, 0.07, 0.09, 0.07, 0.07],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)
    for j in range(len(cols)):
        c = tbl[0, j]
        c.set_facecolor("#1f3a5f")
        c.set_text_props(color="white", fontweight="bold")
    # Highlight any t-stat cell with |t| >= 2 (cols 3, 5, 7, 9, 10)
    t_col_idx = [3, 5, 7, 9, 10]
    t_keys = ["t_1m", "t_2m", "Q1_t", "pure_t", "max_|t|"]
    for i, (_, r) in enumerate(df.iterrows()):
        for j, k in zip(t_col_idx, t_keys):
            v = r[k]
            if pd.notna(v) and abs(v) >= 2.0:
                c = tbl[i + 1, j]
                c.set_facecolor("#cfe6cf")
                c.set_text_props(fontweight="bold")
        tbl[i + 1, 0].set_text_props(ha="left", fontweight="bold" if r["max_|t|"] >= 2 else "normal")
    note = ("Fundamental Law of Active Management (Grinold): IR ~ E[IC] x sqrt(N_independent). "
            "Combining several borderline-significant factors should lift the composite t-stat above 2.  "
            "Z-score composites: cross-sectional winsorise -> standardise -> equal-weight average.")
    fig.text(0.02, 0.02, note, fontsize=8.5, style="italic", color="#555")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup_style()
    panel = build_factor_panel().sort_values(["date", "symbol"]).reset_index(drop=True)
    logger.info(f"composite evaluation: {len(panel):,} rows, {panel['date'].nunique()} months")

    rows = []
    # Family composites: use the pre-built columns directly
    for name, col, n in FAMILY_COMPOSITES:
        if col not in panel.columns:
            continue
        logger.info(f"  family composite: {name}")
        rows.append(composite_stats(panel, panel[col], name, n))

    # Cross-family z-score composites
    for name, raw_cols in CROSS_FAMILY:
        missing = [c for c in raw_cols if c not in panel.columns]
        if missing:
            logger.warning(f"  skipping {name}: missing {missing}")
            continue
        logger.info(f"  z-score composite: {name}  ({len(raw_cols)} factors)")
        comp = composite_zscore(panel, raw_cols)
        rows.append(composite_stats(panel, comp, name, len(raw_cols)))

    df = pd.DataFrame(rows).sort_values("max_|t|", ascending=False).reset_index(drop=True)
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(PROJECT_ROOT / "reports" / "composites_summary.csv", index=False)
    render_table(df, settings.charts_dir / "composites_significance.png")
    logger.info("composites significance (sorted by max |t|):\n" + df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
