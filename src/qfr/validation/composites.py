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

# --------------------------------------------------------------------------
# Within-group "curated" composites: combine ONLY the genuinely-good factors
# inside each style family (drop the duds the full family composite carries).
# These are TRUE composites in the textbook sense - one composite per group.
# --------------------------------------------------------------------------
CURATED_WITHIN_GROUP: list[tuple[str, list[str]]] = [
    ("Value curated (3)", ["freeCashFlowYield", "earningsYield", "salesYield"]),
    ("Quality curated (4)", ["returnOnEquity", "returnOnInvestedCapital",
                             "returnOnAssets", "interestCoverageRatio"]),
    ("Growth curated (4)", ["revenueGrowth", "epsgrowth", "netIncomeGrowth", "ebitdaGrowth"]),
]

# --------------------------------------------------------------------------
# Two-stage multi-factor MODEL: stage 1 = within-group composites (equal-weight
# z-scores), stage 2 = equal-weight combine the group composites + style
# standalones (Momentum, Sentiment) so each STYLE gets equal weight regardless
# of how many factors it has - the canonical institutional construction.
# --------------------------------------------------------------------------
TWO_STAGE_SPEC: list[tuple[str, list[str]]] = [
    ("Value group",     ["freeCashFlowYield", "earningsYield", "salesYield"]),
    ("Quality group",   ["returnOnEquity", "returnOnInvestedCapital",
                         "returnOnAssets", "interestCoverageRatio"]),
    ("Growth group",    ["revenueGrowth", "epsgrowth", "netIncomeGrowth", "ebitdaGrowth"]),
    ("Momentum group",  ["mom_12_1"]),
    ("Sentiment group", ["rating_rev_6m"]),
]

# --------------------------------------------------------------------------
# Family composites already pre-built in the panel (rank-based mean, INCLUDES
# the dud factors). Shown for comparison - typically dominated by the curated
# within-group versions.
# --------------------------------------------------------------------------
FAMILY_COMPOSITES = [
    ("Value family — full (5)", "value", 5),
    ("Quality family — full (8)", "quality", 8),
    ("Momentum family — full (3)", "momentum", 3),
    ("Growth family — full (4)", "growth", 4),
    ("Risk family — full (2)", "risk", 2),
    ("Sentiment family — full (3)", "sentiment", 3),
]

# --------------------------------------------------------------------------
# Cross-group "selection" composites (NOT strict composites, more like a
# multi-factor model selected by t-stat). Included as a context comparison.
# --------------------------------------------------------------------------
CROSS_GROUP_SELECTION = [
    ("Top-5 cross-group", ["returnOnInvestedCapital", "returnOnEquity", "freeCashFlowYield",
                            "revenueGrowth", "epsgrowth"]),
    ("Top-9 cross-group", ["returnOnInvestedCapital", "returnOnEquity", "freeCashFlowYield",
                            "revenueGrowth", "epsgrowth", "salesYield", "ebitdaGrowth",
                            "rating_rev_6m", "netIncomeGrowth"]),
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


def two_stage_composite(panel: pd.DataFrame, groups: list[tuple[str, list[str]]]) -> pd.Series:
    """Two-stage construction:
       1. For each group, equal-weight z-score of its component factors (within).
       2. Equal-weight average of the group z-scores (across).
    So each style gets equal weight regardless of how many factors it contains.
    """
    group_zs = []
    for _, cols in groups:
        if len(cols) == 1:
            group_zs.append(cs_z(panel, cols[0]))
        else:
            group_zs.append(composite_zscore(panel, cols))
    return sum(group_zs) / len(group_zs)


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

    fig, ax = plt.subplots(figsize=(16, max(3, 0.42 * len(cell_text) + 1.6)))
    ax.axis("off")
    ax.set_title(
        "Composite factor significance — within-group composites, two-stage multi-factor model, "
        "family composites, cross-group selection  (bold/green = strict |t| >= 2)",
        fontsize=11.5, fontweight="bold", loc="left", pad=14,
    )
    tbl = ax.table(
        cellText=cell_text, colLabels=cols, cellLoc="center", loc="center",
        colWidths=[0.22, 0.035, 0.075, 0.055, 0.075, 0.055, 0.075, 0.06, 0.075, 0.06, 0.06],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)
    for j in range(len(cols)):
        c = tbl[0, j]
        c.set_facecolor("#1f3a5f")
        c.set_text_props(color="white", fontweight="bold")

    # Per-type row banding (background colour)
    type_bg = {
        "Within-group composite (curated)": "#e8f0fb",   # light blue
        "Multi-factor model (two-stage)":   "#fff6d6",   # light gold
        "Family composite (full, rank-based)": "#f0f0f0",  # light gray
        "Cross-group selection":            "#ffffff",
    }
    # Highlight any t-stat cell with |t| >= 2 (cols 3, 5, 7, 9, 10)
    t_col_idx = [3, 5, 7, 9, 10]
    t_keys = ["t_1m", "t_2m", "Q1_t", "pure_t", "max_|t|"]
    for i, (_, r) in enumerate(df.iterrows()):
        row_bg = type_bg.get(r.get("type", ""), "#ffffff")
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor(row_bg)
        for j, k in zip(t_col_idx, t_keys):
            v = r[k]
            if pd.notna(v) and abs(v) >= 2.0:
                c = tbl[i + 1, j]
                c.set_facecolor("#cfe6cf")
                c.set_text_props(fontweight="bold")
        tbl[i + 1, 0].set_text_props(ha="left", fontweight="bold" if r["max_|t|"] >= 2 else "normal")

    note = (
        "Within-group composites (light blue) combine factors measuring the same concept "
        "(value-only, quality-only, growth-only).  Two-stage model (gold) = equal-weight z-scores "
        "within each group, then equal-weight across groups (style-balanced).  Family composites "
        "(gray) are the existing rank-based mean of ALL components in a family including duds.  "
        "Cross-group selection picks the best individual factors by t-stat regardless of group."
    )
    fig.text(0.02, 0.02, note, fontsize=8.5, style="italic", color="#555")
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup_style()
    panel = build_factor_panel().sort_values(["date", "symbol"]).reset_index(drop=True)
    logger.info(f"composite evaluation: {len(panel):,} rows, {panel['date'].nunique()} months")

    rows = []

    # 1. WITHIN-GROUP CURATED COMPOSITES (the strict-textbook "composites")
    logger.info("-- within-group curated composites --")
    for name, raw_cols in CURATED_WITHIN_GROUP:
        missing = [c for c in raw_cols if c not in panel.columns]
        if missing:
            logger.warning(f"  skipping {name}: missing {missing}")
            continue
        logger.info(f"  {name}  ({len(raw_cols)} factors)")
        comp = composite_zscore(panel, raw_cols)
        rec = composite_stats(panel, comp, name, len(raw_cols))
        rec["type"] = "Within-group composite (curated)"
        rows.append(rec)

    # 2. TWO-STAGE MULTI-FACTOR MODEL
    logger.info("-- two-stage multi-factor model --")
    multi = two_stage_composite(panel, TWO_STAGE_SPEC)
    n_total = sum(len(c) for _, c in TWO_STAGE_SPEC)
    rec = composite_stats(panel, multi,
                          f"Two-stage multi-factor model ({len(TWO_STAGE_SPEC)} groups, {n_total} factors)",
                          n_total)
    rec["type"] = "Multi-factor model (two-stage)"
    rows.append(rec)

    # 3. FAMILY COMPOSITES (full, rank-based, with duds) - comparison
    logger.info("-- family composites (full, for comparison) --")
    for name, col, n in FAMILY_COMPOSITES:
        if col not in panel.columns:
            continue
        rec = composite_stats(panel, panel[col], name, n)
        rec["type"] = "Family composite (full, rank-based)"
        rows.append(rec)

    # 4. CROSS-GROUP SELECTION (multi-factor selection by t-stat) - comparison
    logger.info("-- cross-group t-stat selection (comparison) --")
    for name, raw_cols in CROSS_GROUP_SELECTION:
        missing = [c for c in raw_cols if c not in panel.columns]
        if missing:
            continue
        comp = composite_zscore(panel, raw_cols)
        rec = composite_stats(panel, comp, name, len(raw_cols))
        rec["type"] = "Cross-group selection"
        rows.append(rec)

    df = pd.DataFrame(rows)
    # keep the order intentional (group blocks); but sort within each block by max_|t|
    type_order = ["Within-group composite (curated)", "Multi-factor model (two-stage)",
                  "Family composite (full, rank-based)", "Cross-group selection"]
    df["__t"] = df["type"].map({t: i for i, t in enumerate(type_order)})
    df = df.sort_values(["__t", "max_|t|"], ascending=[True, False]).drop(columns="__t").reset_index(drop=True)

    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(PROJECT_ROOT / "reports" / "composites_summary.csv", index=False)
    render_table(df, settings.charts_dir / "composites_significance.png")
    logger.info("composites significance:\n" + df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
