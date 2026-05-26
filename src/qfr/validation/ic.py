"""Part 4 - Factor validation via the Information Coefficient (IC).

The IC is the workhorse diagnostic of cross-sectional equity research: for each
rebalance date, the cross-sectional **rank correlation** between a factor's score
and the subsequent forward return. A factor with a positive, stable, statistically
significant mean IC carries genuine ranking information.

We report, per factor x horizon (1/3/6m):
    mean_ic   - average rank IC (the headline signal strength)
    ic_std    - volatility of the IC (consistency)
    ic_ir     - IC information ratio = mean/std (risk-adjusted signal)
    t_stat    - mean/std * sqrt(N): is the mean IC distinguishable from zero?
    hit_rate  - share of months with IC > 0

Plus rolling IC (regime behaviour) and an IC-decay curve across horizons.

Run::

    uv run python -m qfr.validation.ic

Outputs: reports/ic_summary.csv + charts/04_*.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.utils.config import PROJECT_ROOT, settings
from qfr.utils.io import read_parquet
from qfr.utils.logging import logger

FACTORS = ["value", "quality", "momentum", "growth", "risk", "sentiment", "size"]
HORIZONS = (1, 3, 6)
MIN_NAMES = 20  # need a reasonable cross-section to compute a meaningful IC


def compute_ic(f: pd.DataFrame, factors=FACTORS, horizons=HORIZONS) -> pd.DataFrame:
    """Monthly cross-sectional rank IC, long format [date, factor, horizon, ic]."""
    rows = []
    for d, g in f.groupby("date"):
        for fac in factors:
            for h in horizons:
                sub = g[[fac, f"ret_fwd_{h}m"]].dropna()
                if len(sub) >= MIN_NAMES:
                    ic = sub[fac].corr(sub[f"ret_fwd_{h}m"], method="spearman")
                    rows.append((d, fac, h, ic))
    return pd.DataFrame(rows, columns=["date", "factor", "horizon", "ic"])


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    g = ic.groupby(["factor", "horizon"])["ic"]
    s = g.agg(mean_ic="mean", ic_std="std", n="count").reset_index()
    s["ic_ir"] = s["mean_ic"] / s["ic_std"]
    s["t_stat"] = s["mean_ic"] / s["ic_std"] * np.sqrt(s["n"])
    hr = g.apply(lambda x: float((x > 0).mean())).rename("hit_rate").reset_index()
    s = s.merge(hr, on=["factor", "horizon"])
    return s.round({"mean_ic": 4, "ic_std": 4, "ic_ir": 3, "t_stat": 2, "hit_rate": 3})


def make_figures(ic: pd.DataFrame, summ: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    from qfr.utils.viz import PALETTE, save_fig, set_plot_style

    set_plot_style()

    # 1) mean rank IC at 1m, with significance shading
    s1 = summ[summ["horizon"] == 1].set_index("factor").reindex(FACTORS)
    fig, ax = plt.subplots()
    colors = [PALETTE["green"] if abs(t) >= 2 else PALETTE["muted"] for t in s1["t_stat"]]
    ax.bar(s1.index, s1["mean_ic"], color=colors)
    for i, (m, t) in enumerate(zip(s1["mean_ic"], s1["t_stat"])):
        ax.text(i, m + (0.001 if m >= 0 else -0.001), f"t={t:.1f}", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_title("Mean rank IC vs forward 1-month return (green = |t| >= 2)")
    ax.set_ylabel("mean rank IC")
    save_fig(fig, "04_ic_summary")

    # 2) rolling 12m IC (1m horizon)
    piv = (ic[ic["horizon"] == 1].pivot_table(index="date", columns="factor", values="ic")
           .reindex(columns=FACTORS).sort_index())
    roll = piv.rolling(12, min_periods=6).mean()
    fig, ax = plt.subplots(figsize=(13, 7))
    for fac in FACTORS:
        ax.plot(roll.index, roll[fac], lw=1.6, label=fac)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_title("Rolling 12-month rank IC (1-month horizon)")
    ax.set_ylabel("rolling mean IC")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save_fig(fig, "04_rolling_ic")

    # 3) IC decay across horizons
    fig, ax = plt.subplots()
    for fac in FACTORS:
        d = summ[summ["factor"] == fac].sort_values("horizon")
        ax.plot(d["horizon"], d["mean_ic"], marker="o", lw=1.8, label=fac)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(list(HORIZONS))
    ax.set_title("IC decay: mean rank IC by forward horizon")
    ax.set_xlabel("forward horizon (months)")
    ax.set_ylabel("mean rank IC")
    ax.legend(ncol=3, fontsize=9)
    save_fig(fig, "04_ic_decay")


def main() -> None:
    f = read_parquet(settings.processed_dir / "factors.parquet")
    ic = compute_ic(f)
    summ = ic_summary(ic)
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    summ.to_csv(PROJECT_ROOT / "reports" / "ic_summary.csv", index=False)
    make_figures(ic, summ)

    logger.info("IC summary (all horizons):\n" + summ.to_string(index=False))
    logger.info("\n--- factor ranking by 1-month mean IC ---")
    r = summ[summ["horizon"] == 1].sort_values("mean_ic", ascending=False)
    logger.info("\n" + r[["factor", "mean_ic", "ic_ir", "t_stat", "hit_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
