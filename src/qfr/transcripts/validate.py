"""Validation: do transcript-derived features predict forward returns?

For each LM feature (free dictionary) and each LLM feature (Haiku 4.5), we
compute the Spearman rank IC of the feature score vs the realised return over
the period AFTER the earnings call (1m / 3m / 6m forward windows).

Decision rule (set in advance, not after seeing the numbers):
  - If at least one feature reaches |t-stat| >= 2.0 with a sign that makes
    economic sense (positive sentiment -> positive forward return), we
    declare the experiment a 'go' and expand to the full universe.
  - Otherwise we record the negative result, save the spend, and move on.

Aligning to forward returns:
  Transcript date is the EARNINGS CALL date (typically a day or two after
  fiscal period end, possibly intra-month). We use the realised price return
  from the earnings call date through 1/3/6 months later, sourced from FMP
  dividend-adjusted closes for the same tickers.

Output:
  reports/transcripts_validation.csv     IC + t-stat per feature per horizon
  charts/transcripts_validation.png      bar chart summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.data.fmp_client import FMPClient
from qfr.transcripts.llm_features import LLM_FEATURES_PARQUET, FEATURE_SCHEMA
from qfr.transcripts.lm_features import LM_FEATURES_PARQUET
from qfr.utils.config import PROJECT_ROOT
from qfr.utils.logging import logger

VALIDATION_CSV = PROJECT_ROOT / "reports" / "transcripts_validation.csv"
VALIDATION_CHART = "transcripts_validation"

LM_FEATURE_COLS = ["lm_positive", "lm_negative", "lm_net", "lm_uncertainty",
                   "lm_litigious", "lm_modal_strong", "lm_modal_weak"]
LLM_NUMERIC_FEATURES = [k for k in FEATURE_SCHEMA
                        if FEATURE_SCHEMA[k]["type"] in ("int", "float")]
LLM_CATEGORICAL_FEATURES = [k for k in FEATURE_SCHEMA
                            if FEATURE_SCHEMA[k]["type"] == "str"]

HORIZONS_MONTHS = (1, 3, 6)


# --------------------------------------------------------------------------
# Forward return lookup
# --------------------------------------------------------------------------
def fetch_prices(symbols: list[str]) -> pd.DataFrame:
    """Daily dividend-adjusted closes for each transcript symbol."""
    c = FMPClient()
    frames = []
    for sym in symbols:
        try:
            data = c.historical_prices(sym, from_date="2019-06-01",
                                       to_date="2024-12-31",
                                       series="dividend-adjusted")
            df = pd.DataFrame(data)
            if not len(df):
                continue
            df["symbol"] = sym
            frames.append(df[["symbol", "date", "adjClose"]])
        except Exception as e:
            logger.warning(f"  prices fail for {sym}: {str(e)[:60]}")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def add_forward_returns(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """For each transcript date, look up the close on that date (or the most
    recent prior trading day) and the close at +1m/+3m/+6m. Then compute returns."""
    f = features.copy()
    f["date"] = pd.to_datetime(f["date"])

    p = prices.sort_values(["symbol", "date"]).copy()
    # asof match: transcript date -> latest close on/before that date
    f = f.sort_values("date")
    base = pd.merge_asof(
        f[["symbol", "date"]], p, by="symbol", on="date", direction="backward"
    ).rename(columns={"adjClose": "base_close"})
    f = f.merge(base[["symbol", "date", "base_close"]], on=["symbol", "date"], how="left")

    for m in HORIZONS_MONTHS:
        target_dates = f.copy()
        target_dates["lookup_date"] = target_dates["date"] + pd.DateOffset(months=m)
        target_dates = target_dates.sort_values("lookup_date").rename(
            columns={"lookup_date": "date_lookup"})
        # Forward asof
        joined = pd.merge_asof(
            target_dates[["symbol", "date_lookup"]].rename(columns={"date_lookup": "date"}),
            p, by="symbol", on="date", direction="backward"
        ).rename(columns={"adjClose": f"close_{m}m"})
        f = f.merge(joined[["symbol", f"close_{m}m"]].assign(_idx=range(len(joined))),
                    left_index=True, right_on="_idx", how="left").drop(columns="_idx")
        f[f"ret_fwd_{m}m"] = f[f"close_{m}m"] / f["base_close"] - 1.0
    return f


# --------------------------------------------------------------------------
# IC computation
# --------------------------------------------------------------------------
def ic_table(features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """For each (feature, horizon), compute Spearman rank IC + t-stat across
    all (symbol, date) observations (pooled — no monthly grouping since transcripts
    are scattered across dates, not aligned to month-ends)."""
    rows = []
    for col in feature_cols:
        for m in HORIZONS_MONTHS:
            ret_col = f"ret_fwd_{m}m"
            sub = features[[col, ret_col]].dropna()
            if len(sub) < 30:
                rows.append({"feature": col, "horizon_m": m, "n": len(sub),
                             "ic": np.nan, "t": np.nan, "hit_rate_%": np.nan})
                continue
            ic = sub[col].corr(sub[ret_col], method="spearman")
            # t-stat assuming IID pooled obs
            n = len(sub)
            t = ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2) if abs(ic) < 1 else np.nan
            # Hit rate: share of obs where (feature - feature.median) has same sign as return
            f_centered = sub[col] - sub[col].median()
            r_centered = sub[ret_col]
            hit = ((f_centered > 0) == (r_centered > 0)).mean() * 100
            rows.append({"feature": col, "horizon_m": m, "n": n,
                         "ic": ic * 100, "t": t, "hit_rate_%": hit})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
def chart_validation(ic_df: pd.DataFrame, out_name: str = VALIDATION_CHART) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import save_fig, set_plot_style
    set_plot_style()
    df = ic_df.dropna(subset=["ic"]).copy()
    df["abs_t"] = df["t"].abs()
    feats = df.groupby("feature")["abs_t"].max().sort_values(ascending=True).index
    horizons = sorted(df["horizon_m"].unique())
    colors = {1: "#1f3a5f", 3: "#0a7a3a", 6: "#bc4a3c"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, max(6, 0.35 * len(feats) + 1.5)),
                                  sharey=True)
    y = np.arange(len(feats))
    width = 0.25
    for i, m in enumerate(horizons):
        sub = df[df["horizon_m"] == m].set_index("feature").reindex(feats)
        a1.barh(y + (i - 1) * width, sub["ic"], width,
                color=colors.get(m, "#888"), alpha=0.85, label=f"{m}m")
        a2.barh(y + (i - 1) * width, sub["t"], width,
                color=colors.get(m, "#888"), alpha=0.85, label=f"{m}m")
    for ax, title in [(a1, "Spearman IC (%) vs forward returns"),
                       (a2, "t-stat of IC")]:
        ax.set_yticks(y)
        ax.set_yticklabels(feats, fontsize=9)
        ax.axvline(0, color="#444", lw=0.7)
        ax.grid(alpha=0.3, axis="x")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.legend(title="Horizon", fontsize=9, loc="lower right")
    a2.axvline(1.96, color="#bc4a3c", ls="--", lw=0.7, alpha=0.6)
    a2.axvline(-1.96, color="#bc4a3c", ls="--", lw=0.7, alpha=0.6)
    fig.suptitle("Transcript-derived features — predictive power vs forward S&P 500 returns",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_name)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main() -> None:
    # Load features
    if not LM_FEATURES_PARQUET.exists():
        logger.warning("No LM features. Run qfr.transcripts.lm_features first.")
        return
    lm = pd.read_parquet(LM_FEATURES_PARQUET)
    logger.info(f"LM features: {len(lm)} rows, {lm['symbol'].nunique()} symbols")

    llm = pd.DataFrame()
    if LLM_FEATURES_PARQUET.exists():
        llm = pd.read_parquet(LLM_FEATURES_PARQUET)
        logger.info(f"LLM features: {len(llm)} rows, {llm['symbol'].nunique()} symbols")
    else:
        logger.warning("No LLM features yet — set ANTHROPIC_API_KEY and run "
                       "qfr.transcripts.llm_features to add LLM scoring.")

    # Combine on (symbol, year, quarter, date)
    if len(llm):
        merged = lm.merge(llm, on=["symbol", "year", "quarter", "date"],
                          how="outer", suffixes=("_lm", ""))
    else:
        merged = lm

    # Forward returns
    symbols = sorted(merged["symbol"].unique())
    logger.info(f"Fetching prices for {len(symbols)} symbols...")
    prices = fetch_prices(symbols)
    enriched = add_forward_returns(merged, prices)

    # IC per feature
    feature_cols = LM_FEATURE_COLS.copy()
    for c in LLM_NUMERIC_FEATURES:
        if c in enriched.columns:
            feature_cols.append(c)
    available_cols = [c for c in feature_cols if c in enriched.columns]
    ic_df = ic_table(enriched, available_cols)

    # Save + chart
    VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(VALIDATION_CSV, index=False)
    chart_validation(ic_df)

    logger.info(f"\n=== Transcript feature validation (n={len(enriched)} transcripts) ===\n"
                f"{ic_df.round(3).to_string(index=False)}")

    # Decision summary
    significant = ic_df.dropna(subset=["t"]).query("abs(t) >= 2.0")
    if len(significant):
        logger.info(f"\n*** {len(significant)} feature-horizon pairs have |t| >= 2.0: ***")
        logger.info(significant.round(3).to_string(index=False))
        logger.info("*** Decision: signal present — escalate to full universe ***")
    else:
        max_t = ic_df["t"].abs().max()
        logger.info(f"\n*** No feature-horizon pairs reach |t| >= 2.0 "
                    f"(best |t| = {max_t:.2f}). Decision: negative result. ***")


if __name__ == "__main__":
    main()
