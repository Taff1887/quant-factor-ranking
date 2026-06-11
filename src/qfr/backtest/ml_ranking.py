"""Walk-forward ML ranking model — can a non-linear ranker beat the linear composite?

The linear composite in ``portfolio.py`` is an equal-weight average of factor
ranks. This module tests whether a gradient-boosted tree ranker (LightGBM /
XGBoost) extracts more cross-sectional information — *strictly out-of-sample,
under a walk-forward scheme*. If it can't beat the linear composite OOS, that's
itself a useful (and very common) finding: factor interactions are weak and the
linear model is hard to beat.

Timing / no-look-ahead convention
---------------------------------
Each observation is a (symbol, month-end ``date``). Features ``X_m`` are known
as of month-end ``m``; the target ``ret_fwd_1m[m]`` is the return realised over
the *following* month (m -> m+1). The portfolio decision for month ``m+1`` is
made at the end of month ``m`` using ``X_m``.

Walk-forward:
  - Expanding training window, refit every ``refit_every`` months.
  - At a refit anchored to test-block start ``t`` (index into the sorted month
    list), we train on every observation with month index ``<= t - 1 - embargo``.
    With ``embargo=1`` the last training label is for month ``t-2`` — strictly
    before the test block and with no calendar overlap with test-month returns.
  - We then predict every observation in months ``[t, t + refit_every)``.
  - Early-stopping validation uses the most recent ``valid_months`` of the
    training window (still strictly historical — no leakage).

Features are cross-sectionally percentile-ranked within each date so they are
stationary across time (the family composites already are; we rank the raw
momentum / volatility columns too for consistency). Tree models handle the
remaining NaNs (e.g. sparse sentiment) natively.

Run::  uv run python -m qfr.backtest.ml_ranking
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfr.backtest.portfolio import (
    ANN,
    DEFAULT_COST_PER_SIDE,
    _spearman_ic_monthly,
    capm,
    fetch_spy_monthly,
    perf_metrics,
    portfolio_monthly,
    summary_metrics_table,
)
from qfr.factors.build import build_factor_panel
from qfr.utils.config import PROJECT_ROOT, settings
from qfr.utils.logging import logger

FEATURES = ["value", "quality", "momentum", "growth", "risk", "sentiment",
            "size", "reversal", "mom_12_1", "vol_12m"]
TARGET = "ret_fwd_1m"

MIN_TRAIN_MONTHS = 60     # 5-year minimum initial training window
REFIT_EVERY = 12          # annual refit
EMBARGO_MONTHS = 1        # drop the last training label (airtight no-leakage)
VALID_MONTHS = 12         # most-recent slice of train used for early stopping
RANDOM_STATE = 7          # fixed for reproducibility


# --------------------------------------------------------------------------
# Feature prep: cross-sectional percentile rank within each date
# --------------------------------------------------------------------------
def cs_rank(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Percentile-rank each column within each date -> [0, 1]. NaNs stay NaN."""
    out = {}
    g = panel.groupby("date")
    for c in cols:
        out[c] = g[c].rank(pct=True)
    return pd.DataFrame(out, index=panel.index)


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------
def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=15, max_depth=4,
        min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=1.0,
        importance_type="gain",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )


def _xgb():
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=4,
        min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5.0, reg_alpha=1.0, gamma=0.0,
        random_state=RANDOM_STATE, n_jobs=-1,
        early_stopping_rounds=50, eval_metric="rmse",
    )


MODEL_FACTORIES = {"lgbm": _lgbm, "xgb": _xgb}


# --------------------------------------------------------------------------
# Walk-forward OOS prediction
# --------------------------------------------------------------------------
def walk_forward_oos(panel: pd.DataFrame, feature_cols: list[str],
                     model_name: str, *,
                     target: str = TARGET,
                     min_train_months: int = MIN_TRAIN_MONTHS,
                     refit_every: int = REFIT_EVERY,
                     embargo_months: int = EMBARGO_MONTHS,
                     valid_months: int = VALID_MONTHS,
                     ) -> tuple[pd.Series, pd.DataFrame]:
    """Return (oos_predictions aligned to panel.index, feature_importance_df).

    Predictions are NaN for the warmup period (first ``min_train_months``).

    The training target is the CROSS-SECTIONALLY DEMEANED forward return
    (each stock's ``ret_fwd_1m`` minus that month's universe mean). This focuses
    the model on relative cross-sectional performance rather than the common
    market move. It introduces no look-ahead — the demeaning uses only the
    contemporaneous cross-section of fully-realised training-month returns — and
    it does not change the IC evaluation, since ranking by demeaned return is
    identical to ranking by raw return within any month.
    """
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    X_all = panel[feature_cols]
    # Cross-sectionally demeaned training target
    y_all = (panel[target] - panel.groupby("date")[target].transform("mean"))

    preds = pd.Series(np.nan, index=panel.index, dtype=float)
    importances: list[pd.Series] = []

    t = min_train_months
    n_refits = 0
    while t < len(dates):
        test_block = dates[t: t + refit_every]
        # Training months: strictly before the embargo gap
        last_train_idx = t - 1 - embargo_months
        if last_train_idx < 10:
            t += refit_every
            continue
        train_months = dates[: last_train_idx + 1]

        train_mask = panel["date"].isin(train_months) & y_all.notna()
        Xtr, ytr = X_all[train_mask], y_all[train_mask]
        if len(Xtr) < 500:
            t += refit_every
            continue

        # Early-stopping validation = most-recent slice of the training window
        valid_cut = train_months[-valid_months] if len(train_months) > valid_months else train_months[0]
        in_valid = panel["date"][train_mask] >= valid_cut
        Xtr_fit, ytr_fit = Xtr[~in_valid.values], ytr[~in_valid.values]
        Xval, yval = Xtr[in_valid.values], ytr[in_valid.values]
        if len(Xval) < 200 or len(Xtr_fit) < 300:
            Xtr_fit, ytr_fit, Xval, yval = Xtr, ytr, None, None

        model = MODEL_FACTORIES[model_name]()
        is_lgbm = type(model).__module__.startswith("lightgbm")
        if is_lgbm:
            import lightgbm as lgb
            if Xval is not None:
                model.fit(Xtr_fit, ytr_fit, eval_set=[(Xval, yval)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                model.fit(Xtr_fit, ytr_fit)
        else:  # xgboost
            if Xval is not None:
                model.fit(Xtr_fit, ytr_fit, eval_set=[(Xval, yval)], verbose=False)
            else:
                model.set_params(early_stopping_rounds=None)
                model.fit(Xtr_fit, ytr_fit)

        test_mask = panel["date"].isin(test_block)
        if test_mask.any():
            preds.loc[test_mask] = model.predict(X_all[test_mask])

        imp = pd.Series(model.feature_importances_, index=feature_cols)
        importances.append(imp / imp.sum() if imp.sum() > 0 else imp)
        n_refits += 1
        t += refit_every

    logger.info(f"  [{model_name}] {n_refits} refits, "
                f"{preds.notna().sum():,} OOS predictions "
                f"({panel.loc[preds.notna(), 'date'].min().date()} -> "
                f"{panel.loc[preds.notna(), 'date'].max().date()})")
    imp_df = (pd.concat(importances, axis=1).mean(axis=1)
              .sort_values(ascending=False).rename("importance").to_frame()
              if importances else pd.DataFrame())
    return preds, imp_df


# --------------------------------------------------------------------------
# Evaluation: OOS IC table
# --------------------------------------------------------------------------
def oos_ic(panel: pd.DataFrame, pred_col: str, target: str = TARGET) -> dict:
    s = _spearman_ic_monthly(panel, pred_col, target).dropna()
    n = len(s)
    sd = s.std()
    return {
        "mean_ic_%": s.mean() * 100,
        "ic_ir": (s.mean() / sd) if sd > 0 else np.nan,
        "t_stat": (s.mean() / sd * np.sqrt(n)) if sd > 0 else np.nan,
        "hit_rate_%": (s > 0).mean() * 100,
        "n_months": n,
    }


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def _save(fig, name):
    from qfr.utils.viz import save_fig
    save_fig(fig, name)


def chart_ml_cumulative(strats: dict[str, dict], spy: pd.Series, cps: float) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    colors = {"LightGBM": "#1f3a5f", "XGBoost": "#0a7a3a", "Linear composite": "#bc4a3c"}
    for label, s in strats.items():
        net = (s["gross"] - s["total_traded"] * cps).fillna(0)
        cum = (1 + net).cumprod()
        ax.plot(cum.index, cum.values, lw=1.9, color=colors.get(label, "#888"), label=label)
    spy_a = spy.reindex(next(iter(strats.values()))["gross"].index).fillna(0)
    ax.plot(spy_a.index, (1 + spy_a).cumprod().values, lw=1.6, color="#222", ls="--", label="SPY (TR)")
    ax.set_yscale("log")
    ax.set_title("Walk-forward ML ranking vs linear composite — top-quintile CW, OOS, net of 10 bps/side",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, "ml_cumulative")


def chart_ml_ic(ic_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(ic_df))
    a1.bar(x, ic_df["mean_ic_%"], color="#1f3a5f", alpha=0.85)
    for i, v in enumerate(ic_df["mean_ic_%"]):
        a1.text(i, v + 0.02, f"{v:.2f}%", ha="center", fontsize=9)
    a1.set_xticks(x); a1.set_xticklabels(ic_df.index, fontsize=9)
    a1.set_ylabel("Mean OOS rank IC (%)"); a1.axhline(0, color="#444", lw=0.7)
    a1.set_title("OOS rank IC (1m)", fontsize=11, fontweight="bold", loc="left")
    a2.bar(x, ic_df["t_stat"], color="#0a7a3a", alpha=0.85)
    for i, v in enumerate(ic_df["t_stat"]):
        a2.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    a2.set_xticks(x); a2.set_xticklabels(ic_df.index, fontsize=9)
    a2.set_ylabel("t-stat of IC"); a2.axhline(1.96, color="#bc4a3c", ls="--", lw=0.7, alpha=0.7)
    a2.set_title("OOS IC t-stat", fontsize=11, fontweight="bold", loc="left")
    fig.suptitle("Out-of-sample signal quality — ML rankers vs linear composite (same OOS window)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "ml_ic_comparison")


def chart_feature_importance(imp_lgbm: pd.DataFrame, imp_xgb: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from qfr.utils.viz import set_plot_style
    set_plot_style()
    feats = imp_lgbm.index.tolist()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(feats) + 1)))
    y = np.arange(len(feats))
    ax.barh(y - 0.2, imp_lgbm["importance"].values * 100, 0.4, color="#1f3a5f", alpha=0.85, label="LightGBM")
    xgb_aligned = imp_xgb.reindex(feats)["importance"].values * 100
    ax.barh(y + 0.2, xgb_aligned, 0.4, color="#0a7a3a", alpha=0.85, label="XGBoost")
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("Mean gain importance (%, normalised, avg across refits)")
    ax.set_title("Feature importance — averaged across walk-forward refits", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "ml_feature_importance")


def chart_shap(panel: pd.DataFrame, feature_cols: list[str]) -> None:
    """SHAP summary on a model fit on the full panel (explainability only —
    NOT used for any OOS metric)."""
    import matplotlib.pyplot as plt
    import shap
    from qfr.utils.viz import save_fig
    d = panel.dropna(subset=[TARGET])
    model = _lgbm()
    model.set_params(n_estimators=300)
    model.fit(d[feature_cols], d[TARGET])
    expl = shap.TreeExplainer(model)
    sample = d[feature_cols].sample(min(4000, len(d)), random_state=RANDOM_STATE)
    sv = expl.shap_values(sample)
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, sample, show=False, plot_size=None)
    fig = plt.gcf()
    fig.suptitle("SHAP — feature contributions to predicted forward return (full-sample fit)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "ml_shap_summary")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main() -> None:
    panel = build_factor_panel().sort_values(["date", "symbol"]).reset_index(drop=True)

    # Cross-sectionally rank features within each date (stationary, comparable)
    ranks = cs_rank(panel, FEATURES)
    for c in FEATURES:
        panel[c] = ranks[c]
    # Linear baseline = equal-weight mean of the ranked features (nanmean)
    panel["pred_linear"] = panel[FEATURES].mean(axis=1)

    logger.info("Running walk-forward LightGBM...")
    panel["pred_lgbm"], imp_lgbm = walk_forward_oos(panel, FEATURES, "lgbm")
    logger.info("Running walk-forward XGBoost...")
    panel["pred_xgb"], imp_xgb = walk_forward_oos(panel, FEATURES, "xgb")

    # Restrict every comparison to the OOS window (where ML has predictions)
    oos = panel[panel["pred_lgbm"].notna() & panel["pred_xgb"].notna()].copy()
    logger.info(f"OOS comparison window: {oos['date'].min().date()} -> {oos['date'].max().date()} "
                f"({oos['date'].nunique()} months, {len(oos):,} obs)")

    # --- OOS IC table ---
    ic_rows = {
        "LightGBM": oos_ic(oos, "pred_lgbm"),
        "XGBoost": oos_ic(oos, "pred_xgb"),
        "Linear composite": oos_ic(oos, "pred_linear"),
    }
    ic_df = pd.DataFrame(ic_rows).T

    # --- Portfolio backtests (top quintile CW) on the OOS window ---
    spy = fetch_spy_monthly()
    strat_defs = [
        ("LightGBM", "pred_lgbm"), ("XGBoost", "pred_xgb"),
        ("Linear composite", "pred_linear"),
    ]
    cum_strats: dict[str, dict] = {}
    summary_rows = []
    for label, col in strat_defs:
        for n_buckets, bname in [(5, "Q"), (10, "D")]:
            for weight in ["equal", "cap"]:
                s = portfolio_monthly(oos, col, n_buckets=n_buckets, weight=weight,
                                      name=f"{label} top-{bname} {weight[:2].upper()}")
                m = summary_metrics_table([s], spy)
                summary_rows.append(m)
                if n_buckets == 5 and weight == "cap":
                    cum_strats[label] = s
    summary = pd.concat(summary_rows, ignore_index=True)

    # --- Save artefacts ---
    out = PROJECT_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(out / "ml_oos_ic.csv")
    summary.to_csv(out / "ml_portfolio_summary.csv", index=False)
    imp_lgbm.to_csv(out / "ml_feature_importance_lgbm.csv")
    imp_xgb.to_csv(out / "ml_feature_importance_xgb.csv")

    # --- Charts ---
    chart_ml_cumulative(cum_strats, spy, DEFAULT_COST_PER_SIDE)
    chart_ml_ic(ic_df)
    chart_feature_importance(imp_lgbm, imp_xgb)
    try:
        chart_shap(panel, FEATURES)
    except Exception as e:
        logger.warning(f"SHAP chart skipped: {str(e)[:120]}")

    # --- Console summary ---
    logger.info(f"\n=== OOS rank IC (same window, 1m forward) ===\n{ic_df.round(3).to_string()}")
    logger.info(f"\n=== Feature importance (LightGBM, avg across refits) ===\n{(imp_lgbm*100).round(1).to_string()}")
    logger.info(f"\n=== Portfolio summary (net of 10 bps/side, OOS, vs SPY) ===\n"
                f"{summary.round(2).to_string(index=False)}")


if __name__ == "__main__":
    main()
