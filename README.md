# Quant Factor Ranking

Cross-sectional multi-factor equity research on the point-in-time S&P 500 (2010–2026), with a cross-market extension to the ASX 200. Pulls fundamentals + prices from Financial Modeling Prep, screens 26 factors, builds an equal-weight 5-factor composite, and runs the usual battery of long-only / long-short / sector-neutral diagnostics.

**Full research write-up:** see [`report.md`](report.md).

---

## Headline result

| Universe | Strategy | CAGR | Sharpe | CAPM α |
|---|---|---|---|---|
| S&P 500 | Top quintile cap-weighted, net of 10 bps/side | 16.9 % | 1.06 | **+1.65 %** vs SPY |
| ASX 200 (current-listed)¹ | Top quintile CW (ff-adj), net of 25 bps/side | 13.1 % | 0.86 | **+6.01 %** vs IOZ |

¹ ASX results are upper-bound estimates inflated by survivorship bias — see [`report.md` §13](report.md#13-cross-market-extension--asx-200) for full disclosure.

The 5-factor composite (ROIC + ROE + FCF yield + revenue growth + EPS growth, equal-weight z-scores) delivers a Spearman rank IC of 2.58 % at 3-month horizon (t = 4.41) on S&P 500, and 8.72 % (t = 11.21) on ASX 200 — same methodology, dramatically different market efficiency.

---

## Quick start

Requires Python 3.13 + [uv](https://docs.astral.sh/uv/) + an [FMP](https://site.financialmodelingprep.com) API key in `.env`.

```bash
uv sync                                          # install deps
cp .env.example .env && vi .env                  # add FMP_API_KEY

# Data layer (one-time, ~30 min for full S&P 500 pull)
uv run python -m qfr.data.collect
uv run python -m qfr.data.validate_yahoo
uv run python -m qfr.data.clean

# Factor panel + per-factor tearsheets
uv run python -m qfr.factors.build
uv run python -m qfr.validation.factor_report
uv run python -m qfr.validation.factor_screen

# Portfolio backtest + composite IC + diagnostics
uv run python -m qfr.backtest.portfolio

# Optional: composite weighting variants (EW vs IC-IR vs t²-shrunk)
uv run python -m qfr.backtest.composite_variants

# Optional: ASX 200 cross-market extension
uv run python -m qfr.backtest.asx_pull_data
uv run python -m qfr.backtest.asx_assemble
uv run python -m qfr.backtest.asx_extension

# Optional: walk-forward ML ranking (LightGBM/XGBoost vs linear composite)
uv run python -m qfr.backtest.ml_ranking
```

---

## Repo structure

```
src/qfr/
  data/                Data pipeline: universe, prices, fundamentals, estimates,
                       Yahoo cross-validation, cleaning, panel assembly
  factors/             Factor construction (winsorisation, cross-sectional z-scores,
                       7 family composites + 26 standalone factors)
  validation/          Per-factor tearsheets (IC, IC decay, deciles, Fama-MacBeth,
                       cumulative returns, drawdowns) + the cross-section screen
  backtest/            Portfolio construction & diagnostics:
                         portfolio.py             core S&P 500 backtest + IC
                         composite_variants.py    factor-weighting variants
                         asx_*.py                 ASX 200 cross-market extension
  utils/               IO, config, plotting, logging

data/                  Cached raw FMP responses + processed panels (gitignored)
charts/                Generated PNGs (factor tearsheets + portfolio plots)
reports/               Generated CSVs (screen tables, backtest summaries)
report.md              Full research walkthrough (this is where the writing lives)
```

---

## What's in [`report.md`](report.md)

1. Data pipeline — point-in-time S&P 500 reconstruction (1,049 unique constituents), SEC filing-date-lagged fundamentals, Yahoo Finance cross-validation
2. Validation & cleaning
3. Factor construction (26 factors across value / quality / momentum / growth / risk / sentiment / size)
4. Per-factor testing (rank IC, decay, deciles, Fama-MacBeth)
5. Significance & screening
6. Composite construction + composite rank IC + factor-weighting variants
7. Long-only results (top quintile / decile, EW / CW)
8. Long-short diagnostics (dollar-neutral, beta-neutral, sector-neutral, cost sensitivity, turnover)
9. Why the long-short underwhelms
10. Next steps to make the long-short investable
11. Honest caveats
12. Reproduce instructions
13. Cross-market extension to ASX 200 (with survivorship-bias disclosure)

---

*Python 3.13 / uv / pandas / matplotlib. Data layer: FMP stable API + Yahoo Finance (free-float). Not investment advice; this is a showcase research project.*
