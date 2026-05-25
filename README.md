# Quant Factor Ranking — Cross-Sectional ML Multi-Factor Equity Model (S&P 500)

> Can a machine-learning *ranking* model improve cross-sectional stock selection
> over traditional, equally-weighted multi-factor approaches?

This repository is an end-to-end, institution-style quantitative equities
research project. It builds a **point-in-time, survivorship-bias-free** dataset
for the S&P 500, engineers a library of academically-motivated factors,
validates them with information-coefficient analysis, and then tests whether
**learning-to-rank** models (LightGBM, XGBoost) can out-select a transparent
traditional factor benchmark — all under walk-forward validation with explicit
controls for look-ahead bias and overlapping-horizon leakage.

The goal is not to predict exact returns. It is to predict **relative ordering**:
*which stocks should rank higher than their peers over the next 1–3 months?*

---

## Research questions

1. Do classic factors (Value, Quality, Momentum, Growth, Risk) carry
   cross-sectional information for S&P 500 names, and how strong/stable is it?
2. Can a non-linear ranking model exploit factor *interactions* that an
   equal-weight composite cannot?
3. Does any edge survive transaction costs, turnover, and honest out-of-sample
   testing?

## Methodology highlights

- **Point-in-time universe.** S&P 500 membership is reconstructed as-of every
  rebalance date from FMP historical constituent changes — no survivorship bias.
- **Look-ahead discipline.** Fundamentals are lagged to their filing date; a
  signal is only usable once the underlying filing was public.
- **Learning-to-rank.** Targets are cross-sectional return deciles; models
  optimise ranking objectives (`lambdarank` / `rank:pairwise`), grouped by date.
- **Leakage-aware validation.** Walk-forward with a purge gap equal to the
  forecast horizon plus an embargo, to neutralise overlapping-label leakage.
- **Honest benchmark.** ML is compared against an equal-weight composite
  z-score model and a Ridge cross-sectional regression.
- **Explainability.** Feature importance and SHAP throughout.

## Repository structure

```
quant-factor-ranking/
├── data/                  # raw / processed / external (gitignored; reproducible)
├── notebooks/             # 01..07 staged research narrative
├── src/qfr/               # importable library
│   ├── data/              # FMP client, point-in-time universe, prices, fundamentals
│   ├── factors/           # factor construction
│   ├── validation/        # IC / rank-IC analysis
│   ├── models/            # baselines + LightGBM/XGBoost rankers
│   ├── portfolio/         # decile / long-short formation
│   ├── backtest/          # walk-forward, cost-aware engine
│   └── utils/             # config, logging, IO
├── charts/                # publication-quality figures
├── paper/                 # research_paper.md (full write-up)
├── pyproject.toml
└── requirements.txt
```

## The research narrative — `notebooks/research.ipynb`

The whole study lives in **one consolidated, executed notebook** so you can open a single file and
read the entire story top to bottom. Each stage carries its own *data collection* and *data
validation* before analysis. Heavy computation lives in `src/qfr/`; the notebook loads and displays
the results.

| Section | Stage |
|---|---|
| 1 | Data collection & the point-in-time dataset |
| 1b | Independent data validation (FMP vs Yahoo Finance) |
| 2 | Exploratory data analysis |
| 3–8 | Factors → validation → models → portfolio → paper *(in progress)* |

Pipeline modules: `qfr.data.collect` → `qfr.data.assemble` → `qfr.data.validate_yahoo` →
`qfr.data.clean` → `qfr.eda`.

## Quick start

```bash
# 1. Install uv (https://docs.astral.sh/uv/)
# 2. Install the environment (everything needed to run the notebooks)
uv sync

# 3. Add your FMP key
cp .env.example .env      # then edit .env and set FMP_API_KEY

# 4. Launch the research environment
uv run jupyter lab
```

**Data source:** [Financial Modeling Prep](https://financialmodelingprep.com/)
(Premium/Ultimate recommended for historical constituents + deep fundamentals).

## Status

🚧 Work in progress — built progressively, stage by stage. See `paper/research_paper.md`
for the evolving write-up and the notebooks for the full narrative.

## Disclaimer

This is a research and educational project. Nothing here is investment advice.
