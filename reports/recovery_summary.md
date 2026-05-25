# Missing Early-Year Price Coverage — Investigation & Recovery Summary

**Date:** 2026-05-25 · **Scope:** point-in-time S&P 500, 2000-01 → 2026-04 · **Sources tried:** FMP
`dividend-adjusted`, FMP `full` (raw), Yahoo Finance (`yfinance`).

## The problem
The point-in-time universe correctly has ~500 members every month, but FMP's price API does not
serve many old/delisted/renamed tickers. **25,589 member-months (362 tickers)** lack usable price.

## Why they are missing (diagnosis)
| Cause | Tickers | Notes |
|---|---|---|
| No FMP price at all (delisted) | 170 | e.g. C.R. Bard, Yahoo!, Tyco, Safeway, Rowan, Enron, WorldCom, Lehman. Neither FMP nor Yahoo serve them. |
| Recycled ticker (FMP serves a *different* company) | 124 | e.g. APC→ARKO, CA→an ETF, EMC→an ETF, ALTR→Altair, STI→Solidion. Data exists but in the wrong (post-delisting) window. |
| Partial FMP coverage (left index) | 51 | FMP's adjusted history is truncated; early years missing. |
| Partial FMP coverage (still in index) | 17 | e.g. current members whose FMP adjusted history is shallow. |

The key safeguard: a recovered series is only accepted if it **overlaps the ticker's actual
membership window**. This automatically rejects recycled tickers (their data starts after delisting).

## Recovery results
| Metric | Value |
|---|---|
| Missing ticker-months (before) | **25,589** |
| Recovered (Yahoo, overlap-validated, clipped to membership) | **3,909 (15.3%)** |
| Still missing | **21,680 (84.7%)** |
| Tickers safely recovered | **40** (36 high-confidence, 4 medium) |
| Best source | **Yahoo Finance** — the only usable source |
| FMP `full` (raw) | 68 tickers overlapped, but **not used** (raw ≠ dividend-adjusted → would corrupt total returns) |

**Recovery by year (share of that year's missing that was recovered):** 2000 = 3.5%, 2003 = 5.2%,
2005 = 7.2%, 2008 = 7.7%, 2010 = 8.0%, then declining to 0% after 2013. Recovery is concentrated in
the **mid-2000s–2012**, not the earliest years, and adds only ~8 names/month in 2000.

## Did early-year coverage materially improve? **No.**
2000 investable goes from ~266 to ~274 names/month; 2000–2005 remains ~45–55% of the universe
missing. See `charts/recovery_before_after.png` — the "after" line is barely above "before".

## Is 2000–2005 viable? **Not for robust cross-sectional work.**
Recommendation: use **2008+ (ideally 2010+)** as the primary high-coverage sample (price coverage
≥80% from 2008, ≥90% from 2015, ~100% from 2020). Keep the full 2000–2026 history for supplementary
/ robustness reporting only, with the coverage caveat stated.

## Risks / assumptions introduced by recovery
- **Vendor mixing:** recovered cells use Yahoo's adjusted prices while the rest use FMP. The two
  differ by a small, mostly-systematic amount (~0.5%/month for high-dividend names; see Part 1b).
  If integrated, a `price_source` flag should mark Yahoo-sourced cells.
- **Reused tickers handled correctly:** for tickers with two membership spells (e.g. `Q` = Qwest then
  Qnity), only the spell Yahoo actually covers is filled; the old spell stays missing.
- **No contamination:** recovered series are clipped to ≤ the last membership month (never extended
  past delisting); recycled tickers are rejected by the overlap test.
- The bulk of missing data (84.7%) is genuinely unavailable from FMP or Yahoo and would require a
  survivorship-bias-free vendor (CRSP / Bloomberg / Refinitiv).

## Recommendation
1. **Do not** force-fill. Treat the 40 recoveries as an optional, clearly-flagged enhancement.
2. Run the **primary** analysis on **2010+** (or 2008+) where coverage is ~90–100% and unbiased.
3. Report 2000–2009 only as a secondary robustness window with the coverage caveat.

## Artifacts
`reports/missing_price_coverage_by_year.csv`, `reports/missing_ticker_detail.csv`,
`reports/recovered_ticker_mapping.csv`, `reports/recovery_attempt_log.csv`,
`charts/recovery_before_after.png`, `data/processed/recovered_prices.parquet` (gitignored).
