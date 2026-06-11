"""Loughran-McDonald financial sentiment scoring for earnings transcripts.

The LM dictionaries (positive, negative, uncertainty, litigious, modal-strong,
modal-weak) are the academic standard for financial text — they're the
benchmark Diener, Loughran, & McDonald 2020 used to publish their 18,000+
citation paper, and most quant shops still use them as a baseline before
moving to more complex models.

Features returned per transcript (all normalised by total word count):

  lm_positive       fraction of positive-tone words
  lm_negative       fraction of negative-tone words
  lm_net            positive - negative   (signed sentiment)
  lm_uncertainty    fraction of uncertainty / hedging words
  lm_litigious      fraction of legal / regulatory words
  lm_modal_strong   fraction of strong modal verbs ("must", "will")
  lm_modal_weak     fraction of weak modal verbs ("might", "could")
  lm_word_count     total word count (control)

We compute these on the full transcript text. They're computationally free
(dictionary lookups, no model) and well-validated in the academic literature.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pysentiment2 as ps

from qfr.transcripts.pull import (
    TRANSCRIPTS_INDEX,
    load_transcript,
)
from qfr.utils.config import PROJECT_ROOT
from qfr.utils.logging import logger

LM_FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "transcripts_lm_features.parquet"
LM_FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)


# Strip the front-matter "Operator: Welcome to ..." etc. Optional but cleaner.
_PREAMBLE_RE = re.compile(r"^.*?(?:Operator|Good morning|Good afternoon|Welcome).*?\n", re.DOTALL | re.IGNORECASE)


def _clean_transcript(text: str) -> str:
    """Light cleanup: drop the very first line (usually metadata), normalise whitespace."""
    text = text.strip()
    # Don't actually strip the preamble — the call body is the signal-rich part
    # and LM is robust to a small amount of operator/welcome text.
    text = re.sub(r"\s+", " ", text)
    return text


def _lm_analyzer() -> ps.LM:
    """Cached singleton — LM loads its dictionaries lazily on first use."""
    if not hasattr(_lm_analyzer, "_inst"):
        _lm_analyzer._inst = ps.LM()
    return _lm_analyzer._inst


def lm_score(transcript: str) -> dict:
    """Compute LM features for one transcript."""
    lm = _lm_analyzer()
    text = _clean_transcript(transcript)
    tokens = lm.tokenize(text)
    word_count = len(tokens)
    if word_count == 0:
        return {k: 0.0 for k in ("lm_positive", "lm_negative", "lm_net",
                                 "lm_uncertainty", "lm_litigious",
                                 "lm_modal_strong", "lm_modal_weak")} | {"lm_word_count": 0}

    scores = lm.get_score(tokens)
    # pysentiment2 LM returns: Positive, Negative, Polarity, Subjectivity (counts not fractions)
    pos = scores.get("Positive", 0)
    neg = scores.get("Negative", 0)
    # We also want uncertainty, litigious, modal — go via raw dictionaries
    pos_pct = pos / word_count if word_count else 0.0
    neg_pct = neg / word_count if word_count else 0.0
    # The raw dictionary access:
    d = lm.lexicon
    n_uncertainty = sum(1 for t in tokens if d.get(t, {}).get("Uncertainty"))
    n_litigious = sum(1 for t in tokens if d.get(t, {}).get("Litigious"))
    n_modal_strong = sum(1 for t in tokens if d.get(t, {}).get("Strong_Modal"))
    n_modal_weak = sum(1 for t in tokens if d.get(t, {}).get("Weak_Modal"))
    return {
        "lm_positive": pos_pct * 100,    # express as percent
        "lm_negative": neg_pct * 100,
        "lm_net": (pos_pct - neg_pct) * 100,
        "lm_uncertainty": n_uncertainty / word_count * 100,
        "lm_litigious": n_litigious / word_count * 100,
        "lm_modal_strong": n_modal_strong / word_count * 100,
        "lm_modal_weak": n_modal_weak / word_count * 100,
        "lm_word_count": word_count,
    }


def score_all() -> pd.DataFrame:
    """Score every transcript in the index. Cached to LM_FEATURES_PARQUET."""
    if not TRANSCRIPTS_INDEX.exists():
        raise FileNotFoundError(f"No transcript index at {TRANSCRIPTS_INDEX}. "
                                f"Run qfr.transcripts.pull first.")
    index = pd.read_parquet(TRANSCRIPTS_INDEX)
    if not len(index):
        raise RuntimeError("Transcript index is empty.")

    rows: list[dict] = []
    n = len(index)
    for i, r in index.iterrows():
        text = load_transcript(r["symbol"], int(r["year"]), int(r["quarter"]))
        if not text:
            continue
        feats = lm_score(text)
        feats.update({
            "symbol": r["symbol"], "year": int(r["year"]), "quarter": int(r["quarter"]),
            "date": r["date"], "content_chars": int(r["content_chars"]),
        })
        rows.append(feats)
        if (i + 1) % 100 == 0:
            logger.info(f"  LM scoring: {i + 1}/{n}")
    df = pd.DataFrame(rows)
    df.to_parquet(LM_FEATURES_PARQUET, index=False)
    logger.info(f"LM features saved: {len(df):,} rows to {LM_FEATURES_PARQUET}")
    return df


def main() -> None:
    df = score_all()
    logger.info(f"\n=== LM feature distributions across {len(df)} transcripts ===")
    cols = ["lm_positive", "lm_negative", "lm_net", "lm_uncertainty",
            "lm_litigious", "lm_modal_strong", "lm_modal_weak"]
    logger.info(df[cols].describe().round(3).to_string())


if __name__ == "__main__":
    main()
