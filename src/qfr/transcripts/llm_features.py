"""LLM-based structured feature extraction from earnings transcripts.

Uses Claude Haiku 4.5 to extract a small set of forward-looking, quant-relevant
features per transcript. Each call returns a JSON object that we cache to disk
so re-runs are free.

Features extracted (the "signal hypothesis"):

  forward_guidance_tone        1-10: how optimistic mgmt is about NEXT quarter
  mgmt_confidence              1-10: based on hedging language / qualifiers
  demand_strength              1-10: described demand environment
  cost_pressure                -5 to +5: margin pressure tone (- = headwinds)
  risk_mentions_count          int: explicit macro / competitive / regulatory risks raised
  capital_return_stance        category: "buyback", "dividend", "capex", "balanced", "deleveraging"

Each is anchored to specific calibration prompts so the LLM produces consistent
scoring across transcripts.

Cost estimate at the validation sample size (~480 transcripts):
  ~6M input tokens at $1/M Haiku 4.5 = ~$6
  ~250K output tokens at $5/M = ~$1.25
  Total: ~$7-10
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from qfr.transcripts.pull import (
    TRANSCRIPTS_INDEX,
    load_transcript,
)
from qfr.utils.config import PROJECT_ROOT
from qfr.utils.logging import logger

LLM_FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "transcripts_llm_features.parquet"
LLM_FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)

LLM_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "transcripts" / "_llm_features"
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model: Haiku 4.5 — small, fast, cheap enough to run on hundreds of transcripts.
# Use Sonnet 4.6 if validation passes and we want to scale up quality.
MODEL_ID = "claude-haiku-4-5"

# Cap transcript length to keep cost predictable. ~30K chars ≈ ~7500 tokens.
# Full transcripts are 40-60K chars; the prepared remarks + Q&A first half
# captures the bulk of the forward-guidance signal.
MAX_TRANSCRIPT_CHARS = 30_000

# Schema we ask the LLM to populate. JSON Schema-like, used in the prompt.
FEATURE_SCHEMA = {
    "forward_guidance_tone": {
        "type": "int", "range": "1-10",
        "desc": "1=very pessimistic about next quarter, 10=very optimistic. "
                "Anchor on explicit guidance phrases like 'we expect', 'we project'."},
    "mgmt_confidence": {
        "type": "int", "range": "1-10",
        "desc": "1=lots of hedging ('might', 'could', 'depending on'), "
                "10=direct/unqualified ('we will', 'we are confident')."},
    "demand_strength": {
        "type": "int", "range": "1-10",
        "desc": "1=very weak described demand, 10=very strong described demand."},
    "cost_pressure": {
        "type": "int", "range": "-5 to +5",
        "desc": "-5=severe margin headwinds described, 0=neutral, +5=margin tailwinds. "
                "Score the NET tone of cost / margin commentary."},
    "risk_mentions_count": {
        "type": "int", "range": "0-20",
        "desc": "Count of explicit risks raised: macro, competitive, regulatory, "
                "supply chain, geopolitical, etc."},
    "capital_return_stance": {
        "type": "str",
        "range": "one of: buyback / dividend / capex / balanced / deleveraging / unspecified",
        "desc": "The dominant capital-allocation message in management commentary."},
}

SYSTEM_PROMPT = (
    "You are a quant equity analyst reading earnings-call transcripts to "
    "extract a small set of structured features. You must return ONLY a JSON "
    "object that matches the requested schema — no prose, no preamble. "
    "Be consistent in your scoring across calls: anchor on the SAME meaning of "
    "each score across every transcript."
)

PROMPT_TEMPLATE = """Extract the following features from this earnings-call transcript and return them as a JSON object.

{schema}

If the transcript is missing context for a feature (e.g. no forward guidance), use the neutral midpoint of the range and set it. Do not return null.

Transcript:

{transcript}

Return only the JSON object."""


def _build_prompt(transcript: str) -> str:
    schema_lines = []
    for k, v in FEATURE_SCHEMA.items():
        schema_lines.append(f'  "{k}" ({v["range"]}): {v["desc"]}')
    schema_str = "\n".join(schema_lines)
    truncated = transcript[:MAX_TRANSCRIPT_CHARS]
    return PROMPT_TEMPLATE.format(schema=schema_str, transcript=truncated)


def _cache_path(symbol: str, year: int, quarter: int) -> Path:
    return LLM_CACHE_DIR / f"{symbol}_{year}_Q{quarter}.json"


def _get_client():
    """Lazy import + key check so the module loads without the API key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env (or export it) "
            "before running LLM extraction.")
    from anthropic import Anthropic
    return Anthropic(api_key=key)


def extract_one(symbol: str, year: int, quarter: int, *,
                force: bool = False) -> dict | None:
    """Extract features for one transcript. Cached to disk."""
    path = _cache_path(symbol, year, quarter)
    if path.exists() and not force:
        with open(path) as f:
            return json.load(f)

    text = load_transcript(symbol, year, quarter)
    if not text:
        return None

    client = _get_client()
    prompt = _build_prompt(text)

    try:
        resp = client.messages.create(
            model=MODEL_ID,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip code fences if the model wrapped JSON in them
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip("` \n")
        parsed = json.loads(raw)
    except Exception as e:
        logger.warning(f"  {symbol} {year}Q{quarter} LLM error: {str(e)[:100]}")
        return None

    parsed["_input_tokens"] = resp.usage.input_tokens
    parsed["_output_tokens"] = resp.usage.output_tokens
    with open(path, "w") as f:
        json.dump(parsed, f)
    return parsed


def extract_all(*, max_to_run: int | None = None) -> pd.DataFrame:
    """Run LLM extraction on every transcript in the index. Cached per-call."""
    if not TRANSCRIPTS_INDEX.exists():
        raise FileNotFoundError(f"No transcript index. Run qfr.transcripts.pull first.")
    index = pd.read_parquet(TRANSCRIPTS_INDEX)

    rows: list[dict] = []
    total_in = total_out = 0
    n_done = 0
    for _, r in index.iterrows():
        if max_to_run and n_done >= max_to_run:
            break
        feats = extract_one(r["symbol"], int(r["year"]), int(r["quarter"]))
        if not feats:
            continue
        total_in += feats.pop("_input_tokens", 0)
        total_out += feats.pop("_output_tokens", 0)
        feats.update({
            "symbol": r["symbol"], "year": int(r["year"]), "quarter": int(r["quarter"]),
            "date": r["date"],
        })
        rows.append(feats)
        n_done += 1
        if n_done % 25 == 0:
            cost_so_far = total_in / 1e6 * 1.0 + total_out / 1e6 * 5.0  # Haiku 4.5
            logger.info(f"  LLM extracted {n_done}/{len(index)}  "
                        f"(tokens in: {total_in:,}, out: {total_out:,}, "
                        f"~${cost_so_far:.2f})")
        # Gentle rate limit
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    df.to_parquet(LLM_FEATURES_PARQUET, index=False)
    cost_total = total_in / 1e6 * 1.0 + total_out / 1e6 * 5.0
    logger.info(f"LLM features saved: {len(df):,} rows. "
                f"Total tokens: in={total_in:,}, out={total_out:,}. "
                f"~${cost_total:.2f} at Haiku 4.5 pricing.")
    return df


def main() -> None:
    df = extract_all()
    logger.info(f"\n=== LLM feature distributions across {len(df)} transcripts ===")
    cols = [k for k in FEATURE_SCHEMA if k in df.columns]
    num_cols = [c for c in cols if df[c].dtype in ("int64", "float64", "int32", "float32")]
    if num_cols:
        logger.info(df[num_cols].describe().round(2).to_string())
    if "capital_return_stance" in df.columns:
        logger.info(f"\ncapital_return_stance counts:\n{df['capital_return_stance'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
