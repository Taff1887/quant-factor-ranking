"""IO and on-disk caching helpers.

* Parquet for tabular data (fast, typed, columnar).
* JSON for raw API payloads, used by the FMP client's response cache so that
  pipeline re-runs are deterministic and do not re-hit the network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from qfr.utils.config import settings


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- Parquet tables -------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_parquet(path)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# --- JSON response cache --------------------------------------------------
def hash_key(*parts: Any) -> str:
    """Stable short hash for cache keys (excludes secrets by construction)."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(key: str, suffix: str = ".json") -> Path:
    ensure_dir(settings.cache_dir)
    return settings.cache_dir / f"{key}{suffix}"


def read_json_cache(key: str) -> Any | None:
    path = _cache_path(key)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def write_json_cache(key: str, obj: Any) -> Path:
    path = _cache_path(key)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return path
