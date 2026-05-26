"""Financial Modeling Prep (FMP) REST client.

A thin wrapper around the FMP API with three properties that matter for
reproducible research:

* **On-disk caching** — every raw JSON payload is cached under
  ``data/raw/fmp_cache`` keyed by endpoint + parameters (the API key is never
  part of the key), so a pipeline re-run is deterministic and avoids re-hitting
  the network. Pass ``force_refresh=True`` to bypass.
* **Robust retries** — transient network errors, rate-limit (429) and 5xx
  responses are retried with exponential backoff.
* **Polite rate limiting** — a minimum interval between calls keeps us inside
  tier limits even when looping over hundreds of symbols.

Endpoints target the FMP *stable* API (the current API; the legacy ``/api/v3``
host is closed to keys created after 2025-08-31). Symbol-scoped endpoints take
the ticker as a ``?symbol=`` query parameter. Override ``FMP_BASE_URL`` in
``.env`` only if you need a different host.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from qfr.utils.config import Settings
from qfr.utils.config import settings as default_settings
from qfr.utils.io import hash_key, read_json_cache, write_json_cache
from qfr.utils.logging import logger


class FMPError(RuntimeError):
    """Non-retryable FMP API error (bad key, malformed request, ...)."""


class FMPTransientError(RuntimeError):
    """Retryable rate-limit (429) or server (5xx) error."""


class FMPClient:
    """Caching, rate-limited client for the FMP REST API."""

    def __init__(self, settings: Settings | None = None, *, force_refresh: bool = False):
        self.settings = settings or default_settings
        if not self.settings.fmp_api_key:
            logger.warning("FMP_API_KEY is empty — set it in .env before making live calls.")
        self.base_url = self.settings.fmp_base_url.rstrip("/")
        self.force_refresh = force_refresh
        self._session = requests.Session()
        self._min_interval = 60.0 / max(self.settings.fmp_calls_per_minute, 1)
        self._last_call = 0.0
        self._lock = threading.Lock()

    # -- low level ---------------------------------------------------------
    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    @retry(
        retry=retry_if_exception_type((FMPTransientError, requests.RequestException)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(default_settings.fmp_max_retries),
        reraise=True,
    )
    def _request(self, url: str, params: dict[str, Any]) -> Any:
        self._throttle()
        resp = self._session.get(url, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning(f"FMP transient {resp.status_code} for {url} — retrying")
            raise FMPTransientError(f"HTTP {resp.status_code}")
        if resp.status_code in (401, 403):
            raise FMPError(f"Auth error {resp.status_code}: check FMP_API_KEY / subscription tier")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "Error Message" in data:
            raise FMPError(str(data["Error Message"]))
        return data

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        force_refresh: bool | None = None,
    ) -> Any:
        """GET an endpoint (relative to base URL), with transparent caching."""
        params = dict(params or {})
        key = hash_key(endpoint, sorted(params.items()))
        refresh = self.force_refresh if force_refresh is None else force_refresh
        if not refresh:
            cached = read_json_cache(key)
            if cached is not None:
                return cached
        params["apikey"] = self.settings.fmp_api_key
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        data = self._request(url, params)
        write_json_cache(key, data)
        return data

    # -- universe ----------------------------------------------------------
    def sp500_constituents(self, *, force_refresh: bool | None = None) -> list[dict]:
        """Current S&P 500 members (symbol, name, sector, sub-sector, ...)."""
        return self.get("sp500-constituent", force_refresh=force_refresh) or []

    def historical_sp500_constituents(self, *, force_refresh: bool | None = None) -> list[dict]:
        """Log of historical index changes (additions / removals with dates)."""
        return self.get("historical-sp500-constituent", force_refresh=force_refresh) or []

    # -- prices ------------------------------------------------------------
    def historical_prices(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
        *,
        series: str = "dividend-adjusted",
        force_refresh: bool | None = None,
    ) -> list[dict]:
        """Daily EOD bars for ``symbol``.

        ``series`` selects the endpoint variant:
        * ``dividend-adjusted`` (default) -> split- & dividend-adjusted
          ``adjOpen/adjHigh/adjLow/adjClose`` + ``volume`` (use for returns);
        * ``full`` -> raw OHLCV + ``change``/``changePercent``/``vwap``;
        * ``light`` -> ``price`` (close) + ``volume`` only.
        """
        params: dict[str, Any] = {"symbol": symbol}
        if from_date:
            params["from"] = str(from_date)
        if to_date:
            params["to"] = str(to_date)
        data = self.get(
            f"historical-price-eod/{series}", params=params, force_refresh=force_refresh
        )
        return data or []

    # -- company profile ---------------------------------------------------
    def profile(self, symbol: str, *, force_refresh: bool | None = None) -> dict:
        """Company profile (sector, industry, beta, market cap, ...)."""
        data = self.get("profile", params={"symbol": symbol}, force_refresh=force_refresh)
        return data[0] if isinstance(data, list) and data else {}

    # -- fundamentals ------------------------------------------------------
    def _statement(
        self,
        endpoint: str,
        symbol: str,
        period: str,
        limit: int,
        force_refresh: bool | None,
    ) -> list[dict]:
        return (
            self.get(
                endpoint,
                params={"symbol": symbol, "period": period, "limit": limit},
                force_refresh=force_refresh,
            )
            or []
        )

    def income_statement(self, symbol, period="quarter", limit=400, *, force_refresh=None):
        return self._statement("income-statement", symbol, period, limit, force_refresh)

    def balance_sheet(self, symbol, period="quarter", limit=400, *, force_refresh=None):
        return self._statement("balance-sheet-statement", symbol, period, limit, force_refresh)

    def cash_flow(self, symbol, period="quarter", limit=400, *, force_refresh=None):
        return self._statement("cash-flow-statement", symbol, period, limit, force_refresh)

    def ratios(self, symbol, period="quarter", limit=400, *, force_refresh=None):
        return self._statement("ratios", symbol, period, limit, force_refresh)

    def key_metrics(self, symbol, period="quarter", limit=400, *, force_refresh=None):
        return self._statement("key-metrics", symbol, period, limit, force_refresh)

    def enterprise_values(self, symbol, period="quarter", limit=400, *, force_refresh=None):
        return self._statement("enterprise-values", symbol, period, limit, force_refresh)

    def financial_growth(self, symbol, period="quarter", limit=400, *, force_refresh=None):
        return self._statement("financial-growth", symbol, period, limit, force_refresh)

    # -- analyst sentiment -------------------------------------------------
    def grades(self, symbol, *, force_refresh=None) -> list[dict]:
        """Dated log of individual analyst rating actions (upgrade/downgrade/...).

        Point-in-time safe: each row carries the action ``date``, plus
        ``previousGrade``/``newGrade``/``action``/``gradingCompany``. Used to build
        a recommendation-revision factor. (Unlike ``analyst-estimates``, which is a
        forward snapshot and not look-ahead-free.)
        """
        return self.get("grades", params={"symbol": symbol}, force_refresh=force_refresh) or []
