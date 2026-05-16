"""Shared application context for the dashboard sidecar.

A ``DashboardContext`` bundles the PortfolioAnalyzer, the data
provider, the per-strategy cache, the loaded ticker-category
map, and the resolved configuration paths. FastAPI route
handlers read it from ``request.app.state.ctx``; tests can
construct one directly with stubbed dependencies.

Heavy computations (analyzer construction, Monte Carlo runs)
are produced through the shared ``SnapshotCache`` so a tab
switch is cheap.
"""

import json
from typing import Dict, Optional

from src.analysis.core.analyzer import PortfolioAnalyzer
from src.dashboard_api import config
from src.dashboard_api.cache import SnapshotCache
from src.shared.data_provider import (
    DataProvider, YFinanceProvider)


class DashboardContext:
    """App-wide singleton for the FastAPI sidecar.

    Attributes:
        analyzer: Lazily-constructed PortfolioAnalyzer reading
            both Degiro and IBKR CSVs.
        data_provider: ``DataProvider`` used for prices, FX,
            and benchmark lookups.
        cache: Shared TTL cache for per-key results.
        categories: Ticker -> category map (read from the
            configured JSON file). Empty dict when the file is
            absent.
        base_currency: ISO-4217 currency for CHF roll-ups.
    """

    def __init__(
            self,
            analyzer: Optional[PortfolioAnalyzer] = None,
            data_provider: Optional[DataProvider] = None,
            cache: Optional[SnapshotCache] = None,
            categories: Optional[Dict[str, str]] = None,
            base_currency: Optional[str] = None):
        self.data_provider = (
            data_provider or YFinanceProvider())
        self.cache = (
            cache or SnapshotCache(
                ttl_seconds=config.cache_ttl_seconds))
        self.base_currency = (
            base_currency or config.base_currency)
        self.categories = (
            categories if categories is not None
            else self._load_categories())
        self._analyzer = analyzer

    @property
    def analyzer(self) -> PortfolioAnalyzer:
        """Lazy PortfolioAnalyzer wired to the configured CSVs."""
        if self._analyzer is None:
            self._analyzer = PortfolioAnalyzer(
                degiro_csv_file_path=_existing_or_none(
                    config.degiro_csv_path),
                ibkr_csv_file_path=_existing_or_none(
                    config.ibkr_csv_path),
                data_provider=self.data_provider,
                base_currency=self.base_currency)
        return self._analyzer

    def invalidate(self) -> None:
        """Drop the analyzer + every cached entry.

        Forces the next request to rebuild the analyzer from
        the source CSVs and to re-run any Monte Carlo / fee
        aggregation.
        """
        self.cache.invalidate()
        self._analyzer = None

    @staticmethod
    def _load_categories() -> Dict[str, str]:
        """Read ticker_categories.json or return an empty map."""
        path = config.ticker_categories_path
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)


def _existing_or_none(path) -> Optional[str]:
    """Return `str(path)` when the file exists, else None.

    Handles the case where `config.*_csv_path` itself is None
    because no statement was found on disk and no env override
    was set.
    """
    if path is None or not path.exists():
        return None
    return str(path)
