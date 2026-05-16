"""
Criterion ABC and concrete subclasses for the Screener.

A Criterion turns a Listing into a raw signal value (higher =
better) or returns None to drop the ticker from ranking. The
Screener percentile-ranks the surviving signals per criterion
and combines them into a composite score.

Concrete criteria self-register in `_registry` via
__init_subclass__; this module re-exports it as
`criterion_registry` and exposes `get_criterion(name)`.

This file holds the ABC, four price-based criteria
(MomentumCriterion, CagrCriterion, SharpeCriterion,
MaxDrawdownCriterion), three fundamentals-based criteria
(PERatioCriterion, GrowthCriterion, LiquidityCriterion),
and two hard-filter criteria (SectorCriterion,
SwissTaxCriterion).
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np
import pandas as pd

from src.screening.listing import Listing


_registry: Dict[str, Type['Criterion']] = {}

weeks_per_year = 52
momentum_lookback_weeks = 13
min_observations_for_long_window = 52
min_observations_for_short_window = 26


class Criterion(ABC):
    """
    Abstract screening criterion.

    Attributes:
        name: Registry key. Subclasses with a non-empty `name`
            register themselves automatically.
        weight: Composite-score weighting (default 1.0).
    """

    name: str = ''
    weight: float = 1.0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            _registry[cls.name] = cls

    @abstractmethod
    def evaluate(self, listing: Listing) -> Optional[float]:
        """
        Return a raw signal value, higher = better.

        Args:
            listing: One Listing from an ExchangeUniverse.

        Returns:
            Float signal (higher = better) or None to drop the
            ticker from ranking.
        """
        raise NotImplementedError


def get_criterion(name: str) -> Criterion:
    """
    Instantiate a registered criterion by name.

    Args:
        name: Registry key (e.g. 'momentum', 'cagr').

    Returns:
        Fresh instance of the matching subclass.

    Raises:
        KeyError: When `name` is not registered. The message
            lists known names.
    """
    if name not in _registry:
        raise KeyError(
            f'Unknown criterion {name!r}. Known: '
            f'{sorted(_registry)}')
    return _registry[name]()


criterion_registry = _registry


class MomentumCriterion(Criterion):
    """13-week price momentum: last close / 13-weeks-ago close - 1."""

    name = 'momentum'

    def evaluate(self, listing: Listing) -> Optional[float]:
        closes = listing.closes_weekly
        if len(closes) < momentum_lookback_weeks:
            return None
        last = closes[-1]
        ref = closes[-momentum_lookback_weeks]
        if np.isnan(last) or np.isnan(ref) or ref <= 0:
            return None
        return float(last / ref - 1.0)


class CagrCriterion(Criterion):
    """Annualised return over the available weekly history."""

    name = 'cagr'

    def evaluate(self, listing: Listing) -> Optional[float]:
        closes = listing.closes_weekly
        valid = closes[~np.isnan(closes)]
        if len(valid) < min_observations_for_long_window:
            return None
        years = len(valid) / weeks_per_year
        if valid[0] <= 0:
            return None
        return float((valid[-1] / valid[0]) ** (1.0 / years) - 1.0)


class SharpeCriterion(Criterion):
    """Annualised Sharpe ratio of weekly returns (rf assumed 0)."""

    name = 'sharpe'

    def evaluate(self, listing: Listing) -> Optional[float]:
        closes = listing.closes_weekly
        valid = closes[~np.isnan(closes)]
        if len(valid) < min_observations_for_long_window:
            return None
        returns = np.diff(valid) / valid[:-1]
        std_w = float(np.std(returns, ddof=1))
        if std_w < 1e-10:
            return None
        mean_w = float(np.mean(returns))
        return float(
            (mean_w * weeks_per_year) /
            (std_w * np.sqrt(weeks_per_year)))


class MaxDrawdownCriterion(Criterion):
    """Negative max drawdown (less negative = better)."""

    name = 'max_drawdown'

    def evaluate(self, listing: Listing) -> Optional[float]:
        closes = listing.closes_weekly
        valid = closes[~np.isnan(closes)]
        if len(valid) < min_observations_for_short_window:
            return None
        cum_max = np.maximum.accumulate(valid)
        if np.any(cum_max <= 0):
            return None
        drawdown = (valid - cum_max) / cum_max
        return float(np.min(drawdown))


liquidity_lookback_weeks = 13
high_dividend_yield_threshold_pct = 5.0
bond_ticker_substrings = ('BOND', 'TREAS')
bond_sector_substring = 'Bond'


class PERatioCriterion(Criterion):
    """Value tilt: lower P/E is better; signal is -P/E."""

    name = 'pe_value'

    def evaluate(self, listing: Listing) -> Optional[float]:
        pe = listing.fundamentals['pe_ratio_ttm']
        if np.isnan(pe) or pe <= 0:
            return None
        return -float(pe)


class GrowthCriterion(Criterion):
    """Mean of revenue and earnings YoY growth."""

    name = 'growth'

    def evaluate(self, listing: Listing) -> Optional[float]:
        rev = listing.fundamentals['revenue_growth_yoy']
        eps = listing.fundamentals['earnings_growth_yoy']
        valid = []
        for x in (rev, eps):
            if not np.isnan(x):
                valid.append(float(x))
        if not valid:
            return None
        return float(np.mean(valid))


class LiquidityCriterion(Criterion):
    """Average weekly turnover (close * volume) over recent weeks."""

    name = 'liquidity'

    def evaluate(self, listing: Listing) -> Optional[float]:
        closes = listing.closes_weekly[-liquidity_lookback_weeks:]
        vols = listing.volumes_weekly[-liquidity_lookback_weeks:]
        valid_mask = ~(np.isnan(closes) | np.isnan(vols))
        if int(valid_mask.sum()) < 5:
            return None
        return float(np.mean(closes[valid_mask] * vols[valid_mask]))


class SectorCriterion(Criterion):
    """
    Hard sector inclusion / exclusion filter.

    With non-empty `included_sectors`, only those sectors pass.
    Always drops sectors in `excluded_sectors`.
    """

    name = 'sector'

    def __init__(
        self,
        included_sectors=None,
        excluded_sectors=None):
        self.included_sectors = (
            tuple(included_sectors)
            if included_sectors else ())
        self.excluded_sectors = (
            tuple(excluded_sectors)
            if excluded_sectors else ())

    def evaluate(self, listing: Listing) -> Optional[float]:
        if listing.sector in self.excluded_sectors:
            return None
        if (self.included_sectors
                and listing.sector not in self.included_sectors):
            return None
        return 1.0


class SwissTaxCriterion(Criterion):
    """Drop bond ETFs and high-dividend names per Swiss tax policy."""

    name = 'swiss_tax'

    def evaluate(self, listing: Listing) -> Optional[float]:
        ticker_upper = listing.ticker.upper()
        for sub in bond_ticker_substrings:
            if sub in ticker_upper:
                return None
        if bond_sector_substring in (listing.sector or ''):
            return None
        div_yield = listing.fundamentals['dividend_yield_ttm']
        if (not np.isnan(div_yield)
                and div_yield > high_dividend_yield_threshold_pct):
            return None
        return 1.0
