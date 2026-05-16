"""Tests for Criterion ABC and concrete criteria."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.screening.listing import Listing, fundamental_field_names


def _listing(closes=None, volumes=None, fundamentals=None,
             sector='Technology', ticker='AAPL',
             dividend_yield=0.0):
    n = 156
    dates = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    if closes is None:
        closes = np.linspace(100.0, 200.0, n)
    if volumes is None:
        volumes = np.full(n, 1.0e6)
    fund = {k: 1.0 for k in fundamental_field_names}
    fund['dividend_yield_ttm'] = dividend_yield
    if fundamentals:
        fund.update(fundamentals)
    return Listing(
        ticker=ticker, exchange='NASDAQ', currency='USD',
        sector=sector, market_cap=1.0e9,
        closes_weekly=np.asarray(closes, dtype=float),
        closes_weekly_dates=dates,
        volumes_weekly=np.asarray(volumes, dtype=float),
        dividends_weekly=np.zeros(n),
        fundamentals=fund,
        refreshed_at=datetime(2026, 5, 14, 12, 0))


def test_criterion_registry_has_nine_criteria():
    """All 9 documented criteria self-register."""
    from src.screening.criteria import criterion_registry
    expected = {
        'momentum', 'cagr', 'sharpe', 'max_drawdown',
        'pe_value', 'growth', 'liquidity', 'sector',
        'swiss_tax'}
    assert set(criterion_registry) == expected


def test_get_criterion_returns_instance():
    from src.screening.criteria import (
        get_criterion, Criterion)
    c = get_criterion('momentum')
    assert isinstance(c, Criterion)
    assert c.name == 'momentum'


def test_get_criterion_unknown_raises():
    from src.screening.criteria import get_criterion
    with pytest.raises(KeyError, match='Unknown criterion'):
        get_criterion('not_real')


def test_momentum_uses_13_week_ratio():
    """13-week return: closes[-1] / closes[-13] - 1."""
    from src.screening.criteria import MomentumCriterion
    closes = np.full(156, 100.0)
    closes[-1] = 110.0    # last week
    closes[-13] = 100.0   # 13 weeks ago
    score = MomentumCriterion().evaluate(_listing(closes=closes))
    assert score == pytest.approx(0.10, abs=1e-6)


def test_cagr_compounds_weekly_to_annual():
    """3y CAGR from valid (non-NaN) closes."""
    from src.screening.criteria import CagrCriterion
    closes = np.linspace(100.0, 200.0, 156)
    score = CagrCriterion().evaluate(_listing(closes=closes))
    # 156 weeks = 3 years; (200/100)^(1/3) - 1 ~ 0.2599
    assert score == pytest.approx(0.2599, abs=0.01)


def test_sharpe_returns_annualized_ratio():
    """Sharpe on weekly returns; output is annualised."""
    from src.screening.criteria import SharpeCriterion
    rng = np.random.default_rng(0)
    closes = (
        100.0 * np.cumprod(
            1.0 + rng.normal(0.002, 0.005, 156)))
    score = SharpeCriterion().evaluate(_listing(closes=closes))
    assert score > 0.5


def test_max_drawdown_returns_negative_absolute():
    """MaxDD result is negative (more negative = worse)."""
    from src.screening.criteria import MaxDrawdownCriterion
    closes = np.concatenate([
        np.linspace(100.0, 200.0, 78),
        np.linspace(200.0, 100.0, 78)])
    score = MaxDrawdownCriterion().evaluate(
        _listing(closes=closes))
    # Drawdown from 200 to 100 = -50%, score = -0.5
    assert score == pytest.approx(-0.5, abs=1e-6)


def test_momentum_returns_none_with_nan_endpoints():
    """If closes[-1] or closes[-13] is NaN, momentum returns None."""
    from src.screening.criteria import MomentumCriterion
    closes = np.full(156, 100.0)
    closes[-1] = np.nan
    assert MomentumCriterion().evaluate(
        _listing(closes=closes)) is None


def test_cagr_returns_none_with_too_few_observations():
    """Need >= 52 valid (non-NaN) weekly closes."""
    from src.screening.criteria import CagrCriterion
    closes = np.full(156, np.nan)
    closes[-30:] = np.linspace(100.0, 110.0, 30)
    assert CagrCriterion().evaluate(
        _listing(closes=closes)) is None


def test_pe_value_returns_negative_pe():
    """Lower P/E is better; criterion returns -P/E."""
    from src.screening.criteria import PERatioCriterion
    score = PERatioCriterion().evaluate(
        _listing(fundamentals={'pe_ratio_ttm': 15.0}))
    assert score == pytest.approx(-15.0)


def test_pe_value_returns_none_for_nonpositive_pe():
    """Negative or NaN P/E -> drop."""
    from src.screening.criteria import PERatioCriterion
    assert PERatioCriterion().evaluate(
        _listing(fundamentals={'pe_ratio_ttm': -5.0})) is None
    assert PERatioCriterion().evaluate(
        _listing(
            fundamentals={'pe_ratio_ttm': float('nan')})) is None


def test_growth_averages_revenue_and_earnings():
    """Mean of revenue + earnings YoY growth."""
    from src.screening.criteria import GrowthCriterion
    score = GrowthCriterion().evaluate(_listing(
        fundamentals={
            'revenue_growth_yoy': 0.10,
            'earnings_growth_yoy': 0.20}))
    assert score == pytest.approx(0.15)


def test_liquidity_returns_avg_recent_turnover():
    """Mean of close * volume over the last 13 weeks."""
    from src.screening.criteria import LiquidityCriterion
    closes = np.full(156, 100.0)
    volumes = np.full(156, 1.0e6)
    score = LiquidityCriterion().evaluate(
        _listing(closes=closes, volumes=volumes))
    assert score == pytest.approx(1.0e8)


def test_sector_filter_drops_excluded():
    """Excluded sectors return None."""
    from src.screening.criteria import SectorCriterion
    crit = SectorCriterion(excluded_sectors=('Real Estate',))
    listing = _listing(sector='Real Estate')
    assert crit.evaluate(listing) is None


def test_sector_filter_passes_included():
    """Included sector returns 1.0."""
    from src.screening.criteria import SectorCriterion
    crit = SectorCriterion(included_sectors=('Technology',))
    assert crit.evaluate(_listing(sector='Technology')) == 1.0
    assert crit.evaluate(_listing(sector='Energy')) is None


def test_swiss_tax_drops_high_dividend():
    """Dividend yield > 5% triggers a drop."""
    from src.screening.criteria import SwissTaxCriterion
    assert SwissTaxCriterion().evaluate(
        _listing(dividend_yield=6.0)) is None


def test_swiss_tax_drops_bond_etf_by_ticker():
    """Tickers with BOND or TREAS substrings are dropped."""
    from src.screening.criteria import SwissTaxCriterion
    assert SwissTaxCriterion().evaluate(
        _listing(ticker='AGG-BOND')) is None
    assert SwissTaxCriterion().evaluate(
        _listing(ticker='SHORTTREAS')) is None
