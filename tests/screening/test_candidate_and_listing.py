"""Tests for CandidateTicker and Listing dataclasses."""

import dataclasses

import pytest

from src.screening.candidate import CandidateTicker


def test_candidate_construct():
    c = CandidateTicker(
        ticker='AAPL', exchange='NASDAQ', currency='USD',
        sector='Technology', market_cap=3.0e12,
        composite_score=0.87,
        metrics={'momentum': 0.12, 'sharpe': 1.4})
    assert c.ticker == 'AAPL'
    assert c.composite_score == 0.87
    assert c.metrics['momentum'] == 0.12


def test_candidate_is_frozen():
    c = CandidateTicker(
        ticker='AAPL', exchange='NASDAQ', currency='USD',
        sector='Technology', market_cap=3.0e12,
        composite_score=0.87, metrics={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.composite_score = 0.5


def test_candidate_score_must_be_in_unit_interval():
    with pytest.raises(ValueError, match='composite_score'):
        CandidateTicker(
            ticker='AAPL', exchange='NASDAQ', currency='USD',
            sector='Technology', market_cap=3.0e12,
            composite_score=1.5, metrics={})
    with pytest.raises(ValueError, match='composite_score'):
        CandidateTicker(
            ticker='AAPL', exchange='NASDAQ', currency='USD',
            sector='Technology', market_cap=3.0e12,
            composite_score=-0.1, metrics={})


def test_candidate_metrics_dict_isolated_per_instance():
    a = CandidateTicker(
        ticker='AAPL', exchange='NASDAQ', currency='USD',
        sector='Technology', market_cap=3e12,
        composite_score=0.5, metrics={'x': 1.0})
    b = CandidateTicker(
        ticker='MSFT', exchange='NASDAQ', currency='USD',
        sector='Technology', market_cap=2e12,
        composite_score=0.5, metrics={'x': 1.0})
    assert a.metrics is not b.metrics


from datetime import datetime

import numpy as np
import pandas as pd

from src.screening.listing import Listing, fundamental_field_names


def _sample_listing(ticker='AAPL', exchange='NASDAQ'):
    n = 156
    dates = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    closes = np.linspace(100.0, 200.0, n)
    volumes = np.full(n, 1.0e7)
    dividends = np.zeros(n)
    fundamentals = {k: 1.0 for k in fundamental_field_names}
    return Listing(
        ticker=ticker, exchange=exchange, currency='USD',
        sector='Technology', market_cap=3.0e12,
        closes_weekly=closes, closes_weekly_dates=dates,
        volumes_weekly=volumes, dividends_weekly=dividends,
        fundamentals=fundamentals,
        refreshed_at=datetime(2026, 5, 14, 12, 0))


def test_listing_construct():
    l = _sample_listing()
    assert l.ticker == 'AAPL'
    assert len(l.closes_weekly) == 156
    assert l.fundamentals['pe_ratio_ttm'] == 1.0


def test_listing_fundamental_field_names_count_is_thirty():
    """The Tier-B fundamentals contract has exactly 30 fields."""
    assert len(fundamental_field_names) == 30


def test_listing_fundamental_field_names_match_contract():
    """Field names match the per-exchange parquet cache contract."""
    expected = {
        # Original 17 (Phase 3)
        'pe_ratio_ttm', 'pe_ratio_forward', 'pb_ratio',
        'ps_ratio', 'dividend_yield_ttm', 'eps_ttm',
        'revenue_growth_yoy', 'earnings_growth_yoy',
        'profit_margin', 'operating_margin', 'roe', 'roa',
        'debt_to_equity', 'free_cash_flow', 'beta_yf',
        'shares_outstanding', 'short_ratio',
        # Phase 8 additions (defeatbeta US-only enrichment)
        'peg_ratio', 'roic', 'roce', 'wacc',
        'equity_multiplier', 'asset_turnover',
        'enterprise_value', 'enterprise_to_revenue',
        'enterprise_to_ebitda', 'ttm_revenue',
        'ebitda_growth_yoy', 'market_cap_chf',
        'dcf_implied_upside'}
    assert set(fundamental_field_names) == expected


def test_listing_accepts_30_fundamental_fields():
    """Listing validates with the full 30-field envelope."""
    n = 156
    fundamentals = {
        name: float('nan') for name in fundamental_field_names}
    assert len(fundamentals) == 30
    listing = Listing(
        ticker='NVDA', exchange='NASDAQ', currency='USD',
        sector='Tech', market_cap=3e12,
        closes_weekly=np.full(n, 100.0),
        closes_weekly_dates=pd.date_range(
            '2024-01-01', periods=n, freq='W'),
        volumes_weekly=np.full(n, 1e6),
        dividends_weekly=np.full(n, 0.0),
        fundamentals=fundamentals,
        refreshed_at=pd.Timestamp.now().to_pydatetime())
    # NaN check via inequality with itself
    val = listing.fundamentals['dcf_implied_upside']
    assert val != val


def test_listing_rejects_wrong_series_length():
    """closes_weekly et al. must be length 156."""
    n_bad = 100
    dates = pd.date_range(
        '2023-05-15', periods=n_bad, freq='W-FRI')
    closes = np.linspace(100.0, 200.0, n_bad)
    volumes = np.full(n_bad, 1.0e7)
    dividends = np.zeros(n_bad)
    fundamentals = {k: 1.0 for k in fundamental_field_names}
    with pytest.raises(ValueError, match='closes_weekly'):
        Listing(
            ticker='AAPL', exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1.0e9,
            closes_weekly=closes,
            closes_weekly_dates=dates,
            volumes_weekly=volumes,
            dividends_weekly=dividends,
            fundamentals=fundamentals,
            refreshed_at=datetime.now())


def test_listing_rejects_missing_fundamental_field():
    """Every Tier-B field must be present (NaN is acceptable)."""
    l_args = _sample_listing()
    bad_fundamentals = {
        k: 1.0 for k in fundamental_field_names if k != 'roe'}
    with pytest.raises(ValueError, match='roe'):
        Listing(
            ticker='AAPL', exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1.0e9,
            closes_weekly=l_args.closes_weekly,
            closes_weekly_dates=l_args.closes_weekly_dates,
            volumes_weekly=l_args.volumes_weekly,
            dividends_weekly=l_args.dividends_weekly,
            fundamentals=bad_fundamentals,
            refreshed_at=datetime.now())
