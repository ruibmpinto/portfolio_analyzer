"""Tests for Screener: load universes, apply criteria, rank."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.screening.candidate import CandidateTicker
from src.screening.listing import Listing, fundamental_field_names
from src.screening.universe import ExchangeUniverse


def _listing(ticker, momentum_pct=0.10, sector='Technology'):
    n = 156
    dates = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    closes = np.full(n, 100.0)
    closes[-1] = 100.0 * (1.0 + momentum_pct)
    fund = {k: 1.0 for k in fundamental_field_names}
    return Listing(
        ticker=ticker, exchange='NASDAQ', currency='USD',
        sector=sector, market_cap=1.0e9,
        closes_weekly=closes, closes_weekly_dates=dates,
        volumes_weekly=np.full(n, 1.0e6),
        dividends_weekly=np.zeros(n),
        fundamentals=fund,
        refreshed_at=datetime(2026, 5, 14, 12, 0))


def _universe(*listings):
    return ExchangeUniverse('NASDAQ', list(listings))


def test_screener_rank_returns_candidates(monkeypatch):
    """rank() returns a list of CandidateTicker objects."""
    from src.screening.screener import Screener
    from src.screening.criteria import MomentumCriterion
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('A', 0.05),
            _listing('B', 0.20),
            _listing('C', 0.10)))
    s = Screener(['NASDAQ']).add_criterion(MomentumCriterion())
    out = s.rank()
    assert all(isinstance(c, CandidateTicker) for c in out)


def test_screener_orders_by_composite_desc(monkeypatch):
    """Best momentum lands first."""
    from src.screening.screener import Screener
    from src.screening.criteria import MomentumCriterion
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('A', 0.05),
            _listing('B', 0.20),
            _listing('C', 0.10)))
    s = Screener(['NASDAQ']).add_criterion(MomentumCriterion())
    out = s.rank()
    assert [c.ticker for c in out] == ['B', 'C', 'A']


def test_screener_top_n_truncates(monkeypatch):
    """top_n returns at most N candidates."""
    from src.screening.screener import Screener
    from src.screening.criteria import MomentumCriterion
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('A', 0.05),
            _listing('B', 0.20),
            _listing('C', 0.10)))
    s = Screener(['NASDAQ']).add_criterion(MomentumCriterion())
    out = s.rank(top_n=2)
    assert len(out) == 2
    assert [c.ticker for c in out] == ['B', 'C']


def test_screener_drops_listings_when_criterion_returns_none(monkeypatch):
    """Listings whose criterion returns None are excluded."""
    from src.screening.screener import Screener
    from src.screening.criteria import SwissTaxCriterion
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('AAPL', 0.05),
            _listing('AGG-BOND', 0.20)))
    s = Screener(['NASDAQ']).add_criterion(SwissTaxCriterion())
    out = s.rank()
    assert {c.ticker for c in out} == {'AAPL'}


def test_screener_combines_multiple_criteria(monkeypatch):
    """Composite score is the weighted mean of percentile ranks."""
    from src.screening.screener import Screener
    from src.screening.criteria import (
        MomentumCriterion, CagrCriterion)
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('A', 0.05),
            _listing('B', 0.20),
            _listing('C', 0.10)))
    s = (Screener(['NASDAQ'])
         .add_criterion(MomentumCriterion())
         .add_criterion(CagrCriterion()))
    out = s.rank()
    for c in out:
        assert 0.0 <= c.composite_score <= 1.0
    assert out[0].ticker == 'B'


def test_screener_rank_without_criteria_raises(monkeypatch):
    """rank() with no criteria raises a clear error."""
    from src.screening.screener import Screener
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(_listing('A')))
    with pytest.raises(RuntimeError, match='criterion'):
        Screener(['NASDAQ']).rank()


def test_screener_to_dataframe_columns(monkeypatch):
    """to_dataframe surfaces ticker / exchange / score columns."""
    from src.screening.screener import Screener
    from src.screening.criteria import MomentumCriterion
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('A', 0.05), _listing('B', 0.20)))
    s = Screener(['NASDAQ']).add_criterion(MomentumCriterion())
    s.rank()
    df = s.to_dataframe()
    expected = {
        'ticker', 'exchange', 'currency', 'sector',
        'market_cap', 'composite_score'}
    assert expected.issubset(df.columns)
