"""Tests for MaxGrowthStrategy."""

import pandas as pd
import pytest

from src.modelling.rebalancing.strategies import get_strategy
from src.modelling.rebalancing.strategies import (  # noqa: F401
    max_growth as _max_growth_mod)
from src.screening.candidate import CandidateTicker
from src.shared.constraints import RebalanceConstraints


def _candidates(rows):
    return [
        CandidateTicker(
            ticker=t, exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=score, metrics={})
        for t, score in rows]


def test_max_growth_registered():
    """MaxGrowthStrategy registers as 'max_growth'."""
    s = get_strategy('max_growth')
    assert s.name == 'max_growth'


def test_max_growth_blends_holds_and_picks():
    """60% to eligible holds, 40% equal-weighted to top picks."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 60.0, 'category': 'core'},
        {'ticker': 'GOOGL', 'value_chf': 400.0,
         'weight_pct': 40.0, 'category': 'core'},
    ])
    candidates = _candidates([
        ('NVDA', 0.95), ('META', 0.90),
        ('AAPL', 0.80), ('TSLA', 0.70),
        ('ORCL', 0.60),
    ])
    s = get_strategy('max_growth')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    # 5 candidates + 2 holds = 7 targets.
    assert set(w) == {
        'AMZN', 'GOOGL', 'NVDA', 'META',
        'AAPL', 'TSLA', 'ORCL'}
    # 60% retained on holds, proportional to current weight
    assert w['AMZN'] == pytest.approx(0.6 * 0.6)
    assert w['GOOGL'] == pytest.approx(0.6 * 0.4)
    # 40% across 5 picks equally
    assert w['NVDA'] == pytest.approx(0.4 / 5)
    assert sum(w.values()) == pytest.approx(1.0)


def test_max_growth_drops_candidates_already_held():
    """A candidate that's already held is not double-counted."""
    holdings = pd.DataFrame([
        {'ticker': 'NVDA', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('NVDA', 0.99), ('META', 0.90), ('AAPL', 0.80)])
    s = get_strategy('max_growth')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    # NVDA picked up once, as a hold (60% slice)
    assert w['NVDA'] == pytest.approx(0.6)
    assert 'META' in w
    assert 'AAPL' in w
    assert sum(w.values()) == pytest.approx(1.0)


def test_max_growth_top_n_truncates_picks():
    """Only top-N candidates by composite_score are kept."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', 0.99), ('B', 0.95), ('C', 0.90),
        ('D', 0.85), ('E', 0.80), ('F', 0.75)])
    from src.modelling.rebalancing.strategies.max_growth import (
        MaxGrowthStrategy)
    s = MaxGrowthStrategy(top_n=3)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    picks = set(w) - {'AMZN'}
    assert picks == {'A', 'B', 'C'}


def test_max_growth_raises_when_candidates_missing():
    """Loud failure if the screener feed is empty/None."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    s = get_strategy('max_growth')
    with pytest.raises(RuntimeError, match='candidates'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(
                excluded_categories=(), category_caps={}),
            candidates=None)


def test_max_growth_raises_when_no_eligible_holds_or_picks():
    """No holds and no picks raises (degenerate plan)."""
    holdings = pd.DataFrame([
        {'ticker': 'JUNK', 'value_chf': 100.0,
         'weight_pct': 100.0, 'category': 'speculative'}])
    s = get_strategy('max_growth')
    with pytest.raises(RuntimeError):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(),
            candidates=[])


def test_max_growth_with_cash_weight_emits_cash_entry():
    """cash_weight > 0 adds 'CASH' to the output weights."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 60.0, 'category': 'core'},
        {'ticker': 'GOOGL', 'value_chf': 400.0,
         'weight_pct': 40.0, 'category': 'core'},
    ])
    candidates = _candidates([
        ('NVDA', 0.95), ('META', 0.90),
        ('AAPL', 0.80), ('TSLA', 0.70),
        ('ORCL', 0.60),
    ])
    from src.modelling.rebalancing.strategies.max_growth import (
        MaxGrowthStrategy)
    s = MaxGrowthStrategy(top_n=5, cash_weight=0.20)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert 'CASH' in w
    assert w['CASH'] == pytest.approx(0.20)
    # Equity slice = 80%; holds = 60% of equity = 0.48; picks = 0.32
    assert w['AMZN'] == pytest.approx(0.80 * 0.60 * 0.60)
    assert w['NVDA'] == pytest.approx(0.80 * 0.40 / 5)
    assert sum(w.values()) == pytest.approx(1.0)
