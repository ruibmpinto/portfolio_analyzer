"""Tests for RiskAdjustedStrategy."""

import pandas as pd
import pytest

from src.modelling.rebalancing.strategies import get_strategy
from src.modelling.rebalancing.strategies import (  # noqa: F401
    risk_adjusted as _risk_adjusted_mod)
from src.screening.candidate import CandidateTicker
from src.shared.constraints import RebalanceConstraints


def _candidates(rows):
    return [
        CandidateTicker(
            ticker=t, exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=0.7,
            metrics={'sharpe': sharpe,
                     'max_drawdown': md})
        for t, sharpe, md in rows]


def test_risk_adjusted_registered():
    s = get_strategy('risk_adjusted')
    assert s.name == 'risk_adjusted'


def test_risk_adjusted_higher_sharpe_picks_chosen():
    """Top-N candidates by Sharpe are selected."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', 2.0, -0.10), ('B', 1.8, -0.12),
        ('C', 1.5, -0.15), ('D', 1.0, -0.20),
        ('E', 0.5, -0.25), ('F', 0.1, -0.30)])
    from src.modelling.rebalancing.strategies.risk_adjusted \
        import RiskAdjustedStrategy
    s = RiskAdjustedStrategy(
        top_n=3, holds_weight=0.6, max_per_name=0.5,
        cash_weight=0.0)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    picks = set(w) - {'AMZN'}
    assert picks == {'A', 'B', 'C'}


def test_risk_adjusted_lower_drawdown_weighted_higher():
    """Lower max_drawdown magnitude -> larger weight."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('LOW_DD', 1.5, -0.05),
        ('HIGH_DD', 1.4, -0.50)])
    from src.modelling.rebalancing.strategies.risk_adjusted \
        import RiskAdjustedStrategy
    s = RiskAdjustedStrategy(
        top_n=2, holds_weight=0.5, max_per_name=0.5,
        cash_weight=0.0)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert w['LOW_DD'] > w['HIGH_DD']
    assert sum(w.values()) == pytest.approx(1.0)


def test_risk_adjusted_per_name_cap_enforced():
    """No candidate exceeds max_per_name."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    # One candidate has tiny drawdown so it dominates pre-cap
    candidates = _candidates([
        ('DOMINANT', 2.0, -0.001),
        ('B', 1.5, -0.20),
        ('C', 1.4, -0.20)])
    # Sanity: pre-cap, DOMINANT would dwarf B/C. Inverse-DD
    # weights are 1000, 5, 5 -> DOMINANT's share of the picks
    # slice = 1000/1010 ~= 0.99. With defaults (cash 0.10,
    # holds 0.6 -> picks slice 0.36), DOMINANT would land
    # at ~0.36 pre-cap -- well above the 0.15 cap.
    pre_cap_dominant_share = 1000.0 / (1000.0 + 5.0 + 5.0)
    picks_slice = 0.9 * 0.4  # equity_mass * picks_frac
    pre_cap_dominant_weight = pre_cap_dominant_share * picks_slice
    assert pre_cap_dominant_weight > 0.15, (
        'test setup invalid: DOMINANT does not exceed cap '
        'pre-clipping, so the cap loop never fires')
    s = get_strategy('risk_adjusted')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    candidate_weights = {
        t: v for t, v in w.items()
        if t not in ('AMZN', 'CASH')}
    assert max(candidate_weights.values()) <= 0.15 + 1e-9
    assert sum(w.values()) == pytest.approx(1.0)


def test_risk_adjusted_raises_on_missing_metric():
    """Missing 'sharpe' or 'max_drawdown' raises."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = [
        CandidateTicker(
            ticker='X', exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=0.7,
            metrics={'cagr': 0.1}),  # no sharpe / max_dd
    ]
    s = get_strategy('risk_adjusted')
    with pytest.raises(KeyError, match='sharpe'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(
                excluded_categories=(), category_caps={}),
            candidates=candidates)


def test_risk_adjusted_raises_on_no_candidates():
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    s = get_strategy('risk_adjusted')
    with pytest.raises(RuntimeError, match='candidates'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(),
            candidates=None)


def test_risk_adjusted_default_cash_weight_is_10pct():
    """Default cash_weight=0.10 emits a CASH entry."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 60.0, 'category': 'core'},
        {'ticker': 'GOOGL', 'value_chf': 400.0,
         'weight_pct': 40.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', 2.0, -0.10), ('B', 1.8, -0.12),
        ('C', 1.5, -0.15), ('D', 1.0, -0.20),
        ('E', 0.5, -0.25)])
    s = get_strategy('risk_adjusted')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert 'CASH' in w
    assert w['CASH'] == pytest.approx(0.10)
    assert sum(w.values()) == pytest.approx(1.0)


def test_risk_adjusted_cash_weight_zero_omits_cash_entry():
    """cash_weight=0 -> no CASH entry."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', 2.0, -0.10), ('B', 1.8, -0.12)])
    from src.modelling.rebalancing.strategies.risk_adjusted \
        import RiskAdjustedStrategy
    s = RiskAdjustedStrategy(cash_weight=0.0)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert 'CASH' not in w
    assert sum(w.values()) == pytest.approx(1.0)
