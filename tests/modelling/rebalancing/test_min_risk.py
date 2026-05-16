"""Tests for MinRiskStrategy."""

import pandas as pd
import pytest

from src.modelling.rebalancing.strategies import get_strategy
from src.modelling.rebalancing.strategies import (  # noqa: F401
    min_risk as _min_risk_mod)
from src.screening.candidate import CandidateTicker
from src.shared.constraints import RebalanceConstraints


def _candidates(rows):
    return [
        CandidateTicker(
            ticker=t, exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=0.5,
            metrics={'max_drawdown': md})
        for t, md in rows]


def test_min_risk_registered():
    s = get_strategy('min_risk')
    assert s.name == 'min_risk'


def test_min_risk_default_allocation():
    """36% holds + 12% ETF + 12% low-vol + 40% cash."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 60.0, 'category': 'core'},
        {'ticker': 'GOOGL', 'value_chf': 400.0,
         'weight_pct': 40.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', -0.05), ('B', -0.07),
        ('C', -0.09), ('D', -0.20)])
    s = get_strategy('min_risk')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert 'CSSPX.SW' in w
    assert w['CSSPX.SW'] == pytest.approx(0.12)
    assert w['AMZN'] == pytest.approx(0.36 * 0.6)
    assert w['GOOGL'] == pytest.approx(0.36 * 0.4)
    assert 'CASH' in w
    assert w['CASH'] == pytest.approx(0.40)
    picks = set(w) - {'AMZN', 'GOOGL', 'CSSPX.SW', 'CASH'}
    assert picks == {'A', 'B', 'C'}
    assert sum(w.values()) == pytest.approx(1.0)


def test_min_risk_inverse_drawdown_weighting():
    """Lower abs(max_drawdown) -> larger weight in lowvol bucket."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('STABLE', -0.02), ('OK', -0.10), ('NOISY', -0.30)])
    s = get_strategy('min_risk')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert w['STABLE'] > w['OK'] > w['NOISY']


def test_min_risk_does_not_duplicate_held_etf():
    """If broad ETF is already held, no separate ETF entry."""
    holdings = pd.DataFrame([
        {'ticker': 'CSSPX.SW', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', -0.05), ('B', -0.07), ('C', -0.09)])
    s = get_strategy('min_risk')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    # CSSPX.SW counted once. Sum still 1.0.
    cs = [t for t in w if t == 'CSSPX.SW']
    assert len(cs) == 1
    assert sum(w.values()) == pytest.approx(1.0)


def test_min_risk_raises_when_candidate_missing_max_drawdown():
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    bad = [
        CandidateTicker(
            ticker='X', exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=0.5,
            metrics={'cagr': 0.1})]
    s = get_strategy('min_risk')
    with pytest.raises(KeyError, match='max_drawdown'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(),
            candidates=bad)


def test_min_risk_raises_when_no_candidates_and_no_etf_or_holds():
    """Degenerate plan raises."""
    holdings = pd.DataFrame([
        {'ticker': 'JUNK', 'value_chf': 100.0,
         'weight_pct': 100.0, 'category': 'speculative'}])
    s = get_strategy('min_risk')
    with pytest.raises(RuntimeError, match='candidates'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(),
            candidates=None)


def test_min_risk_constructor_rejects_weights_not_summing_to_one():
    """etf + lowvol + holds + cash must sum to 1.0."""
    from src.modelling.rebalancing.strategies.min_risk import (
        MinRiskStrategy)
    with pytest.raises(ValueError, match='sum to 1.0'):
        MinRiskStrategy(
            etf_weight=0.20, lowvol_weight=0.20,
            holds_weight=0.20, cash_weight=0.20)


def test_min_risk_zero_cash_weight_omits_cash_entry():
    """cash_weight=0 -> no CASH entry; other weights re-sum to 1."""
    from src.modelling.rebalancing.strategies.min_risk import (
        MinRiskStrategy)
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', -0.05), ('B', -0.07), ('C', -0.09)])
    s = MinRiskStrategy(
        etf_weight=0.20, lowvol_weight=0.20,
        holds_weight=0.60, cash_weight=0.0)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert 'CASH' not in w
    assert sum(w.values()) == pytest.approx(1.0)


def test_min_risk_raises_when_no_picks_remain_and_lowvol_weight_positive():
    """Empty picks after filtering with lowvol_weight > 0 raises."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    # Every candidate is filtered out: swiss_tax=0 marks fail.
    candidates = [
        CandidateTicker(
            ticker='X', exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=0.5,
            metrics={'max_drawdown': -0.05,
                     'swiss_tax': 0.0})]
    s = get_strategy('min_risk')
    with pytest.raises(
            RuntimeError, match='lowvol_weight'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(
                excluded_categories=(), category_caps={}),
            candidates=candidates)


def test_min_risk_with_lowvol_weight_zero_allows_empty_picks():
    """lowvol_weight=0 + empty picks succeeds (no slice to fill)."""
    from src.modelling.rebalancing.strategies.min_risk import (
        MinRiskStrategy)
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    # All candidates filtered by swiss_tax.
    candidates = [
        CandidateTicker(
            ticker='X', exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=0.5,
            metrics={'max_drawdown': -0.05,
                     'swiss_tax': 0.0})]
    s = MinRiskStrategy(
        etf_weight=0.20, lowvol_weight=0.0,
        holds_weight=0.60, cash_weight=0.20)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert w['CASH'] == pytest.approx(0.20)
    assert sum(w.values()) == pytest.approx(1.0)
