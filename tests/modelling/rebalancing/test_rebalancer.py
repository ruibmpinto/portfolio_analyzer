"""Tests for Rebalancer orchestrator."""

import pandas as pd
import pytest

from src.modelling.rebalancing.action import RebalancingAction
from src.modelling.rebalancing.plan import RebalancePlan
from src.modelling.rebalancing.rebalancer import Rebalancer
from src.modelling.rebalancing.strategies import get_strategy
from src.shared.constraints import (
    RebalanceConstraints, default_constraints)


def _holdings(weights_pct, value_total=10000.0):
    """Build a fake holdings_df with given weights summing to 100."""
    rows = []
    for ticker, w in weights_pct.items():
        value = value_total * w / 100.0
        rows.append({
            'ticker': ticker, 'shares': 10,
            'value_chf': value, 'weight_pct': w,
            'currency': 'CHF', 'sector': 'X',
            'price_chf': value / 10.0, 'category': 'core'})
    return pd.DataFrame(rows)


class _FakeAnalyzer:
    """Stand-in for PortfolioAnalyzer in unit tests.

    Provides only the two methods Rebalancer calls.
    """
    def __init__(self, holdings_df, prices):
        self._holdings_df = holdings_df
        self._prices = prices
    def get_holdings_snapshot(self, categories=None):
        return self._holdings_df.copy()
    def get_price_panel(self):
        return self._prices.copy()


def test_rebalancer_returns_plan():
    holdings = _holdings({'A': 40.0, 'B': 60.0})
    prices = pd.DataFrame({
        'A': [100.0, 100.0, 100.0],
        'B': [100.0, 100.0, 100.0]},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        default_constraints())
    plan = rb.propose()
    assert isinstance(plan, RebalancePlan)
    assert plan.strategy_name == 'equal_weight'


def test_rebalancer_buy_action_created_for_underweight():
    holdings = _holdings({'A': 30.0, 'B': 70.0})
    prices = pd.DataFrame({
        'A': [100.0] * 3, 'B': [100.0] * 3},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        constraints)
    plan = rb.propose()
    by_ticker = {a.ticker: a for a in plan.actions}
    assert by_ticker['A'].action == 'BUY'
    assert by_ticker['A'].shares > 0


def test_rebalancer_reduce_action_for_overweight_beyond_band():
    """Overweight beyond -200 CHF gap -> REDUCE."""
    holdings = _holdings({'A': 30.0, 'B': 70.0})
    prices = pd.DataFrame({
        'A': [100.0] * 3, 'B': [100.0] * 3},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        default_constraints())
    plan = rb.propose()
    by_ticker = {a.ticker: a for a in plan.actions}
    assert by_ticker['B'].action == 'REDUCE'
    assert by_ticker['B'].shares > 0
    assert 'held' in by_ticker['B'].note.lower()


def test_rebalancer_hold_inside_band():
    """Tickers near target -> HOLD."""
    holdings = _holdings({'A': 50.0, 'B': 50.0})
    prices = pd.DataFrame({
        'A': [100.0] * 3, 'B': [100.0] * 3},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        default_constraints())
    plan = rb.propose()
    actions_by_type = {a.action for a in plan.actions}
    assert 'HOLD' in actions_by_type


def test_rebalancer_buy_pool_includes_reduce_proceeds_minus_cash():
    """BUY total <= new_capital + total_reduce - target_cash."""
    holdings = _holdings({'A': 10.0, 'B': 90.0})
    prices = pd.DataFrame({
        'A': [100.0] * 3, 'B': [100.0] * 3},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    constraints = RebalanceConstraints(new_capital_chf=500.0)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'), constraints)
    plan = rb.propose()
    total_buy = sum(
        a.est_cost_chf for a in plan.actions
        if a.action == 'BUY')
    total_reduce = sum(
        a.est_cost_chf for a in plan.actions
        if a.action == 'REDUCE')
    total_cash = sum(
        a.est_cost_chf for a in plan.actions
        if a.action == 'CASH')  # 0.0 for equal_weight
    funding_cap = (
        constraints.new_capital_chf
        + total_reduce - total_cash)
    assert total_buy <= funding_cap + 1e-6


def test_rebalancer_passes_categories_to_analyzer():
    """Rebalancer.propose forwards constraints-derived categories."""
    captured = {}

    class _CapturingAnalyzer:
        _holdings_df = _holdings({'A': 50.0, 'B': 50.0})
        _prices = pd.DataFrame({
            'A': [100.0] * 3, 'B': [100.0] * 3},
            index=pd.date_range('2024-01-01', periods=3))

        def get_holdings_snapshot(self, categories=None):
            captured['categories'] = categories
            return self._holdings_df.copy()

        def get_price_panel(self):
            return self._prices.copy()

    analyzer = _CapturingAnalyzer()
    cats = {'A': 'core', 'B': 'core'}
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        default_constraints(),
        ticker_categories=cats)
    rb.propose()
    assert captured['categories'] == cats


def test_buy_allocation_is_proportional_not_greedy():
    """Equal gaps + equal prices -> equal share counts.

    A and B are underweight by the same amount at the same
    price, funded by a pool too small to cover both gaps. The
    old greedy fill handed the first ticker its full gap and
    starved the second (23 vs 17 shares here); proportional fill
    splits the pool by gap, so both get the same allocation.
    """
    holdings = _holdings({'A': 10.0, 'B': 10.0, 'C': 80.0})
    prices = pd.DataFrame({
        'A': [100.0] * 3, 'B': [100.0] * 3, 'C': [100.0] * 3},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        RebalanceConstraints(new_capital_chf=0.0))
    plan = rb.propose()
    by_ticker = {a.ticker: a for a in plan.actions}
    assert by_ticker['A'].action == 'BUY'
    assert by_ticker['B'].action == 'BUY'
    assert by_ticker['A'].shares == by_ticker['B'].shares


def test_achievable_wt_pct_between_current_and_target():
    """Underfunded BUY reaches an achievable weight below target.

    The achievable weight equals the post-trade value over the
    post-trade NAV, and sits between the current and the
    aspirational target weight when the buy is only partly
    funded.
    """
    holdings = _holdings({'A': 10.0, 'B': 10.0, 'C': 80.0})
    prices = pd.DataFrame({
        'A': [100.0] * 3, 'B': [100.0] * 3, 'C': [100.0] * 3},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        RebalanceConstraints(new_capital_chf=0.0))
    plan = rb.propose()
    a = {x.ticker: x for x in plan.actions}['A']
    assert a.current_wt_pct < a.achievable_wt_pct < a.target_wt_pct
    # Post-trade value / post-trade NAV, in percent. No new
    # capital, so NAV stays 10000.
    expected = (1000.0 + a.est_cost_chf) / 10000.0 * 100.0
    assert a.achievable_wt_pct == pytest.approx(expected)
