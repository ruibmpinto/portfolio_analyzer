"""Tests for MonteCarloEngine."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.modelling.monte_carlo.cost_model import (
    ibkr_default_cost_model)
from src.modelling.monte_carlo.engine import MonteCarloEngine
from src.modelling.monte_carlo.path_distribution import (
    PathDistribution)
from src.modelling.rebalancing.action import RebalancingAction
from src.modelling.rebalancing.plan import RebalancePlan
from src.shared.constraints import default_constraints


def _plan(target_weights, strategy_name='test'):
    """Build a RebalancePlan with synthetic HOLDs."""
    actions = []
    holdings_rows = []
    for t, w in target_weights.items():
        if t == 'CASH':
            actions.append(RebalancingAction(
                ticker='CASH', action='CASH', shares=0,
                est_cost_chf=w * 10000.0,
                current_wt_pct=0.0,
                target_wt_pct=w * 100.0, note='Cash reserve'))
        else:
            actions.append(RebalancingAction(
                ticker=t, action='HOLD', shares=0,
                est_cost_chf=0.0,
                current_wt_pct=w * 100.0,
                target_wt_pct=w * 100.0, note=''))
            holdings_rows.append({
                'ticker': t, 'value_chf': w * 10000.0,
                'weight_pct': w * 100.0, 'category': 'core'})
    holdings_df = pd.DataFrame(holdings_rows)
    return RebalancePlan(
        actions=actions, target_weights=target_weights,
        strategy_name=strategy_name,
        constraints=default_constraints(),
        holdings_df=holdings_df,
        generated_at=datetime.now())


def _prices(tickers, n_days=400, seed=0):
    """Daily prices DataFrame with synthetic random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n_days, freq='B')
    cols = {}
    for t in tickers:
        increments = rng.normal(0.0005, 0.012, n_days)
        cols[t] = 100.0 * np.exp(np.cumsum(increments))
    return pd.DataFrame(cols, index=idx)


def test_simulate_returns_path_distribution():
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    engine = MonteCarloEngine(n_paths=50, random_seed=7)
    dist = engine.simulate(
        plan, prices, horizon_days=10, initial_nav_chf=10000.0)
    assert isinstance(dist, PathDistribution)
    assert dist.paths.shape == (50, 11)
    assert dist.plan_name == 'test'
    assert dist.initial_nav_chf == 10000.0
    np.testing.assert_allclose(dist.paths[:, 0], 10000.0)


def test_simulate_cash_only_paths_drift_at_cash_rate():
    """100% CASH allocation produces deterministic paths."""
    plan = _plan({'CASH': 1.0})
    prices = _prices(['A'])
    engine = MonteCarloEngine(
        n_paths=10, random_seed=1, cash_rate_annual=0.0252)
    dist = engine.simulate(
        plan, prices, horizon_days=10, initial_nav_chf=10000.0)
    expected_terminal = 10000.0 * (1.0 + 0.0001) ** 10
    np.testing.assert_allclose(
        dist.paths[:, -1], expected_terminal, rtol=1e-9)


def test_simulate_costless_run_has_none_costs():
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    engine = MonteCarloEngine(
        n_paths=20, random_seed=2, cost_model=None)
    dist = engine.simulate(
        plan, prices, horizon_days=21, initial_nav_chf=10000.0)
    assert dist.cumulative_costs_chf is None


def test_simulate_with_costs_attaches_positive_cost_array():
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    engine = MonteCarloEngine(
        n_paths=20, random_seed=2,
        cost_model=ibkr_default_cost_model(),
        rebalance_frequency='monthly')
    dist = engine.simulate(
        plan, prices, horizon_days=63, initial_nav_chf=10000.0)
    assert dist.cumulative_costs_chf is not None
    assert dist.cumulative_costs_chf.shape == (20,)
    assert (dist.cumulative_costs_chf > 0).all()


def test_simulate_costs_reduce_terminal_nav():
    """Same seed: --no-costs > with-costs at terminal."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    seed = 11
    engine_no = MonteCarloEngine(
        n_paths=200, random_seed=seed, cost_model=None,
        rebalance_frequency='monthly')
    engine_yes = MonteCarloEngine(
        n_paths=200, random_seed=seed,
        cost_model=ibkr_default_cost_model(),
        rebalance_frequency='monthly')
    dist_no = engine_no.simulate(
        plan, prices, horizon_days=126,
        initial_nav_chf=10000.0)
    dist_yes = engine_yes.simulate(
        plan, prices, horizon_days=126,
        initial_nav_chf=10000.0)
    assert dist_yes.paths[:, -1].mean() < dist_no.paths[:, -1].mean()


def test_simulate_raises_when_ticker_missing_from_prices():
    plan = _plan({'A': 0.5, 'GHOST': 0.5})
    prices = _prices(['A'])
    engine = MonteCarloEngine(n_paths=10, random_seed=3)
    with pytest.raises(RuntimeError, match='GHOST'):
        engine.simulate(
            plan, prices, horizon_days=5,
            initial_nav_chf=10000.0)


def test_simulate_random_seed_reproducible():
    """Same seed yields identical paths."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    e1 = MonteCarloEngine(n_paths=30, random_seed=99)
    e2 = MonteCarloEngine(n_paths=30, random_seed=99)
    d1 = e1.simulate(
        plan, prices, horizon_days=15, initial_nav_chf=10000.0)
    d2 = e2.simulate(
        plan, prices, horizon_days=15, initial_nav_chf=10000.0)
    np.testing.assert_array_equal(d1.paths, d2.paths)


def test_simulate_monthly_contribution_grows_paths():
    """Positive contribution increases mean terminal NAV."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    seed = 7
    e_no = MonteCarloEngine(
        n_paths=100, random_seed=seed,
        rebalance_frequency='monthly',
        monthly_contribution_chf=0.0)
    e_yes = MonteCarloEngine(
        n_paths=100, random_seed=seed,
        rebalance_frequency='monthly',
        monthly_contribution_chf=500.0)
    d_no = e_no.simulate(
        plan, prices, horizon_days=126,
        initial_nav_chf=10000.0)
    d_yes = e_yes.simulate(
        plan, prices, horizon_days=126,
        initial_nav_chf=10000.0)
    assert d_yes.paths[:, -1].mean() > d_no.paths[:, -1].mean()


def test_simulate_threshold_rebalance_triggers_on_drift():
    """Threshold mode rebalances when a weight drifts > threshold."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    engine = MonteCarloEngine(
        n_paths=20, random_seed=8,
        rebalance_frequency='threshold',
        drift_threshold_pct=1.0,
        cost_model=ibkr_default_cost_model())
    dist = engine.simulate(
        plan, prices, horizon_days=63,
        initial_nav_chf=10000.0)
    assert dist.cumulative_costs_chf is not None
    assert dist.cumulative_costs_chf.mean() > 0.0


def test_simulate_accumulates_monthly_contributions():
    """Monthly rebalance over 63 days = 3 contributions."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    engine = MonteCarloEngine(
        n_paths=10, random_seed=4,
        rebalance_frequency='monthly',
        monthly_contribution_chf=200.0)
    dist = engine.simulate(
        plan, prices, horizon_days=63,
        initial_nav_chf=10000.0)
    # Monthly rebal fires at day % 21 == 0 for days 1..63
    # i.e. days 21, 42, 63 -> 3 contributions of 200 CHF.
    assert dist.cumulative_contributions_chf == pytest.approx(
        600.0)
    assert dist.total_invested_chf == pytest.approx(10600.0)


def test_simulate_accumulates_quarterly_contributions():
    """Quarterly rebalance over 126 days = 2 quarterly events."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    engine = MonteCarloEngine(
        n_paths=10, random_seed=5,
        rebalance_frequency='quarterly',
        monthly_contribution_chf=200.0)
    dist = engine.simulate(
        plan, prices, horizon_days=126,
        initial_nav_chf=10000.0)
    # Quarterly rebal fires at day % 63 == 0 -> days 63, 126.
    # Each adds 3 * monthly_contribution = 600. Total 1200.
    assert dist.cumulative_contributions_chf == pytest.approx(
        1200.0)
    assert dist.total_invested_chf == pytest.approx(11200.0)


def test_simulate_no_contributions_for_threshold_or_none():
    """Threshold and 'none' modes never add contributions."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    e_threshold = MonteCarloEngine(
        n_paths=10, random_seed=6,
        rebalance_frequency='threshold',
        drift_threshold_pct=1.0,
        monthly_contribution_chf=500.0)
    e_none = MonteCarloEngine(
        n_paths=10, random_seed=6,
        rebalance_frequency='none',
        monthly_contribution_chf=500.0)
    d_threshold = e_threshold.simulate(
        plan, prices, horizon_days=63,
        initial_nav_chf=10000.0)
    d_none = e_none.simulate(
        plan, prices, horizon_days=63,
        initial_nav_chf=10000.0)
    assert d_threshold.cumulative_contributions_chf == 0.0
    assert d_none.cumulative_contributions_chf == 0.0
