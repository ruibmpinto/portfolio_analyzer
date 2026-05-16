"""Tests for PathDistribution."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.modelling.monte_carlo.path_distribution import (
    PathDistribution)


def _make_paths(n_paths=4, n_steps=10, seed=0):
    """Deterministic NAV paths starting at 10000."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(
        loc=10.0, scale=20.0, size=(n_paths, n_steps))
    nav = np.zeros((n_paths, n_steps + 1))
    nav[:, 0] = 10000.0
    for d in range(n_steps):
        nav[:, d + 1] = nav[:, d] + increments[:, d]
    return nav


def test_path_distribution_construction():
    """Frozen dataclass holds paths and metadata."""
    paths = _make_paths()
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    assert pd_.paths.shape == (4, 11)
    assert pd_.plan_name == 'test'


def test_percentiles_shape_and_columns():
    """percentiles() returns DataFrame indexed by step."""
    paths = _make_paths()
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    df = pd_.percentiles(qs=(0.05, 0.5, 0.95))
    assert list(df.columns) == ['p5', 'p50', 'p95']
    assert len(df) == 11
    assert (df['p5'] <= df['p50']).all()
    assert (df['p50'] <= df['p95']).all()


def test_terminal_distribution_returns_last_column():
    paths = _make_paths()
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    terminal = pd_.terminal_distribution()
    assert terminal.shape == (4,)
    np.testing.assert_array_equal(terminal, paths[:, -1])


def test_max_drawdown_distribution_is_negative_fraction():
    """Drawdown is the most-negative running-peak deviation."""
    paths = np.array([
        [100.0, 120.0, 90.0, 110.0, 60.0],
        [100.0, 110.0, 110.0, 115.0, 130.0],
    ])
    pd_ = PathDistribution(
        paths=paths, horizon_days=4,
        initial_nav_chf=100.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    dd = pd_.max_drawdown_distribution()
    assert dd[0] == pytest.approx(-0.5)
    assert dd[1] == pytest.approx(0.0)


def test_prob_below_threshold():
    paths = np.array([
        [10000.0, 12000.0],
        [10000.0, 8000.0],
        [10000.0, 9500.0],
        [10000.0, 7000.0],
    ])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    assert pd_.prob_below(9000.0) == pytest.approx(0.5)


def test_expected_terminal_return():
    paths = np.array([
        [100.0, 110.0],
        [100.0, 120.0],
        [100.0, 90.0],
    ])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=100.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    assert pd_.expected_terminal_return() == pytest.approx(
        0.06666667, abs=1e-6)


def test_to_summary_dict_costless():
    """Costless run reports zero cost fields."""
    paths = _make_paths()
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    s = pd_.to_summary_dict()
    expected_keys = {
        'initial_nav_chf', 'horizon_days', 'n_paths',
        'cumulative_contributions_chf', 'total_invested_chf',
        'terminal_p5', 'terminal_p50', 'terminal_p95',
        'expected_terminal_return',
        'max_drawdown_p5', 'max_drawdown_p50',
        'max_drawdown_p95',
        'prob_loss',
        'cumulative_cost_chf_p50',
        'cumulative_cost_chf_p95',
        'annual_cost_drag_bps',
        'mu_ann_pct', 'sigma_ann_pct',
        'var_5_pct', 'cvar_5_pct',
        'p_gain_10pct', 'p_gain_20pct',
        'p_dd_over_15', 'p_dd_over_25'}
    assert set(s) == expected_keys
    assert s['cumulative_cost_chf_p50'] == 0.0
    assert s['cumulative_cost_chf_p95'] == 0.0
    assert s['annual_cost_drag_bps'] == 0.0
    assert s['n_paths'] == 4
    assert s['horizon_days'] == 10


def test_to_summary_dict_with_costs():
    """Non-None cumulative_costs_chf flows into the summary."""
    paths = _make_paths()
    costs = np.array([100.0, 150.0, 200.0, 250.0])
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=costs)
    s = pd_.to_summary_dict()
    assert s['cumulative_cost_chf_p50'] == pytest.approx(175.0)
    assert s['cumulative_cost_chf_p95'] == pytest.approx(
        242.5, abs=2.5)
    expected_drag = (175.0 / 10000.0) / (10.0 / 252.0) * 1e4
    assert s['annual_cost_drag_bps'] == pytest.approx(
        expected_drag, abs=1.0)


def test_prob_loss_uses_initial_nav():
    """prob_loss is the fraction of paths terminating below initial NAV."""
    paths = np.array([
        [100.0, 110.0],
        [100.0, 80.0],
        [100.0, 95.0],
        [100.0, 130.0]])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=100.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    s = pd_.to_summary_dict()
    assert s['prob_loss'] == pytest.approx(0.5)


def test_mu_and_sigma_ann_pct_are_terminal_return_pct_stats():
    """mu_ann_pct / sigma_ann_pct = mean / std of terminal return %."""
    # Four paths terminate at +10%, -20%, -5%, +30%.
    paths = np.array([
        [100.0, 110.0],
        [100.0, 80.0],
        [100.0, 95.0],
        [100.0, 130.0]])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=100.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    s = pd_.to_summary_dict()
    returns_pct = np.array([10.0, -20.0, -5.0, 30.0])
    assert s['mu_ann_pct'] == pytest.approx(
        float(returns_pct.mean()))
    assert s['sigma_ann_pct'] == pytest.approx(
        float(returns_pct.std()))


def test_var_5_pct_is_5th_percentile_of_returns():
    """var_5_pct is the 5th percentile of terminal return %."""
    rng = np.random.default_rng(0)
    n = 1000
    terminal_pct = rng.normal(5.0, 15.0, n)
    terminal_nav = 10000.0 * (1.0 + terminal_pct / 100.0)
    paths = np.column_stack([
        np.full(n, 10000.0), terminal_nav])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    s = pd_.to_summary_dict()
    expected_var = float(np.percentile(terminal_pct, 5))
    assert s['var_5_pct'] == pytest.approx(
        expected_var, abs=1e-9)


def test_cvar_5_pct_is_mean_below_var():
    """cvar_5_pct is the mean of returns at or below var_5_pct."""
    # 20 paths, deterministic return ladder from -50% to +49%.
    returns_pct = np.arange(-50.0, 50.0, 5.0)
    n = len(returns_pct)
    terminal_nav = 100.0 * (1.0 + returns_pct / 100.0)
    paths = np.column_stack([
        np.full(n, 100.0), terminal_nav])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=100.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    s = pd_.to_summary_dict()
    var = float(np.percentile(returns_pct, 5))
    tail = returns_pct[returns_pct <= var]
    assert s['cvar_5_pct'] == pytest.approx(
        float(tail.mean()))


def test_p_gain_thresholds():
    """p_gain_10pct / p_gain_20pct: % of paths above thresholds."""
    # Returns: -10, 0, 12, 25, 30 -> >10%: 3/5 = 60; >20%: 2/5 = 40
    terminal_nav = np.array([90.0, 100.0, 112.0, 125.0, 130.0])
    paths = np.column_stack([
        np.full(5, 100.0), terminal_nav])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=100.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    s = pd_.to_summary_dict()
    assert s['p_gain_10pct'] == pytest.approx(60.0)
    assert s['p_gain_20pct'] == pytest.approx(40.0)


def test_total_invested_property_and_default():
    """cumulative_contributions_chf defaults to 0.0 and the
    derived total_invested_chf equals initial_nav_chf."""
    paths = _make_paths()
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    assert pd_.cumulative_contributions_chf == 0.0
    assert pd_.total_invested_chf == pytest.approx(10000.0)


def test_contributions_shift_pnl_basis_to_total_invested():
    """pnl_pct uses (terminal - total_invested) / total_invested."""
    # Start 10000 + contributions 24000 = total_invested 34000.
    # Terminal equals total_invested -> pnl_pct = 0 -> mu_ann_pct = 0.
    paths = np.array([
        [10000.0, 34000.0],
        [10000.0, 34000.0]])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None,
        cumulative_contributions_chf=24000.0)
    s = pd_.to_summary_dict()
    assert s['total_invested_chf'] == pytest.approx(34000.0)
    assert s['cumulative_contributions_chf'] == pytest.approx(
        24000.0)
    assert s['mu_ann_pct'] == pytest.approx(0.0)
    assert s['sigma_ann_pct'] == pytest.approx(0.0)


def test_prob_loss_uses_total_invested_when_contributions_set():
    """prob_loss counts paths below total_invested, not initial."""
    # initial 10000, contributions 5000, total_invested 15000.
    # Terminal 14000 (below total_invested but above initial)
    # counts as a loss under contribution-adjusted semantic.
    paths = np.array([
        [10000.0, 14000.0],
        [10000.0, 20000.0]])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None,
        cumulative_contributions_chf=5000.0)
    s = pd_.to_summary_dict()
    assert s['prob_loss'] == pytest.approx(0.5)


def test_p_dd_thresholds():
    """p_dd_over_15 / p_dd_over_25: % of paths with worse drawdown."""
    # Three paths: DD = 0, -20%, -30%.
    # > -15% threshold: 2/3 = 66.67%; > -25%: 1/3 = 33.33%.
    paths = np.array([
        [100.0, 100.0, 100.0],          # DD = 0
        [100.0, 120.0, 96.0],            # DD = (96-120)/120 = -20%
        [100.0, 100.0, 70.0]])           # DD = -30%
    pd_ = PathDistribution(
        paths=paths, horizon_days=2,
        initial_nav_chf=100.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    s = pd_.to_summary_dict()
    assert s['p_dd_over_15'] == pytest.approx(
        2.0 / 3.0 * 100.0)
    assert s['p_dd_over_25'] == pytest.approx(
        1.0 / 3.0 * 100.0)
