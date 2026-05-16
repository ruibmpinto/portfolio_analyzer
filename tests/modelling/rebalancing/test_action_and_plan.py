"""Tests for RebalancingAction and RebalancePlan."""

import dataclasses

import pytest

from src.modelling.rebalancing.action import RebalancingAction


def test_action_construct_buy():
    a = RebalancingAction(
        ticker='AMZN', action='BUY', shares=3,
        est_cost_chf=450.0, current_wt_pct=2.0,
        target_wt_pct=5.0, note='')
    assert a.ticker == 'AMZN'
    assert a.action == 'BUY'
    assert a.shares == 3
    assert a.est_cost_chf == 450.0


def test_action_is_frozen():
    a = RebalancingAction(
        ticker='AMZN', action='HOLD', shares=0,
        est_cost_chf=0.0, current_wt_pct=4.5,
        target_wt_pct=5.0, note='')
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.action = 'BUY'


def test_action_rejects_unknown_action():
    """Action must be one of BUY/REDUCE/HOLD."""
    with pytest.raises(ValueError, match='action'):
        RebalancingAction(
            ticker='AMZN', action='SHORT', shares=1,
            est_cost_chf=10.0, current_wt_pct=1.0,
            target_wt_pct=2.0, note='')


def test_action_rejects_negative_cost():
    """est_cost_chf must be >= 0 (sign carried by `action`)."""
    with pytest.raises(ValueError, match='est_cost_chf'):
        RebalancingAction(
            ticker='AMZN', action='BUY', shares=1,
            est_cost_chf=-50.0, current_wt_pct=1.0,
            target_wt_pct=2.0, note='')


def test_hold_must_have_zero_shares():
    """HOLD action: shares must be 0 (sanity check)."""
    with pytest.raises(ValueError, match='HOLD'):
        RebalancingAction(
            ticker='AMZN', action='HOLD', shares=5,
            est_cost_chf=0.0, current_wt_pct=4.5,
            target_wt_pct=5.0, note='')


from datetime import datetime

import pandas as pd

from src.modelling.rebalancing.plan import RebalancePlan
from src.shared.constraints import default_constraints


def _sample_actions():
    return [
        RebalancingAction(
            ticker='AMZN', action='BUY', shares=3,
            est_cost_chf=450.0, current_wt_pct=2.0,
            target_wt_pct=20.0, note=''),
        RebalancingAction(
            ticker='UBSG.SW', action='REDUCE', shares=1,
            est_cost_chf=25.0, current_wt_pct=8.0,
            target_wt_pct=30.0, note='Only if held >6 months'),
        RebalancingAction(
            ticker='HOLN.SW', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=15.0,
            target_wt_pct=50.0, note=''),
    ]


def _sample_holdings_df():
    return pd.DataFrame([
        {'ticker': 'AMZN', 'shares': 5, 'value_chf': 600.0,
         'weight_pct': 2.0, 'currency': 'USD',
         'sector': 'Consumer Cyclical', 'price_chf': 120.0,
         'category': 'core'},
        {'ticker': 'UBSG.SW', 'shares': 4, 'value_chf': 100.0,
         'weight_pct': 8.0, 'currency': 'CHF',
         'sector': 'Financial Services', 'price_chf': 25.0,
         'category': 'core'},
        {'ticker': 'HOLN.SW', 'shares': 2, 'value_chf': 140.0,
         'weight_pct': 15.0, 'currency': 'CHF',
         'sector': 'Basic Materials', 'price_chf': 70.0,
         'category': 'core'},
    ])


def test_plan_construct():
    plan = RebalancePlan(
        actions=_sample_actions(),
        target_weights={'AMZN': 0.20, 'UBSG.SW': 0.30,
                        'HOLN.SW': 0.50},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=_sample_holdings_df(),
        generated_at=datetime(2026, 5, 14, 10, 0))
    assert plan.strategy_name == 'mvo'
    assert len(plan.actions) == 3


def test_plan_to_dataframe_columns():
    plan = RebalancePlan(
        actions=_sample_actions(),
        target_weights={'AMZN': 0.20, 'UBSG.SW': 0.30,
                        'HOLN.SW': 0.50},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=_sample_holdings_df(),
        generated_at=datetime(2026, 5, 14, 10, 0))
    df = plan.to_dataframe()
    expected = {
        'ticker', 'action', 'shares', 'est_cost_chf',
        'current_wt_pct', 'target_wt_pct', 'note'}
    assert set(df.columns) == expected
    assert len(df) == 3


def test_plan_target_weights_dataframe_columns():
    plan = RebalancePlan(
        actions=_sample_actions(),
        target_weights={'AMZN': 0.20, 'UBSG.SW': 0.30,
                        'HOLN.SW': 0.50},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=_sample_holdings_df(),
        generated_at=datetime(2026, 5, 14, 10, 0))
    df = plan.target_weights_dataframe()
    expected = {
        'ticker', 'current_wt_pct', 'target_wt_pct',
        'deviation_pct'}
    assert set(df.columns) == expected


def test_plan_summary_aggregates():
    plan = RebalancePlan(
        actions=_sample_actions(),
        target_weights={'AMZN': 0.20, 'UBSG.SW': 0.30,
                        'HOLN.SW': 0.50},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=_sample_holdings_df(),
        generated_at=datetime(2026, 5, 14, 10, 0))
    s = plan.summary()
    assert s['n_buy'] == 1
    assert s['n_reduce'] == 1
    assert s['n_hold'] == 1
    assert s['total_buy_chf'] == 450.0
    assert s['total_reduce_chf'] == 25.0


def test_plan_total_value_chf_property():
    plan = RebalancePlan(
        actions=_sample_actions(),
        target_weights={'AMZN': 0.20, 'UBSG.SW': 0.30,
                        'HOLN.SW': 0.50},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=_sample_holdings_df(),
        generated_at=datetime(2026, 5, 14, 10, 0))
    assert plan.total_value_chf == 840.0


def test_plan_target_weights_must_sum_to_one():
    """Constructor validates target_weights sums to ~1."""
    with pytest.raises(ValueError, match='sum'):
        RebalancePlan(
            actions=_sample_actions(),
            target_weights={'AMZN': 0.5, 'UBSG.SW': 0.3},
            strategy_name='mvo',
            constraints=default_constraints(),
            holdings_df=_sample_holdings_df(),
            generated_at=datetime(2026, 5, 14, 10, 0))


def test_plan_rejects_ticker_set_mismatch():
    """target_weights tickers must match actions tickers."""
    with pytest.raises(ValueError, match='ticker set'):
        RebalancePlan(
            actions=_sample_actions(),
            target_weights={'AMZN': 0.5, 'GHOST': 0.5},
            strategy_name='mvo',
            constraints=default_constraints(),
            holdings_df=_sample_holdings_df(),
            generated_at=datetime(2026, 5, 14, 10, 0))


def test_plan_rejects_per_ticker_weight_mismatch():
    """target_weights[t] * 100 must equal action.target_wt_pct."""
    # Sample actions have AMZN/UBSG.SW/HOLN.SW targets 20/30/50.
    # Provide a weights dict that sums to 1.0 but disagrees with
    # AMZN's 20.0 (use 0.40 -> 40% vs the action's 20%).
    with pytest.raises(ValueError, match='Inconsistent target weight'):
        RebalancePlan(
            actions=_sample_actions(),
            target_weights={'AMZN': 0.40, 'UBSG.SW': 0.10,
                            'HOLN.SW': 0.50},
            strategy_name='mvo',
            constraints=default_constraints(),
            holdings_df=_sample_holdings_df(),
            generated_at=datetime(2026, 5, 14, 10, 0))


def test_plan_rejects_duplicate_action_tickers():
    """Two actions for the same ticker must raise.

    Otherwise downstream to_dataframe and summary would
    silently double-count the duplicated ticker.
    """
    duplicate_actions = [
        RebalancingAction(
            ticker='AMZN', action='BUY', shares=3,
            est_cost_chf=450.0, current_wt_pct=2.0,
            target_wt_pct=20.0, note=''),
        RebalancingAction(
            ticker='AMZN', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=2.0,
            target_wt_pct=20.0, note=''),
        RebalancingAction(
            ticker='UBSG.SW', action='REDUCE', shares=1,
            est_cost_chf=25.0, current_wt_pct=8.0,
            target_wt_pct=30.0, note='Only if held >6 months'),
        RebalancingAction(
            ticker='HOLN.SW', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=15.0,
            target_wt_pct=50.0, note=''),
    ]
    with pytest.raises(ValueError, match='duplicate ticker'):
        RebalancePlan(
            actions=duplicate_actions,
            target_weights={'AMZN': 0.20, 'UBSG.SW': 0.30,
                            'HOLN.SW': 0.50},
            strategy_name='mvo',
            constraints=default_constraints(),
            holdings_df=_sample_holdings_df(),
            generated_at=datetime(2026, 5, 14, 10, 0))


def test_cash_action_is_a_valid_action():
    """CASH is in the action vocabulary."""
    a = RebalancingAction(
        ticker='CASH', action='CASH', shares=0,
        est_cost_chf=400.0, current_wt_pct=0.0,
        target_wt_pct=40.0, note='Cash reserve')
    assert a.action == 'CASH'


def test_cash_action_requires_zero_shares():
    """CASH inherits the shares==0 invariant."""
    with pytest.raises(ValueError, match='CASH'):
        RebalancingAction(
            ticker='CASH', action='CASH', shares=1,
            est_cost_chf=400.0, current_wt_pct=0.0,
            target_wt_pct=40.0, note='')


def test_cash_action_allows_positive_est_cost():
    """est_cost_chf carries the target cash magnitude."""
    a = RebalancingAction(
        ticker='CASH', action='CASH', shares=0,
        est_cost_chf=1234.56, current_wt_pct=0.0,
        target_wt_pct=20.0, note='')
    assert a.est_cost_chf == pytest.approx(1234.56)


def test_plan_summary_includes_total_cash_chf():
    """summary() reports total_cash_chf summed over CASH rows."""
    actions = [
        RebalancingAction(
            ticker='AMZN', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=60.0,
            target_wt_pct=60.0, note=''),
        RebalancingAction(
            ticker='CASH', action='CASH', shares=0,
            est_cost_chf=400.0, current_wt_pct=0.0,
            target_wt_pct=40.0, note='Cash reserve'),
    ]
    target_weights = {'AMZN': 0.6, 'CASH': 0.4}
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    plan = RebalancePlan(
        actions=actions,
        target_weights=target_weights,
        strategy_name='fake',
        constraints=default_constraints(),
        holdings_df=holdings,
        generated_at=datetime.now())
    s = plan.summary()
    assert s['total_cash_chf'] == pytest.approx(400.0)


def test_plan_summary_total_cash_zero_when_no_cash_actions():
    """No CASH action -> total_cash_chf == 0."""
    actions = [
        RebalancingAction(
            ticker='AMZN', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=100.0,
            target_wt_pct=100.0, note='')]
    plan = RebalancePlan(
        actions=actions,
        target_weights={'AMZN': 1.0},
        strategy_name='fake',
        constraints=default_constraints(),
        holdings_df=pd.DataFrame([
            {'ticker': 'AMZN', 'value_chf': 1000.0,
             'weight_pct': 100.0, 'category': 'core'}]),
        generated_at=datetime.now())
    assert plan.summary()['total_cash_chf'] == 0.0


def test_plan_per_ticker_tolerance_boundary():
    """Per-ticker weight check accepts diffs up to 0.1 percent.

    Differences strictly larger than 0.1 percentage points are
    rejected; differences at-or-below 0.1 are accepted. This
    locks in the boundary semantics so future edits to
    target_weights_sum_tol don't silently drift.
    """
    # 0.099 percentage points off -> accepted
    near_actions = [
        RebalancingAction(
            ticker='A', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=50.0,
            target_wt_pct=49.901, note=''),
        RebalancingAction(
            ticker='B', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=50.0,
            target_wt_pct=50.099, note=''),
    ]
    holdings = pd.DataFrame([
        {'ticker': 'A', 'shares': 5, 'value_chf': 500.0,
         'weight_pct': 50.0, 'currency': 'CHF',
         'sector': 'X', 'price_chf': 100.0,
         'category': 'core'},
        {'ticker': 'B', 'shares': 5, 'value_chf': 500.0,
         'weight_pct': 50.0, 'currency': 'CHF',
         'sector': 'X', 'price_chf': 100.0,
         'category': 'core'},
    ])
    # target sums to 1.0 exactly; per-ticker diff is 0.099 pp,
    # under the 0.1 pp tolerance.
    RebalancePlan(
        actions=near_actions,
        target_weights={'A': 0.49901, 'B': 0.50099},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=holdings,
        generated_at=datetime(2026, 5, 14, 10, 0))

    # 0.2 percentage points off -> rejected
    far_actions = [
        RebalancingAction(
            ticker='A', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=50.0,
            target_wt_pct=49.8, note=''),
        RebalancingAction(
            ticker='B', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=50.0,
            target_wt_pct=50.2, note=''),
    ]
    with pytest.raises(
        ValueError, match='Inconsistent target weight'):
        RebalancePlan(
            actions=far_actions,
            target_weights={'A': 0.49901, 'B': 0.50099},
            strategy_name='mvo',
            constraints=default_constraints(),
            holdings_df=holdings,
            generated_at=datetime(2026, 5, 14, 10, 0))
