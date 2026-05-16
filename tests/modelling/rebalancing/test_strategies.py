"""Tests for rebalancing strategies and the strategy registry."""

import inspect

import pandas as pd
import pytest

from src.modelling.rebalancing.strategies import (
    Strategy, get_strategy, strategy_registry)
from src.shared.constraints import default_constraints


def test_get_strategy_returns_instance():
    """get_strategy(name) returns a Strategy instance."""
    s = get_strategy('equal_weight')
    assert isinstance(s, Strategy)
    assert s.name == 'equal_weight'


def test_get_strategy_raises_on_unknown():
    """Unknown name raises KeyError listing known names."""
    with pytest.raises(KeyError, match='Unknown strategy'):
        get_strategy('not_a_strategy')


def test_strategy_abc_propose_signature():
    """Strategy.propose must accept the contract args."""
    sig = inspect.signature(Strategy.propose)
    params = set(sig.parameters)
    expected = {
        'self', 'holdings_df', 'prices', 'constraints',
        'candidates', 'risk_free_rate'}
    assert expected.issubset(params)


def test_equal_weight_registered():
    """EqualWeightStrategy registers with name 'equal_weight'."""
    assert 'equal_weight' in strategy_registry


def test_equal_weight_uniform_across_eligible():
    """1/N weights across non-excluded, non-dust holdings."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 30.0, 'category': 'core'},
        {'ticker': 'UBSG.SW', 'value_chf': 400.0,
         'weight_pct': 20.0, 'category': 'core'},
        {'ticker': 'HOLN.SW', 'value_chf': 1000.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    prices = pd.DataFrame()    # equal_weight ignores prices
    s = get_strategy('equal_weight')
    w = s.propose(holdings, prices, default_constraints())
    assert set(w) == {'AMZN', 'UBSG.SW', 'HOLN.SW'}
    for v in w.values():
        assert v == pytest.approx(1.0 / 3.0)


def test_equal_weight_excludes_speculative():
    """excluded_categories drops tickers before weighting."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 30.0, 'category': 'core'},
        {'ticker': 'USAR', 'value_chf': 400.0,
         'weight_pct': 20.0, 'category': 'speculative'},
        {'ticker': 'HOLN.SW', 'value_chf': 1000.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    prices = pd.DataFrame()
    s = get_strategy('equal_weight')
    w = s.propose(holdings, prices, default_constraints())
    assert 'USAR' not in w
    for v in w.values():
        assert v == pytest.approx(0.5)


def test_equal_weight_drops_dust_positions():
    """min_position_pct drops dust before weighting."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 60.0, 'category': 'core'},
        {'ticker': 'TINY', 'value_chf': 1.0,
         'weight_pct': 0.1, 'category': 'core'},
        {'ticker': 'HOLN.SW', 'value_chf': 400.0,
         'weight_pct': 39.9, 'category': 'core'},
    ])
    prices = pd.DataFrame()
    s = get_strategy('equal_weight')
    w = s.propose(holdings, prices, default_constraints())
    assert 'TINY' not in w
    assert len(w) == 2


def test_equal_weight_sums_to_one():
    """Output weights sum to 1.0 within plan tolerance."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 30.0, 'category': 'core'},
        {'ticker': 'UBSG.SW', 'value_chf': 400.0,
         'weight_pct': 20.0, 'category': 'core'},
    ])
    s = get_strategy('equal_weight')
    w = s.propose(holdings, pd.DataFrame(),
                  default_constraints())
    assert sum(w.values()) == pytest.approx(1.0)


def test_inverse_vol_higher_weight_for_lower_vol():
    """Lower-vol asset gets higher weight."""
    import numpy as np
    holdings = pd.DataFrame([
        {'ticker': 'LOWVOL', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
        {'ticker': 'HIGHVOL', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    rng = np.random.default_rng(42)
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    low = pd.Series(
        100 + rng.normal(0, 0.5, n).cumsum(), index=idx)
    high = pd.Series(
        100 + rng.normal(0, 2.5, n).cumsum(), index=idx)
    prices = pd.DataFrame({'LOWVOL': low, 'HIGHVOL': high})
    s = get_strategy('inverse_vol')
    w = s.propose(holdings, prices, default_constraints())
    assert w['LOWVOL'] > w['HIGHVOL']
    assert sum(w.values()) == pytest.approx(1.0)


def test_inverse_vol_raises_when_prices_missing_ticker():
    """Held ticker absent from `prices` raises (no silent skip)."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 30.0, 'category': 'core'},
        {'ticker': 'GHOST', 'value_chf': 400.0,
         'weight_pct': 20.0, 'category': 'core'},
    ])
    prices = pd.DataFrame({
        'AMZN': pd.Series(
            [100.0, 101.0, 102.0, 100.5],
            index=pd.date_range('2024-01-01', periods=4))})
    s = get_strategy('inverse_vol')
    with pytest.raises(RuntimeError, match='GHOST'):
        s.propose(holdings, prices, default_constraints())


def test_min_variance_assigns_more_weight_to_lower_vol():
    """Min-variance over 2 uncorrelated assets favors low-vol.

    Uses a relaxed cap (max_weight_default=1.0) so the
    2-ticker problem is feasible. Cap behavior is exercised
    in test_min_variance_respects_max_weight_cap.
    """
    import numpy as np
    from src.shared.constraints import RebalanceConstraints
    holdings = pd.DataFrame([
        {'ticker': 'STABLE', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
        {'ticker': 'VOLATILE', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    rng = np.random.default_rng(0)
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    stable = pd.Series(
        100 + rng.normal(0, 0.3, n).cumsum(), index=idx)
    volatile = pd.Series(
        100 + rng.normal(0, 3.0, n).cumsum(), index=idx)
    prices = pd.DataFrame({
        'STABLE': stable, 'VOLATILE': volatile})
    relaxed = RebalanceConstraints(max_weight_default=1.0)
    s = get_strategy('min_variance')
    w = s.propose(holdings, prices, relaxed)
    assert w['STABLE'] > w['VOLATILE']
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)


def test_min_variance_respects_max_weight_cap():
    """A heavily preferred asset is still capped at 15%."""
    import numpy as np
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(1)
    rows = []
    cols = {}
    for i in range(10):
        t = f'A{i}'
        rows.append({
            'ticker': t, 'value_chf': 100.0,
            'weight_pct': 10.0, 'category': 'core'})
        cols[t] = pd.Series(
            100 + rng.normal(0, 0.5, n).cumsum(), index=idx)
    holdings = pd.DataFrame(rows)
    prices = pd.DataFrame(cols)
    s = get_strategy('min_variance')
    w = s.propose(holdings, prices, default_constraints())
    assert max(w.values()) <= (
        default_constraints().max_weight_default + 1e-6)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)


def test_mvo_returns_valid_weight_dict():
    """MVO produces long-only weights summing to 1, capped."""
    import numpy as np
    from src.shared.constraints import RebalanceConstraints
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(7)
    rows = []
    cols = {}
    profiles = [(0.10, 0.5), (0.05, 0.3), (0.15, 1.0),
                (0.08, 0.4), (0.12, 0.6)]
    for i, (mu, sig) in enumerate(profiles):
        t = f'A{i}'
        rows.append({
            'ticker': t, 'value_chf': 200.0,
            'weight_pct': 20.0, 'category': 'core'})
        daily_returns = rng.normal(
            mu / 252.0, sig / np.sqrt(252.0), n)
        cols[t] = pd.Series(
            100.0 * (1 + pd.Series(daily_returns)).cumprod().values,
            index=idx)
    holdings = pd.DataFrame(rows)
    prices = pd.DataFrame(cols)
    relaxed = RebalanceConstraints(max_weight_default=0.30)
    s = get_strategy('mvo')
    w = s.propose(holdings, prices, relaxed, risk_free_rate=0.005)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(v >= -1e-9 for v in w.values())
    assert max(w.values()) <= (
        relaxed.max_weight_default + 1e-6)


def test_mvo_uses_live_rfr_when_none(monkeypatch):
    """risk_free_rate=None triggers get_live_risk_free_rate."""
    import numpy as np
    from src.shared.constraints import RebalanceConstraints
    called = {'n': 0}

    def fake_rfr(*args, **kwargs):
        called['n'] += 1
        return 0.005

    monkeypatch.setattr(
        'src.modelling.rebalancing.strategies.mvo.'
        'get_live_risk_free_rate', fake_rfr)
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(0)
    holdings = pd.DataFrame([
        {'ticker': 'A', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
        {'ticker': 'B', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    prices = pd.DataFrame({
        'A': pd.Series(
            100 + rng.normal(0, 1, n).cumsum(), index=idx),
        'B': pd.Series(
            100 + rng.normal(0, 0.5, n).cumsum(), index=idx)})
    relaxed = RebalanceConstraints(max_weight_default=1.0)
    s = get_strategy('mvo')
    s.propose(holdings, prices, relaxed)
    assert called['n'] == 1


def test_strategy_registry_after_phase_4():
    """Phase 4 expands the registry to seven strategies."""
    expected = {
        'mvo', 'equal_weight', 'inverse_vol', 'min_variance',
        'max_growth', 'risk_adjusted', 'min_risk'}
    assert set(strategy_registry) == expected


def test_mvo_raises_on_slsqp_infeasibility():
    """Infeasible per-name caps surface as a SLSQP convergence error.

    Two tickers with default 15% cap each can't satisfy the
    sum-to-1 equality (max achievable 30%). The optimiser must
    fail loudly instead of silently returning a partial weight
    vector.
    """
    import numpy as np
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(13)
    holdings = pd.DataFrame([
        {'ticker': 'A', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
        {'ticker': 'B', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    prices = pd.DataFrame({
        'A': pd.Series(
            100 + rng.normal(0, 1, n).cumsum(), index=idx),
        'B': pd.Series(
            100 + rng.normal(0, 1, n).cumsum(), index=idx),
    })
    s = get_strategy('mvo')
    with pytest.raises(
        RuntimeError, match='SLSQP failed to converge'):
        s.propose(holdings, prices, default_constraints(),
                  risk_free_rate=0.005)


def test_mvo_raises_when_too_few_tickers_survive_nan_filter():
    """Insufficient covariance overlap drops every ticker.

    With only 10 daily observations the covariance returns NaN
    everywhere (min_periods is 30). The valid-mask filter must
    drop all tickers and raise loudly so the caller knows the
    optimisation cannot proceed.
    """
    import numpy as np
    from src.shared.constraints import RebalanceConstraints
    n = 10
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(0)
    holdings = pd.DataFrame([
        {'ticker': 'A', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
        {'ticker': 'B', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    prices = pd.DataFrame({
        'A': pd.Series(
            100 + rng.normal(0, 1, n).cumsum(), index=idx),
        'B': pd.Series(
            100 + rng.normal(0, 1, n).cumsum(), index=idx),
    })
    s = get_strategy('mvo')
    with pytest.raises(RuntimeError, match='NaN filter'):
        s.propose(
            holdings, prices,
            RebalanceConstraints(max_weight_default=1.0),
            risk_free_rate=0.005)


def test_min_variance_raises_on_slsqp_infeasibility():
    """Infeasible per-name caps surface as a SLSQP convergence error.

    Two tickers with default 15% cap each can't satisfy the
    sum-to-1 equality. The optimiser must fail loudly instead
    of silently returning a partial weight vector.
    """
    import numpy as np
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(13)
    holdings = pd.DataFrame([
        {'ticker': 'A', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
        {'ticker': 'B', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    prices = pd.DataFrame({
        'A': pd.Series(
            100 + rng.normal(0, 1, n).cumsum(), index=idx),
        'B': pd.Series(
            100 + rng.normal(0, 1, n).cumsum(), index=idx),
    })
    s = get_strategy('min_variance')
    with pytest.raises(
        RuntimeError, match='SLSQP failed to converge'):
        s.propose(holdings, prices, default_constraints())


def test_min_variance_raises_when_too_few_tickers_survive_nan_filter():
    """Insufficient covariance overlap drops every ticker.

    With only 10 daily observations the covariance returns NaN
    everywhere (min_periods is 30). The valid-mask filter must
    drop all tickers and raise loudly so the caller knows the
    optimisation cannot proceed.
    """
    import numpy as np
    from src.shared.constraints import RebalanceConstraints
    n = 10
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(0)
    holdings = pd.DataFrame([
        {'ticker': 'A', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
        {'ticker': 'B', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    prices = pd.DataFrame({
        'A': pd.Series(
            100 + rng.normal(0, 1, n).cumsum(), index=idx),
        'B': pd.Series(
            100 + rng.normal(0, 1, n).cumsum(), index=idx),
    })
    s = get_strategy('min_variance')
    with pytest.raises(RuntimeError, match='NaN filter'):
        s.propose(
            holdings, prices,
            RebalanceConstraints(max_weight_default=1.0))
