"""Tests for TransactionCostModel."""

import pytest

from src.modelling.monte_carlo.cost_model import (
    TransactionCostModel, ibkr_default_cost_model)


def test_default_factory_returns_dataclass_with_spec_values():
    m = ibkr_default_cost_model()
    assert isinstance(m, TransactionCostModel)
    assert m.commission_bps == 7.0
    assert m.commission_min_chf == 1.5
    assert m.spread_bps == 5.0
    assert m.stamp_duty_bps_swiss == 7.5
    assert m.stamp_duty_bps_foreign == 15.0
    assert m.fx_cost_bps == 2.0


def test_swiss_ticker_chf_no_fx_cost():
    """NESN.SW (CHF) -> no FX cost."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(1000.0, 'NESN.SW', 'CHF')
    # commission = max(1000 * 7/1e4, 1.5) = 1.5
    # spread = 1000 * 5/1e4 = 0.5
    # stamp = 1000 * 7.5/1e4 = 0.75
    # fx = 0
    assert cost == pytest.approx(1.5 + 0.5 + 0.75)


def test_us_ticker_usd_includes_fx_cost():
    """AAPL (USD) -> foreign stamp + FX cost."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(1000.0, 'AAPL', 'USD')
    # commission = max(1000 * 7/1e4, 1.5) = 1.5
    # spread = 0.5
    # stamp = 1000 * 15/1e4 = 1.5
    # fx = 1000 * 2/1e4 = 0.2
    assert cost == pytest.approx(1.5 + 0.5 + 1.5 + 0.2)


def test_small_trade_hits_commission_floor():
    """A 10 CHF trade pays the 1.5 CHF commission floor."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(10.0, 'NESN.SW', 'CHF')
    # commission_proportional = 10 * 7/1e4 = 0.007 -> floored at 1.5
    # spread = 10 * 5/1e4 = 0.005
    # stamp = 10 * 7.5/1e4 = 0.0075
    assert cost == pytest.approx(1.5 + 0.005 + 0.0075)


def test_dataclass_is_frozen():
    m = ibkr_default_cost_model()
    with pytest.raises(Exception):
        m.commission_bps = 99.0


def test_custom_construction():
    """Override defaults to model a different broker."""
    m = TransactionCostModel(
        commission_bps=10.0, commission_min_chf=2.0,
        spread_bps=3.0, stamp_duty_bps_swiss=7.5,
        stamp_duty_bps_foreign=15.0, fx_cost_bps=0.0)
    assert m.commission_bps == 10.0
    cost = m.cost_chf(1000.0, 'NESN.SW', 'CHF')
    # commission = max(1.0, 2.0) = 2.0; spread = 0.3; stamp = 0.75; fx = 0
    assert cost == pytest.approx(2.0 + 0.3 + 0.75)


def test_cost_chf_array_matches_scalar():
    """Vectorised cost matches scalar for each element."""
    import numpy as np
    m = ibkr_default_cost_model()
    trade_values = np.array([10.0, 100.0, 1000.0, 5000.0])
    vec = m.cost_chf_array(trade_values, 'AAPL', 'USD')
    scalar = np.array([
        m.cost_chf(v, 'AAPL', 'USD') for v in trade_values])
    np.testing.assert_allclose(vec, scalar, rtol=1e-12)


def test_cost_chf_array_swiss_ticker():
    """Swiss ticker uses swiss stamp; CHF currency suppresses FX."""
    import numpy as np
    m = ibkr_default_cost_model()
    trade_values = np.array([1000.0, 2000.0])
    vec = m.cost_chf_array(trade_values, 'NESN.SW', 'CHF')
    scalar = np.array([
        m.cost_chf(v, 'NESN.SW', 'CHF') for v in trade_values])
    np.testing.assert_allclose(vec, scalar, rtol=1e-12)
