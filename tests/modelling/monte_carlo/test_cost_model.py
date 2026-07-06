"""Tests for TransactionCostModel."""

import numpy as np
import pytest

from src.modelling.monte_carlo.cost_model import (
    TransactionCostModel, ibkr_default_cost_model)


def test_default_factory_returns_dataclass_with_spec_values():
    m = ibkr_default_cost_model()
    assert isinstance(m, TransactionCostModel)
    # Per-share (fees_breakdown path)
    assert m.commission_per_share_native == 0.0035
    assert m.commission_min_native == 0.35
    assert m.commission_cap_pct == 0.01
    assert m.exchange_clearing_per_share_native == 0.0034
    assert m.sec_fee_rate == 0.0000206
    assert m.finra_taf_per_share == 0.000195
    # FX (used only by cost_chf / MC; fees_breakdown ignores FX)
    assert m.fx_cost_bps == 2.0
    # Bps-of-notional fallback (cost_chf path)
    assert m.commission_bps == 7.0
    assert m.commission_min_chf == 1.5


def test_swiss_ticker_chf_no_fx_cost():
    """NESN.SW (CHF) - swiss stamp on both sides, no FX."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(1000.0, 'NESN.SW', 'CHF')
    # commission = max(1000 * 7/1e4, 1.5) = 1.5
    # spread from venue_taxes: half_spread_bps = 3.0
    # spread = 1000 * 3/1e4 = 0.3
    # avg stamp = (7.5 + 7.5)/2 = 7.5 -> tax = 1000 * 7.5/1e4 = 0.75
    # fx = 0 (CHF)
    assert cost == pytest.approx(1.5 + 0.3 + 0.75)


def test_us_ticker_usd_includes_fx_cost():
    """AAPL (USD) - US default venue: no tax, 2 bps spread, FX."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(1000.0, 'AAPL', 'USD')
    # commission = 1.5
    # spread = 1000 * 2/1e4 = 0.2
    # tax = 0 (US default has buy_bps = sell_bps = 0)
    # fx = 1000 * 2/1e4 = 0.2
    assert cost == pytest.approx(1.5 + 0.2 + 0.2)


def test_uk_ticker_uses_venue_stamp_on_average():
    """RR.L - UK stamp 50 bps on buy only -> avg = 25 bps."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(1000.0, 'RR.L', 'GBP')
    # commission = 1.5
    # spread = 1000 * 3/1e4 = 0.3
    # avg tax = (50 + 0)/2 = 25 -> tax = 1000 * 25/1e4 = 2.5
    # fx = 1000 * 2/1e4 = 0.2
    assert cost == pytest.approx(1.5 + 0.3 + 2.5 + 0.2)


def test_small_trade_hits_commission_floor():
    """A 10 CHF trade pays the 1.5 CHF commission floor."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(10.0, 'NESN.SW', 'CHF')
    # commission_proportional = 10 * 7/1e4 = 0.007 -> floored at 1.5
    # spread = 10 * 3/1e4 = 0.003
    # tax = 10 * 7.5/1e4 = 0.0075
    assert cost == pytest.approx(1.5 + 0.003 + 0.0075)


def test_dataclass_is_frozen():
    m = ibkr_default_cost_model()
    with pytest.raises(Exception):
        m.commission_bps = 99.0


def test_custom_construction():
    """Override defaults to model a different broker."""
    m = TransactionCostModel(
        commission_bps=10.0, commission_min_chf=2.0,
        fx_cost_bps=0.0)
    assert m.commission_bps == 10.0
    cost = m.cost_chf(1000.0, 'NESN.SW', 'CHF')
    # commission = max(1.0, 2.0) = 2.0
    # spread = 1000 * 3/1e4 = 0.3
    # tax = 1000 * 7.5/1e4 = 0.75
    # fx = 0 (override)
    assert cost == pytest.approx(2.0 + 0.3 + 0.75)


def test_cost_chf_array_matches_scalar():
    """Vectorised cost matches scalar for each element."""
    m = ibkr_default_cost_model()
    trade_values = np.array([10.0, 100.0, 1000.0, 5000.0])
    vec = m.cost_chf_array(trade_values, 'AAPL', 'USD')
    scalar = np.array([
        m.cost_chf(v, 'AAPL', 'USD') for v in trade_values])
    np.testing.assert_allclose(vec, scalar, rtol=1e-12)


def test_cost_chf_array_swiss_ticker():
    """Swiss ticker uses swiss stamp; CHF currency suppresses FX."""
    m = ibkr_default_cost_model()
    trade_values = np.array([1000.0, 2000.0])
    vec = m.cost_chf_array(trade_values, 'NESN.SW', 'CHF')
    scalar = np.array([
        m.cost_chf(v, 'NESN.SW', 'CHF') for v in trade_values])
    np.testing.assert_allclose(vec, scalar, rtol=1e-12)


def test_fees_breakdown_us_buy_no_regulatory():
    """AAPL buy: commission + clearing + FX; no SEC/FINRA/tax."""
    m = ibkr_default_cost_model()
    # 100 shares @ USD 50 = USD 5000 notional; assume 1 USD = 0.9 CHF
    result = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='USD', ticker='AAPL', side='buy',
        fx_rate_to_chf=0.9)
    # commission_native = max(0.35, 100*0.0035) = 0.35 (100*0.0035=0.35)
    # exchange_clearing_native = 100 * 0.0034 = 0.34
    # regulatory_native = 0 (buy)
    # venue_tax_native = 0 (US default)
    # no FX component (matched-currency legs)
    assert result['commission_chf'] == pytest.approx(0.35 * 0.9)
    assert result['exchange_clearing_chf'] == pytest.approx(0.34 * 0.9)
    assert result['regulatory_chf'] == 0.0
    assert result['venue_tax_chf'] == 0.0
    assert 'fx_chf' not in result


def test_fees_breakdown_us_sell_adds_regulatory():
    """AAPL sell: same as buy plus SEC + FINRA TAF."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='USD', ticker='AAPL', side='sell',
        fx_rate_to_chf=0.9)
    # regulatory_native = 0.0000206*5000 + 0.000195*100 = 0.103 + 0.0195
    expected_regulatory_native = (
        0.0000206 * 5000.0 + 0.000195 * 100.0)
    assert result['regulatory_chf'] == pytest.approx(
        expected_regulatory_native * 0.9)


def test_fees_breakdown_uk_buy_charges_stamp():
    """RR.L buy: 50 bps UK stamp on the buy leg."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='GBP', ticker='RR.L', side='buy',
        fx_rate_to_chf=1.1)
    # venue_tax_native = 5000 * 50/1e4 = 25.0
    assert result['venue_tax_chf'] == pytest.approx(25.0 * 1.1)


def test_fees_breakdown_uk_sell_no_stamp():
    """RR.L sell: no UK stamp (buy-only tax)."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='GBP', ticker='RR.L', side='sell',
        fx_rate_to_chf=1.1)
    assert result['venue_tax_chf'] == 0.0


def test_fees_breakdown_swiss_taxes_both_sides():
    """NESN.SW: swiss stamp fires on both buy and sell."""
    m = ibkr_default_cost_model()
    buy = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='CHF', ticker='NESN.SW', side='buy',
        fx_rate_to_chf=1.0)
    sell = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='CHF', ticker='NESN.SW', side='sell',
        fx_rate_to_chf=1.0)
    # 7.5 bps each side; 5000 * 7.5/1e4 = 3.75 CHF
    assert buy['venue_tax_chf'] == pytest.approx(3.75)
    assert sell['venue_tax_chf'] == pytest.approx(3.75)


def test_fees_breakdown_no_fx_component():
    """fees_breakdown does not include FX; matched-currency legs."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='USD', ticker='AAPL', side='buy',
        fx_rate_to_chf=0.9)
    # The return dict has no fx_chf key regardless of currency.
    assert 'fx_chf' not in result


def test_fees_breakdown_commission_cap_binds():
    """Very small trade with many shares: 1% cap engages."""
    m = ibkr_default_cost_model()
    # 1000 shares @ USD 0.20 = USD 200 notional
    # per-share = 1000 * 0.0035 = 3.5; min = 0.35; unclipped = 3.5
    # cap = 0.01 * 200 = 2.0; commission = min(3.5, 2.0) = 2.0
    result = m.fees_breakdown(
        shares=1000.0, trade_value_native=200.0,
        currency='USD', ticker='AAPL', side='buy',
        fx_rate_to_chf=0.9)
    assert result['commission_chf'] == pytest.approx(2.0 * 0.9)


def test_fees_breakdown_invalid_side_raises():
    m = ibkr_default_cost_model()
    with pytest.raises(ValueError):
        m.fees_breakdown(
            shares=100.0, trade_value_native=5000.0,
            currency='USD', ticker='AAPL', side='hold',
            fx_rate_to_chf=0.9)


def test_fees_breakdown_zero_notional_zero_bps():
    """total_bps guards against division by zero."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=0.0, trade_value_native=0.0,
        currency='CHF', ticker='NESN.SW', side='buy',
        fx_rate_to_chf=1.0)
    assert result['total_bps'] == 0.0


def test_fees_breakdown_total_is_sum_of_components():
    """total_chf equals the sum of the five component keys."""
    m = ibkr_default_cost_model()
    r = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='USD', ticker='AAPL', side='sell',
        fx_rate_to_chf=0.9)
    expected = (
        r['commission_chf'] + r['exchange_clearing_chf']
        + r['regulatory_chf'] + r['venue_tax_chf'])
    assert r['total_chf'] == pytest.approx(expected)
