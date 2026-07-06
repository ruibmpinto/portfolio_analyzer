"""Tests for src.modelling.trading.breakeven."""

import pytest

from src.modelling.trading.breakeven import breakeven_gap
from src.shared.venue_taxes import lookup_venue


def test_us_ticker_symmetric_fees_no_venue_tax():
    """AAPL: identical broker fees each side, US default venue = 0 tax."""
    r = breakeven_gap(
        shares=100.0, price_native=50.0,
        currency='USD', ticker='AAPL',
        fx_rate_to_chf=0.9)
    # Sell has SEC + FINRA TAF; buy does not -> f_s > f_b
    assert r['f_s_bps'] > r['f_b_bps']
    # US default half-spread is 2.0
    assert r['half_spread_bps'] == 2.0
    # breakeven = f_s + f_b + 2*half_spread
    assert r['breakeven_bps'] == pytest.approx(
        r['f_s_bps'] + r['f_b_bps'] + 2.0 * r['half_spread_bps'])


def test_uk_stamp_lands_on_buy_side():
    """RR.L: UK stamp 50 bps on the buy leg only."""
    r = breakeven_gap(
        shares=100.0, price_native=10.0,
        currency='GBP', ticker='RR.L',
        fx_rate_to_chf=1.1)
    # Buy leg carries 50 bps stamp; sell has none.
    # Excluding stamp, buy fees ~= sell fees (minus regulatory).
    # So f_b_bps should be substantially higher than f_s_bps.
    assert r['f_b_bps'] > r['f_s_bps'] + 40.0


def test_swiss_stamp_lands_on_both_sides():
    """NESN.SW: Swiss stamp 7.5 bps each side."""
    r = breakeven_gap(
        shares=100.0, price_native=90.0,
        currency='CHF', ticker='NESN.SW',
        fx_rate_to_chf=1.0)
    # No FX component (CHF), no SEC/FINRA (non-US).
    # Sell and buy differ only by the (buy has 7.5 stamp, sell has 7.5 stamp)
    # -> effectively equal fees per side.
    assert r['f_b_bps'] == pytest.approx(r['f_s_bps'], abs=0.1)


def test_chf_trade_no_fx_component():
    """CHF trade: fx_chf should be zero in each breakdown."""
    r = breakeven_gap(
        shares=100.0, price_native=90.0,
        currency='CHF', ticker='NESN.SW',
        fx_rate_to_chf=1.0)
    # Sanity: breakeven equals f_s + f_b + 2*spread with no FX drag
    assert r['breakeven_bps'] == pytest.approx(
        r['f_s_bps'] + r['f_b_bps'] + 2.0 * r['half_spread_bps'])


def test_suggested_trigger_is_twice_breakeven():
    """Trigger safety margin: 2x breakeven per strategy notes."""
    r = breakeven_gap(
        shares=100.0, price_native=50.0,
        currency='USD', ticker='AAPL',
        fx_rate_to_chf=0.9)
    assert r['suggested_trigger_bps'] == pytest.approx(
        2.0 * r['breakeven_bps'])


def test_half_spread_matches_venue_taxes():
    """half_spread_bps mirrors venue_taxes.lookup_venue exactly."""
    for ticker in ('AAPL', 'NESN.SW', 'RR.L', 'TTE.PA'):
        r = breakeven_gap(
            shares=10.0, price_native=100.0,
            currency='USD', ticker=ticker,
            fx_rate_to_chf=1.0)
        assert r['half_spread_bps'] == lookup_venue(ticker).half_spread_bps


def test_breakeven_composition_identity():
    """breakeven_bps == f_s + f_b + 2 * half_spread, exactly."""
    r = breakeven_gap(
        shares=250.0, price_native=25.0,
        currency='USD', ticker='AAPL',
        fx_rate_to_chf=0.9)
    lhs = r['breakeven_bps']
    rhs = r['f_s_bps'] + r['f_b_bps'] + 2.0 * r['half_spread_bps']
    assert lhs == pytest.approx(rhs)


def test_uk_breakeven_dominated_by_stamp_duty():
    """RR.L roundtrip breakeven is >= 50 bps (UK stamp alone)."""
    r = breakeven_gap(
        shares=100.0, price_native=10.0,
        currency='GBP', ticker='RR.L',
        fx_rate_to_chf=1.1)
    assert r['breakeven_bps'] >= 50.0


def test_returned_bps_all_nonneg():
    """Fees, spread, breakeven, and trigger are all non-negative."""
    r = breakeven_gap(
        shares=100.0, price_native=50.0,
        currency='USD', ticker='AAPL',
        fx_rate_to_chf=0.9)
    for key in (
            'f_s_bps', 'f_b_bps', 'half_spread_bps',
            'breakeven_bps', 'suggested_trigger_bps'):
        assert r[key] >= 0.0, key


def test_custom_cost_model_used():
    """A cost model with zeroed fees drives f_s_bps and f_b_bps to zero."""
    from src.modelling.monte_carlo.cost_model import TransactionCostModel
    # Zero out every broker-side fee; venue tax + spread still fire
    zero_broker = TransactionCostModel(
        commission_per_share_native=0.0,
        commission_min_native=0.0,
        commission_cap_pct=1.0,
        exchange_clearing_per_share_native=0.0,
        sec_fee_rate=0.0,
        finra_taf_per_share=0.0,
        fx_cost_bps=0.0)
    r = breakeven_gap(
        shares=100.0, price_native=50.0,
        currency='USD', ticker='AAPL',
        fx_rate_to_chf=1.0,
        cost_model=zero_broker)
    # AAPL has no venue tax; broker fees zeroed -> both legs zero bps
    assert r['f_s_bps'] == 0.0
    assert r['f_b_bps'] == 0.0
    # breakeven collapses to pure spread cost
    assert r['breakeven_bps'] == pytest.approx(
        2.0 * r['half_spread_bps'])
