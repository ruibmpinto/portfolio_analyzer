"""Tests for TransactionCostModel."""

import numpy as np
import pytest

from src.modelling.monte_carlo.cost_model import (
    TransactionCostModel, degiro_default_cost_model,
    ibkr_default_cost_model)


def test_default_factory_returns_dataclass_with_spec_values():
    """The trimmed dataclass exposes only cross-venue constants."""
    m = ibkr_default_cost_model()
    assert isinstance(m, TransactionCostModel)
    # US regulatory (statutory, cross-venue)
    assert m.sec_fee_rate == 0.0000206
    assert m.finra_taf_per_share == 0.000195
    # FX (MC only; fees_breakdown ignores FX)
    assert m.fx_cost_bps == 2.0
    # Bps-of-notional MC fallback
    assert m.commission_bps == 7.0
    assert m.commission_min_chf == 1.5


def test_dataclass_is_frozen():
    m = ibkr_default_cost_model()
    with pytest.raises(Exception):
        m.commission_bps = 99.0


# ---------------------------------------------------------------------
# cost_chf — MC symmetric path (bps-of-notional commission proxy)
# ---------------------------------------------------------------------


def test_cost_chf_swiss_ticker():
    """SIX (via IBKR-UK) has zero stamp; venue half-spread 3 bps."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(1000.0, 'NESN.SW', 'CHF')
    # commission = max(1000*7/1e4, 1.5) = 1.5
    # spread = 1000*3/1e4 = 0.3
    # avg tax = (0+0)/2 = 0
    # fx = 0 (CHF)
    assert cost == pytest.approx(1.5 + 0.3)


def test_cost_chf_us_ticker_includes_fx():
    """US (AAPL) - no tax, 2 bps spread, FX cost fires (USD)."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(1000.0, 'AAPL', 'USD')
    # commission = 1.5
    # spread = 1000*2/1e4 = 0.2
    # tax = 0
    # fx = 1000*2/1e4 = 0.2
    assert cost == pytest.approx(1.5 + 0.2 + 0.2)


def test_cost_chf_uk_avg_stamp():
    """RR.L (UK-incorporated) - avg stamp 25 bps + FX."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(1000.0, 'RR.L', 'GBP')
    # commission = 1.5
    # spread = 1000*3/1e4 = 0.3
    # avg tax = (50+0)/2 = 25 -> 1000*25/1e4 = 2.5
    # fx = 1000*2/1e4 = 0.2
    assert cost == pytest.approx(1.5 + 0.3 + 2.5 + 0.2)


def test_cost_chf_small_trade_hits_min():
    """A 10 CHF trade pays the 1.5 CHF commission floor."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(10.0, 'NESN.SW', 'CHF')
    # commission proportional = 10*7/1e4 = 0.007 -> floored at 1.5
    # spread = 10*3/1e4 = 0.003
    # tax = 0
    assert cost == pytest.approx(1.5 + 0.003)


def test_cost_chf_custom_construction():
    """Override the MC-side constants."""
    m = TransactionCostModel(
        commission_bps=10.0, commission_min_chf=2.0,
        fx_cost_bps=0.0)
    cost = m.cost_chf(1000.0, 'NESN.SW', 'CHF')
    # commission = max(1.0, 2.0) = 2.0; spread 0.3; tax 0; fx 0
    assert cost == pytest.approx(2.0 + 0.3)


def test_cost_chf_ntf_mutual_fund_is_zero():
    """NTF funds return 0 without touching the schedule."""
    m = ibkr_default_cost_model()
    # IE00BFZP7W55 is in the NTF list
    assert m.cost_chf(5000.0, 'IE00BFZP7W55', 'CHF') == 0.0


def test_cost_chf_broadcasts_over_array():
    """Passing an array returns per-element costs of the same shape."""
    m = ibkr_default_cost_model()
    trade_values = np.array([10.0, 100.0, 1000.0, 5000.0])
    vec = m.cost_chf(trade_values, 'AAPL', 'USD')
    scalar = np.array([
        float(m.cost_chf(v, 'AAPL', 'USD')) for v in trade_values])
    np.testing.assert_allclose(vec, scalar, rtol=1e-12)


def test_cost_chf_ntf_returns_zero_array():
    """NTF short-circuit yields all-zero output for vector input."""
    m = ibkr_default_cost_model()
    result = m.cost_chf(
        np.array([1000.0, 2000.0]), 'IE00BFZP7W55', 'CHF')
    np.testing.assert_array_equal(result, np.zeros(2))


# ---------------------------------------------------------------------
# fees_breakdown — per-venue schedule (exact IBKR schedule)
# ---------------------------------------------------------------------


def test_fees_breakdown_us_buy_no_regulatory():
    """AAPL buy: per-share commission + NSCC extras, no tax/regulatory."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='USD', ticker='AAPL', side='buy',
        fx_rate_to_chf=0.9)
    # US schedule: per-share 0.0035, min 0.35, cap 1%, extras 0.0034/share
    # commission_native = max(0.35, 100*0.0035) = 0.35
    # broker_extras_native = 100*0.0034 = 0.34
    assert result['commission_chf'] == pytest.approx(0.35 * 0.9)
    assert result['broker_extras_chf'] == pytest.approx(0.34 * 0.9)
    assert result['regulatory_chf'] == 0.0
    assert result['venue_tax_chf'] == 0.0


def test_fees_breakdown_us_sell_adds_regulatory():
    """AAPL sell: SEC + FINRA TAF fire on the sell leg."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='USD', ticker='AAPL', side='sell',
        fx_rate_to_chf=0.9)
    expected_reg_native = (
        0.0000206 * 5000.0 + 0.000195 * 100.0)
    assert result['regulatory_chf'] == pytest.approx(
        expected_reg_native * 0.9)


def test_fees_breakdown_uk_buy_charges_stamp():
    """RR.L buy: 50 bps UK stamp on the buy leg."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='GBP', ticker='RR.L', side='buy',
        fx_rate_to_chf=1.1)
    assert result['venue_tax_chf'] == pytest.approx(25.0 * 1.1)


def test_fees_breakdown_uk_sell_no_stamp():
    """RR.L sell: no UK stamp (buy-only tax)."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='GBP', ticker='RR.L', side='sell',
        fx_rate_to_chf=1.1)
    assert result['venue_tax_chf'] == 0.0


def test_fees_breakdown_ucits_lse_stamp_exempt():
    """VUAA.L is a UCITS ETF - no UK stamp on either side."""
    m = ibkr_default_cost_model()
    for side in ('buy', 'sell'):
        result = m.fees_breakdown(
            shares=12.0, trade_value_native=1511.76,
            currency='USD', ticker='VUAA.L', side=side,
            fx_rate_to_chf=0.9)
        assert result['venue_tax_chf'] == 0.0


def test_fees_breakdown_lse_usd_min_binds():
    """LSE USD-denominated: min 1.70 USD binds on small trades."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=12.0, trade_value_native=1511.76,
        currency='USD', ticker='VUAA.L', side='buy',
        fx_rate_to_chf=0.9)
    # 5 bps * 1511.76 = 0.756, below min 1.70 -> commission = 1.70
    assert result['commission_chf'] == pytest.approx(1.70 * 0.9)


def test_fees_breakdown_swiss_venue_stamp_zero_via_ibkr_uk():
    """SIX schedule (via IBKR-UK) has stamp = 0 by design."""
    m = ibkr_default_cost_model()
    for side in ('buy', 'sell'):
        result = m.fees_breakdown(
            shares=54.0, trade_value_native=3991.68,
            currency='CHF', ticker='HOLN.SW', side=side,
            fx_rate_to_chf=1.0)
        assert result['venue_tax_chf'] == 0.0


def test_fees_breakdown_swiss_commission_seven_bps_of_value():
    """HOLN.SW: 7 bps * value, above the CHF 1.50 min."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=54.0, trade_value_native=3991.68,
        currency='CHF', ticker='HOLN.SW', side='buy',
        fx_rate_to_chf=1.0)
    # 7 bps * 3991.68 = 2.794 CHF, above min 1.50
    assert result['commission_chf'] == pytest.approx(2.7942, rel=1e-4)


def test_fees_breakdown_ntf_mutual_fund_zero():
    """IE00BFZP7W55 short-circuits to all zeros."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=21.34, trade_value_native=5000.0,
        currency='CHF', ticker='IE00BFZP7W55', side='buy',
        fx_rate_to_chf=1.0)
    assert result['total_chf'] == 0.0
    assert result['total_bps'] == 0.0


def test_fees_breakdown_no_fx_component():
    """No fx_chf field even for a USD trade."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='USD', ticker='AAPL', side='buy',
        fx_rate_to_chf=0.9)
    assert 'fx_chf' not in result


def test_fees_breakdown_commission_cap_binds():
    """Very small trade with many shares: 1% cap engages."""
    m = ibkr_default_cost_model()
    # 1000 shares @ USD 0.20 = USD 200 notional
    # per-share = 1000*0.0035 = 3.5; min 0.35; unclipped = 3.5
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


def test_fees_breakdown_unmapped_venue_raises():
    """Unknown (ticker, currency) pair is a hard error."""
    m = ibkr_default_cost_model()
    with pytest.raises(RuntimeError, match='No ibkr schedule'):
        m.fees_breakdown(
            shares=100.0, trade_value_native=5000.0,
            currency='JPY', ticker='7203.T', side='buy',
            fx_rate_to_chf=0.008)


def test_fees_breakdown_zero_notional_zero_bps():
    """total_bps guards against division by zero."""
    m = ibkr_default_cost_model()
    result = m.fees_breakdown(
        shares=0.0, trade_value_native=0.0,
        currency='CHF', ticker='NESN.SW', side='buy',
        fx_rate_to_chf=1.0)
    assert result['total_bps'] == 0.0


def test_fees_breakdown_total_is_sum_of_components():
    """total_chf equals the sum of the four component keys."""
    m = ibkr_default_cost_model()
    r = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='USD', ticker='AAPL', side='sell',
        fx_rate_to_chf=0.9)
    expected = (
        r['commission_chf'] + r['broker_extras_chf']
        + r['regulatory_chf'] + r['venue_tax_chf'])
    assert r['total_chf'] == pytest.approx(expected)


# ---------------------------------------------------------------------
# Degiro tariff (Swiss-resident, 2026-01-01)
# ---------------------------------------------------------------------


def test_degiro_default_factory_returns_expected_defaults():
    """degiro_default_cost_model wires the Degiro broker + FX cost."""
    m = degiro_default_cost_model()
    assert m.broker == 'degiro'
    # Regulatory bundled into Degiro's EUR 1 handling fee
    assert m.sec_fee_rate == 0.0
    assert m.finra_taf_per_share == 0.0
    # 0.25% AutoFX = 25 bps
    assert m.fx_cost_bps == 25.0


def test_degiro_swiss_venue_flat_six_chf():
    """SIX at Degiro: flat CHF 5 + CHF 1 handling = CHF 6."""
    m = degiro_default_cost_model()
    r = m.fees_breakdown(
        shares=54.0, trade_value_native=3991.68,
        currency='CHF', ticker='HOLN.SW', side='buy',
        fx_rate_to_chf=1.0)
    assert r['commission_chf'] == pytest.approx(6.00)
    assert r['venue_tax_chf'] == 0.0
    assert r['regulatory_chf'] == 0.0


def test_degiro_us_flat_billed_in_eur_needs_commission_fx():
    """AAPL at Degiro: EUR 2 flat, must convert via commission FX."""
    m = degiro_default_cost_model()
    r = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='USD', ticker='AAPL', side='buy',
        fx_rate_to_chf=0.90,
        commission_fx_rate_to_chf=0.95)
    # 2 EUR * 0.95 EUR/CHF = 1.90 CHF
    assert r['commission_chf'] == pytest.approx(1.90)
    # No US regulatory at Degiro (bundled into handling)
    assert r['regulatory_chf'] == 0.0
    assert r['venue_tax_chf'] == 0.0


def test_degiro_us_sell_no_regulatory_passthrough():
    """Sell leg on Degiro US carries no SEC/FINRA (bundled)."""
    m = degiro_default_cost_model()
    r = m.fees_breakdown(
        shares=100.0, trade_value_native=5000.0,
        currency='USD', ticker='AAPL', side='sell',
        fx_rate_to_chf=0.90,
        commission_fx_rate_to_chf=0.95)
    assert r['regulatory_chf'] == 0.0
    # Sell fee equals buy fee (symmetric at Degiro US)
    assert r['commission_chf'] == pytest.approx(1.90)


def test_degiro_missing_commission_fx_raises():
    """Cross-currency Degiro row without commission_fx raises."""
    m = degiro_default_cost_model()
    with pytest.raises(
            RuntimeError, match='commission_fx_rate_to_chf'):
        m.fees_breakdown(
            shares=100.0, trade_value_native=5000.0,
            currency='USD', ticker='AAPL', side='buy',
            fx_rate_to_chf=0.90)


def test_degiro_uk_stamp_still_applies():
    """RR.L at Degiro: 50 bps UK stamp on buy, passed to client."""
    m = degiro_default_cost_model()
    r = m.fees_breakdown(
        shares=100.0, trade_value_native=1000.0,
        currency='GBP', ticker='RR.L', side='buy',
        fx_rate_to_chf=1.10,
        commission_fx_rate_to_chf=0.95)
    # UK stamp = 50 bps * 1000 GBP = 5 GBP; * 1.10 = 5.50 CHF
    assert r['venue_tax_chf'] == pytest.approx(5.50)
    # Sell leg carries no stamp
    r_sell = m.fees_breakdown(
        shares=100.0, trade_value_native=1000.0,
        currency='GBP', ticker='RR.L', side='sell',
        fx_rate_to_chf=1.10,
        commission_fx_rate_to_chf=0.95)
    assert r_sell['venue_tax_chf'] == 0.0


def test_degiro_ucits_lse_stamp_exempt():
    """VUAA.L at Degiro: UCITS ETF exemption zeros UK stamp AND
    triggers the ETF flat-fee override (EUR 2 tracker + EUR 1
    handling = EUR 3 vs EUR 4.90 for a stock)."""
    m = degiro_default_cost_model()
    r = m.fees_breakdown(
        shares=12.0, trade_value_native=1511.76,
        currency='USD', ticker='VUAA.L', side='buy',
        fx_rate_to_chf=0.90,
        commission_fx_rate_to_chf=0.95)
    assert r['venue_tax_chf'] == 0.0
    # Flat EUR 3.00 * 0.95 = 2.85 CHF
    assert r['commission_chf'] == pytest.approx(2.85)


def test_degiro_paris_ftt_on_buys():
    """TTE.PA at Degiro: 40 bps FR FTT on buy."""
    m = degiro_default_cost_model()
    r = m.fees_breakdown(
        shares=100.0, trade_value_native=6000.0,
        currency='EUR', ticker='TTE.PA', side='buy',
        fx_rate_to_chf=0.95)
    # 40 bps * 6000 EUR = 24 EUR * 0.95 = 22.80 CHF
    assert r['venue_tax_chf'] == pytest.approx(22.80)
    # Flat EUR 4.90 * 0.95 = 4.655 CHF (same-currency: no
    # commission_fx needed)
    assert r['commission_chf'] == pytest.approx(4.655)


def test_degiro_euro_venue_no_commission_fx_needed():
    """EUR trade at Degiro EUR-billed row: fx_rate_to_chf suffices."""
    m = degiro_default_cost_model()
    r = m.fees_breakdown(
        shares=10.0, trade_value_native=1000.0,
        currency='EUR', ticker='SAP.DE', side='buy',
        fx_rate_to_chf=0.95)
    # commission_currency is None for .DE (billed in EUR = trade
    # currency), so the fallback path uses fx_rate_to_chf.
    assert r['commission_chf'] == pytest.approx(4.90 * 0.95)


def test_degiro_flat_dominates_at_small_trade():
    """A CHF 100 trade at SIX still pays the full CHF 6 flat."""
    m = degiro_default_cost_model()
    r = m.fees_breakdown(
        shares=1.0, trade_value_native=100.0,
        currency='CHF', ticker='NESN.SW', side='buy',
        fx_rate_to_chf=1.0)
    assert r['commission_chf'] == pytest.approx(6.00)


def test_degiro_cost_chf_scalar_and_array():
    """cost_chf uses the Degiro half-spread + MC bps proxy."""
    m = degiro_default_cost_model()
    scalar = m.cost_chf(1000.0, 'NESN.SW', 'CHF')
    vec = m.cost_chf(np.array([1000.0, 1000.0]), 'NESN.SW', 'CHF')
    np.testing.assert_allclose(vec, np.full(2, scalar))


def test_degiro_etf_override_on_xetra():
    """XAIX.DE at Degiro: ETF flat-fee override drops EUR 4.90 to EUR 3."""
    m = degiro_default_cost_model()
    r = m.fees_breakdown(
        shares=1.0, trade_value_native=100.0,
        currency='EUR', ticker='XAIX.DE', side='buy',
        fx_rate_to_chf=0.95)
    # ETF override: EUR 3 * 0.95 = 2.85 CHF, not EUR 4.90 * 0.95
    assert r['commission_chf'] == pytest.approx(2.85)


def test_ibkr_etf_override_does_not_touch_ibkr_commission():
    """The Degiro ETF override does not leak into the IBKR schedule."""
    m = ibkr_default_cost_model()
    # VUAA.L at IBKR still uses the ('.L','USD') row: bps=5,
    # min USD 1.70. Commission_flat stays 0.
    r = m.fees_breakdown(
        shares=12.0, trade_value_native=1511.76,
        currency='USD', ticker='VUAA.L', side='buy',
        fx_rate_to_chf=0.9)
    # min 1.70 USD binds (5 bps * 1511.76 = 0.756)
    assert r['commission_chf'] == pytest.approx(1.70 * 0.9)


def test_degiro_unmapped_venue_raises():
    """Unknown venue at Degiro raises with broker name in message."""
    m = degiro_default_cost_model()
    with pytest.raises(RuntimeError, match='degiro'):
        m.fees_breakdown(
            shares=100.0, trade_value_native=5000.0,
            currency='JPY', ticker='7203.T', side='buy',
            fx_rate_to_chf=0.008,
            commission_fx_rate_to_chf=0.95)
