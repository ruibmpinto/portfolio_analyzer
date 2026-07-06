"""Tests for the GET /api/stops endpoint.

Focus on the ``_row_from_primitives`` helper since the endpoint itself
is a thin snapshot-iteration wrapper. The helper exercises the full
math (breakeven -> compounded trigger and rebuy prices).
"""

import pandas as pd
import pytest

from src.dashboard_api.routes.stops import (
    _latest_native_price, _row_from_primitives)


def test_row_returns_all_required_fields():
    """Every row carries the full bps + price + metadata field set."""
    row = _row_from_primitives(
        ticker='AAPL', shares=100.0, currency='USD',
        weight_pct=25.0, price_native=200.0,
        fx_rate_to_chf=0.9)
    required = {
        'ticker', 'shares', 'currency', 'weight_pct',
        'current_price_native',
        'f_s_bps', 'f_b_bps', 'half_spread_bps',
        'breakeven_bps', 'suggested_trigger_bps',
        'suggested_trigger_price_native',
        'suggested_rebuy_price_native'}
    assert required.issubset(row.keys())


def test_trigger_price_below_current_by_trigger_bps():
    """suggested_trigger_price = current * (1 - trigger_bps / 1e4)."""
    row = _row_from_primitives(
        ticker='AAPL', shares=100.0, currency='USD',
        weight_pct=25.0, price_native=200.0,
        fx_rate_to_chf=0.9)
    expected = 200.0 * (1.0 - row['suggested_trigger_bps'] / 1e4)
    assert row['suggested_trigger_price_native'] == pytest.approx(expected)


def test_rebuy_price_below_trigger_by_breakeven_bps_compounded():
    """rebuy = trigger * (1 - breakeven_bps / 1e4). No linear shortcut."""
    row = _row_from_primitives(
        ticker='AAPL', shares=100.0, currency='USD',
        weight_pct=25.0, price_native=200.0,
        fx_rate_to_chf=0.9)
    trigger = row['suggested_trigger_price_native']
    expected = trigger * (1.0 - row['breakeven_bps'] / 1e4)
    assert row['suggested_rebuy_price_native'] == pytest.approx(expected)


def test_uk_stop_gap_wider_than_us():
    """UK stamp duty inflates breakeven -> wider recommended trigger."""
    us = _row_from_primitives(
        ticker='AAPL', shares=100.0, currency='USD',
        weight_pct=10.0, price_native=200.0, fx_rate_to_chf=0.9)
    uk = _row_from_primitives(
        ticker='RR.L', shares=500.0, currency='GBP',
        weight_pct=10.0, price_native=10.0, fx_rate_to_chf=1.1)
    assert uk['suggested_trigger_bps'] > us['suggested_trigger_bps']


def test_expensive_stock_has_smaller_bps_than_cheap_stock():
    """Per-share fees dilute over larger trade value -> smaller bps."""
    cheap = _row_from_primitives(
        ticker='AAPL', shares=100.0, currency='USD',
        weight_pct=10.0, price_native=5.0, fx_rate_to_chf=0.9)
    expensive = _row_from_primitives(
        ticker='AAPL', shares=100.0, currency='USD',
        weight_pct=10.0, price_native=500.0, fx_rate_to_chf=0.9)
    # Same ticker, same shares, same FX -> only price differs. The
    # per-share commission + clearing + FINRA TAF spread over a 100x
    # larger trade value, so bps shrinks.
    assert expensive['breakeven_bps'] < cheap['breakeven_bps']


def test_trigger_and_rebuy_prices_are_below_current():
    """Both derived prices sit below the current market price."""
    row = _row_from_primitives(
        ticker='AAPL', shares=100.0, currency='USD',
        weight_pct=10.0, price_native=200.0, fx_rate_to_chf=0.9)
    assert row['suggested_trigger_price_native'] < 200.0
    assert (row['suggested_rebuy_price_native']
            < row['suggested_trigger_price_native'])


def test_latest_native_price_returns_last_close():
    """Helper returns the tail non-NaN value."""
    panel = pd.DataFrame({
        'AAPL': [190.0, 195.0, 200.0],
        'MSFT': [300.0, 310.0, float('nan')],
    })
    assert _latest_native_price(panel, 'AAPL') == 200.0
    # MSFT tail is NaN; dropna keeps 310.0 as last
    assert _latest_native_price(panel, 'MSFT') == 310.0


def test_latest_native_price_missing_ticker_raises():
    """Absent ticker column -> RuntimeError (no silent NaN)."""
    panel = pd.DataFrame({'AAPL': [200.0]})
    with pytest.raises(RuntimeError, match='No price history'):
        _latest_native_price(panel, 'NOPE')


def test_latest_native_price_empty_series_raises():
    """All-NaN column -> RuntimeError (no silent NaN)."""
    panel = pd.DataFrame({'AAPL': [float('nan'), float('nan')]})
    with pytest.raises(RuntimeError, match='empty after dropna'):
        _latest_native_price(panel, 'AAPL')


def test_route_registered_on_app():
    """/api/stops is reachable through the FastAPI router."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from src.dashboard_api.context import DashboardContext
    from src.dashboard_api.server import create_app
    ctx = DashboardContext.__new__(DashboardContext)
    with patch.object(
            DashboardContext, '__init__', return_value=None):
        app = create_app(ctx=ctx)
    # raise_server_exceptions=False lets the stub ctx return 500
    # instead of propagating the AttributeError to the test.
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get('/api/stops')
        # We only assert the route is registered (i.e. no 404).
        # 500 is fine — the stub ctx lacks a real analyzer.
        assert response.status_code != 404
