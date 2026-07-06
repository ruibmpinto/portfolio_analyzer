"""Tests for the Swiss safe-harbor tracker service + route.

The service composes four primitives (Jan-1 NAV, YTD turnover,
realized gains, holding-period violations). Tests hit each helper
directly for math correctness, plus a happy-path end-to-end run
against a hand-built context.
"""

from datetime import datetime

import pandas as pd
import pytest

from src.dashboard_api.services.safe_harbor import (
    _status, _turnover_ratio, _violations_records,
    build_safe_harbor_payload)


# --------------------------------------------------------------------
# Pure-function helpers
# --------------------------------------------------------------------


def test_turnover_ratio_positive_baseline():
    assert _turnover_ratio(5000.0, 1000.0) == pytest.approx(5.0)


def test_turnover_ratio_zero_baseline_returns_none():
    """No Jan-1 baseline -> ratio is undefined, not infinity."""
    assert _turnover_ratio(5000.0, 0.0) is None
    assert _turnover_ratio(5000.0, -1.0) is None


def test_status_ok_below_warning():
    assert _status(ratio=2.0, violation_count=0) == 'ok'


def test_status_warning_between_thresholds():
    """4.0 <= ratio < 5.0 warns without breaching."""
    assert _status(ratio=4.5, violation_count=0) == 'warning'


def test_status_breach_on_ratio_at_or_above_threshold():
    assert _status(ratio=5.0, violation_count=0) == 'breach'
    assert _status(ratio=6.5, violation_count=0) == 'breach'


def test_status_breach_on_any_violation_even_when_ratio_ok():
    """One holding-period violation is a breach even at low turnover."""
    assert _status(ratio=1.0, violation_count=1) == 'breach'


def test_status_no_baseline_when_ratio_is_none():
    assert _status(ratio=None, violation_count=0) == 'no_baseline'
    # Even with violations the missing baseline dominates
    assert _status(ratio=None, violation_count=3) == 'no_baseline'


def test_violations_records_empty_frame_returns_empty_list():
    df = pd.DataFrame(columns=[
        'ticker', 'buy_date', 'sell_date', 'days_held', 'shares',
        'realized_chf'])
    assert _violations_records(df) == []


def test_violations_records_serialises_dates_to_iso():
    """Datetimes emerge as ISO-8601 strings for JSON friendliness."""
    df = pd.DataFrame([{
        'ticker': 'AAPL',
        'buy_date': datetime(2024, 1, 5),
        'sell_date': datetime(2024, 4, 5),
        'days_held': 91,
        'shares': 10.0,
        'realized_chf': 123.4,
    }])
    records = _violations_records(df)
    assert len(records) == 1
    r = records[0]
    assert r['ticker'] == 'AAPL'
    assert r['buy_date'] == '2024-01-05T00:00:00'
    assert r['sell_date'] == '2024-04-05T00:00:00'
    assert r['days_held'] == 91
    assert r['shares'] == pytest.approx(10.0)
    assert r['realized_chf'] == pytest.approx(123.4)


# --------------------------------------------------------------------
# End-to-end with a stub context
# --------------------------------------------------------------------


class _StubPortfolio:
    """Minimum Portfolio-shaped stub for the service."""

    def __init__(self, holdings_by_date, turnover_by_currency):
        self._holdings_by_date = holdings_by_date
        self._turnover_by_currency = turnover_by_currency
        self.transactions = []  # RealizedGains iterates this list

    def get_holdings_at_date(self, date, operations=None):
        return self._holdings_by_date.get(date, {})

    def get_turnover(self, since, until, ops=None):
        return dict(self._turnover_by_currency)


class _StubAnalyzer:
    """Minimum PortfolioAnalyzer-shaped stub for the service."""

    def __init__(self, portfolio, jan1_nav_chf, fx_by_currency):
        self.portfolio = portfolio
        self._jan1_nav_chf = jan1_nav_chf
        self._fx = fx_by_currency
        # Empty by default; _first_trading_day_of_year falls back
        # to pd.bdate_range when this dict has no non-empty series.
        self._market_data = {}

    def _fetch_market_data(self):
        pass

    def _calculate_value(
            self, holdings, date, apply_dividend_correction=False):
        return self._jan1_nav_chf

    def _native_to_base(self, amount, currency, date):
        return amount * self._fx.get(currency, 1.0)


class _StubContext:
    def __init__(self, analyzer):
        self.analyzer = analyzer


def test_build_payload_populated_portfolio_shape_and_math():
    """Payload carries every documented field with correct arithmetic."""
    year_start = datetime(2026, 1, 1)
    today = datetime(2026, 7, 6)
    portfolio = _StubPortfolio(
        holdings_by_date={year_start: {'AAPL': 10.0}},
        turnover_by_currency={'USD': 20000.0, 'CHF': 5000.0})
    analyzer = _StubAnalyzer(
        portfolio=portfolio,
        jan1_nav_chf=10000.0,
        fx_by_currency={'USD': 0.9, 'CHF': 1.0})
    ctx = _StubContext(analyzer)

    payload = build_safe_harbor_payload(ctx, today=today)
    # Jan-1 NAV = 10000 (from stub _calculate_value)
    assert payload['jan1_nav_chf'] == pytest.approx(10000.0)
    # Turnover = 20000 * 0.9 (USD->CHF) + 5000 * 1.0 (CHF) = 23000
    assert payload['ytd_turnover_chf'] == pytest.approx(23000.0)
    # Ratio = 23000 / 10000 = 2.3
    assert payload['ytd_turnover_ratio'] == pytest.approx(2.3)
    assert payload['threshold_ratio'] == 5.0
    # No transactions in the stub -> zero realized, no violations
    assert payload['ytd_realized_chf'] == 0.0
    assert payload['holding_period_violations'] == []
    # Ratio 2.3 < 4.0 warning threshold -> 'ok'
    assert payload['status'] == 'ok'


def test_build_payload_new_portfolio_no_baseline():
    """No holdings on Jan 1 -> jan1_nav 0, ratio None, status no_baseline."""
    year_start = datetime(2026, 1, 1)
    today = datetime(2026, 7, 6)
    portfolio = _StubPortfolio(
        holdings_by_date={},  # empty for every date
        turnover_by_currency={'USD': 1000.0})
    analyzer = _StubAnalyzer(
        portfolio=portfolio,
        jan1_nav_chf=0.0,
        fx_by_currency={'USD': 0.9})
    ctx = _StubContext(analyzer)

    payload = build_safe_harbor_payload(ctx, today=today)
    assert payload['jan1_nav_chf'] == 0.0
    assert payload['ytd_turnover_chf'] == pytest.approx(900.0)
    assert payload['ytd_turnover_ratio'] is None
    assert payload['status'] == 'no_baseline'


def test_build_payload_breach_when_turnover_exceeds_5x_nav():
    year_start = datetime(2026, 1, 1)
    today = datetime(2026, 7, 6)
    portfolio = _StubPortfolio(
        holdings_by_date={year_start: {'AAPL': 10.0}},
        turnover_by_currency={'CHF': 60000.0})
    analyzer = _StubAnalyzer(
        portfolio=portfolio,
        jan1_nav_chf=10000.0,
        fx_by_currency={'CHF': 1.0})
    ctx = _StubContext(analyzer)

    payload = build_safe_harbor_payload(ctx, today=today)
    assert payload['ytd_turnover_ratio'] == pytest.approx(6.0)
    assert payload['status'] == 'breach'


# --------------------------------------------------------------------
# Route registration
# --------------------------------------------------------------------


def test_route_registered_on_app():
    """/api/safe_harbor is reachable through the FastAPI router."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from src.dashboard_api.context import DashboardContext
    from src.dashboard_api.server import create_app
    ctx = DashboardContext.__new__(DashboardContext)
    with patch.object(
            DashboardContext, '__init__', return_value=None):
        app = create_app(ctx=ctx)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get('/api/safe_harbor')
        assert response.status_code != 404
