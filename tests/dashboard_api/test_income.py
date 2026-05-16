"""Tests for the dividend + tax aggregator."""

from datetime import datetime

import pandas as pd
import pytest

from src.analysis.core.portfolio import Portfolio
from src.analysis.core.transaction import Transaction
from src.dashboard_api.services.income import cumulative_income


class _StubProvider:
    def __init__(self, rates):
        self.rates = rates

    def get_price_history(
            self, ticker, start_date=None, end_date=None):
        from_ccy = ticker[:3]
        to_ccy = ticker[3:6]
        rate = self.rates.get((from_ccy, to_ccy))
        if rate is None:
            return pd.Series(dtype=float)
        return pd.Series([rate])


def _dividend(ticker, date, price, currency='USD'):
    return Transaction(
        ticker=ticker, operation='dividend', date=date,
        price=price, amount=1, fee=0.0, auto_fx_fee=0.0,
        currency=currency)


def _tax(ticker, date, price, currency='USD'):
    """Tax transaction; convention: price is negative magnitude."""
    return Transaction(
        ticker=ticker, operation='tax', date=date,
        price=price, amount=1, fee=0.0, auto_fx_fee=0.0,
        currency=currency)


def test_dividends_only_per_currency_with_chf_conversion():
    txs = [
        _dividend('NVDA', datetime(2024, 3, 28), 0.08),
        _dividend('NVDA', datetime(2024, 7, 1), 0.20),
        _dividend('UBSG.SW', datetime(2025, 4, 1), 0.43, 'CHF'),
    ]
    pf = Portfolio(transactions=txs)
    provider = _StubProvider({('USD', 'CHF'): 0.9})
    result = cumulative_income(pf, 'dividend', provider)
    assert result['by_currency'] == {
        'USD': pytest.approx(0.28), 'CHF': pytest.approx(0.43)}
    # 0.28 * 0.9 + 0.43 = 0.682
    assert result['total_chf'] == pytest.approx(0.682, rel=1e-9)
    assert result['fx_rates_used'] == {'USD': 0.9, 'CHF': 1.0}


def test_taxes_preserve_negative_sign():
    """Withholding tax rows ship with negative `price`."""
    txs = [
        _tax('NVDA', datetime(2024, 3, 28), -0.05),
        _tax('NVDA', datetime(2024, 7, 1), -0.10),
    ]
    pf = Portfolio(transactions=txs)
    provider = _StubProvider({('USD', 'CHF'): 0.9})
    result = cumulative_income(pf, 'tax', provider)
    assert result['by_currency'] == {'USD': pytest.approx(-0.15)}
    assert result['total_chf'] == pytest.approx(-0.135, rel=1e-9)


def test_dividends_and_taxes_isolated_per_call():
    """Filtering by `operation` keeps the two buckets distinct."""
    txs = [
        _dividend('NVDA', datetime(2024, 3, 28), 0.20),
        _tax('NVDA', datetime(2024, 3, 28), -0.05),
    ]
    pf = Portfolio(transactions=txs)
    provider = _StubProvider({('USD', 'CHF'): 0.9})
    divs = cumulative_income(pf, 'dividend', provider)
    taxes = cumulative_income(pf, 'tax', provider)
    assert divs['by_currency'] == {'USD': pytest.approx(0.20)}
    assert taxes['by_currency'] == {'USD': pytest.approx(-0.05)}


def test_unknown_operation_raises():
    pf = Portfolio(transactions=[])
    with pytest.raises(ValueError, match='must be'):
        cumulative_income(pf, 'buy', _StubProvider({}))


def test_missing_fx_raises_runtime_error():
    txs = [_dividend('X', datetime(2024, 1, 1), 1.0, 'JPY')]
    pf = Portfolio(transactions=txs)
    with pytest.raises(RuntimeError, match='JPYCHF'):
        cumulative_income(pf, 'dividend', _StubProvider({}))


def test_zero_price_rows_skipped():
    """Zero-cash dividend / tax rows must not pollute the buckets."""
    txs = [_dividend('X', datetime(2024, 1, 1), 0.0)]
    pf = Portfolio(transactions=txs)
    result = cumulative_income(pf, 'dividend', _StubProvider({}))
    assert result['by_currency'] == {}
    assert result['total_chf'] == 0.0
