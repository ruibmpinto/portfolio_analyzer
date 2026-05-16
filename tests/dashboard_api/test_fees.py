"""Tests for cumulative_fees aggregator."""

from datetime import datetime

import pandas as pd
import pytest

from src.analysis.core.portfolio import Portfolio
from src.analysis.core.transaction import Transaction
from src.dashboard_api.services.fees import cumulative_fees


class _StubProvider:
    """Returns a constant FX close for every (from, to) pair."""

    def __init__(self, rates):
        # rates[(from_ccy, to_ccy)] -> float
        self.rates = rates

    def get_price_history(
            self, ticker, start_date=None, end_date=None):
        # ticker is e.g. 'USDCHF=X'
        from_ccy = ticker[:3]
        to_ccy = ticker[3:6]
        rate = self.rates.get((from_ccy, to_ccy))
        if rate is None:
            return pd.Series(dtype=float)
        return pd.Series([rate])


def _portfolio():
    txs = [
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 1, 5), price=180.0,
            amount=10, fee=1.0, auto_fx_fee=0.5,
            currency='USD'),
        Transaction(
            ticker='AAPL', operation='sell',
            date=datetime(2024, 8, 15), price=200.0,
            amount=5, fee=2.0, auto_fx_fee=0.0,
            currency='USD'),
        Transaction(
            ticker='NESN.SW', operation='buy',
            date=datetime(2024, 3, 1), price=100.0,
            amount=20, fee=3.0, auto_fx_fee=0.0,
            currency='CHF'),
    ]
    return Portfolio(transactions=txs)


def test_cumulative_fees_aggregates_per_currency():
    pf = _portfolio()
    provider = _StubProvider({('USD', 'CHF'): 0.9})
    result = cumulative_fees(
        pf, data_provider=provider, base_currency='CHF')
    # USD: 1.0 + 0.5 + 2.0 = 3.5; CHF: 3.0
    assert result['by_currency'] == {'USD': 3.5, 'CHF': 3.0}
    # Total CHF: 3.0 + 3.5 * 0.9 = 6.15
    assert result['total_chf'] == pytest.approx(6.15, rel=1e-9)
    assert result['fx_rates_used'] == {'USD': 0.9, 'CHF': 1.0}


def test_cumulative_fees_skips_zero_fee_transactions():
    """Zero-fee rows must not pollute the by_currency dict."""
    txs = [
        Transaction(
            ticker='X', operation='buy',
            date=datetime(2024, 1, 1), price=10.0,
            amount=1, fee=0.0, auto_fx_fee=0.0,
            currency='EUR'),
    ]
    pf = Portfolio(transactions=txs)
    provider = _StubProvider({('EUR', 'CHF'): 0.95})
    result = cumulative_fees(pf, provider)
    assert result['by_currency'] == {}
    assert result['total_chf'] == 0.0


def test_cumulative_fees_raises_when_fx_missing():
    """Missing FX history must raise rather than silently zero."""
    txs = [
        Transaction(
            ticker='X', operation='buy',
            date=datetime(2024, 1, 1), price=10.0,
            amount=1, fee=1.0, auto_fx_fee=0.0,
            currency='JPY'),
    ]
    pf = Portfolio(transactions=txs)
    provider = _StubProvider({})  # no JPYCHF=X
    with pytest.raises(RuntimeError, match='JPYCHF'):
        cumulative_fees(pf, provider)
