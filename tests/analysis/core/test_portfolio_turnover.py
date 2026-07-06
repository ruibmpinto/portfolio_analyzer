"""Tests for Portfolio.get_turnover and the refactored _sum_by_currency."""

from datetime import datetime

import pytest

from src.analysis.core.portfolio import Portfolio
from src.analysis.core.transaction import Transaction


def _portfolio():
    """A portfolio spanning two years, three currencies, four ops."""
    return Portfolio(transactions=[
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 3, 1), price=100.0,
            amount=10, fee=1.0, auto_fx_fee=0.0,
            currency='USD'),
        Transaction(
            ticker='AAPL', operation='sell',
            date=datetime(2024, 8, 1), price=120.0,
            amount=5, fee=1.0, auto_fx_fee=0.0,
            currency='USD'),
        Transaction(
            ticker='NESN.SW', operation='buy',
            date=datetime(2025, 2, 15), price=90.0,
            amount=20, fee=2.0, auto_fx_fee=0.0,
            currency='CHF'),
        Transaction(
            ticker='RR.L', operation='buy',
            date=datetime(2025, 6, 1), price=5.0,
            amount=100, fee=1.5, auto_fx_fee=0.5,
            currency='GBP'),
        Transaction(
            ticker='AAPL', operation='dividend',
            date=datetime(2025, 7, 1), price=0.30,
            amount=1, fee=0.0, auto_fx_fee=0.0,
            currency='USD'),
    ])


def test_turnover_defaults_to_buys_and_sells_all_history():
    """No date bounds and default ops: all buy/sell notional."""
    p = _portfolio()
    t = p.get_turnover()
    # USD: buy 10*100+1 = 1001, sell 5*120-1 = 599 -> total 1600
    assert t['USD'] == pytest.approx(1001.0 + 599.0)
    # CHF: buy 20*90+2 = 1802
    assert t['CHF'] == pytest.approx(1802.0)
    # GBP: buy 100*5+1.5+0.5 = 502
    assert t['GBP'] == pytest.approx(502.0)


def test_turnover_ignores_dividends_by_default():
    """Dividend rows do not contribute to buy/sell turnover."""
    p = _portfolio()
    t = p.get_turnover()
    # USD total excludes the dividend row (which is 0.30 * 1 = 0.30)
    # If it were included, USD would be 1600.3, not 1600.
    assert t['USD'] == pytest.approx(1600.0)


def test_turnover_since_filter_excludes_earlier_transactions():
    """`since` cuts off the 2024 AAPL activity."""
    p = _portfolio()
    t = p.get_turnover(since=datetime(2025, 1, 1))
    # 2024 AAPL buy/sell dropped; only 2025 buys remain
    assert 'USD' not in t     # no USD trades in 2025
    assert t['CHF'] == pytest.approx(1802.0)
    assert t['GBP'] == pytest.approx(502.0)


def test_turnover_until_filter_excludes_later_transactions():
    """`until` keeps only the 2024 rows."""
    p = _portfolio()
    t = p.get_turnover(until=datetime(2024, 12, 31))
    assert t == pytest.approx({'USD': 1600.0})


def test_turnover_window_is_inclusive_on_both_ends():
    """Trades exactly on `since` or `until` are counted."""
    p = _portfolio()
    t = p.get_turnover(
        since=datetime(2024, 3, 1),
        until=datetime(2024, 8, 1))
    # Both AAPL trades sit on the boundary dates -> both counted
    assert t['USD'] == pytest.approx(1600.0)


def test_turnover_custom_ops_restricts_the_operation_set():
    """Passing ops=('buy',) mirrors the hold-strategy turnover."""
    p = _portfolio()
    t = p.get_turnover(ops=('buy',))
    # Only buys: USD 1001, CHF 1802, GBP 502
    assert t == pytest.approx(
        {'USD': 1001.0, 'CHF': 1802.0, 'GBP': 502.0})


def test_turnover_empty_window_returns_empty_dict():
    """A window that excludes everything returns {}."""
    p = _portfolio()
    t = p.get_turnover(
        since=datetime(2030, 1, 1),
        until=datetime(2030, 12, 31))
    assert t == {}


def test_sum_by_currency_unsigned_matches_turnover_for_one_op():
    """Refactor invariant: _sum_by_currency(op, signed=False) == turnover."""
    p = _portfolio()
    assert (p._sum_by_currency('buy')
            == p.get_turnover(ops=('buy',)))


def test_get_total_invested_still_works_after_refactor():
    """Public API unchanged: sums abs(total_cost) of every buy row."""
    p = _portfolio()
    invested = p.get_total_invested()
    assert invested == pytest.approx(
        {'USD': 1001.0, 'CHF': 1802.0, 'GBP': 502.0})


def test_get_total_proceeds_signed_path_still_works():
    """Signed path unaffected: proceeds keep positive sign."""
    p = _portfolio()
    # sell total_cost = 5*120 - 1 = 599 (positive per Transaction)
    assert p.get_total_proceeds() == pytest.approx({'USD': 599.0})
