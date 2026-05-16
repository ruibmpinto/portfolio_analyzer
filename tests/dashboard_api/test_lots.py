"""Tests for FIFO lot tracking + 6-month-hold eligibility."""

from datetime import datetime, timedelta

from src.analysis.core.portfolio import Portfolio
from src.analysis.core.transaction import Transaction
from src.dashboard_api.services.lots import (
    held_lots, sellable_today, min_hold_days_default)


def _buy(ticker, date, amount, fee=0.0):
    return Transaction(
        ticker=ticker, operation='buy', date=date,
        price=10.0, amount=amount, fee=fee, auto_fx_fee=0.0,
        currency='USD')


def _sell(ticker, date, amount, fee=0.0):
    return Transaction(
        ticker=ticker, operation='sell', date=date,
        price=12.0, amount=amount, fee=fee, auto_fx_fee=0.0,
        currency='USD')


def test_held_lots_single_buy_unsold():
    pf = Portfolio([_buy('AAPL', datetime(2025, 1, 1), 10)])
    lots = held_lots(pf)
    assert list(lots.keys()) == ['AAPL']
    assert lots['AAPL'] == [(datetime(2025, 1, 1), 10.0)]


def test_held_lots_fifo_partial_sell_consumes_oldest():
    """Selling 7 of 15 must leave 3 in lot1 and 5 in lot2."""
    pf = Portfolio([
        _buy('AAPL', datetime(2025, 1, 1), 10),
        _buy('AAPL', datetime(2025, 6, 1), 5),
        _sell('AAPL', datetime(2025, 8, 1), 7)])
    lots = held_lots(pf)
    assert lots['AAPL'] == [
        (datetime(2025, 1, 1), 3.0),
        (datetime(2025, 6, 1), 5.0)]


def test_held_lots_drops_fully_closed_position():
    """Selling everything must remove the ticker entirely."""
    pf = Portfolio([
        _buy('AAPL', datetime(2025, 1, 1), 10),
        _sell('AAPL', datetime(2025, 8, 1), 10)])
    lots = held_lots(pf)
    assert 'AAPL' not in lots


def test_sellable_today_boundary_at_exactly_183_days():
    """Boundary check: oldest lot aged exactly min_hold_days is eligible."""
    as_of = datetime(2025, 12, 31)
    earliest = as_of - timedelta(days=min_hold_days_default)
    pf = Portfolio([_buy('AAPL', earliest, 10)])
    info = sellable_today(pf, as_of=as_of)
    assert info['AAPL']['eligible'] is True
    assert info['AAPL']['days_held'] == min_hold_days_default


def test_sellable_today_one_day_short_is_locked():
    as_of = datetime(2025, 12, 31)
    earliest = as_of - timedelta(
        days=min_hold_days_default - 1)
    pf = Portfolio([_buy('AAPL', earliest, 10)])
    info = sellable_today(pf, as_of=as_of)
    rec = info['AAPL']
    assert rec['eligible'] is False
    assert rec['days_held'] == min_hold_days_default - 1
    assert rec['unlock_date'] == (
        earliest + timedelta(days=min_hold_days_default))


def test_sellable_today_uses_oldest_open_lot():
    """A recent top-up must not reset the 6-month clock."""
    as_of = datetime(2025, 12, 31)
    old = as_of - timedelta(days=200)
    recent = as_of - timedelta(days=30)
    pf = Portfolio([
        _buy('AAPL', old, 10),
        _buy('AAPL', recent, 5)])
    info = sellable_today(pf, as_of=as_of)
    assert info['AAPL']['eligible'] is True
    assert info['AAPL']['days_held'] == 200
    assert info['AAPL']['shares'] == 15.0


def test_sellable_today_after_partial_sell_uses_new_oldest_lot():
    """If FIFO consumes the original lot, the next-oldest takes over."""
    as_of = datetime(2025, 12, 31)
    very_old = as_of - timedelta(days=400)
    medium = as_of - timedelta(days=100)
    pf = Portfolio([
        _buy('AAPL', very_old, 10),
        _buy('AAPL', medium, 8),
        _sell('AAPL', as_of - timedelta(days=10), 10)])
    info = sellable_today(pf, as_of=as_of)
    # The 400-day lot is fully consumed; 100-day lot becomes oldest
    assert info['AAPL']['shares'] == 8.0
    assert info['AAPL']['days_held'] == 100
    assert info['AAPL']['eligible'] is False
