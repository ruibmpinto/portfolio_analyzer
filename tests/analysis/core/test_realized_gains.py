"""Tests for RealizedGains FIFO lot matcher."""

from datetime import datetime

import pandas as pd
import pytest

from src.analysis.core.portfolio import Portfolio
from src.analysis.core.realized_gains import RealizedGains
from src.analysis.core.transaction import Transaction


def _identity_fx(amount, currency, date):
    """Test FX converter: leaves the amount unchanged."""
    return amount


def test_empty_portfolio_returns_empty_frame():
    """A portfolio with no transactions yields no closed lots."""
    rg = RealizedGains(Portfolio(transactions=[]), _identity_fx)
    df = rg.closed_lots()
    assert df.empty
    expected_cols = {
        'ticker', 'buy_date', 'sell_date', 'shares',
        'cost_basis_native', 'proceeds_native', 'realized_native',
        'realized_chf', 'days_held', 'currency'}
    assert set(df.columns) == expected_cols


def test_full_lot_match_with_fees():
    """Buy 10 @ 100 (fees 1), sell 10 @ 120 (fees 2) -> +197 realized."""
    p = Portfolio(transactions=[
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=10, fee=1.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='sell',
            date=datetime(2024, 6, 5), price=120.0,
            amount=10, fee=2.0, auto_fx_fee=0.0, currency='USD'),
    ])
    df = RealizedGains(p, _identity_fx).closed_lots()
    assert len(df) == 1
    row = df.iloc[0]
    # cost_basis = 10*100 + 1 = 1001; proceeds = 10*120 - 2 = 1198
    assert row['cost_basis_native'] == pytest.approx(1001.0)
    assert row['proceeds_native'] == pytest.approx(1198.0)
    assert row['realized_native'] == pytest.approx(197.0)
    assert row['days_held'] == 152


def test_partial_sell_leaves_lot_open_with_reduced_basis():
    """Sell 5 of a 10-share lot: cost basis of remaining 5 stays intact."""
    p = Portfolio(transactions=[
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='sell',
            date=datetime(2024, 6, 5), price=120.0,
            amount=5, fee=0.0, auto_fx_fee=0.0, currency='USD'),
    ])
    df = RealizedGains(p, _identity_fx).closed_lots()
    assert len(df) == 1
    row = df.iloc[0]
    # 50% of lot: cost=500, proceeds=600, realized=100
    assert row['shares'] == pytest.approx(5.0)
    assert row['cost_basis_native'] == pytest.approx(500.0)
    assert row['proceeds_native'] == pytest.approx(600.0)
    assert row['realized_native'] == pytest.approx(100.0)


def test_fifo_across_multiple_lots():
    """Buys 10@100 then 10@110; sell 15@130 hits lot1 fully + lot2 partial."""
    p = Portfolio(transactions=[
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 2, 5), price=110.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='sell',
            date=datetime(2024, 6, 5), price=130.0,
            amount=15, fee=0.0, auto_fx_fee=0.0, currency='USD'),
    ])
    df = RealizedGains(p, _identity_fx).closed_lots()
    assert len(df) == 2
    row1, row2 = df.iloc[0], df.iloc[1]
    # Row 1: lot @ 100, 10 shares consumed
    assert row1['buy_date'] == datetime(2024, 1, 5)
    assert row1['shares'] == pytest.approx(10.0)
    assert row1['cost_basis_native'] == pytest.approx(1000.0)
    assert row1['proceeds_native'] == pytest.approx(
        1950.0 * 10.0 / 15.0)
    # Row 2: lot @ 110, 5 shares consumed
    assert row2['buy_date'] == datetime(2024, 2, 5)
    assert row2['shares'] == pytest.approx(5.0)
    assert row2['cost_basis_native'] == pytest.approx(550.0)
    assert row2['proceeds_native'] == pytest.approx(
        1950.0 * 5.0 / 15.0)
    # Total realized: 15 * 130 - (1000 + 550) = 1950 - 1550 = 400
    total_realized = df['realized_native'].sum()
    assert total_realized == pytest.approx(400.0)


def test_split_adjusted_cost_basis():
    """Buy 10 @ 100, 2-for-1 split, sell 5 @ 60 -> +50 realized."""
    p = Portfolio(transactions=[
        Transaction(
            ticker='TSLA', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='TSLA', operation='sell',
            date=datetime(2024, 12, 5), price=60.0,
            amount=5, fee=0.0, auto_fx_fee=0.0, currency='USD'),
    ])
    # 2-for-1 split between buy and sell
    splits = {'TSLA': pd.Series(
        [2.0], index=[pd.Timestamp('2024-06-01')])}
    p.set_splits(splits)
    df = RealizedGains(p, _identity_fx).closed_lots()
    assert len(df) == 1
    row = df.iloc[0]
    # Buy: 10 pre-split -> 20 post-today. Sell: 5 post-today.
    # Sold 5/20 = 25% of the lot. Cost slice = 0.25 * 1000 = 250.
    # Proceeds = 5*60 = 300. Realized = 50.
    assert row['shares'] == pytest.approx(5.0)
    assert row['cost_basis_native'] == pytest.approx(250.0)
    assert row['proceeds_native'] == pytest.approx(300.0)
    assert row['realized_native'] == pytest.approx(50.0)


def test_sell_exceeds_open_lots_raises():
    """A sell without enough matching buys must fail loud (no silent zero)."""
    p = Portfolio(transactions=[
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=5, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='sell',
            date=datetime(2024, 6, 5), price=120.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
    ])
    with pytest.raises(RuntimeError, match='exceeds open lots'):
        RealizedGains(p, _identity_fx).closed_lots()


def test_ytd_realized_filters_by_calendar_year():
    """Two sells, one in 2024, one in 2025: YTD includes only current year."""
    p = Portfolio(transactions=[
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2023, 1, 5), price=100.0,
            amount=20, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='sell',
            date=datetime(2024, 6, 5), price=110.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='sell',
            date=datetime(2025, 6, 5), price=130.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
    ])
    rg = RealizedGains(p, _identity_fx)
    # 2025 realized (10 * 130 - 10 * 100 = 300)
    ytd_2025 = rg.ytd_realized_chf(datetime(2025, 12, 31))
    assert ytd_2025 == pytest.approx(300.0)
    # 2024 realized (10 * 110 - 10 * 100 = 100)
    ytd_2024 = rg.ytd_realized_chf(datetime(2024, 12, 31))
    assert ytd_2024 == pytest.approx(100.0)
    # Pre-history year with no sells
    assert rg.ytd_realized_chf(datetime(2020, 12, 31)) == 0.0


def test_holding_period_violations_flags_short_holds():
    """A 90-day hold is flagged; a 200-day hold is not."""
    p = Portfolio(transactions=[
        Transaction(
            ticker='SHORT', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=1, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='SHORT', operation='sell',
            date=datetime(2024, 4, 5), price=110.0,
            amount=1, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='LONG', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=1, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='LONG', operation='sell',
            date=datetime(2024, 9, 5), price=110.0,
            amount=1, fee=0.0, auto_fx_fee=0.0, currency='USD'),
    ])
    rg = RealizedGains(p, _identity_fx)
    violations = rg.holding_period_violations(min_days=180)
    assert set(violations['ticker']) == {'SHORT'}


def test_fx_converter_applied_at_sell_date():
    """realized_chf equals realized_native * fx_rate the callable returns."""
    calls = []

    def fx(amount, currency, date):
        calls.append((amount, currency, date))
        return amount * 0.9  # e.g. USD -> CHF at 0.9

    p = Portfolio(transactions=[
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='sell',
            date=datetime(2024, 6, 5), price=120.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
    ])
    rg = RealizedGains(p, fx)
    row = rg.closed_lots().iloc[0]
    # realized_native = 200; * 0.9 = 180
    assert row['realized_chf'] == pytest.approx(180.0)
    # FX call passes the sell date, not the buy date
    assert calls[0][2] == datetime(2024, 6, 5)


def test_dividend_and_tax_rows_do_not_close_lots():
    """Dividend/tax rows are ignored by the FIFO walk."""
    p = Portfolio(transactions=[
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='dividend',
            date=datetime(2024, 3, 5), price=0.3,
            amount=1, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='tax',
            date=datetime(2024, 3, 5), price=-0.09,
            amount=1, fee=0.0, auto_fx_fee=0.0, currency='USD'),
    ])
    df = RealizedGains(p, _identity_fx).closed_lots()
    assert df.empty


def test_closed_lots_cached_across_calls():
    """Second call returns the same object (not a fresh compute)."""
    p = Portfolio(transactions=[])
    rg = RealizedGains(p, _identity_fx)
    first = rg.closed_lots()
    second = rg.closed_lots()
    assert first is second


def test_zero_share_buy_row_raises():
    """A buy resolving to zero post-split shares is a malformed row."""
    p = Portfolio(transactions=[
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=0, fee=0.0, auto_fx_fee=0.0, currency='USD'),
    ])
    with pytest.raises(
            RuntimeError, match='resolves to zero post-split shares'):
        RealizedGains(p, _identity_fx).closed_lots()


def test_zero_share_sell_row_raises():
    """A sell resolving to zero post-split shares is a malformed row."""
    p = Portfolio(transactions=[
        Transaction(
            ticker='AAPL', operation='buy',
            date=datetime(2024, 1, 5), price=100.0,
            amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AAPL', operation='sell',
            date=datetime(2024, 6, 5), price=120.0,
            amount=0, fee=0.0, auto_fx_fee=0.0, currency='USD'),
    ])
    with pytest.raises(
            RuntimeError, match='resolves to zero post-split shares'):
        RealizedGains(p, _identity_fx).closed_lots()


def test_unknown_operation_raises():
    """An operation outside buy/sell/dividend/tax must not be dropped."""
    # Bypass Transaction validation by editing operation after
    # construction (the class doesn't runtime-enforce the Literal).
    t = Transaction(
        ticker='AAPL', operation='buy',
        date=datetime(2024, 1, 5), price=100.0,
        amount=10, fee=0.0, auto_fx_fee=0.0, currency='USD')
    t.operation = 'transfer'  # unknown op
    p = Portfolio(transactions=[t])
    with pytest.raises(
            RuntimeError, match='Unknown transaction operation'):
        RealizedGains(p, _identity_fx).closed_lots()
