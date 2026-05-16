"""Tests for Portfolio's synthetic-split-buy filter.

Some brokers (Degiro) record a stock split as a zero-fee buy of
the post-split additional shares. When this is combined with our
own split-factor mechanism the corporate action would be double-
counted. Portfolio drops those synthetic rows as soon as it
learns about the split.
"""

from datetime import datetime

import pandas as pd

from src.analysis.core.portfolio import Portfolio
from src.analysis.core.transaction import Transaction


def _buy(ticker, date, amount, fee=1.0, price=100.0):
    """Real buy with a non-zero fee (default 1.0 USD)."""
    return Transaction(
        ticker=ticker, operation='buy', date=date,
        price=price, amount=amount, fee=fee,
        auto_fx_fee=0.0, currency='USD')


def _synthetic_split_buy(ticker, date, amount, price=100.0):
    """Synthetic split-adjustment row: zero fee on both legs."""
    return Transaction(
        ticker=ticker, operation='buy', date=date,
        price=price, amount=amount, fee=0.0,
        auto_fx_fee=0.0, currency='USD')


def _sell(ticker, date, amount, fee=1.0, price=130.0):
    return Transaction(
        ticker=ticker, operation='sell', date=date,
        price=price, amount=amount, fee=fee,
        auto_fx_fee=0.0, currency='USD')


def _split_series(dates_and_ratios):
    """Build a pd.Series ex-date -> ratio."""
    idx = pd.to_datetime([d for d, _ in dates_and_ratios])
    vals = [r for _, r in dates_and_ratios]
    return pd.Series(vals, index=idx)


def test_synthetic_split_row_dropped_when_splits_attached():
    """NVDA-shaped scenario: net should be 0 after splits attach."""
    # Two pre-split buys, the synthetic adjustment row, two real
    # post-split buys, and one sell that closes everything out.
    txs = [
        _buy('NVDA', datetime(2024, 1, 30), 1, price=628.49),
        _buy('NVDA', datetime(2024, 2, 6), 1, price=694.00),
        _synthetic_split_buy(
            'NVDA', datetime(2024, 6, 10), 20, price=120.89),
        _buy('NVDA', datetime(2024, 6, 12), 1, price=124.76),
        _buy('NVDA', datetime(2025, 1, 28), 2, price=124.43),
        _sell('NVDA', datetime(2025, 5, 19), 23, price=132.75),
    ]
    splits = {
        'NVDA': _split_series([(datetime(2024, 6, 10), 10.0)])}
    pf = Portfolio(transactions=txs, splits=splits)
    # The synthetic row is gone
    buy_dates = sorted(
        t.date for t in pf.transactions if t.operation == 'buy')
    assert datetime(2024, 6, 10) not in buy_dates
    assert len(pf.transactions) == 5  # 4 buys + 1 sell
    # Holdings: 1*10 + 1*10 + 1 + 2 - 23 = 0
    holdings = pf.get_current_holdings()
    assert holdings.get('NVDA', 0.0) == 0.0


def test_no_filtering_when_splits_dict_empty():
    """Without split data we have no signal to filter on."""
    txs = [
        _buy('NVDA', datetime(2024, 1, 30), 1),
        _synthetic_split_buy(
            'NVDA', datetime(2024, 6, 10), 20),
        _sell('NVDA', datetime(2024, 7, 1), 21)]
    pf = Portfolio(transactions=txs)
    # All three rows survive — no filter ran
    assert len(pf.transactions) == 3


def test_zero_fee_buy_on_non_split_day_preserved():
    """Commission-free buys far from any split must not be dropped."""
    txs = [
        # Real zero-fee trade on a non-split day
        _synthetic_split_buy(
            'AAPL', datetime(2024, 3, 15), 5, price=170.0)]
    # Only a totally unrelated split is in the table
    splits = {
        'AAPL': _split_series([(datetime(2020, 8, 31), 4.0)])}
    pf = Portfolio(transactions=txs, splits=splits)
    # Row survives because trade date does not match any split
    assert len(pf.transactions) == 1
    assert pf.transactions[0].ticker == 'AAPL'


def test_filter_runs_on_set_splits_after_construction():
    """Splits attached post-construction must still trigger filter."""
    txs = [
        _buy('NVDA', datetime(2024, 1, 30), 1),
        _synthetic_split_buy(
            'NVDA', datetime(2024, 6, 10), 20),
        _buy('NVDA', datetime(2024, 6, 12), 1),
        _sell('NVDA', datetime(2024, 7, 1), 11)]
    pf = Portfolio(transactions=txs)
    # Before splits attach: all 4 rows present
    assert len(pf.transactions) == 4
    # Attach splits — synthetic row gets dropped
    pf.set_splits({
        'NVDA': _split_series([(datetime(2024, 6, 10), 10.0)])})
    assert len(pf.transactions) == 3
    assert all(
        not (t.fee == 0.0 and t.amount == 20)
        for t in pf.transactions)
    # 1*10 + 1 - 11 = 0
    assert pf.get_current_holdings().get('NVDA', 0.0) == 0.0


def test_paired_zero_fee_sell_and_buy_on_split_day_both_dropped():
    """PANW-shaped scenario: 2-for-1 split recorded as paired sell+buy."""
    # Real buy of 1 share pre-split (with fee)
    # Synthetic zero-fee SELL of 1 on the split day (removes pre-split)
    # Synthetic zero-fee BUY of 2 on the split day (adds post-split)
    # Real sell of 2 shares post-split (with fee)
    txs = [
        _buy('PANW', datetime(2024, 3, 13), 1, price=285.0),
        Transaction(
            ticker='PANW', operation='sell',
            date=datetime(2024, 12, 16), price=393.12,
            amount=1, fee=0.0, auto_fx_fee=0.0,
            currency='USD'),
        _synthetic_split_buy(
            'PANW', datetime(2024, 12, 16), 2, price=196.56),
        _sell('PANW', datetime(2025, 5, 14), 2, price=190.47),
    ]
    splits = {
        'PANW': _split_series([(datetime(2024, 12, 16), 2.0)])}
    pf = Portfolio(transactions=txs, splits=splits)
    # Both synthetic rows (sell 1 + buy 2) are dropped
    assert len(pf.transactions) == 2
    operations = [t.operation for t in pf.transactions]
    assert operations == ['buy', 'sell']
    # 1*2 - 2 = 0
    assert pf.get_current_holdings().get('PANW', 0.0) == 0.0


def test_filter_allows_split_at_plus_two_days_slack():
    """Trades within +/- 2 days of a split date count as synthetic.

    Brokers sometimes record the split on the next trading day or
    a previous one. The +/- 2 day window absorbs that drift.
    """
    txs = [
        _synthetic_split_buy(
            'NVDA', datetime(2024, 6, 11), 20)]  # +1 day
    splits = {
        'NVDA': _split_series([(datetime(2024, 6, 10), 10.0)])}
    pf = Portfolio(transactions=txs, splits=splits)
    assert pf.transactions == []
