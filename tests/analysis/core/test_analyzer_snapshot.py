"""Tests for PortfolioAnalyzer.get_holdings_snapshot and get_price_panel."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.analysis.core.transaction import Transaction
from src.analysis.core.analyzer import PortfolioAnalyzer
from src.analysis.loaders.csv_loader import CSVLoader


def test_snapshot_returns_dataframe(analyzer):
    """Returns a pandas DataFrame."""
    df = analyzer.get_holdings_snapshot()
    assert isinstance(df, pd.DataFrame)


def test_snapshot_columns_without_categories(analyzer):
    """When categories is None, no 'category' column."""
    df = analyzer.get_holdings_snapshot()
    expected = {
        'ticker', 'shares', 'value_chf', 'weight_pct',
        'currency', 'sector', 'price_chf'}
    assert set(df.columns) == expected


def test_snapshot_columns_with_categories(analyzer):
    """When categories is provided, 'category' column added."""
    cats = {
        'AMZN': 'core',
        'UBSG.SW': 'core',
        'HOLN.SW': 'core'}
    df = analyzer.get_holdings_snapshot(categories=cats)
    expected = {
        'ticker', 'shares', 'value_chf', 'weight_pct',
        'currency', 'sector', 'price_chf', 'category'}
    assert set(df.columns) == expected


def test_snapshot_includes_all_held_tickers(analyzer):
    """Every fixture-held ticker appears in the output."""
    df = analyzer.get_holdings_snapshot()
    assert set(df['ticker']) == {'AMZN', 'UBSG.SW', 'HOLN.SW'}


def test_snapshot_weights_sum_close_to_100(analyzer):
    """weight_pct values sum to ~100 (rounding tolerance)."""
    df = analyzer.get_holdings_snapshot()
    assert df['weight_pct'].sum() == pytest.approx(100.0, abs=1e-6)


def test_snapshot_sorted_by_value_descending(analyzer):
    """Rows are ordered largest-position-first."""
    df = analyzer.get_holdings_snapshot()
    values = df['value_chf'].tolist()
    assert values == sorted(values, reverse=True)


def test_snapshot_raises_on_unclassified_ticker(analyzer):
    """Per no-silent-defaults policy: missing category -> raise."""
    cats = {'AMZN': 'core'}    # UBSG.SW + HOLN.SW absent
    with pytest.raises(RuntimeError, match='missing category'):
        analyzer.get_holdings_snapshot(categories=cats)


def test_snapshot_currency_column_matches_transactions(analyzer):
    """Currency column reflects transaction currencies."""
    df = analyzer.get_holdings_snapshot()
    by_ticker = dict(zip(df['ticker'], df['currency']))
    assert by_ticker['AMZN'] == 'USD'
    assert by_ticker['UBSG.SW'] == 'CHF'
    assert by_ticker['HOLN.SW'] == 'CHF'


def test_snapshot_sector_populated_from_fundamentals(analyzer):
    """Sector column reflects data_provider.get_fundamental_data."""
    df = analyzer.get_holdings_snapshot()
    by_ticker = dict(zip(df['ticker'], df['sector']))
    assert by_ticker['AMZN'] == 'Consumer Cyclical'
    assert by_ticker['UBSG.SW'] == 'Financial Services'
    assert by_ticker['HOLN.SW'] == 'Basic Materials'


def test_snapshot_sector_falls_back_to_unknown(analyzer, monkeypatch):
    """Sector defaults to 'Unknown' when fundamentals empty."""
    monkeypatch.setattr(
        analyzer.data_provider,
        'get_fundamental_data',
        lambda ticker: {})
    df = analyzer.get_holdings_snapshot()
    assert (df['sector'] == 'Unknown').all()


def test_snapshot_value_chf_matches_position_values(analyzer):
    """value_chf column equals get_current_position_values output."""
    df = analyzer.get_holdings_snapshot()
    snapshot_values = dict(zip(df['ticker'], df['value_chf']))
    position_values = analyzer.get_current_position_values()
    held = set(snapshot_values)
    for ticker in held:
        assert snapshot_values[ticker] == pytest.approx(
            position_values[ticker])


def test_snapshot_price_chf_derives_from_value_and_shares(analyzer):
    """price_chf == value_chf / shares for every row."""
    df = analyzer.get_holdings_snapshot()
    for _, row in df.iterrows():
        assert row['price_chf'] == pytest.approx(
            row['value_chf'] / row['shares'])


def test_snapshot_drops_closed_positions(
    monkeypatch, fake_data_provider):
    """Buy-then-sell that nets to zero shares is omitted."""
    base = datetime(2024, 1, 15)
    closing_pair = [
        Transaction(
            ticker='META', operation='buy', date=base,
            price=300.0, amount=2, fee=0.0,
            auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='META', operation='sell',
            date=base + timedelta(days=5),
            price=310.0, amount=2, fee=0.0,
            auto_fx_fee=0.0, currency='USD'),
        Transaction(
            ticker='AMZN', operation='buy',
            date=base + timedelta(days=10),
            price=150.0, amount=3, fee=0.0,
            auto_fx_fee=0.0, currency='USD'),
    ]
    monkeypatch.setattr(
        CSVLoader, 'load_csv_degiro',
        lambda self, path: closing_pair)
    monkeypatch.setattr(
        CSVLoader, 'load_csv_ibkr',
        lambda self, path: [])
    a = PortfolioAnalyzer(
        degiro_csv_file_path='ignored.csv',
        ibkr_csv_file_path='ignored.csv',
        data_provider=fake_data_provider,
        base_currency='CHF')
    df = a.get_holdings_snapshot()
    assert 'META' not in set(df['ticker'])
    assert 'AMZN' in set(df['ticker'])


def test_panel_returns_dataframe(analyzer):
    """Returns a pandas DataFrame."""
    df = analyzer.get_price_panel()
    assert isinstance(df, pd.DataFrame)


def test_panel_has_one_column_per_held_ticker(analyzer):
    """Columns are exactly the currently-held tickers."""
    df = analyzer.get_price_panel()
    assert set(df.columns) == {'AMZN', 'UBSG.SW', 'HOLN.SW'}


def test_panel_columns_are_numeric(analyzer):
    """Each column carries float prices, not NaN-only."""
    df = analyzer.get_price_panel()
    for ticker in df.columns:
        col = df[ticker].dropna()
        assert len(col) > 0
        assert pd.api.types.is_numeric_dtype(col)


def test_panel_index_is_datetime(analyzer):
    """Index is a DatetimeIndex spanning the holding window."""
    df = analyzer.get_price_panel()
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df.index) > 1


def test_panel_raises_when_held_ticker_missing_from_cache(analyzer):
    """Held ticker absent from _market_data must raise.

    Per the no-silent-defaults policy: a held position missing
    from the price cache is a real failure, not a partial result.
    """
    analyzer._fetch_market_data()
    assert 'AMZN' in analyzer._market_data
    del analyzer._market_data['AMZN']
    with pytest.raises(RuntimeError, match='AMZN'):
        analyzer.get_price_panel()


def test_fetch_raises_when_provider_has_no_data(
        monkeypatch, fake_data_provider):
    """A held ticker the provider cannot serve must be fatal.

    Per the no-silent-defaults policy the fetch loop no longer
    warns-and-continues; a data-less held ticker raises so the
    symbol/ISIN mapping gets fixed rather than silently dropped.
    """
    base = datetime(2024, 1, 15)
    monkeypatch.setattr(
        CSVLoader, 'load_csv_degiro',
        lambda self, path: [
            Transaction(
                ticker='BADX', operation='buy', date=base,
                price=10.0, amount=1, fee=0.0,
                auto_fx_fee=0.0, currency='USD')])
    monkeypatch.setattr(
        CSVLoader, 'load_csv_ibkr', lambda self, path: [])

    def _raise_for_badx(ticker, start, end):
        raise ValueError(f'No price data for {ticker}')
    monkeypatch.setattr(
        fake_data_provider, 'get_price_history', _raise_for_badx)

    a = PortfolioAnalyzer(
        degiro_csv_file_path='ignored.csv',
        ibkr_csv_file_path=None,
        data_provider=fake_data_provider,
        base_currency='CHF')
    with pytest.raises(RuntimeError, match='BADX'):
        a.get_holdings_snapshot()
