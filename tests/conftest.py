"""
Shared pytest fixtures for the stock_market test suite.

Provides:
    fake_transactions: minimal cross-broker transaction list
        for analyzer tests.
    fake_data_provider: DataProvider stub that returns
        synthetic price, split, dividend, and fundamentals data
        so analyzer tests run without yfinance.
    patched_csv_loader: monkeypatches CSVLoader.load_csv_degiro
        and load_csv_ibkr to return preset Transaction lists so
        analyzer tests don't need on-disk CSV fixtures.
    analyzer: fully wired PortfolioAnalyzer using the three
        fixtures above.
"""

import pathlib
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.analysis.core.transaction import Transaction
from src.analysis.core.analyzer import PortfolioAnalyzer
from src.analysis.loaders.csv_loader import CSVLoader


@pytest.fixture
def fake_transactions():
    """
    Minimal cross-broker transaction list.

    Returns:
        Dict {'degiro': list[Transaction], 'ibkr':
        list[Transaction]}.
    """
    base = datetime(2024, 1, 15)
    degiro = [
        Transaction(
            ticker='AMZN', operation='buy', date=base,
            price=150.0, amount=5, fee=0.5,
            auto_fx_fee=0.1, currency='USD'),
        Transaction(
            ticker='UBSG.SW', operation='buy',
            date=base + timedelta(days=10),
            price=25.0, amount=1, fee=0.5,
            auto_fx_fee=0.0, currency='CHF'),
    ]
    ibkr = [
        Transaction(
            ticker='HOLN.SW', operation='buy',
            date=base + timedelta(days=20),
            price=70.0, amount=2, fee=1.5,
            auto_fx_fee=0.0, currency='CHF'),
    ]
    return {'degiro': degiro, 'ibkr': ibkr}


class _FakeDataProvider:
    """
    DataProvider stub returning deterministic synthetic data.

    Methods mirror the abstract DataProvider surface used by
    PortfolioAnalyzer: get_price_history, get_split_history,
    get_dividend_price_factors, get_dividend_history,
    get_current_price, get_fundamental_data.
    """

    sector_map = {
        'AMZN': 'Consumer Cyclical',
        'UBSG.SW': 'Financial Services',
        'HOLN.SW': 'Basic Materials',
    }

    def get_price_history(self, ticker, start, end):
        dates = pd.date_range(start=start, end=end, freq='D')
        seed = sum(ord(c) for c in ticker) % 50
        prices = 100.0 + seed + np.sin(
            np.arange(len(dates)) / 30.0) * 5.0
        return pd.Series(prices, index=dates)

    def get_split_history(self, ticker):
        return pd.Series(dtype=float)

    def get_dividend_price_factors(self, ticker):
        return pd.Series(dtype=float)

    def get_dividend_history(self, ticker, start):
        return pd.Series(dtype=float)

    def get_current_price(self, ticker):
        # Deterministic, ticker-seeded last value
        seed = sum(ord(c) for c in ticker) % 50
        return 100.0 + seed

    def get_fundamental_data(self, ticker):
        sector = self.sector_map.get(ticker)
        if sector is None:
            return {}
        return {'sector': sector}


@pytest.fixture
def fake_data_provider():
    """Instance of _FakeDataProvider for tests."""
    return _FakeDataProvider()


@pytest.fixture
def patched_csv_loader(monkeypatch, fake_transactions):
    """
    Patch CSVLoader to return preset transactions, no disk I/O.

    Routes:
        load_csv_degiro -> fake_transactions['degiro']
        load_csv_ibkr   -> fake_transactions['ibkr']
    """
    monkeypatch.setattr(
        CSVLoader, 'load_csv_degiro',
        lambda self, path: fake_transactions['degiro'])
    monkeypatch.setattr(
        CSVLoader, 'load_csv_ibkr',
        lambda self, path: fake_transactions['ibkr'])


@pytest.fixture
def analyzer(patched_csv_loader, fake_data_provider):
    """
    PortfolioAnalyzer wired to fake transactions + provider.

    Both CSV paths are non-None so the loader is invoked, but
    the monkeypatched loader returns the fake transactions
    regardless of path content.
    """
    return PortfolioAnalyzer(
        degiro_csv_file_path='ignored.csv',
        ibkr_csv_file_path='ignored.csv',
        data_provider=fake_data_provider,
        base_currency='CHF')
