"""Integration tests for refresh_screener_cache.py and run_screener.py CLIs."""

import json
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.screening.listing import fundamental_field_names


def _fake_history_df():
    n = 156
    idx = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    return pd.DataFrame({
        'Open': np.full(n, 100.0),
        'High': np.full(n, 105.0),
        'Low': np.full(n, 95.0),
        'Close': np.linspace(100.0, 200.0, n),
        'Volume': np.full(n, 1.0e6),
        'Dividends': np.zeros(n),
        'Stock Splits': np.zeros(n),
    }, index=idx)


def _fake_ticker_info():
    return {
        'currency': 'USD', 'sector': 'Technology',
        'marketCap': 3.0e12,
        'trailingPE': 28.0, 'forwardPE': 25.0,
        'priceToBook': 45.0,
        'priceToSalesTrailing12Months': 7.0,
        'dividendYield': 0.005, 'trailingEps': 6.5,
        'revenueGrowth': 0.10, 'earningsGrowth': 0.15,
        'profitMargins': 0.25, 'operatingMargins': 0.30,
        'returnOnEquity': 1.5, 'returnOnAssets': 0.30,
        'debtToEquity': 200.0, 'freeCashflow': 1.0e11,
        'beta': 1.2, 'sharesOutstanding': 1.5e10,
        'shortRatio': 1.5,
    }


def _patch_yfinance(monkeypatch):
    fake = MagicMock()
    fake.history.return_value = _fake_history_df()
    fake.info = _fake_ticker_info()
    monkeypatch.setattr(
        'src.screening.cache_refresh.yf.Ticker',
        lambda ticker: fake)


def test_refresh_cli_writes_parquet(monkeypatch, tmp_path):
    """refresh_screener_cache --exchange NASDAQ writes a parquet."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    tickers_dir = tmp_path / 'tickers'
    tickers_dir.mkdir()
    (tickers_dir / 'NASDAQ.txt').write_text('AAPL\nMSFT\n')
    _patch_yfinance(monkeypatch)
    from src.user_scripts import refresh_screener_cache
    monkeypatch.setattr(
        refresh_screener_cache,
        'tickers_dir', tickers_dir)
    monkeypatch.setattr(sys, 'argv', [
        'refresh_screener_cache.py',
        '--exchange', 'NASDAQ',
        '--rate-limit', '0',
    ])
    refresh_screener_cache.main()
    shards = list((tmp_path / 'NASDAQ').glob('*.parquet'))
    assert len(shards) >= 1


def test_refresh_cli_missing_ticker_list_raises(
    monkeypatch, tmp_path):
    """Missing ticker list file raises a clear error."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    from src.user_scripts import refresh_screener_cache
    monkeypatch.setattr(
        refresh_screener_cache,
        'tickers_dir', tmp_path / 'no_dir')
    monkeypatch.setattr(sys, 'argv', [
        'refresh_screener_cache.py',
        '--exchange', 'XETRA'])
    with pytest.raises(SystemExit, match='ticker list'):
        refresh_screener_cache.main()


def _seed_universe(monkeypatch, tmp_path):
    """Write a small parquet for the run_screener test."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    from src.screening.listing import (
        Listing, fundamental_field_names)
    from src.screening.universe import ExchangeUniverse
    from datetime import datetime
    n = 156
    dates = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    listings = []
    for ticker, mom in (('A', 0.05), ('B', 0.20), ('C', 0.10)):
        closes = np.full(n, 100.0)
        closes[-1] = 100.0 * (1.0 + mom)
        fund = {k: 1.0 for k in fundamental_field_names}
        listings.append(Listing(
            ticker=ticker, exchange='NASDAQ',
            currency='USD', sector='Technology',
            market_cap=1.0e9,
            closes_weekly=closes,
            closes_weekly_dates=dates,
            volumes_weekly=np.full(n, 1.0e6),
            dividends_weekly=np.zeros(n),
            fundamentals=fund,
            refreshed_at=datetime(2026, 5, 14, 12, 0)))
    ExchangeUniverse('NASDAQ', listings).save()


def test_run_screener_writes_excel(monkeypatch, tmp_path):
    """run_screener --output-format excel writes an .xlsx."""
    _seed_universe(monkeypatch, tmp_path)
    out_path = tmp_path / 'ranked.xlsx'
    monkeypatch.setattr(sys, 'argv', [
        'run_screener.py',
        '--exchange', 'NASDAQ',
        '--criteria', 'momentum',
        '--top-n', '10',
        '--output-format', 'excel',
        '--output', str(out_path)])
    from src.user_scripts import run_screener
    run_screener.main()
    assert out_path.exists()
    df = pd.read_excel(out_path)
    assert 'composite_score' in df.columns
    assert df.iloc[0]['ticker'] == 'B'    # best momentum


def test_run_screener_writes_json(monkeypatch, tmp_path):
    """run_screener --output-format json writes a .json."""
    _seed_universe(monkeypatch, tmp_path)
    out_path = tmp_path / 'ranked.json'
    monkeypatch.setattr(sys, 'argv', [
        'run_screener.py',
        '--exchange', 'NASDAQ',
        '--criteria', 'momentum',
        '--output-format', 'json',
        '--output', str(out_path)])
    from src.user_scripts import run_screener
    run_screener.main()
    data = json.loads(out_path.read_text())
    assert isinstance(data, list)
    assert data[0]['ticker'] == 'B'
    assert 'composite_score' in data[0]


def test_run_screener_unknown_criterion_raises(
    monkeypatch, tmp_path):
    """Unknown --criteria value raises."""
    _seed_universe(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, 'argv', [
        'run_screener.py',
        '--exchange', 'NASDAQ',
        '--criteria', 'not_real',
        '--output-format', 'json',
        '--output', str(tmp_path / 'r.json')])
    from src.user_scripts import run_screener
    with pytest.raises(KeyError):
        run_screener.main()
