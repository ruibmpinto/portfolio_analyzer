"""Tests for cache_refresh: resumable yfinance fetcher."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.screening.listing import (
    Listing, fundamental_field_names, weekly_series_length)


def _fake_history_df(n_rows=156):
    """Fake yfinance.Ticker.history(period='3y', interval='1wk') output."""
    idx = pd.date_range('2023-05-15', periods=n_rows, freq='W-FRI')
    return pd.DataFrame({
        'Open': np.full(n_rows, 100.0),
        'High': np.full(n_rows, 105.0),
        'Low': np.full(n_rows, 95.0),
        'Close': np.linspace(100.0, 200.0, n_rows),
        'Volume': np.full(n_rows, 1.0e6),
        'Dividends': np.zeros(n_rows),
        'Stock Splits': np.zeros(n_rows),
    }, index=idx)


def _fake_ticker_info():
    """Fake yfinance.Ticker.info dict."""
    return {
        'currency': 'USD',
        'sector': 'Technology',
        'marketCap': 3.0e12,
        'trailingPE': 28.0,
        'forwardPE': 25.0,
        'priceToBook': 45.0,
        'priceToSalesTrailing12Months': 7.0,
        'dividendYield': 0.005,
        'trailingEps': 6.5,
        'revenueGrowth': 0.10,
        'earningsGrowth': 0.15,
        'profitMargins': 0.25,
        'operatingMargins': 0.30,
        'returnOnEquity': 1.5,
        'returnOnAssets': 0.30,
        'debtToEquity': 200.0,
        'freeCashflow': 1.0e11,
        'beta': 1.2,
        'sharesOutstanding': 1.5e10,
        'shortRatio': 1.5,
    }


def _patch_yfinance(monkeypatch):
    """Install a fake yf.Ticker that returns canned history + info."""
    fake = MagicMock()
    fake.history.return_value = _fake_history_df()
    fake.info = _fake_ticker_info()
    monkeypatch.setattr(
        'src.screening.cache_refresh.yf.Ticker',
        lambda ticker: fake)


def test_fetch_one_listing_returns_listing(monkeypatch):
    """One ticker fetch produces a valid Listing."""
    from src.screening.cache_refresh import _fetch_one_listing
    _patch_yfinance(monkeypatch)
    l = _fetch_one_listing('AAPL', 'NASDAQ')
    assert isinstance(l, Listing)
    assert l.ticker == 'AAPL'
    assert l.exchange == 'NASDAQ'
    assert l.currency == 'USD'
    assert l.sector == 'Technology'
    assert len(l.closes_weekly) == weekly_series_length
    assert l.fundamentals['pe_ratio_ttm'] == 28.0


def test_fetch_one_listing_handles_short_history(monkeypatch):
    """If yfinance returns fewer than 156 weeks, pad with NaN."""
    from src.screening.cache_refresh import _fetch_one_listing
    fake = MagicMock()
    fake.history.return_value = _fake_history_df(n_rows=80)
    fake.info = _fake_ticker_info()
    monkeypatch.setattr(
        'src.screening.cache_refresh.yf.Ticker',
        lambda ticker: fake)
    l = _fetch_one_listing('AAPL', 'NASDAQ')
    assert len(l.closes_weekly) == weekly_series_length
    # First 76 entries (156 - 80) are pad NaNs
    assert np.isnan(l.closes_weekly[0])
    assert not np.isnan(l.closes_weekly[-1])


def test_refresh_exchange_writes_parquet(
    monkeypatch, tmp_path):
    """End-to-end: refresh writes shards readable by ExchangeUniverse."""
    from src.screening.cache_refresh import refresh_exchange
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    _patch_yfinance(monkeypatch)
    # Use a non-US exchange so dispatch lands on the yfinance
    # path (NASDAQ / NYSE now route to defeatbeta).
    refresh_exchange(
        'LSE', tickers=['AAPL.L', 'MSFT.L'],
        rate_limit_sleep=0.0)
    from src.screening.universe import ExchangeUniverse
    u = ExchangeUniverse.load('LSE')
    assert {l.ticker for l in u} == {'AAPL.L', 'MSFT.L'}
    # At least one shard file exists in the exchange dir
    shards = list((tmp_path / 'LSE').glob('*.parquet'))
    assert len(shards) >= 1


def test_refresh_exchange_resume_skips_existing(
    monkeypatch, tmp_path):
    """Second call with resume=True skips already-cached tickers."""
    from src.screening.cache_refresh import refresh_exchange
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    call_log = {'tickers': []}

    fake = MagicMock()
    fake.history.return_value = _fake_history_df()
    fake.info = _fake_ticker_info()

    def fake_ticker(t):
        call_log['tickers'].append(t)
        return fake

    monkeypatch.setattr(
        'src.screening.cache_refresh.yf.Ticker', fake_ticker)
    refresh_exchange(
        'LSE', tickers=['AAPL.L'],
        rate_limit_sleep=0.0)
    assert call_log['tickers'] == ['AAPL.L']

    # Resume: AAPL.L already cached; MSFT.L is new
    refresh_exchange(
        'LSE', tickers=['AAPL.L', 'MSFT.L'],
        rate_limit_sleep=0.0, resume=True)
    # Only MSFT.L was fetched on the second call
    assert call_log['tickers'] == ['AAPL.L', 'MSFT.L']


def test_refresh_exchange_no_resume_refetches(
    monkeypatch, tmp_path):
    """resume=False refetches everything."""
    from src.screening.cache_refresh import refresh_exchange
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    call_log = {'tickers': []}

    fake = MagicMock()
    fake.history.return_value = _fake_history_df()
    fake.info = _fake_ticker_info()

    def fake_ticker(t):
        call_log['tickers'].append(t)
        return fake

    monkeypatch.setattr(
        'src.screening.cache_refresh.yf.Ticker', fake_ticker)
    refresh_exchange(
        'LSE', tickers=['AAPL.L'],
        rate_limit_sleep=0.0)
    refresh_exchange(
        'LSE', tickers=['AAPL.L'],
        rate_limit_sleep=0.0, resume=False)
    # Both calls fetched AAPL.L
    assert call_log['tickers'] == ['AAPL.L', 'AAPL.L']


def test_refresh_exchange_uses_defeatbeta_for_nyse(
        tmp_path, monkeypatch):
    """NYSE / NASDAQ exchanges route to the defeatbeta path."""
    import src.screening.cache_refresh as cr

    captured = {'paths': []}

    def fake_refresh_dfb(exchange, tickers, resume, provider):
        captured['paths'].append(
            ('defeatbeta', exchange, list(tickers or [])))

    def fake_refresh_yf(*args, **kwargs):
        captured['paths'].append(('yfinance', args, kwargs))

    monkeypatch.setattr(
        cr, '_refresh_via_defeatbeta', fake_refresh_dfb)
    monkeypatch.setattr(
        cr, '_refresh_via_yfinance', fake_refresh_yf)

    cr.refresh_exchange('NASDAQ', tickers=['NVDA', 'MSFT'])
    assert captured['paths'][-1][0] == 'defeatbeta'

    cr.refresh_exchange('NYSE', tickers=['AAPL'])
    assert captured['paths'][-1][0] == 'defeatbeta'

    cr.refresh_exchange('SIX', tickers=['NESN.SW'])
    assert captured['paths'][-1][0] == 'yfinance'


def test_build_listing_via_defeatbeta_round_trips(
        tmp_path, monkeypatch):
    """End-to-end: bulk defeatbeta path -> parquet -> load."""
    from datetime import datetime, timedelta
    from unittest.mock import MagicMock

    import src.screening.cache_refresh as cr
    import src.screening.universe as universe_mod
    from src.screening.universe import ExchangeUniverse
    from src.shared.data_provider import DefeatBetaProvider

    # Build a fake DefeatBetaProvider that returns canned data
    # for one ticker.
    n_days = 1200  # ~5y of daily data
    end = datetime.now()
    dates = pd.date_range(
        end - timedelta(days=n_days), periods=n_days, freq='B')

    fake_price_df = pd.DataFrame({
        'symbol': ['NVDA'] * n_days,
        'report_date': dates.strftime('%Y-%m-%d'),
        'open': np.linspace(100, 200, n_days),
        'close': np.linspace(101, 201, n_days),
        'high': np.linspace(102, 202, n_days),
        'low': np.linspace(99, 199, n_days),
        'volume': [1_000_000] * n_days})

    fake_info_df = pd.DataFrame({'sector': ['Tech']})
    fake_div_df = pd.DataFrame({
        'symbol': ['NVDA'], 'report_date': ['2025-06-01'],
        'amount': [0.04]})

    fake_ticker = MagicMock()
    fake_ticker.price.return_value = fake_price_df
    fake_ticker.info.return_value = fake_info_df
    fake_ticker.dividends.return_value = fake_div_df
    one_row = lambda col, val: pd.DataFrame({col: [val]})
    for method, col, val in [
        ('ttm_pe', 'ttm_pe', 25.0),
        ('peg_ratio', 'peg_ratio', 1.5),
        ('roic', 'roic', 0.25),
        ('wacc', 'wacc', 0.08),
        ('market_capitalization', 'market_capitalization',
         3e12)]:
        getattr(fake_ticker, method).return_value = (
            one_row(col, val))
    fake_ticker.dcf_data.return_value = {
        'implied_upside': 0.32}
    for method in [
        'pb_ratio', 'ps_ratio', 'ttm_eps',
        'quarterly_revenue_yoy_growth',
        'quarterly_eps_yoy_growth',
        'quarterly_ebitda_yoy_growth',
        'roe', 'roa', 'debt_to_equity', 'ttm_fcf',
        'equity_multiplier', 'asset_turnover',
        'enterprise_value', 'enterprise_to_revenue',
        'enterprise_to_ebitda', 'ttm_revenue']:
        getattr(fake_ticker, method).return_value = (
            pd.DataFrame())
    fake_ticker.splits.return_value = pd.DataFrame()

    monkeypatch.setattr(
        'src.shared.data_provider.Ticker',
        lambda s: fake_ticker)

    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)

    cr.refresh_exchange(
        'NASDAQ',
        tickers=['NVDA'],
        defeatbeta_provider=DefeatBetaProvider())

    universe = ExchangeUniverse.load('NASDAQ')
    assert len(universe) == 1
    listing = list(universe)[0]
    assert listing.ticker == 'NVDA'
    assert listing.exchange == 'NASDAQ'
    assert listing.currency == 'USD'
    assert listing.sector == 'Tech'
    assert listing.fundamentals['peg_ratio'] == 1.5
    assert listing.fundamentals['roic'] == 0.25
    assert listing.fundamentals['wacc'] == 0.08
    assert listing.fundamentals['dcf_implied_upside'] == 0.32
    assert len(listing.closes_weekly) == 156
