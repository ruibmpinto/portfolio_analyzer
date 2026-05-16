"""Tests for ExchangeUniverse: shard I/O + iteration helpers."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.screening.listing import Listing, fundamental_field_names
from src.screening.universe import ExchangeUniverse


def _make_listing(ticker, exchange='NASDAQ', refreshed_at=None):
    n = 156
    dates = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    closes = np.linspace(100.0, 200.0, n)
    volumes = np.full(n, 1.0e6)
    dividends = np.zeros(n)
    fundamentals = {k: 1.0 for k in fundamental_field_names}
    return Listing(
        ticker=ticker, exchange=exchange, currency='USD',
        sector='Technology', market_cap=1.0e9,
        closes_weekly=closes, closes_weekly_dates=dates,
        volumes_weekly=volumes, dividends_weekly=dividends,
        fundamentals=fundamentals,
        refreshed_at=refreshed_at or datetime(2026, 5, 14, 12, 0))


def test_universe_construct():
    listings = [_make_listing('A'), _make_listing('B')]
    u = ExchangeUniverse('NASDAQ', listings)
    assert len(u) == 2
    assert list(u)[0].ticker == 'A'


def test_universe_save_and_load_round_trip(tmp_path, monkeypatch):
    """Save then load returns equivalent listings."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    listings = [
        _make_listing('AAPL'),
        _make_listing('MSFT')]
    u = ExchangeUniverse('NASDAQ', listings)
    u.save()
    loaded = ExchangeUniverse.load('NASDAQ')
    assert len(loaded) == 2
    by_ticker = {l.ticker: l for l in loaded}
    assert set(by_ticker) == {'AAPL', 'MSFT'}
    aapl = by_ticker['AAPL']
    assert aapl.exchange == 'NASDAQ'
    assert aapl.currency == 'USD'
    assert aapl.sector == 'Technology'
    assert aapl.market_cap == pytest.approx(1.0e9)
    assert len(aapl.closes_weekly) == 156
    assert aapl.closes_weekly[0] == pytest.approx(100.0)
    assert aapl.closes_weekly[-1] == pytest.approx(200.0)
    assert aapl.fundamentals['pe_ratio_ttm'] == 1.0


def test_universe_save_writes_to_exchange_subdir(
    tmp_path, monkeypatch):
    """save() creates `<exchange>/<timestamp>.parquet`."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    u = ExchangeUniverse('NASDAQ', [_make_listing('AAPL')])
    u.save()
    exchange_dir = tmp_path / 'NASDAQ'
    assert exchange_dir.is_dir()
    shards = list(exchange_dir.glob('*.parquet'))
    assert len(shards) == 1
    leftovers = list(exchange_dir.glob('*.tmp*'))
    assert leftovers == []


def test_universe_save_appends_new_shards(tmp_path, monkeypatch):
    """Each save() writes a new shard; existing shards untouched."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    ExchangeUniverse(
        'NASDAQ', [_make_listing('AAPL')]).save()
    ExchangeUniverse(
        'NASDAQ', [_make_listing('MSFT')]).save()
    shards = list((tmp_path / 'NASDAQ').glob('*.parquet'))
    assert len(shards) == 2


def test_universe_load_dedupes_by_latest_refreshed_at(
    tmp_path, monkeypatch):
    """When two shards reference the same ticker, latest wins."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    early = _make_listing(
        'AAPL', refreshed_at=datetime(2026, 5, 1, 10, 0))
    late = _make_listing(
        'AAPL', refreshed_at=datetime(2026, 5, 14, 10, 0))
    ExchangeUniverse('NASDAQ', [early]).save()
    ExchangeUniverse('NASDAQ', [late]).save()
    loaded = ExchangeUniverse.load('NASDAQ')
    by_ticker = {l.ticker: l for l in loaded}
    assert len(by_ticker) == 1
    assert by_ticker['AAPL'].refreshed_at == datetime(
        2026, 5, 14, 10, 0)


def test_universe_load_missing_directory_raises(
    tmp_path, monkeypatch):
    """Loading a non-existent exchange raises FileNotFoundError."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    with pytest.raises(FileNotFoundError):
        ExchangeUniverse.load('NOPE')


def test_universe_load_empty_directory_raises(
    tmp_path, monkeypatch):
    """Empty exchange directory also raises."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    (tmp_path / 'NASDAQ').mkdir()
    with pytest.raises(FileNotFoundError):
        ExchangeUniverse.load('NASDAQ')


def test_universe_filter_returns_subset():
    """filter() returns a new ExchangeUniverse with the subset."""
    listings = [_make_listing('A'), _make_listing('B')]
    u = ExchangeUniverse('NASDAQ', listings)
    sub = u.filter(lambda l: l.ticker == 'A')
    assert len(sub) == 1
    assert next(iter(sub)).ticker == 'A'
    assert sub.exchange == 'NASDAQ'


def test_universe_to_dataframe_one_row_per_listing():
    listings = [_make_listing('A'), _make_listing('B')]
    df = ExchangeUniverse('NASDAQ', listings).to_dataframe()
    assert len(df) == 2
    assert set(df['ticker']) == {'A', 'B'}
    assert {'currency', 'sector', 'market_cap',
            'pe_ratio_ttm'}.issubset(df.columns)


def test_universe_staleness_days_property():
    """staleness_days returns the age of the oldest refreshed_at."""
    fresh = _make_listing(
        'A', refreshed_at=datetime.now() - timedelta(days=2))
    stale = _make_listing(
        'B', refreshed_at=datetime.now() - timedelta(days=10))
    u = ExchangeUniverse('NASDAQ', [fresh, stale])
    assert 9.0 <= u.staleness_days <= 11.0


def test_universe_load_back_compat_with_legacy_17_field_shards(
    tmp_path, monkeypatch):
    """Old parquet shards (17 fundamental cols) still load.

    Older shards were written before Phase 8 added the 13
    new fields. Loading them should fill missing columns
    with NaN rather than raise.
    """
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    n = 156
    legacy_row = {
        'ticker': 'OLDTICK',
        'currency': 'USD',
        'sector': 'Tech',
        'market_cap': 1.0e10,
        'refreshed_at': datetime(
            2026, 5, 14, 12, 0).isoformat(),
        'closes_weekly': [100.0] * n,
        'closes_weekly_dates': [
            (pd.Timestamp('2023-05-15')
             + pd.Timedelta(weeks=i)).isoformat()
            for i in range(n)],
        'volumes_weekly': [1e6] * n,
        'dividends_weekly': [0.0] * n,
        # Original 17 fundamentals only
        'pe_ratio_ttm': 25.0,
        'pe_ratio_forward': 22.0,
        'pb_ratio': 5.0,
        'ps_ratio': 8.0,
        'dividend_yield_ttm': 0.01,
        'eps_ttm': 4.0,
        'revenue_growth_yoy': 0.20,
        'earnings_growth_yoy': 0.30,
        'profit_margin': 0.25,
        'operating_margin': 0.30,
        'roe': 0.30,
        'roa': 0.18,
        'debt_to_equity': 0.4,
        'free_cash_flow': 5e10,
        'beta_yf': 1.5,
        'shares_outstanding': 2.5e9,
        'short_ratio': 1.2,
    }
    shard_dir = tmp_path / 'NASDAQ'
    shard_dir.mkdir(parents=True)
    df = pd.DataFrame([legacy_row])
    df.to_parquet(
        shard_dir / '20260101T000000.parquet', engine='pyarrow')

    u = ExchangeUniverse.load('NASDAQ')
    assert len(u) == 1
    l = list(u)[0]
    assert l.ticker == 'OLDTICK'
    # Original fields preserved
    assert l.fundamentals['pe_ratio_ttm'] == 25.0
    assert l.fundamentals['short_ratio'] == 1.2
    # New fields NaN-defaulted
    for new_field in (
        'peg_ratio', 'roic', 'roce', 'wacc',
        'equity_multiplier', 'asset_turnover',
        'enterprise_value', 'enterprise_to_revenue',
        'enterprise_to_ebitda', 'ttm_revenue',
        'ebitda_growth_yoy', 'market_cap_chf',
        'dcf_implied_upside'):
        v = l.fundamentals[new_field]
        assert v != v, (
            f'{new_field} should be NaN, got {v!r}')


def test_universe_save_empty_is_noop(tmp_path, monkeypatch):
    """Saving an empty universe writes nothing."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    ExchangeUniverse('NASDAQ', []).save()
    # Either no directory or an empty directory is acceptable
    exchange_dir = tmp_path / 'NASDAQ'
    if exchange_dir.exists():
        assert list(exchange_dir.glob('*.parquet')) == []
