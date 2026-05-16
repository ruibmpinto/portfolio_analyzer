"""Tests for DefeatBetaProvider."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.shared.data_provider import (
    DataProvider, DefeatBetaProvider)


class FakeFallback(DataProvider):
    """Records calls; returns sentinel values."""

    def __init__(self):
        self.calls = []

    def _record(self, name, *args):
        self.calls.append((name, args))

    def get_price_history(self, t, s, e):
        self._record('price', t, s, e)
        return pd.Series([1.0], index=[pd.Timestamp(s)])

    def get_fundamental_data(self, t):
        self._record('fund', t)
        return {'pe_ratio_ttm': 99.9}

    def get_dividend_history(self, t, s):
        self._record('div', t, s)
        return pd.Series([], dtype=float)

    def get_split_history(self, t):
        self._record('split', t)
        return pd.Series([], dtype=float)

    def get_dividend_price_factors(self, t):
        self._record('factors', t)
        return pd.Series([], dtype=float)

    def get_current_price(self, t):
        self._record('current', t)
        return 42.0


def _fake_price_df(symbol='NVDA', n_days=10):
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    return pd.DataFrame({
        'symbol': [symbol] * n_days,
        'report_date': dates.strftime('%Y-%m-%d'),
        'open': np.linspace(100, 110, n_days),
        'close': np.linspace(101, 111, n_days),
        'high': np.linspace(102, 112, n_days),
        'low': np.linspace(99, 109, n_days),
        'volume': [1_000_000] * n_days})


def test_is_native_for_us_ticker():
    p = DefeatBetaProvider()
    assert p.is_native('NVDA')
    assert p.is_native('BRK-B')


def test_is_native_false_for_dotted_or_fx_ticker():
    p = DefeatBetaProvider()
    assert not p.is_native('NESN.SW')
    assert not p.is_native('BARC.L')
    assert not p.is_native('USDCHF=X')


def test_get_price_history_routes_us_to_defeatbeta(monkeypatch):
    p = DefeatBetaProvider()
    fake_ticker = MagicMock()
    fake_ticker.price.return_value = _fake_price_df()
    monkeypatch.setattr(
        'src.shared.data_provider.Ticker', lambda s: fake_ticker)

    out = p.get_price_history(
        'NVDA', datetime(2024, 1, 1), datetime(2024, 1, 8))
    assert isinstance(out, pd.Series)
    assert len(out) > 0


def test_get_price_history_falls_back_for_non_us(monkeypatch):
    fb = FakeFallback()
    p = DefeatBetaProvider(fallback=fb)
    out = p.get_price_history(
        'NESN.SW', datetime(2024, 1, 1), datetime(2024, 1, 8))
    assert fb.calls[0][0] == 'price'
    assert out.iloc[0] == 1.0


def test_non_us_ticker_raises_when_no_fallback():
    p = DefeatBetaProvider(fallback=None)
    with pytest.raises(RuntimeError, match='NESN.SW'):
        p.get_price_history(
            'NESN.SW', datetime(2024, 1, 1), datetime(2024, 1, 8))


def test_get_current_price_returns_last_close(monkeypatch):
    p = DefeatBetaProvider()
    fake_ticker = MagicMock()
    fake_ticker.price.return_value = _fake_price_df()
    monkeypatch.setattr(
        'src.shared.data_provider.Ticker', lambda s: fake_ticker)
    assert p.get_current_price('NVDA') == pytest.approx(111.0)


def test_get_current_price_raises_on_empty_history(monkeypatch):
    p = DefeatBetaProvider()
    fake_ticker = MagicMock()
    fake_ticker.price.return_value = pd.DataFrame(
        columns=['symbol', 'report_date', 'close'])
    monkeypatch.setattr(
        'src.shared.data_provider.Ticker', lambda s: fake_ticker)
    with pytest.raises(ValueError, match='NVDA'):
        p.get_current_price('NVDA')


def test_get_fundamental_data_assembles_30_keys(monkeypatch):
    p = DefeatBetaProvider()
    fake_ticker = MagicMock()
    one_row_df = lambda col, val: pd.DataFrame({col: [val]})
    fake_ticker.ttm_pe.return_value = one_row_df('ttm_pe', 25.0)
    fake_ticker.ttm_eps.return_value = one_row_df('ttm_eps', 4.0)
    fake_ticker.pb_ratio.return_value = one_row_df('pb_ratio', 5.0)
    fake_ticker.ps_ratio.return_value = one_row_df('ps_ratio', 8.0)
    fake_ticker.peg_ratio.return_value = one_row_df('peg_ratio', 1.5)
    fake_ticker.market_capitalization.return_value = one_row_df(
        'market_capitalization', 3e12)
    fake_ticker.roe.return_value = one_row_df('roe', 0.30)
    fake_ticker.roa.return_value = one_row_df('roa', 0.18)
    fake_ticker.roic.return_value = one_row_df('roic', 0.25)
    fake_ticker.wacc.return_value = one_row_df('wacc', 0.08)
    fake_ticker.equity_multiplier.return_value = one_row_df(
        'equity_multiplier', 1.7)
    fake_ticker.asset_turnover.return_value = one_row_df(
        'asset_turnover', 0.6)
    fake_ticker.debt_to_equity.return_value = one_row_df(
        'debt_to_equity', 0.4)
    fake_ticker.enterprise_value.return_value = one_row_df(
        'enterprise_value', 2.9e12)
    fake_ticker.enterprise_to_revenue.return_value = one_row_df(
        'enterprise_to_revenue', 23.0)
    fake_ticker.enterprise_to_ebitda.return_value = one_row_df(
        'enterprise_to_ebitda', 50.0)
    fake_ticker.ttm_revenue.return_value = one_row_df(
        'ttm_revenue', 1.3e11)
    fake_ticker.ttm_fcf.return_value = one_row_df(
        'ttm_fcf', 5e10)
    fake_ticker.quarterly_revenue_yoy_growth.return_value = (
        one_row_df('revenue_yoy_growth', 1.2))
    fake_ticker.quarterly_ebitda_yoy_growth.return_value = (
        one_row_df('ebitda_yoy_growth', 1.5))
    fake_ticker.quarterly_eps_yoy_growth.return_value = (
        one_row_df('eps_yoy_growth', 1.4))
    fake_ticker.dcf_data.return_value = {'implied_upside': 0.30}
    monkeypatch.setattr(
        'src.shared.data_provider.Ticker', lambda s: fake_ticker)

    fund = p.get_fundamental_data('NVDA')
    assert fund['pe_ratio_ttm'] == pytest.approx(25.0)
    assert fund['peg_ratio'] == pytest.approx(1.5)
    assert fund['roic'] == pytest.approx(0.25)
    assert fund['wacc'] == pytest.approx(0.08)
    assert fund['dcf_implied_upside'] == pytest.approx(0.30)
    assert 'shares_outstanding' in fund   # 30-key envelope


def test_ticker_instance_cached_across_calls(monkeypatch):
    """Two calls for the same symbol create only one Ticker."""
    construct_count = {'n': 0}
    fake_ticker = MagicMock()
    fake_ticker.price.return_value = _fake_price_df()

    def fake_factory(s):
        construct_count['n'] += 1
        return fake_ticker

    monkeypatch.setattr(
        'src.shared.data_provider.Ticker', fake_factory)
    p = DefeatBetaProvider()
    p.get_current_price('NVDA')
    p.get_current_price('NVDA')
    assert construct_count['n'] == 1
