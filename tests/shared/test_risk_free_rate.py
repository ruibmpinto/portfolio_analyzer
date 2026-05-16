"""Tests for src.shared.risk_free_rate."""

import warnings

import pandas as pd
import pytest

from src.shared import risk_free_rate


def test_returns_float_from_fred(monkeypatch):
    """Live fetch returns the latest FRED value as a decimal."""
    risk_free_rate._cache.clear()
    fake_df = pd.DataFrame(
        {'IRLTLT01CHM156N': [0.42, 0.55, 0.61]},
        index=pd.date_range('2024-01-01', periods=3, freq='ME'))

    def fake_reader(series_id, source):
        assert series_id == 'IRLTLT01CHM156N'
        assert source == 'fred'
        return fake_df

    monkeypatch.setattr(
        'src.shared.risk_free_rate.pdr.DataReader',
        fake_reader)
    rate = risk_free_rate.get_live_risk_free_rate()
    assert isinstance(rate, float)
    # FRED reports percent (0.61); function returns decimal.
    assert rate == pytest.approx(0.0061)


def test_caches_within_process(monkeypatch):
    """Second call hits cache, not pandas_datareader."""
    risk_free_rate._cache.clear()
    call_count = {'n': 0}

    def fake_reader(series_id, source):
        call_count['n'] += 1
        return pd.DataFrame(
            {'x': [1.23]},
            index=pd.date_range('2024-01-01', periods=1, freq='ME'))

    monkeypatch.setattr(
        'src.shared.risk_free_rate.pdr.DataReader',
        fake_reader)
    risk_free_rate.get_live_risk_free_rate()
    risk_free_rate.get_live_risk_free_rate()
    assert call_count['n'] == 1


def test_cache_can_be_disabled(monkeypatch):
    """use_cache=False forces a re-fetch."""
    risk_free_rate._cache.clear()
    call_count = {'n': 0}

    def fake_reader(series_id, source):
        call_count['n'] += 1
        return pd.DataFrame(
            {'x': [0.7]},
            index=pd.date_range('2024-01-01', periods=1, freq='ME'))

    monkeypatch.setattr(
        'src.shared.risk_free_rate.pdr.DataReader',
        fake_reader)
    risk_free_rate.get_live_risk_free_rate(use_cache=False)
    risk_free_rate.get_live_risk_free_rate(use_cache=False)
    assert call_count['n'] == 2


def test_falls_back_on_fetch_failure(monkeypatch):
    """Network failure triggers fallback rate + warning."""
    risk_free_rate._cache.clear()

    def fake_reader(series_id, source):
        raise ConnectionError('FRED unreachable')

    monkeypatch.setattr(
        'src.shared.risk_free_rate.pdr.DataReader',
        fake_reader)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rate = risk_free_rate.get_live_risk_free_rate()
    assert rate == pytest.approx(
        risk_free_rate.fallback_rate_decimal)
    assert len(caught) == 1
    assert 'fallback' in str(caught[0].message).lower()


def test_fallback_not_cached(monkeypatch):
    """A failed fetch must not poison the cache.

    First call fails -> fallback returned. Second call (after
    fixing the data source) must perform a live fetch, not
    return the cached fallback.
    """
    risk_free_rate._cache.clear()
    state = {'mode': 'fail'}

    def fake_reader(series_id, source):
        if state['mode'] == 'fail':
            raise ConnectionError('FRED unreachable')
        return pd.DataFrame(
            {'x': [0.88]},
            index=pd.date_range('2024-01-01', periods=1, freq='ME'))

    monkeypatch.setattr(
        'src.shared.risk_free_rate.pdr.DataReader',
        fake_reader)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        first = risk_free_rate.get_live_risk_free_rate()
    assert first == pytest.approx(
        risk_free_rate.fallback_rate_decimal)

    state['mode'] = 'ok'
    second = risk_free_rate.get_live_risk_free_rate()
    # FRED 0.88% -> decimal 0.0088.
    assert second == pytest.approx(0.0088)
