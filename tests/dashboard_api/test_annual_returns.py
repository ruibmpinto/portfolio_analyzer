"""Tests for the annual_returns aggregator."""

import numpy as np
import pandas as pd
import pytest

from src.dashboard_api.services.annual_returns import annual_returns


def _daily_constant_return(year, daily_rate, n_days=252):
    """Build a daily-return Series with a constant rate."""
    dates = pd.bdate_range(f'{year}-01-02', periods=n_days)
    return pd.Series([daily_rate] * n_days, index=dates)


def test_annual_returns_constant_daily_compounds_correctly():
    """A constant 0.1%/day for 252 trading days = ~28.6% annual."""
    daily = _daily_constant_return(2024, 0.001, n_days=252)
    df = annual_returns(daily, {})
    assert list(df.columns) == ['portfolio_pct']
    assert df.index.tolist() == [2024]
    expected = ((1.001 ** 252) - 1.0) * 100.0
    assert df.loc[2024, 'portfolio_pct'] == pytest.approx(
        expected, rel=1e-9)


def test_annual_returns_two_years_separately_compounded():
    s24 = _daily_constant_return(2024, 0.001, n_days=252)
    s25 = _daily_constant_return(2025, 0.0005, n_days=252)
    daily = pd.concat([s24, s25])
    df = annual_returns(daily, {})
    assert df.index.tolist() == [2024, 2025]
    exp24 = ((1.001 ** 252) - 1.0) * 100.0
    exp25 = ((1.0005 ** 252) - 1.0) * 100.0
    assert df.loc[2024, 'portfolio_pct'] == pytest.approx(
        exp24, rel=1e-9)
    assert df.loc[2025, 'portfolio_pct'] == pytest.approx(
        exp25, rel=1e-9)


def test_annual_returns_includes_benchmark_first_last_pct():
    """Benchmark returns are last/first - 1 per year."""
    daily = _daily_constant_return(2024, 0.0, n_days=252)
    dates = pd.bdate_range('2024-01-02', periods=252)
    spy = pd.Series(
        np.linspace(100.0, 130.0, 252), index=dates)
    df = annual_returns(daily, {'SPY': spy})
    assert 'SPY_pct' in df.columns
    assert df.loc[2024, 'SPY_pct'] == pytest.approx(
        30.0, rel=1e-9)


def test_annual_returns_empty_input_returns_empty_frame():
    df = annual_returns(pd.Series(dtype=float), {})
    assert df.empty
    assert 'portfolio_pct' in df.columns
