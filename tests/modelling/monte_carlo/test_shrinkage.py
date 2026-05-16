"""Tests for james_stein_shrink."""

import numpy as np
import pandas as pd
import pytest

from src.modelling.monte_carlo.shrinkage import (
    james_stein_shrink)


def _returns(n_days=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'A': rng.normal(0.0008, 0.012, n_days),
        'B': rng.normal(0.0002, 0.010, n_days),
        'C': rng.normal(-0.0001, 0.014, n_days),
        'D': rng.normal(0.0005, 0.011, n_days)})


def test_shrinks_toward_anchor():
    """Shrunk values lie between raw mean and the anchor."""
    df = _returns()
    raw = df.mean()
    shrunk = james_stein_shrink(df, anchor_annual=0.06)
    anchor_daily = 0.06 / 252.0
    for t in df.columns:
        lo = min(raw[t], anchor_daily)
        hi = max(raw[t], anchor_daily)
        assert lo - 1e-12 <= shrunk[t] <= hi + 1e-12


def test_explicit_intensity_one_returns_raw_means():
    """intensity=1.0 disables shrinkage."""
    df = _returns()
    raw = df.mean()
    shrunk = james_stein_shrink(
        df, anchor_annual=0.06, intensity=1.0)
    for t in df.columns:
        assert shrunk[t] == pytest.approx(raw[t])


def test_explicit_intensity_zero_collapses_to_anchor():
    """intensity=0.0 collapses every ticker to the anchor."""
    df = _returns()
    shrunk = james_stein_shrink(
        df, anchor_annual=0.06, intensity=0.0)
    anchor_daily = 0.06 / 252.0
    for t in df.columns:
        assert shrunk[t] == pytest.approx(anchor_daily)


def test_raises_when_fewer_than_two_columns():
    df = _returns().iloc[:, :1]
    with pytest.raises(ValueError, match='at least'):
        james_stein_shrink(df)


def test_returns_series_indexed_by_ticker():
    df = _returns()
    s = james_stein_shrink(df)
    assert isinstance(s, pd.Series)
    assert set(s.index) == set(df.columns)
