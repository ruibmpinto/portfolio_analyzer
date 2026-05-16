"""Tests for the weighted strategy-CAGR forecast helper."""

import pytest

from src.dashboard_api.services.strategy_cagr import (
    forecast_strategy_cagr)


def test_forecast_full_coverage_weighted_average():
    weights = {'A': 0.5, 'B': 0.5}
    cagr = {'A': 0.20, 'B': 0.10}
    forecast_pct, missing, coverage = forecast_strategy_cagr(
        weights, cagr)
    assert forecast_pct == pytest.approx(15.0, rel=1e-9)
    assert missing == []
    assert coverage == 1.0


def test_forecast_partial_coverage_renormalises():
    """Missing tickers must be reported and matched-weight renormalised."""
    weights = {'A': 0.5, 'B': 0.3, 'C': 0.2}
    cagr = {'A': 0.20, 'C': 0.05}  # B missing
    forecast_pct, missing, coverage = forecast_strategy_cagr(
        weights, cagr)
    # weighted = 0.5*0.20 + 0.2*0.05 = 0.11; matched = 0.7
    assert forecast_pct == pytest.approx(
        0.11 / 0.7 * 100.0, rel=1e-9)
    assert missing == ['B']
    assert coverage == pytest.approx(0.7, rel=1e-9)


def test_forecast_no_coverage_returns_zero():
    weights = {'A': 1.0}
    cagr = {'B': 0.5}
    forecast_pct, missing, coverage = forecast_strategy_cagr(
        weights, cagr)
    assert forecast_pct == 0.0
    assert missing == ['A']
    assert coverage == 0.0
