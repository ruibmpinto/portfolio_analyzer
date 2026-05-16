"""Overview endpoint: top-line KPIs + benchmark comparison.

Returns the headline numbers (NAV, day P&L, YTD/1Y/3Y return)
plus a rebased portfolio-vs-benchmark series suitable for the
Overview tab's normalised line chart.

Routes
------
GET /api/overview
"""

import math
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, Request

from src.dashboard_api.services import benchmarks as bench_mod
from src.shared.metrics.volatility import VolatilityAnalyzer


router = APIRouter()

trading_days_per_year = 252
ytd_lookback_days = 365
one_year_days = 252
three_year_days = 252 * 3


@router.get('/api/overview')
def get_overview(request: Request):
    """Return headline KPIs and normalised performance series."""
    ctx = request.app.state.ctx
    return ctx.cache.get_or_compute(
        'overview', lambda: _build_payload(ctx))


def _build_payload(ctx) -> Dict:
    analyzer = ctx.analyzer
    snapshot = analyzer.get_holdings_snapshot(
        categories=ctx.categories or None)
    total_value_chf = float(snapshot['value_chf'].sum())
    daily_returns = analyzer._get_portfolio_returns()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # NAV series for the performance chart (normalised)
    nav_series = _nav_series_from_returns(daily_returns)
    benchmarks_raw = bench_mod.benchmark_prices(
        ctx.data_provider)
    normalised = bench_mod.normalised_levels(
        portfolio_nav=nav_series,
        benchmark_series_by_label=benchmarks_raw)
    perf_series = _df_to_records(normalised)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # KPI: trailing returns
    return_ytd = _trailing_return(daily_returns, _ytd_index(daily_returns))
    return_1y = _trailing_return(
        daily_returns, daily_returns.index[-one_year_days:])
    return_3y = _trailing_return(
        daily_returns, daily_returns.index[-three_year_days:])
    last_day_return = _safe_last(daily_returns)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Risk metrics: Sharpe / Sortino / MaxDD / CAGR
    risk = _risk_block(analyzer, daily_returns)
    return {
        'total_value_chf': total_value_chf,
        'day_pnl_pct': last_day_return * 100.0,
        'return_ytd_pct': return_ytd * 100.0,
        'return_1y_pct': return_1y * 100.0,
        'return_3y_pct': return_3y * 100.0,
        'performance_series': perf_series,
        'benchmark_labels': list(benchmarks_raw.keys()),
        'risk': risk,
    }


def _risk_block(analyzer, daily_returns) -> Dict:
    """Compute Sharpe, Sortino, MaxDD, and CAGR from the backcast."""
    sharpe = analyzer.get_sharpe()
    sortino = analyzer.get_sortino_ratio()
    # Max drawdown (annualised analyzer expects daily returns)
    max_dd_pct = VolatilityAnalyzer().compute_max_drawdown(
        pd.Series(daily_returns).dropna())
    # CAGR from compounded daily returns over the elapsed window
    cagr_pct = _cagr_pct(daily_returns)
    return {
        'sharpe': float(sharpe) if sharpe is not None else None,
        'sortino': float(sortino) if sortino is not None else None,
        'max_drawdown_pct': float(max_dd_pct),
        'cagr_pct': float(cagr_pct),
    }


def _cagr_pct(daily_returns) -> float:
    """Compound annual growth rate as percent.

    Computed as ``(prod(1+r) ** (252/N) - 1) * 100`` over the
    non-NaN daily-return window. Returns 0.0 on empty input.
    """
    daily = pd.Series(daily_returns).dropna()
    if daily.empty:
        return 0.0
    total = float((1.0 + daily).prod())
    n_days = len(daily)
    if n_days <= 0 or total <= 0:
        return 0.0
    annual = total ** (trading_days_per_year / n_days) - 1.0
    return annual * 100.0


def _nav_series_from_returns(daily_returns) -> pd.Series:
    """Compound daily returns to a notional NAV series (base=100)."""
    daily = pd.Series(daily_returns).dropna()
    if daily.empty:
        return daily
    nav = (1.0 + daily).cumprod() * 100.0
    nav.iloc[0] = 100.0
    return nav


def _ytd_index(series):
    """Slice index of `series` starting at Jan 1 of the latest year."""
    if series is None or len(series) == 0:
        return []
    latest_year = series.index[-1].year
    mask = series.index.year == latest_year
    return series.index[mask]


def _trailing_return(daily_returns, idx) -> float:
    """Cumulative return over the dates in `idx`. 0.0 on empty."""
    if idx is None or len(idx) == 0:
        return 0.0
    sliced = pd.Series(daily_returns).reindex(idx).dropna()
    if sliced.empty:
        return 0.0
    return float((1.0 + sliced).prod() - 1.0)


def _safe_last(series) -> float:
    s = pd.Series(series).dropna()
    if s.empty:
        return 0.0
    return float(s.iloc[-1])


def _df_to_records(df: pd.DataFrame) -> List[Dict]:
    """Convert a date-indexed DataFrame to JSON-serialisable list."""
    records = []
    for ts, row in df.iterrows():
        rec = {'date': ts.strftime('%Y-%m-%d')}
        for col in df.columns:
            val = row[col]
            rec[col] = (
                None if (val is None or _is_nan(val))
                else float(val))
        records.append(rec)
    return records


def _is_nan(x) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False
