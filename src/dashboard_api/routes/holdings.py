"""Holdings endpoint: per-position breakdown with fees footer.

Combines the analyzer's CHF-denominated snapshot with the most
recent native-currency close for each ticker and the lifetime
cumulative fees so the frontend can render one table.

Routes
------
GET /api/holdings
    Per-position rows plus cumulative fees + annual returns.
"""

import math
from typing import Dict

from fastapi import APIRouter, Request

from src.dashboard_api.services import (
    annual_returns as ann_mod,
    benchmarks as bench_mod,
    fees as fees_mod,
    income as income_mod)


router = APIRouter()


@router.get('/api/holdings')
def get_holdings(request: Request):
    """Return holdings rows + fees aggregate + annual returns."""
    ctx = request.app.state.ctx
    return ctx.cache.get_or_compute(
        'holdings', lambda: _build_payload(ctx))


def _build_payload(ctx) -> Dict:
    analyzer = ctx.analyzer
    snapshot = analyzer.get_holdings_snapshot(
        categories=ctx.categories or None)
    price_panel = analyzer.get_price_panel()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-row payload with native + CHF prices
    rows = []
    for _, row in snapshot.iterrows():
        ticker = row['ticker']
        native_price = _latest_native_price(price_panel, ticker)
        rows.append({
            'ticker': ticker,
            'shares': float(row['shares']),
            'currency': row['currency'],
            'price_native': native_price,
            'price_chf': float(row['price_chf']),
            'value_chf': float(row['value_chf']),
            'weight_pct': float(row['weight_pct']),
            'sector': row.get('sector', 'Unknown'),
        })
    rows.sort(key=lambda r: r['value_chf'], reverse=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cumulative fees, dividends, and withholding tax (all in CHF)
    fees = fees_mod.cumulative_fees(
        portfolio=analyzer.portfolio,
        data_provider=ctx.data_provider,
        base_currency=ctx.base_currency)
    dividends = income_mod.cumulative_income(
        portfolio=analyzer.portfolio, operation='dividend',
        data_provider=ctx.data_provider,
        base_currency=ctx.base_currency)
    taxes = income_mod.cumulative_income(
        portfolio=analyzer.portfolio, operation='tax',
        data_provider=ctx.data_provider,
        base_currency=ctx.base_currency)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Annual return comparison (portfolio vs benchmarks)
    annual = _annual_returns_payload(ctx, analyzer)
    return {
        'rows': rows,
        'fees': fees,
        'dividends': dividends,
        'taxes': taxes,
        'annual_returns': annual,
        'total_value_chf': sum(r['value_chf'] for r in rows),
    }


def _latest_native_price(price_panel, ticker) -> float:
    """Most-recent non-NaN close from the analyzer's price panel."""
    if ticker not in price_panel.columns:
        return float('nan')
    series = price_panel[ticker].dropna()
    if series.empty:
        return float('nan')
    value = float(series.iloc[-1])
    if math.isnan(value):
        return float('nan')
    return value


def _annual_returns_payload(ctx, analyzer):
    """Return list-of-dicts ready for JSON serialisation."""
    daily = analyzer._get_portfolio_returns()
    bench_series = bench_mod.benchmark_prices(
        ctx.data_provider, tickers=tuple(['SPY', 'VT']))
    df = ann_mod.annual_returns(daily, bench_series)
    out = []
    for year, row in df.iterrows():
        rec = {'year': int(year)}
        for col in df.columns:
            val = row[col]
            rec[col] = (
                None if (val is None or _is_nan(val))
                else float(val))
        out.append(rec)
    return out


def _is_nan(x) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False
