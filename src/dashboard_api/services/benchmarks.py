"""Benchmark price fetcher used by the overview and annual-returns
endpoints.

Pulls daily close history for the configured benchmark tickers
through whatever ``DataProvider`` the analyzer is using, then
aligns each series to the portfolio's own date index so the
frontend can plot like-for-like.

Functions
---------
benchmark_prices
    Per-ticker daily close ``pd.Series`` for the configured set.
normalised_levels
    Rebase each price series to 100 at the first shared date.
"""

from datetime import datetime, timedelta

import pandas as pd


default_benchmarks = ('SPY', 'DIA', 'VT')
default_lookback_days = 10 * 365


def benchmark_prices(
        data_provider,
        tickers=default_benchmarks,
        lookback_days=default_lookback_days):
    """Fetch daily close ``pd.Series`` for each benchmark ticker.

    Args:
        data_provider: ``DataProvider`` instance.
        tickers: Iterable of benchmark ticker strings.
        lookback_days: How far back to request data, in calendar
            days. Defaults to ~10 years.

    Returns:
        Dict[str, pd.Series] mapping ticker to daily close prices.
        Tickers whose lookup fails are omitted from the dict so
        a single benchmark outage cannot break the dashboard.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    out = {}
    for ticker in tickers:
        try:
            series = data_provider.get_price_history(
                ticker, start_date=start, end_date=end)
        except Exception:
            continue
        if series is None or len(series) == 0:
            continue
        out[ticker] = series
    return out


def normalised_levels(portfolio_nav, benchmark_series_by_label):
    """Rebase portfolio NAV and benchmarks to 100 at first date.

    Args:
        portfolio_nav: ``pd.Series`` of portfolio NAV (CHF or
            otherwise) indexed by date.
        benchmark_series_by_label: Dict label -> ``pd.Series``
            of daily closes.

    Returns:
        DataFrame indexed by date with columns ``portfolio`` and
        one column per benchmark, each rebased to 100.0 at the
        first date the column has a value.
    """
    cols = {'portfolio': pd.Series(portfolio_nav).dropna()}
    for label, series in benchmark_series_by_label.items():
        cols[label] = pd.Series(series).dropna()
    df = pd.DataFrame(cols)
    # Align on the intersection of all series so the rebase
    # anchor is a shared date.
    df = df.dropna(how='any')
    if df.empty:
        return df
    base = df.iloc[0]
    return df.divide(base).multiply(100.0)
