"""Per-calendar-year portfolio returns joined with benchmarks.

Resamples the analyzer's daily backcast return series to annual
buckets and pairs each year with the calendar-year close-to-close
return of each configured benchmark.

Functions
---------
annual_returns
    Year-keyed DataFrame of portfolio vs benchmark percent returns.
"""

import pandas as pd


def annual_returns(daily_returns, benchmark_prices):
    """Build the per-year comparison table.

    Args:
        daily_returns: ``pd.Series`` of daily simple returns
            (NOT cumulative). Indexed by trading-day date.
            Typically the output of
            ``PortfolioAnalyzer._get_portfolio_returns()`` or
            ``._get_backcast_returns()``.
        benchmark_prices: Dict[str, pd.Series] mapping benchmark
            label (e.g. 'SPY', 'DIA', 'VT') to a daily close
            ``pd.Series``.

    Returns:
        DataFrame indexed by calendar year (int). One column
        named ``portfolio_pct`` plus one column per benchmark
        suffixed ``_pct`` carrying year-over-year percent return.
        Years where the portfolio series has no data are omitted.
    """
    if daily_returns is None or len(daily_returns) == 0:
        return pd.DataFrame(
            columns=['portfolio_pct'] + [
                f'{label}_pct'
                for label in benchmark_prices.keys()])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Portfolio annual: compound daily returns within each year
    daily = pd.Series(daily_returns).dropna()
    growth = (1.0 + daily)
    portfolio_annual = (
        growth.groupby(growth.index.year).prod() - 1.0) * 100.0
    portfolio_annual.name = 'portfolio_pct'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Benchmark annual: last-close / first-close - 1 per year
    benchmark_cols = {}
    for label, series in benchmark_prices.items():
        if series is None or len(series) == 0:
            continue
        s = pd.Series(series).dropna()
        if s.empty:
            continue
        first = s.groupby(s.index.year).first()
        last = s.groupby(s.index.year).last()
        benchmark_cols[f'{label}_pct'] = (
            (last / first) - 1.0) * 100.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Join everything on the year axis
    df = pd.DataFrame({'portfolio_pct': portfolio_annual})
    for col, series in benchmark_cols.items():
        df[col] = series
    df.index.name = 'year'
    return df
