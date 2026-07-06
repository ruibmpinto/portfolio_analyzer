"""Swiss safe-harbor tracker service.

Assembles the payload for ``GET /api/safe_harbor`` from three
existing primitives:

- ``PortfolioAnalyzer._calculate_value`` — for the Jan-1 NAV of the
  current calendar year.
- ``Portfolio.get_turnover`` — for the YTD buy+sell notional per
  currency.
- ``RealizedGains`` — for YTD realized P&L and the sub-6-month
  holding-period violations.

The two Swiss safe-harbor tests exposed here:

- **Turnover < 5x portfolio value at Jan 1**: ``ytd_turnover_ratio``.
- **Every closed position held >= 6 months**:
  ``holding_period_violations`` list.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.analysis.core.realized_gains import RealizedGains


def _first_trading_day_of_year(analyzer, year: int) -> datetime:
    """Return the earliest date >= Jan 1 of ``year`` for which the
    analyzer has FX + price data.

    Jan 1 itself is often a market holiday and has no FX quote; a
    NAV valuation there would raise inside ``_native_to_base``. We
    look for the first date in the analyzer's cached price panel
    that lands on or after Jan 1. Falls back to the first business
    day (via ``pd.bdate_range``) when market data is empty.
    """
    year_start = pd.Timestamp(datetime(year, 1, 1))
    for series in analyzer._market_data.values():
        if series is None or series.empty:
            continue
        in_year = series[series.index >= year_start]
        if not in_year.empty:
            return in_year.index[0].to_pydatetime()
    # No market data cached: use the first business day of the year.
    return pd.bdate_range(
        start=year_start, periods=1)[0].to_pydatetime()


# Swiss safe-harbor 5x-of-Jan-1-NAV turnover ceiling.
_turnover_threshold_ratio = 5.0
# Below this ratio the tracker reports 'ok'; between here and
# threshold, 'warning' (approaching the limit).
_turnover_warning_ratio = 4.0
# Safe-harbor minimum holding period, calendar days.
_holding_period_min_days = 180


def build_safe_harbor_payload(
        ctx, today: Optional[datetime] = None) -> Dict:
    """Build the safe-harbor payload.

    Args:
        ctx: DashboardContext with ``analyzer`` attached.
        today: Optional cutoff date. Defaults to ``datetime.now()``.
            Exposed for deterministic tests.

    Returns:
        Dict with:
          - ``jan1_nav_chf``: portfolio value on Jan 1 of the current
            year (0.0 if the portfolio didn't exist yet).
          - ``ytd_turnover_chf``: YTD buy+sell notional, summed across
            currencies at today's FX rate (an approximation — exact
            attribution would price each trade at its own trade-date
            rate).
          - ``ytd_turnover_ratio``: ``ytd_turnover_chf / jan1_nav_chf``,
            or ``None`` when the Jan-1 NAV is zero (no baseline).
          - ``threshold_ratio``: the safe-harbor 5x ceiling.
          - ``ytd_realized_chf``: signed realized P&L YTD, in CHF.
          - ``holding_period_violations``: list of closed sub-lots
            held for < 180 days, ready for JSON.
          - ``status``: ``'ok' | 'warning' | 'breach' |
            'no_baseline'``.
    """
    if today is None:
        today = datetime.now()
    year_start = datetime(today.year, 1, 1)
    analyzer = ctx.analyzer
    # Ensures market data + FX cache are populated for _calculate_value
    # and _native_to_base further down.
    analyzer._fetch_market_data()

    # NAV is valued on the first trading day of the year — Jan 1 is
    # usually a market holiday with no FX quote.
    nav_date = _first_trading_day_of_year(analyzer, today.year)
    jan1_nav_chf = _compute_jan1_nav(analyzer, nav_date)
    # Turnover uses the calendar-year window regardless: the tax rule
    # is calendar-based, and any weekend-1 trades still count.
    ytd_turnover_chf = _compute_ytd_turnover_chf(
        analyzer, year_start, today)
    ratio = _turnover_ratio(ytd_turnover_chf, jan1_nav_chf)

    rg = RealizedGains(analyzer.portfolio, analyzer._native_to_base)
    ytd_realized_chf = rg.ytd_realized_chf(today)
    violations = _violations_records(
        rg.holding_period_violations(_holding_period_min_days))

    return {
        'jan1_nav_chf': jan1_nav_chf,
        'ytd_turnover_chf': ytd_turnover_chf,
        'ytd_turnover_ratio': ratio,
        'threshold_ratio': _turnover_threshold_ratio,
        'ytd_realized_chf': ytd_realized_chf,
        'holding_period_violations': violations,
        'status': _status(ratio, len(violations)),
    }


def _compute_jan1_nav(analyzer, nav_date: datetime) -> float:
    """Portfolio value on the year's baseline date (first trading day).

    Uses ``get_holdings_at_date`` + ``_calculate_value`` directly rather
    than walking the full daily NAV series so a single date query stays
    cheap. Returns 0.0 when the portfolio held nothing on that date (a
    new account opened later in the year).
    """
    holdings = analyzer.portfolio.get_holdings_at_date(nav_date)
    if not holdings:
        return 0.0
    return float(analyzer._calculate_value(
        holdings, nav_date, apply_dividend_correction=True))


def _compute_ytd_turnover_chf(
        analyzer, year_start: datetime, today: datetime) -> float:
    """Sum YTD buy+sell notional across currencies into CHF.

    Uses today's FX rate for the conversion — an approximation. Exact
    accounting would apply each trade's own date rate, which requires
    walking the transactions again inside this loop; not worth the
    complexity for a safe-harbor tracker where a few percent of FX
    drift is well below the 5x threshold's granularity.
    """
    per_currency = analyzer.portfolio.get_turnover(
        since=year_start, until=today)
    total_chf = 0.0
    for currency, native_amount in per_currency.items():
        total_chf += analyzer._native_to_base(
            native_amount, currency, today)
    return total_chf


def _turnover_ratio(
        ytd_turnover_chf: float,
        jan1_nav_chf: float) -> Optional[float]:
    """Turnover / Jan-1-NAV, or None when the baseline is zero."""
    if jan1_nav_chf <= 0.0:
        return None
    return ytd_turnover_chf / jan1_nav_chf


def _violations_records(df: pd.DataFrame) -> List[Dict]:
    """Serialise the holding-period-violations DataFrame for JSON.

    Empty frame -> empty list. Dates are emitted as ISO-8601 strings so
    the frontend can render them without a date-parser.
    """
    if df.empty:
        return []
    records = []
    for _, row in df.iterrows():
        records.append({
            'ticker': row['ticker'],
            'buy_date': row['buy_date'].isoformat(),
            'sell_date': row['sell_date'].isoformat(),
            'days_held': int(row['days_held']),
            'shares': float(row['shares']),
            'realized_chf': float(row['realized_chf']),
        })
    return records


def _status(
        ratio: Optional[float], violation_count: int) -> str:
    """Compute the traffic-light status.

    Any holding-period violation or a turnover ratio at/above the 5x
    threshold is a breach. Between the warning ratio and the threshold
    is 'warning'. A None ratio means the portfolio had no Jan-1
    baseline (opened this year) — 'no_baseline'.
    """
    if ratio is None:
        return 'no_baseline'
    if ratio >= _turnover_threshold_ratio or violation_count > 0:
        return 'breach'
    if ratio >= _turnover_warning_ratio:
        return 'warning'
    return 'ok'
