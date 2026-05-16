"""Cumulative-fee aggregation across the lifetime of a portfolio.

Walks ``portfolio.transactions`` and sums every transaction's
``fee`` and ``auto_fx_fee`` per currency, then translates each
currency bucket into CHF using the most recent FX close fetched
through the provided DataProvider.

Functions
---------
cumulative_fees
    Return per-currency fee totals plus a CHF roll-up.
"""

from datetime import datetime, timedelta
from typing import Dict


fx_history_days = 30


def cumulative_fees(portfolio, data_provider, base_currency='CHF'):
    """Sum lifetime transaction fees grouped by trade currency.

    Args:
        portfolio: ``analysis.core.portfolio.Portfolio`` instance
            providing ``portfolio.transactions``.
        data_provider: ``DataProvider`` used to look up CCY=X FX
            price history when converting non-base currencies to
            ``base_currency``.
        base_currency: ISO-4217 currency for the total roll-up.

    Returns:
        Dict with keys:
            by_currency : Dict[str, float]
                Sum of ``fee + auto_fx_fee`` per trade currency.
            total_chf : float
                Same totals converted to ``base_currency`` and
                summed.
            fx_rates_used : Dict[str, float]
                Per-currency FX rate that was applied (1.0 when
                the currency already equals ``base_currency``).
    """
    by_currency: Dict[str, float] = {}
    for t in portfolio.transactions:
        fee = float(t.fee) + float(t.auto_fx_fee)
        if fee == 0.0:
            continue
        by_currency[t.currency] = (
            by_currency.get(t.currency, 0.0) + fee)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert each currency bucket to base_currency.
    fx_rates_used: Dict[str, float] = {}
    total_chf = 0.0
    for currency, amount in by_currency.items():
        rate = _fx_rate(
            data_provider, currency, base_currency)
        fx_rates_used[currency] = rate
        total_chf += amount * rate
    return {
        'by_currency': by_currency,
        'total_chf': total_chf,
        'fx_rates_used': fx_rates_used,
    }


def _fx_rate(data_provider, from_ccy, to_ccy):
    """Return the latest FX close converting `from_ccy` -> `to_ccy`."""
    if from_ccy == to_ccy:
        return 1.0
    pair = f'{from_ccy}{to_ccy}=X'
    end = datetime.now()
    start = end - timedelta(days=fx_history_days)
    try:
        series = data_provider.get_price_history(
            pair, start_date=start, end_date=end)
    except Exception as exc:
        raise RuntimeError(
            f'cumulative_fees: no FX history for {pair}; '
            f'cannot convert {from_ccy} -> {to_ccy} ({exc}).')
    if series is None or len(series) == 0:
        raise RuntimeError(
            f'cumulative_fees: no FX history for {pair}; '
            f'cannot convert {from_ccy} -> {to_ccy}.')
    return float(series.iloc[-1])
