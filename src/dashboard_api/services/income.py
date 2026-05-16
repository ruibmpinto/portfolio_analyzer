"""Lifetime dividend and withholding-tax aggregation.

Sums ``Transaction.price`` for every row whose operation is
``dividend`` or ``tax``, grouped by trade currency, and converts
each currency bucket to the base currency using the latest FX
close fetched through the provided DataProvider. Same shape as
``services.fees`` so the frontend can render them with one
component.

Functions
---------
cumulative_income
    Per-currency dividend or tax totals plus a base-currency
    roll-up.
"""

from datetime import datetime, timedelta
from typing import Dict


fx_history_days = 30


def cumulative_income(
        portfolio,
        operation: str,
        data_provider,
        base_currency: str = 'CHF') -> Dict:
    """Sum lifetime dividend or tax cash flows by currency.

    Args:
        portfolio: ``analysis.core.portfolio.Portfolio`` instance
            providing ``portfolio.transactions``.
        operation: Either ``'dividend'`` or ``'tax'``. Tax rows
            ship with a negative price (per the loaders'
            convention), so the result preserves the sign:
            dividends are positive, taxes are negative.
        data_provider: ``DataProvider`` used to look up CCY=X
            FX history when converting non-base currencies to
            ``base_currency``.
        base_currency: ISO-4217 currency for the total roll-up.

    Returns:
        Dict with keys:
            by_currency : Dict[str, float]
                Sum of cash flows per trade currency.
            total_chf : float
                Same totals converted to ``base_currency`` and
                summed.
            fx_rates_used : Dict[str, float]
                Per-currency FX rate applied (1.0 when the
                currency already equals ``base_currency``).
    """
    if operation not in ('dividend', 'tax'):
        raise ValueError(
            f'cumulative_income: operation must be '
            f'"dividend" or "tax"; got {operation!r}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Walk transactions; accumulate per currency
    by_currency: Dict[str, float] = {}
    for t in portfolio.transactions:
        if t.operation != operation:
            continue
        # `price` carries the cash amount per share (sign-aware:
        # negative for tax, positive for dividend); `amount` is 1
        cash = float(t.price) * float(t.amount)
        if cash == 0.0:
            continue
        by_currency[t.currency] = (
            by_currency.get(t.currency, 0.0) + cash)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert each currency bucket to base_currency
    fx_rates_used: Dict[str, float] = {}
    total_chf = 0.0
    for currency, amount in by_currency.items():
        rate = _fx_rate(data_provider, currency, base_currency)
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
            f'cumulative_income: no FX history for {pair}; '
            f'cannot convert {from_ccy} -> {to_ccy} ({exc}).')
    if series is None or len(series) == 0:
        raise RuntimeError(
            f'cumulative_income: no FX history for {pair}; '
            f'cannot convert {from_ccy} -> {to_ccy}.')
    return float(series.iloc[-1])
