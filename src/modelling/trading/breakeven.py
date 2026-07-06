"""Sell-and-rebuy roundtrip breakeven-gap calculator.

Answers the question: how far must the price drop between
selling a lot and rebuying the same number of shares for the
roundtrip to at least recover its fees + market friction?

Consumed by:
- ``src/dashboard_api/routes/rules.py`` (Rules page examples)
- ``src/dashboard_api/routes/stops.py`` (per-holding stop-loss
  gap and suggested trigger price)

Math
----
Let ``f_s`` and ``f_b`` be the sell- and buy-leg fee fractions
of trade value (from ``TransactionCostModel.fees_breakdown``),
and ``h`` the venue's half-spread (from ``venue_taxes``). To
end up with at least the same number of shares after a
sell-and-rebuy at prices ``S`` and ``B``:

    S * (1 - f_s) >= B * (1 + f_b)
    B / S         <= (1 - f_s) / (1 + f_b)  ~=  1 - f_s - f_b

Rebuy price must therefore be below ``S * (1 - f_s - f_b - 2h)``.
The minimum gap in bps is

    breakeven_bps = f_s_bps + f_b_bps + 2 * half_spread_bps

The suggested stop-loss trigger is set at ``2 * breakeven`` — a
one-times-breakeven safety margin against noise-triggered stops,
per the strategy notes.
"""

from typing import Dict, Optional

from src.modelling.monte_carlo.cost_model import (
    TransactionCostModel, ibkr_default_cost_model)
from src.shared.venue_taxes import lookup_venue


def breakeven_gap(
    shares: float,
    price_native: float,
    currency: str,
    ticker: str,
    fx_rate_to_chf: float,
    cost_model: Optional[TransactionCostModel] = None
) -> Dict[str, float]:
    """Compute the roundtrip breakeven gap for a stop-and-rebuy.

    Args:
        shares: Positive share count for the hypothetical trade.
        price_native: Per-share price in the trade's native
            currency (usually the current market price).
        currency: ISO-4217 trade currency.
        ticker: Yahoo-style ticker; drives the venue tax and
            half-spread lookup.
        fx_rate_to_chf: Native-to-CHF conversion rate. Pass
            ``1.0`` when the trade is already CHF-denominated.
        cost_model: Broker cost model. Defaults to
            ``ibkr_default_cost_model()``; pass a Degiro-flavoured
            sibling when one lands.

    Returns:
        Dict with:
            - ``f_s_bps``: sell-leg fee fraction of trade value
              (bps), from the cost model.
            - ``f_b_bps``: buy-leg fee fraction (bps).
            - ``half_spread_bps``: venue half-spread heuristic
              (bps), from ``venue_taxes``.
            - ``breakeven_bps``: minimum sell-to-rebuy gap for
              the roundtrip to net zero on fees + spread
              (``f_s + f_b + 2 * half_spread``).
            - ``suggested_trigger_bps``: recommended stop-loss
              distance below current price
              (``2 * breakeven_bps``).
    """
    # Fall back to the shipped IBKR defaults when no model given
    if cost_model is None:
        cost_model = ibkr_default_cost_model()

    # Total notional in the trade's native currency
    trade_value_native = float(shares) * float(price_native)

    # Per-leg exact breakdown from the cost model
    sell = cost_model.fees_breakdown(
        shares=shares,
        trade_value_native=trade_value_native,
        currency=currency,
        ticker=ticker,
        side='sell',
        fx_rate_to_chf=fx_rate_to_chf)
    buy = cost_model.fees_breakdown(
        shares=shares,
        trade_value_native=trade_value_native,
        currency=currency,
        ticker=ticker,
        side='buy',
        fx_rate_to_chf=fx_rate_to_chf)

    # Half-spread is a market-friction cost, not a broker fee;
    # it comes from venue_taxes so all venue references stay
    # in one place.
    half_spread_bps = lookup_venue(ticker).half_spread_bps

    # Roundtrip breakeven: both legs' fees plus two half-spreads
    breakeven_bps = (
        sell['total_bps'] + buy['total_bps']
        + 2.0 * half_spread_bps)

    # 2x breakeven trigger gives one-times-breakeven margin
    # against a stop-out that reverses immediately
    suggested_trigger_bps = 2.0 * breakeven_bps

    return {
        'f_s_bps': sell['total_bps'],
        'f_b_bps': buy['total_bps'],
        'half_spread_bps': half_spread_bps,
        'breakeven_bps': breakeven_bps,
        'suggested_trigger_bps': suggested_trigger_bps,
    }
