"""TransactionCostModel + IBKR Swiss-resident defaults.

Two entry points serve different callers:

- ``cost_chf`` returns the symmetric CHF cost for MC rebalance
  simulation. Accepts either a scalar notional or a 1-D array
  (NumPy broadcasts uniformly); the shape of the return follows
  the shape of the input. Used by ``MonteCarloEngine``. Because
  share counts are not available in the MC hot path, a bps-of-
  notional commission proxy is used instead of the per-share
  IBKR schedule.
- ``fees_breakdown`` returns a per-component CHF breakdown for the
  dashboard's forward-cost preview and breakeven-gap calculator.
  Uses the exact per-(venue, currency) schedule from ``venue_taxes``.

Both entry points consume ``src.shared.venue_taxes`` for the
per-venue statutory tax, IBKR Tiered commission constants, US
regulatory-fee flag, and half-spread heuristic — one source of
truth for every venue-specific number.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.shared.venue_taxes import (
    is_ntf_mutual_fund, lookup_schedule)


bps_per_unit = 1e4
chf_currency = 'CHF'


@dataclass(frozen=True)
class TransactionCostModel:
    """Broker-side constants that don't fit the per-venue schedule.

    Every venue-specific fee constant lives on ``VenueSchedule`` in
    ``venue_taxes``. This class carries only the small set of
    numbers that are not venue-parameterised: US regulatory rates
    (SEC + FINRA are fixed statutory constants) and the MC-only
    FX/commission-bps fallback used by ``cost_chf``.

    Attributes:
        sec_fee_rate: SEC Section 31 fee as a fraction of sale
            value (0.0000206 for 2024-2025). Applied to sells on
            US-listed venues only.
        finra_taf_per_share: FINRA Trading Activity Fee per share
            sold (0.000195). Applied to US sells only.
        fx_cost_bps: IBKR FX conversion cost in bps of notional
            (0.2 bps). Consumed only by ``cost_chf`` (Monte Carlo);
            ``fees_breakdown`` assumes matched-currency legs and
            does not charge FX.
        commission_bps: Bps-of-notional commission for
            ``cost_chf``'s share-agnostic path (7.0). Matches the
            per-share model at ~USD 50/share.
        commission_min_chf: Minimum CHF commission for
            ``cost_chf`` (1.5 CHF).
    """

    # US regulatory (fixed statutory rates, not per-venue)
    sec_fee_rate: float = 0.0000206
    finra_taf_per_share: float = 0.000195

    # FX conversion (MC only; fees_breakdown assumes matched-currency legs)
    fx_cost_bps: float = 2.0

    # Bps-of-notional MC fallback (share counts unavailable in MC path)
    commission_bps: float = 7.0
    commission_min_chf: float = 1.5

    def cost_chf(
        self,
        trade_values_chf,
        ticker: str,
        currency: str):
        """Symmetric CHF cost for MC rebalance simulation.

        Accepts either a scalar notional or a 1-D array; NumPy
        broadcasts uniformly and the return shape follows the
        input. Averages the venue's buy/sell taxes so the returned
        cost is direction-agnostic (MC does not distinguish sides
        on rebalance steps). Half-spread and tax bps come from
        ``VenueSchedule``; commission uses the coarse bps-of-
        notional proxy since MC has no share counts.

        Args:
            trade_values_chf: Trade notional(s) in CHF, scalar or
                array-like. Sign is irrelevant.
            ticker: Yahoo-style ticker; drives the schedule lookup.
            currency: ISO-4217 trade currency. ``'CHF'`` suppresses
                the FX cost component.

        Returns:
            CHF cost with the same shape as the input. All-zero
            when ``ticker`` is an IBKR NTF fund.
        """
        notional = np.abs(np.asarray(
            trade_values_chf, dtype=float))
        # NTF mutual funds: zero broker fee across the vector
        if is_ntf_mutual_fund(ticker):
            return np.zeros_like(notional)
        commission = np.maximum(
            notional * self.commission_bps / bps_per_unit,
            self.commission_min_chf)
        sched = lookup_schedule(ticker, currency)
        # MC has no side; use symmetric average per roundtrip step
        avg_tax_bps = (
            sched.tax_buy_bps + sched.tax_sell_bps) / 2.0
        tax = notional * avg_tax_bps / bps_per_unit
        # Half-spread is a market cost; retained here so MC still
        # captures round-trip friction end-to-end.
        spread = notional * sched.half_spread_bps / bps_per_unit
        if currency != chf_currency:
            fx = notional * self.fx_cost_bps / bps_per_unit
        else:
            fx = 0.0
        return commission + spread + tax + fx

    def fees_breakdown(
        self,
        shares: float,
        trade_value_native: float,
        currency: str,
        ticker: str,
        side: str,
        fx_rate_to_chf: float) -> Dict[str, float]:
        """Per-component CHF fee breakdown for a single trade.

        One uniform formula, one schedule lookup. Every venue-specific
        constant lives on the ``VenueSchedule`` row; this method just
        composes commission, broker per-share extras, US regulatory
        (sell-side), and statutory venue tax. Buy and sell legs are
        assumed to run in the same currency (no FX component).

        Args:
            shares: Positive share count; fractional shares OK.
            trade_value_native: ``shares * price_native``, positive,
                in the trade's native currency.
            currency: ISO-4217 trade currency.
            ticker: Yahoo-style ticker.
            side: ``'buy'`` or ``'sell'``. Any other value raises
                ``ValueError`` — venue taxes are asymmetric.
            fx_rate_to_chf: Native-to-CHF conversion rate. Pass
                ``1.0`` when the trade is CHF-denominated.

        Returns:
            Dict with keys ``commission_chf``, ``broker_extras_chf``,
            ``regulatory_chf``, ``venue_tax_chf``, ``total_chf``,
            ``total_bps``. The four components sum to ``total_chf``.
            NTF mutual funds return all zeros.

        Raises:
            ValueError: If ``side`` is not ``'buy'`` or ``'sell'``.
            RuntimeError: If no schedule row matches
                ``(ticker, currency)`` — data-mapping error surfaced
                loudly (no silent default).
        """
        if side not in ('buy', 'sell'):
            raise ValueError(
                f"side must be 'buy' or 'sell', got {side!r}.")
        notional = float(trade_value_native)
        shares_f = float(shares)

        # NTF mutual funds pay zero broker fee at IBKR; short-circuit
        # before the schedule lookup so the caller doesn't need a
        # dedicated table row per fund.
        if is_ntf_mutual_fund(ticker):
            return self._zero_breakdown(notional * fx_rate_to_chf)

        sched = lookup_schedule(ticker, currency)

        # Commission: sum both rate types (each schedule row has
        # one nonzero — US uses per-share, non-US uses bps).
        # Cap applies FIRST, then min. Order matters: on very small
        # trades the cap can fall below the min, and IBKR bills the
        # min in that case (observed on SMICHA.SW at CHF 267 -> min).
        commission_native = (
            sched.commission_bps * notional / bps_per_unit
            + sched.commission_per_share_native * shares_f)
        commission_native = min(
            commission_native,
            sched.commission_cap_pct * notional)
        commission_native = max(
            commission_native, sched.commission_min_native)

        # Broker per-share extras (NSCC/DTC/CAT on US; zero elsewhere)
        broker_extras_native = (
            sched.broker_extras_per_share_native * shares_f)

        # SEC Section 31 + FINRA TAF fire on US sells only
        if side == 'sell' and sched.charges_us_regulatory:
            regulatory_native = (
                self.sec_fee_rate * notional
                + self.finra_taf_per_share * shares_f)
        else:
            regulatory_native = 0.0

        # Statutory venue tax: asymmetric per side
        tax_bps = (
            sched.tax_buy_bps if side == 'buy'
            else sched.tax_sell_bps)
        venue_tax_native = notional * tax_bps / bps_per_unit

        # Every native component converts to CHF via the same rate
        commission_chf = commission_native * fx_rate_to_chf
        broker_extras_chf = broker_extras_native * fx_rate_to_chf
        regulatory_chf = regulatory_native * fx_rate_to_chf
        venue_tax_chf = venue_tax_native * fx_rate_to_chf
        total_chf = (
            commission_chf + broker_extras_chf
            + regulatory_chf + venue_tax_chf)

        # bps of the CHF trade value; guard against zero notional
        trade_value_chf = notional * fx_rate_to_chf
        if trade_value_chf > 0:
            total_bps = total_chf / trade_value_chf * bps_per_unit
        else:
            total_bps = 0.0

        return {
            'commission_chf': commission_chf,
            'broker_extras_chf': broker_extras_chf,
            'regulatory_chf': regulatory_chf,
            'venue_tax_chf': venue_tax_chf,
            'total_chf': total_chf,
            'total_bps': total_bps,
        }

    @staticmethod
    def _zero_breakdown(trade_value_chf: float) -> Dict[str, float]:
        """All-zero breakdown for NTF mutual funds."""
        # trade_value_chf is unused today but reserved for the
        # future case where NTFs still incur a tiny custody fee.
        return {
            'commission_chf': 0.0,
            'broker_extras_chf': 0.0,
            'regulatory_chf': 0.0,
            'venue_tax_chf': 0.0,
            'total_chf': 0.0,
            'total_bps': 0.0,
        }


def ibkr_default_cost_model() -> TransactionCostModel:
    """IBKR defaults for a Swiss tax-resident retail investor.

    Numbers reflect IBKR Tiered top-retail-tier commissions across
    every venue documented in ``venue_taxes.schedule_table()``.

    Returns:
        TransactionCostModel with dataclass field defaults.
    """
    return TransactionCostModel()
