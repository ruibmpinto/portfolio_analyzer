"""TransactionCostModel + IBKR/Degiro Swiss-resident defaults.

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
  Uses the exact per-(broker, venue, currency) schedule from
  ``venue_taxes``.

Both entry points consume ``src.shared.venue_taxes`` for the
per-venue statutory tax, broker commission constants, US
regulatory-fee flag, and half-spread heuristic - one source of
truth for every venue-specific number.
"""

from dataclasses import dataclass
from typing import Dict, Optional

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
    (SEC + FINRA are fixed statutory constants), the MC-only
    FX/commission-bps fallback used by ``cost_chf``, and the
    ``broker`` selector used to route schedule lookups.

    Attributes:
        broker: Which broker's schedule to consult. ``'ibkr'`` or
            ``'degiro'``. Every call ``fees_breakdown``/``cost_chf``
            makes into ``lookup_schedule`` uses this value.
        sec_fee_rate: SEC Section 31 fee as a fraction of sale
            value (0.0000206 for 2024-2025). Applied on the sell
            leg only when the row has ``charges_us_regulatory``.
            Set 0 for Degiro (regulatory bundled in handling fee).
        finra_taf_per_share: FINRA Trading Activity Fee per share
            sold (0.000195). Same gating as ``sec_fee_rate``.
        fx_cost_bps: FX conversion cost in bps of notional.
            Consumed only by ``cost_chf`` (Monte Carlo);
            ``fees_breakdown`` assumes matched-currency legs and
            does not charge FX. IBKR: 2.0 bps. Degiro: 25.0 bps
            (0.25% AutoFX).
        commission_bps: Bps-of-notional commission for
            ``cost_chf``'s share-agnostic MC path. Matches the
            per-share IBKR model at ~USD 50/share (7 bps).
        commission_min_chf: Minimum CHF commission for
            ``cost_chf`` (1.5 CHF).
    """

    broker: str = 'ibkr'
    sec_fee_rate: float = 0.0000206
    finra_taf_per_share: float = 0.000195
    fx_cost_bps: float = 2.0
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
            when ``ticker`` is an IBKR NTF fund (Degiro has no
            NTF-parity, so callers should not exercise that path
            when ``broker='degiro'``).
        """
        notional = np.abs(np.asarray(
            trade_values_chf, dtype=float))
        if is_ntf_mutual_fund(ticker):
            return np.zeros_like(notional)
        commission = np.maximum(
            notional * self.commission_bps / bps_per_unit,
            self.commission_min_chf)
        sched = lookup_schedule(ticker, currency, self.broker)
        avg_tax_bps = (
            sched.tax_buy_bps + sched.tax_sell_bps) / 2.0
        tax = notional * avg_tax_bps / bps_per_unit
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
        fx_rate_to_chf: float,
        commission_fx_rate_to_chf: Optional[float] = None
    ) -> Dict[str, float]:
        """Per-component CHF fee breakdown for a single trade.

        One uniform formula, one schedule lookup. Every venue-
        specific constant lives on the ``VenueSchedule`` row; this
        method just composes commission, broker per-share extras,
        US regulatory (sell-side), and statutory venue tax. Buy
        and sell legs are assumed to run in the same trade
        currency (no FX component).

        Commission may be billed in a currency other than the
        trade currency (Degiro US: fee in EUR, trade in USD). When
        the schedule row has ``commission_currency`` set,
        ``commission_fx_rate_to_chf`` is required and used to
        convert the commission (plus broker per-share extras) to
        CHF. Regulatory + venue tax always convert via
        ``fx_rate_to_chf`` because they are levied in the trade
        currency.

        Args:
            shares: Positive share count; fractional shares OK.
            trade_value_native: ``shares * price_native``, positive,
                in the trade's native currency.
            currency: ISO-4217 trade currency.
            ticker: Yahoo-style ticker.
            side: ``'buy'`` or ``'sell'``. Any other value raises
                ``ValueError`` - venue taxes are asymmetric.
            fx_rate_to_chf: Trade-currency-to-CHF conversion rate.
                Pass ``1.0`` when the trade is CHF-denominated.
            commission_fx_rate_to_chf: Rate for converting the
                commission from its billing currency to CHF. Used
                only when the schedule row has
                ``commission_currency`` set. When left ``None``,
                falls back to ``fx_rate_to_chf`` - which is correct
                for IBKR (billed in trade currency) but wrong for
                Degiro cross-currency trades.

        Returns:
            Dict with keys ``commission_chf``, ``broker_extras_chf``,
            ``regulatory_chf``, ``venue_tax_chf``, ``total_chf``,
            ``total_bps``. The four components sum to ``total_chf``.
            NTF mutual funds return all zeros.

        Raises:
            ValueError: If ``side`` is not ``'buy'`` or ``'sell'``.
            RuntimeError: If no schedule row matches
                ``(broker, ticker, currency)`` - data-mapping error
                surfaced loudly (no silent default).
        """
        if side not in ('buy', 'sell'):
            raise ValueError(
                f"side must be 'buy' or 'sell', got {side!r}.")
        notional = float(trade_value_native)
        shares_f = float(shares)

        if is_ntf_mutual_fund(ticker):
            return self._zero_breakdown(notional * fx_rate_to_chf)

        sched = lookup_schedule(ticker, currency, self.broker)

        # Commission: flat + bps*value + per-share*shares. Cap
        # applies FIRST, then min. Order matters: on very small
        # trades the cap can fall below the min, and the broker
        # bills the min in that case (IBKR SMICHA.SW at CHF 267 ->
        # min; Degiro cap=1.0 so cap effectively inactive).
        # Note: the commission may be in EUR while the trade is in
        # USD (Degiro). We keep the whole computation in the
        # commission's own currency until final CHF conversion.
        commission_native = (
            sched.commission_flat_native
            + sched.commission_bps * notional / bps_per_unit
            + sched.commission_per_share_native * shares_f)
        commission_native = min(
            commission_native,
            sched.commission_cap_pct * notional)
        commission_native = max(
            commission_native, sched.commission_min_native)

        broker_extras_native = (
            sched.broker_extras_per_share_native * shares_f)

        if side == 'sell' and sched.charges_us_regulatory:
            regulatory_native = (
                self.sec_fee_rate * notional
                + self.finra_taf_per_share * shares_f)
        else:
            regulatory_native = 0.0

        tax_bps = (
            sched.tax_buy_bps if side == 'buy'
            else sched.tax_sell_bps)
        venue_tax_native = notional * tax_bps / bps_per_unit

        # Commission (and broker per-share extras, which are also
        # billed by the broker) convert via commission FX rate;
        # regulatory and venue tax always via trade FX rate.
        if sched.commission_currency is None:
            commission_rate = fx_rate_to_chf
        else:
            if commission_fx_rate_to_chf is None:
                raise RuntimeError(
                    f'Schedule row for {ticker!r} bills commission '
                    f'in {sched.commission_currency!r}, but no '
                    f'commission_fx_rate_to_chf was supplied.')
            commission_rate = commission_fx_rate_to_chf
        commission_chf = commission_native * commission_rate
        broker_extras_chf = broker_extras_native * commission_rate
        regulatory_chf = regulatory_native * fx_rate_to_chf
        venue_tax_chf = venue_tax_native * fx_rate_to_chf
        total_chf = (
            commission_chf + broker_extras_chf
            + regulatory_chf + venue_tax_chf)

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
    every venue documented in ``venue_taxes._ibkr_schedule()``.
    """
    return TransactionCostModel(broker='ibkr')


def degiro_default_cost_model() -> TransactionCostModel:
    """Degiro Swiss-resident tariff defaults (2026-01-01).

    Regulatory constants zero because Degiro bundles US SEC/FINRA
    passthrough into its EUR 1.00 handling fee (which is already
    baked into ``commission_flat_native`` on each schedule row).
    FX cost 25 bps = 0.25% AutoFX.
    """
    return TransactionCostModel(
        broker='degiro',
        sec_fee_rate=0.0,
        finra_taf_per_share=0.0,
        fx_cost_bps=25.0)
