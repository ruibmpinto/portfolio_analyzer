"""TransactionCostModel + IBKR Swiss-resident defaults.

Two entry points serve different callers:

- ``cost_chf`` and its vectorised sibling ``cost_chf_array`` return
  a single CHF cost for symmetric-roundtrip MC rebalance
  simulation. Used by ``MonteCarloEngine``. Because share counts
  are not available in the MC hot path, a bps-of-notional
  commission proxy is used instead of the per-share IBKR schedule.
- ``fees_breakdown`` returns a per-component CHF breakdown for the
  dashboard's forward-cost preview and breakeven-gap calculator.
  Uses the exact per-share IBKR schedule; requires ``shares``,
  ``trade_value_native``, and ``fx_rate_to_chf`` from the caller.

Both entry points consume ``src.shared.venue_taxes`` for
statutory taxes (UK stamp, FR/IT FTT, Swiss federal stamp) and
for the half-spread heuristic — one source of truth for both
broker and market-microstructure data.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.shared.venue_taxes import lookup_venue


bps_per_unit = 1e4
chf_currency = 'CHF'


@dataclass(frozen=True)
class TransactionCostModel:
    """IBKR US-Tiered trade cost model, defaults for Swiss residents.

    Per-share fields express IBKR's US Tiered top-tier retail
    schedule and drive ``fees_breakdown``. The two bps fields
    (``commission_bps``, ``commission_min_chf``) are a coarser
    fallback for ``cost_chf`` where share counts are unavailable;
    the default ``commission_bps=7`` matches the per-share default
    at ~USD 50/share so MC output stays close to reality.

    Attributes:
        commission_per_share_native: IBKR US-Tiered commission per
            share (USD 0.0035/share for top-tier retail).
        commission_min_native: Minimum per-order commission in
            native currency (USD 0.35).
        commission_cap_pct: Fraction of trade value at which
            commission is capped (0.01 = 1% cap).
        exchange_clearing_per_share_native: Per-share sum of
            NSCC/DTC clearing (0.0002), FINRA CAT (0.000003),
            and average exchange remove-liquidity fees (~0.003)
            for IBKR US trades.
        sec_fee_rate: SEC Section 31 fee as a fraction of sale
            value (0.0000206 for 2024-2025). Applied to sells only.
        finra_taf_per_share: FINRA Trading Activity Fee per share
            sold (0.000195). Applied to sells only.
        fx_cost_bps: IBKR FX conversion cost in bps of notional
            (0.2 bps). Consumed only by ``cost_chf`` (Monte Carlo);
            ``fees_breakdown`` assumes the sell and rebuy legs
            run in the same currency and does not charge FX.
        commission_bps: Bps-of-notional commission for
            ``cost_chf``'s share-agnostic path (7.0). Matches
            the per-share model at ~USD 50/share.
        commission_min_chf: Minimum CHF commission for
            ``cost_chf`` (1.5 CHF). Slightly conservative
            relative to USD 0.35 * FX; kept for MC baseline
            stability.
    """

    # IBKR US Tiered per-share schedule (used by fees_breakdown)
    commission_per_share_native: float = 0.0035
    commission_min_native: float = 0.35
    commission_cap_pct: float = 0.01
    exchange_clearing_per_share_native: float = 0.0034
    sec_fee_rate: float = 0.0000206
    finra_taf_per_share: float = 0.000195

    # FX conversion (Monte Carlo cost_chf only; fees_breakdown
    # assumes matched-currency legs and does not charge FX).
    fx_cost_bps: float = 2.0

    # Bps-of-notional fallback (used by cost_chf / cost_chf_array)
    commission_bps: float = 7.0
    commission_min_chf: float = 1.5

    def cost_chf(
        self,
        trade_value_chf: float,
        ticker: str,
        currency: str) -> float:
        """Symmetric CHF cost for MC rebalance simulation.

        Averages the venue's buy/sell taxes so the returned cost
        is direction-agnostic (MC does not distinguish sides on
        rebalance steps). Uses ``venue_taxes.half_spread_bps``
        for a venue-aware market-friction proxy.

        Args:
            trade_value_chf: Notional in CHF; sign is irrelevant.
            ticker: Yahoo-style ticker; drives the venue lookup.
            currency: ISO-4217 trade currency. ``'CHF'``
                suppresses the FX cost component.

        Returns:
            Total CHF cost = commission + spread + venue tax
            + FX cost.
        """
        notional = abs(float(trade_value_chf))
        commission = max(
            notional * self.commission_bps / bps_per_unit,
            self.commission_min_chf)
        venue = lookup_venue(ticker)
        # MC has no side; use symmetric average per roundtrip step
        avg_tax_bps = (venue.buy_bps + venue.sell_bps) / 2.0
        tax = notional * avg_tax_bps / bps_per_unit
        # Half-spread is a market cost; retained here so MC still
        # captures round-trip friction end-to-end.
        spread = notional * venue.half_spread_bps / bps_per_unit
        fx = 0.0
        if currency != chf_currency:
            fx = notional * self.fx_cost_bps / bps_per_unit
        return commission + spread + tax + fx

    def cost_chf_array(
        self,
        trade_values_chf,
        ticker: str,
        currency: str):
        """Vectorised cost calculation across multiple paths.

        Equivalent to applying ``cost_chf`` element-wise to
        ``trade_values_chf``, but executes in a single NumPy
        expression to avoid Python-level looping in
        performance-critical Monte Carlo paths.

        Args:
            trade_values_chf: 1D numpy array of trade
                notionals in CHF. Sign is irrelevant.
            ticker: Exchange symbol (used to detect Swiss
                listings).
            currency: ISO-4217 listing currency.

        Returns:
            1D numpy array of CHF costs, same shape as
            ``trade_values_chf``.
        """
        notional = np.abs(np.asarray(
            trade_values_chf, dtype=float))
        commission = np.maximum(
            notional * self.commission_bps / bps_per_unit,
            self.commission_min_chf)
        venue = lookup_venue(ticker)
        avg_tax_bps = (venue.buy_bps + venue.sell_bps) / 2.0
        tax = notional * avg_tax_bps / bps_per_unit
        spread = notional * venue.half_spread_bps / bps_per_unit
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

        Every component is modelled at its natural granularity
        (per-share for broker fees, per-value for taxes) then
        converted to CHF via ``fx_rate_to_chf``. ``side``
        determines which venue-tax leg fires: UK stamp and FR/IT
        FTT tax the buy leg only; Swiss federal stamp taxes both.

        Args:
            shares: Positive share count; fractional shares OK.
            trade_value_native: ``shares * price_native``,
                positive, in the trade's native currency.
            currency: ISO-4217 trade currency.
            ticker: Yahoo-style ticker; drives the venue lookup.
            side: ``'buy'`` or ``'sell'``. Any other value raises
                ``ValueError`` — venue taxes are asymmetric, so a
                silent default would produce wrong numbers.
            fx_rate_to_chf: Native-to-CHF conversion rate. Pass
                ``1.0`` when the trade is already CHF-denominated.

        Returns:
            Dict with keys ``commission_chf``,
            ``exchange_clearing_chf``, ``regulatory_chf``,
            ``venue_tax_chf``, ``total_chf``, ``total_bps``.
            The four component fields sum to ``total_chf``.
            ``total_bps`` is 0.0 when ``trade_value_native`` is 0
            (mathematically correct, not a silent fallback). No
            FX component: buy and sell legs are assumed to run
            in the same currency.

        Raises:
            ValueError: If ``side`` is not ``'buy'`` or ``'sell'``.
        """
        if side not in ('buy', 'sell'):
            raise ValueError(
                f"side must be 'buy' or 'sell', got {side!r}.")
        notional = float(trade_value_native)
        shares_f = float(shares)

        # IBKR commission: max(per-share, minimum), capped at
        # a percentage of trade value
        commission_native = max(
            self.commission_per_share_native * shares_f,
            self.commission_min_native)
        commission_native = min(
            commission_native,
            self.commission_cap_pct * notional)

        # Exchange + clearing + CAT: per-share, side-agnostic
        exchange_clearing_native = (
            self.exchange_clearing_per_share_native * shares_f)

        # Venue determines both the statutory tax leg and whether
        # US regulatory fees (SEC, FINRA) apply.
        venue = lookup_venue(ticker)

        # SEC Section 31 and FINRA TAF fire on US-listed sells only
        if side == 'sell' and venue.is_us:
            regulatory_native = (
                self.sec_fee_rate * notional
                + self.finra_taf_per_share * shares_f)
        else:
            regulatory_native = 0.0

        # Statutory venue tax: asymmetric per side (buy_bps for
        # UK/FR/IT, both sides for Switzerland).
        if side == 'buy':
            venue_tax_bps = venue.buy_bps
        else:
            venue_tax_bps = venue.sell_bps
        venue_tax_native = notional * venue_tax_bps / bps_per_unit

        # No FX component: the breakeven / forward-cost consumers
        # assume the sell and rebuy legs run in the same currency
        # (holding the native-currency cash between them). IBKR
        # bills FX conversions as separate Forex trades — those are
        # captured by the retrospective cumulative-fee panel via
        # ``Transaction.auto_fx_fee``, not here.

        # Every native component converts to CHF via the same rate
        commission_chf = commission_native * fx_rate_to_chf
        exchange_clearing_chf = (
            exchange_clearing_native * fx_rate_to_chf)
        regulatory_chf = regulatory_native * fx_rate_to_chf
        venue_tax_chf = venue_tax_native * fx_rate_to_chf
        total_chf = (
            commission_chf + exchange_clearing_chf
            + regulatory_chf + venue_tax_chf)

        # bps of the CHF trade value; guard against zero notional
        trade_value_chf = notional * fx_rate_to_chf
        if trade_value_chf > 0:
            total_bps = total_chf / trade_value_chf * bps_per_unit
        else:
            total_bps = 0.0

        return {
            'commission_chf': commission_chf,
            'exchange_clearing_chf': exchange_clearing_chf,
            'regulatory_chf': regulatory_chf,
            'venue_tax_chf': venue_tax_chf,
            'total_chf': total_chf,
            'total_bps': total_bps,
        }


def ibkr_default_cost_model() -> TransactionCostModel:
    """IBKR defaults for a Swiss tax-resident retail investor.

    Numbers reflect IBKR US Tiered top-retail-tier commissions.
    European IBKR trades follow a bps-of-value schedule which the
    current dataclass does not yet model; a sibling model will
    land alongside when that broker/venue combination is added.

    Returns:
        TransactionCostModel with dataclass field defaults.
    """
    return TransactionCostModel()
