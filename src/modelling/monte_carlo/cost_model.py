"""TransactionCostModel + IBKR Swiss-resident defaults.

Implements spec §5. The model returns a CHF cost for a trade
of the given trade_value_chf, ticker (used to detect Swiss
listings via the .SW suffix), and currency (used to detect
non-CHF FX cost). Used by MonteCarloEngine on every rebalance
step when the engine is configured with a cost_model.
"""

from dataclasses import dataclass

import numpy as np


bps_per_unit = 1e4
swiss_suffix = '.SW'
chf_currency = 'CHF'


@dataclass(frozen=True)
class TransactionCostModel:
    """IBKR Swiss-resident trade cost model.

    Attributes:
        commission_bps: Per-trade commission in basis points.
        commission_min_chf: Minimum per-trade commission.
        spread_bps: Half-spread cost in basis points.
        stamp_duty_bps_swiss: Swiss federal stamp on CH
            securities (ticker ending in '.SW').
        stamp_duty_bps_foreign: Swiss federal stamp on
            foreign securities.
        fx_cost_bps: FX conversion cost in basis points when
            the trade currency is not CHF.
    """

    commission_bps: float = 7.0
    commission_min_chf: float = 1.5
    spread_bps: float = 5.0
    stamp_duty_bps_swiss: float = 7.5
    stamp_duty_bps_foreign: float = 15.0
    fx_cost_bps: float = 2.0

    def cost_chf(
        self,
        trade_value_chf: float,
        ticker: str,
        currency: str) -> float:
        """CHF cost of a trade.

        Args:
            trade_value_chf: Notional in CHF. Sign is
                irrelevant — cost is applied symmetrically to
                buys and sells via abs().
            ticker: Exchange symbol; used to detect Swiss
                listings via the '.SW' suffix.
            currency: ISO-4217 listing currency. 'CHF'
                suppresses the FX cost component.

        Returns:
            Total CHF cost = commission + spread + stamp + fx.
        """
        notional = abs(float(trade_value_chf))
        commission = max(
            notional * self.commission_bps / bps_per_unit,
            self.commission_min_chf)
        spread = notional * self.spread_bps / bps_per_unit
        if ticker.endswith(swiss_suffix):
            stamp_bps = self.stamp_duty_bps_swiss
        else:
            stamp_bps = self.stamp_duty_bps_foreign
        stamp = notional * stamp_bps / bps_per_unit
        fx = 0.0
        if currency != chf_currency:
            fx = notional * self.fx_cost_bps / bps_per_unit
        return commission + spread + stamp + fx

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
        spread = notional * self.spread_bps / bps_per_unit
        if ticker.endswith(swiss_suffix):
            stamp_bps = self.stamp_duty_bps_swiss
        else:
            stamp_bps = self.stamp_duty_bps_foreign
        stamp = notional * stamp_bps / bps_per_unit
        if currency != chf_currency:
            fx = notional * self.fx_cost_bps / bps_per_unit
        else:
            fx = 0.0
        return commission + spread + stamp + fx


def ibkr_default_cost_model() -> TransactionCostModel:
    """IBKR defaults for a Swiss tax-resident retail investor.

    Sensible across the CHF 200-2000 trade size range.

    Returns:
        TransactionCostModel with the dataclass field defaults.
    """
    return TransactionCostModel()
