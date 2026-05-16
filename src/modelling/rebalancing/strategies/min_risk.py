"""MinRiskStrategy: low-vol picks + broad equity ETF + cash.

Anchored by a 40% CHF cash reserve (per
feedback_swiss_tax_no_bonds.md: defensive risk lever is cash,
never bonds), with the rest split across a global equity index
(default CSSPX.SW), the N lowest-drawdown screener candidates,
and eligible current holdings. Defaults port the legacy S3
scenario from the deleted construct_scenarios.py.

Honours the no-bond Swiss-tax policy implicitly: the screener
upstream is expected to have applied SwissTaxCriterion. As a
defensive secondary filter, candidates whose metrics include a
'swiss_tax' key with value 0.0 are dropped. The default of 1.0
when the key is absent is the documented exception to the
no-silent-defaults rule.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.modelling.rebalancing.action import cash_ticker
from src.modelling.rebalancing.strategies.base import Strategy
from src.modelling.rebalancing.strategies.equal_weight import (
    filter_eligible)
from src.shared.constraints import RebalanceConstraints


default_broad_etf = 'CSSPX.SW'
default_etf_weight = 0.12
default_n_lowvol = 3
default_lowvol_weight = 0.12
default_holds_weight = 0.36
default_cash_weight = 0.40
drawdown_floor = 1e-4


class MinRiskStrategy(Strategy):
    """Holds + broad equity ETF + low-DD picks + cash."""

    name = 'min_risk'

    def __init__(
        self,
        broad_etf: str = default_broad_etf,
        etf_weight: float = default_etf_weight,
        n_lowvol: int = default_n_lowvol,
        lowvol_weight: float = default_lowvol_weight,
        holds_weight: float = default_holds_weight,
        cash_weight: float = default_cash_weight):
        for label, val in (
            ('etf_weight', etf_weight),
            ('lowvol_weight', lowvol_weight),
            ('holds_weight', holds_weight),
            ('cash_weight', cash_weight)):
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f'{label} must be in [0, 1], got {val}.')
        total = (
            etf_weight + lowvol_weight + holds_weight
            + cash_weight)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f'MinRiskStrategy: etf_weight + '
                f'lowvol_weight + holds_weight + cash_weight '
                f'must sum to 1.0, got {total}.')
        if n_lowvol <= 0:
            raise ValueError(
                f'n_lowvol must be positive, got {n_lowvol}.')
        self.broad_etf = broad_etf
        self.etf_weight = etf_weight
        self.n_lowvol = n_lowvol
        self.lowvol_weight = lowvol_weight
        self.holds_weight = holds_weight
        self.cash_weight = cash_weight

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[
            str, float]:
        """Return defensively-allocated target weights.

        Args:
            holdings_df: Snapshot DataFrame; must include
                'ticker', 'category', 'weight_pct'.
            prices: Ignored.
            constraints: Reads excluded_categories +
                min_position_pct via filter_eligible.
            candidates: List of CandidateTicker, required.
            risk_free_rate: Ignored.

        Returns:
            Dict {ticker: fraction} summing to 1.0. Includes a
            'CASH' entry equal to ``cash_weight`` when
            ``cash_weight > 0``.

        Raises:
            RuntimeError: When candidates is None.
            KeyError: When any candidate lacks
                'max_drawdown' in its metrics.
        """
        if candidates is None:
            raise RuntimeError(
                'MinRiskStrategy: candidates is required '
                '(screener-fed strategy).')

        eligible = filter_eligible(holdings_df, constraints)
        held = set(eligible['ticker'])

        for c in candidates:
            if 'max_drawdown' not in c.metrics:
                raise KeyError(
                    f'MinRiskStrategy: candidate '
                    f'{c.ticker!r} missing metric '
                    f"'max_drawdown'.")

        filtered = [
            c for c in candidates
            if c.metrics.get('swiss_tax', 1.0) != 0.0
            and c.ticker not in held
            and c.ticker != self.broad_etf]
        picks = sorted(
            filtered,
            key=lambda c: abs(c.metrics['max_drawdown']))[
                :self.n_lowvol]
        if not picks and self.lowvol_weight > 0.0:
            raise RuntimeError(
                f'MinRiskStrategy: no candidate picks remain '
                f'after filtering (held/swiss_tax/broad_etf '
                f'exclusion), but lowvol_weight='
                f'{self.lowvol_weight} requires at least one. '
                f'Provide more candidates or construct with '
                f'lowvol_weight=0.')

        weights: Dict[str, float] = {}
        if not eligible.empty:
            total_wpct = float(eligible['weight_pct'].sum())
            for _, row in eligible.iterrows():
                weights[row['ticker']] = self.holds_weight * (
                    row['weight_pct'] / total_wpct)
        # Broad ETF: standalone entry if not already held, else
        # fold its weight into the existing hold to keep the key
        # unique.
        if self.broad_etf not in held:
            weights[self.broad_etf] = self.etf_weight
        else:
            if self.broad_etf in weights:
                weights[self.broad_etf] += self.etf_weight
        if picks:
            inv = {
                c.ticker: 1.0 / max(
                    abs(c.metrics['max_drawdown']),
                    drawdown_floor)
                for c in picks}
            inv_sum = sum(inv.values())
            for c in picks:
                weights[c.ticker] = self.lowvol_weight * (
                    inv[c.ticker] / inv_sum)
        if self.cash_weight > 0.0:
            weights[cash_ticker] = self.cash_weight

        total = sum(weights.values())
        if total <= 0:
            raise RuntimeError(
                'MinRiskStrategy: all weights sum to zero; '
                'check constructor parameters and inputs.')
        return {t: w / total for t, w in weights.items()}
