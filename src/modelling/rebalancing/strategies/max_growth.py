"""MaxGrowthStrategy: top-composite picks blended with holds.

High-risk growth: keeps a configurable fraction of NAV in
eligible current holdings (proportional to their current
weights), allocates the rest equal-weighted across the
top-N CandidateTickers by composite_score (excluding tickers
already held), and supports an optional CHF cash bucket.

Honours the no-bond Swiss-tax policy implicitly: the screener
upstream is expected to have applied SwissTaxCriterion. As a
defensive secondary filter, candidates whose metrics include a
'swiss_tax' key with value 0.0 are dropped. The default of 1.0
in the metrics.get('swiss_tax', 1.0) lookup is intentional and
documented: 'swiss_tax' is an optional fail signal, and absent
keys mean 'no fail signal recorded, keep the candidate'.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.modelling.rebalancing.action import cash_ticker
from src.modelling.rebalancing.strategies.base import Strategy
from src.modelling.rebalancing.strategies.equal_weight import (
    filter_eligible)
from src.shared.constraints import RebalanceConstraints


default_top_n = 5
default_holds_weight = 0.6
default_cash_weight = 0.0


class MaxGrowthStrategy(Strategy):
    """Blend eligible holds + top-N composite candidates."""

    name = 'max_growth'

    def __init__(
        self,
        top_n: int = default_top_n,
        holds_weight: float = default_holds_weight,
        cash_weight: float = default_cash_weight):
        if not 0.0 <= holds_weight <= 1.0:
            raise ValueError(
                f'holds_weight must be in [0, 1], '
                f'got {holds_weight}.')
        if not 0.0 <= cash_weight <= 1.0:
            raise ValueError(
                f'cash_weight must be in [0, 1], '
                f'got {cash_weight}.')
        if top_n <= 0:
            raise ValueError(
                f'top_n must be positive, got {top_n}.')
        self.top_n = top_n
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
        """Return blended target weights.

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
            RuntimeError: When candidates is None, or when
                neither eligible holds nor candidate picks
                remain (degenerate plan).
        """
        if candidates is None:
            raise RuntimeError(
                'MaxGrowthStrategy: candidates is required '
                '(screener-fed strategy).')
        eligible = filter_eligible(holdings_df, constraints)
        held = set(eligible['ticker'])
        filtered = [
            c for c in candidates
            if c.metrics.get('swiss_tax', 1.0) != 0.0]
        picks = sorted(
            (c for c in filtered if c.ticker not in held),
            key=lambda c: -c.composite_score)[:self.top_n]

        if eligible.empty and not picks:
            raise RuntimeError(
                'MaxGrowthStrategy: no eligible holds and no '
                'candidate picks; nothing to allocate.')
        equity_mass = 1.0 - self.cash_weight
        # Effective splits collapse when one side is missing
        if eligible.empty:
            holds_frac, picks_frac = 0.0, 1.0
        elif not picks:
            holds_frac, picks_frac = 1.0, 0.0
        else:
            holds_frac = self.holds_weight
            picks_frac = 1.0 - holds_frac

        weights: Dict[str, float] = {}
        if not eligible.empty:
            total_wpct = float(eligible['weight_pct'].sum())
            for _, row in eligible.iterrows():
                w = equity_mass * holds_frac * (
                    row['weight_pct'] / total_wpct)
                weights[row['ticker']] = w
        if picks:
            per_pick = (
                equity_mass * picks_frac / len(picks))
            for c in picks:
                weights[c.ticker] = per_pick
        if self.cash_weight > 0.0:
            weights[cash_ticker] = self.cash_weight

        total = sum(weights.values())
        return {t: w / total for t, w in weights.items()}
