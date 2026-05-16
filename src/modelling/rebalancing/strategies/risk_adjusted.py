"""RiskAdjustedStrategy: Sharpe-ranked picks, inverse-drawdown.

Ports the spirit of the legacy S2 scenario from the deleted
construct_scenarios.py: top-Sharpe candidates, inverse-vol
weighted with a per-name cap, blended with eligible current
holdings, plus a 10% cash reserve by default. Uses
1 / abs(max_drawdown) as the vol proxy since the Phase 3
criteria set has no plain volatility metric.

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


default_top_n = 5
default_holds_weight = 0.6
default_max_per_name = 0.15
default_cash_weight = 0.10
drawdown_floor = 1e-4
# Upper bound: convergence is typically < top_n (one pick pinned
# per pass). 50 is a defensive ceiling for pathological inputs.
cap_iter_max = 50


class RiskAdjustedStrategy(Strategy):
    """Top-Sharpe picks, inverse-drawdown weighted, capped."""

    name = 'risk_adjusted'

    def __init__(
        self,
        top_n: int = default_top_n,
        holds_weight: float = default_holds_weight,
        max_per_name: float = default_max_per_name,
        cash_weight: float = default_cash_weight):
        if not 0.0 <= holds_weight <= 1.0:
            raise ValueError(
                f'holds_weight must be in [0, 1], got '
                f'{holds_weight}.')
        if not 0.0 < max_per_name <= 1.0:
            raise ValueError(
                f'max_per_name must be in (0, 1], got '
                f'{max_per_name}.')
        if not 0.0 <= cash_weight <= 1.0:
            raise ValueError(
                f'cash_weight must be in [0, 1], got '
                f'{cash_weight}.')
        if top_n <= 0:
            raise ValueError(
                f'top_n must be positive, got {top_n}.')
        self.top_n = top_n
        self.holds_weight = holds_weight
        self.max_per_name = max_per_name
        self.cash_weight = cash_weight

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[
            str, float]:
        """Return Sharpe-ranked, capped target weights.

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
                remain.
            KeyError: When any candidate is missing 'sharpe'
                or 'max_drawdown' from its metrics.
        """
        if candidates is None:
            raise RuntimeError(
                'RiskAdjustedStrategy: candidates is required '
                '(screener-fed strategy).')

        eligible = filter_eligible(holdings_df, constraints)
        held = set(eligible['ticker'])

        for c in candidates:
            for key in ('sharpe', 'max_drawdown'):
                if key not in c.metrics:
                    raise KeyError(
                        f'RiskAdjustedStrategy: candidate '
                        f'{c.ticker!r} missing metric {key!r}.')

        filtered = [
            c for c in candidates
            if c.metrics.get('swiss_tax', 1.0) != 0.0
            and c.ticker not in held]
        picks = sorted(
            filtered, key=lambda c: -c.metrics['sharpe'])[
                :self.top_n]

        if eligible.empty and not picks:
            raise RuntimeError(
                'RiskAdjustedStrategy: no eligible holds and '
                'no candidate picks remain.')
        equity_mass = 1.0 - self.cash_weight
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
                weights[row['ticker']] = equity_mass * (
                    holds_frac * row['weight_pct'] / total_wpct)
        if picks:
            inv = {
                c.ticker: 1.0 / max(
                    abs(c.metrics['max_drawdown']),
                    drawdown_floor)
                for c in picks}
            inv_sum = sum(inv.values())
            for c in picks:
                weights[c.ticker] = equity_mass * picks_frac * (
                    inv[c.ticker] / inv_sum)
            self._enforce_cap(
                weights, {c.ticker for c in picks},
                hold_tickers=set(eligible['ticker'])
                if not eligible.empty else set())
        if self.cash_weight > 0.0:
            weights[cash_ticker] = self.cash_weight

        total = sum(weights.values())
        return {t: w / total for t, w in weights.items()}

    def _enforce_cap(
        self,
        weights: Dict[str, float],
        cap_tickers: set,
        hold_tickers: set) -> None:
        """Cap picks at max_per_name, redistribute the excess.

        Args:
            weights: Mutable dict of {ticker: fraction}; will be
                modified in place.
            cap_tickers: Set of pick tickers subject to the cap.
                Holds are excluded so the cap only applies to
                candidates.
            hold_tickers: Set of eligible-holding tickers. Holds
                bypass the cap and absorb leftover excess when
                every pick is already pinned at max_per_name.

        Raises:
            RuntimeError: When redistribution cannot converge
                in cap_iter_max iterations, or when every pick
                is already at the cap and there are no holds to
                absorb the excess.
        """
        for _ in range(cap_iter_max):
            over = {
                t: w - self.max_per_name
                for t, w in weights.items()
                if t in cap_tickers and w > self.max_per_name}
            if not over:
                return
            excess = sum(over.values())
            for t in over:
                weights[t] = self.max_per_name
            uncapped = [
                t for t in cap_tickers
                if weights[t] < self.max_per_name]
            if uncapped:
                share = excess / len(uncapped)
                for t in uncapped:
                    weights[t] += share
            elif hold_tickers:
                # Picks slice exceeds n_picks * cap; spill the
                # remainder into holds proportional to current
                # hold weights (holds bypass the cap by spec).
                hold_total = sum(
                    weights[t] for t in hold_tickers)
                if hold_total <= 0.0:
                    raise RuntimeError(
                        'RiskAdjustedStrategy: cap '
                        'redistribution has nowhere to put '
                        f'excess weight {excess:.4f}.')
                for t in hold_tickers:
                    weights[t] += excess * (
                        weights[t] / hold_total)
                return
            else:
                raise RuntimeError(
                    'RiskAdjustedStrategy: cap redistribution '
                    'has nowhere to put excess weight '
                    f'{excess:.4f}; all picks at cap and no '
                    'holds to absorb the remainder.')
        raise RuntimeError(
            'RiskAdjustedStrategy: cap redistribution did '
            f'not converge in {cap_iter_max} iterations.')
