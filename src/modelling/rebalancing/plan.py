"""
Rebalance plan aggregator.

Implements Contract 5 of the refactor design spec.
RebalancePlan bundles a list of RebalancingAction with the
inputs that produced them (target weights, strategy name,
constraints, holdings snapshot) and exposes DataFrame views
+ summary aggregates for the Excel report.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.modelling.rebalancing.action import RebalancingAction
from src.shared.constraints import RebalanceConstraints


target_weights_sum_tol = 1e-3


@dataclass(frozen=True)
class RebalancePlan:
    """
    Bundle of rebalance actions + the inputs that produced them.

    Attributes:
        actions: Ordered list of RebalancingAction.
        target_weights: Strategy output, ticker -> fraction.
            Must sum to 1.0 within `target_weights_sum_tol`.
        strategy_name: Registry key of the strategy used (e.g.
            'mvo', 'equal_weight').
        constraints: The RebalanceConstraints applied.
        holdings_df: Snapshot input passed to the strategy
            (output of analyzer.get_holdings_snapshot).
        generated_at: Timestamp the plan was produced.
    """

    actions: List[RebalancingAction]
    target_weights: Dict[str, float]
    strategy_name: str
    constraints: RebalanceConstraints
    holdings_df: pd.DataFrame
    generated_at: datetime

    def __post_init__(self):
        total = sum(self.target_weights.values())
        if abs(total - 1.0) > target_weights_sum_tol:
            raise ValueError(
                f'target_weights must sum to 1.0 within '
                f'{target_weights_sum_tol}; got {total:.6f}.')
        action_tickers = {a.ticker for a in self.actions}
        weight_tickers = set(self.target_weights)
        if action_tickers != weight_tickers:
            raise ValueError(
                f'target_weights ticker set must match actions '
                f'ticker set; weights-only='
                f'{sorted(weight_tickers - action_tickers)}, '
                f'actions-only='
                f'{sorted(action_tickers - weight_tickers)}.')
        if len(self.actions) != len(action_tickers):
            counts = Counter(a.ticker for a in self.actions)
            duplicates = sorted(
                t for t, c in counts.items() if c > 1)
            raise ValueError(
                f'actions list contains duplicate ticker(s): '
                f'{duplicates}.')
        action_by_ticker = {a.ticker: a for a in self.actions}
        per_ticker_tol = target_weights_sum_tol * 100.0
        for ticker, frac in self.target_weights.items():
            expected_pct = frac * 100.0
            actual_pct = action_by_ticker[ticker].target_wt_pct
            if abs(actual_pct - expected_pct) > per_ticker_tol:
                raise ValueError(
                    f'Inconsistent target weight for '
                    f'{ticker!r}: target_weights says '
                    f'{expected_pct:.4f}%, action says '
                    f'{actual_pct:.4f}%.')

    @property
    def total_value_chf(self) -> float:
        """Sum of `value_chf` over the input holdings snapshot."""
        return float(self.holdings_df['value_chf'].sum())

    def to_dataframe(self) -> pd.DataFrame:
        """
        Action-table DataFrame ready for the Excel sheet.

        Returns:
            DataFrame with one row per action, columns: ticker,
            action, shares, est_cost_chf, current_wt_pct,
            target_wt_pct, note.
        """
        rows = []
        for a in self.actions:
            rows.append({
                'ticker': a.ticker,
                'action': a.action,
                'shares': a.shares,
                'est_cost_chf': a.est_cost_chf,
                'current_wt_pct': a.current_wt_pct,
                'target_wt_pct': a.target_wt_pct,
                'note': a.note,
            })
        return pd.DataFrame(rows)

    def target_weights_dataframe(self) -> pd.DataFrame:
        """
        Side-by-side current vs target weight table.

        Returns:
            DataFrame with one row per action's ticker, columns:
            ticker, current_wt_pct, target_wt_pct,
            deviation_pct (= target - current).
        """
        rows = []
        for a in self.actions:
            rows.append({
                'ticker': a.ticker,
                'current_wt_pct': a.current_wt_pct,
                'target_wt_pct': a.target_wt_pct,
                'deviation_pct':
                    a.target_wt_pct - a.current_wt_pct,
            })
        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, float]:
        """
        Aggregate plan stats for headline reporting.

        Returns:
            Dict with: n_buy, n_reduce, n_hold,
            total_buy_chf, total_reduce_chf, total_cash_chf,
            capital_remaining_chf,
            weight_drift_pre_pct, weight_drift_post_pct.
            'total_cash_chf' = sum of est_cost_chf over CASH
            rows (0.0 when no cash bucket exists).
            'capital_remaining_chf' = constraints.new_capital_chf
            minus sum of BUY est_cost_chf, floored at 0.
            'weight_drift_*_pct' = mean absolute deviation of
            current vs target weights, before and after acting.
        """
        n_buy = 0
        n_reduce = 0
        n_hold = 0
        total_buy = 0.0
        total_reduce = 0.0
        total_cash = 0.0
        drift_pre = 0.0
        for a in self.actions:
            drift_pre += abs(a.target_wt_pct - a.current_wt_pct)
            if a.action == 'BUY':
                n_buy += 1
                total_buy += a.est_cost_chf
            elif a.action == 'REDUCE':
                n_reduce += 1
                total_reduce += a.est_cost_chf
            elif a.action == 'HOLD':
                n_hold += 1
            elif a.action == 'CASH':
                total_cash += a.est_cost_chf
        n_actions = len(self.actions)
        drift_pre_mean = (
            drift_pre / n_actions if n_actions else 0.0)
        # Post drift assumes BUY/REDUCE close the gap exactly;
        # HOLD leaves the gap as-is. Mean over all actions.
        drift_post = 0.0
        for a in self.actions:
            if a.action == 'HOLD':
                drift_post += abs(
                    a.target_wt_pct - a.current_wt_pct)
        drift_post_mean = (
            drift_post / n_actions if n_actions else 0.0)
        capital_remaining = max(
            0.0, self.constraints.new_capital_chf - total_buy)
        return {
            'n_buy': n_buy,
            'n_reduce': n_reduce,
            'n_hold': n_hold,
            'total_buy_chf': total_buy,
            'total_reduce_chf': total_reduce,
            'total_cash_chf': total_cash,
            'capital_remaining_chf': capital_remaining,
            'weight_drift_pre_pct': drift_pre_mean,
            'weight_drift_post_pct': drift_post_mean,
        }
