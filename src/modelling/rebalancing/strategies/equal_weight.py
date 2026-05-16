"""
EqualWeightStrategy: 1/N across eligible current holdings.

Eligibility filters: drop holdings whose category is in
constraints.excluded_categories, then drop dust holdings whose
weight_pct is below constraints.min_position_pct. Ignores
prices, candidates, and risk_free_rate -- pure portfolio-
snapshot strategy. Useful as a sanity baseline.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.modelling.rebalancing.strategies.base import Strategy
from src.shared.constraints import RebalanceConstraints


class EqualWeightStrategy(Strategy):
    """
    Uniform weights across eligible current holdings.
    """

    name = 'equal_weight'

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Return uniform target weights across eligible holdings.

        Args:
            holdings_df: Snapshot from
                analyzer.get_holdings_snapshot(); must include
                'ticker', 'category', 'weight_pct'.
            prices: Ignored.
            constraints: Reads excluded_categories and
                min_position_pct.
            candidates: Ignored.
            risk_free_rate: Ignored.

        Returns:
            Dict {ticker: 1/N} where N is the number of
            eligible holdings.

        Raises:
            RuntimeError: When no holdings remain after
                applying the eligibility filter.
        """
        eligible = filter_eligible(holdings_df, constraints)
        n = len(eligible)
        if n == 0:
            raise RuntimeError(
                'EqualWeightStrategy: no eligible holdings '
                'after applying constraints.')
        weight = 1.0 / n
        return {ticker: weight for ticker in eligible['ticker']}


def filter_eligible(
    holdings_df: pd.DataFrame,
    constraints: RebalanceConstraints) -> pd.DataFrame:
    """
    Drop excluded categories and dust positions.

    Args:
        holdings_df: Snapshot DataFrame; must include
            'category', 'weight_pct', 'ticker' columns.
        constraints: RebalanceConstraints; reads
            excluded_categories and min_position_pct.

    Returns:
        Filtered holdings DataFrame, same columns.

    Raises:
        KeyError: When holdings_df is missing 'category',
            'weight_pct', or 'ticker'.
    """
    for col in ('category', 'weight_pct', 'ticker'):
        if col not in holdings_df.columns:
            raise KeyError(
                f'filter_eligible: holdings_df missing '
                f'column {col!r}.')
    df = holdings_df[
        ~holdings_df['category'].isin(
            constraints.excluded_categories)]
    df = df[df['weight_pct'] >= constraints.min_position_pct]
    return df
