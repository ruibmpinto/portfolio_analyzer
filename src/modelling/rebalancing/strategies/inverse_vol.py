"""
InverseVolStrategy: weight proportional to 1/sigma_i.

Risk-balanced without estimating a covariance matrix; tends
to over-weight bond-like assets and under-weight equities.
Operates on current holdings only (no screener input).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.modelling.rebalancing.strategies.base import Strategy
from src.modelling.rebalancing.strategies.equal_weight import (
    filter_eligible)
from src.shared.constraints import RebalanceConstraints


trading_days_per_year = 252
volatility_floor = 1e-10


class InverseVolStrategy(Strategy):
    """
    w_i proportional to 1/sigma_i, normalised to sum to 1.

    sigma_i is the annualised standard deviation of the
    ticker's daily returns over the available price-history
    window.
    """

    name = 'inverse_vol'

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Return inverse-volatility target weights.

        Args:
            holdings_df: Snapshot DataFrame; must include
                'ticker', 'category', 'weight_pct'.
            prices: Wide DataFrame; must have one column per
                eligible ticker.
            constraints: Reads excluded_categories and
                min_position_pct via filter_eligible.
            candidates: Ignored.
            risk_free_rate: Ignored.

        Returns:
            Dict {ticker: fraction} summing to 1.0.

        Raises:
            RuntimeError: When no holdings are eligible, or
                when an eligible ticker is missing from
                `prices`.
        """
        eligible = filter_eligible(holdings_df, constraints)
        tickers = list(eligible['ticker'])
        if len(tickers) == 0:
            raise RuntimeError(
                'InverseVolStrategy: no eligible holdings '
                'after applying constraints.')

        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise RuntimeError(
                f'InverseVolStrategy: prices missing column(s) '
                f'for held ticker(s) {sorted(missing)}.')

        returns = prices[tickers].pct_change().dropna()
        annual_vols = (
            returns.std() * np.sqrt(trading_days_per_year))
        # Numerical floor so a flat-line ticker doesn't blow up
        annual_vols = annual_vols.where(
            annual_vols > volatility_floor,
            other=volatility_floor)
        inv = 1.0 / annual_vols
        weights = inv / inv.sum()
        return {t: float(weights[t]) for t in tickers}
