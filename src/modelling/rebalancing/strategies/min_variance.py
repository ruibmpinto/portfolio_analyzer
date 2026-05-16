"""
MinVarianceStrategy: minimise w^T Sigma w subject to long-only,
per-name caps, weights sum to 1.

Operates on current holdings only. Uses scipy SLSQP.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.modelling.rebalancing.strategies.base import Strategy
from src.modelling.rebalancing.strategies.equal_weight import (
    filter_eligible)
from src.shared.constraints import RebalanceConstraints


trading_days_per_year = 252
covariance_min_periods = 30
slsqp_max_iter = 1000
slsqp_ftol = 1e-12


class MinVarianceStrategy(Strategy):
    """
    Minimise portfolio variance under long-only + per-name caps.

    Uses an annualised pairwise-complete sample covariance from
    daily returns. Per-name caps come from
    `constraints.max_weight_default` with category overrides
    in `constraints.category_caps`.
    """

    name = 'min_variance'

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Return min-variance target weights.

        Args:
            holdings_df: Snapshot DataFrame; must include
                'ticker', 'category', 'weight_pct'.
            prices: Wide DataFrame; one column per held ticker.
            constraints: Reads excluded_categories,
                min_position_pct, max_weight_default,
                category_caps.
            candidates: Ignored.
            risk_free_rate: Ignored.

        Returns:
            Dict {ticker: fraction} summing to 1.0 (within
            SLSQP tolerance).

        Raises:
            RuntimeError: When fewer than 2 holdings are
                eligible, when an eligible ticker is missing
                from `prices`, or when SLSQP fails to converge.
        """
        eligible = filter_eligible(holdings_df, constraints)
        tickers = list(eligible['ticker'])
        if len(tickers) < 2:
            raise RuntimeError(
                'MinVarianceStrategy: need at least 2 '
                'eligible holdings for a meaningful '
                'optimisation.')

        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise RuntimeError(
                f'MinVarianceStrategy: prices missing '
                f'column(s) for held ticker(s) '
                f'{sorted(missing)}.')

        returns = prices[tickers].pct_change().iloc[1:]
        cov = (returns.cov(min_periods=covariance_min_periods)
               * trading_days_per_year)

        # Drop tickers with NaN covariance (insufficient overlap)
        valid_mask = cov.notna().all()
        if not valid_mask.all():
            tickers = [t for t in tickers if valid_mask[t]]
            cov = cov.loc[tickers, tickers]
        if len(tickers) < 2:
            raise RuntimeError(
                'MinVarianceStrategy: <2 tickers survived '
                'covariance NaN filter.')

        ticker_to_category = dict(
            zip(eligible['ticker'], eligible['category']))
        bounds = []
        for t in tickers:
            cat = ticker_to_category[t]
            cap = constraints.category_caps.get(
                cat, constraints.max_weight_default)
            bounds.append((0.0, cap))

        n = len(tickers)
        w0 = np.array([1.0 / n] * n)
        cov_values = cov.values

        def objective(w):
            return float(w @ cov_values @ w)

        result = minimize(
            objective, w0, method='SLSQP',
            bounds=bounds,
            constraints=[{
                'type': 'eq',
                'fun': lambda w: float(np.sum(w) - 1.0),
            }],
            options={'maxiter': slsqp_max_iter,
                     'ftol': slsqp_ftol})
        if not result.success:
            raise RuntimeError(
                f'MinVarianceStrategy: SLSQP failed to '
                f'converge: {result.message}')
        return {t: float(w) for t, w in zip(tickers, result.x)}
