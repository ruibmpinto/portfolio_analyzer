"""
MVOStrategy: maximise Sharpe ratio under long-only + caps.

Lift-and-shift of the SLSQP optimiser previously embedded in
the legacy report.py. Operates on current holdings only.
risk_free_rate=None triggers a live FRED fetch.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.modelling.rebalancing.strategies.base import Strategy
from src.modelling.rebalancing.strategies.equal_weight import (
    filter_eligible)
from src.shared.constraints import RebalanceConstraints
from src.shared.risk_free_rate import get_live_risk_free_rate


trading_days_per_year = 252
covariance_min_periods = 30
slsqp_max_iter = 1000
slsqp_ftol = 1e-12
volatility_floor = 1e-10


class MVOStrategy(Strategy):
    """
    Maximise (w^T mu - rf) / sqrt(w^T Sigma w).

    mu = annualised mean of daily returns per ticker.
    Sigma = annualised pairwise-complete sample covariance with
    `covariance_min_periods` overlapping observations.
    """

    name = 'mvo'

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Return max-Sharpe target weights.

        Args:
            holdings_df: Snapshot DataFrame; must include
                'ticker', 'category', 'weight_pct'.
            prices: Wide DataFrame; one column per held ticker.
            constraints: Reads excluded_categories,
                min_position_pct, max_weight_default,
                category_caps.
            candidates: Ignored.
            risk_free_rate: Annual rate as a decimal fraction
                (e.g. 0.005 means 0.5%). None triggers a live
                FRED fetch via get_live_risk_free_rate().

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
                'MVOStrategy: need at least 2 eligible '
                'holdings for a meaningful optimisation.')

        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise RuntimeError(
                f'MVOStrategy: prices missing column(s) for '
                f'held ticker(s) {sorted(missing)}.')

        if risk_free_rate is None:
            risk_free_rate = get_live_risk_free_rate()
        # risk_free_rate is already a decimal fraction.
        rf_decimal = float(risk_free_rate)

        returns = prices[tickers].pct_change().iloc[1:]
        mu = returns.mean() * trading_days_per_year
        cov = (returns.cov(min_periods=covariance_min_periods)
               * trading_days_per_year)

        valid_mask = mu.notna() & cov.notna().all()
        if not valid_mask.all():
            tickers = [t for t in tickers if valid_mask[t]]
            mu = mu.loc[tickers]
            cov = cov.loc[tickers, tickers]
            eligible = eligible[
                eligible['ticker'].isin(tickers)]
        if len(tickers) < 2:
            raise RuntimeError(
                'MVOStrategy: <2 tickers survived NaN filter.')

        ticker_to_category = dict(
            zip(eligible['ticker'], eligible['category']))
        bounds = []
        for t in tickers:
            cat = ticker_to_category[t]
            cap = constraints.category_caps.get(
                cat, constraints.max_weight_default)
            bounds.append((0.0, cap))

        mu_values = mu.values
        cov_values = cov.values
        n = len(tickers)

        def neg_sharpe(w):
            port_return = float(w @ mu_values)
            port_vol = float(np.sqrt(w @ cov_values @ w))
            if port_vol < volatility_floor:
                return 1e6
            return -(port_return - rf_decimal) / port_vol

        w0 = np.array([1.0 / n] * n)
        result = minimize(
            neg_sharpe, w0, method='SLSQP',
            bounds=bounds,
            constraints=[{
                'type': 'eq',
                'fun': lambda w: float(np.sum(w) - 1.0),
            }],
            options={'maxiter': slsqp_max_iter,
                     'ftol': slsqp_ftol})
        if not result.success:
            raise RuntimeError(
                f'MVOStrategy: SLSQP failed to converge: '
                f'{result.message}')
        return {t: float(w) for t, w in zip(tickers, result.x)}
