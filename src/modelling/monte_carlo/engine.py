"""Monte Carlo engine for portfolio path simulation.

Consumes a Phase-4 RebalancePlan + daily prices panel and
produces a PathDistribution. Uses Clayton copula joint
sampling, per-ticker Student-t marginals (with James-Stein
shrunk loc), deterministic cash drift, and optional
IBKR-style transaction costs on initial execution + every
rebalance step.
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from scipy.stats import t as t_dist

from src.modelling.monte_carlo.copula import sample_clayton
from src.modelling.monte_carlo.cost_model import (
    TransactionCostModel)
from src.modelling.monte_carlo.path_distribution import (
    PathDistribution)
from src.modelling.monte_carlo.shrinkage import (
    james_stein_shrink)
from src.modelling.rebalancing.action import cash_ticker
from src.modelling.rebalancing.plan import RebalancePlan


df_floor = 2.1
df_ceil = 50.0
theta_floor = 0.1
theta_ceil = 20.0
tau_floor = 0.01
tau_ceil = 0.95
fallback_marginal = (30.0, 3e-4, 1.5e-2)
trading_days_per_year = 252.0
monthly_rebalance_days = 21
quarterly_rebalance_days = 63
default_currency = 'USD'
valid_rebalance_frequencies = (
    'none', 'monthly', 'quarterly', 'threshold')
min_history_for_fit = 60
single_ticker_theta = 1.0


class MonteCarloEngine:
    """Clayton-copula + Student-t Monte Carlo engine.

    Forward-simulates portfolio NAV paths from a RebalancePlan
    and a daily price panel. Non-CASH tickers share a Clayton
    copula whose theta is estimated from median pairwise
    Kendall tau (or supplied explicitly); each ticker's
    marginal is a Student-t fit with James-Stein-shrunk loc.
    Cash earns a deterministic daily rate. Optional transaction
    costs are charged on the initial execution and on every
    rebalance step.

    Attributes:
        n_paths: Number of Monte Carlo paths.
        copula_theta: Clayton parameter; None to fit from data.
        shrinkage_intensity: JS shrinkage; None for JS-optimal.
        anchor_annual: Annual anchor return for JS shrinkage.
        cost_model: TransactionCostModel or None to disable.
        rebalance_frequency: One of 'none', 'monthly',
            'quarterly', 'threshold'.
        monthly_contribution_chf: Cash inflow added each
            monthly rebalance (3x at quarterly rebalances).
        drift_threshold_pct: Per-ticker drift trigger for
            threshold rebalancing, in percentage points.
        random_seed: Seed for the RNG.
        cash_rate_annual: Annual cash drift rate.
        ticker_currencies: Optional ticker -> ISO-4217 map for
            cost-model FX lookup. Missing keys fall back to
            ``default_currency``.
    """

    def __init__(
        self,
        n_paths: int = 10_000,
        copula_theta: Optional[float] = None,
        shrinkage_intensity: Optional[float] = None,
        anchor_annual: float = 0.06,
        cost_model: Optional[TransactionCostModel] = None,
        rebalance_frequency: str = 'monthly',
        monthly_contribution_chf: float = 0.0,
        drift_threshold_pct: float = 5.0,
        random_seed: Optional[int] = None,
        cash_rate_annual: float = 0.015,
        ticker_currencies: Optional[Dict[str, str]] = None):
        if n_paths < 1:
            raise ValueError(
                f'MonteCarloEngine: n_paths must be >= 1, '
                f'got {n_paths}.')
        if rebalance_frequency not in valid_rebalance_frequencies:
            raise ValueError(
                f'MonteCarloEngine: rebalance_frequency must be '
                f'one of {valid_rebalance_frequencies}, got '
                f'{rebalance_frequency!r}.')
        self.n_paths = int(n_paths)
        self.copula_theta = copula_theta
        self.shrinkage_intensity = shrinkage_intensity
        self.anchor_annual = float(anchor_annual)
        self.cost_model = cost_model
        self.rebalance_frequency = rebalance_frequency
        self.monthly_contribution_chf = float(
            monthly_contribution_chf)
        self.drift_threshold_pct = float(drift_threshold_pct)
        self.random_seed = random_seed
        self.cash_rate_annual = float(cash_rate_annual)
        self.ticker_currencies = ticker_currencies or {}

    def simulate(
        self,
        plan: RebalancePlan,
        prices: pd.DataFrame,
        horizon_days: int,
        initial_nav_chf: float) -> PathDistribution:
        """Forward-simulate NAV paths.

        Args:
            plan: RebalancePlan whose target_weights drives
                target allocations. May contain a 'CASH' entry.
            prices: Daily price panel; one column per non-CASH
                ticker in plan.target_weights.
            horizon_days: Number of trading days to simulate.
                Must be >= 1.
            initial_nav_chf: Starting portfolio NAV in CHF.

        Returns:
            PathDistribution with shape (n_paths, horizon_days+1).

        Raises:
            RuntimeError: When horizon_days < 1 or non-CASH
                tickers in plan.target_weights are missing from
                ``prices.columns``.
        """
        if horizon_days < 1:
            raise RuntimeError(
                f'MonteCarloEngine: horizon_days must be >= 1, '
                f'got {horizon_days}.')
        # Split target weights into equity vs cash buckets.
        target_weights = plan.target_weights
        equity_tickers = [
            t for t in target_weights if t != cash_ticker]
        cash_weight = float(
            target_weights.get(cash_ticker, 0.0))
        equity_weights = np.array(
            [float(target_weights[t]) for t in equity_tickers],
            dtype=float)

        # Validate price coverage.
        missing = [
            t for t in equity_tickers
            if t not in prices.columns]
        if missing:
            raise RuntimeError(
                f'MonteCarloEngine: tickers missing from prices '
                f'panel: {missing}.')
        n_equity = len(equity_tickers)

        # Fit Student-t marginals per equity ticker.
        dfs = np.zeros(n_equity)
        locs = np.zeros(n_equity)
        scales = np.zeros(n_equity)
        per_ticker_returns: Dict[str, np.ndarray] = {}
        for k, ticker in enumerate(equity_tickers):
            r = prices[ticker].pct_change().dropna().values
            per_ticker_returns[ticker] = r
            if len(r) < min_history_for_fit:
                df_fit, loc_fit, scale_fit = fallback_marginal
            else:
                df_fit, loc_fit, scale_fit = t_dist.fit(r)
                df_fit = float(
                    np.clip(df_fit, df_floor, df_ceil))
            dfs[k] = df_fit
            locs[k] = loc_fit
            scales[k] = scale_fit

        # JS-shrink loc parameters when we have >= 2 equity legs.
        if n_equity >= 2:
            returns_df = (
                prices[equity_tickers]
                .pct_change()
                .dropna(how='all'))
            shrunk = james_stein_shrink(
                returns_df,
                anchor_annual=self.anchor_annual,
                intensity=self.shrinkage_intensity)
            for k, ticker in enumerate(equity_tickers):
                locs[k] = float(shrunk[ticker])

        # Determine Clayton theta.
        theta = self._resolve_theta(
            equity_tickers, per_ticker_returns)

        # Initial execution: holdings, cash, t0 cost.
        rng = np.random.default_rng(self.random_seed)
        eq_total_weight = float(equity_weights.sum())
        eq_value_t0 = initial_nav_chf * eq_total_weight
        if n_equity > 0 and eq_total_weight > 0:
            per_leg_value = eq_value_t0 * (
                equity_weights / eq_total_weight)
        else:
            per_leg_value = np.zeros(n_equity)
        holdings = np.tile(
            per_leg_value, (self.n_paths, 1)).astype(float)
        cash = np.full(
            self.n_paths,
            initial_nav_chf * cash_weight,
            dtype=float)
        if self.cost_model is not None:
            t0_cost = 0.0
            for k, ticker in enumerate(equity_tickers):
                trade_value = float(per_leg_value[k])
                t0_cost += self.cost_model.cost_chf(
                    trade_value, ticker,
                    self._currency_of(ticker))
            cash -= t0_cost
            cumulative_costs = np.full(
                self.n_paths, t0_cost, dtype=float)
        else:
            cumulative_costs = None

        # NAV history allocation.
        nav = np.zeros(
            (self.n_paths, horizon_days + 1), dtype=float)
        nav[:, 0] = holdings.sum(axis=1) + cash
        cash_daily_factor = (
            1.0 + self.cash_rate_annual / trading_days_per_year)
        total_contributions_chf = 0.0

        # Per-step loop.
        for d in range(horizon_days):
            if n_equity > 0:
                u = sample_clayton(
                    theta, self.n_paths, n_equity, rng)
                returns_d = np.zeros_like(u)
                for k in range(n_equity):
                    returns_d[:, k] = t_dist.ppf(
                        u[:, k],
                        df=dfs[k],
                        loc=locs[k],
                        scale=scales[k])
                holdings *= (1.0 + returns_d)
            cash *= cash_daily_factor
            day = d + 1
            total_contributions_chf += self._maybe_rebalance(
                day, holdings, cash, equity_tickers,
                equity_weights, eq_total_weight, cash_weight,
                cumulative_costs)
            nav[:, day] = holdings.sum(axis=1) + cash
        return PathDistribution(
            paths=nav,
            horizon_days=horizon_days,
            initial_nav_chf=float(initial_nav_chf),
            plan_name=plan.strategy_name,
            generated_at=datetime.now(),
            cumulative_costs_chf=cumulative_costs,
            cumulative_contributions_chf=total_contributions_chf)
    # -------------------------------------------------------------
    def _resolve_theta(
        self,
        equity_tickers: List[str],
        per_ticker_returns: Dict[str, np.ndarray]) -> float:
        """Pick Clayton theta from data or constructor.

        Args:
            equity_tickers: Ordered ticker list.
            per_ticker_returns: Map ticker -> daily return array.

        Returns:
            Clayton theta in ``[theta_floor, theta_ceil]``.
        """
        if self.copula_theta is not None:
            return float(np.clip(
                self.copula_theta, theta_floor, theta_ceil))
        if len(equity_tickers) < 2:
            return single_ticker_theta
        taus = []
        for i in range(len(equity_tickers)):
            for j in range(i + 1, len(equity_tickers)):
                r_i = per_ticker_returns[equity_tickers[i]]
                r_j = per_ticker_returns[equity_tickers[j]]
                n = min(len(r_i), len(r_j))
                if n < 2:
                    continue
                tau, _ = kendalltau(r_i[-n:], r_j[-n:])
                if np.isnan(tau):
                    continue
                taus.append(float(tau))
        if not taus:
            return single_ticker_theta
        med_tau = float(np.median(taus))
        med_tau = float(np.clip(med_tau, tau_floor, tau_ceil))
        theta = 2.0 * med_tau / (1.0 - med_tau)
        return float(np.clip(theta, theta_floor, theta_ceil))
    # -------------------------------------------------------------
    def _currency_of(self, ticker: str) -> str:
        """Look up listing currency for cost model FX cost.

        Args:
            ticker: Equity ticker.

        Returns:
            ISO-4217 currency code; ``default_currency`` when
            ticker missing from the mapping.
        """
        return self.ticker_currencies.get(
            ticker, default_currency)
    # -------------------------------------------------------------
    def _maybe_rebalance(
        self,
        day: int,
        holdings: np.ndarray,
        cash: np.ndarray,
        equity_tickers: List[str],
        equity_weights: np.ndarray,
        eq_total_weight: float,
        cash_weight: float,
        cumulative_costs: Optional[np.ndarray]) -> float:
        """Apply rebalancing on the current trading day.

        Mutates ``holdings``, ``cash`` and ``cumulative_costs``
        in place. Picks the rebalance trigger from
        ``self.rebalance_frequency``:

        - ``'none'``: no rebalance.
        - ``'monthly'``: every ``monthly_rebalance_days``; add
          one ``monthly_contribution_chf`` to cash on each.
        - ``'quarterly'``: every ``quarterly_rebalance_days``;
          add three contributions on each.
        - ``'threshold'``: rebalance whenever any path's max
          per-ticker |current_w - target_w| exceeds
          ``drift_threshold_pct / 100``. No contribution.

        Args:
            day: 1-indexed trading day count.
            holdings: (n_paths, n_equity) equity values.
            cash: (n_paths,) cash balance.
            equity_tickers: Ticker list aligned with holdings cols.
            equity_weights: Raw target weight vector.
            eq_total_weight: Sum of equity_weights.
            cash_weight: Target cash fraction.
            cumulative_costs: Running per-path cost array;
                None when cost_model is disabled.

        Returns:
            CHF contribution applied to cash on this day (0.0
            when no rebalance fired or the trigger added no
            cash).
        """
        n_equity = len(equity_tickers)
        if n_equity == 0:
            return 0.0
        freq = self.rebalance_frequency
        if freq == 'none':
            return 0.0
        contribution = 0.0
        if freq == 'monthly':
            if day % monthly_rebalance_days != 0:
                return 0.0
            contribution = self.monthly_contribution_chf
            cash += contribution
        elif freq == 'quarterly':
            if day % quarterly_rebalance_days != 0:
                return 0.0
            contribution = 3.0 * self.monthly_contribution_chf
            cash += contribution
        elif freq == 'threshold':
            if not self._threshold_drift_breached(
                    holdings, cash, equity_weights,
                    eq_total_weight, cash_weight):
                return 0.0

        # Compute provisional targets from pre-cost NAV, then
        # charge the cost from cash, then rebalance into the
        # post-cost NAV split.
        pre_nav = holdings.sum(axis=1) + cash
        provisional_eq_total = pre_nav * eq_total_weight
        if eq_total_weight > 0:
            provisional_target = (
                provisional_eq_total[:, None]
                * (equity_weights / eq_total_weight)[None, :])
        else:
            provisional_target = np.zeros_like(holdings)
        trade_value = np.abs(provisional_target - holdings)
        if self.cost_model is not None:
            step_cost = np.zeros(holdings.shape[0])
            for k, ticker in enumerate(equity_tickers):
                currency = self._currency_of(ticker)
                step_cost += self.cost_model.cost_chf_array(
                    trade_value[:, k], ticker, currency)
            if cumulative_costs is not None:
                cumulative_costs += step_cost
        else:
            step_cost = np.zeros(holdings.shape[0])
        # Post-cost NAV gets re-split between equity and cash.
        post_nav = pre_nav - step_cost
        target_eq_total = post_nav * eq_total_weight
        if eq_total_weight > 0:
            holdings[:, :] = (
                target_eq_total[:, None]
                * (equity_weights / eq_total_weight)[None, :])
        else:
            holdings[:, :] = 0.0
        cash[:] = post_nav - target_eq_total
        return contribution
    # -------------------------------------------------------------
    def _threshold_drift_breached(
        self,
        holdings: np.ndarray,
        cash: np.ndarray,
        equity_weights: np.ndarray,
        eq_total_weight: float,
        cash_weight: float) -> bool:
        """Return True if any path's drift exceeds the threshold.

        Uses a uniform-day rebalance policy: if any single path
        has a per-ticker drift greater than the configured
        threshold, all paths rebalance together. This is the
        approximation called out in the spec.

        Args:
            holdings: (n_paths, n_equity) values.
            cash: (n_paths,) cash balance.
            equity_weights: Raw target weight vector.
            eq_total_weight: Sum of equity_weights.
            cash_weight: Target cash fraction.

        Returns:
            True if any path triggers; False otherwise.
        """
        nav_total = holdings.sum(axis=1) + cash
        safe_nav = np.where(nav_total > 0, nav_total, 1.0)
        current_w_eq = holdings / safe_nav[:, None]
        eq_drift = np.max(
            np.abs(current_w_eq - equity_weights[None, :]),
            axis=1)
        cash_drift = np.abs(cash / safe_nav - cash_weight)
        max_drift = np.maximum(eq_drift, cash_drift)
        threshold = self.drift_threshold_pct / 100.0
        return bool((max_drift > threshold).any())
