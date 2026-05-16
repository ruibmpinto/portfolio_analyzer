"""PathDistribution: NAV paths + query methods.

Implements Contract 6 of the refactor design spec. Returned by
MonteCarloEngine.simulate. Stores the raw (n_paths, n_steps+1)
NAV array; query methods compute percentiles, drawdowns,
terminal-return statistics, and cumulative-cost CHF aggregates
on demand.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


trading_days_per_year = 252.0
drawdown_peak_floor = 1e-12


@dataclass(frozen=True)
class PathDistribution:
    """Distribution of CHF NAV paths from a Monte Carlo run.

    Attributes:
        paths: 2D array shape (n_paths, n_steps+1). Column 0
            equals initial_nav_chf for every path.
        horizon_days: Number of trading days simulated.
        initial_nav_chf: NAV at t=0 (column 0 of paths).
        plan_name: Identifier of the RebalancePlan that
            produced the simulation.
        generated_at: Timestamp the simulation finished.
        cumulative_costs_chf: 1D array shape (n_paths,) of
            total transaction costs per path, or None when
            cost_model was disabled.
        cumulative_contributions_chf: Total CHF added to cash
            over the horizon by monthly/quarterly
            contributions. Identical across paths (deterministic
            calendar schedule). Default 0.0 means the run had
            no contributions.
    """

    paths: np.ndarray
    horizon_days: int
    initial_nav_chf: float
    plan_name: str
    generated_at: datetime
    cumulative_costs_chf: Optional[np.ndarray]
    cumulative_contributions_chf: float = 0.0

    @property
    def total_invested_chf(self) -> float:
        """Initial NAV plus cumulative contributions over horizon."""
        return float(
            self.initial_nav_chf
            + self.cumulative_contributions_chf)

    def percentiles(
        self,
        qs: Tuple[float, ...] = (
            0.05, 0.25, 0.50, 0.75, 0.95)
    ) -> pd.DataFrame:
        """Per-step quantile DataFrame.

        Args:
            qs: Quantile levels in [0, 1]. Column labels are
                ``f'p{int(q*100)}'``.

        Returns:
            DataFrame indexed by trading day (0..horizon_days),
            one column per quantile in qs.
        """
        cols = {
            f'p{int(round(q * 100))}': np.quantile(
                self.paths, q, axis=0)
            for q in qs}
        return pd.DataFrame(
            cols, index=np.arange(self.paths.shape[1]))

    def terminal_distribution(self) -> np.ndarray:
        """Terminal NAV per path (1D, length n_paths)."""
        return self.paths[:, -1].copy()

    def max_drawdown_distribution(self) -> np.ndarray:
        """Maximum drawdown per path as negative fractions.

        Returns:
            1D array shape (n_paths,). Each value is
            ``min((nav - running_peak) / running_peak)`` over
            the path. All values are <= 0.0; 0.0 means no
            drawdown occurred (path is monotonically
            non-decreasing).
        """
        peak = np.maximum.accumulate(self.paths, axis=1)
        safe_peak = np.maximum(peak, drawdown_peak_floor)
        dd = (self.paths - peak) / safe_peak
        return dd.min(axis=1)

    def prob_below(self, threshold_chf: float) -> float:
        """Fraction of paths terminating strictly below the threshold."""
        terminal = self.terminal_distribution()
        return float((terminal < threshold_chf).mean())

    def expected_terminal_return(self) -> float:
        """Mean terminal return relative to initial NAV."""
        terminal = self.terminal_distribution()
        return float(
            (terminal / self.initial_nav_chf - 1.0).mean())

    def to_summary_dict(self) -> Dict[str, float]:
        """Headline statistics for the Excel/JSON report.

        Returns:
            Dict with terminal percentiles, expected return,
            drawdown percentiles, loss probability, cost drag
            aggregates, plus the legacy MC stats. The legacy
            P&L stats (mu_ann_pct, sigma_ann_pct, var_5_pct,
            cvar_5_pct, p_gain_10pct, p_gain_20pct) compute
            against ``total_invested_chf`` (initial NAV plus
            cumulative contributions), matching the legacy
            ``pnl_pct = (final_nav - total_invested) /
            total_invested`` semantic. ``prob_loss`` likewise
            counts paths terminating below total_invested, not
            initial NAV. ``expected_terminal_return`` and
            ``terminal_pX`` remain NAV-multiplier semantic
            (against initial_nav_chf) because they describe the
            path's growth multiple, not capital-deployment
            return. Cost fields are 0.0 when
            ``cumulative_costs_chf is None``.
        """
        terminal = self.terminal_distribution()
        dd = self.max_drawdown_distribution()
        invested = self.total_invested_chf
        pnl_pct = (terminal - invested) / invested * 100.0
        var_5 = float(np.percentile(pnl_pct, 5))
        tail = pnl_pct[pnl_pct <= var_5]
        cvar_5 = float(tail.mean()) if tail.size else var_5
        # Headline-stat SEs: SE(mean)=sigma/sqrt(N), SE(p)=sqrt(p(1-p)/N).
        n_paths = int(self.paths.shape[0])
        if n_paths > 1:
            sqrt_n = float(np.sqrt(n_paths))
            mu_se_pct = float(pnl_pct.std(ddof=1) / sqrt_n)
            terminal_se_chf = float(
                terminal.std(ddof=1) / sqrt_n)
            term_ret = (
                terminal / self.initial_nav_chf - 1.0)
            expected_terminal_return_se = float(
                term_ret.std(ddof=1) / sqrt_n)
        else:
            mu_se_pct = 0.0
            terminal_se_chf = 0.0
            expected_terminal_return_se = 0.0
        prob_loss = float((terminal < invested).mean())
        if n_paths > 0:
            prob_loss_se = float(np.sqrt(
                prob_loss * (1.0 - prob_loss) / n_paths))
        else:
            prob_loss_se = 0.0
        if self.cumulative_costs_chf is None:
            cost_p50 = 0.0
            cost_p95 = 0.0
        else:
            cost_p50 = float(np.quantile(
                self.cumulative_costs_chf, 0.5))
            cost_p95 = float(np.quantile(
                self.cumulative_costs_chf, 0.95))
        if self.horizon_days > 0:
            years = self.horizon_days / trading_days_per_year
            drag_bps = (
                (cost_p50 / self.initial_nav_chf) / years
                * 1e4)
        else:
            drag_bps = 0.0
        return {
            'initial_nav_chf': float(self.initial_nav_chf),
            'cumulative_contributions_chf':
                float(self.cumulative_contributions_chf),
            'total_invested_chf': float(invested),
            'horizon_days': int(self.horizon_days),
            'n_paths': n_paths,
            'terminal_p5': float(np.quantile(terminal, 0.05)),
            'terminal_p50': float(np.quantile(terminal, 0.5)),
            'terminal_p95': float(np.quantile(terminal, 0.95)),
            'terminal_mean_chf': float(terminal.mean()),
            'terminal_se_chf': terminal_se_chf,
            'expected_terminal_return':
                self.expected_terminal_return(),
            'expected_terminal_return_se':
                expected_terminal_return_se,
            'max_drawdown_p5': float(np.quantile(dd, 0.05)),
            'max_drawdown_p50': float(np.quantile(dd, 0.5)),
            'max_drawdown_p95': float(np.quantile(dd, 0.95)),
            'prob_loss': prob_loss,
            'prob_loss_se': prob_loss_se,
            'cumulative_cost_chf_p50': cost_p50,
            'cumulative_cost_chf_p95': cost_p95,
            'annual_cost_drag_bps': float(drag_bps),
            'mu_ann_pct': float(pnl_pct.mean()),
            'mu_se_pct': mu_se_pct,
            'sigma_ann_pct': float(pnl_pct.std()),
            'var_5_pct': var_5,
            'cvar_5_pct': cvar_5,
            'p_gain_10pct': float(
                (pnl_pct > 10.0).mean() * 100.0),
            'p_gain_20pct': float(
                (pnl_pct > 20.0).mean() * 100.0),
            'p_dd_over_15': float(
                (dd < -0.15).mean() * 100.0),
            'p_dd_over_25': float(
                (dd < -0.25).mean() * 100.0),
        }
