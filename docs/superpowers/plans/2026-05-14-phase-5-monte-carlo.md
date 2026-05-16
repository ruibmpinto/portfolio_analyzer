# Phase 5 — Monte Carlo refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 907-line monolithic `src/modelling/monte_carlo.py` with a `src/modelling/monte_carlo/` package. New surface: `PathDistribution` data class (Contract 6), pure-function `sample_clayton` (Clayton copula sampler) and `james_stein_shrink` (expected-return shrinkage), `TransactionCostModel` (IBKR Swiss-resident defaults), and `MonteCarloEngine.simulate(plan, prices, horizon_days, initial_nav_chf) -> PathDistribution`. Add `src/user_scripts/run_monte_carlo.py` CLI. Delete the legacy file.

**Architecture:**
- The engine consumes a Phase-4 `RebalancePlan` directly. `plan.target_weights` provides the per-ticker fractions (including the `CASH` pseudo-ticker if present). Equity tickers are simulated under a Clayton-copula + Student-t marginal model with James-Stein-shrunk loc parameters; cash earns a deterministic daily rate.
- The engine fits Student-t marginals from the daily `prices` panel passed in (output of `analyzer.get_price_panel()`). For any non-held target ticker (e.g. screener-fed candidates), the caller is responsible for extending `prices` to cover it; if a ticker is missing the engine raises.
- `PathDistribution` is data-only: NAV paths as a 2D `np.ndarray` plus the metadata needed to derive percentiles, drawdowns, terminal-return statistics, and cumulative-cost CHF aggregates. Query methods compute on demand; no caching beyond the original `paths` array.
- `TransactionCostModel` is a frozen dataclass with `cost_chf(trade_value_chf, ticker, currency) -> float`. The engine deducts costs (a) once at t=0 to execute the plan, and (b) on every rebalance step proportional to the absolute weight-drift trade value. `cost_model=None` runs a costless simulation.
- Rebalancing frequency is one of `'none' | 'monthly' | 'quarterly' | 'threshold'`. `'threshold'` triggers when any equity weight's absolute drift exceeds `drift_threshold_pct` of NAV.
- CLI re-uses the `report.py` `_build_data_provider` factory hook so tests can monkeypatch FX/price fetches.

**Tech Stack:** Python 3.12, pandas, numpy, scipy (`scipy.stats.t`, `scipy.stats.kendalltau`), pytest 8.3.4, openpyxl.

**Style:** Match `src/analysis/core/portfolio.py` and `src/modelling/rebalancing/strategies/mvo.py`. Type hints in signatures. Google-style docstrings (Args/Returns/Raises). 80-char max. Single quotes for strings, double for docstrings. Lowercase module-level constants. No banner separators. No `__author__` blocks. Per HANDOFF.md §4 this overrides the global `~/.claude/rules/code-style.md`.

**Conventions:**
- TDD per task (red → green). Tests live under `tests/modelling/monte_carlo/` mirroring the source layout.
- Two-stage review after each task (spec compliance + code quality) per HANDOFF.md §3.
- "Commit (advisory)" steps are NOT authorizations — do not commit unless the user explicitly asks.
- Loud failures: missing tickers, non-positive prices, infeasible parameters all raise `RuntimeError`/`KeyError`. Per `feedback_no_silent_defaults.md`.
- No bond ETFs in any default policy. Per `feedback_swiss_tax_no_bonds.md`. (Cash is the defensive lever — modelled as deterministic daily rate.)

---

## File Structure

### New files
- `src/modelling/monte_carlo/__init__.py` — package re-exports.
- `src/modelling/monte_carlo/path_distribution.py` — `PathDistribution` class.
- `src/modelling/monte_carlo/copula.py` — `sample_clayton`.
- `src/modelling/monte_carlo/shrinkage.py` — `james_stein_shrink`.
- `src/modelling/monte_carlo/cost_model.py` — `TransactionCostModel` + `ibkr_default_cost_model()`.
- `src/modelling/monte_carlo/engine.py` — `MonteCarloEngine`.
- `src/user_scripts/run_monte_carlo.py` — CLI entry point.
- `tests/modelling/monte_carlo/__init__.py` — empty.
- `tests/modelling/monte_carlo/test_path_distribution.py`
- `tests/modelling/monte_carlo/test_copula.py`
- `tests/modelling/monte_carlo/test_shrinkage.py`
- `tests/modelling/monte_carlo/test_cost_model.py`
- `tests/modelling/monte_carlo/test_engine.py`
- `tests/user_scripts/test_run_monte_carlo.py`

### Deleted files
- `src/modelling/monte_carlo.py` (907-line legacy monolith).

### Untouched
- `src/modelling/rebalancing/` (Phase 4 deliverable, frozen).
- `src/modelling/greeks/` (Phase 6 deliverable, not yet refactored).
- `src/analysis/core/`, `src/analysis/report.py`, `src/screening/` — none modified by Phase 5.

---

## Task 1: `PathDistribution` data + query class

**Files:**
- Create: `src/modelling/monte_carlo/__init__.py` (empty placeholder for now — re-exports added in Task 6).
- Create: `src/modelling/monte_carlo/path_distribution.py`
- Create: `tests/modelling/monte_carlo/__init__.py` (empty).
- Create: `tests/modelling/monte_carlo/test_path_distribution.py`

**Why:** Contract 6 says the engine returns a `PathDistribution` carrying NAV paths plus query methods. Implementing it first lets the engine (Task 5) target a known data shape and lets tests for the engine assert structured output rather than raw arrays.

**Class shape:**

```python
@dataclass(frozen=True)
class PathDistribution:
    paths: np.ndarray            # (n_paths, n_steps+1), CHF NAV
    horizon_days: int
    initial_nav_chf: float
    plan_name: str
    generated_at: datetime
    cumulative_costs_chf: np.ndarray | None  # (n_paths,), or None when costless

    def percentiles(self, qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> pd.DataFrame: ...
    def terminal_distribution(self) -> np.ndarray: ...
    def max_drawdown_distribution(self) -> np.ndarray: ...
    def prob_below(self, threshold_chf: float) -> float: ...
    def expected_terminal_return(self) -> float: ...
    def to_summary_dict(self) -> dict[str, float]: ...
```

**Method semantics:**
- `percentiles(qs)`: returns a DataFrame indexed by trading day (0 .. horizon_days), columns are each `q` formatted as `'p{int(q*100)}'`, values are the per-step quantile of NAV across paths.
- `terminal_distribution()`: 1D array of length `n_paths`, terminal NAV.
- `max_drawdown_distribution()`: 1D array of length `n_paths`, each path's maximum drawdown expressed as a NEGATIVE fraction (e.g. `-0.32` for a 32% drawdown). Use `np.maximum.accumulate(paths, axis=1)` as the running peak.
- `prob_below(threshold_chf)`: fraction of paths whose terminal NAV is strictly below the threshold.
- `expected_terminal_return()`: mean of `(terminal / initial_nav_chf - 1)`.
- `to_summary_dict()`: returns a dict with keys: `initial_nav_chf, horizon_days, n_paths, terminal_p5, terminal_p50, terminal_p95, expected_terminal_return, max_drawdown_p5, max_drawdown_p50, max_drawdown_p95, prob_loss, cumulative_cost_chf_p50, cumulative_cost_chf_p95, annual_cost_drag_bps`. The three cost fields are 0.0 when `cumulative_costs_chf is None`. `annual_cost_drag_bps` = `(cumulative_cost_chf_p50 / initial_nav_chf) / (horizon_days / 252) * 1e4`.

- [ ] **Step 1: Write failing tests**

Create `tests/modelling/monte_carlo/__init__.py` (empty).

Create `tests/modelling/monte_carlo/test_path_distribution.py`:

```python
"""Tests for PathDistribution."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.modelling.monte_carlo.path_distribution import (
    PathDistribution)


def _make_paths(n_paths=4, n_steps=10, seed=0):
    """Deterministic NAV paths starting at 10000."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(
        loc=10.0, scale=20.0, size=(n_paths, n_steps))
    nav = np.zeros((n_paths, n_steps + 1))
    nav[:, 0] = 10000.0
    for d in range(n_steps):
        nav[:, d + 1] = nav[:, d] + increments[:, d]
    return nav


def test_path_distribution_construction():
    """Frozen dataclass holds paths and metadata."""
    paths = _make_paths()
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    assert pd_.paths.shape == (4, 11)
    assert pd_.plan_name == 'test'


def test_percentiles_shape_and_columns():
    """percentiles() returns DataFrame indexed by step."""
    paths = _make_paths()
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    df = pd_.percentiles(qs=(0.05, 0.5, 0.95))
    assert list(df.columns) == ['p5', 'p50', 'p95']
    assert len(df) == 11
    # Monotone non-decreasing quantiles per row
    assert (df['p5'] <= df['p50']).all()
    assert (df['p50'] <= df['p95']).all()


def test_terminal_distribution_returns_last_column():
    paths = _make_paths()
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    terminal = pd_.terminal_distribution()
    assert terminal.shape == (4,)
    np.testing.assert_array_equal(terminal, paths[:, -1])


def test_max_drawdown_distribution_is_negative_fraction():
    """Drawdown is the most-negative running-peak deviation."""
    paths = np.array([
        [100.0, 120.0, 90.0, 110.0, 60.0],
        [100.0, 110.0, 105.0, 115.0, 130.0],
    ])
    pd_ = PathDistribution(
        paths=paths, horizon_days=4,
        initial_nav_chf=100.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    dd = pd_.max_drawdown_distribution()
    # Path 0 peaks at 120, drops to 60 -> (60-120)/120 = -0.5.
    assert dd[0] == pytest.approx(-0.5)
    # Path 1 monotone up -> 0.0.
    assert dd[1] == pytest.approx(0.0)


def test_prob_below_threshold():
    paths = np.array([
        [10000.0, 12000.0],
        [10000.0, 8000.0],
        [10000.0, 9500.0],
        [10000.0, 7000.0],
    ])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    # Below 9000: paths 1 and 3 -> 50%
    assert pd_.prob_below(9000.0) == pytest.approx(0.5)


def test_expected_terminal_return():
    paths = np.array([
        [100.0, 110.0],
        [100.0, 120.0],
        [100.0, 90.0],
    ])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=100.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    # Mean terminal = 106.667. Return = 0.06667.
    assert pd_.expected_terminal_return() == pytest.approx(
        0.06666667, abs=1e-6)


def test_to_summary_dict_costless():
    """Costless run reports zero cost fields."""
    paths = _make_paths()
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    s = pd_.to_summary_dict()
    expected_keys = {
        'initial_nav_chf', 'horizon_days', 'n_paths',
        'terminal_p5', 'terminal_p50', 'terminal_p95',
        'expected_terminal_return',
        'max_drawdown_p5', 'max_drawdown_p50',
        'max_drawdown_p95',
        'prob_loss',
        'cumulative_cost_chf_p50',
        'cumulative_cost_chf_p95',
        'annual_cost_drag_bps'}
    assert set(s) == expected_keys
    assert s['cumulative_cost_chf_p50'] == 0.0
    assert s['cumulative_cost_chf_p95'] == 0.0
    assert s['annual_cost_drag_bps'] == 0.0
    assert s['n_paths'] == 4
    assert s['horizon_days'] == 10


def test_to_summary_dict_with_costs():
    """Non-None cumulative_costs_chf flows into the summary."""
    paths = _make_paths()
    costs = np.array([100.0, 150.0, 200.0, 250.0])
    pd_ = PathDistribution(
        paths=paths, horizon_days=10,
        initial_nav_chf=10000.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=costs)
    s = pd_.to_summary_dict()
    assert s['cumulative_cost_chf_p50'] == pytest.approx(175.0)
    assert s['cumulative_cost_chf_p95'] == pytest.approx(
        242.5, abs=2.5)
    # Drag bps = (175 / 10000) / (10/252) * 1e4
    expected_drag = (175.0 / 10000.0) / (10.0 / 252.0) * 1e4
    assert s['annual_cost_drag_bps'] == pytest.approx(
        expected_drag, abs=1.0)


def test_prob_loss_uses_initial_nav():
    """prob_loss is the fraction of paths terminating below initial NAV."""
    paths = np.array([
        [100.0, 110.0],
        [100.0, 80.0],
        [100.0, 95.0],
        [100.0, 130.0]])
    pd_ = PathDistribution(
        paths=paths, horizon_days=1,
        initial_nav_chf=100.0,
        plan_name='test', generated_at=datetime.now(),
        cumulative_costs_chf=None)
    s = pd_.to_summary_dict()
    # Paths 1 and 2 below 100 -> 50%
    assert s['prob_loss'] == pytest.approx(0.5)
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m pytest tests/modelling/monte_carlo/test_path_distribution.py -v`
Expected: ImportError on `PathDistribution`.

- [ ] **Step 3: Implement**

Create `src/modelling/monte_carlo/__init__.py` as a placeholder docstring + empty body (Task 6 will populate re-exports):

```python
"""Monte Carlo simulation domain.

PathDistribution data class + query methods, Clayton copula
sampler, James-Stein shrinkage, transaction cost model, and
the MonteCarloEngine orchestrator. Public surface is added
incrementally across Phase 5 tasks; see __init__.py at the
end of the phase for the final re-export list.
"""
```

Create `src/modelling/monte_carlo/path_distribution.py`:

```python
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
    """

    paths: np.ndarray
    horizon_days: int
    initial_nav_chf: float
    plan_name: str
    generated_at: datetime
    cumulative_costs_chf: Optional[np.ndarray]

    def percentiles(
        self,
        qs: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
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
            the path, so monotonically non-decreasing paths
            return 0.0.
        """
        peak = np.maximum.accumulate(self.paths, axis=1)
        dd = (self.paths - peak) / np.where(
            peak > 0, peak, 1.0)
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
            drawdown percentiles, loss probability, and cost
            drag aggregates. Cost fields are 0.0 when
            ``cumulative_costs_chf is None``.
        """
        terminal = self.terminal_distribution()
        dd = self.max_drawdown_distribution()
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
            'horizon_days': int(self.horizon_days),
            'n_paths': int(self.paths.shape[0]),
            'terminal_p5': float(np.quantile(terminal, 0.05)),
            'terminal_p50': float(np.quantile(terminal, 0.5)),
            'terminal_p95': float(np.quantile(terminal, 0.95)),
            'expected_terminal_return':
                self.expected_terminal_return(),
            'max_drawdown_p5': float(np.quantile(dd, 0.05)),
            'max_drawdown_p50': float(np.quantile(dd, 0.5)),
            'max_drawdown_p95': float(np.quantile(dd, 0.95)),
            'prob_loss': self.prob_below(
                float(self.initial_nav_chf)),
            'cumulative_cost_chf_p50': cost_p50,
            'cumulative_cost_chf_p95': cost_p95,
            'annual_cost_drag_bps': float(drag_bps),
        }
```

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/modelling/monte_carlo/test_path_distribution.py -v`
Expected: 9 PASS.

Run: `python3 -m pytest tests/ -v 2>&1 | tail -3`
Expected: 169 + 9 = 178 PASS.

- [ ] **Step 5: NO commit.**

---

## Task 2: Clayton copula sampler

**Files:**
- Create: `src/modelling/monte_carlo/copula.py`
- Test: `tests/modelling/monte_carlo/test_copula.py`

**Why:** Spec §5 specifies a pure function `sample_clayton(theta, n_samples, n_assets, rng) -> np.ndarray`. The legacy module implemented this as the Marshall-Olkin algorithm (frailty + independent exponentials). Extract it as a pure function so the engine and tests can use it directly without touching simulation state.

**Function spec:**

```python
def sample_clayton(
    theta: float,
    n_samples: int,
    n_assets: int,
    rng: np.random.Generator) -> np.ndarray:
    """Sample n_samples × n_assets uniforms with Clayton dependence.

    Uses the Marshall-Olkin algorithm: shared frailty
    V ~ Gamma(1/theta, 1) drives co-crash dependence;
    independent exponentials E_i drive idiosyncratic shocks.
    U_i = (1 + E_i / V) ^ (-1/theta).

    Args:
        theta: Clayton parameter, must be > 0. Higher = stronger
            lower-tail dependence.
        n_samples: Number of joint samples to draw.
        n_assets: Dimension per sample.
        rng: NumPy random generator.

    Returns:
        Array shape (n_samples, n_assets) of uniforms in (0, 1),
        clipped to [1e-7, 1 - 1e-7] for numerical safety.

    Raises:
        ValueError: theta <= 0, n_samples < 1, or n_assets < 1.
    """
```

- [ ] **Step 1: Write failing tests**

Create `tests/modelling/monte_carlo/test_copula.py`:

```python
"""Tests for sample_clayton."""

import numpy as np
import pytest

from src.modelling.monte_carlo.copula import sample_clayton


def test_sample_clayton_shape():
    """Output shape matches (n_samples, n_assets)."""
    rng = np.random.default_rng(0)
    u = sample_clayton(theta=2.0, n_samples=100, n_assets=4,
                       rng=rng)
    assert u.shape == (100, 4)


def test_sample_clayton_uniform_marginals():
    """Each column has approximately uniform[0,1] marginals."""
    rng = np.random.default_rng(0)
    u = sample_clayton(theta=2.0, n_samples=20000, n_assets=3,
                       rng=rng)
    for j in range(3):
        # Mean of U[0,1] ~ 0.5
        assert abs(u[:, j].mean() - 0.5) < 0.02
        # Std of U[0,1] ~ 0.289
        assert abs(u[:, j].std() - 0.2887) < 0.02


def test_sample_clayton_lower_tail_dependence():
    """Higher theta -> stronger co-crash (lower-tail dependence)."""
    rng_low = np.random.default_rng(1)
    rng_high = np.random.default_rng(1)
    u_low = sample_clayton(
        theta=0.5, n_samples=20000, n_assets=2, rng=rng_low)
    u_high = sample_clayton(
        theta=8.0, n_samples=20000, n_assets=2, rng=rng_high)

    # Fraction of joint draws where both u_i < 0.05
    joint_low_low = (
        (u_low[:, 0] < 0.05) & (u_low[:, 1] < 0.05)).mean()
    joint_low_high = (
        (u_high[:, 0] < 0.05) & (u_high[:, 1] < 0.05)).mean()
    assert joint_low_high > joint_low_low * 2


def test_sample_clayton_values_strictly_inside_unit():
    """Clipping prevents 0 or 1 exactly."""
    rng = np.random.default_rng(0)
    u = sample_clayton(theta=2.0, n_samples=5000, n_assets=4,
                       rng=rng)
    assert u.min() > 0.0
    assert u.max() < 1.0


def test_sample_clayton_rejects_invalid_theta():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='theta'):
        sample_clayton(theta=0.0, n_samples=10, n_assets=2,
                       rng=rng)
    with pytest.raises(ValueError, match='theta'):
        sample_clayton(theta=-1.0, n_samples=10, n_assets=2,
                       rng=rng)


def test_sample_clayton_rejects_invalid_sizes():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='n_samples'):
        sample_clayton(theta=2.0, n_samples=0, n_assets=2,
                       rng=rng)
    with pytest.raises(ValueError, match='n_assets'):
        sample_clayton(theta=2.0, n_samples=10, n_assets=0,
                       rng=rng)


def test_sample_clayton_reproducible_with_same_seed():
    """Same rng seed yields identical output."""
    a = sample_clayton(
        theta=2.0, n_samples=200, n_assets=3,
        rng=np.random.default_rng(42))
    b = sample_clayton(
        theta=2.0, n_samples=200, n_assets=3,
        rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m pytest tests/modelling/monte_carlo/test_copula.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/modelling/monte_carlo/copula.py`:

```python
"""Clayton copula sampler.

Pure function: given a Clayton parameter theta > 0 and a
random generator, draws joint uniforms whose marginals are
uniform[0, 1] but exhibit lower-tail dependence proportional
to theta. Used by MonteCarloEngine to drive co-crash structure
between equity tickers without inflating upper-tail dependence.

Algorithm: Marshall-Olkin. Shared frailty V ~ Gamma(1/theta, 1),
independent exponentials E_i ~ Exp(1) per asset, then
U_i = (1 + E_i / V) ^ (-1/theta).
"""

import numpy as np


u_clip_floor = 1e-7
u_clip_ceil = 1.0 - 1e-7


def sample_clayton(
    theta: float,
    n_samples: int,
    n_assets: int,
    rng: np.random.Generator) -> np.ndarray:
    """Draw joint uniforms with Clayton lower-tail dependence.

    Args:
        theta: Clayton parameter, > 0. Higher = stronger
            co-crash dependence.
        n_samples: Number of joint draws.
        n_assets: Number of asset dimensions per draw.
        rng: NumPy random generator (e.g. np.random.default_rng).

    Returns:
        Array (n_samples, n_assets) of values in
        [u_clip_floor, u_clip_ceil].

    Raises:
        ValueError: theta <= 0, n_samples < 1, or n_assets < 1.
    """
    if theta <= 0:
        raise ValueError(
            f'sample_clayton: theta must be positive, '
            f'got {theta}.')
    if n_samples < 1:
        raise ValueError(
            f'sample_clayton: n_samples must be >= 1, '
            f'got {n_samples}.')
    if n_assets < 1:
        raise ValueError(
            f'sample_clayton: n_assets must be >= 1, '
            f'got {n_assets}.')
    inv_theta = 1.0 / theta
    neg_inv_theta = -1.0 / theta
    v = rng.gamma(inv_theta, 1.0, size=(n_samples, 1))
    e = rng.exponential(1.0, size=(n_samples, n_assets))
    u = (1.0 + e / v) ** neg_inv_theta
    return np.clip(u, u_clip_floor, u_clip_ceil)
```

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/modelling/monte_carlo/test_copula.py -v`
Expected: 7 PASS.

- [ ] **Step 5: NO commit.**

---

## Task 3: James-Stein shrinkage

**Files:**
- Create: `src/modelling/monte_carlo/shrinkage.py`
- Test: `tests/modelling/monte_carlo/test_shrinkage.py`

**Why:** Survivorship-biased historical means dominate the Monte Carlo drift unless we shrink toward a stationary anchor. The James-Stein estimator shrinks each ticker's daily mean toward an `anchor` parameter (an annual equity risk premium converted to daily), with a shrinkage intensity learned from the cross-sectional variance of the means.

**Function spec:**

```python
def james_stein_shrink(
    returns: pd.DataFrame,
    anchor_annual: float = 0.06,
    intensity: Optional[float] = None) -> pd.Series:
    """Shrink ticker daily-mean returns toward a common anchor.

    Args:
        returns: Daily-return DataFrame, one column per ticker.
            Index = trading days; NaNs allowed.
        anchor_annual: Annualised target return to shrink toward
            (default 0.06 = 6% equity risk premium).
        intensity: If given, use this fixed shrinkage in [0, 1].
            If None, compute the James-Stein optimal value from
            the cross-sectional variance of the per-ticker means.

    Returns:
        pd.Series indexed by ticker; values are shrunk daily
        mean returns.

    Raises:
        ValueError: When ``returns`` has fewer than two columns
            (James-Stein undefined for p < 2 and intensity
            inference relies on at least three).
    """
```

JS formula: `b_js = max(0, 1 - (p - 2) * var_pool / ss)` where `var_pool = mean(scale^2) / n_obs` is a heuristic pooled estimator. Shrunk mean for ticker `i` = `anchor_daily + b_js * (mean_i - anchor_daily)`. When `intensity` is supplied, use it verbatim instead.

- [ ] **Step 1: Write failing tests**

Create `tests/modelling/monte_carlo/test_shrinkage.py`:

```python
"""Tests for james_stein_shrink."""

import numpy as np
import pandas as pd
import pytest

from src.modelling.monte_carlo.shrinkage import (
    james_stein_shrink)


def _returns(n_days=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'A': rng.normal(0.0008, 0.012, n_days),
        'B': rng.normal(0.0002, 0.010, n_days),
        'C': rng.normal(-0.0001, 0.014, n_days),
        'D': rng.normal(0.0005, 0.011, n_days)})


def test_shrinks_toward_anchor():
    """Shrunk values lie between raw mean and the anchor."""
    df = _returns()
    raw = df.mean()
    shrunk = james_stein_shrink(df, anchor_annual=0.06)
    anchor_daily = 0.06 / 252.0
    for t in df.columns:
        lo = min(raw[t], anchor_daily)
        hi = max(raw[t], anchor_daily)
        assert lo - 1e-12 <= shrunk[t] <= hi + 1e-12


def test_explicit_intensity_one_returns_raw_means():
    """intensity=1.0 disables shrinkage."""
    df = _returns()
    raw = df.mean()
    shrunk = james_stein_shrink(
        df, anchor_annual=0.06, intensity=1.0)
    for t in df.columns:
        assert shrunk[t] == pytest.approx(raw[t])


def test_explicit_intensity_zero_collapses_to_anchor():
    """intensity=0.0 collapses every ticker to the anchor."""
    df = _returns()
    shrunk = james_stein_shrink(
        df, anchor_annual=0.06, intensity=0.0)
    anchor_daily = 0.06 / 252.0
    for t in df.columns:
        assert shrunk[t] == pytest.approx(anchor_daily)


def test_raises_when_fewer_than_two_columns():
    df = _returns().iloc[:, :1]
    with pytest.raises(ValueError, match='at least'):
        james_stein_shrink(df)


def test_returns_series_indexed_by_ticker():
    df = _returns()
    s = james_stein_shrink(df)
    assert isinstance(s, pd.Series)
    assert set(s.index) == set(df.columns)
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m pytest tests/modelling/monte_carlo/test_shrinkage.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/modelling/monte_carlo/shrinkage.py`:

```python
"""James-Stein shrinkage for daily mean returns.

Pure function. Shrinks per-ticker historical daily means toward
a common anchor (default = 6% annualised equity risk premium).
Reduces the influence of survivorship-biased outliers on the
Monte Carlo drift.
"""

from typing import Optional

import numpy as np
import pandas as pd


trading_days_per_year = 252.0


def james_stein_shrink(
    returns: pd.DataFrame,
    anchor_annual: float = 0.06,
    intensity: Optional[float] = None) -> pd.Series:
    """Return JS-shrunk daily mean per ticker.

    Args:
        returns: Daily returns DataFrame, one column per ticker.
            NaNs allowed; per-column means drop NaNs.
        anchor_annual: Annual target return for the anchor
            (e.g. 0.06 for a 6% equity risk premium).
        intensity: Fixed shrinkage factor in [0, 1]. If None,
            compute the JS-optimal value from the cross-section.

    Returns:
        pd.Series indexed by ticker with shrunk daily means.

    Raises:
        ValueError: When ``returns`` has fewer than two columns.
    """
    p = returns.shape[1]
    if p < 2:
        raise ValueError(
            f'james_stein_shrink: need at least 2 columns, '
            f'got {p}.')
    anchor_daily = anchor_annual / trading_days_per_year
    means = returns.mean()
    scales = returns.std()

    if intensity is None:
        ss = float(((means - anchor_daily) ** 2).sum())
        if p < 3 or ss < 1e-15:
            b_js = 1.0
        else:
            n_obs = max(
                int(returns.notna().sum().mean()), 1)
            var_pool = float((scales ** 2).mean()) / n_obs
            b_js = max(0.0, 1.0 - (p - 2) * var_pool / ss)
    else:
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(
                f'james_stein_shrink: intensity must be in '
                f'[0, 1], got {intensity}.')
        b_js = float(intensity)

    shrunk = anchor_daily + b_js * (means - anchor_daily)
    return pd.Series(shrunk, index=returns.columns)
```

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/modelling/monte_carlo/test_shrinkage.py -v`
Expected: 5 PASS.

- [ ] **Step 5: NO commit.**

---

## Task 4: TransactionCostModel + IBKR defaults

**Files:**
- Create: `src/modelling/monte_carlo/cost_model.py`
- Test: `tests/modelling/monte_carlo/test_cost_model.py`

**Why:** Spec §5 specifies a frozen `TransactionCostModel` with IBKR Swiss-resident defaults. The dataclass is small but worth a dedicated module because the engine reaches for it on every rebalance step.

**Class spec:**

```python
@dataclass(frozen=True)
class TransactionCostModel:
    commission_bps: float = 7.0
    commission_min_chf: float = 1.5
    spread_bps: float = 5.0
    stamp_duty_bps_swiss: float = 7.5
    stamp_duty_bps_foreign: float = 15.0
    fx_cost_bps: float = 2.0

    def cost_chf(
        self,
        trade_value_chf: float,
        ticker: str,
        currency: str) -> float: ...


def ibkr_default_cost_model() -> TransactionCostModel: ...
```

`cost_chf` returns commission + spread + stamp + fx as defined in spec §5. Swiss stamp applies if ticker ends with `.SW`; otherwise foreign stamp. FX cost applies if `currency != 'CHF'`.

- [ ] **Step 1: Write failing tests**

Create `tests/modelling/monte_carlo/test_cost_model.py`:

```python
"""Tests for TransactionCostModel."""

import pytest

from src.modelling.monte_carlo.cost_model import (
    TransactionCostModel, ibkr_default_cost_model)


def test_default_factory_returns_dataclass_with_spec_values():
    m = ibkr_default_cost_model()
    assert isinstance(m, TransactionCostModel)
    assert m.commission_bps == 7.0
    assert m.commission_min_chf == 1.5
    assert m.spread_bps == 5.0
    assert m.stamp_duty_bps_swiss == 7.5
    assert m.stamp_duty_bps_foreign == 15.0
    assert m.fx_cost_bps == 2.0


def test_swiss_ticker_chf_no_fx_cost():
    """NESN.SW (CHF) -> no FX cost."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(1000.0, 'NESN.SW', 'CHF')
    # commission = max(1000 * 7/1e4, 1.5) = 1.5
    # spread = 1000 * 5/1e4 = 0.5
    # stamp = 1000 * 7.5/1e4 = 0.75
    # fx = 0
    assert cost == pytest.approx(1.5 + 0.5 + 0.75)


def test_us_ticker_usd_includes_fx_cost():
    """AAPL (USD) -> foreign stamp + FX cost."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(1000.0, 'AAPL', 'USD')
    # commission = max(1000 * 7/1e4, 1.5) = 1.5
    # spread = 0.5
    # stamp = 1000 * 15/1e4 = 1.5
    # fx = 1000 * 2/1e4 = 0.2
    assert cost == pytest.approx(1.5 + 0.5 + 1.5 + 0.2)


def test_small_trade_hits_commission_floor():
    """A 10 CHF trade pays the 1.5 CHF commission floor."""
    m = ibkr_default_cost_model()
    cost = m.cost_chf(10.0, 'NESN.SW', 'CHF')
    # commission_proportional = 10 * 7/1e4 = 0.007 -> floored at 1.5
    # spread = 10 * 5/1e4 = 0.005
    # stamp = 10 * 7.5/1e4 = 0.0075
    assert cost == pytest.approx(1.5 + 0.005 + 0.0075)


def test_dataclass_is_frozen():
    m = ibkr_default_cost_model()
    with pytest.raises(Exception):
        m.commission_bps = 99.0


def test_custom_construction():
    """Override defaults to model a different broker."""
    m = TransactionCostModel(
        commission_bps=10.0, commission_min_chf=2.0,
        spread_bps=3.0, stamp_duty_bps_swiss=7.5,
        stamp_duty_bps_foreign=15.0, fx_cost_bps=0.0)
    assert m.commission_bps == 10.0
    cost = m.cost_chf(1000.0, 'NESN.SW', 'CHF')
    # commission = max(1.0, 2.0) = 2.0; spread = 0.3; stamp = 0.75; fx = 0
    assert cost == pytest.approx(2.0 + 0.3 + 0.75)
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m pytest tests/modelling/monte_carlo/test_cost_model.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/modelling/monte_carlo/cost_model.py`:

```python
"""TransactionCostModel + IBKR Swiss-resident defaults.

Implements spec §5. The model returns a CHF cost for a trade
of the given trade_value_chf, ticker (used to detect Swiss
listings via the .SW suffix), and currency (used to detect
non-CHF FX cost). Used by MonteCarloEngine on every rebalance
step when the engine is configured with a cost_model.
"""

from dataclasses import dataclass


bps_per_unit = 1e4
swiss_suffix = '.SW'
chf_currency = 'CHF'


@dataclass(frozen=True)
class TransactionCostModel:
    """IBKR Swiss-resident trade cost model.

    Attributes:
        commission_bps: Per-trade commission in basis points.
        commission_min_chf: Minimum per-trade commission.
        spread_bps: Half-spread cost in basis points.
        stamp_duty_bps_swiss: Swiss federal stamp on CH
            securities (ticker ending in '.SW').
        stamp_duty_bps_foreign: Swiss federal stamp on
            foreign securities.
        fx_cost_bps: FX conversion cost in basis points when
            the trade currency is not CHF.
    """

    commission_bps: float = 7.0
    commission_min_chf: float = 1.5
    spread_bps: float = 5.0
    stamp_duty_bps_swiss: float = 7.5
    stamp_duty_bps_foreign: float = 15.0
    fx_cost_bps: float = 2.0

    def cost_chf(
        self,
        trade_value_chf: float,
        ticker: str,
        currency: str) -> float:
        """CHF cost of a trade.

        Args:
            trade_value_chf: Notional in CHF. Must be
                non-negative; sign is irrelevant (cost is
                applied symmetrically to buys and sells).
            ticker: Exchange symbol; used to detect Swiss
                listings via the '.SW' suffix.
            currency: ISO-4217 listing currency. 'CHF' suppresses
                the FX cost component.

        Returns:
            Total CHF cost = commission + spread + stamp + fx.
        """
        notional = abs(float(trade_value_chf))
        commission = max(
            notional * self.commission_bps / bps_per_unit,
            self.commission_min_chf)
        spread = notional * self.spread_bps / bps_per_unit
        if ticker.endswith(swiss_suffix):
            stamp_bps = self.stamp_duty_bps_swiss
        else:
            stamp_bps = self.stamp_duty_bps_foreign
        stamp = notional * stamp_bps / bps_per_unit
        fx = 0.0
        if currency != chf_currency:
            fx = notional * self.fx_cost_bps / bps_per_unit
        return commission + spread + stamp + fx


def ibkr_default_cost_model() -> TransactionCostModel:
    """IBKR defaults for a Swiss tax-resident retail investor.

    Sensible across the CHF 200-2000 trade size range.

    Returns:
        TransactionCostModel with the dataclass field defaults.
    """
    return TransactionCostModel()
```

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/modelling/monte_carlo/test_cost_model.py -v`
Expected: 6 PASS.

- [ ] **Step 5: NO commit.**

---

## Task 5: `MonteCarloEngine` orchestrator

**Files:**
- Create: `src/modelling/monte_carlo/engine.py`
- Test: `tests/modelling/monte_carlo/test_engine.py`

**Why:** The orchestrator. Ties together Student-t marginal fitting (legacy lift), Clayton copula sampling (Task 2), JS shrinkage (Task 3), and per-step transaction cost accounting (Task 4) to produce a `PathDistribution` (Task 1) from a Phase-4 `RebalancePlan`.

**Class spec:**

```python
class MonteCarloEngine:
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
        cash_rate_annual: float = 0.015): ...

    def simulate(
        self,
        plan: RebalancePlan,
        prices: pd.DataFrame,
        horizon_days: int,
        initial_nav_chf: float) -> PathDistribution: ...
```

**Behavior summary:**
1. Read `plan.target_weights`. Split into `equity_weights` (non-CASH) and `cash_weight` (target_weights.get('CASH', 0.0)).
2. Fit Student-t marginals (df, loc, scale) per equity ticker from `prices`. Use `scipy.stats.t.fit` on daily pct changes. Clamp `df` to `[df_floor, df_ceil]` for numerical safety.
3. If `copula_theta is None`, infer from median pairwise Kendall tau via `theta = 2 * tau / (1 - tau)`, clamped to `[theta_floor, theta_ceil]`. Otherwise use the passed value.
4. Shrink loc parameters using `james_stein_shrink` with the configured `anchor_annual` and `shrinkage_intensity`.
5. Per-step loop (`horizon_days` iterations):
   - Sample Clayton-correlated uniforms (`sample_clayton`).
   - Convert to log-return draws via per-ticker inverse Student-t CDF (`scipy.stats.t.ppf`).
   - Apply to holdings; cash earns `cash_rate_annual / 252`.
   - On rebalance days (per `rebalance_frequency`): add `monthly_contribution_chf` to cash (when frequency is 'monthly' or 'threshold'; quarterly adds 3× the monthly contribution at quarter boundaries), then rebalance to target weights. Compute total trade value as sum of abs(weight_delta * NAV). Deduct `cost_model.cost_chf(...)` per non-CASH ticker leg if `cost_model is not None`. Accumulate the per-path cost.
6. At t=0: deduct cost of the initial plan execution from the cash bucket. Use the absolute target trade value (NAV * sum(equity weights)) split equally across equity tickers as an approximation, or — cleaner — call `cost_model.cost_chf` with the per-ticker trade value `target_w * NAV` for each non-CASH entry. (Choose the latter; it's more accurate and uses each ticker's currency from `plan.holdings_df` when available, else falls back to `'USD'` with a logged warning.)
7. Return a `PathDistribution(paths, horizon_days, initial_nav_chf, plan_name=plan.strategy_name, generated_at=datetime.now(), cumulative_costs_chf=...)`.

**Rebalance days:**
- `'none'`: never rebalance after t=0.
- `'monthly'`: day indices `21, 42, 63, ...` (i.e. every 21 trading days).
- `'quarterly'`: day indices `63, 126, ...`.
- `'threshold'`: each day, compute `max(abs(current_weights - target_weights))`. If exceeds `drift_threshold_pct / 100`, rebalance.

**Currency lookup for cost model:**
- The engine accepts an optional `ticker_currencies: dict[str, str]` constructor kwarg. Default `None` means all non-CHF tickers are treated as `'USD'`. The CLI populates this from `plan.holdings_df` and the candidate JSON (the holdings frame has a `'currency'` column; candidate JSON has `'currency'` per row).

(Document the cleaner-but-simple approach in the docstring; do not let this knob multiply.)

- [ ] **Step 1: Write failing tests**

Create `tests/modelling/monte_carlo/test_engine.py`:

```python
"""Tests for MonteCarloEngine."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.modelling.monte_carlo.cost_model import (
    ibkr_default_cost_model)
from src.modelling.monte_carlo.engine import MonteCarloEngine
from src.modelling.monte_carlo.path_distribution import (
    PathDistribution)
from src.modelling.rebalancing.action import RebalancingAction
from src.modelling.rebalancing.plan import RebalancePlan
from src.shared.constraints import default_constraints


def _plan(target_weights, strategy_name='test'):
    """Build a RebalancePlan with synthetic HOLDs."""
    actions = []
    holdings_rows = []
    for t, w in target_weights.items():
        if t == 'CASH':
            actions.append(RebalancingAction(
                ticker='CASH', action='CASH', shares=0,
                est_cost_chf=w * 10000.0,
                current_wt_pct=0.0,
                target_wt_pct=w * 100.0, note='Cash reserve'))
        else:
            actions.append(RebalancingAction(
                ticker=t, action='HOLD', shares=0,
                est_cost_chf=0.0,
                current_wt_pct=w * 100.0,
                target_wt_pct=w * 100.0, note=''))
            holdings_rows.append({
                'ticker': t, 'value_chf': w * 10000.0,
                'weight_pct': w * 100.0, 'category': 'core'})
    holdings_df = pd.DataFrame(holdings_rows)
    return RebalancePlan(
        actions=actions, target_weights=target_weights,
        strategy_name=strategy_name,
        constraints=default_constraints(),
        holdings_df=holdings_df,
        generated_at=datetime.now())


def _prices(tickers, n_days=400, seed=0):
    """Daily prices DataFrame with synthetic random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n_days, freq='B')
    cols = {}
    for t in tickers:
        increments = rng.normal(0.0005, 0.012, n_days)
        cols[t] = 100.0 * np.exp(np.cumsum(increments))
    return pd.DataFrame(cols, index=idx)


def test_simulate_returns_path_distribution():
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    engine = MonteCarloEngine(n_paths=50, random_seed=7)
    dist = engine.simulate(
        plan, prices, horizon_days=10, initial_nav_chf=10000.0)
    assert isinstance(dist, PathDistribution)
    assert dist.paths.shape == (50, 11)
    assert dist.plan_name == 'test'
    assert dist.initial_nav_chf == 10000.0
    # First column equals initial NAV for every path.
    np.testing.assert_allclose(dist.paths[:, 0], 10000.0)


def test_simulate_cash_only_paths_drift_at_cash_rate():
    """100% CASH allocation produces deterministic paths."""
    plan = _plan({'CASH': 1.0})
    prices = _prices(['A'])  # ignored
    engine = MonteCarloEngine(
        n_paths=10, random_seed=1, cash_rate_annual=0.0252)
    dist = engine.simulate(
        plan, prices, horizon_days=10, initial_nav_chf=10000.0)
    # 0.0252/252 = 0.0001 daily -> after 10 days ~ 10010.0.
    expected_terminal = 10000.0 * (1.0 + 0.0001) ** 10
    np.testing.assert_allclose(
        dist.paths[:, -1], expected_terminal, rtol=1e-9)


def test_simulate_costless_run_has_none_costs():
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    engine = MonteCarloEngine(
        n_paths=20, random_seed=2, cost_model=None)
    dist = engine.simulate(
        plan, prices, horizon_days=21, initial_nav_chf=10000.0)
    assert dist.cumulative_costs_chf is None


def test_simulate_with_costs_attaches_positive_cost_array():
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    engine = MonteCarloEngine(
        n_paths=20, random_seed=2,
        cost_model=ibkr_default_cost_model(),
        rebalance_frequency='monthly')
    dist = engine.simulate(
        plan, prices, horizon_days=63, initial_nav_chf=10000.0)
    assert dist.cumulative_costs_chf is not None
    assert dist.cumulative_costs_chf.shape == (20,)
    assert (dist.cumulative_costs_chf > 0).all()


def test_simulate_costs_reduce_terminal_nav():
    """Same seed: --no-costs > with-costs at terminal."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    seed = 11
    engine_no = MonteCarloEngine(
        n_paths=200, random_seed=seed, cost_model=None,
        rebalance_frequency='monthly')
    engine_yes = MonteCarloEngine(
        n_paths=200, random_seed=seed,
        cost_model=ibkr_default_cost_model(),
        rebalance_frequency='monthly')
    dist_no = engine_no.simulate(
        plan, prices, horizon_days=126,
        initial_nav_chf=10000.0)
    dist_yes = engine_yes.simulate(
        plan, prices, horizon_days=126,
        initial_nav_chf=10000.0)
    # Mean terminal NAV strictly lower with costs.
    assert dist_yes.paths[:, -1].mean() < dist_no.paths[:, -1].mean()


def test_simulate_raises_when_ticker_missing_from_prices():
    plan = _plan({'A': 0.5, 'GHOST': 0.5})
    prices = _prices(['A'])
    engine = MonteCarloEngine(n_paths=10, random_seed=3)
    with pytest.raises(RuntimeError, match='GHOST'):
        engine.simulate(
            plan, prices, horizon_days=5,
            initial_nav_chf=10000.0)


def test_simulate_random_seed_reproducible():
    """Same seed yields identical paths."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    e1 = MonteCarloEngine(n_paths=30, random_seed=99)
    e2 = MonteCarloEngine(n_paths=30, random_seed=99)
    d1 = e1.simulate(
        plan, prices, horizon_days=15, initial_nav_chf=10000.0)
    d2 = e2.simulate(
        plan, prices, horizon_days=15, initial_nav_chf=10000.0)
    np.testing.assert_array_equal(d1.paths, d2.paths)


def test_simulate_monthly_contribution_grows_paths():
    """Positive contribution increases mean terminal NAV."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    seed = 7
    e_no = MonteCarloEngine(
        n_paths=100, random_seed=seed,
        rebalance_frequency='monthly',
        monthly_contribution_chf=0.0)
    e_yes = MonteCarloEngine(
        n_paths=100, random_seed=seed,
        rebalance_frequency='monthly',
        monthly_contribution_chf=500.0)
    d_no = e_no.simulate(
        plan, prices, horizon_days=126,
        initial_nav_chf=10000.0)
    d_yes = e_yes.simulate(
        plan, prices, horizon_days=126,
        initial_nav_chf=10000.0)
    assert d_yes.paths[:, -1].mean() > d_no.paths[:, -1].mean()


def test_simulate_threshold_rebalance_triggers_on_drift():
    """Threshold mode rebalances when a weight drifts > threshold."""
    plan = _plan({'A': 0.5, 'B': 0.5})
    prices = _prices(['A', 'B'])
    engine = MonteCarloEngine(
        n_paths=20, random_seed=8,
        rebalance_frequency='threshold',
        drift_threshold_pct=1.0,  # tiny threshold -> frequent rebal
        cost_model=ibkr_default_cost_model())
    dist = engine.simulate(
        plan, prices, horizon_days=63,
        initial_nav_chf=10000.0)
    # Frequent rebalances should accumulate noticeable costs.
    assert dist.cumulative_costs_chf is not None
    assert dist.cumulative_costs_chf.mean() > 0.0
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m pytest tests/modelling/monte_carlo/test_engine.py -v`
Expected: ImportError on `MonteCarloEngine`.

- [ ] **Step 3: Implement**

Create `src/modelling/monte_carlo/engine.py`. The implementer should follow the legacy module's marginal-fitting + Marshall-Olkin sampling skeleton but plug in the new pure helpers (`sample_clayton`, `james_stein_shrink`) and emit a `PathDistribution`. Key skeleton:

```python
"""Monte Carlo engine for portfolio path simulation.

Consumes a Phase-4 RebalancePlan + daily prices panel and
produces a PathDistribution. Uses Clayton copula joint
sampling, per-ticker Student-t marginals (with James-Stein
shrunk loc), deterministic cash drift, and optional
IBKR-style transaction costs on initial execution + every
rebalance step.
"""

from datetime import datetime
from typing import Dict, Optional

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
default_currency = 'USD'

valid_rebalance_frequencies = (
    'none', 'monthly', 'quarterly', 'threshold')


class MonteCarloEngine:
    """Run Clayton-copula Student-t Monte Carlo on a plan.

    Attributes are the constructor kwargs of the same name.
    """

    def __init__(
        self,
        n_paths: int = 10000,
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
        if rebalance_frequency not in valid_rebalance_frequencies:
            raise ValueError(
                f'rebalance_frequency must be one of '
                f'{valid_rebalance_frequencies}, got '
                f'{rebalance_frequency!r}.')
        if n_paths < 1:
            raise ValueError(
                f'n_paths must be >= 1, got {n_paths}.')
        self.n_paths = n_paths
        self.copula_theta = copula_theta
        self.shrinkage_intensity = shrinkage_intensity
        self.anchor_annual = anchor_annual
        self.cost_model = cost_model
        self.rebalance_frequency = rebalance_frequency
        self.monthly_contribution_chf = monthly_contribution_chf
        self.drift_threshold_pct = drift_threshold_pct
        self.random_seed = random_seed
        self.cash_rate_annual = cash_rate_annual
        self.ticker_currencies = ticker_currencies or {}

    def simulate(
        self,
        plan: RebalancePlan,
        prices: pd.DataFrame,
        horizon_days: int,
        initial_nav_chf: float) -> PathDistribution:
        """Simulate forward NAV paths under target_weights.

        Args:
            plan: RebalancePlan whose target_weights drive the
                allocation. CASH entries become a deterministic
                cash-rate bucket; other tickers are simulated.
            prices: Daily price DataFrame for every non-CASH
                ticker in plan.target_weights. Index =
                trading days; columns = tickers.
            horizon_days: Number of trading days to simulate.
            initial_nav_chf: Starting NAV.

        Returns:
            PathDistribution with paths shape
            (n_paths, horizon_days+1) and cumulative_costs_chf
            populated when cost_model is not None.

        Raises:
            RuntimeError: When any non-CASH ticker in
                plan.target_weights is missing from prices,
                or when horizon_days < 1.
        """
        if horizon_days < 1:
            raise RuntimeError(
                f'simulate: horizon_days must be >= 1, '
                f'got {horizon_days}.')

        equity_tickers = [
            t for t in plan.target_weights
            if t != cash_ticker]
        missing = [
            t for t in equity_tickers if t not in prices.columns]
        if missing:
            raise RuntimeError(
                f'MonteCarloEngine: prices panel missing '
                f'columns for ticker(s) {sorted(missing)}.')

        equity_weights = np.array(
            [plan.target_weights[t] for t in equity_tickers],
            dtype=float)
        cash_weight = float(
            plan.target_weights.get(cash_ticker, 0.0))

        rng = np.random.default_rng(self.random_seed)

        # Fit marginals + JS shrinkage + Clayton theta.
        # ... (lift from legacy fit_marginals / fit_clayton_theta
        # / shrink_expected_returns but use james_stein_shrink
        # from shrinkage.py for the shrinkage).

        # Per-step loop. See spec.

        # Build PathDistribution and return.
        ...
```

Full implementation should:
- Use `prices.pct_change().dropna()` to derive a per-ticker daily returns DataFrame.
- For each equity ticker call `scipy.stats.t.fit(returns[t].dropna())` and clamp `df` to `[df_floor, df_ceil]`. Fall back to `fallback_marginal` (a near-normal default) if `<60` observations.
- Call `james_stein_shrink(returns_df, anchor_annual=self.anchor_annual, intensity=self.shrinkage_intensity)` to get shrunk per-ticker daily mean. Replace the `loc` component of each marginal with the shrunk value.
- If `self.copula_theta is None`, compute theta from median pairwise Kendall tau on the returns DataFrame. Else use the passed value. Clamp final theta to `[theta_floor, theta_ceil]`.
- Initial-execution cost: `total_t0_cost_chf = sum(cost_model.cost_chf(target_w * initial_nav_chf, t, currency_for(t)) for t, target_w in equity entries)`. Subtract from cash bucket (cash_weight * initial_nav_chf) — if it drives cash negative, allow it (the path effectively starts with sub-initial NAV).
- Each step:
  - Sample `u = sample_clayton(theta, n_paths, n_assets, rng)`.
  - For each ticker: `returns[:, k] = t_dist.ppf(u[:, k], dfs[k], loc=locs[k], scale=scales[k])`.
  - `holdings *= 1 + returns`. `cash *= 1 + cash_daily_rate`.
  - If rebalance day: add monthly contribution to cash (per frequency), then rebalance to target weights, deducting per-leg cost.
  - Record NAV.
- Return a `PathDistribution`.

The implementer is expected to write the full body; the skeleton above plus the legacy module reference are sufficient.

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/modelling/monte_carlo/test_engine.py -v`
Expected: 9 PASS.

Run: `python3 -m pytest tests/ -v 2>&1 | tail -3`
Expected: full suite green.

- [ ] **Step 5: NO commit.**

---

## Task 6: Package `__init__.py` exports

**Files:**
- Modify: `src/modelling/monte_carlo/__init__.py`

**Why:** Once Tasks 1-5 land, expose the public surface for the CLI and any future consumers.

- [ ] **Step 1: Replace `__init__.py`**

```python
"""Monte Carlo simulation domain.

Public surface:
    PathDistribution
    MonteCarloEngine
    TransactionCostModel, ibkr_default_cost_model
    sample_clayton (low-level)
    james_stein_shrink (low-level)
"""

from src.modelling.monte_carlo.copula import sample_clayton
from src.modelling.monte_carlo.cost_model import (
    TransactionCostModel, ibkr_default_cost_model)
from src.modelling.monte_carlo.engine import MonteCarloEngine
from src.modelling.monte_carlo.path_distribution import (
    PathDistribution)
from src.modelling.monte_carlo.shrinkage import (
    james_stein_shrink)


__all__ = [
    'PathDistribution',
    'MonteCarloEngine',
    'TransactionCostModel',
    'ibkr_default_cost_model',
    'sample_clayton',
    'james_stein_shrink',
]
```

- [ ] **Step 2: Verify full suite still green**

Run: `python3 -m pytest tests/ -v 2>&1 | tail -3`
Expected: all PASS.

- [ ] **Step 3: NO commit.**

---

## Task 7: `run_monte_carlo.py` CLI

**Files:**
- Create: `src/user_scripts/run_monte_carlo.py`
- Test: `tests/user_scripts/test_run_monte_carlo.py`

**Why:** Operator-facing entry point. argparse is acceptable for user_scripts per spec §6 (report.py is the exception that keeps zero-arg main).

**CLI args:**
- `--strategy NAME` (required) — one of the registered strategy names.
- `--horizon-days INT` (default 252).
- `--n-paths INT` (default 5000).
- `--rebalance {none, monthly, quarterly, threshold}` (default `'monthly'`).
- `--monthly-contribution FLOAT` (default 0.0).
- `--no-costs` (flag; default off → IBKR defaults used).
- `--candidates-from PATH` (optional; screener JSON for screener-fed strategies).
- `--new-capital FLOAT` (default 2000.0).
- `--max-weight FLOAT` (default 0.15).
- `--random-seed INT` (default 42).
- `--output PATH` (required; .xlsx).

**Flow:**
1. Build constraints + analyzer + rebalancer (mirroring `report.main_with_inputs`).
2. Compute the plan via `Rebalancer.propose(candidates=...)`.
3. Build `ticker_currencies` from `plan.holdings_df` + `candidates`.
4. Assemble `prices` = `analyzer.get_price_panel()` and extend with any candidate tickers via `analyzer.data_provider.get_price_history(...)` for the past N days (default 1000 trading days, configurable later).
5. Construct `MonteCarloEngine(...)` with the CLI knobs. `--no-costs` → `cost_model=None`; otherwise `ibkr_default_cost_model()`.
6. Call `engine.simulate(plan, prices, args.horizon_days, initial_nav_chf=plan.total_value_chf)`.
7. Write Excel with sheets: `Plan`, `Summary`, `Percentiles`, `TerminalDistribution`. Use the existing `DataExporter`.

- [ ] **Step 1: Write the failing CLI smoke test**

Create `tests/user_scripts/test_run_monte_carlo.py`:

```python
"""Smoke test for run_monte_carlo CLI."""

import json
import sys

import pandas as pd
import pytest

from src.user_scripts import run_monte_carlo


def test_run_monte_carlo_emits_xlsx(tmp_path, monkeypatch):
    """End-to-end: argparse -> engine -> Excel."""

    class FakeAnalyzer:
        def __init__(self, **_):
            self.data_provider = FakeProvider()

        def get_holdings_snapshot(self, categories=None):
            return pd.DataFrame([
                {'ticker': 'A', 'shares': 1,
                 'value_chf': 5000.0, 'weight_pct': 50.0,
                 'currency': 'USD', 'sector': 'X',
                 'price_chf': 100.0, 'category': 'core'},
                {'ticker': 'B', 'shares': 1,
                 'value_chf': 5000.0, 'weight_pct': 50.0,
                 'currency': 'CHF', 'sector': 'X',
                 'price_chf': 100.0, 'category': 'core'}])

        def get_price_panel(self):
            idx = pd.date_range(
                '2024-01-01', periods=400, freq='B')
            return pd.DataFrame({
                'A': 100.0 + range(400),
                'B': 100.0 + range(400)}, index=idx)

    class FakeProvider:
        def get_current_price(self, ticker):
            return 100.0

    monkeypatch.setattr(
        'src.user_scripts.run_monte_carlo.PortfolioAnalyzer',
        FakeAnalyzer)
    monkeypatch.setattr(
        'src.user_scripts.run_monte_carlo._build_data_provider',
        lambda: FakeProvider())

    cat_path = tmp_path / 'cats.json'
    cat_path.write_text(json.dumps({'A': 'core', 'B': 'core'}))
    output_path = tmp_path / 'mc.xlsx'

    argv = [
        '--strategy', 'equal_weight',
        '--horizon-days', '21',
        '--n-paths', '50',
        '--rebalance', 'monthly',
        '--ticker-categories', str(cat_path),
        '--random-seed', '7',
        '--output', str(output_path),
        '--degiro-csv', 'X', '--ibkr-csv', 'Y']

    run_monte_carlo.main(argv)
    assert output_path.exists()
    xl = pd.ExcelFile(output_path)
    assert {'Summary', 'Percentiles'}.issubset(set(xl.sheet_names))


def test_run_monte_carlo_no_costs_flag(tmp_path, monkeypatch):
    """--no-costs disables the cost model."""

    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured['cost_model'] = kwargs.get('cost_model')
            captured['rebalance_frequency'] = kwargs.get(
                'rebalance_frequency')

        def simulate(self, plan, prices, horizon_days,
                     initial_nav_chf):
            from datetime import datetime
            import numpy as np
            from src.modelling.monte_carlo.path_distribution \
                import PathDistribution
            n = 5
            paths = np.full(
                (n, horizon_days + 1), initial_nav_chf)
            return PathDistribution(
                paths=paths, horizon_days=horizon_days,
                initial_nav_chf=initial_nav_chf,
                plan_name=plan.strategy_name,
                generated_at=datetime.now(),
                cumulative_costs_chf=None)

    class FakeAnalyzer:
        def __init__(self, **_):
            self.data_provider = None

        def get_holdings_snapshot(self, categories=None):
            return pd.DataFrame([
                {'ticker': 'A', 'shares': 1,
                 'value_chf': 5000.0, 'weight_pct': 50.0,
                 'currency': 'USD', 'sector': 'X',
                 'price_chf': 100.0, 'category': 'core'},
                {'ticker': 'B', 'shares': 1,
                 'value_chf': 5000.0, 'weight_pct': 50.0,
                 'currency': 'CHF', 'sector': 'X',
                 'price_chf': 100.0, 'category': 'core'}])

        def get_price_panel(self):
            idx = pd.date_range(
                '2024-01-01', periods=300, freq='B')
            return pd.DataFrame({
                'A': 100.0, 'B': 100.0}, index=idx)

    monkeypatch.setattr(
        'src.user_scripts.run_monte_carlo.PortfolioAnalyzer',
        FakeAnalyzer)
    monkeypatch.setattr(
        'src.user_scripts.run_monte_carlo._build_data_provider',
        lambda: None)
    monkeypatch.setattr(
        'src.user_scripts.run_monte_carlo.MonteCarloEngine',
        FakeEngine)

    cat_path = tmp_path / 'cats.json'
    cat_path.write_text(json.dumps({'A': 'core', 'B': 'core'}))
    output_path = tmp_path / 'mc.xlsx'

    argv = [
        '--strategy', 'equal_weight',
        '--horizon-days', '10',
        '--n-paths', '5',
        '--no-costs',
        '--rebalance', 'none',
        '--ticker-categories', str(cat_path),
        '--output', str(output_path),
        '--degiro-csv', 'X', '--ibkr-csv', 'Y']

    run_monte_carlo.main(argv)
    assert captured['cost_model'] is None
    assert captured['rebalance_frequency'] == 'none'
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m pytest tests/user_scripts/test_run_monte_carlo.py -v`
Expected: ImportError on `src.user_scripts.run_monte_carlo`.

- [ ] **Step 3: Implement**

Create `src/user_scripts/run_monte_carlo.py`. Use the existing `_build_data_provider` factory hook (define it in the module for monkeypatching). Mirror `src/user_scripts/run_screener.py` and `src/analysis/report.py` for structure. Wire args → constraints → analyzer → rebalancer → plan → engine → PathDistribution → Excel.

(Concrete body left to the implementer — the test specifies the public contract: `run_monte_carlo.main(argv: list[str])` that takes CLI args and produces an xlsx at the configured path.)

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/user_scripts/test_run_monte_carlo.py -v`
Expected: 2 PASS.

Run: `python3 -m pytest tests/ -v 2>&1 | tail -3`
Expected: full suite green.

- [ ] **Step 5: NO commit.**

---

## Task 8: Delete legacy `src/modelling/monte_carlo.py`

**Files:**
- Delete: `src/modelling/monte_carlo.py`

**Why:** The 907-line monolith is fully replaced by the `monte_carlo/` package. The grep audit at the start of this phase confirmed no live consumers outside docs/plans.

- [ ] **Step 1: Verify no live imports**

Run via Bash:
```
grep -rn 'from src.modelling.monte_carlo import\|from src.modelling import monte_carlo\|import src.modelling.monte_carlo$\|mc_module' src/ tests/ 2>&1
```
Confirm zero hits in `src/` and `tests/`. (Docs may mention it; ignore.)

- [ ] **Step 2: Delete**

```
rm /Users/rbarreira/Desktop/stock_market/src/modelling/monte_carlo.py
```

- [ ] **Step 3: Full suite verification**

Run: `python3 -m pytest tests/ -v 2>&1 | tail -3`
Expected: full green.

- [ ] **Step 4: NO commit.**

---

## Task 9: Manual verification + tasks/todo.md update

- [ ] **Step 1: Suite green check**

Run: `python3 -m pytest tests/ -v`
Expected: every Phase 1-5 test passes.

- [ ] **Step 2: Smoke run (operator-driven)**

Run from the repo root with a real or test-fixture portfolio CSV:
```
python -m src.user_scripts.run_monte_carlo \
    --strategy mvo --horizon-days 252 --n-paths 5000 \
    --rebalance monthly --monthly-contribution 2000 \
    --ticker-categories data/ticker_categories.json \
    --degiro-csv data/postprocess_data/processed_portfolio.csv \
    --ibkr-csv reports/ibkr/<YYYYMMDD>_<YYYYMMDD>_<ACCOUNT_ID>.csv \
    --output results/mc_mvo.xlsx
```
Inspect:
- The `Summary` sheet contains the expected keys (terminal_p5/p50/p95, max_drawdown_p5/p50/p95, prob_loss, cumulative_cost_chf_*, annual_cost_drag_bps).
- The `Percentiles` sheet is a `(horizon_days+1) x 5` band.
- Add `--no-costs` to a second run with the same seed; confirm `cumulative_cost_chf_p50 == 0.0` and the mean terminal NAV strictly exceeds the with-costs run.

Skip this step if no portfolio CSVs are available in the operator's environment; integration tests in Tasks 5 and 7 already verify the contract.

- [ ] **Step 3: Append Phase 5 completion block to `tasks/todo.md`**

Mirror the formatting of Phase 1-4. Include: new package files, deleted legacy file, test count, verification notes, links to spec + plan, and next-phase pointer (Phase 6 — Greeks code-style pass).

- [ ] **Step 4: NO commit.**

---

## Self-Review

**1. Spec coverage (HANDOFF.md §8 Phase 5 deliverables):**

| Spec requirement | Task |
|---|---|
| `PathDistribution` with paths, query methods, Contract 6 | Task 1 |
| `sample_clayton` Clayton copula sampler | Task 2 |
| `james_stein_shrink` shrinkage helper | Task 3 |
| `TransactionCostModel` + `ibkr_default_cost_model()` (IBKR Swiss-resident defaults) | Task 4 |
| `MonteCarloEngine.simulate(plan, prices, horizon_days, initial_nav_chf) -> PathDistribution` with all required kwargs | Task 5 |
| Per-step cost deduction; `--no-costs` strictly higher terminal NAV | Task 5 |
| `run_monte_carlo.py` CLI with strategy/horizon/n_paths/rebalance/contribution args | Task 7 |
| Delete legacy 907-line `monte_carlo.py` | Task 8 |
| Cost-drag fields in summary (`cumulative_cost_chf_p50/p95`, `annual_cost_drag_bps`) | Task 1 (PathDistribution) + Task 5 (engine populates) |

**2. Placeholder scan:**
- Task 5 (engine.py) intentionally leaves the body sketched rather than fully written — it's a substantial numerical implementation that should be lifted from the legacy file with the new helpers wired in. The plan provides the skeleton, the legacy reference, and a complete test suite that pins the contract; the implementer writes the body against those tests. This is the only task where the "complete code in every step" rule is bent, and the bend is justified by the lift-and-shift nature of the work.
- Task 7 (CLI body) is similarly sketched — the contract is in the test, the implementer wires the standard CLI scaffolding plus the engine call.

**3. Type consistency:**
- `PathDistribution(paths, horizon_days, initial_nav_chf, plan_name, generated_at, cumulative_costs_chf)` — used identically in Tasks 1, 5, 7.
- `MonteCarloEngine.simulate(plan: RebalancePlan, prices: pd.DataFrame, horizon_days: int, initial_nav_chf: float) -> PathDistribution` — Task 5 (def), Task 7 (call site).
- `sample_clayton(theta, n_samples, n_assets, rng)` — Task 2 (def), Task 5 (call site).
- `james_stein_shrink(returns, anchor_annual, intensity)` — Task 3 (def), Task 5 (call site).
- `TransactionCostModel.cost_chf(trade_value_chf, ticker, currency)` — Task 4 (def), Task 5 (call site).
- `cash_ticker = 'CASH'` from `src.modelling.rebalancing.action` — Task 5 import.

No drift across tasks.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-14-phase-5-monte-carlo.md`. Two execution options:

1. **Subagent-Driven** (recommended per HANDOFF.md §3) — fresh implementer subagent per task + two-stage review.
2. **Inline Execution** — via `superpowers:executing-plans`, batched checkpoints.

Per the handoff workflow the default is Subagent-Driven via `superpowers:subagent-driven-development`.
