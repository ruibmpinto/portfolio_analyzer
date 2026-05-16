# Phase 2 — Rebalancing domain + rewritten report.py implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean rebalancing domain (`src/modelling/rebalancing/`) exposing four portfolio-only strategies via a strategy registry, plus a `Rebalancer` orchestrator that turns target weights into BUY/REDUCE/HOLD actions; rewrite `src/analysis/report.py` as a thin CLI orchestrator that wires `PortfolioAnalyzer` + `Rebalancer` + `DataExporter` and writes the Excel workbook; delete the legacy 1064-line `src/modelling/rebalancing.py`.

**Architecture:** Mirrors `src/analysis/core/`: data model (`RebalancingAction`) → state aggregator (`RebalancePlan`) → orchestrator (`Rebalancer`). Strategies live in a sibling subpackage as a registry of subclasses of a single `Strategy` ABC; each strategy takes `(holdings_df, prices, constraints, candidates, risk_free_rate)` and returns target weights. Action generation is shared logic on `Rebalancer`, never duplicated per strategy. `report.py` is the only cross-domain entry point; it imports from `src.analysis.core`, `src.modelling.rebalancing`, `src.analysis.loaders`, `src.shared`.

**Tech Stack:** Python 3, pandas, numpy, scipy.optimize.minimize (SLSQP), openpyxl (already used by `DataExporter`), pytest, argparse (stdlib), pathlib (stdlib), json (stdlib).

**Source spec:** `docs/superpowers/specs/2026-05-13-repo-refactor-design.md` Sections 2 (Contracts 4-5), 4 (rebalancing domain detail), 6 (`report.py` rewrite spec), 7 (Phase 2 verification).

**Style:** Match `src/analysis/core/portfolio.py` and `src/analysis/core/analyzer.py`:
- Plain module docstring (no `# ===` banners).
- Imports grouped (stdlib / third-party / local) separated by blank lines, no banner comments.
- No `__author__` / `__credits__` / `__status__` blocks.
- Type hints in signatures.
- Google-style docstrings (`Args:`, `Returns:`, `Raises:`, `Example:`).
- Lowercase module-level constants.
- No `# ~~~~~` subsection separators inside method bodies — use blank lines + lead-in comments.
- 80-char max line width. Single quotes for strings, double quotes for docstrings only.

**Commit policy:** This plan describes commit boundaries. Per `~/.claude/CLAUDE.md`, only commit when the user requests. Commit steps in tasks are advisory.

---

## File map

**New files (10):**
- `src/modelling/rebalancing/__init__.py` — re-exports public surface.
- `src/modelling/rebalancing/action.py` — `RebalancingAction` dataclass (Contract 5).
- `src/modelling/rebalancing/plan.py` — `RebalancePlan` class (Contract 5).
- `src/modelling/rebalancing/rebalancer.py` — `Rebalancer` orchestrator.
- `src/modelling/rebalancing/strategies/__init__.py` — `strategy_registry` + `get_strategy()`.
- `src/modelling/rebalancing/strategies/base.py` — `Strategy` ABC.
- `src/modelling/rebalancing/strategies/mvo.py` — `MVOStrategy` (max Sharpe, SLSQP).
- `src/modelling/rebalancing/strategies/equal_weight.py` — `EqualWeightStrategy`.
- `src/modelling/rebalancing/strategies/inverse_vol.py` — `InverseVolStrategy`.
- `src/modelling/rebalancing/strategies/min_variance.py` — `MinVarianceStrategy`.

**New test files (4):**
- `tests/modelling/__init__.py` (empty marker)
- `tests/modelling/rebalancing/__init__.py` (empty marker)
- `tests/modelling/rebalancing/test_action_and_plan.py`
- `tests/modelling/rebalancing/test_strategies.py`
- `tests/modelling/rebalancing/test_rebalancer.py`
- `tests/analysis/test_report.py` — integration test for the CLI orchestrator.

**Rewritten files (1):**
- `src/analysis/report.py` — thin orchestrator + CLI entry point.

**Deleted files (1):**
- `src/modelling/rebalancing.py` (1064-line legacy with hardcoded holdings + scenarios).

**Modified files (1):**
- `tests/conftest.py` — add a `holdings_df` fixture and a `prices_panel` fixture for strategy tests (so they don't all rebuild the same input).

---

## Task 1 — `RebalancingAction` dataclass

**Files:**
- Create: `tests/modelling/__init__.py` (empty)
- Create: `tests/modelling/rebalancing/__init__.py` (empty)
- Create: `tests/modelling/rebalancing/test_action_and_plan.py`
- Create: `src/modelling/rebalancing/__init__.py` (empty for now; re-exports added in Task 8)
- Create: `src/modelling/rebalancing/action.py`

- [ ] **Step 1: Create the directory tree + empty markers**

```bash
mkdir -p /Users/rbarreira/Desktop/stock_market/tests/modelling/rebalancing
mkdir -p /Users/rbarreira/Desktop/stock_market/src/modelling/rebalancing/strategies
touch /Users/rbarreira/Desktop/stock_market/tests/modelling/__init__.py
touch /Users/rbarreira/Desktop/stock_market/tests/modelling/rebalancing/__init__.py
touch /Users/rbarreira/Desktop/stock_market/src/modelling/rebalancing/__init__.py
touch /Users/rbarreira/Desktop/stock_market/src/modelling/rebalancing/strategies/__init__.py
```

- [ ] **Step 2: Write `tests/modelling/rebalancing/test_action_and_plan.py` with the action tests**

```python
"""Tests for RebalancingAction and RebalancePlan."""

import dataclasses

import pytest

from src.modelling.rebalancing.action import RebalancingAction


def test_action_construct_buy():
    a = RebalancingAction(
        ticker='AMZN', action='BUY', shares=3,
        est_cost_chf=450.0, current_wt_pct=2.0,
        target_wt_pct=5.0, note='')
    assert a.ticker == 'AMZN'
    assert a.action == 'BUY'
    assert a.shares == 3
    assert a.est_cost_chf == 450.0


def test_action_is_frozen():
    a = RebalancingAction(
        ticker='AMZN', action='HOLD', shares=0,
        est_cost_chf=0.0, current_wt_pct=4.5,
        target_wt_pct=5.0, note='')
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.action = 'BUY'


def test_action_rejects_unknown_action():
    """Action must be one of BUY/REDUCE/HOLD."""
    with pytest.raises(ValueError, match='action'):
        RebalancingAction(
            ticker='AMZN', action='SHORT', shares=1,
            est_cost_chf=10.0, current_wt_pct=1.0,
            target_wt_pct=2.0, note='')


def test_action_rejects_negative_cost():
    """est_cost_chf must be >= 0 (sign carried by `action`)."""
    with pytest.raises(ValueError, match='est_cost_chf'):
        RebalancingAction(
            ticker='AMZN', action='BUY', shares=1,
            est_cost_chf=-50.0, current_wt_pct=1.0,
            target_wt_pct=2.0, note='')


def test_hold_must_have_zero_shares():
    """HOLD action: shares must be 0 (sanity check)."""
    with pytest.raises(ValueError, match='HOLD'):
        RebalancingAction(
            ticker='AMZN', action='HOLD', shares=5,
            est_cost_chf=0.0, current_wt_pct=4.5,
            target_wt_pct=5.0, note='')
```

- [ ] **Step 3: Run tests, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_action_and_plan.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.modelling.rebalancing.action'`.

- [ ] **Step 4: Implement `src/modelling/rebalancing/action.py`**

```python
"""
Rebalancing action data model.

Implements Contract 5 of the refactor design spec. One
RebalancingAction = one row in the rebalance plan: what to do
(BUY/REDUCE/HOLD), for which ticker, and at what cost.

Validation enforces the action vocabulary, non-negative cost
magnitude (sign is carried by the `action` field), and the
HOLD-must-be-zero-shares invariant.
"""

from dataclasses import dataclass


valid_actions = ('BUY', 'REDUCE', 'HOLD')


@dataclass(frozen=True)
class RebalancingAction:
    """
    Single row of a rebalance plan.

    Attributes:
        ticker: Stock or ETF ticker symbol.
        action: One of 'BUY', 'REDUCE', 'HOLD'.
        shares: Number of shares affected. 0 for HOLD; positive
            for BUY (shares to add); positive for REDUCE
            (shares to drop). Always non-negative — direction
            is carried by `action`.
        est_cost_chf: Magnitude of CHF cash flow. 0 for HOLD;
            positive for BUY and REDUCE. Always non-negative.
        current_wt_pct: Position's weight before the action,
            in percent.
        target_wt_pct: Position's target weight after the
            rebalance, in percent.
        note: Free-form annotation (e.g.
            'Only if held >6 months' for REDUCE actions).
    """

    ticker: str
    action: str
    shares: int
    est_cost_chf: float
    current_wt_pct: float
    target_wt_pct: float
    note: str

    def __post_init__(self):
        if self.action not in valid_actions:
            raise ValueError(
                f'Unknown action {self.action!r}; must be one '
                f'of {valid_actions}.')
        if self.shares < 0:
            raise ValueError(
                f'shares must be non-negative '
                f'(got {self.shares}); direction is carried '
                f'by `action`.')
        if self.est_cost_chf < 0:
            raise ValueError(
                f'est_cost_chf must be non-negative '
                f'(got {self.est_cost_chf}); magnitude only.')
        if self.action == 'HOLD' and self.shares != 0:
            raise ValueError(
                f'HOLD action must have shares=0, '
                f'got {self.shares}.')
```

- [ ] **Step 5: Run tests, confirm 5 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_action_and_plan.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit (advisory)**

```bash
git add src/modelling/rebalancing/__init__.py \
        src/modelling/rebalancing/action.py \
        src/modelling/rebalancing/strategies/__init__.py \
        tests/modelling/__init__.py \
        tests/modelling/rebalancing/__init__.py \
        tests/modelling/rebalancing/test_action_and_plan.py
git commit -m "feat(rebalancing): add RebalancingAction dataclass per Contract 5"
```

---

## Task 2 — `RebalancePlan` class

**Files:**
- Modify: `tests/modelling/rebalancing/test_action_and_plan.py` — append plan tests.
- Create: `src/modelling/rebalancing/plan.py`

- [ ] **Step 1: Append plan tests to the existing file**

```python


from datetime import datetime

import pandas as pd

from src.modelling.rebalancing.plan import RebalancePlan
from src.shared.constraints import default_constraints


def _sample_actions():
    return [
        RebalancingAction(
            ticker='AMZN', action='BUY', shares=3,
            est_cost_chf=450.0, current_wt_pct=2.0,
            target_wt_pct=5.0, note=''),
        RebalancingAction(
            ticker='UBSG.SW', action='REDUCE', shares=1,
            est_cost_chf=25.0, current_wt_pct=8.0,
            target_wt_pct=5.0, note='Only if held >6 months'),
        RebalancingAction(
            ticker='HOLN.SW', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=15.0,
            target_wt_pct=15.0, note=''),
    ]


def _sample_holdings_df():
    return pd.DataFrame([
        {'ticker': 'AMZN', 'shares': 5, 'value_chf': 600.0,
         'weight_pct': 2.0, 'currency': 'USD',
         'sector': 'Consumer Cyclical', 'price_chf': 120.0,
         'category': 'core'},
        {'ticker': 'UBSG.SW', 'shares': 4, 'value_chf': 100.0,
         'weight_pct': 8.0, 'currency': 'CHF',
         'sector': 'Financial Services', 'price_chf': 25.0,
         'category': 'core'},
        {'ticker': 'HOLN.SW', 'shares': 2, 'value_chf': 140.0,
         'weight_pct': 15.0, 'currency': 'CHF',
         'sector': 'Basic Materials', 'price_chf': 70.0,
         'category': 'core'},
    ])


def test_plan_construct():
    plan = RebalancePlan(
        actions=_sample_actions(),
        target_weights={'AMZN': 0.05, 'UBSG.SW': 0.05,
                        'HOLN.SW': 0.15},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=_sample_holdings_df(),
        generated_at=datetime(2026, 5, 14, 10, 0))
    assert plan.strategy_name == 'mvo'
    assert len(plan.actions) == 3


def test_plan_to_dataframe_columns():
    plan = RebalancePlan(
        actions=_sample_actions(),
        target_weights={'AMZN': 0.05, 'UBSG.SW': 0.05,
                        'HOLN.SW': 0.15},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=_sample_holdings_df(),
        generated_at=datetime(2026, 5, 14, 10, 0))
    df = plan.to_dataframe()
    expected = {
        'ticker', 'action', 'shares', 'est_cost_chf',
        'current_wt_pct', 'target_wt_pct', 'note'}
    assert set(df.columns) == expected
    assert len(df) == 3


def test_plan_target_weights_dataframe_columns():
    plan = RebalancePlan(
        actions=_sample_actions(),
        target_weights={'AMZN': 0.05, 'UBSG.SW': 0.05,
                        'HOLN.SW': 0.15},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=_sample_holdings_df(),
        generated_at=datetime(2026, 5, 14, 10, 0))
    df = plan.target_weights_dataframe()
    expected = {
        'ticker', 'current_wt_pct', 'target_wt_pct',
        'deviation_pct'}
    assert set(df.columns) == expected


def test_plan_summary_aggregates():
    plan = RebalancePlan(
        actions=_sample_actions(),
        target_weights={'AMZN': 0.05, 'UBSG.SW': 0.05,
                        'HOLN.SW': 0.15},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=_sample_holdings_df(),
        generated_at=datetime(2026, 5, 14, 10, 0))
    s = plan.summary()
    assert s['n_buy'] == 1
    assert s['n_reduce'] == 1
    assert s['n_hold'] == 1
    assert s['total_buy_chf'] == 450.0
    assert s['total_reduce_chf'] == 25.0


def test_plan_total_value_chf_property():
    plan = RebalancePlan(
        actions=_sample_actions(),
        target_weights={'AMZN': 0.05, 'UBSG.SW': 0.05,
                        'HOLN.SW': 0.15},
        strategy_name='mvo',
        constraints=default_constraints(),
        holdings_df=_sample_holdings_df(),
        generated_at=datetime(2026, 5, 14, 10, 0))
    assert plan.total_value_chf == 840.0


def test_plan_target_weights_must_sum_to_one():
    """Constructor validates target_weights sums to ~1."""
    with pytest.raises(ValueError, match='sum'):
        RebalancePlan(
            actions=_sample_actions(),
            target_weights={'AMZN': 0.5, 'UBSG.SW': 0.3},
            strategy_name='mvo',
            constraints=default_constraints(),
            holdings_df=_sample_holdings_df(),
            generated_at=datetime(2026, 5, 14, 10, 0))
```

- [ ] **Step 2: Run tests, confirm 6 plan tests fail**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_action_and_plan.py -v`
Expected: 5 action tests pass, 6 plan tests fail with `ModuleNotFoundError: No module named 'src.modelling.rebalancing.plan'`.

- [ ] **Step 3: Implement `src/modelling/rebalancing/plan.py`**

```python
"""
Rebalance plan aggregator.

Implements Contract 5 of the refactor design spec.
RebalancePlan bundles a list of RebalancingAction with the
inputs that produced them (target weights, strategy name,
constraints, holdings snapshot) and exposes DataFrame views
+ summary aggregates for the Excel report.
"""

from dataclasses import dataclass, field
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
            total_buy_chf, total_reduce_chf,
            capital_remaining_chf,
            weight_drift_pre_pct, weight_drift_post_pct.
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
            'capital_remaining_chf': capital_remaining,
            'weight_drift_pre_pct': drift_pre_mean,
            'weight_drift_post_pct': drift_post_mean,
        }
```

- [ ] **Step 4: Run tests, confirm 11 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_action_and_plan.py -v`
Expected: 11 passed (5 action + 6 plan).

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/modelling/rebalancing/plan.py \
        tests/modelling/rebalancing/test_action_and_plan.py
git commit -m "feat(rebalancing): add RebalancePlan with summary + DataFrame views"
```

---

## Task 3 — `Strategy` ABC + registry skeleton

**Files:**
- Create: `tests/modelling/rebalancing/test_strategies.py`
- Create: `src/modelling/rebalancing/strategies/base.py`
- Modify: `src/modelling/rebalancing/strategies/__init__.py` — registry skeleton.

- [ ] **Step 1: Write the strategy-registry tests**

Create `tests/modelling/rebalancing/test_strategies.py`:

```python
"""Tests for rebalancing strategies and the strategy registry."""

import pandas as pd
import pytest

from src.shared.constraints import default_constraints


def test_strategy_registry_has_four_initial_strategies():
    """Phase 2 ships exactly four portfolio-only strategies."""
    from src.modelling.rebalancing.strategies import (
        strategy_registry)
    expected = {
        'mvo', 'equal_weight', 'inverse_vol', 'min_variance'}
    assert set(strategy_registry) == expected


def test_get_strategy_returns_instance():
    """get_strategy(name) returns a Strategy instance."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    from src.modelling.rebalancing.strategies.base import (
        Strategy)
    s = get_strategy('equal_weight')
    assert isinstance(s, Strategy)
    assert s.name == 'equal_weight'


def test_get_strategy_raises_on_unknown():
    """Unknown name raises KeyError listing known names."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    with pytest.raises(KeyError, match='Unknown strategy'):
        get_strategy('not_a_strategy')


def test_strategy_abc_propose_signature():
    """Strategy.propose must accept the contract args."""
    from src.modelling.rebalancing.strategies.base import (
        Strategy)
    import inspect
    sig = inspect.signature(Strategy.propose)
    params = set(sig.parameters)
    expected = {
        'self', 'holdings_df', 'prices', 'constraints',
        'candidates', 'risk_free_rate'}
    assert expected.issubset(params)
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_strategies.py -v`
Expected: 4 collection errors — `ModuleNotFoundError: No module named 'src.modelling.rebalancing.strategies.base'` (and similar for the four concrete strategy modules referenced via the registry).

- [ ] **Step 3: Implement `src/modelling/rebalancing/strategies/base.py`**

```python
"""
Strategy ABC for rebalancing.

Every concrete strategy subclasses Strategy and implements
propose(...) returning target weights {ticker: fraction}.
Action generation is shared logic on Rebalancer, never
duplicated per strategy.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from src.shared.constraints import RebalanceConstraints


class Strategy(ABC):
    """
    Abstract rebalancing strategy.

    Attributes:
        name: Registry key. Must be set on every concrete
            subclass; used by `strategy_registry` and the CLI's
            `--strategy` flag.
    """

    name: str = ''

    @abstractmethod
    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Return target portfolio weights.

        Args:
            holdings_df: Snapshot from
                analyzer.get_holdings_snapshot(); columns include
                ticker, value_chf, weight_pct, category.
            prices: Wide DataFrame from
                analyzer.get_price_panel(); one column per held
                ticker, daily close, native currency.
            constraints: Per-run RebalanceConstraints.
            candidates: Optional list of CandidateTicker (set by
                screener). None for portfolio-only strategies;
                screener-fed strategies (Phase 4) require it.
            risk_free_rate: Annual rate in percent. None means
                the strategy fetches via
                src.shared.risk_free_rate.get_live_risk_free_rate().

        Returns:
            Dict {ticker: fraction} summing to 1.0 within
            target_weights_sum_tol of RebalancePlan.
        """
        raise NotImplementedError
```

- [ ] **Step 4: Implement skeleton `src/modelling/rebalancing/strategies/__init__.py`**

The four concrete classes don't exist yet. Build the registry incrementally — Tasks 4-7 each add one strategy. For now write a skeleton that imports each class lazily via `__init_subclass__` so the registry assembles itself as concrete modules are imported.

Replace `src/modelling/rebalancing/strategies/__init__.py` (currently empty) with:

```python
"""
Strategy registry: name -> Strategy subclass.

Concrete strategies self-register via __init_subclass__ on
the Strategy ABC. Importing this module triggers import of
every concrete strategy module so the registry is fully
populated.
"""

from typing import Dict, Type

from src.modelling.rebalancing.strategies.base import Strategy
# Importing each concrete module triggers its class definition,
# which registers itself in `strategy_registry` via
# Strategy.__init_subclass__.
from src.modelling.rebalancing.strategies import mvo
from src.modelling.rebalancing.strategies import equal_weight
from src.modelling.rebalancing.strategies import inverse_vol
from src.modelling.rebalancing.strategies import min_variance


strategy_registry: Dict[str, Type[Strategy]] = {}


def get_strategy(name: str) -> Strategy:
    """
    Instantiate a registered strategy by name.

    Args:
        name: Registry key (e.g. 'mvo', 'equal_weight').

    Returns:
        Fresh instance of the matching Strategy subclass.

    Raises:
        KeyError: When `name` is not a registered strategy.
            The message lists the known names.
    """
    if name not in strategy_registry:
        raise KeyError(
            f'Unknown strategy {name!r}. Known: '
            f'{sorted(strategy_registry)}')
    return strategy_registry[name]()
```

Now extend `src/modelling/rebalancing/strategies/base.py` with the auto-registration hook. Append (inside the file, after the imports but before the `Strategy` class definition):

```python
# Populated lazily by Strategy.__init_subclass__ as concrete
# strategy modules are imported. Defined here (not in this
# package's __init__.py) so subclasses can register themselves
# at class-definition time without importing the registry
# module first (avoids a circular import).
_registry: Dict[str, type] = {}
```

Then update the `Strategy` class to add `__init_subclass__`:

```python
class Strategy(ABC):
    """[unchanged docstring]"""

    name: str = ''

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            _registry[cls.name] = cls

    @abstractmethod
    def propose(
        self,
        ...):
        # ... unchanged
```

Finally rewire the package `__init__.py` to expose `strategy_registry` as a live view of `_registry`. Replace its body with:

```python
"""
Strategy registry: name -> Strategy subclass.

Concrete strategies self-register via Strategy.__init_subclass__.
Importing this module triggers import of every concrete strategy
module so the registry is fully populated.
"""

from src.modelling.rebalancing.strategies.base import (
    Strategy, _registry as strategy_registry)
from src.modelling.rebalancing.strategies import mvo
from src.modelling.rebalancing.strategies import equal_weight
from src.modelling.rebalancing.strategies import inverse_vol
from src.modelling.rebalancing.strategies import min_variance


def get_strategy(name: str) -> Strategy:
    """
    Instantiate a registered strategy by name.

    Args:
        name: Registry key (e.g. 'mvo', 'equal_weight').

    Returns:
        Fresh instance of the matching Strategy subclass.

    Raises:
        KeyError: When `name` is not registered. Message lists
            the known names.
    """
    if name not in strategy_registry:
        raise KeyError(
            f'Unknown strategy {name!r}. Known: '
            f'{sorted(strategy_registry)}')
    return strategy_registry[name]()
```

Note: `mvo`, `equal_weight`, `inverse_vol`, `min_variance` modules don't exist yet — `__init__.py` will fail to import. Don't worry, Tasks 4-7 add them. For now, comment out three of the four imports so Task 3 can ship in isolation. Pick `equal_weight` (simplest) to land first in Task 4 and keep imported. Remove the three other lines (mvo, inverse_vol, min_variance) — re-add in Tasks 5/6/7. ALSO: skip running test_strategy_registry_has_four_initial_strategies until all four land (we'll re-run after Task 7).

So the working `strategies/__init__.py` for Task 3 only:

```python
"""
Strategy registry: name -> Strategy subclass.

Concrete strategies self-register via Strategy.__init_subclass__.
Importing this module triggers import of every concrete strategy
module so the registry is fully populated.
"""

from src.modelling.rebalancing.strategies.base import (
    Strategy, _registry as strategy_registry)
from src.modelling.rebalancing.strategies import equal_weight


def get_strategy(name: str) -> Strategy:
    """[same docstring as above]"""
    if name not in strategy_registry:
        raise KeyError(
            f'Unknown strategy {name!r}. Known: '
            f'{sorted(strategy_registry)}')
    return strategy_registry[name]()
```

The `equal_weight` strategy is implemented in Task 4. To unblock Task 3 testing, write a temporary minimal `equal_weight.py` placeholder now:

Create `src/modelling/rebalancing/strategies/equal_weight.py` with:

```python
"""Equal-weight strategy. Full implementation in Task 4."""

from typing import Dict, List, Optional

import pandas as pd

from src.modelling.rebalancing.strategies.base import Strategy
from src.shared.constraints import RebalanceConstraints


class EqualWeightStrategy(Strategy):
    """1/N across non-excluded current holdings."""

    name = 'equal_weight'

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        # Placeholder: full implementation in Task 4.
        raise NotImplementedError(
            'Implemented in Task 4 of the Phase 2 plan.')
```

- [ ] **Step 5: Run tests, confirm 3 of 4 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_strategies.py -v -k "not four_initial"`
Expected: 3 passed (registry returns instance; raises on unknown; ABC has correct signature). The four-strategy test is skipped until Task 7.

- [ ] **Step 6: Commit (advisory)**

```bash
git add src/modelling/rebalancing/strategies/base.py \
        src/modelling/rebalancing/strategies/__init__.py \
        src/modelling/rebalancing/strategies/equal_weight.py \
        tests/modelling/rebalancing/test_strategies.py
git commit -m "feat(rebalancing): add Strategy ABC + auto-registration registry"
```

---

## Task 4 — `EqualWeightStrategy`

**Files:**
- Modify: `tests/modelling/rebalancing/test_strategies.py` — add equal-weight tests.
- Modify: `src/modelling/rebalancing/strategies/equal_weight.py` — replace placeholder with full implementation.

- [ ] **Step 1: Append tests**

```python


def test_equal_weight_uniform_across_eligible():
    """1/N weights across non-excluded, non-dust holdings."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 30.0, 'category': 'core'},
        {'ticker': 'UBSG.SW', 'value_chf': 400.0,
         'weight_pct': 20.0, 'category': 'core'},
        {'ticker': 'HOLN.SW', 'value_chf': 1000.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    prices = pd.DataFrame()    # equal_weight ignores prices
    s = get_strategy('equal_weight')
    w = s.propose(holdings, prices, default_constraints())
    assert set(w) == {'AMZN', 'UBSG.SW', 'HOLN.SW'}
    for v in w.values():
        assert v == pytest.approx(1.0 / 3.0)


def test_equal_weight_excludes_speculative():
    """excluded_categories drops tickers before weighting."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 30.0, 'category': 'core'},
        {'ticker': 'USAR', 'value_chf': 400.0,
         'weight_pct': 20.0, 'category': 'speculative'},
        {'ticker': 'HOLN.SW', 'value_chf': 1000.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    prices = pd.DataFrame()
    s = get_strategy('equal_weight')
    w = s.propose(holdings, prices, default_constraints())
    assert 'USAR' not in w
    for v in w.values():
        assert v == pytest.approx(0.5)


def test_equal_weight_drops_dust_positions():
    """min_position_pct drops dust before weighting."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 60.0, 'category': 'core'},
        {'ticker': 'TINY', 'value_chf': 1.0,
         'weight_pct': 0.1, 'category': 'core'},
        {'ticker': 'HOLN.SW', 'value_chf': 400.0,
         'weight_pct': 39.9, 'category': 'core'},
    ])
    prices = pd.DataFrame()
    s = get_strategy('equal_weight')
    w = s.propose(holdings, prices, default_constraints())
    assert 'TINY' not in w
    assert len(w) == 2


def test_equal_weight_sums_to_one():
    """Output weights sum to 1.0 within plan tolerance."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 30.0, 'category': 'core'},
        {'ticker': 'UBSG.SW', 'value_chf': 400.0,
         'weight_pct': 20.0, 'category': 'core'},
    ])
    s = get_strategy('equal_weight')
    w = s.propose(holdings, pd.DataFrame(),
                  default_constraints())
    assert sum(w.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run, confirm 4 fail with NotImplementedError**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_strategies.py -v -k "equal_weight"`
Expected: 4 failures — `NotImplementedError: Implemented in Task 4`.

- [ ] **Step 3: Replace `src/modelling/rebalancing/strategies/equal_weight.py`**

```python
"""
EqualWeightStrategy: 1/N across eligible current holdings.

Eligibility filters: drop holdings whose category is in
constraints.excluded_categories, then drop dust holdings whose
weight_pct is below constraints.min_position_pct.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.modelling.rebalancing.strategies.base import Strategy
from src.shared.constraints import RebalanceConstraints


class EqualWeightStrategy(Strategy):
    """
    Uniform weights across eligible current holdings.

    Ignores `prices`, `candidates`, and `risk_free_rate` —
    pure portfolio-snapshot strategy. Useful as a sanity
    baseline.
    """

    name = 'equal_weight'

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        eligible = _filter_eligible(holdings_df, constraints)
        n = len(eligible)
        if n == 0:
            raise RuntimeError(
                'EqualWeightStrategy: no eligible holdings '
                'after applying constraints.')
        weight = 1.0 / n
        return {ticker: weight for ticker in eligible['ticker']}


def _filter_eligible(
    holdings_df: pd.DataFrame,
    constraints: RebalanceConstraints) -> pd.DataFrame:
    """
    Drop excluded categories and dust positions.

    Args:
        holdings_df: Snapshot from
            analyzer.get_holdings_snapshot(); must include
            'category' and 'weight_pct'.
        constraints: RebalanceConstraints; reads
            excluded_categories and min_position_pct.

    Returns:
        Filtered holdings DataFrame, same columns.

    Raises:
        KeyError: When holdings_df is missing 'category' or
            'weight_pct' columns.
    """
    for col in ('category', 'weight_pct', 'ticker'):
        if col not in holdings_df.columns:
            raise KeyError(
                f'_filter_eligible: holdings_df missing '
                f'column {col!r}.')
    df = holdings_df[
        ~holdings_df['category'].isin(
            constraints.excluded_categories)]
    df = df[df['weight_pct'] >= constraints.min_position_pct]
    return df
```

- [ ] **Step 4: Run, confirm 4 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_strategies.py -v -k "equal_weight"`
Expected: 4 passed.

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/modelling/rebalancing/strategies/equal_weight.py \
        tests/modelling/rebalancing/test_strategies.py
git commit -m "feat(rebalancing): implement EqualWeightStrategy + filters"
```

---

## Task 5 — `InverseVolStrategy`

**Files:**
- Modify: `tests/modelling/rebalancing/test_strategies.py` — add inverse-vol tests.
- Create: `src/modelling/rebalancing/strategies/inverse_vol.py`
- Modify: `src/modelling/rebalancing/strategies/__init__.py` — re-add the `inverse_vol` import.

- [ ] **Step 1: Append tests**

```python


def test_inverse_vol_higher_weight_for_lower_vol():
    """Lower-vol asset gets higher weight."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    holdings = pd.DataFrame([
        {'ticker': 'LOWVOL', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
        {'ticker': 'HIGHVOL', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    # Build flat-then-noisy synthetic price series so HIGHVOL
    # has ~5x the std of LOWVOL.
    import numpy as np
    rng = np.random.default_rng(42)
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    low = pd.Series(
        100 + rng.normal(0, 0.5, n).cumsum(), index=idx)
    high = pd.Series(
        100 + rng.normal(0, 2.5, n).cumsum(), index=idx)
    prices = pd.DataFrame({'LOWVOL': low, 'HIGHVOL': high})
    s = get_strategy('inverse_vol')
    w = s.propose(holdings, prices, default_constraints())
    assert w['LOWVOL'] > w['HIGHVOL']
    assert sum(w.values()) == pytest.approx(1.0)


def test_inverse_vol_raises_when_prices_missing_ticker():
    """Held ticker absent from `prices` raises (no silent skip)."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 30.0, 'category': 'core'},
        {'ticker': 'GHOST', 'value_chf': 400.0,
         'weight_pct': 20.0, 'category': 'core'},
    ])
    prices = pd.DataFrame({
        'AMZN': pd.Series(
            [100.0, 101.0, 102.0, 100.5],
            index=pd.date_range('2024-01-01', periods=4))})
    s = get_strategy('inverse_vol')
    with pytest.raises(RuntimeError, match='GHOST'):
        s.propose(holdings, prices, default_constraints())
```

- [ ] **Step 2: Run, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_strategies.py -v -k "inverse_vol"`
Expected: 2 failures — `KeyError: 'Unknown strategy 'inverse_vol''`.

- [ ] **Step 3: Implement `src/modelling/rebalancing/strategies/inverse_vol.py`**

```python
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
    _filter_eligible)
from src.shared.constraints import RebalanceConstraints


trading_days_per_year = 252


class InverseVolStrategy(Strategy):
    """
    w_i ∝ 1/σ_i, normalised to sum to 1.

    σ_i is the annualised standard deviation of the ticker's
    daily returns over the available price-history window.
    """

    name = 'inverse_vol'

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        eligible = _filter_eligible(holdings_df, constraints)
        tickers = list(eligible['ticker'])
        if len(tickers) == 0:
            raise RuntimeError(
                'InverseVolStrategy: no eligible holdings '
                'after applying constraints.')

        # Every eligible ticker must have a price column
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise RuntimeError(
                f'InverseVolStrategy: prices missing column(s) '
                f'for held ticker(s) {sorted(missing)}.')

        # Annualised volatility per ticker
        returns = prices[tickers].pct_change().dropna()
        annual_vols = returns.std() * np.sqrt(trading_days_per_year)
        # Numerical floor so a flat-line ticker doesn't blow up
        annual_vols = annual_vols.where(annual_vols > 1e-10,
                                        other=1e-10)
        inv = 1.0 / annual_vols
        weights = inv / inv.sum()
        return {t: float(weights[t]) for t in tickers}
```

- [ ] **Step 4: Re-add `inverse_vol` import in `strategies/__init__.py`**

Insert below the `from src.modelling.rebalancing.strategies import equal_weight` line:

```python
from src.modelling.rebalancing.strategies import inverse_vol
```

- [ ] **Step 5: Run, confirm 2 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_strategies.py -v -k "inverse_vol"`
Expected: 2 passed.

- [ ] **Step 6: Commit (advisory)**

```bash
git add src/modelling/rebalancing/strategies/inverse_vol.py \
        src/modelling/rebalancing/strategies/__init__.py \
        tests/modelling/rebalancing/test_strategies.py
git commit -m "feat(rebalancing): implement InverseVolStrategy"
```

---

## Task 6 — `MinVarianceStrategy`

**Files:**
- Modify: `tests/modelling/rebalancing/test_strategies.py` — add min-variance tests.
- Create: `src/modelling/rebalancing/strategies/min_variance.py`
- Modify: `src/modelling/rebalancing/strategies/__init__.py` — add `min_variance` import.

- [ ] **Step 1: Append tests**

```python


def test_min_variance_assigns_more_weight_to_lower_vol():
    """Min-variance over 2 uncorrelated assets favors low-vol."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    import numpy as np
    holdings = pd.DataFrame([
        {'ticker': 'STABLE', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
        {'ticker': 'VOLATILE', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    rng = np.random.default_rng(0)
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    stable = pd.Series(
        100 + rng.normal(0, 0.3, n).cumsum(), index=idx)
    volatile = pd.Series(
        100 + rng.normal(0, 3.0, n).cumsum(), index=idx)
    prices = pd.DataFrame({
        'STABLE': stable, 'VOLATILE': volatile})
    s = get_strategy('min_variance')
    w = s.propose(holdings, prices, default_constraints())
    assert w['STABLE'] > w['VOLATILE']
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    # Per-name cap honored
    assert max(w.values()) <= (
        default_constraints().max_weight_default + 1e-6)


def test_min_variance_respects_max_weight_cap():
    """A heavily preferred asset is still capped at 15%."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    import numpy as np
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(1)
    rows = []
    cols = {}
    # 10 nearly-identical low-vol assets so cap binds
    for i in range(10):
        t = f'A{i}'
        rows.append({
            'ticker': t, 'value_chf': 100.0,
            'weight_pct': 10.0, 'category': 'core'})
        cols[t] = pd.Series(
            100 + rng.normal(0, 0.5, n).cumsum(), index=idx)
    holdings = pd.DataFrame(rows)
    prices = pd.DataFrame(cols)
    s = get_strategy('min_variance')
    w = s.propose(holdings, prices, default_constraints())
    assert max(w.values()) <= (
        default_constraints().max_weight_default + 1e-6)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
```

- [ ] **Step 2: Run, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_strategies.py -v -k "min_variance"`
Expected: 2 failures — `KeyError: 'Unknown strategy 'min_variance''`.

- [ ] **Step 3: Implement `src/modelling/rebalancing/strategies/min_variance.py`**

```python
"""
MinVarianceStrategy: minimise w^T Σ w subject to long-only,
per-name caps, weights sum to 1.

Operates on current holdings only. Uses scipy SLSQP.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.modelling.rebalancing.strategies.base import Strategy
from src.modelling.rebalancing.strategies.equal_weight import (
    _filter_eligible)
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
        eligible = _filter_eligible(holdings_df, constraints)
        tickers = list(eligible['ticker'])
        if len(tickers) < 2:
            raise RuntimeError(
                'MinVarianceStrategy: need at least 2 eligible '
                'holdings for a meaningful optimisation.')

        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise RuntimeError(
                f'MinVarianceStrategy: prices missing column(s) '
                f'for held ticker(s) {sorted(missing)}.')

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
```

- [ ] **Step 4: Add `min_variance` import in `strategies/__init__.py`**

Insert below the `inverse_vol` line:

```python
from src.modelling.rebalancing.strategies import min_variance
```

- [ ] **Step 5: Run, confirm 2 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_strategies.py -v -k "min_variance"`
Expected: 2 passed.

- [ ] **Step 6: Commit (advisory)**

```bash
git add src/modelling/rebalancing/strategies/min_variance.py \
        src/modelling/rebalancing/strategies/__init__.py \
        tests/modelling/rebalancing/test_strategies.py
git commit -m "feat(rebalancing): implement MinVarianceStrategy via SLSQP"
```

---

## Task 7 — `MVOStrategy` (max Sharpe)

**Files:**
- Modify: `tests/modelling/rebalancing/test_strategies.py` — add MVO tests + the deferred 4-strategy registry test.
- Create: `src/modelling/rebalancing/strategies/mvo.py`
- Modify: `src/modelling/rebalancing/strategies/__init__.py` — add `mvo` import.

- [ ] **Step 1: Append tests**

```python


def test_mvo_returns_valid_weight_dict():
    """MVO produces long-only weights summing to 1, capped."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    import numpy as np
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(7)
    rows = []
    cols = {}
    # 5 assets with mixed return / vol profiles
    profiles = [(0.10, 0.5), (0.05, 0.3), (0.15, 1.0),
                (0.08, 0.4), (0.12, 0.6)]
    for i, (mu, sig) in enumerate(profiles):
        t = f'A{i}'
        rows.append({
            'ticker': t, 'value_chf': 200.0,
            'weight_pct': 20.0, 'category': 'core'})
        # Daily mean ≈ mu/252; daily std ≈ sig/sqrt(252)
        daily_returns = rng.normal(
            mu / 252.0, sig / np.sqrt(252.0), n)
        cols[t] = pd.Series(
            100.0 * (1 + pd.Series(daily_returns)).cumprod().values,
            index=idx)
    holdings = pd.DataFrame(rows)
    prices = pd.DataFrame(cols)
    s = get_strategy('mvo')
    w = s.propose(holdings, prices, default_constraints(),
                  risk_free_rate=0.5)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(v >= -1e-9 for v in w.values())
    assert max(w.values()) <= (
        default_constraints().max_weight_default + 1e-6)


def test_mvo_uses_live_rfr_when_none(monkeypatch):
    """risk_free_rate=None triggers get_live_risk_free_rate."""
    from src.modelling.rebalancing.strategies import (
        get_strategy)
    called = {'n': 0}

    def fake_rfr(*args, **kwargs):
        called['n'] += 1
        return 0.5

    monkeypatch.setattr(
        'src.modelling.rebalancing.strategies.mvo.'
        'get_live_risk_free_rate', fake_rfr)
    import numpy as np
    n = 252
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(0)
    holdings = pd.DataFrame([
        {'ticker': 'A', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
        {'ticker': 'B', 'value_chf': 500.0,
         'weight_pct': 50.0, 'category': 'core'},
    ])
    prices = pd.DataFrame({
        'A': pd.Series(
            100 + rng.normal(0, 1, n).cumsum(), index=idx),
        'B': pd.Series(
            100 + rng.normal(0, 0.5, n).cumsum(), index=idx)})
    s = get_strategy('mvo')
    s.propose(holdings, prices, default_constraints())
    assert called['n'] == 1


def test_strategy_registry_has_four_initial_strategies():
    """Phase 2 ships exactly four portfolio-only strategies."""
    from src.modelling.rebalancing.strategies import (
        strategy_registry)
    expected = {
        'mvo', 'equal_weight', 'inverse_vol', 'min_variance'}
    assert set(strategy_registry) == expected
```

(Note: this re-defines `test_strategy_registry_has_four_initial_strategies` from Task 3 — that one was deferred. If the function already exists with the same name from Task 3, replace it; pytest will pick up the latest definition from the file.)

- [ ] **Step 2: Run, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_strategies.py -v -k "mvo or four_initial"`
Expected: 3 failures — `KeyError: 'Unknown strategy 'mvo''`.

- [ ] **Step 3: Implement `src/modelling/rebalancing/strategies/mvo.py`**

```python
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
    _filter_eligible)
from src.shared.constraints import RebalanceConstraints
from src.shared.risk_free_rate import get_live_risk_free_rate


trading_days_per_year = 252
covariance_min_periods = 30
slsqp_max_iter = 1000
slsqp_ftol = 1e-12
volatility_floor = 1e-10


class MVOStrategy(Strategy):
    """
    Maximise (w^T mu - rf) / sqrt(w^T Σ w).

    mu = annualised mean of daily returns per ticker.
    Σ = annualised pairwise-complete sample covariance with
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
        eligible = _filter_eligible(holdings_df, constraints)
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
        # rf is in percent; convert to decimal for the ratio
        rf_decimal = risk_free_rate / 100.0

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
```

- [ ] **Step 4: Add `mvo` import in `strategies/__init__.py`**

Insert below the `min_variance` line:

```python
from src.modelling.rebalancing.strategies import mvo
```

- [ ] **Step 5: Run all strategy tests, confirm pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_strategies.py -v`
Expected: at least 15 passed (4 abc/registry + 4 equal_weight + 2 inverse_vol + 2 min_variance + 3 mvo = 15; the deferred test from Task 3 now passes too).

- [ ] **Step 6: Commit (advisory)**

```bash
git add src/modelling/rebalancing/strategies/mvo.py \
        src/modelling/rebalancing/strategies/__init__.py \
        tests/modelling/rebalancing/test_strategies.py
git commit -m "feat(rebalancing): implement MVOStrategy (max Sharpe via SLSQP)"
```

---

## Task 8 — `Rebalancer` orchestrator

**Files:**
- Create: `tests/modelling/rebalancing/test_rebalancer.py`
- Create: `src/modelling/rebalancing/rebalancer.py`
- Modify: `src/modelling/rebalancing/__init__.py` — populate re-exports.

- [ ] **Step 1: Write tests**

```python
"""Tests for Rebalancer orchestrator."""

import pandas as pd
import pytest

from src.modelling.rebalancing.action import RebalancingAction
from src.modelling.rebalancing.plan import RebalancePlan
from src.modelling.rebalancing.rebalancer import Rebalancer
from src.modelling.rebalancing.strategies import get_strategy
from src.shared.constraints import (
    RebalanceConstraints, default_constraints)


def _holdings(weights_pct, value_total=10000.0):
    """Build a fake holdings_df with given weights summing to 100."""
    rows = []
    for ticker, w in weights_pct.items():
        value = value_total * w / 100.0
        rows.append({
            'ticker': ticker, 'shares': 10,
            'value_chf': value, 'weight_pct': w,
            'currency': 'CHF', 'sector': 'X',
            'price_chf': value / 10.0, 'category': 'core'})
    return pd.DataFrame(rows)


class _FakeAnalyzer:
    """Stand-in for PortfolioAnalyzer in unit tests.

    Provides only the two methods Rebalancer calls.
    """
    def __init__(self, holdings_df, prices):
        self._holdings_df = holdings_df
        self._prices = prices
    def get_holdings_snapshot(self, categories=None):
        return self._holdings_df.copy()
    def get_price_panel(self):
        return self._prices.copy()


def test_rebalancer_returns_plan():
    """Rebalancer.propose returns a RebalancePlan."""
    holdings = _holdings({'A': 40.0, 'B': 60.0})
    prices = pd.DataFrame({
        'A': [100.0, 100.0, 100.0],
        'B': [100.0, 100.0, 100.0]},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        default_constraints())
    plan = rb.propose()
    assert isinstance(plan, RebalancePlan)
    assert plan.strategy_name == 'equal_weight'


def test_rebalancer_buy_action_created_for_underweight():
    """Underweight ticker with new capital -> BUY action."""
    holdings = _holdings({'A': 30.0, 'B': 70.0})
    prices = pd.DataFrame({
        'A': [100.0] * 3, 'B': [100.0] * 3},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        constraints)
    plan = rb.propose()
    by_ticker = {a.ticker: a for a in plan.actions}
    # A is underweight (30% vs target 50%); should BUY
    assert by_ticker['A'].action == 'BUY'
    assert by_ticker['A'].shares > 0


def test_rebalancer_reduce_action_for_overweight_beyond_band():
    """Overweight beyond -200 CHF gap -> REDUCE."""
    # Total NAV 10000; 70/30; equal-weight target 50/50;
    # target_value(B) = 5000; current_value(B) = 7000;
    # delta = -2000 < -200 -> REDUCE
    holdings = _holdings({'A': 30.0, 'B': 70.0})
    prices = pd.DataFrame({
        'A': [100.0] * 3, 'B': [100.0] * 3},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        default_constraints())
    plan = rb.propose()
    by_ticker = {a.ticker: a for a in plan.actions}
    assert by_ticker['B'].action == 'REDUCE'
    assert by_ticker['B'].shares > 0
    assert 'held' in by_ticker['B'].note.lower()


def test_rebalancer_hold_inside_band():
    """Tickers near target -> HOLD."""
    # Already balanced: gap is 0
    holdings = _holdings({'A': 50.0, 'B': 50.0})
    prices = pd.DataFrame({
        'A': [100.0] * 3, 'B': [100.0] * 3},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        default_constraints())
    plan = rb.propose()
    actions_by_type = {a.action for a in plan.actions}
    assert 'HOLD' in actions_by_type


def test_rebalancer_buy_capital_capped_at_new_capital():
    """Total BUY est_cost_chf doesn't exceed new_capital_chf."""
    holdings = _holdings({'A': 10.0, 'B': 90.0})
    prices = pd.DataFrame({
        'A': [100.0] * 3, 'B': [100.0] * 3},
        index=pd.date_range('2024-01-01', periods=3))
    analyzer = _FakeAnalyzer(holdings, prices)
    constraints = RebalanceConstraints(new_capital_chf=500.0)
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'), constraints)
    plan = rb.propose()
    total_buy = sum(
        a.est_cost_chf for a in plan.actions
        if a.action == 'BUY')
    assert total_buy <= constraints.new_capital_chf + 1e-6


def test_rebalancer_passes_categories_to_analyzer():
    """Rebalancer.propose forwards constraints-derived categories."""
    captured = {}

    class _CapturingAnalyzer:
        _holdings_df = _holdings({'A': 50.0, 'B': 50.0})
        _prices = pd.DataFrame({
            'A': [100.0] * 3, 'B': [100.0] * 3},
            index=pd.date_range('2024-01-01', periods=3))

        def get_holdings_snapshot(self, categories=None):
            captured['categories'] = categories
            return self._holdings_df.copy()

        def get_price_panel(self):
            return self._prices.copy()

    analyzer = _CapturingAnalyzer()
    cats = {'A': 'core', 'B': 'core'}
    rb = Rebalancer(
        analyzer, get_strategy('equal_weight'),
        default_constraints(),
        ticker_categories=cats)
    rb.propose()
    assert captured['categories'] == cats
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_rebalancer.py -v`
Expected: 6 failures — `ModuleNotFoundError: No module named 'src.modelling.rebalancing.rebalancer'`.

- [ ] **Step 3: Implement `src/modelling/rebalancing/rebalancer.py`**

```python
"""
Rebalancer orchestrator.

Wires PortfolioAnalyzer + Strategy + RebalanceConstraints into
a RebalancePlan. Holds the shared action-generation logic that
turns a strategy's target weights into BUY/REDUCE/HOLD actions:
allocate new_capital_chf to underweight names first
(largest-gap-first), flag overweight names beyond the
rebalance band as REDUCE, leave gaps inside the band as HOLD.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.modelling.rebalancing.action import RebalancingAction
from src.modelling.rebalancing.plan import RebalancePlan
from src.modelling.rebalancing.strategies.base import Strategy
from src.shared.constraints import (
    RebalanceConstraints, default_constraints)


class Rebalancer:
    """
    Apply a Strategy to current holdings and produce a plan.

    Attributes:
        analyzer: Provides get_holdings_snapshot() and
            get_price_panel().
        strategy: A concrete Strategy implementation.
        constraints: RebalanceConstraints for this run.
        ticker_categories: Optional ticker -> category map
            forwarded to analyzer.get_holdings_snapshot. Required
            when constraints reference excluded_categories or
            category_caps and the analyzer's snapshot won't
            otherwise have a 'category' column.
    """

    def __init__(
        self,
        analyzer,
        strategy: Strategy,
        constraints: Optional[RebalanceConstraints] = None,
        ticker_categories: Optional[Dict[str, str]] = None):
        self.analyzer = analyzer
        self.strategy = strategy
        self.constraints = constraints or default_constraints()
        self.ticker_categories = ticker_categories

    def propose(
        self, candidates: Optional[List] = None) -> RebalancePlan:
        """
        Build a RebalancePlan from the analyzer's current state.

        Args:
            candidates: Optional list of CandidateTicker; only
                screener-fed strategies (Phase 4) require it.

        Returns:
            RebalancePlan bundling actions, target weights, the
            input holdings snapshot, the strategy name, the
            applied constraints, and a generation timestamp.
        """
        holdings_df = self.analyzer.get_holdings_snapshot(
            categories=self.ticker_categories)
        prices = self.analyzer.get_price_panel()
        target_weights = self.strategy.propose(
            holdings_df, prices, self.constraints,
            candidates=candidates)
        actions = self._generate_actions(
            target_weights, holdings_df)
        return RebalancePlan(
            actions=actions,
            target_weights=target_weights,
            strategy_name=self.strategy.name,
            constraints=self.constraints,
            holdings_df=holdings_df,
            generated_at=datetime.now())

    def _generate_actions(
        self,
        target_weights: Dict[str, float],
        holdings_df: pd.DataFrame) -> List[RebalancingAction]:
        """
        Turn target weights + current snapshot into actions.

        Algorithm:
            1. For each ticker with a target weight, compute the
               gap delta = target_value - current_value, where
               target_value uses NAV + new_capital_chf.
            2. Underweight (delta > band upper bound):
               BUY rows. Allocate new_capital_chf
               largest-gap-first; cap each allocation at the
               gap and at the remaining pool. Convert CHF to a
               whole-share count using the position's price_chf.
            3. Overweight (delta < band lower bound): REDUCE
               rows. Note='Only if held >6 months' (Swiss tax-
               loss rule).
            4. Inside the band: HOLD rows.
        """
        if not target_weights:
            return []
        # Lookup tables from the snapshot
        snap_by_ticker = holdings_df.set_index('ticker')
        total_value = float(holdings_df['value_chf'].sum())
        target_total = (
            total_value + self.constraints.new_capital_chf)

        # Per-ticker gap analysis
        deltas = []
        for ticker, target_w in target_weights.items():
            if ticker not in snap_by_ticker.index:
                # Strategy proposed a ticker we don't hold (only
                # screener-fed strategies do this); Phase 2 has
                # no such strategies, so this is an error.
                raise RuntimeError(
                    f'Rebalancer: target weight for ticker '
                    f'{ticker!r} not in current holdings.')
            row = snap_by_ticker.loc[ticker]
            current_value = float(row['value_chf'])
            current_w_pct = float(row['weight_pct'])
            target_value = target_w * target_total
            delta = target_value - current_value
            deltas.append({
                'ticker': ticker,
                'current_w_pct': current_w_pct,
                'target_w_pct': target_w * 100.0,
                'delta_chf': delta,
                'price_chf': float(row['price_chf']),
            })

        actions: List[RebalancingAction] = []
        lower, upper = self.constraints.rebalance_band_chf

        # BUY pass: allocate new capital to underweights largest-gap-first
        underweight = sorted(
            [d for d in deltas if d['delta_chf'] > upper],
            key=lambda d: -d['delta_chf'])
        remaining = self.constraints.new_capital_chf
        bought_tickers = set()
        for d in underweight:
            if remaining <= 0:
                break
            alloc = min(remaining, d['delta_chf'])
            shares = 0
            if d['price_chf'] > 0:
                shares = int(alloc / d['price_chf'])
                alloc = shares * d['price_chf']
            if shares <= 0:
                continue
            actions.append(RebalancingAction(
                ticker=d['ticker'], action='BUY',
                shares=shares, est_cost_chf=alloc,
                current_wt_pct=d['current_w_pct'],
                target_wt_pct=d['target_w_pct'],
                note=''))
            remaining -= alloc
            bought_tickers.add(d['ticker'])

        # Underweight tickers we couldn't BUY become HOLD
        for d in underweight:
            if d['ticker'] in bought_tickers:
                continue
            actions.append(RebalancingAction(
                ticker=d['ticker'], action='HOLD',
                shares=0, est_cost_chf=0.0,
                current_wt_pct=d['current_w_pct'],
                target_wt_pct=d['target_w_pct'],
                note='Underweight but no capital available'))

        # REDUCE pass: overweight beyond band lower bound
        overweight = sorted(
            [d for d in deltas if d['delta_chf'] < lower],
            key=lambda d: d['delta_chf'])
        for d in overweight:
            shares = 0
            magnitude = abs(d['delta_chf'])
            if d['price_chf'] > 0:
                shares = int(magnitude / d['price_chf'])
            actions.append(RebalancingAction(
                ticker=d['ticker'], action='REDUCE',
                shares=shares,
                est_cost_chf=shares * d['price_chf'],
                current_wt_pct=d['current_w_pct'],
                target_wt_pct=d['target_w_pct'],
                note='Only if held >6 months'))

        # HOLD pass: inside the band
        in_band = [
            d for d in deltas
            if lower <= d['delta_chf'] <= upper]
        for d in in_band:
            actions.append(RebalancingAction(
                ticker=d['ticker'], action='HOLD',
                shares=0, est_cost_chf=0.0,
                current_wt_pct=d['current_w_pct'],
                target_wt_pct=d['target_w_pct'],
                note=''))
        return actions
```

- [ ] **Step 4: Populate `src/modelling/rebalancing/__init__.py` with re-exports**

Replace its current empty content with:

```python
"""
Rebalancing domain: data model, plan, strategies, orchestrator.

Mirrors src.analysis.core's data-model -> aggregator ->
orchestrator shape. Strategies live in the
src.modelling.rebalancing.strategies subpackage.
"""

from src.modelling.rebalancing.action import RebalancingAction
from src.modelling.rebalancing.plan import RebalancePlan
from src.modelling.rebalancing.rebalancer import Rebalancer
from src.modelling.rebalancing.strategies import (
    get_strategy, strategy_registry, Strategy)


__all__ = [
    'RebalancingAction',
    'RebalancePlan',
    'Rebalancer',
    'Strategy',
    'get_strategy',
    'strategy_registry',
]
```

- [ ] **Step 5: Run rebalancer tests, confirm 6 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/modelling/rebalancing/test_rebalancer.py -v`
Expected: 6 passed.

- [ ] **Step 6: Run the full Phase-2 test suite to confirm no regressions**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: 28 (Phase 1) + 11 (Task 1+2) + at least 15 (Tasks 3-7) + 6 (Task 8) = 60+ passed.

- [ ] **Step 7: Commit (advisory)**

```bash
git add src/modelling/rebalancing/rebalancer.py \
        src/modelling/rebalancing/__init__.py \
        tests/modelling/rebalancing/test_rebalancer.py
git commit -m "feat(rebalancing): add Rebalancer orchestrator + action generation"
```

---

## Task 9 — Rewrite `src/analysis/report.py` as CLI orchestrator

**Files:**
- Create: `tests/analysis/test_report.py`
- Modify: `src/analysis/report.py` — full rewrite (delete old contents, replace).

- [ ] **Step 1: Write the integration test**

Create `tests/analysis/test_report.py`:

```python
"""Integration tests for the rewritten src.analysis.report CLI."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _write_categories(tmp_path):
    """Tiny ticker categories file matching the conftest fixtures."""
    p = tmp_path / 'categories.json'
    p.write_text(json.dumps({
        'AMZN': 'core',
        'UBSG.SW': 'core',
        'HOLN.SW': 'core',
    }))
    return p


def test_report_main_writes_excel(
    monkeypatch, tmp_path, patched_csv_loader, fake_data_provider):
    """End-to-end: report.main writes a multi-sheet xlsx."""
    cats_path = _write_categories(tmp_path)
    out_path = tmp_path / 'report.xlsx'
    # Argparse reads sys.argv; simulate the invocation
    args = [
        'report.py',
        '--strategy', 'equal_weight',
        '--new-capital', '1000',
        '--max-weight', '0.15',
        '--degiro-csv', 'ignored.csv',
        '--ibkr-csv', 'ignored.csv',
        '--ticker-categories', str(cats_path),
        '--output', str(out_path),
    ]
    monkeypatch.setattr(sys, 'argv', args)
    # Inject the fake data provider via the factory hook
    from src.analysis import report
    monkeypatch.setattr(
        report, '_build_data_provider', lambda: fake_data_provider)
    report.main()
    assert out_path.exists()
    # Inspect sheet names
    with pd.ExcelFile(out_path) as xf:
        names = set(xf.sheet_names)
    expected = {
        'Holdings', 'Target Weights', 'Rebalancing Actions',
        'Summary'}
    assert expected.issubset(names)


def test_report_main_unknown_strategy_exits(
    monkeypatch, tmp_path, patched_csv_loader, fake_data_provider):
    """Unknown --strategy raises SystemExit (argparse) or KeyError."""
    cats_path = _write_categories(tmp_path)
    args = [
        'report.py',
        '--strategy', 'nope',
        '--degiro-csv', 'ignored.csv',
        '--ibkr-csv', 'ignored.csv',
        '--ticker-categories', str(cats_path),
        '--output', str(tmp_path / 'r.xlsx'),
    ]
    monkeypatch.setattr(sys, 'argv', args)
    from src.analysis import report
    monkeypatch.setattr(
        report, '_build_data_provider', lambda: fake_data_provider)
    with pytest.raises((SystemExit, KeyError)):
        report.main()
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/analysis/test_report.py -v`
Expected: failures — likely `ImportError` on `_build_data_provider` (does not yet exist) or `argparse` mismatch.

- [ ] **Step 3: Replace `src/analysis/report.py` with the new orchestrator**

Delete the entire contents of `src/analysis/report.py` and write:

```python
"""
Portfolio rebalancing report — CLI orchestrator.

Wires PortfolioAnalyzer + Rebalancer + DataExporter into a
single command. Reads transaction CSVs, runs the chosen
strategy against current holdings, and writes a multi-sheet
Excel workbook with current holdings, target weights, and
BUY/REDUCE/HOLD actions.

Usage:
    python -m src.analysis.report \\
        --strategy mvo \\
        --new-capital 2000 \\
        --max-weight 0.15 \\
        --degiro-csv data/postprocess_data/processed_portfolio.csv \\
        --ibkr-csv reports/ibkr/<file>.csv \\
        --ticker-categories data/ticker_categories.json \\
        --output results/portfolio_report.xlsx
"""

import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import Dict

import pandas as pd

from src.analysis.core.analyzer import PortfolioAnalyzer
from src.analysis.loaders.data_exporter import DataExporter
from src.modelling.rebalancing import (
    Rebalancer, get_strategy)
from src.shared.constraints import RebalanceConstraints
from src.shared.data_provider import (
    DataProvider, YFinanceProvider)


def _build_data_provider() -> DataProvider:
    """Factory hook — overridden in tests to inject fakes."""
    return YFinanceProvider()


def _parse_args(argv) -> argparse.Namespace:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog='report.py',
        description=(
            'Generate the portfolio rebalancing Excel report.'))
    p.add_argument(
        '--strategy', required=True,
        choices=['mvo', 'equal_weight',
                 'inverse_vol', 'min_variance'],
        help='Rebalancing strategy to apply.')
    p.add_argument(
        '--new-capital', type=float, default=0.0,
        help='New CHF capital to deploy in BUY actions.')
    p.add_argument(
        '--max-weight', type=float, default=0.15,
        help='Per-name maximum weight as a fraction.')
    p.add_argument(
        '--degiro-csv',
        help='Path to the Degiro transaction CSV.')
    p.add_argument(
        '--ibkr-csv',
        help='Path to the IBKR Activity Statement CSV.')
    p.add_argument(
        '--ticker-categories', required=True,
        help='Path to ticker_categories.json.')
    p.add_argument(
        '--output', required=True,
        help='Output .xlsx path.')
    return p.parse_args(argv)


def _load_categories(path: str) -> Dict[str, str]:
    """Load ticker -> category map from JSON."""
    with open(path) as f:
        return json.load(f)


def _build_constraints(args: argparse.Namespace) -> RebalanceConstraints:
    """Translate CLI args into a RebalanceConstraints."""
    return RebalanceConstraints(
        max_weight_default=args.max_weight,
        new_capital_chf=args.new_capital)


def _summary_dataframe(plan_summary: Dict[str, float]) -> pd.DataFrame:
    """Single-row DataFrame for the Summary sheet."""
    return pd.DataFrame([plan_summary])


def main(argv=None):
    """Entry point for `python -m src.analysis.report`."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if not args.degiro_csv and not args.ibkr_csv:
        raise SystemExit(
            'report.py: must supply at least one of '
            '--degiro-csv or --ibkr-csv.')

    categories = _load_categories(args.ticker_categories)
    constraints = _build_constraints(args)
    provider = _build_data_provider()

    analyzer = PortfolioAnalyzer(
        degiro_csv_file_path=args.degiro_csv,
        ibkr_csv_file_path=args.ibkr_csv,
        data_provider=provider,
        base_currency='CHF')

    rebalancer = Rebalancer(
        analyzer=analyzer,
        strategy=get_strategy(args.strategy),
        constraints=constraints,
        ticker_categories=categories)
    plan = rebalancer.propose()

    sheets = {
        'Holdings': plan.holdings_df,
        'Target Weights': plan.target_weights_dataframe(),
        'Rebalancing Actions': plan.to_dataframe(),
        'Summary': _summary_dataframe(plan.summary()),
    }
    out_dir = pathlib.Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    DataExporter().export_to_excel(sheets, args.output)
    print(
        f'Report saved: {args.output} '
        f'(strategy={args.strategy}, '
        f'NAV CHF {plan.total_value_chf:,.2f}, '
        f'{len(plan.actions)} actions, '
        f'generated {plan.generated_at:%Y-%m-%d %H:%M})')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run integration test, confirm pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/analysis/test_report.py -v`
Expected: 2 passed.

- [ ] **Step 5: Smoke-test against real CSVs (optional but recommended before commit)**

Run:
```
cd /Users/rbarreira/Desktop/stock_market && python3 -m src.analysis.report \
    --strategy equal_weight \
    --new-capital 0 \
    --degiro-csv data/postprocess_data/processed_portfolio.csv \
    --ibkr-csv reports/ibkr/<YYYYMMDD>_<YYYYMMDD>_<ACCOUNT_ID>.csv \
    --ticker-categories data/ticker_categories.json \
    --output /tmp/report_smoketest.xlsx
```

Expected: prints `Report saved: /tmp/report_smoketest.xlsx ...` and the file exists. If it fails on a missing-category error, the user must add the ticker(s) to `data/ticker_categories.json` before retrying — known data gap (e.g. `PANW`).

- [ ] **Step 6: Commit (advisory)**

```bash
git add src/analysis/report.py tests/analysis/test_report.py
git commit -m "feat(report): rewrite as CLI orchestrator over Rebalancer"
```

---

## Task 10 — Delete legacy `src/modelling/rebalancing.py` + Phase 2 verification

**Files:**
- Delete: `src/modelling/rebalancing.py` (1064-line legacy).
- Modify: `tasks/todo.md` — append Phase 2 completion note.

- [ ] **Step 1: Confirm no current code imports the legacy module**

Run: `cd /Users/rbarreira/Desktop/stock_market && grep -rn "from src.modelling.rebalancing import\|import src.modelling.rebalancing$" src/ tests/ 2>&1 | grep -v "src.modelling.rebalancing\."`

Expected: only matches against the new `src.modelling.rebalancing.*` subpackage modules; NO matches against `src.modelling.rebalancing` as a module (with no submodule). The legacy file is now shadowed by the new subpackage of the same name — safe to delete.

If anything matches the legacy module form, fix the importer first before deleting.

- [ ] **Step 2: Delete the legacy file**

Run:
```
rm /Users/rbarreira/Desktop/stock_market/src/modelling/rebalancing.py
```

- [ ] **Step 3: Run the full test suite, confirm everything still passes**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: all tests still pass.

- [ ] **Step 4: Append Phase 2 completion to `tasks/todo.md`**

Append this block to `/Users/rbarreira/Desktop/stock_market/tasks/todo.md`:

```markdown

## Phase 2 — Rebalancing domain + rewritten report.py: COMPLETE (2026-05-14)

- src/modelling/rebalancing/                    [new package]
  - action.py    (RebalancingAction)
  - plan.py      (RebalancePlan)
  - rebalancer.py
  - strategies/  (base, mvo, equal_weight, inverse_vol, min_variance)
- src/analysis/report.py                        [rewritten as CLI orchestrator]
- src/modelling/rebalancing.py (1064-line legacy) [DELETED]

Verification:
- Full pytest suite passes.
- `python -m src.analysis.report --strategy equal_weight --new-capital 0 ...`
  end-to-end against real CSVs writes a multi-sheet xlsx.
- Strategy registry exposes 4 strategies: mvo, equal_weight, inverse_vol,
  min_variance.

Spec: docs/superpowers/specs/2026-05-13-repo-refactor-design.md
Plan: docs/superpowers/plans/2026-05-14-phase-2-rebalancing-and-report.md

Next: Phase 3 — Screening domain (per-exchange parquet caches + screener).
Plan to be authored by re-invoking superpowers:writing-plans against the
same spec, scoped to Phase 3.
```

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/modelling/rebalancing.py tasks/todo.md
git commit -m "chore(rebalancing): delete legacy 1064-line module; mark Phase 2 done"
```
(`git add src/modelling/rebalancing.py` stages the deletion.)

---

## Self-review

**Spec coverage** (cross-referenced against `docs/superpowers/specs/2026-05-13-repo-refactor-design.md` Section 7, Phase 2):
- [x] `src/modelling/rebalancing/action.py` — Task 1.
- [x] `src/modelling/rebalancing/plan.py` — Task 2.
- [x] `src/modelling/rebalancing/strategies/{base,mvo,equal_weight,inverse_vol,min_variance}.py` — Tasks 3-7.
- [x] `src/modelling/rebalancing/rebalancer.py` — Task 8.
- [x] `src/analysis/report.py` rewrite — Task 9.
- [x] Delete legacy `src/modelling/rebalancing.py` — Task 10.
- [x] Verification: `python -m src.analysis.report --strategy mvo` end-to-end — Task 9 Step 5.

**Placeholder scan:** no TBD/TODO/incomplete sections. Every code step shows the actual code. Test names are concrete. Paths are absolute.

**Type / signature consistency:**
- `Strategy.propose(holdings_df, prices, constraints, candidates=None, risk_free_rate=None) -> Dict[str, float]` — same signature in ABC, all 4 concrete subclasses, and the test that verifies it via `inspect`.
- `Rebalancer(analyzer, strategy, constraints, ticker_categories)` — constructor identical in implementation and tests.
- `RebalancingAction` field set + types — match Contract 5 in spec; tests verify each field.
- `RebalancePlan(actions, target_weights, strategy_name, constraints, holdings_df, generated_at)` — match Contract 5; tests verify with same args.
- `analyzer.get_holdings_snapshot(categories=...)` and `analyzer.get_price_panel()` — match the Phase 1 method signatures we added.

**Style compliance:** no `# ===` separators, no `# ~~~~~` separators, no authorship blocks. Type hints in signatures. Google-style docstrings throughout. 80-char width. snake_case + PascalCase.

**Anti-pattern audit:**
- No silent defaults: `_filter_eligible` raises `KeyError` on missing columns; strategies raise `RuntimeError` when held tickers lack price data; `Rebalancer._generate_actions` raises when a strategy proposes an unheld ticker.
- No bond ETFs in any strategy default. The `swiss_tax_filter` flag on `RebalanceConstraints` is wired to be honored at strategy level once the screener-fed strategies (Phase 4) need it.
- Module-level constants are lowercase (`trading_days_per_year`, `slsqp_max_iter`, `volatility_floor`, `target_weights_sum_tol`, `valid_actions`, `covariance_min_periods`).
