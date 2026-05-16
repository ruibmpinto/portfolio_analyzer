# Phase 4 — Screener-fed strategies wired into report.py

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three screener-fed rebalancing strategies (`MaxGrowthStrategy`, `RiskAdjustedStrategy`, `MinRiskStrategy`) that consume a list of `CandidateTicker` from `run_screener.py`'s JSON output, extend the `Rebalancer` to produce BUY actions for candidate tickers not currently held, wire a `candidates_from` JSON path into `report.py`'s zero-arg `main()`, and delete `src/orchestration/` entirely.

**Architecture:**
- Strategy signature (`propose(holdings_df, prices, constraints, candidates, risk_free_rate)`) is unchanged. The new strategies treat `candidates` as required (raise loudly when `None` or empty).
- Each strategy returns target weights summing to 1.0 over the union of (eligible current holdings, new candidate tickers, optional broad ETF, optional `CASH` pseudo-ticker).
- `RebalancingAction.valid_actions` is extended to `('BUY', 'REDUCE', 'HOLD', 'CASH')`. CASH inherits HOLD's `shares=0` invariant and carries the target cash CHF in `est_cost_chf`. A module-level `cash_ticker = 'CASH'` constant lives in `action.py` (so strategies can reference it without importing the rebalancer, keeping the dependency graph one-directional). The rebalancer emits the CASH action directly without going through the BUY/REDUCE/HOLD passes.
- Each screener-fed strategy exposes a `cash_weight: float` constructor parameter. Defaults restore legacy S1/S2/S3 from the (deleted) `construct_scenarios.py`: `MaxGrowth.cash_weight = 0.0`, `RiskAdjusted.cash_weight = 0.10`, `MinRisk.cash_weight = 0.40`. Per `feedback_swiss_tax_no_bonds.md` the user wants CHF cash as the defensive lever — never bond ETFs.
- Phase 4 supersedes the Phase 2 BUY-from-new_capital-only contract. `_generate_actions` now runs REDUCEs first and then funds BUYs from `new_capital_chf + Σ(REDUCE est_cost_chf) − target_cash_chf`. Implication: target cash is honoured as a hard reservation when funding is sufficient; CASH actions report intent that the realised post-trade cash matches (modulo whole-share rounding). This is a deliberate Phase 2-invariant change; the existing rebalancer test that asserts `total_buy ≤ new_capital_chf` is rewritten in Task 3 to the new funding invariant.
- The `Rebalancer` also handles non-CASH target tickers that aren't in `holdings_df` by fetching CHF prices via `analyzer.data_provider` (FX-converted using the candidate's `currency`).
- `report.py` accepts an optional `candidates_from: str` local variable. When set, candidates are loaded from JSON and passed to `Rebalancer.propose(candidates=...)`.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, pytest 8.3.4, pyarrow, yfinance, openpyxl.

**Style:** Match `src/analysis/core/portfolio.py` and `src/modelling/rebalancing/strategies/mvo.py`: type hints in signatures, Google-style docstrings, 80-char line width, single quotes for strings (double for docstrings), lowercase module-level constants, no banner separators, no `__author__` blocks. Per HANDOFF.md §4 this overrides the global `~/.claude/rules/code-style.md` for this repo.

**Conventions:**
- TDD per task (red → green → refactor). Tests live under `tests/` mirroring the source layout.
- Two-stage review after each task (spec compliance + code quality) per HANDOFF.md §3.
- "Commit (advisory)" steps are not authorizations — do not commit unless the user explicitly asks.
- Loud failures: raise `RuntimeError`/`KeyError` with the offending key/list, never silently fall back. Per `feedback_no_silent_defaults.md`.
- No bond ETFs in any default policy. Per `feedback_swiss_tax_no_bonds.md`.

---

## File Structure

### New files
- `src/modelling/rebalancing/pricing.py` — CHF pricing helper for non-held tickers.
- `src/modelling/rebalancing/strategies/max_growth.py` — `MaxGrowthStrategy`.
- `src/modelling/rebalancing/strategies/risk_adjusted.py` — `RiskAdjustedStrategy`.
- `src/modelling/rebalancing/strategies/min_risk.py` — `MinRiskStrategy`.
- `tests/modelling/rebalancing/test_pricing.py`
- `tests/modelling/rebalancing/test_max_growth.py`
- `tests/modelling/rebalancing/test_risk_adjusted.py`
- `tests/modelling/rebalancing/test_min_risk.py`
- `tests/modelling/rebalancing/test_rebalancer_with_candidates.py`
- `tests/analysis/test_report_with_candidates.py`
- `tests/screening/test_candidate_io.py`

### Modified files
- `src/screening/candidate.py` — add `load_candidates_from_json(path)` helper.
- `src/screening/__init__.py` — re-export `load_candidates_from_json`.
- `src/modelling/rebalancing/action.py` — extend `valid_actions` with `'CASH'`; require `shares == 0` for CASH actions.
- `src/modelling/rebalancing/rebalancer.py` — handle non-held tickers; accept and use candidate price lookup; recognise `cash_ticker` and emit CASH actions.
- `src/modelling/rebalancing/strategies/__init__.py` — register the three new strategies.
- `src/modelling/rebalancing/plan.py` — `summary()` gains `total_cash_chf`.
- `src/analysis/report.py` — add `candidates_from` local var and JSON loading.
- `tests/modelling/rebalancing/test_action_and_plan.py` — extend with CASH-action validation + `total_cash_chf` assertions.

### Deleted files
- `src/orchestration/__init__.py`
- `src/orchestration/construct_scenarios.py`
- `src/orchestration/` (directory)

---

## Task 1: Add a JSON loader for `CandidateTicker`

**Files:**
- Modify: `src/screening/candidate.py`
- Modify: `src/screening/__init__.py` (re-export)
- Test: `tests/screening/test_candidate_io.py` (new)

**Why:** `run_screener.py` writes JSON rows shaped like `{ticker, exchange, currency, sector, market_cap, composite_score, metrics: {...}}`. `report.py` needs to reconstruct `CandidateTicker` instances from that file. Centralise the loader in `src/screening/candidate.py` so both runtime code and tests use the same path.

- [ ] **Step 1: Write the failing test**

Create `tests/screening/test_candidate_io.py`:

```python
"""Tests for CandidateTicker JSON round-tripping."""

import json
from types import MappingProxyType

import pytest

from src.screening.candidate import (
    CandidateTicker, load_candidates_from_json)


def test_load_candidates_from_json_roundtrips(tmp_path):
    """Loader reconstructs CandidateTicker fields verbatim."""
    rows = [
        {
            'ticker': 'AAPL', 'exchange': 'NASDAQ',
            'currency': 'USD', 'sector': 'Technology',
            'market_cap': 3.0e12, 'composite_score': 0.87,
            'metrics': {'momentum': 0.72, 'sharpe': 1.4},
        },
        {
            'ticker': 'NESN.SW', 'exchange': 'SIX',
            'currency': 'CHF', 'sector': 'Consumer Defensive',
            'market_cap': 2.5e11, 'composite_score': 0.61,
            'metrics': {'momentum': 0.35, 'sharpe': 0.9},
        },
    ]
    path = tmp_path / 'screener.json'
    path.write_text(json.dumps(rows))
    candidates = load_candidates_from_json(str(path))
    assert [c.ticker for c in candidates] == ['AAPL', 'NESN.SW']
    assert candidates[0].composite_score == pytest.approx(0.87)
    assert candidates[1].currency == 'CHF'
    assert isinstance(candidates[0].metrics, MappingProxyType)
    assert candidates[0].metrics['momentum'] == pytest.approx(0.72)


def test_load_candidates_raises_on_missing_field(tmp_path):
    """Loader raises KeyError listing the missing field."""
    rows = [{
        'ticker': 'AAPL', 'exchange': 'NASDAQ',
        # currency missing
        'sector': 'Technology', 'market_cap': 3e12,
        'composite_score': 0.5, 'metrics': {}}]
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(rows))
    with pytest.raises(KeyError, match='currency'):
        load_candidates_from_json(str(path))


def test_load_candidates_empty_file(tmp_path):
    """Empty list yields an empty list."""
    path = tmp_path / 'empty.json'
    path.write_text('[]')
    assert load_candidates_from_json(str(path)) == []
```

- [ ] **Step 2: Verify test fails**

Run: `pytest tests/screening/test_candidate_io.py -v`
Expected: ImportError on `load_candidates_from_json`.

- [ ] **Step 3: Implement the loader**

Append to `src/screening/candidate.py`:

```python
import json
from typing import List


def load_candidates_from_json(path: str) -> List[CandidateTicker]:
    """Load a list of CandidateTicker from a JSON file.

    Args:
        path: Path to a JSON file containing a list of dicts
            with keys ``ticker, exchange, currency, sector,
            market_cap, composite_score, metrics``. The shape
            matches what ``run_screener.py`` writes for
            ``--output-format json``.

    Returns:
        List of CandidateTicker in the order they appear in the
        file. Returns an empty list when the file contains
        ``[]``.

    Raises:
        KeyError: When a required field is missing from any row.
        ValueError: When ``composite_score`` is outside [0, 1].
    """
    with open(path) as f:
        rows = json.load(f)
    candidates: List[CandidateTicker] = []
    for row in rows:
        required = (
            'ticker', 'exchange', 'currency', 'sector',
            'market_cap', 'composite_score', 'metrics')
        for key in required:
            if key not in row:
                raise KeyError(
                    f'load_candidates_from_json: row missing '
                    f'required field {key!r}.')
        candidates.append(CandidateTicker(
            ticker=row['ticker'],
            exchange=row['exchange'],
            currency=row['currency'],
            sector=row['sector'],
            market_cap=float(row['market_cap']),
            composite_score=float(row['composite_score']),
            metrics=dict(row['metrics'])))
    return candidates
```

- [ ] **Step 4: Re-export from the package**

Modify `src/screening/__init__.py`:

```python
from src.screening.candidate import (
    CandidateTicker, load_candidates_from_json)
```

and add `'load_candidates_from_json'` to `__all__`.

- [ ] **Step 5: Verify tests pass**

Run: `pytest tests/screening/test_candidate_io.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Full suite green-check**

Run: `pytest tests/ -v`
Expected: 122 prior + 3 new = 125 PASS.

- [ ] **Step 7: Commit (advisory)**

```bash
git add src/screening/candidate.py src/screening/__init__.py \
        tests/screening/test_candidate_io.py
git commit -m "feat(screening): JSON loader for CandidateTicker"
```

---

## Task 2: CHF pricing helper for non-held tickers

**Files:**
- Create: `src/modelling/rebalancing/pricing.py`
- Test: `tests/modelling/rebalancing/test_pricing.py`

**Why:** Screener-fed strategies will propose target weights for tickers that are not currently held. `Rebalancer._generate_actions` needs a CHF price for each such ticker to convert CHF allocation into whole shares. Centralise the FX-aware lookup so the Rebalancer stays focused on action generation.

**Behaviour:**
- `fetch_chf_price(data_provider, ticker, currency)` returns the latest CHF price.
- For `currency == 'CHF'`: returns `data_provider.get_current_price(ticker)` directly.
- For `currency == 'GBp'` (London pence): native price / 100 × GBPCHF latest.
- For other currencies (USD, EUR, GBP, ...): native price × `<CURRENCY>CHF=X` latest.
- Raises `RuntimeError` on lookup failure (no silent fallback).

- [ ] **Step 1: Write failing tests**

Create `tests/modelling/rebalancing/test_pricing.py`:

```python
"""Tests for CHF pricing helper for non-held tickers."""

import pytest

from src.modelling.rebalancing.pricing import fetch_chf_price


class FakeProvider:
    """Minimal data_provider stub for pricing tests."""

    def __init__(self, prices):
        self._prices = prices

    def get_current_price(self, ticker):
        if ticker not in self._prices:
            raise ValueError(
                f'FakeProvider: no price for {ticker!r}.')
        return self._prices[ticker]


def test_fetch_chf_price_for_chf_ticker():
    """No FX call for a CHF-denominated ticker."""
    fp = FakeProvider({'NESN.SW': 95.0})
    assert fetch_chf_price(fp, 'NESN.SW', 'CHF') == 95.0


def test_fetch_chf_price_for_usd_ticker():
    """USD price multiplied by USDCHF=X."""
    fp = FakeProvider({'AAPL': 200.0, 'USDCHF=X': 0.9})
    price = fetch_chf_price(fp, 'AAPL', 'USD')
    assert price == pytest.approx(180.0)


def test_fetch_chf_price_for_gbp_ticker():
    """GBP price multiplied by GBPCHF=X."""
    fp = FakeProvider({'BARC.L': 1.50, 'GBPCHF=X': 1.1})
    price = fetch_chf_price(fp, 'BARC.L', 'GBP')
    assert price == pytest.approx(1.65)


def test_fetch_chf_price_for_gbx_pence_ticker():
    """GBp pence price divided by 100, then GBPCHF=X."""
    fp = FakeProvider({'AAF.L': 150.0, 'GBPCHF=X': 1.1})
    price = fetch_chf_price(fp, 'AAF.L', 'GBp')
    assert price == pytest.approx(1.65)


def test_fetch_chf_price_raises_when_native_missing():
    """No native quote raises a RuntimeError."""
    fp = FakeProvider({})
    with pytest.raises(RuntimeError, match='AAPL'):
        fetch_chf_price(fp, 'AAPL', 'USD')


def test_fetch_chf_price_raises_when_fx_missing():
    """Missing FX pair raises a RuntimeError naming the pair."""
    fp = FakeProvider({'AAPL': 200.0})
    with pytest.raises(RuntimeError, match='USDCHF=X'):
        fetch_chf_price(fp, 'AAPL', 'USD')
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/modelling/rebalancing/test_pricing.py -v`
Expected: ImportError (module does not exist).

- [ ] **Step 3: Implement the helper**

Create `src/modelling/rebalancing/pricing.py`:

```python
"""CHF price lookup for tickers not present in current holdings.

Used by Rebalancer._generate_actions when a strategy proposes a
target weight on a ticker that the analyzer's holdings_df does
not contain. The helper converts a native-currency quote from
the data_provider to CHF via a single FX pair lookup, supporting
'CHF', 'GBp' (London pence), and any major currency for which
yfinance exposes a <CURRENCY>CHF=X pair.

Loud failures only: native or FX lookup misses raise
RuntimeError. No silent fallbacks.
"""

from typing import Protocol


pence_per_pound = 100.0


class _PriceProvider(Protocol):
    """Minimal interface of DataProvider used here."""

    def get_current_price(self, ticker: str) -> float: ...


def fetch_chf_price(
    data_provider: _PriceProvider,
    ticker: str,
    currency: str) -> float:
    """Return the latest price for `ticker` in CHF.

    Args:
        data_provider: A DataProvider with ``get_current_price``.
        ticker: Native exchange symbol.
        currency: ISO-4217 listing currency, or 'GBp' for London
            pence.

    Returns:
        Price in CHF, strictly positive.

    Raises:
        RuntimeError: If the native quote or the FX pair quote
            cannot be fetched.
    """
    native_price = _fetch_native(data_provider, ticker)
    if currency == 'CHF':
        return native_price
    if currency == 'GBp':
        fx = _fetch_native(data_provider, 'GBPCHF=X')
        return (native_price / pence_per_pound) * fx
    pair = f'{currency}CHF=X'
    fx = _fetch_native(data_provider, pair)
    return native_price * fx


def _fetch_native(data_provider, ticker: str) -> float:
    try:
        return float(data_provider.get_current_price(ticker))
    except Exception as e:
        raise RuntimeError(
            f'fetch_chf_price: data_provider.get_current_price'
            f'({ticker!r}) failed: {type(e).__name__}: {e}')
```

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/modelling/rebalancing/test_pricing.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/modelling/rebalancing/pricing.py \
        tests/modelling/rebalancing/test_pricing.py
git commit -m "feat(rebalancing): CHF pricing helper for non-held tickers"
```

---

## Task 3: Rebalancer + Action extension (CASH bucket + non-held tickers)

**Files:**
- Modify: `src/modelling/rebalancing/action.py` (extend `valid_actions`).
- Modify: `src/modelling/rebalancing/rebalancer.py` (handle non-held tickers + CASH).
- Modify: `src/modelling/rebalancing/plan.py` (`summary()` gains `total_cash_chf`).
- Modify: `tests/modelling/rebalancing/test_action_and_plan.py` (CASH-action + summary tests).
- Test (new): `tests/modelling/rebalancing/test_rebalancer_with_candidates.py`.

**Why:** Four coupled extensions land together:

1. The `Rebalancer` today raises `RuntimeError` for any target ticker missing from `holdings_df`. Screener-fed strategies need to propose brand-new positions. For these the action generator should treat `current_value` / `current_w_pct` as 0 and source the CHF price from a candidate lookup.
2. Strategies (Tasks 4-6) opt into an explicit cash bucket via a `cash_ticker = 'CASH'` pseudo-ticker in `target_weights`. The Rebalancer must recognise it without going through the BUY/REDUCE/HOLD passes and emit a single `CASH` action whose `est_cost_chf` records `target_w_cash * (NAV + new_capital)`.
3. The plan's `summary()` exposes a `total_cash_chf` aggregate for the Excel report.
4. The BUY pool is augmented to recycle REDUCE proceeds and respect the cash reservation, so realised post-trade cash actually matches the CASH action's target intent (modulo whole-share rounding). This deliberately changes Phase 2 semantics for portfolio-only strategies; one existing test asserting `total_buy ≤ new_capital_chf` is rewritten to the new funding invariant.

**Behaviour:**
- `RebalancingAction.valid_actions = ('BUY', 'REDUCE', 'HOLD', 'CASH')`. CASH inherits HOLD's `shares == 0` invariant. `est_cost_chf` carries the target cash CHF magnitude (non-negative); `note` is free-form (default `'Cash reserve'`).
- `cash_ticker = 'CASH'` is a module-level constant in `action.py` (lowercase per `feedback_no_caps_globals.md`). The rebalancer and the screener-fed strategies import it from there.
- `Rebalancer.propose(candidates=...)`:
  - Skips `cash_ticker` when building the `candidate_prices_chf` lookup (CASH has no CHF price to fetch).
  - Routes the CASH entry to a dedicated emit, bypassing BUY/REDUCE/HOLD classification.
- For non-held, non-CASH target tickers: synthesise a delta entry with `current_w_pct = 0`, `price_chf` from the candidate lookup, `current_value = 0`.
- New BUY pool funding (Option A — always recycle): the rebalancer now runs the REDUCE pass before the BUY pass and sets `buy_pool = max(0, new_capital_chf + Σ(REDUCE est_cost_chf) − target_cash_chf)`. The largest-gap-first BUY allocation then runs against `buy_pool` (instead of just `new_capital_chf`). `target_cash_chf` is `cash_weight × (NAV + new_capital)` extracted from `target_weights` before the equity passes; defaults to 0 when no CASH entry exists.
- A target ticker that is neither in `holdings_df`, nor in `candidate_prices_chf`, nor equal to `cash_ticker` raises `RuntimeError` naming it.
- `RebalancePlan.summary()` gains `total_cash_chf` = sum of `est_cost_chf` over CASH actions (0.0 when no cash bucket).
- Existing test `test_rebalancer_buy_capital_capped_at_new_capital` is renamed to `test_rebalancer_buy_pool_includes_reduce_proceeds_minus_cash` and rewritten to assert `total_buy ≤ new_capital + total_reduce - target_cash` (the new funding invariant).

- [ ] **Step 1a: Write failing action.py tests**

Append to `tests/modelling/rebalancing/test_action_and_plan.py` (or add fresh `test_action_cash.py` if cleaner; either placement is fine — the implementer chooses, but pick one and stick with it):

```python
import pytest

from src.modelling.rebalancing.action import RebalancingAction


def test_cash_action_is_a_valid_action():
    """CASH is in the action vocabulary."""
    a = RebalancingAction(
        ticker='CASH', action='CASH', shares=0,
        est_cost_chf=400.0, current_wt_pct=0.0,
        target_wt_pct=40.0, note='Cash reserve')
    assert a.action == 'CASH'


def test_cash_action_requires_zero_shares():
    """CASH inherits the shares==0 invariant."""
    with pytest.raises(ValueError, match='CASH'):
        RebalancingAction(
            ticker='CASH', action='CASH', shares=1,
            est_cost_chf=400.0, current_wt_pct=0.0,
            target_wt_pct=40.0, note='')


def test_cash_action_allows_positive_est_cost():
    """est_cost_chf carries the target cash magnitude."""
    a = RebalancingAction(
        ticker='CASH', action='CASH', shares=0,
        est_cost_chf=1234.56, current_wt_pct=0.0,
        target_wt_pct=20.0, note='')
    assert a.est_cost_chf == pytest.approx(1234.56)
```

- [ ] **Step 1b: Write failing plan.py summary test**

Append to `tests/modelling/rebalancing/test_action_and_plan.py`:

```python
from datetime import datetime

import pandas as pd

from src.modelling.rebalancing.plan import RebalancePlan
from src.shared.constraints import default_constraints


def test_plan_summary_includes_total_cash_chf():
    """summary() reports total_cash_chf summed over CASH rows."""
    actions = [
        RebalancingAction(
            ticker='AMZN', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=60.0,
            target_wt_pct=60.0, note=''),
        RebalancingAction(
            ticker='CASH', action='CASH', shares=0,
            est_cost_chf=400.0, current_wt_pct=0.0,
            target_wt_pct=40.0, note='Cash reserve'),
    ]
    target_weights = {'AMZN': 0.6, 'CASH': 0.4}
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    plan = RebalancePlan(
        actions=actions,
        target_weights=target_weights,
        strategy_name='fake',
        constraints=default_constraints(),
        holdings_df=holdings,
        generated_at=datetime.now())
    s = plan.summary()
    assert s['total_cash_chf'] == pytest.approx(400.0)


def test_plan_summary_total_cash_zero_when_no_cash_actions():
    """No CASH action -> total_cash_chf == 0."""
    actions = [
        RebalancingAction(
            ticker='AMZN', action='HOLD', shares=0,
            est_cost_chf=0.0, current_wt_pct=100.0,
            target_wt_pct=100.0, note='')]
    plan = RebalancePlan(
        actions=actions,
        target_weights={'AMZN': 1.0},
        strategy_name='fake',
        constraints=default_constraints(),
        holdings_df=pd.DataFrame([
            {'ticker': 'AMZN', 'value_chf': 1000.0,
             'weight_pct': 100.0, 'category': 'core'}]),
        generated_at=datetime.now())
    assert plan.summary()['total_cash_chf'] == 0.0
```

- [ ] **Step 1c: Write failing rebalancer tests (non-held + CASH)**

Create `tests/modelling/rebalancing/test_rebalancer_with_candidates.py`:

```python
"""Tests for Rebalancer with screener-fed strategies + CASH."""

import pandas as pd
import pytest

from src.modelling.rebalancing.rebalancer import Rebalancer
from src.modelling.rebalancing.strategies.base import Strategy
from src.screening.candidate import CandidateTicker
from src.shared.constraints import RebalanceConstraints


class FakeAnalyzer:
    """Minimal PortfolioAnalyzer stub."""

    def __init__(self, holdings_df, prices, data_provider):
        self._h = holdings_df
        self._p = prices
        self.data_provider = data_provider

    def get_holdings_snapshot(self, categories=None):
        return self._h.copy()

    def get_price_panel(self):
        return self._p.copy()


class FakeProvider:
    def __init__(self, prices):
        self._prices = prices

    def get_current_price(self, ticker):
        if ticker not in self._prices:
            raise ValueError(f'no price for {ticker}')
        return self._prices[ticker]


class FakeStrategy(Strategy):
    """Returns fixed target weights for testing."""

    name = 'fake_phase4'

    def __init__(self, target_weights):
        self._tw = target_weights

    def propose(
        self, holdings_df, prices, constraints,
        candidates=None, risk_free_rate=None):
        return dict(self._tw)


def _holdings(values_chf, prices_chf):
    rows = []
    total = sum(values_chf.values())
    for t, v in values_chf.items():
        rows.append({
            'ticker': t, 'shares': 1, 'value_chf': v,
            'weight_pct': v / total * 100.0,
            'currency': 'CHF', 'sector': 'X',
            'price_chf': prices_chf[t], 'category': 'core'})
    return pd.DataFrame(rows)


def _make_candidate(ticker, currency='USD'):
    return CandidateTicker(
        ticker=ticker, exchange='NASDAQ',
        currency=currency, sector='Tech',
        market_cap=1.0e11, composite_score=0.7,
        metrics={'sharpe': 1.2, 'max_drawdown': -0.2})


def test_rebalancer_generates_buy_for_new_candidate_ticker():
    """Target weight on a non-held ticker yields a BUY action."""
    holdings = _holdings(
        {'AMZN': 1000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0, 199.0]})
    provider = FakeProvider(
        {'NEWCO': 100.0, 'USDCHF=X': 0.9})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({'AMZN': 0.5, 'NEWCO': 0.5})
    candidates = [_make_candidate('NEWCO', 'USD')]
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=candidates)

    actions_by_ticker = {a.ticker: a for a in plan.actions}
    assert 'NEWCO' in actions_by_ticker
    new = actions_by_ticker['NEWCO']
    assert new.action == 'BUY'
    assert new.shares > 0
    assert new.current_wt_pct == pytest.approx(0.0)
    assert new.target_wt_pct == pytest.approx(50.0)


def test_rebalancer_raises_when_candidate_price_missing():
    """Target weight on unknown ticker raises."""
    holdings = _holdings(
        {'AMZN': 1000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0]})
    provider = FakeProvider({})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({'AMZN': 0.5, 'GHOST': 0.5})
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    with pytest.raises(RuntimeError, match='GHOST'):
        rebalancer.propose(candidates=[])


def test_rebalancer_uses_candidate_currency_for_fx():
    """A USD candidate uses USDCHF=X for the CHF conversion."""
    holdings = _holdings(
        {'AMZN': 10000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0]})
    provider = FakeProvider(
        {'NEWCO': 100.0, 'USDCHF=X': 0.5})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({'AMZN': 0.5, 'NEWCO': 0.5})
    candidates = [_make_candidate('NEWCO', 'USD')]
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=candidates)

    new = next(a for a in plan.actions if a.ticker == 'NEWCO')
    assert new.shares == 40
    assert new.est_cost_chf == pytest.approx(2000.0)


def test_rebalancer_emits_cash_action_for_cash_ticker():
    """target_weights['CASH'] -> single CASH action with target CHF."""
    holdings = _holdings(
        {'AMZN': 1000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0]})
    provider = FakeProvider({})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    # NAV=1000, new_capital=2000 -> target_total=3000.
    # CASH 30% -> est_cost_chf = 900.
    strategy = FakeStrategy({'AMZN': 0.7, 'CASH': 0.3})
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=None)

    cash_actions = [a for a in plan.actions if a.ticker == 'CASH']
    assert len(cash_actions) == 1
    cash = cash_actions[0]
    assert cash.action == 'CASH'
    assert cash.shares == 0
    assert cash.est_cost_chf == pytest.approx(900.0)
    assert cash.current_wt_pct == pytest.approx(0.0)
    assert cash.target_wt_pct == pytest.approx(30.0)
    # No FX lookup attempted for CASH (FakeProvider has empty map)
    assert plan.summary()['total_cash_chf'] == pytest.approx(900.0)


def test_rebalancer_does_not_request_price_for_cash():
    """CASH ticker is never sent to fetch_chf_price."""
    holdings = _holdings(
        {'AMZN': 1000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0]})
    # Provider with no prices at all - any FX call would fail.
    provider = FakeProvider({})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({'AMZN': 0.5, 'CASH': 0.5})
    constraints = RebalanceConstraints(
        new_capital_chf=1000.0,
        excluded_categories=(), category_caps={})
    rebalancer = Rebalancer(analyzer, strategy, constraints)
    # Should not raise even though provider has nothing.
    plan = rebalancer.propose(candidates=None)
    assert 'CASH' in {a.ticker for a in plan.actions}


def test_rebalancer_recycles_reduce_proceeds_into_buy_pool():
    """REDUCE proceeds + new_capital - target_cash fund BUYs.

    NAV=10000 in AMZN, new_capital=2000, target
    AMZN 50% / NEWCO 30% / CASH 20%.

    With Phase 2 logic NEWCO would BUY only 2000 (capped at
    new_capital). With Phase 4 recycling: AMZN REDUCEs 4000,
    target_cash = 0.20*12000 = 2400, BUY pool =
    2000 + 4000 - 2400 = 3600 -> NEWCO BUY hits its 3600
    target exactly.
    """
    holdings = _holdings(
        {'AMZN': 10000.0}, {'AMZN': 200.0})
    prices = pd.DataFrame({'AMZN': [200.0, 201.0]})
    # NEWCO priced in CHF at 100 (provider returns CHF directly
    # when candidate currency is CHF).
    provider = FakeProvider({'NEWCO': 100.0})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    strategy = FakeStrategy({
        'AMZN': 0.5, 'NEWCO': 0.3, 'CASH': 0.2})
    candidates = [CandidateTicker(
        ticker='NEWCO', exchange='SIX', currency='CHF',
        sector='Tech', market_cap=1e10, composite_score=0.7,
        metrics={})]
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=candidates)

    by_ticker = {a.ticker: a for a in plan.actions}
    assert by_ticker['NEWCO'].action == 'BUY'
    assert by_ticker['NEWCO'].est_cost_chf == pytest.approx(
        3600.0)
    assert by_ticker['NEWCO'].shares == 36
    assert by_ticker['AMZN'].action == 'REDUCE'
    assert by_ticker['CASH'].action == 'CASH'
    assert by_ticker['CASH'].est_cost_chf == pytest.approx(
        2400.0)


def test_rebalancer_cash_reservation_caps_buy_pool():
    """target_cash > funding -> BUY pool is 0, not negative.

    NAV=10000, new_capital=2000, target AMZN 50% / CASH 50%.
    AMZN REDUCEs 1000 (target 5000). target_cash =
    0.5*12000 = 6000. Funding = 2000 + 1000 - 6000 = -3000
    -> clamped to 0 -> no BUYs even if any ticker were
    underweight.
    """
    holdings = _holdings(
        {'AMZN': 6000.0, 'B': 4000.0},
        {'AMZN': 200.0, 'B': 100.0})
    prices = pd.DataFrame({
        'AMZN': [200.0, 201.0],
        'B': [100.0, 101.0]})
    provider = FakeProvider({})
    analyzer = FakeAnalyzer(holdings, prices, provider)
    # Force B underweight (current 40% -> target 50%) while
    # AMZN overweight enough that BUY pool would be negative.
    strategy = FakeStrategy({
        'AMZN': 0.3, 'B': 0.2, 'CASH': 0.5})
    constraints = RebalanceConstraints(
        new_capital_chf=2000.0,
        excluded_categories=(), category_caps={})

    rebalancer = Rebalancer(analyzer, strategy, constraints)
    plan = rebalancer.propose(candidates=None)

    total_buy = sum(
        a.est_cost_chf for a in plan.actions
        if a.action == 'BUY')
    assert total_buy == pytest.approx(0.0)
```

- [ ] **Step 2: Verify all the new tests fail**

Run: `pytest tests/modelling/rebalancing/test_action_and_plan.py tests/modelling/rebalancing/test_rebalancer_with_candidates.py -v`
Expected: action-CASH tests fail (`'CASH' not in valid_actions`), summary-total_cash_chf tests fail (`KeyError: 'total_cash_chf'`), rebalancer-non-held tests fail with `target weight for ticker ... not in current holdings`, rebalancer-CASH tests fail (CASH treated as missing ticker).

- [ ] **Step 3: Extend `RebalancingAction`**

Edit `src/modelling/rebalancing/action.py`:

```python
valid_actions = ('BUY', 'REDUCE', 'HOLD', 'CASH')
cash_ticker = 'CASH'
```

And in `__post_init__` replace the `HOLD` zero-shares branch with:

```python
if self.action in ('HOLD', 'CASH') and self.shares != 0:
    raise ValueError(
        f'{self.action} action must have shares=0, '
        f'got {self.shares}.')
```

(Leave the other invariants — non-negative shares, non-negative est_cost_chf, action-in-vocab — untouched. The `est_cost_chf >= 0` rule already covers the CASH magnitude requirement.)

Update the module docstring's "BUY/REDUCE/HOLD" mention to "BUY/REDUCE/HOLD/CASH" and the Attributes block for `action` accordingly. Document `cash_ticker` in the module docstring as the canonical name for the cash pseudo-ticker.

- [ ] **Step 4: Extend `RebalancePlan.summary` with `total_cash_chf`**

In `src/modelling/rebalancing/plan.py`, in `summary()`:

```python
total_cash = 0.0
# in the existing per-action loop, add:
elif a.action == 'CASH':
    total_cash += a.est_cost_chf
```

Add `'total_cash_chf': total_cash` to the returned dict. Update `summary()`'s docstring to mention the new key.

- [ ] **Step 5: Extend the Rebalancer**

Edit `src/modelling/rebalancing/rebalancer.py`. Add at module scope:

```python
from src.modelling.rebalancing.action import (
    RebalancingAction, cash_ticker)
from src.modelling.rebalancing.pricing import fetch_chf_price


cash_action_note = 'Cash reserve'
```

(`RebalancingAction` is already imported in the existing file; merge with the existing import line. `cash_ticker` is the new addition.)

Replace `propose` and `_generate_actions`. Add `_build_candidate_prices` and `_make_cash_action`. Final relevant portions:

```python
class Rebalancer:
    # ... (existing docstring + __init__ unchanged)

    def propose(
        self, candidates: Optional[List] = None) -> RebalancePlan:
        """Build a RebalancePlan from the analyzer's state."""
        holdings_df = self.analyzer.get_holdings_snapshot(
            categories=self.ticker_categories)
        prices = self.analyzer.get_price_panel()
        target_weights = self.strategy.propose(
            holdings_df, prices, self.constraints,
            candidates=candidates)
        held = set(holdings_df['ticker'])
        new_targets = [
            t for t in target_weights
            if t not in held and t != cash_ticker]
        candidate_prices_chf = self._build_candidate_prices(
            new_targets, candidates)
        actions = self._generate_actions(
            target_weights, holdings_df, candidate_prices_chf)
        return RebalancePlan(
            actions=actions,
            target_weights=target_weights,
            strategy_name=self.strategy.name,
            constraints=self.constraints,
            holdings_df=holdings_df,
            generated_at=datetime.now())

    def _build_candidate_prices(
        self,
        new_targets: List[str],
        candidates: Optional[List]) -> Dict[str, float]:
        """CHF price lookup for non-held, non-CASH targets.

        Args:
            new_targets: Tickers in target_weights that are not
                in holdings_df and are not the CASH pseudo-ticker.
            candidates: List of CandidateTicker passed by the
                caller; may be None.

        Returns:
            Dict {ticker: chf_price} covering every new_target.

        Raises:
            RuntimeError: When a new_target has no matching
                CandidateTicker, or when fetch_chf_price fails.
        """
        if not new_targets:
            return {}
        by_ticker = {
            c.ticker: c for c in (candidates or [])}
        prices_chf: Dict[str, float] = {}
        for t in new_targets:
            if t not in by_ticker:
                raise RuntimeError(
                    f'Rebalancer: target weight on ticker '
                    f'{t!r} is not in current holdings and no '
                    f'matching CandidateTicker was supplied.')
            c = by_ticker[t]
            prices_chf[t] = fetch_chf_price(
                self.analyzer.data_provider,
                c.ticker, c.currency)
        return prices_chf

    def _generate_actions(
        self,
        target_weights: Dict[str, float],
        holdings_df: pd.DataFrame,
        candidate_prices_chf: Dict[str, float]) -> List[
            RebalancingAction]:
        """Turn target weights + snapshot into actions.

        Phase 4 order: emit the CASH action first, run the
        REDUCE pass to size the recycling pool, then run the
        BUY pass against ``new_capital + total_reduce -
        target_cash``, then close out in-band tickers as HOLD.
        Tickers not in holdings_df are treated as
        current_value=0 / current_w_pct=0 and priced from
        candidate_prices_chf. The CASH pseudo-ticker bypasses
        the BUY/REDUCE/HOLD passes; it emits a single CASH
        action recording target_w * (NAV + new_capital).
        """
        if not target_weights:
            return []
        snap_by_ticker = holdings_df.set_index('ticker')
        total_value = float(holdings_df['value_chf'].sum())
        target_total = (
            total_value + self.constraints.new_capital_chf)

        actions: List[RebalancingAction] = []
        target_cash_chf = 0.0
        deltas = []
        for ticker, target_w in target_weights.items():
            if ticker == cash_ticker:
                cash_action = self._make_cash_action(
                    target_w, target_total)
                actions.append(cash_action)
                target_cash_chf = cash_action.est_cost_chf
                continue
            if ticker in snap_by_ticker.index:
                row = snap_by_ticker.loc[ticker]
                current_value = float(row['value_chf'])
                current_w_pct = float(row['weight_pct'])
                price_chf = float(row['price_chf'])
            else:
                if ticker not in candidate_prices_chf:
                    raise RuntimeError(
                        f'Rebalancer: target weight for '
                        f'ticker {ticker!r} has no current '
                        f'holding and no candidate price.')
                current_value = 0.0
                current_w_pct = 0.0
                price_chf = float(candidate_prices_chf[ticker])
            target_value = target_w * target_total
            deltas.append({
                'ticker': ticker,
                'current_w_pct': current_w_pct,
                'target_w_pct': target_w * 100.0,
                'delta_chf': target_value - current_value,
                'price_chf': price_chf,
            })

        lower, upper = self.constraints.rebalance_band_chf

        # REDUCE pass FIRST. Whole-share rounding decides
        # actual proceeds; total_reduce_chf feeds the BUY pool.
        overweight = sorted(
            [d for d in deltas if d['delta_chf'] < lower],
            key=lambda d: d['delta_chf'])
        total_reduce_chf = 0.0
        for d in overweight:
            magnitude = abs(d['delta_chf'])
            shares = 0
            if d['price_chf'] > 0:
                shares = int(magnitude / d['price_chf'])
            if shares <= 0:
                actions.append(RebalancingAction(
                    ticker=d['ticker'], action='HOLD',
                    shares=0, est_cost_chf=0.0,
                    current_wt_pct=d['current_w_pct'],
                    target_wt_pct=d['target_w_pct'],
                    note='Overweight but <1 share to reduce'))
                continue
            proceeds = shares * d['price_chf']
            total_reduce_chf += proceeds
            actions.append(RebalancingAction(
                ticker=d['ticker'], action='REDUCE',
                shares=shares, est_cost_chf=proceeds,
                current_wt_pct=d['current_w_pct'],
                target_wt_pct=d['target_w_pct'],
                note='Only if held >6 months'))

        # BUY pool augments new_capital with REDUCE proceeds
        # and reserves the cash target.
        buy_pool = max(
            0.0,
            self.constraints.new_capital_chf
            + total_reduce_chf
            - target_cash_chf)

        underweight = sorted(
            [d for d in deltas if d['delta_chf'] > upper],
            key=lambda d: -d['delta_chf'])
        remaining = buy_pool
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

        for d in underweight:
            if d['ticker'] in bought_tickers:
                continue
            actions.append(RebalancingAction(
                ticker=d['ticker'], action='HOLD',
                shares=0, est_cost_chf=0.0,
                current_wt_pct=d['current_w_pct'],
                target_wt_pct=d['target_w_pct'],
                note='Underweight but no capital available'))

        # In-band HOLDs
        for d in deltas:
            if lower <= d['delta_chf'] <= upper:
                actions.append(RebalancingAction(
                    ticker=d['ticker'], action='HOLD',
                    shares=0, est_cost_chf=0.0,
                    current_wt_pct=d['current_w_pct'],
                    target_wt_pct=d['target_w_pct'],
                    note=''))
        return actions

    def _make_cash_action(
        self,
        target_w: float,
        target_total: float) -> RebalancingAction:
        """Emit the CASH bucket as a single CASH action.

        Args:
            target_w: Cash weight fraction in [0, 1].
            target_total: NAV + new_capital_chf.

        Returns:
            A CASH RebalancingAction whose ``est_cost_chf`` is
            the target cash CHF magnitude.
        """
        target_cash_chf = float(target_w) * float(target_total)
        return RebalancingAction(
            ticker=cash_ticker,
            action='CASH',
            shares=0,
            est_cost_chf=max(0.0, target_cash_chf),
            current_wt_pct=0.0,
            target_wt_pct=float(target_w) * 100.0,
            note=cash_action_note)
```

**Preservation rules:**
- Every action that was emitted before Phase 4 is still emitted (REDUCE rows, underweight-HOLD fallbacks, in-band HOLDs). The only behavioural change is the BUY pool magnitude.
- Do NOT silently downgrade the "ticker not in current holdings" `RuntimeError` — preserve the loud failure path for non-held tickers absent from both `candidate_prices_chf` and `cash_ticker`.

- [ ] **Step 6: Rewrite the Phase 2 BUY-cap test**

The new BUY pool funding contract obsoletes
`test_rebalancer_buy_capital_capped_at_new_capital` at
[tests/modelling/rebalancing/test_rebalancer.py:105-119](tests/modelling/rebalancing/test_rebalancer.py:105).
Rename and rewrite to assert the new funding invariant. Replace those lines with:

```python
def test_rebalancer_buy_pool_includes_reduce_proceeds_minus_cash():
    """BUY total <= new_capital + total_reduce - target_cash."""
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
    total_reduce = sum(
        a.est_cost_chf for a in plan.actions
        if a.action == 'REDUCE')
    total_cash = sum(
        a.est_cost_chf for a in plan.actions
        if a.action == 'CASH')  # 0.0 for equal_weight
    funding_cap = (
        constraints.new_capital_chf
        + total_reduce - total_cash)
    assert total_buy <= funding_cap + 1e-6
```

- [ ] **Step 7: Verify all the new tests pass**

Run: `pytest tests/modelling/rebalancing/test_action_and_plan.py tests/modelling/rebalancing/test_rebalancer_with_candidates.py -v`
Expected: every new test PASS.

- [ ] **Step 8: Verify the existing rebalancer tests still pass**

Run: `pytest tests/modelling/rebalancing/test_rebalancer.py -v`
Expected: every prior test still PASS. The rewritten BUY-cap test now reflects the recycle invariant. If any other Phase 2 test breaks unexpectedly, do NOT mute it — re-read the test and decide whether the breakage is a legitimate semantic change (update the assertion in this same task) or a regression (fix the rebalancer).

- [ ] **Step 9: Full suite**

Run: `pytest tests/ -v`
Expected: full green, prior count + 5 new action/plan tests + 7 new rebalancer-with-candidates tests, with the renamed BUY-cap test passing under the new contract.

- [ ] **Step 10: Commit (advisory)**

```bash
git add src/modelling/rebalancing/action.py \
        src/modelling/rebalancing/plan.py \
        src/modelling/rebalancing/rebalancer.py \
        tests/modelling/rebalancing/test_action_and_plan.py \
        tests/modelling/rebalancing/test_rebalancer.py \
        tests/modelling/rebalancing/test_rebalancer_with_candidates.py
git commit -m "feat(rebalancing): CASH action + recycle REDUCE proceeds"
```

---

## Task 4: `MaxGrowthStrategy`

**Files:**
- Create: `src/modelling/rebalancing/strategies/max_growth.py`
- Test: `tests/modelling/rebalancing/test_max_growth.py`

**Why:** Top-N candidates by `composite_score`, blended with eligible current holdings. Default 60% to eligible holdings (weighted by their current `weight_pct`, renormalised), 40% equal-weighted across the new top-N picks not already held, 0% cash (S1 from legacy `construct_scenarios.py`). The strategy is "high-risk growth": no per-name cap beyond the slot size.

**Constructor parameters:**
- `top_n: int = 5` — number of new picks.
- `holds_weight: float = 0.6` — fraction of NAV retained in eligible current holdings.
- `cash_weight: float = 0.0` — fraction of NAV reserved as CHF cash (S1 default 0%).

**Behaviour:**
- `candidates` is required. Raise `RuntimeError` when `None` or empty.
- Filter eligible holdings via `filter_eligible` (shared helper from `equal_weight.py`).
- New picks: top `top_n` candidates by `composite_score`, excluding tickers already in eligible holdings.
- Equity slice mass = `1 - cash_weight`. Output weights:
  - For each eligible holding: `(1 - cash_weight) * holds_weight * (h.weight_pct / sum(eligible.weight_pct))`.
  - For each new pick: `(1 - cash_weight) * (1 - holds_weight) / len(picks)`.
  - If `cash_weight > 0`: add `{cash_ticker: cash_weight}` (cash_ticker imported from rebalancer.py).
  - Renormalise the full dict to exactly 1.0 (defensive floating-point cleanup).
- Honours `constraints.swiss_tax_filter` (a no-op here because the screener should have applied `SwissTaxCriterion` upstream; but if any candidate's `metrics` contains a `swiss_tax` key == 0.0 (a fail signal), drop it).
- Validation: `cash_weight ∈ [0, 1]`, `holds_weight ∈ [0, 1]`, `top_n > 0`. Out-of-range raises `ValueError`.

- [ ] **Step 1: Write failing tests**

Create `tests/modelling/rebalancing/test_max_growth.py`:

```python
"""Tests for MaxGrowthStrategy."""

import pandas as pd
import pytest

from src.modelling.rebalancing.strategies import get_strategy
from src.screening.candidate import CandidateTicker
from src.shared.constraints import RebalanceConstraints


def _candidates(rows):
    return [
        CandidateTicker(
            ticker=t, exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=score, metrics={})
        for t, score in rows]


def test_max_growth_registered():
    """MaxGrowthStrategy registers as 'max_growth'."""
    s = get_strategy('max_growth')
    assert s.name == 'max_growth'


def test_max_growth_blends_holds_and_picks():
    """60% to eligible holds, 40% equal-weighted to top picks."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 60.0, 'category': 'core'},
        {'ticker': 'GOOGL', 'value_chf': 400.0,
         'weight_pct': 40.0, 'category': 'core'},
    ])
    candidates = _candidates([
        ('NVDA', 0.95), ('META', 0.90),
        ('AAPL', 0.80), ('TSLA', 0.70),
        ('ORCL', 0.60),
    ])
    s = get_strategy('max_growth')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    # 5 candidates + 2 holds = 7 targets.
    assert set(w) == {
        'AMZN', 'GOOGL', 'NVDA', 'META',
        'AAPL', 'TSLA', 'ORCL'}
    # 60% retained on holds, proportional to current weight
    assert w['AMZN'] == pytest.approx(0.6 * 0.6)
    assert w['GOOGL'] == pytest.approx(0.6 * 0.4)
    # 40% across 5 picks equally
    assert w['NVDA'] == pytest.approx(0.4 / 5)
    assert sum(w.values()) == pytest.approx(1.0)


def test_max_growth_drops_candidates_already_held():
    """A candidate that's already held is not double-counted."""
    holdings = pd.DataFrame([
        {'ticker': 'NVDA', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('NVDA', 0.99), ('META', 0.90), ('AAPL', 0.80)])
    s = get_strategy('max_growth')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    # NVDA picked up once, as a hold (60% slice)
    assert w['NVDA'] == pytest.approx(0.6)
    assert 'META' in w
    assert 'AAPL' in w
    assert sum(w.values()) == pytest.approx(1.0)


def test_max_growth_top_n_truncates_picks():
    """Only top-N candidates by composite_score are kept."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', 0.99), ('B', 0.95), ('C', 0.90),
        ('D', 0.85), ('E', 0.80), ('F', 0.75)])
    from src.modelling.rebalancing.strategies.max_growth import (
        MaxGrowthStrategy)
    s = MaxGrowthStrategy(top_n=3)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    picks = set(w) - {'AMZN'}
    assert picks == {'A', 'B', 'C'}


def test_max_growth_raises_when_candidates_missing():
    """Loud failure if the screener feed is empty/None."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    s = get_strategy('max_growth')
    with pytest.raises(RuntimeError, match='candidates'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(
                excluded_categories=(), category_caps={}),
            candidates=None)


def test_max_growth_raises_when_no_eligible_holds_or_picks():
    """No holds and no picks raises (degenerate plan)."""
    holdings = pd.DataFrame([
        {'ticker': 'JUNK', 'value_chf': 100.0,
         'weight_pct': 100.0, 'category': 'speculative'}])
    s = get_strategy('max_growth')
    with pytest.raises(RuntimeError):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(),
            candidates=[])


def test_max_growth_with_cash_weight_emits_cash_entry():
    """cash_weight > 0 adds 'CASH' to the output weights."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 60.0, 'category': 'core'},
        {'ticker': 'GOOGL', 'value_chf': 400.0,
         'weight_pct': 40.0, 'category': 'core'},
    ])
    candidates = _candidates([
        ('NVDA', 0.95), ('META', 0.90),
        ('AAPL', 0.80), ('TSLA', 0.70),
        ('ORCL', 0.60),
    ])
    from src.modelling.rebalancing.strategies.max_growth import (
        MaxGrowthStrategy)
    s = MaxGrowthStrategy(top_n=5, cash_weight=0.20)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert 'CASH' in w
    assert w['CASH'] == pytest.approx(0.20)
    # Equity slice = 80%; holds = 60% of equity = 0.48; picks = 0.32
    assert w['AMZN'] == pytest.approx(0.80 * 0.60 * 0.60)
    assert w['NVDA'] == pytest.approx(0.80 * 0.40 / 5)
    assert sum(w.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/modelling/rebalancing/test_max_growth.py -v`
Expected: ImportError or KeyError ('max_growth' not registered).

- [ ] **Step 3: Implement**

Create `src/modelling/rebalancing/strategies/max_growth.py`:

```python
"""MaxGrowthStrategy: top-composite picks blended with holds.

High-risk growth: keeps a configurable fraction of NAV in
eligible current holdings (proportional to their current
weights), and allocates the rest equal-weighted across the
top-N CandidateTickers by composite_score (excluding tickers
already held).

Honours the no-bond Swiss-tax policy implicitly: the screener
upstream is expected to have applied SwissTaxCriterion. As a
defensive secondary filter, candidates whose metrics include a
'swiss_tax' key with value 0.0 are dropped.
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
default_cash_weight = 0.0


class MaxGrowthStrategy(Strategy):
    """Blend eligible holds + top-N composite candidates."""

    name = 'max_growth'

    def __init__(
        self,
        top_n: int = default_top_n,
        holds_weight: float = default_holds_weight,
        cash_weight: float = default_cash_weight):
        if not 0.0 <= holds_weight <= 1.0:
            raise ValueError(
                f'holds_weight must be in [0, 1], '
                f'got {holds_weight}.')
        if not 0.0 <= cash_weight <= 1.0:
            raise ValueError(
                f'cash_weight must be in [0, 1], '
                f'got {cash_weight}.')
        if top_n <= 0:
            raise ValueError(
                f'top_n must be positive, got {top_n}.')
        self.top_n = top_n
        self.holds_weight = holds_weight
        self.cash_weight = cash_weight

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[
            str, float]:
        """Return blended target weights.

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
                remain (degenerate plan).
        """
        if candidates is None:
            raise RuntimeError(
                'MaxGrowthStrategy: candidates is required '
                '(screener-fed strategy).')
        eligible = filter_eligible(holdings_df, constraints)
        held = set(eligible['ticker'])
        filtered = [
            c for c in candidates
            if c.metrics.get('swiss_tax', 1.0) != 0.0]
        picks = sorted(
            (c for c in filtered if c.ticker not in held),
            key=lambda c: -c.composite_score)[:self.top_n]

        if eligible.empty and not picks:
            raise RuntimeError(
                'MaxGrowthStrategy: no eligible holds and no '
                'candidate picks; nothing to allocate.')
        equity_mass = 1.0 - self.cash_weight
        # Effective splits collapse when one side is missing
        h_w = self.holds_weight if not eligible.empty else 0.0
        p_w = (1.0 - h_w) if picks else 0.0
        if h_w == 0.0:
            p_w = 1.0
        if p_w == 0.0:
            h_w = 1.0

        weights: Dict[str, float] = {}
        if not eligible.empty:
            total_wpct = float(eligible['weight_pct'].sum())
            for _, row in eligible.iterrows():
                w = equity_mass * h_w * (
                    row['weight_pct'] / total_wpct)
                weights[row['ticker']] = w
        if picks:
            per_pick = equity_mass * p_w / len(picks)
            for c in picks:
                weights[c.ticker] = per_pick
        if self.cash_weight > 0.0:
            weights[cash_ticker] = self.cash_weight

        total = sum(weights.values())
        return {t: w / total for t, w in weights.items()}
```

- [ ] **Step 4: Verify the tests pass**

Run: `pytest tests/modelling/rebalancing/test_max_growth.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/modelling/rebalancing/strategies/max_growth.py \
        tests/modelling/rebalancing/test_max_growth.py
git commit -m "feat(rebalancing): MaxGrowthStrategy"
```

---

## Task 5: `RiskAdjustedStrategy`

**Files:**
- Create: `src/modelling/rebalancing/strategies/risk_adjusted.py`
- Test: `tests/modelling/rebalancing/test_risk_adjusted.py`

**Why:** Top-N candidates by `metrics['sharpe']`, weighted inversely by `abs(metrics['max_drawdown'])` (drawdown as a vol proxy — the existing criteria set has no plain volatility metric), with a per-name cap, blended with eligible current holdings, plus a 10% cash reserve (S2 default from legacy `construct_scenarios.py`).

**Constructor parameters:**
- `top_n: int = 5`
- `holds_weight: float = 0.6`
- `max_per_name: float = 0.15` — cap on any single candidate's weight after renormalisation. If breached, the over-cap excess is redistributed pro-rata to the other candidates until all are at or below cap. Holds bypass the cap (their split is governed by `holds_weight`).
- `cash_weight: float = 0.10` — CHF cash reserve fraction (S2 default 10%).

**Behaviour:**
- Requires `metrics['sharpe']` and `metrics['max_drawdown']` on every supplied candidate; raises `KeyError` listing the offending ticker on the first missing key.
- Top-N candidates ordered by Sharpe descending.
- Equity slice mass = `1 - cash_weight`. Inside the equity slice, holds get `holds_weight` (proportional to current weight_pct), picks get `(1 - holds_weight)`.
- Inverse-drawdown weights: `1 / max(abs(md), eps)`, normalised so the picks slice sums to `equity_mass * (1 - holds_weight)`.
- Cap loop: `max_per_name` is the cap on each pick's **total** weight (post-normalisation, including cash bucket dilution). While any pick > `max_per_name`, clip to the cap and redistribute the excess equally among the uncapped picks. Bail with `RuntimeError` if convergence fails after `cap_iter_max` rounds.
- If `cash_weight > 0`: add `{cash_ticker: cash_weight}` to the output.
- Same Swiss-tax defensive filter as MaxGrowthStrategy.

- [ ] **Step 1: Write failing tests**

Create `tests/modelling/rebalancing/test_risk_adjusted.py`:

```python
"""Tests for RiskAdjustedStrategy."""

import pandas as pd
import pytest

from src.modelling.rebalancing.strategies import get_strategy
from src.screening.candidate import CandidateTicker
from src.shared.constraints import RebalanceConstraints


def _candidates(rows):
    return [
        CandidateTicker(
            ticker=t, exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=0.7,
            metrics={'sharpe': sharpe,
                     'max_drawdown': md})
        for t, sharpe, md in rows]


def test_risk_adjusted_registered():
    s = get_strategy('risk_adjusted')
    assert s.name == 'risk_adjusted'


def test_risk_adjusted_higher_sharpe_picks_chosen():
    """Top-N candidates by Sharpe are selected."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', 2.0, -0.10), ('B', 1.8, -0.12),
        ('C', 1.5, -0.15), ('D', 1.0, -0.20),
        ('E', 0.5, -0.25), ('F', 0.1, -0.30)])
    from src.modelling.rebalancing.strategies.risk_adjusted \
        import RiskAdjustedStrategy
    s = RiskAdjustedStrategy(
        top_n=3, holds_weight=0.6, max_per_name=0.5,
        cash_weight=0.0)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    picks = set(w) - {'AMZN'}
    assert picks == {'A', 'B', 'C'}


def test_risk_adjusted_lower_drawdown_weighted_higher():
    """Lower max_drawdown magnitude -> larger weight."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('LOW_DD', 1.5, -0.05),
        ('HIGH_DD', 1.4, -0.50)])
    from src.modelling.rebalancing.strategies.risk_adjusted \
        import RiskAdjustedStrategy
    s = RiskAdjustedStrategy(
        top_n=2, holds_weight=0.5, max_per_name=0.5,
        cash_weight=0.0)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert w['LOW_DD'] > w['HIGH_DD']
    assert sum(w.values()) == pytest.approx(1.0)


def test_risk_adjusted_per_name_cap_enforced():
    """No candidate exceeds max_per_name."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    # One candidate has tiny drawdown so it dominates pre-cap
    candidates = _candidates([
        ('DOMINANT', 2.0, -0.001),
        ('B', 1.5, -0.20),
        ('C', 1.4, -0.20)])
    s = get_strategy('risk_adjusted')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    candidate_weights = {
        t: v for t, v in w.items() if t != 'AMZN'}
    assert max(candidate_weights.values()) <= 0.15 + 1e-9
    assert sum(w.values()) == pytest.approx(1.0)


def test_risk_adjusted_raises_on_missing_metric():
    """Missing 'sharpe' or 'max_drawdown' raises."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = [
        CandidateTicker(
            ticker='X', exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=0.7,
            metrics={'cagr': 0.1}),  # no sharpe / max_dd
    ]
    s = get_strategy('risk_adjusted')
    with pytest.raises(KeyError, match='sharpe'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(
                excluded_categories=(), category_caps={}),
            candidates=candidates)


def test_risk_adjusted_raises_on_no_candidates():
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    s = get_strategy('risk_adjusted')
    with pytest.raises(RuntimeError, match='candidates'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(),
            candidates=None)


def test_risk_adjusted_default_cash_weight_is_10pct():
    """Default cash_weight=0.10 emits a CASH entry."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 60.0, 'category': 'core'},
        {'ticker': 'GOOGL', 'value_chf': 400.0,
         'weight_pct': 40.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', 2.0, -0.10), ('B', 1.8, -0.12),
        ('C', 1.5, -0.15), ('D', 1.0, -0.20),
        ('E', 0.5, -0.25)])
    s = get_strategy('risk_adjusted')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert 'CASH' in w
    assert w['CASH'] == pytest.approx(0.10)
    assert sum(w.values()) == pytest.approx(1.0)


def test_risk_adjusted_cash_weight_zero_omits_cash_entry():
    """cash_weight=0 -> no CASH entry."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', 2.0, -0.10), ('B', 1.8, -0.12)])
    from src.modelling.rebalancing.strategies.risk_adjusted \
        import RiskAdjustedStrategy
    s = RiskAdjustedStrategy(cash_weight=0.0)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert 'CASH' not in w
    assert sum(w.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/modelling/rebalancing/test_risk_adjusted.py -v`
Expected: ImportError / 'risk_adjusted' not registered.

- [ ] **Step 3: Implement**

Create `src/modelling/rebalancing/strategies/risk_adjusted.py`:

```python
"""RiskAdjustedStrategy: Sharpe-ranked picks, inverse-drawdown.

Ports the spirit of the legacy S2 scenario from the deleted
construct_scenarios.py: top-Sharpe candidates, inverse-vol
weighted with a per-name cap, blended with eligible current
holdings, plus a 10% cash reserve by default. Uses
1 / abs(max_drawdown) as the vol proxy since the Phase 3
criteria set has no plain volatility metric.
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
        h_w = self.holds_weight if not eligible.empty else 0.0
        p_w = (1.0 - h_w) if picks else 0.0
        if h_w == 0.0:
            p_w = 1.0
        if p_w == 0.0:
            h_w = 1.0

        weights: Dict[str, float] = {}
        if not eligible.empty:
            total_wpct = float(eligible['weight_pct'].sum())
            for _, row in eligible.iterrows():
                weights[row['ticker']] = equity_mass * h_w * (
                    row['weight_pct'] / total_wpct)
        if picks:
            inv = {
                c.ticker: 1.0 / max(
                    abs(c.metrics['max_drawdown']),
                    drawdown_floor)
                for c in picks}
            inv_sum = sum(inv.values())
            for c in picks:
                weights[c.ticker] = equity_mass * p_w * (
                    inv[c.ticker] / inv_sum)
            self._enforce_cap(
                weights, {c.ticker for c in picks})
        if self.cash_weight > 0.0:
            weights[cash_ticker] = self.cash_weight

        total = sum(weights.values())
        return {t: w / total for t, w in weights.items()}

    def _enforce_cap(
        self,
        weights: Dict[str, float],
        cap_tickers: set) -> None:
        """Cap candidate weights at max_per_name, redistribute."""
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
            if not uncapped:
                raise RuntimeError(
                    'RiskAdjustedStrategy: cap redistribution '
                    'has nowhere to put excess weight '
                    f'{excess:.4f}; all picks already at cap.')
            share = excess / len(uncapped)
            for t in uncapped:
                weights[t] += share
        raise RuntimeError(
            'RiskAdjustedStrategy: cap redistribution did '
            f'not converge in {cap_iter_max} iterations.')
```

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/modelling/rebalancing/test_risk_adjusted.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/modelling/rebalancing/strategies/risk_adjusted.py \
        tests/modelling/rebalancing/test_risk_adjusted.py
git commit -m "feat(rebalancing): RiskAdjustedStrategy"
```

---

## Task 6: `MinRiskStrategy`

**Files:**
- Create: `src/modelling/rebalancing/strategies/min_risk.py`
- Test: `tests/modelling/rebalancing/test_min_risk.py`

**Why:** Lowest-drawdown candidates plus a broad accumulating ETF (`CSSPX.SW` by default, a global equity index — NOT a bond fund) blended with eligible current holdings, anchored by a 40% CHF cash reserve. The "min risk" character comes from a defensive cash buffer plus low-volatility equity (per `feedback_swiss_tax_no_bonds.md`: cash, never bonds). Defaults port S3 from the deleted `construct_scenarios.py`.

**Constructor parameters:**
- `broad_etf: str = 'CSSPX.SW'`
- `etf_weight: float = 0.12`
- `n_lowvol: int = 3`
- `lowvol_weight: float = 0.12`
- `holds_weight: float = 0.36`
- `cash_weight: float = 0.40`

(Defaults sum to 1.0: 0.12 + 0.12 + 0.36 + 0.40.)

**Behaviour:**
- Requires every candidate to have `metrics['max_drawdown']`.
- Low-vol picks = `n_lowvol` candidates with the **smallest** `abs(max_drawdown)`, excluding already-held tickers and excluding the broad ETF itself.
- Weights:
  - Eligible holds split `holds_weight` proportional to current weight_pct.
  - Broad ETF receives `etf_weight` as a singleton entry. If the broad ETF is already in `holdings_df` as an eligible holding, fold its allocation into the eligible-holdings slice (do not duplicate it).
  - Low-vol picks split `lowvol_weight` inversely by `abs(max_drawdown)` (lower drawdown → larger weight).
  - If `cash_weight > 0`: add `{cash_ticker: cash_weight}`.
- Validation: all weights ∈ [0, 1]; the four weights must sum to 1.0 ± 1e-6.
- Renormalise to sum to 1.0 (defensive floating-point cleanup).

- [ ] **Step 1: Write failing tests**

Create `tests/modelling/rebalancing/test_min_risk.py`:

```python
"""Tests for MinRiskStrategy."""

import pandas as pd
import pytest

from src.modelling.rebalancing.strategies import get_strategy
from src.screening.candidate import CandidateTicker
from src.shared.constraints import RebalanceConstraints


def _candidates(rows):
    return [
        CandidateTicker(
            ticker=t, exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=0.5,
            metrics={'max_drawdown': md})
        for t, md in rows]


def test_min_risk_registered():
    s = get_strategy('min_risk')
    assert s.name == 'min_risk'


def test_min_risk_default_allocation():
    """36% holds + 12% ETF + 12% low-vol + 40% cash."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 600.0,
         'weight_pct': 60.0, 'category': 'core'},
        {'ticker': 'GOOGL', 'value_chf': 400.0,
         'weight_pct': 40.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', -0.05), ('B', -0.07),
        ('C', -0.09), ('D', -0.20)])
    s = get_strategy('min_risk')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert 'CSSPX.SW' in w
    assert w['CSSPX.SW'] == pytest.approx(0.12)
    assert w['AMZN'] == pytest.approx(0.36 * 0.6)
    assert w['GOOGL'] == pytest.approx(0.36 * 0.4)
    assert 'CASH' in w
    assert w['CASH'] == pytest.approx(0.40)
    picks = set(w) - {'AMZN', 'GOOGL', 'CSSPX.SW', 'CASH'}
    assert picks == {'A', 'B', 'C'}
    assert sum(w.values()) == pytest.approx(1.0)


def test_min_risk_inverse_drawdown_weighting():
    """Lower abs(max_drawdown) -> larger weight in lowvol bucket."""
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('STABLE', -0.02), ('OK', -0.10), ('NOISY', -0.30)])
    s = get_strategy('min_risk')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert w['STABLE'] > w['OK'] > w['NOISY']


def test_min_risk_does_not_duplicate_held_etf():
    """If broad ETF is already held, no separate ETF entry."""
    holdings = pd.DataFrame([
        {'ticker': 'CSSPX.SW', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', -0.05), ('B', -0.07), ('C', -0.09)])
    s = get_strategy('min_risk')
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    # CSSPX.SW counted once. Sum still 1.0.
    cs = [t for t in w if t == 'CSSPX.SW']
    assert len(cs) == 1
    assert sum(w.values()) == pytest.approx(1.0)


def test_min_risk_raises_when_candidate_missing_max_drawdown():
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    bad = [
        CandidateTicker(
            ticker='X', exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1e11,
            composite_score=0.5,
            metrics={'cagr': 0.1})]
    s = get_strategy('min_risk')
    with pytest.raises(KeyError, match='max_drawdown'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(),
            candidates=bad)


def test_min_risk_raises_when_no_candidates_and_no_etf_or_holds():
    """Degenerate plan raises."""
    holdings = pd.DataFrame([
        {'ticker': 'JUNK', 'value_chf': 100.0,
         'weight_pct': 100.0, 'category': 'speculative'}])
    s = get_strategy('min_risk')
    with pytest.raises(RuntimeError, match='candidates'):
        s.propose(
            holdings, pd.DataFrame(),
            RebalanceConstraints(),
            candidates=None)


def test_min_risk_constructor_rejects_weights_not_summing_to_one():
    """etf + lowvol + holds + cash must sum to 1.0."""
    from src.modelling.rebalancing.strategies.min_risk import (
        MinRiskStrategy)
    with pytest.raises(ValueError, match='sum to 1.0'):
        MinRiskStrategy(
            etf_weight=0.20, lowvol_weight=0.20,
            holds_weight=0.20, cash_weight=0.20)


def test_min_risk_zero_cash_weight_omits_cash_entry():
    """cash_weight=0 -> no CASH entry; other weights re-sum to 1."""
    from src.modelling.rebalancing.strategies.min_risk import (
        MinRiskStrategy)
    holdings = pd.DataFrame([
        {'ticker': 'AMZN', 'value_chf': 1000.0,
         'weight_pct': 100.0, 'category': 'core'}])
    candidates = _candidates([
        ('A', -0.05), ('B', -0.07), ('C', -0.09)])
    s = MinRiskStrategy(
        etf_weight=0.20, lowvol_weight=0.20,
        holds_weight=0.60, cash_weight=0.0)
    w = s.propose(
        holdings, pd.DataFrame(),
        RebalanceConstraints(
            excluded_categories=(), category_caps={}),
        candidates=candidates)
    assert 'CASH' not in w
    assert sum(w.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/modelling/rebalancing/test_min_risk.py -v`
Expected: ImportError / 'min_risk' not registered.

- [ ] **Step 3: Implement**

Create `src/modelling/rebalancing/strategies/min_risk.py`:

```python
"""MinRiskStrategy: low-vol picks + broad equity ETF + cash.

Anchored by a 40% CHF cash reserve (per
feedback_swiss_tax_no_bonds.md: defensive risk lever is cash,
never bonds), with the rest split across a global equity index
(default CSSPX.SW), the N lowest-drawdown screener candidates,
and eligible current holdings. Defaults port the legacy S3
scenario from the deleted construct_scenarios.py.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.modelling.rebalancing.action import cash_ticker
from src.modelling.rebalancing.strategies.base import Strategy
from src.modelling.rebalancing.strategies.equal_weight import (
    filter_eligible)
from src.shared.constraints import RebalanceConstraints


default_broad_etf = 'CSSPX.SW'
default_etf_weight = 0.12
default_n_lowvol = 3
default_lowvol_weight = 0.12
default_holds_weight = 0.36
default_cash_weight = 0.40
drawdown_floor = 1e-4


class MinRiskStrategy(Strategy):
    """Holds + broad equity ETF + low-DD picks + cash."""

    name = 'min_risk'

    def __init__(
        self,
        broad_etf: str = default_broad_etf,
        etf_weight: float = default_etf_weight,
        n_lowvol: int = default_n_lowvol,
        lowvol_weight: float = default_lowvol_weight,
        holds_weight: float = default_holds_weight,
        cash_weight: float = default_cash_weight):
        for label, val in (
            ('etf_weight', etf_weight),
            ('lowvol_weight', lowvol_weight),
            ('holds_weight', holds_weight),
            ('cash_weight', cash_weight)):
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f'{label} must be in [0, 1], got {val}.')
        total = (
            etf_weight + lowvol_weight + holds_weight
            + cash_weight)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f'MinRiskStrategy: etf_weight + '
                f'lowvol_weight + holds_weight + cash_weight '
                f'must sum to 1.0, got {total}.')
        if n_lowvol <= 0:
            raise ValueError(
                f'n_lowvol must be positive, got {n_lowvol}.')
        self.broad_etf = broad_etf
        self.etf_weight = etf_weight
        self.n_lowvol = n_lowvol
        self.lowvol_weight = lowvol_weight
        self.holds_weight = holds_weight
        self.cash_weight = cash_weight

    def propose(
        self,
        holdings_df: pd.DataFrame,
        prices: pd.DataFrame,
        constraints: RebalanceConstraints,
        candidates: Optional[List] = None,
        risk_free_rate: Optional[float] = None) -> Dict[
            str, float]:
        """Return defensively-allocated target weights.

        Returns:
            Dict {ticker: fraction} summing to 1.0. Includes a
            'CASH' entry equal to ``cash_weight`` when
            ``cash_weight > 0``.

        Raises:
            RuntimeError: When candidates is None.
            KeyError: When any candidate lacks
                'max_drawdown' in its metrics.
        """
        if candidates is None:
            raise RuntimeError(
                'MinRiskStrategy: candidates is required '
                '(screener-fed strategy).')

        eligible = filter_eligible(holdings_df, constraints)
        held = set(eligible['ticker'])

        for c in candidates:
            if 'max_drawdown' not in c.metrics:
                raise KeyError(
                    f'MinRiskStrategy: candidate '
                    f'{c.ticker!r} missing metric '
                    f"'max_drawdown'.")

        filtered = [
            c for c in candidates
            if c.metrics.get('swiss_tax', 1.0) != 0.0
            and c.ticker not in held
            and c.ticker != self.broad_etf]
        picks = sorted(
            filtered,
            key=lambda c: abs(c.metrics['max_drawdown']))[
                :self.n_lowvol]

        weights: Dict[str, float] = {}
        if not eligible.empty:
            total_wpct = float(eligible['weight_pct'].sum())
            for _, row in eligible.iterrows():
                weights[row['ticker']] = self.holds_weight * (
                    row['weight_pct'] / total_wpct)
        # Broad ETF: only add as standalone if not already held
        if self.broad_etf not in held:
            weights[self.broad_etf] = self.etf_weight
        else:
            # Fold ETF allocation into the holds bucket
            if self.broad_etf in weights:
                weights[self.broad_etf] += self.etf_weight
        if picks:
            inv = {
                c.ticker: 1.0 / max(
                    abs(c.metrics['max_drawdown']),
                    drawdown_floor)
                for c in picks}
            inv_sum = sum(inv.values())
            for c in picks:
                weights[c.ticker] = self.lowvol_weight * (
                    inv[c.ticker] / inv_sum)
        if self.cash_weight > 0.0:
            weights[cash_ticker] = self.cash_weight

        total = sum(weights.values())
        if total <= 0:
            raise RuntimeError(
                'MinRiskStrategy: empty allocation '
                '(no holds, no picks, no ETF entry).')
        return {t: w / total for t, w in weights.items()}
```

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/modelling/rebalancing/test_min_risk.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/modelling/rebalancing/strategies/min_risk.py \
        tests/modelling/rebalancing/test_min_risk.py
git commit -m "feat(rebalancing): MinRiskStrategy"
```

---

## Task 7: Register the three new strategies

**Files:**
- Modify: `src/modelling/rebalancing/strategies/__init__.py`
- Modify: `tests/modelling/rebalancing/test_strategies.py` — update the registry-size assertion.

**Why:** The strategies self-register via `Strategy.__init_subclass__`, but the package `__init__.py` must import the new modules so the classes are actually loaded. Update the registry-size test (`test_strategy_registry_has_four_initial_strategies`) to expect seven.

- [ ] **Step 1: Update the package `__init__`**

Modify `src/modelling/rebalancing/strategies/__init__.py`:

```python
"""
Strategy registry: name -> Strategy subclass.

Concrete strategies self-register via Strategy.__init_subclass__.
Importing this module triggers import of every concrete
strategy module so the registry is fully populated.
"""

from src.modelling.rebalancing.strategies.base import (
    Strategy, _registry as strategy_registry)
from src.modelling.rebalancing.strategies import equal_weight  # noqa: F401
from src.modelling.rebalancing.strategies import inverse_vol  # noqa: F401
from src.modelling.rebalancing.strategies import max_growth  # noqa: F401
from src.modelling.rebalancing.strategies import min_risk  # noqa: F401
from src.modelling.rebalancing.strategies import min_variance  # noqa: F401
from src.modelling.rebalancing.strategies import mvo  # noqa: F401
from src.modelling.rebalancing.strategies import risk_adjusted  # noqa: F401


def get_strategy(name: str) -> Strategy:
    """Instantiate a registered strategy by name.

    Args:
        name: Registry key.

    Returns:
        Fresh instance of the matching Strategy subclass.

    Raises:
        KeyError: When `name` is not registered.
    """
    if name not in strategy_registry:
        raise KeyError(
            f'Unknown strategy {name!r}. Known: '
            f'{sorted(strategy_registry)}')
    return strategy_registry[name]()
```

- [ ] **Step 2: Update the registry-size test**

In `tests/modelling/rebalancing/test_strategies.py` find:

```python
def test_strategy_registry_has_four_initial_strategies():
    """Phase 2 ships exactly four portfolio-only strategies."""
    expected = {
        'mvo', 'equal_weight', 'inverse_vol', 'min_variance'}
    assert set(strategy_registry) == expected
```

Rename + rewrite to:

```python
def test_strategy_registry_after_phase_4():
    """Phase 4 expands the registry to seven strategies."""
    expected = {
        'mvo', 'equal_weight', 'inverse_vol', 'min_variance',
        'max_growth', 'risk_adjusted', 'min_risk'}
    assert set(strategy_registry) == expected
```

- [ ] **Step 3: Verify the registry tests pass**

Run: `pytest tests/modelling/rebalancing/test_strategies.py -v`
Expected: all PASS, registry-size test reflects seven.

- [ ] **Step 4: Full-suite green-check**

Run: `pytest tests/ -v`
Expected: previous count + new tests from tasks 4/5/6 all PASS.

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/modelling/rebalancing/strategies/__init__.py \
        tests/modelling/rebalancing/test_strategies.py
git commit -m "feat(rebalancing): register screener-fed strategies"
```

---

## Task 8: Wire `candidates_from` into `report.py`

**Files:**
- Modify: `src/analysis/report.py`
- Test: `tests/analysis/test_report_with_candidates.py`

**Why:** The user-edited orchestrator must accept a JSON file path for screener-fed runs. Per HANDOFF.md §2, `main()` stays zero-arg with local-variable inputs — **do not** reintroduce argparse.

- [ ] **Step 1: Write the failing integration test**

Create `tests/analysis/test_report_with_candidates.py`:

```python
"""Smoke test: report.main runs with a screener-fed strategy."""

import json
import pathlib

import pandas as pd
import pytest

from src.analysis import report


def test_report_main_with_max_growth_candidates(
        tmp_path, monkeypatch):
    """report.main runs end-to-end with max_growth + JSON."""
    # Stub the analyzer with fake holdings + prices.
    holdings_df = pd.DataFrame([
        {'ticker': 'AMZN', 'shares': 5,
         'value_chf': 1000.0, 'weight_pct': 50.0,
         'currency': 'USD', 'sector': 'Tech',
         'price_chf': 200.0, 'category': 'core'},
        {'ticker': 'GOOGL', 'shares': 1,
         'value_chf': 1000.0, 'weight_pct': 50.0,
         'currency': 'USD', 'sector': 'Tech',
         'price_chf': 1000.0, 'category': 'core'},
    ])
    prices = pd.DataFrame({
        'AMZN': [200.0, 201.0, 199.0],
        'GOOGL': [1000.0, 1010.0, 990.0]})

    class FakeAnalyzer:
        def __init__(self, **_):
            self.data_provider = FakeProvider()
        def get_holdings_snapshot(self, categories=None):
            return holdings_df.copy()
        def get_price_panel(self):
            return prices.copy()

    class FakeProvider:
        def get_current_price(self, ticker):
            table = {
                'NVDA': 1000.0, 'META': 500.0, 'AAPL': 200.0,
                'TSLA': 250.0, 'ORCL': 100.0,
                'USDCHF=X': 0.9}
            if ticker not in table:
                raise ValueError(ticker)
            return table[ticker]

    monkeypatch.setattr(
        'src.analysis.report.PortfolioAnalyzer', FakeAnalyzer)
    monkeypatch.setattr(
        'src.analysis.report._build_data_provider',
        lambda: FakeProvider())
    # Categories file stub
    cat_path = tmp_path / 'cats.json'
    cat_path.write_text(json.dumps(
        {'AMZN': 'core', 'GOOGL': 'core', 'NVDA': 'core',
         'META': 'core', 'AAPL': 'core', 'TSLA': 'core',
         'ORCL': 'core'}))
    # Candidates JSON
    cand_path = tmp_path / 'candidates.json'
    rows = [
        {'ticker': t, 'exchange': 'NASDAQ',
         'currency': 'USD', 'sector': 'Tech',
         'market_cap': 1e12, 'composite_score': s,
         'metrics': {'sharpe': 1.5,
                     'max_drawdown': -0.10}}
        for t, s in [
            ('NVDA', 0.95), ('META', 0.90),
            ('AAPL', 0.85), ('TSLA', 0.80),
            ('ORCL', 0.75)]]
    cand_path.write_text(json.dumps(rows))

    # Patch the local-var inputs by monkeypatching `main`'s
    # constants via a tiny wrapper.
    output_path = tmp_path / 'report.xlsx'
    monkeypatch.setattr(report, 'main', report.main)

    # Run via a vars-injection helper (see implementation).
    report.main_with_inputs(
        strategy='max_growth',
        new_capital_chf=2000.0,
        max_weight=0.15,
        degiro_csv=None,
        ibkr_csv='IGNORED',
        ticker_categories_path=str(cat_path),
        candidates_from=str(cand_path),
        output_path=str(output_path))

    assert output_path.exists()
    # Confirm Excel has the expected sheets
    xl = pd.ExcelFile(output_path)
    assert {'Holdings', 'Target Weights',
            'Rebalancing Actions', 'Summary'}.issubset(
                set(xl.sheet_names))
```

- [ ] **Step 2: Verify it fails**

Run: `pytest tests/analysis/test_report_with_candidates.py -v`
Expected: AttributeError on `main_with_inputs` (not yet defined) OR ImportError chain.

- [ ] **Step 3: Refactor `report.main` to expose `main_with_inputs`**

Replace `src/analysis/report.py`:

```python
"""Portfolio rebalancing report - programmatic orchestrator.

Reads transaction CSVs, runs the chosen rebalance strategy
against current holdings, and writes a multi-sheet Excel
workbook with current holdings, target weights, and BUY /
REDUCE / HOLD actions.

Edit the values at the top of ``main()`` to change the run
inputs (strategy, capital, file paths, etc.), then either:
    python -m src.analysis.report
or call from a session / notebook:
    from src.analysis import report
    report.main()
"""

import json
import pathlib
from typing import Dict, Optional

import pandas as pd

from src.analysis.core.analyzer import PortfolioAnalyzer
from src.analysis.loaders.data_exporter import DataExporter
from src.modelling.rebalancing import (
    Rebalancer, get_strategy)
from src.screening.candidate import load_candidates_from_json
from src.shared.constraints import RebalanceConstraints
from src.shared.data_provider import (
    DataProvider, YFinanceProvider)


def _build_data_provider() -> DataProvider:
    """Factory hook returning the live data provider."""
    return YFinanceProvider()


def _load_categories(path: str) -> Dict[str, str]:
    """Load ticker -> category map from JSON."""
    with open(path) as f:
        return json.load(f)


def _summary_dataframe(
    plan_summary: Dict[str, float]) -> pd.DataFrame:
    """Single-row DataFrame for the Summary sheet."""
    return pd.DataFrame([plan_summary])


def main_with_inputs(
    strategy: str,
    new_capital_chf: float,
    max_weight: float,
    degiro_csv: Optional[str],
    ibkr_csv: Optional[str],
    ticker_categories_path: str,
    candidates_from: Optional[str],
    output_path: str) -> None:
    """Generate the rebalancing Excel report.

    Args:
        strategy: Registered strategy name (e.g. 'mvo',
            'max_growth').
        new_capital_chf: Fresh capital to deploy this run.
        max_weight: Per-name cap fed into RebalanceConstraints.
        degiro_csv: Path to Degiro transaction CSV, or None.
        ibkr_csv: Path to IBKR transaction CSV, or None.
        ticker_categories_path: Path to the ticker -> category
            JSON map.
        candidates_from: Path to a screener JSON output (for
            screener-fed strategies); ``None`` for
            portfolio-only strategies.
        output_path: Excel destination.

    Raises:
        ValueError: When neither degiro_csv nor ibkr_csv is set.
        KeyError: When ``strategy`` is not registered.
    """
    if not degiro_csv and not ibkr_csv:
        raise ValueError(
            'report.main_with_inputs: must supply at least '
            'one of degiro_csv or ibkr_csv.')

    categories = _load_categories(ticker_categories_path)
    constraints = RebalanceConstraints(
        max_weight_default=max_weight,
        new_capital_chf=new_capital_chf)
    provider = _build_data_provider()

    analyzer = PortfolioAnalyzer(
        degiro_csv_file_path=degiro_csv,
        ibkr_csv_file_path=ibkr_csv,
        data_provider=provider,
        base_currency='CHF')

    candidates = None
    if candidates_from is not None:
        candidates = load_candidates_from_json(candidates_from)

    rebalancer = Rebalancer(
        analyzer=analyzer,
        strategy=get_strategy(strategy),
        constraints=constraints,
        ticker_categories=categories)
    plan = rebalancer.propose(candidates=candidates)

    sheets = {
        'Holdings': plan.holdings_df,
        'Target Weights': plan.target_weights_dataframe(),
        'Rebalancing Actions': plan.to_dataframe(),
        'Summary': _summary_dataframe(plan.summary())}

    out_dir = pathlib.Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    DataExporter().export_to_excel(sheets, output_path)
    print(
        f'Report saved: {output_path} '
        f'(strategy={strategy}, '
        f'NAV CHF {plan.total_value_chf:,.2f}, '
        f'{len(plan.actions)} actions, '
        f'generated {plan.generated_at:%Y-%m-%d %H:%M})')


def main() -> None:
    """Zero-arg entry point. Edit the locals to change inputs."""
    strategy = 'mvo'
    new_capital_chf = 2000.0
    max_weight = 0.15
    degiro_csv = 'data/postprocess_data/processed_portfolio.csv'
    ibkr_csv = 'reports/ibkr/<YYYYMMDD>_<YYYYMMDD>_<ACCOUNT_ID>.csv'
    ticker_categories_path = 'data/ticker_categories.json'
    candidates_from = None  # set to a JSON path for screener-fed
    output_path = 'results/portfolio_report.xlsx'

    main_with_inputs(
        strategy=strategy,
        new_capital_chf=new_capital_chf,
        max_weight=max_weight,
        degiro_csv=degiro_csv,
        ibkr_csv=ibkr_csv,
        ticker_categories_path=ticker_categories_path,
        candidates_from=candidates_from,
        output_path=output_path)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Verify the integration test passes**

Run: `pytest tests/analysis/test_report_with_candidates.py -v`
Expected: PASS.

- [ ] **Step 5: Verify the original `main()` smoke path still works against existing fixtures**

Run: `pytest tests/ -v`
Expected: all prior tests + new ones PASS.

- [ ] **Step 6: Commit (advisory)**

```bash
git add src/analysis/report.py \
        tests/analysis/test_report_with_candidates.py
git commit -m "feat(report): candidates_from JSON for screener-fed strategies"
```

---

## Task 9: Delete `src/orchestration/`

**Files:**
- Delete: `src/orchestration/__init__.py`
- Delete: `src/orchestration/construct_scenarios.py`
- Delete: `src/orchestration/` (directory)

**Why:** Per spec §7 Phase 4: the orchestration domain collapses into `report.py`. The legacy `construct_scenarios.py` is fully replaced by the new strategies + `candidates_from` wiring. There is no public consumer of the package outside its `__main__` block.

- [ ] **Step 1: Confirm no live imports remain**

Run: `grep -rn 'from src.orchestration\|import src.orchestration' src/ tests/ docs/`
Expected: zero hits. If any exist, fix them first (they are bugs).

- [ ] **Step 2: Delete the directory**

```bash
rm -rf src/orchestration/
```

- [ ] **Step 3: Verify the suite still passes**

Run: `pytest tests/ -v`
Expected: full suite green.

- [ ] **Step 4: Commit (advisory)**

```bash
git add -A src/orchestration/
git commit -m "chore: remove src/orchestration; collapsed into report.py"
```

---

## Task 10: Full-pipeline manual verification (per spec §7 Phase 4)

This step is operator-driven and produces no code; it is the closeout verification before marking Phase 4 complete in `tasks/todo.md`. **Do not skip.**

- [ ] **Step 1: Confirm the test suite is fully green**

Run: `pytest tests/ -v`
Expected: every prior test (122 baseline) + every new test from tasks 1-9 PASS. No xfails, no skips beyond pre-existing harmless ones.

- [ ] **Step 2: Spot-check one screener-fed run end-to-end**

This requires a refreshed `data/screener_cache/<exchange>/` (Phase 3 deliverable; per HANDOFF.md §11 the cache is currently empty in CI, populated only by tests). For a real run the operator:

1. Refreshes one small exchange. Example:
   ```bash
   python -m src.user_scripts.refresh_screener_cache \
       --exchange SIX
   ```
2. Runs the screener:
   ```bash
   python -m src.user_scripts.run_screener \
       --exchange SIX \
       --criteria momentum,cagr,sharpe,max_drawdown,swiss_tax \
       --top-n 30 \
       --output-format json \
       --output results/screener_SIX.json
   ```
3. Edits the top of `src/analysis/report.py` `main()`:
   - `strategy = 'max_growth'`
   - `candidates_from = 'results/screener_SIX.json'`
4. Runs the report:
   ```bash
   python -m src.analysis.report
   ```
5. Opens `results/portfolio_report.xlsx`. Confirms:
   - `Rebalancing Actions` sheet contains BUY rows for new tickers from the screener output.
   - `Target Weights` sheet's `target_wt_pct` column sums to ~100.
   - `Summary` sheet shows non-zero `total_buy_chf` ≤ `new_capital_chf`.

Skip this step if the cache is empty in the operator's environment — the integration tests cover the contract in tests/.

- [ ] **Step 3: Append the Phase 4 block to `tasks/todo.md`**

Mirror the formatting of the Phase 1/2/3 blocks. Include: files added, files deleted, test count, verification notes, links to the spec + this plan, and the next-phase pointer (Phase 5 — Monte Carlo).

- [ ] **Step 4: Commit (advisory)**

```bash
git add tasks/todo.md
git commit -m "docs: Phase 4 completion log"
```

---

## Self-Review

**1. Spec coverage (HANDOFF.md §2 Phase 4 deliverables, plus the cash-bucket addition agreed inline):**

| Spec requirement | Task |
|---|---|
| `MaxGrowthStrategy` consumes `candidates`, picks top-N by composite_score, equal-weight | Task 4 |
| `RiskAdjustedStrategy` top-Sharpe picks, inverse-vol weighting, 15% per-name cap | Task 5 |
| `MinRiskStrategy` low-vol picks + broad ETF, no bonds | Task 6 |
| CHF cash bucket (S1/S2/S3 defaults 0% / 10% / 40%, per `feedback_swiss_tax_no_bonds.md`) | Tasks 3-6 |
| `RebalancingAction` extended with `'CASH'` action type; CASH action emits target cash CHF in `est_cost_chf` | Task 3 |
| `RebalancePlan.summary()` reports `total_cash_chf` | Task 3 |
| BUY pool recycles REDUCE proceeds and respects the cash reservation (Option A) | Task 3 |
| Phase 2 BUY-cap test rewritten to the new funding invariant | Task 3 step 6 |
| Each strategy auto-registers; `strategies/__init__.py` is the only edit needed | Task 7 |
| `candidates_from` local var in `main()`, no argparse | Task 8 |
| Delete `src/orchestration/` entirely | Task 9 |
| TDD per strategy | Tasks 4-6 |
| Integration test: `report.main()` with screener-fed strategy + JSON fixture | Task 8 |
| `Rebalancer` handles non-held target tickers | Task 3 |
| Loud failures, no silent defaults | Tasks 2-6 |

**2. Placeholder scan:** every code block contains complete, runnable code. No TBDs, no "implement later", no "similar to Task N" without the actual code, no references to undefined functions.

**3. Type consistency:**
- `CandidateTicker(ticker, exchange, currency, sector, market_cap, composite_score, metrics)` — used identically in Tasks 1, 3, 4, 5, 6, 8.
- `Strategy.propose(holdings_df, prices, constraints, candidates, risk_free_rate)` — matches base.py signature; new strategies all declare `candidates: Optional[List] = None` per ABC and raise when None.
- `Rebalancer.propose(candidates=None)` — preserved across Task 3.
- `RebalancingAction(ticker, action, shares, est_cost_chf, current_wt_pct, target_wt_pct, note)` — unchanged signature; only `valid_actions` is extended.
- `cash_ticker = 'CASH'` — defined in `action.py` (Task 3), imported by `rebalancer.py` (Task 3) and by all three new strategy modules (Tasks 4-6). No circular import: strategies depend on `action.py`, not on `rebalancer.py`.
- `fetch_chf_price(data_provider, ticker, currency)` — used in Task 2 (definition) and Task 3 (caller).
- `load_candidates_from_json(path)` — defined in Task 1, called in Task 8.
- `filter_eligible(holdings_df, constraints)` — existing helper, imported from `equal_weight.py` in Tasks 4, 5, 6.
- `cash_weight: float` constructor parameter — present on `MaxGrowthStrategy` (default 0.0), `RiskAdjustedStrategy` (default 0.10), `MinRiskStrategy` (default 0.40). Per-strategy defaults match S1/S2/S3 from the deleted `construct_scenarios.py`.

No type or name drift across tasks.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-14-phase-4-screener-fed-strategies.md`. Two execution options:

1. **Subagent-Driven** (recommended per HANDOFF.md §3) — dispatch a fresh implementer subagent per task, then a spec reviewer + a code-quality reviewer before marking complete. Continuous execution; no pause unless something blocks.
2. **Inline Execution** — execute tasks in this session via `superpowers:executing-plans`, batched with review checkpoints.

Per the handoff workflow (skill stack item 2), the default is Subagent-Driven via `superpowers:subagent-driven-development`.
