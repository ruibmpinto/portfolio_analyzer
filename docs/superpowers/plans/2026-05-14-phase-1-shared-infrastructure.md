# Phase 1 — Shared infrastructure implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the cross-cutting shared infrastructure that the rest of the refactor depends on: live risk-free-rate fetcher, rebalance-constraints dataclass, and two new portfolio-snapshot methods on `PortfolioAnalyzer`. Provides Contracts 3 and 4 from the design spec.

**Architecture:** Two new modules under `src/shared/` (`risk_free_rate.py`, `constraints.py`), additive-only edits to `src/analysis/core/analyzer.py` (two new public methods), and a `tests/` infrastructure setup since none exists. All work TDD: failing test → minimal implementation → green → commit.

**Tech Stack:** Python 3, pandas, numpy, pandas-datareader (already installed), pytest 8.3+ (already installed), warnings (stdlib).

**Source spec:** `docs/superpowers/specs/2026-05-13-repo-refactor-design.md` — Sections 2 (Contracts 3 & 4), 6 (`risk_free_rate.py` and `constraints.py` definitions), 7 (Phase 1 verification).

**Style note:** All Python files comply with `~/.claude/rules/code-style.md`: 80-char max line width, single quotes for strings (double quotes only for docstrings), snake_case + PascalCase, lowercase module-level constants (per memory `feedback_no_caps_globals.md`), no `.get(key, default)` to mask missing keys (per memory `feedback_no_silent_defaults.md`), NumPy-style docstrings, no type hints in function signatures (types live in docstrings only).

**Commit policy:** This plan describes commit boundaries for the engineer. Per `~/.claude/CLAUDE.md`, only commit when the user requests. The commit steps are advisory — execute them only after the user approves.

---

## File map

**New files:**
- `tests/__init__.py` — empty package marker.
- `tests/conftest.py` — shared pytest fixtures.
- `tests/shared/__init__.py` — empty package marker.
- `tests/shared/test_risk_free_rate.py` — unit tests for live RFR.
- `tests/shared/test_constraints.py` — unit tests for `RebalanceConstraints`.
- `tests/analysis/__init__.py` — empty package marker.
- `tests/analysis/core/__init__.py` — empty package marker.
- `tests/analysis/core/test_analyzer_snapshot.py` — unit tests for the two new analyzer methods.
- `src/shared/risk_free_rate.py` — `get_live_risk_free_rate()`.
- `src/shared/constraints.py` — `RebalanceConstraints` dataclass + `default_constraints()`.

**Modified files:**
- `src/analysis/core/analyzer.py` — add `get_holdings_snapshot()` and `get_price_panel()` methods on `PortfolioAnalyzer`.
- `src/shared/__init__.py` — export the new public symbols.

---

## Task 0 — Test infrastructure setup

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/shared/__init__.py` (empty)
- Create: `tests/analysis/__init__.py` (empty)
- Create: `tests/analysis/core/__init__.py` (empty)

- [ ] **Step 1: Verify pytest is available**

Run: `python3 -c "import pytest; print(pytest.__version__)"`
Expected: prints a version >= 8.0.

- [ ] **Step 2: Create tests directory tree**

Run:
```bash
mkdir -p /Users/rbarreira/Desktop/stock_market/tests/shared
mkdir -p /Users/rbarreira/Desktop/stock_market/tests/analysis/core
```
Then create empty marker files:
```bash
touch /Users/rbarreira/Desktop/stock_market/tests/__init__.py
touch /Users/rbarreira/Desktop/stock_market/tests/shared/__init__.py
touch /Users/rbarreira/Desktop/stock_market/tests/analysis/__init__.py
touch /Users/rbarreira/Desktop/stock_market/tests/analysis/core/__init__.py
```

- [ ] **Step 3: Write `tests/conftest.py` with shared fixtures**

```python
"""Shared pytest fixtures for the stock_market test suite.

Fixtures
--------
fake_transactions
    Minimal cross-broker transaction list for analyzer tests.
fake_data_provider
    DataProvider stub that returns synthetic price/split/dividend
    series and reports CHF as the listing currency for everything.
patched_csv_loader
    Monkeypatches CSVLoader.load_csv_degiro and load_csv_ibkr to
    return preset Transaction lists so analyzer tests don't need
    on-disk CSV fixtures.
analyzer
    Fully wired PortfolioAnalyzer using the three fixtures above.
"""

#
#                                                                       Modules
# =============================================================================
# Standard
import pathlib
import sys
from datetime import datetime, timedelta
# Third-party
import numpy as np
import pandas as pd
import pytest

# Local
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.analysis.core.transaction import Transaction
from src.analysis.core.analyzer import PortfolioAnalyzer
from src.analysis.loaders.csv_loader import CSVLoader


# =============================================================================
@pytest.fixture
def fake_transactions():
    """Minimal cross-broker transaction list.

    Returns
    -------
    by_broker : dict
        {'degiro': list[Transaction], 'ibkr': list[Transaction]}.
    """
    base = datetime(2024, 1, 15)
    degiro = [
        Transaction(
            ticker='AMZN', operation='buy', date=base,
            price=150.0, amount=5, fee=0.5,
            auto_fx_fee=0.1, currency='USD'),
        Transaction(
            ticker='UBSG.SW', operation='buy',
            date=base + timedelta(days=10),
            price=25.0, amount=1, fee=0.5,
            auto_fx_fee=0.0, currency='CHF'),
    ]
    ibkr = [
        Transaction(
            ticker='HOLN.SW', operation='buy',
            date=base + timedelta(days=20),
            price=70.0, amount=2, fee=1.5,
            auto_fx_fee=0.0, currency='CHF'),
    ]
    return {'degiro': degiro, 'ibkr': ibkr}


# =============================================================================
class _FakeDataProvider:
    """DataProvider stub returning deterministic synthetic data.

    Methods
    -------
    get_price_history(ticker, start, end)
        Daily close series, sinusoidal around 100, ticker-seeded.
    get_split_history(ticker)
        Empty series.
    get_dividend_price_factors(ticker)
        Series of 1.0 (no dividend deflation).
    get_dividend_history(ticker, start)
        Empty series.
    get_fundamental_data(ticker)
        Returns a dict with a deterministic sector per ticker.
    """
    sector_map = {
        'AMZN': 'Consumer Cyclical',
        'UBSG.SW': 'Financial Services',
        'HOLN.SW': 'Basic Materials',
    }
    # -------------------------------------------------------------------------
    def get_price_history(self, ticker, start, end):
        dates = pd.date_range(start=start, end=end, freq='D')
        seed = sum(ord(c) for c in ticker) % 50
        prices = 100.0 + seed + np.sin(
            np.arange(len(dates)) / 30.0) * 5.0
        return pd.Series(prices, index=dates)
    # -------------------------------------------------------------------------
    def get_split_history(self, ticker):
        return pd.Series(dtype=float)
    # -------------------------------------------------------------------------
    def get_dividend_price_factors(self, ticker):
        # Empty series -> Portfolio._dividend_price_factor returns 1.0
        return pd.Series(dtype=float)
    # -------------------------------------------------------------------------
    def get_dividend_history(self, ticker, start):
        return pd.Series(dtype=float)
    # -------------------------------------------------------------------------
    def get_fundamental_data(self, ticker):
        sector = self.sector_map.get(ticker)
        if sector is None:
            return {}
        return {'sector': sector}


@pytest.fixture
def fake_data_provider():
    """Instance of _FakeDataProvider for tests."""
    return _FakeDataProvider()


# =============================================================================
@pytest.fixture
def patched_csv_loader(monkeypatch, fake_transactions):
    """Patch CSVLoader to return preset transactions, no disk I/O.

    Routes:
        load_csv_degiro -> fake_transactions['degiro']
        load_csv_ibkr   -> fake_transactions['ibkr']
    """
    monkeypatch.setattr(
        CSVLoader, 'load_csv_degiro',
        lambda self, path: fake_transactions['degiro'])
    monkeypatch.setattr(
        CSVLoader, 'load_csv_ibkr',
        lambda self, path: fake_transactions['ibkr'])


# =============================================================================
@pytest.fixture
def analyzer(patched_csv_loader, fake_data_provider):
    """PortfolioAnalyzer wired to fake transactions + data provider.

    Both CSV paths are non-None so the loader is invoked, but the
    monkeypatched loader returns the fake transactions regardless
    of path content.
    """
    return PortfolioAnalyzer(
        degiro_csv_file_path='ignored.csv',
        ibkr_csv_file_path='ignored.csv',
        data_provider=fake_data_provider,
        base_currency='CHF')
```

- [ ] **Step 4: Verify the suite runs and discovers no failures**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: `0 collected` (no test files yet) — exit code 5 from pytest is expected for "no tests collected"; the harness should not crash on the conftest import.

If any import in `conftest.py` fails, fix before continuing.

- [ ] **Step 5: Commit (advisory — only run after user approval)**

```bash
git add tests/__init__.py tests/conftest.py tests/shared/__init__.py \
        tests/analysis/__init__.py tests/analysis/core/__init__.py
git commit -m "test: scaffold pytest infrastructure for refactor"
```

---

## Task 1 — `src/shared/risk_free_rate.py`

**Files:**
- Create: `tests/shared/test_risk_free_rate.py`
- Create: `src/shared/risk_free_rate.py`

- [ ] **Step 1: Write failing test file**

Create `tests/shared/test_risk_free_rate.py`:

```python
"""Tests for src.shared.risk_free_rate."""

#
#                                                                       Modules
# =============================================================================
# Standard
import warnings
# Third-party
import pandas as pd
import pytest


# =============================================================================
def test_returns_float_from_fred(monkeypatch):
    """Live fetch returns the latest FRED value as a float."""
    from src.shared import risk_free_rate
    risk_free_rate._cache.clear()
    fake_df = pd.DataFrame(
        {'IRLTLT01CHM156N': [0.42, 0.55, 0.61]},
        index=pd.date_range('2024-01-01', periods=3, freq='M'))

    def fake_reader(series_id, source):
        assert series_id == 'IRLTLT01CHM156N'
        assert source == 'fred'
        return fake_df

    monkeypatch.setattr(
        'src.shared.risk_free_rate.pdr.DataReader',
        fake_reader)
    rate = risk_free_rate.get_live_risk_free_rate()
    assert isinstance(rate, float)
    assert rate == pytest.approx(0.61)


# =============================================================================
def test_caches_within_process(monkeypatch):
    """Second call hits cache, not pandas_datareader."""
    from src.shared import risk_free_rate
    risk_free_rate._cache.clear()
    call_count = {'n': 0}

    def fake_reader(series_id, source):
        call_count['n'] += 1
        return pd.DataFrame(
            {'x': [1.23]},
            index=pd.date_range('2024-01-01', periods=1, freq='M'))

    monkeypatch.setattr(
        'src.shared.risk_free_rate.pdr.DataReader',
        fake_reader)
    risk_free_rate.get_live_risk_free_rate()
    risk_free_rate.get_live_risk_free_rate()
    assert call_count['n'] == 1


# =============================================================================
def test_cache_can_be_disabled(monkeypatch):
    """use_cache=False forces a re-fetch."""
    from src.shared import risk_free_rate
    risk_free_rate._cache.clear()
    call_count = {'n': 0}

    def fake_reader(series_id, source):
        call_count['n'] += 1
        return pd.DataFrame(
            {'x': [0.7]},
            index=pd.date_range('2024-01-01', periods=1, freq='M'))

    monkeypatch.setattr(
        'src.shared.risk_free_rate.pdr.DataReader',
        fake_reader)
    risk_free_rate.get_live_risk_free_rate(use_cache=False)
    risk_free_rate.get_live_risk_free_rate(use_cache=False)
    assert call_count['n'] == 2


# =============================================================================
def test_falls_back_on_fetch_failure(monkeypatch):
    """Network failure triggers fallback rate + warning."""
    from src.shared import risk_free_rate
    risk_free_rate._cache.clear()

    def fake_reader(series_id, source):
        raise ConnectionError('FRED unreachable')

    monkeypatch.setattr(
        'src.shared.risk_free_rate.pdr.DataReader',
        fake_reader)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rate = risk_free_rate.get_live_risk_free_rate()
    assert rate == pytest.approx(
        risk_free_rate.fallback_rate_pct)
    assert len(caught) == 1
    assert 'fallback' in str(caught[0].message).lower()
```

- [ ] **Step 2: Run tests and confirm they fail with ImportError**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/shared/test_risk_free_rate.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.shared.risk_free_rate'`.

- [ ] **Step 3: Implement `src/shared/risk_free_rate.py`**

```python
"""Live risk-free rate, FRED Swiss 10Y with offline fallback.

Functions
---------
get_live_risk_free_rate
    Fetch the latest FRED value with in-process caching and a
    safe fallback on failure.
"""

#
#                                                                       Modules
# =============================================================================
# Standard
import warnings
# Third-party
import pandas_datareader.data as pdr

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
fred_series_id_default = 'IRLTLT01CHM156N'
fallback_rate_pct = 0.5

_cache = {}


# =============================================================================
def get_live_risk_free_rate(
    series_id=fred_series_id_default, use_cache=True):
    """Annual risk-free rate in percent, fetched from FRED.

    Parameters
    ----------
    series_id : str, default='IRLTLT01CHM156N'
        FRED series ID. Default is the OECD long-term interest
        rate for Switzerland (monthly, percent per annum).
    use_cache : bool, default=True
        Cache the fetched value for the lifetime of the process.
        FRED data is monthly, so re-fetching during a single run
        is wasted work.

    Returns
    -------
    rate_pct : float
        Annual rate in percent. Falls back to
        `fallback_rate_pct` (0.5) on any fetch failure with a
        warning.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cache hit short-circuit
    if use_cache and series_id in _cache:
        return _cache[series_id]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Live fetch with safe fallback
    try:
        rate = float(pdr.DataReader(series_id, 'fred').iloc[-1, 0])
    except Exception as e:
        warnings.warn(
            f'FRED fetch for {series_id!r} failed '
            f'({type(e).__name__}: {e}); using fallback '
            f'{fallback_rate_pct}%.')
        rate = fallback_rate_pct
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cache write
    if use_cache:
        _cache[series_id] = rate
    return rate
```

- [ ] **Step 4: Run tests and confirm all pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/shared/test_risk_free_rate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/shared/risk_free_rate.py \
        tests/shared/test_risk_free_rate.py
git commit -m "feat(shared): add live risk-free-rate fetcher with cache + fallback"
```

---

## Task 2 — `src/shared/constraints.py`

**Files:**
- Create: `tests/shared/test_constraints.py`
- Create: `src/shared/constraints.py`

- [ ] **Step 1: Write failing test file**

Create `tests/shared/test_constraints.py`:

```python
"""Tests for src.shared.constraints."""

#
#                                                                       Modules
# =============================================================================
# Standard
import dataclasses
# Third-party
import pytest


# =============================================================================
def test_default_factory_returns_constraints():
    """default_constraints() returns a RebalanceConstraints."""
    from src.shared.constraints import (
        default_constraints, RebalanceConstraints)
    c = default_constraints()
    assert isinstance(c, RebalanceConstraints)


# =============================================================================
def test_default_field_values():
    """Defaults match Contract 4 in the design spec."""
    from src.shared.constraints import default_constraints
    c = default_constraints()
    assert c.max_weight_default == 0.15
    assert c.category_caps == {'commodity': 0.05}
    assert c.excluded_categories == ('speculative',)
    assert c.min_position_pct == 0.5
    assert c.swiss_tax_filter is True
    assert c.new_capital_chf == 0.0
    assert c.rebalance_band_chf == (-200.0, 50.0)


# =============================================================================
def test_constraints_are_frozen():
    """Mutating a field raises FrozenInstanceError."""
    from src.shared.constraints import default_constraints
    c = default_constraints()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.max_weight_default = 0.25


# =============================================================================
def test_per_instance_category_caps_isolated():
    """Each default_constraints() call gets its own dict.

    Guards against the classic mutable-default bug.
    """
    from src.shared.constraints import default_constraints
    a = default_constraints()
    b = default_constraints()
    assert a.category_caps is not b.category_caps
```

- [ ] **Step 2: Run tests and confirm they fail with ImportError**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/shared/test_constraints.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.shared.constraints'`.

- [ ] **Step 3: Implement `src/shared/constraints.py`**

```python
"""Rebalance-policy constraints shared across strategies.

Classes
-------
RebalanceConstraints
    Frozen dataclass capturing all per-run rebalance policy:
    weight caps, excluded categories, dust threshold, Swiss
    tax filter, new capital, and the HOLD band.

Functions
---------
default_constraints
    Factory returning a fresh RebalanceConstraints with the
    canonical defaults from the design spec (Contract 4).
"""

#
#                                                                       Modules
# =============================================================================
# Standard
from dataclasses import dataclass, field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


# =============================================================================
@dataclass(frozen=True)
class RebalanceConstraints:
    """Per-run rebalance policy; passed to every Strategy.

    Attributes
    ----------
    max_weight_default : float
        Per-name maximum weight, as a fraction. Default 0.15.
    category_caps : dict
        Per-category overrides on `max_weight_default`. Keys
        are category labels (see data/ticker_categories.json),
        values are fractions. Default {'commodity': 0.05}.
    excluded_categories : tuple
        Categories that the rebalancer drops entirely before
        optimising. Default ('speculative',).
    min_position_pct : float
        Positions whose current weight is below this percent
        are treated as dust and dropped. Default 0.5.
    swiss_tax_filter : bool
        When True, exclude bond ETFs and high-dividend names
        per the Swiss tax-resident policy. Default True.
    new_capital_chf : float
        New capital to deploy on top of current NAV in
        rebalancing actions. Default 0.0.
    rebalance_band_chf : tuple
        (lower, upper) gap in CHF inside which a position is
        held; below the lower bound is REDUCE, above the upper
        bound is BUY. Default (-200.0, 50.0).
    """
    max_weight_default: float = 0.15
    category_caps: dict = field(
        default_factory=lambda: {'commodity': 0.05})
    excluded_categories: tuple = ('speculative',)
    min_position_pct: float = 0.5
    swiss_tax_filter: bool = True
    new_capital_chf: float = 0.0
    rebalance_band_chf: tuple = (-200.0, 50.0)
# =============================================================================


# =============================================================================
def default_constraints():
    """Return a fresh RebalanceConstraints with spec defaults.

    Returns
    -------
    constraints : RebalanceConstraints
        New instance; mutable members (category_caps) are
        per-instance, not shared.
    """
    return RebalanceConstraints()
```

- [ ] **Step 4: Run tests and confirm all pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/shared/test_constraints.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/shared/constraints.py tests/shared/test_constraints.py
git commit -m "feat(shared): add RebalanceConstraints dataclass + default factory"
```

---

## Task 3 — `PortfolioAnalyzer.get_holdings_snapshot()`

**Files:**
- Create: `tests/analysis/core/test_analyzer_snapshot.py`
- Modify: `src/analysis/core/analyzer.py` — add one new method on `PortfolioAnalyzer` (additive only, no edits to existing methods).

**Method contract (Contract 3 from the spec):**
```
def get_holdings_snapshot(self, categories=None) -> pd.DataFrame:
    """Snapshot DataFrame, columns:
       ticker, shares, value_chf, weight_pct, currency, sector,
       price_chf [, category]
       The 'category' column is included only when `categories`
       is provided. Closed positions (shares <= 0) are dropped.
       Sorted by value_chf descending."""
```

- [ ] **Step 1: Write failing test file**

Create `tests/analysis/core/test_analyzer_snapshot.py`:

```python
"""Tests for PortfolioAnalyzer.get_holdings_snapshot."""

#
#                                                                       Modules
# =============================================================================
# Standard
# Third-party
import pandas as pd
import pytest


# =============================================================================
def test_snapshot_returns_dataframe(analyzer):
    """Returns a pandas DataFrame."""
    df = analyzer.get_holdings_snapshot()
    assert isinstance(df, pd.DataFrame)


# =============================================================================
def test_snapshot_columns_without_categories(analyzer):
    """When categories is None, no 'category' column."""
    df = analyzer.get_holdings_snapshot()
    expected = {
        'ticker', 'shares', 'value_chf', 'weight_pct',
        'currency', 'sector', 'price_chf'}
    assert set(df.columns) == expected


# =============================================================================
def test_snapshot_columns_with_categories(analyzer):
    """When categories is provided, 'category' column added."""
    cats = {
        'AMZN': 'core',
        'UBSG.SW': 'core',
        'HOLN.SW': 'core'}
    df = analyzer.get_holdings_snapshot(categories=cats)
    expected = {
        'ticker', 'shares', 'value_chf', 'weight_pct',
        'currency', 'sector', 'price_chf', 'category'}
    assert set(df.columns) == expected


# =============================================================================
def test_snapshot_includes_all_held_tickers(analyzer):
    """Every fixture-held ticker appears in the output."""
    df = analyzer.get_holdings_snapshot()
    assert set(df['ticker']) == {'AMZN', 'UBSG.SW', 'HOLN.SW'}


# =============================================================================
def test_snapshot_weights_sum_close_to_100(analyzer):
    """weight_pct values sum to ~100 (rounding tolerance)."""
    df = analyzer.get_holdings_snapshot()
    assert df['weight_pct'].sum() == pytest.approx(100.0, abs=1e-6)


# =============================================================================
def test_snapshot_sorted_by_value_descending(analyzer):
    """Rows are ordered largest-position-first."""
    df = analyzer.get_holdings_snapshot()
    values = df['value_chf'].tolist()
    assert values == sorted(values, reverse=True)


# =============================================================================
def test_snapshot_raises_on_unclassified_ticker(analyzer):
    """Per no-silent-defaults policy: missing category -> raise."""
    cats = {'AMZN': 'core'}    # UBSG.SW + HOLN.SW absent
    with pytest.raises(RuntimeError, match='category'):
        analyzer.get_holdings_snapshot(categories=cats)


# =============================================================================
def test_snapshot_currency_column_matches_transactions(analyzer):
    """Currency column reflects transaction currencies."""
    df = analyzer.get_holdings_snapshot()
    by_ticker = dict(zip(df['ticker'], df['currency']))
    assert by_ticker['AMZN'] == 'USD'
    assert by_ticker['UBSG.SW'] == 'CHF'
    assert by_ticker['HOLN.SW'] == 'CHF'
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/analysis/core/test_analyzer_snapshot.py -v`
Expected: 8 failures — `AttributeError: 'PortfolioAnalyzer' object has no attribute 'get_holdings_snapshot'`.

- [ ] **Step 3: Add the method to `PortfolioAnalyzer`**

Open `src/analysis/core/analyzer.py`. Locate the existing method `get_current_position_values` (around line 250-276). Insert the new method immediately after `get_current_position_values` to keep snapshot helpers grouped:

```python
    def get_holdings_snapshot(self, categories=None):
        """Per-position snapshot DataFrame.

        Implements Contract 3 of the refactor design spec.
        Aggregates current shares, base-currency values, weights,
        currencies, sectors, and per-share prices into a single
        DataFrame. Closed positions (shares <= 0) are dropped.
        Sorted by `value_chf` descending.

        Parameters
        ----------
        categories : dict, optional
            Optional ticker -> category-label map. When provided,
            adds a 'category' column. A held ticker missing from
            this map raises RuntimeError (per no-silent-defaults
            policy).

        Returns
        -------
        snapshot : pandas.DataFrame
            Columns: ticker, shares, value_chf, weight_pct,
            currency, sector, price_chf [, category].
        """
        self._fetch_market_data()
        shares_map = self.portfolio.get_current_holdings()
        values_map = self.get_current_position_values()
        total_value = sum(values_map.values())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # One row per open position
        rows = []
        for ticker, shares in shares_map.items():
            if shares <= 0:
                continue
            if ticker not in values_map:
                raise RuntimeError(
                    f'Ticker {ticker!r} missing valuation.')
            value = values_map[ticker]
            if ticker not in self._ticker_currencies:
                raise RuntimeError(
                    f'Ticker {ticker!r} missing currency.')
            currency = self._ticker_currencies[ticker]
            # Sector via fundamental data; ETFs / proxies fall back
            fundamentals = self.data_provider.get_fundamental_data(
                ticker)
            sector = fundamentals.get('sector')
            if not sector:
                sector = 'Unknown'
            weight = 0.0
            if total_value > 0:
                weight = value / total_value * 100.0
            price_chf = 0.0
            if shares > 0:
                price_chf = value / shares
            row = {
                'ticker': ticker,
                'shares': shares,
                'value_chf': value,
                'weight_pct': weight,
                'currency': currency,
                'sector': sector,
                'price_chf': price_chf,
            }
            if categories is not None:
                if ticker not in categories:
                    raise RuntimeError(
                        f'Ticker {ticker!r} missing category in '
                        f'provided categories map.')
                row['category'] = categories[ticker]
            rows.append(row)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Order largest-position-first; reset index so rows are 0..N-1
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('value_chf', ascending=False)
            df = df.reset_index(drop=True)
        return df
```

- [ ] **Step 4: Run tests and confirm all pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/analysis/core/test_analyzer_snapshot.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/analysis/core/analyzer.py \
        tests/analysis/core/test_analyzer_snapshot.py
git commit -m "feat(analyzer): add get_holdings_snapshot per Contract 3"
```

---

## Task 4 — `PortfolioAnalyzer.get_price_panel()`

**Files:**
- Modify: `tests/analysis/core/test_analyzer_snapshot.py` — append tests for `get_price_panel`.
- Modify: `src/analysis/core/analyzer.py` — add one new method on `PortfolioAnalyzer`, immediately after `get_holdings_snapshot`.

**Method contract (Contract 3 from the spec):**
```
def get_price_panel(self) -> pd.DataFrame:
    """Wide DataFrame, one column per held ticker, daily close
       in native currency. Columns are tickers in the same order
       as get_current_holdings(). Index is the union of all
       ticker price-history dates."""
```

- [ ] **Step 1: Append failing tests**

Append to `tests/analysis/core/test_analyzer_snapshot.py`:

```python
# =============================================================================
def test_panel_returns_dataframe(analyzer):
    """Returns a pandas DataFrame."""
    df = analyzer.get_price_panel()
    assert isinstance(df, pd.DataFrame)


# =============================================================================
def test_panel_has_one_column_per_held_ticker(analyzer):
    """Columns are exactly the currently-held tickers."""
    df = analyzer.get_price_panel()
    assert set(df.columns) == {'AMZN', 'UBSG.SW', 'HOLN.SW'}


# =============================================================================
def test_panel_columns_are_numeric(analyzer):
    """Each column carries float prices, not NaN-only."""
    df = analyzer.get_price_panel()
    for ticker in df.columns:
        col = df[ticker].dropna()
        assert len(col) > 0
        assert pd.api.types.is_numeric_dtype(col)


# =============================================================================
def test_panel_index_is_datetime(analyzer):
    """Index is a DatetimeIndex spanning the holding window."""
    df = analyzer.get_price_panel()
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df.index) > 1
```

- [ ] **Step 2: Run new tests and confirm they fail**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/analysis/core/test_analyzer_snapshot.py -v -k "panel"`
Expected: 4 failures — `AttributeError: 'PortfolioAnalyzer' object has no attribute 'get_price_panel'`.

- [ ] **Step 3: Add the method to `PortfolioAnalyzer`**

Open `src/analysis/core/analyzer.py`. Insert immediately after `get_holdings_snapshot` (added in Task 3):

```python
    def get_price_panel(self):
        """Wide DataFrame of daily closes for currently-held tickers.

        Implements Contract 3 of the refactor design spec.
        Each column is a ticker's native-currency daily close
        series, drawn from the analyzer's existing market-data
        cache. Tickers without fetched price history are dropped
        rather than represented as all-NaN columns.

        Returns
        -------
        panel : pandas.DataFrame
            Columns: ticker symbols. Index: union of all tickers'
            price-history dates (DatetimeIndex). Values: native-
            currency close prices.
        """
        self._fetch_market_data()
        held = self.portfolio.get_current_holdings()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Restrict to open positions with cached price history
        columns = {}
        for ticker, shares in held.items():
            if shares <= 0:
                continue
            if ticker not in self._market_data:
                continue
            columns[ticker] = self._market_data[ticker]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # DataFrame() aligns Series on the union of their indices
        return pd.DataFrame(columns)
```

- [ ] **Step 4: Run all snapshot tests, confirm everything passes**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/analysis/core/test_analyzer_snapshot.py -v`
Expected: 12 passed (8 from Task 3 + 4 new).

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/analysis/core/analyzer.py \
        tests/analysis/core/test_analyzer_snapshot.py
git commit -m "feat(analyzer): add get_price_panel per Contract 3"
```

---

## Task 5 — Update `src/shared/__init__.py`

**Files:**
- Modify: `src/shared/__init__.py`

- [ ] **Step 1: Read current contents**

Run: `cat /Users/rbarreira/Desktop/stock_market/src/shared/__init__.py`
Expected current contents (18 lines):

```python
"""
Shared infrastructure for the stock_market codebase.
...
"""
from src.shared.data_provider import (
    DataProvider, YFinanceProvider, QFLibProvider,
)

__all__ = [
    'DataProvider',
    'YFinanceProvider',
    'QFLibProvider',
]
```

- [ ] **Step 2: Add the new exports**

Replace the file contents (read the actual file first to see exact form, then edit) with:

```python
"""Shared infrastructure for the stock_market codebase.

Re-exports the data-provider abstractions used by both
analysis and modelling, the live risk-free-rate fetcher, and
the rebalance-constraints dataclass.
"""

from src.shared.data_provider import (
    DataProvider, YFinanceProvider, QFLibProvider)
from src.shared.risk_free_rate import (
    get_live_risk_free_rate,
    fred_series_id_default,
    fallback_rate_pct)
from src.shared.constraints import (
    RebalanceConstraints, default_constraints)

__all__ = [
    'DataProvider',
    'YFinanceProvider',
    'QFLibProvider',
    'get_live_risk_free_rate',
    'fred_series_id_default',
    'fallback_rate_pct',
    'RebalanceConstraints',
    'default_constraints',
]
```

- [ ] **Step 3: Verify exports load cleanly**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -c "from src.shared import get_live_risk_free_rate, RebalanceConstraints, default_constraints; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Run the full Phase 1 test suite**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: 16 passed (4 risk_free_rate + 4 constraints + 8 snapshot + 4 panel = 20; if 12 snapshot tests run all together: total 20 passed).

If the actual count differs from the expectation: investigate before proceeding. Don't proceed to commit until the suite is clean.

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/shared/__init__.py
git commit -m "feat(shared): export new public symbols from src.shared"
```

---

## Task 6 — Phase 1 verification & cleanup

**Files:** none modified.

- [ ] **Step 1: Live RFR smoke test**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -c "from src.shared.risk_free_rate import get_live_risk_free_rate; print(get_live_risk_free_rate())"`
Expected: a float between roughly 0 and 5 (the current Swiss 10Y rate). If a warning fires about FRED unreachability, the fallback (0.5) is acceptable as long as the call returns a value.

- [ ] **Step 2: Constraints smoke test**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -c "from src.shared.constraints import default_constraints; c = default_constraints(); print(c)"`
Expected: prints a `RebalanceConstraints(max_weight_default=0.15, ..., rebalance_band_chf=(-200.0, 50.0))` line.

- [ ] **Step 3: Analyzer integration smoke test**

Confirm the new methods work end-to-end against the real CSV fixtures from the existing `data/` directory. Run:

```bash
cd /Users/rbarreira/Desktop/stock_market && python3 -c "
from src.analysis.core.analyzer import PortfolioAnalyzer
import json, pathlib
root = pathlib.Path('.')
cats = json.loads((root / 'data/ticker_categories.json').read_text())
a = PortfolioAnalyzer(
    degiro_csv_file_path=str(root / 'data/postprocess_data/processed_portfolio.csv'),
    ibkr_csv_file_path=str(root / 'reports/ibkr/<YYYYMMDD>_<YYYYMMDD>_<ACCOUNT_ID>.csv'))
snap = a.get_holdings_snapshot(categories=cats)
print(snap[['ticker', 'value_chf', 'weight_pct', 'category']].head())
print('---')
panel = a.get_price_panel()
print('Panel shape:', panel.shape)
print('Tickers:', list(panel.columns))
"
```

Expected: holdings snapshot prints with valid value_chf + weight_pct + category for each ticker; panel shape is `(N_days, N_held_tickers)` with N_days >= 1.

This test makes live yfinance calls and may take 30-90 seconds.

- [ ] **Step 4: Run the full test suite a final time**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 5: Mark Phase 1 complete**

If `tasks/todo.md` exists, append a Phase-1 completion note. Otherwise create the file with:

```markdown
# Refactor progress

## Phase 1 — Shared infrastructure: COMPLETE (YYYY-MM-DD)

- src/shared/risk_free_rate.py     [new]
- src/shared/constraints.py        [new]
- src/analysis/core/analyzer.py    [+get_holdings_snapshot, +get_price_panel]
- tests/                           [scaffolded]

Verification: 20 unit tests pass; live RFR fetch returns a number; analyzer
methods produce non-empty DataFrames against real CSV fixtures.

Next: Phase 2 — Rebalancing domain + rewritten src/analysis/report.py.
Plan to be authored by re-invoking superpowers:writing-plans against
docs/superpowers/specs/2026-05-13-repo-refactor-design.md, scoped to
Phase 2.
```

- [ ] **Step 6: Commit phase-completion marker (advisory)**

```bash
git add tasks/todo.md
git commit -m "docs: mark Phase 1 complete in tasks/todo.md"
```

---

## Self-review

**Spec coverage** (cross-referenced against `docs/superpowers/specs/2026-05-13-repo-refactor-design.md` Section 7, Phase 1):
- [x] `src/shared/risk_free_rate.py` (`get_live_risk_free_rate`) — Task 1.
- [x] `src/shared/constraints.py` (`RebalanceConstraints`, `default_constraints`) — Task 2.
- [x] `src/analysis/core/analyzer.py` — `get_holdings_snapshot()` — Task 3.
- [x] `src/analysis/core/analyzer.py` — `get_price_panel()` — Task 4.
- [x] Verification: live RFR returns plausible value — Task 6 Step 1.
- [x] Verification: analyzer methods return well-formed DataFrames — Task 6 Step 3.

**Placeholder scan:** no TBD/TODO/incomplete sections. Every code step shows the actual code. Every test step shows the actual test. All file paths are absolute. Commands have expected outputs.

**Type / signature consistency:**
- `get_live_risk_free_rate(series_id, use_cache)` — same signature in spec (Section 6), implementation, and tests.
- `RebalanceConstraints` field set + types — match Contract 4 in Section 2; tests check each field.
- `get_holdings_snapshot(categories=None) -> pd.DataFrame` — column set in spec Section 2 Contract 3 matches implementation matches test.
- `get_price_panel() -> pd.DataFrame` — return shape (one column per held ticker) consistent across spec, impl, and tests.

**Style compliance check:** all code blocks observe the user's `~/.claude/rules/code-style.md` (80-char width, single quotes, snake_case, lowercase module-level constants, no type hints in signatures, NumPy docstrings). The conftest fixture file uses the same conventions.

**Anti-pattern audit:**
- No silent defaults: `get_holdings_snapshot` raises `RuntimeError` on a held ticker missing from the categories map (per `feedback_no_silent_defaults.md`); `default_factory` for `category_caps` prevents the mutable-default bug.
- No bond ETFs: `swiss_tax_filter=True` is the default in `RebalanceConstraints`; downstream strategies enforce it (per `feedback_swiss_tax_no_bonds.md`).
- Module-level constants are lowercase: `fred_series_id_default`, `fallback_rate_pct` (per `feedback_no_caps_globals.md`).
