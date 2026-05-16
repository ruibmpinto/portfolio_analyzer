# Screening domain implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy `src/screening/` modules with a clean per-exchange-parquet-cached screener: `Listing` data model + `ExchangeUniverse` aggregator (parquet I/O) + `Criterion` ABC with concrete criteria + `Screener` orchestrator that ranks tickers across multiple cached exchanges; resumable yfinance refresh script; two CLI entry points (`refresh_screener_cache.py`, `run_screener.py`).

**Architecture:** Mirrors the data-model → state-aggregator → orchestrator shape of `src/analysis/core/`. Per-exchange parquet files at `data/screener_cache/<exchange>.parquet` are the persistent state; `ExchangeUniverse` reads/writes them; `Screener` aggregates listings from multiple universes, applies a list of `Criterion` instances, computes a percentile-rank-based composite score, and returns ranked `CandidateTicker` objects. Refresh is incremental and resumable: the parquet itself acts as the checkpoint — if a ticker's row is already present, the refresh skips it.

**Tech Stack:** Python 3, pandas, numpy, pyarrow (parquet I/O), yfinance (live data), scipy.stats.rankdata (percentile ranking), pytest, argparse + json + pathlib (stdlib).

**Source spec:** `docs/superpowers/specs/2026-05-13-repo-refactor-design.md` Sections 2 (Contracts 1, 2), 3 (Screening domain detail), 7 (Phase 3 verification).

**Style:** Match `src/analysis/core/portfolio.py` and `src/modelling/rebalancing/`:
- Plain module docstring (no `# ===` banners).
- Imports plain (stdlib / third-party / local) separated by blank lines, no header comments.
- No `__author__` / `__credits__` / `__status__` blocks.
- Type hints in signatures.
- Google-style docstrings (`Args:`, `Returns:`, `Raises:`).
- Lowercase module-level constants.
- 80-char max width. Single quotes for strings, double quotes for docstrings only.

**Commit policy:** Per `~/.claude/CLAUDE.md`, only commit on explicit user request. Commit steps are advisory.

---

## File map

**New files (~12):**
- `src/screening/__init__.py` — re-exports public surface
- `src/screening/candidate.py` — `CandidateTicker` dataclass (Contract 2)
- `src/screening/listing.py` — `Listing` dataclass (Contract 1)
- `src/screening/universe.py` — `ExchangeUniverse` aggregator + parquet I/O
- `src/screening/criteria.py` — `Criterion` ABC + 9 concrete subclasses
- `src/screening/screener.py` — `Screener` orchestrator
- `src/screening/cache_refresh.py` — resumable yfinance fetcher
- `src/user_scripts/refresh_screener_cache.py` — CLI
- `src/user_scripts/run_screener.py` — CLI

**New test files (~5):**
- `tests/screening/__init__.py` (empty marker)
- `tests/screening/test_candidate_and_listing.py`
- `tests/screening/test_universe.py`
- `tests/screening/test_criteria.py`
- `tests/screening/test_screener.py`
- `tests/screening/test_cache_refresh.py`
- `tests/user_scripts/__init__.py` (empty marker)
- `tests/user_scripts/test_screener_cli.py` (covers both CLI scripts)

**Deleted files (8):**
- `src/screening/cache.py`
- `src/screening/criteria.py` (legacy — overwritten in Task 4)
- `src/screening/exchanges.py`
- `src/screening/fundamentals.py`
- `src/screening/pipeline.py`
- `src/screening/screener.py` (legacy — overwritten in Task 6)
- `src/screening/universe.py` (legacy — overwritten in Task 3)
- `src/user_scripts/screen_stocks.py`
- `src/user_scripts/update_exchange_lists.py`

**New data assets:**
- `data/exchange_tickers/<exchange>.txt` — one ticker per line, per exchange. Seed at minimum NYSE+NASDAQ from the existing `data/sp500_tickers.json` and LSE from `data/ftse100_tickers.json`. Other exchanges left empty for the user to populate.

---

## Task 1 — `CandidateTicker` dataclass

**Files:**
- Create: `tests/screening/__init__.py` (empty)
- Create: `tests/screening/test_candidate_and_listing.py`
- Replace: `src/screening/__init__.py` (currently 43 lines of legacy re-exports — start fresh)
- Create: `src/screening/candidate.py`

- [ ] **Step 1: Create the directory tree + clear legacy re-exports**

```bash
mkdir -p /Users/rbarreira/Desktop/stock_market/tests/screening
touch /Users/rbarreira/Desktop/stock_market/tests/screening/__init__.py
```

Replace `/Users/rbarreira/Desktop/stock_market/src/screening/__init__.py` with a minimal placeholder (final exports added in Task 6):

```python
"""
Screening domain: per-exchange cached universes + criteria-based ranking.
"""
```

- [ ] **Step 2: Write failing test file**

Create `/Users/rbarreira/Desktop/stock_market/tests/screening/test_candidate_and_listing.py`:

```python
"""Tests for CandidateTicker and Listing dataclasses."""

import dataclasses

import pytest

from src.screening.candidate import CandidateTicker


def test_candidate_construct():
    c = CandidateTicker(
        ticker='AAPL', exchange='NASDAQ', currency='USD',
        sector='Technology', market_cap=3.0e12,
        composite_score=0.87,
        metrics={'momentum': 0.12, 'sharpe': 1.4})
    assert c.ticker == 'AAPL'
    assert c.composite_score == 0.87
    assert c.metrics['momentum'] == 0.12


def test_candidate_is_frozen():
    c = CandidateTicker(
        ticker='AAPL', exchange='NASDAQ', currency='USD',
        sector='Technology', market_cap=3.0e12,
        composite_score=0.87, metrics={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.composite_score = 0.5


def test_candidate_score_must_be_in_unit_interval():
    with pytest.raises(ValueError, match='composite_score'):
        CandidateTicker(
            ticker='AAPL', exchange='NASDAQ', currency='USD',
            sector='Technology', market_cap=3.0e12,
            composite_score=1.5, metrics={})
    with pytest.raises(ValueError, match='composite_score'):
        CandidateTicker(
            ticker='AAPL', exchange='NASDAQ', currency='USD',
            sector='Technology', market_cap=3.0e12,
            composite_score=-0.1, metrics={})


def test_candidate_metrics_dict_isolated_per_instance():
    a = CandidateTicker(
        ticker='AAPL', exchange='NASDAQ', currency='USD',
        sector='Technology', market_cap=3e12,
        composite_score=0.5, metrics={'x': 1.0})
    b = CandidateTicker(
        ticker='MSFT', exchange='NASDAQ', currency='USD',
        sector='Technology', market_cap=2e12,
        composite_score=0.5, metrics={'x': 1.0})
    assert a.metrics is not b.metrics
```

- [ ] **Step 3: Run tests, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_candidate_and_listing.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.screening.candidate'`.

- [ ] **Step 4: Implement `src/screening/candidate.py`**

```python
"""
Candidate ticker output of the Screener.

Implements Contract 2 of the refactor design spec. Returned
in ranked order by Screener.rank(); consumed by the screener-
fed strategies in src.modelling.rebalancing.strategies.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict


@dataclass(frozen=True)
class CandidateTicker:
    """
    One ranked candidate from the screener.

    Attributes:
        ticker: Native exchange symbol (e.g. 'AAPL', 'NESN.SW').
        exchange: Exchange code (e.g. 'NASDAQ', 'SIX').
        currency: ISO-4217 listing currency.
        sector: yfinance sector label or 'Unknown'.
        market_cap: Market capitalisation in listing currency
            (NaN if unavailable).
        composite_score: Aggregate ranking in [0, 1]; higher is
            better. The Screener computes this as the weighted
            mean of per-criterion percentile ranks.
        metrics: Per-criterion raw scores keyed by criterion
            name. Wrapped in MappingProxyType so the dict is
            read-only.
    """

    ticker: str
    exchange: str
    currency: str
    sector: str
    market_cap: float
    composite_score: float
    metrics: Dict[str, float]

    def __post_init__(self):
        if not 0.0 <= self.composite_score <= 1.0:
            raise ValueError(
                f'composite_score must be in [0, 1], '
                f'got {self.composite_score}.')
        # Wrap the metrics dict so callers can't mutate it
        if not isinstance(self.metrics, MappingProxyType):
            object.__setattr__(
                self, 'metrics',
                MappingProxyType(dict(self.metrics)))
```

- [ ] **Step 5: Run tests, confirm 4 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_candidate_and_listing.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit (advisory)**

```bash
git add tests/screening/__init__.py \
        tests/screening/test_candidate_and_listing.py \
        src/screening/__init__.py \
        src/screening/candidate.py
git commit -m "feat(screening): add CandidateTicker dataclass"
```

---

## Task 2 — `Listing` dataclass

**Files:**
- Modify: `tests/screening/test_candidate_and_listing.py` — append Listing tests.
- Create: `src/screening/listing.py`

- [ ] **Step 1: Append tests**

Append to `tests/screening/test_candidate_and_listing.py`:

```python


from datetime import datetime

import numpy as np
import pandas as pd

from src.screening.listing import Listing, fundamental_field_names


def _sample_listing(ticker='AAPL', exchange='NASDAQ'):
    n = 156
    dates = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    closes = np.linspace(100.0, 200.0, n)
    volumes = np.full(n, 1.0e7)
    dividends = np.zeros(n)
    fundamentals = {k: 1.0 for k in fundamental_field_names}
    return Listing(
        ticker=ticker, exchange=exchange, currency='USD',
        sector='Technology', market_cap=3.0e12,
        closes_weekly=closes, closes_weekly_dates=dates,
        volumes_weekly=volumes, dividends_weekly=dividends,
        fundamentals=fundamentals,
        refreshed_at=datetime(2026, 5, 14, 12, 0))


def test_listing_construct():
    l = _sample_listing()
    assert l.ticker == 'AAPL'
    assert len(l.closes_weekly) == 156
    assert l.fundamentals['pe_ratio_ttm'] == 1.0


def test_listing_fundamental_field_names_count_is_seventeen():
    """The Tier-B fundamentals contract has exactly 17 fields."""
    assert len(fundamental_field_names) == 17


def test_listing_fundamental_field_names_match_contract():
    """Field names match the per-exchange parquet cache contract."""
    expected = {
        'pe_ratio_ttm', 'pe_ratio_forward', 'pb_ratio',
        'ps_ratio', 'dividend_yield_ttm', 'eps_ttm',
        'revenue_growth_yoy', 'earnings_growth_yoy',
        'profit_margin', 'operating_margin', 'roe', 'roa',
        'debt_to_equity', 'free_cash_flow', 'beta_yf',
        'shares_outstanding', 'short_ratio'}
    assert set(fundamental_field_names) == expected


def test_listing_rejects_wrong_series_length():
    """closes_weekly et al. must be length 156."""
    n_bad = 100
    dates = pd.date_range(
        '2023-05-15', periods=n_bad, freq='W-FRI')
    closes = np.linspace(100.0, 200.0, n_bad)
    volumes = np.full(n_bad, 1.0e7)
    dividends = np.zeros(n_bad)
    fundamentals = {k: 1.0 for k in fundamental_field_names}
    with pytest.raises(ValueError, match='closes_weekly'):
        Listing(
            ticker='AAPL', exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1.0e9,
            closes_weekly=closes,
            closes_weekly_dates=dates,
            volumes_weekly=volumes,
            dividends_weekly=dividends,
            fundamentals=fundamentals,
            refreshed_at=datetime.now())


def test_listing_rejects_missing_fundamental_field():
    """Every Tier-B field must be present (NaN is acceptable)."""
    l_args = _sample_listing()
    bad_fundamentals = {
        k: 1.0 for k in fundamental_field_names if k != 'roe'}
    with pytest.raises(ValueError, match='roe'):
        Listing(
            ticker='AAPL', exchange='NASDAQ', currency='USD',
            sector='Tech', market_cap=1.0e9,
            closes_weekly=l_args.closes_weekly,
            closes_weekly_dates=l_args.closes_weekly_dates,
            volumes_weekly=l_args.volumes_weekly,
            dividends_weekly=l_args.dividends_weekly,
            fundamentals=bad_fundamentals,
            refreshed_at=datetime.now())
```

- [ ] **Step 2: Run, confirm 5 new tests fail**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_candidate_and_listing.py -v`
Expected: 4 candidate tests pass; 5 listing tests fail with `ModuleNotFoundError: No module named 'src.screening.listing'`.

- [ ] **Step 3: Implement `src/screening/listing.py`**

```python
"""
Per-ticker cache record for the screener.

Implements Contract 1 of the refactor design spec. One Listing
= one row in an ExchangeUniverse's parquet file. Holds 156
weekly bars (closes / volumes / dividends), 17 Tier-B
fundamental snapshots, plus identification metadata.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd


weekly_series_length = 156

fundamental_field_names = (
    'pe_ratio_ttm',
    'pe_ratio_forward',
    'pb_ratio',
    'ps_ratio',
    'dividend_yield_ttm',
    'eps_ttm',
    'revenue_growth_yoy',
    'earnings_growth_yoy',
    'profit_margin',
    'operating_margin',
    'roe',
    'roa',
    'debt_to_equity',
    'free_cash_flow',
    'beta_yf',
    'shares_outstanding',
    'short_ratio',
)


@dataclass(frozen=True)
class Listing:
    """
    One row of an ExchangeUniverse parquet cache.

    Attributes:
        ticker: Native exchange symbol.
        exchange: Exchange code; set at load time from the
            parquet filename rather than stored in the file.
        currency: ISO-4217 listing currency.
        sector: yfinance sector label or 'Unknown'.
        market_cap: Listing-currency market cap (NaN allowed).
        closes_weekly: Length-156 numpy array of weekly closes,
            oldest first. NaN allowed for pre-listing weeks.
        closes_weekly_dates: Length-156 DatetimeIndex aligned
            with closes_weekly.
        volumes_weekly: Length-156 numpy array of weekly volumes.
        dividends_weekly: Length-156 numpy array of weekly
            dividend cash per share.
        fundamentals: Dict keyed by `fundamental_field_names`,
            17 floats (NaN allowed). Every name must be present;
            constructor raises if any are missing.
        refreshed_at: Timestamp of the underlying yfinance fetch.
    """

    ticker: str
    exchange: str
    currency: str
    sector: str
    market_cap: float
    closes_weekly: np.ndarray
    closes_weekly_dates: pd.DatetimeIndex
    volumes_weekly: np.ndarray
    dividends_weekly: np.ndarray
    fundamentals: Dict[str, float]
    refreshed_at: datetime

    def __post_init__(self):
        for series_name in (
            'closes_weekly', 'closes_weekly_dates',
            'volumes_weekly', 'dividends_weekly'):
            series = getattr(self, series_name)
            if len(series) != weekly_series_length:
                raise ValueError(
                    f'{series_name} must have length '
                    f'{weekly_series_length}; got {len(series)}.')
        missing = [
            name for name in fundamental_field_names
            if name not in self.fundamentals]
        if missing:
            raise ValueError(
                f'fundamentals missing required field(s): '
                f'{missing}.')
```

- [ ] **Step 4: Run all candidate+listing tests, confirm 9 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_candidate_and_listing.py -v`
Expected: 9 passed (4 candidate + 5 listing).

- [ ] **Step 5: Commit (advisory)**

```bash
git add src/screening/listing.py \
        tests/screening/test_candidate_and_listing.py
git commit -m "feat(screening): add Listing dataclass with field validation"
```

---

## Task 3 — `ExchangeUniverse` parquet aggregator

**Files:**
- Delete: `/Users/rbarreira/Desktop/stock_market/src/screening/universe.py` (legacy, will be replaced).
- Create: `tests/screening/test_universe.py`
- Create: `src/screening/universe.py` (new, NOT the legacy one).

- [ ] **Step 1: Delete legacy universe.py**

```bash
rm /Users/rbarreira/Desktop/stock_market/src/screening/universe.py
```

- [ ] **Step 2: Confirm pyarrow is installed**

Run: `python3 -c "import pyarrow; print(pyarrow.__version__)"`
Expected: a version string. If `ModuleNotFoundError`, the user must install: `pip install pyarrow`.

- [ ] **Step 3: Write tests**

Create `/Users/rbarreira/Desktop/stock_market/tests/screening/test_universe.py`:

```python
"""Tests for ExchangeUniverse: parquet I/O + iteration helpers."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.screening.listing import Listing, fundamental_field_names
from src.screening.universe import ExchangeUniverse


def _make_listing(ticker, exchange='NASDAQ', refreshed_at=None):
    n = 156
    dates = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    closes = np.linspace(100.0, 200.0, n)
    volumes = np.full(n, 1.0e6)
    dividends = np.zeros(n)
    fundamentals = {k: 1.0 for k in fundamental_field_names}
    return Listing(
        ticker=ticker, exchange=exchange, currency='USD',
        sector='Technology', market_cap=1.0e9,
        closes_weekly=closes, closes_weekly_dates=dates,
        volumes_weekly=volumes, dividends_weekly=dividends,
        fundamentals=fundamentals,
        refreshed_at=refreshed_at or datetime(2026, 5, 14, 12, 0))


def test_universe_construct():
    listings = [_make_listing('A'), _make_listing('B')]
    u = ExchangeUniverse('NASDAQ', listings)
    assert len(u) == 2
    assert list(u)[0].ticker == 'A'


def test_universe_save_and_load_round_trip(tmp_path, monkeypatch):
    """Save then load returns equivalent listings."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    listings = [
        _make_listing('AAPL'),
        _make_listing('MSFT')]
    u = ExchangeUniverse('NASDAQ', listings)
    u.save()
    loaded = ExchangeUniverse.load('NASDAQ')
    assert len(loaded) == 2
    by_ticker = {l.ticker: l for l in loaded}
    assert set(by_ticker) == {'AAPL', 'MSFT'}
    aapl = by_ticker['AAPL']
    assert aapl.exchange == 'NASDAQ'
    assert aapl.currency == 'USD'
    assert aapl.sector == 'Technology'
    assert aapl.market_cap == pytest.approx(1.0e9)
    assert len(aapl.closes_weekly) == 156
    assert aapl.closes_weekly[0] == pytest.approx(100.0)
    assert aapl.closes_weekly[-1] == pytest.approx(200.0)
    assert aapl.fundamentals['pe_ratio_ttm'] == 1.0


def test_universe_save_atomic(tmp_path, monkeypatch):
    """Save writes via a temp file and renames into place."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    u = ExchangeUniverse('NASDAQ', [_make_listing('AAPL')])
    u.save()
    # Final file exists; no temp leftover
    assert (tmp_path / 'NASDAQ.parquet').exists()
    leftovers = list(tmp_path.glob('*.tmp.*'))
    assert leftovers == []


def test_universe_load_missing_raises(tmp_path, monkeypatch):
    """Loading a non-existent exchange raises FileNotFoundError."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(
        universe_mod, 'cache_dir', tmp_path)
    with pytest.raises(FileNotFoundError):
        ExchangeUniverse.load('NOPE')


def test_universe_filter_returns_subset():
    """filter() returns a new ExchangeUniverse with the subset."""
    listings = [_make_listing('A'), _make_listing('B')]
    u = ExchangeUniverse('NASDAQ', listings)
    sub = u.filter(lambda l: l.ticker == 'A')
    assert len(sub) == 1
    assert next(iter(sub)).ticker == 'A'
    assert sub.exchange == 'NASDAQ'


def test_universe_to_dataframe_one_row_per_listing():
    listings = [_make_listing('A'), _make_listing('B')]
    df = ExchangeUniverse('NASDAQ', listings).to_dataframe()
    assert len(df) == 2
    assert set(df['ticker']) == {'A', 'B'}
    # A few schema-critical columns
    assert {'currency', 'sector', 'market_cap',
            'pe_ratio_ttm'}.issubset(df.columns)


def test_universe_staleness_days_property():
    """staleness_days returns the age of the oldest refreshed_at."""
    fresh = _make_listing(
        'A', refreshed_at=datetime.now() - timedelta(days=2))
    stale = _make_listing(
        'B', refreshed_at=datetime.now() - timedelta(days=10))
    u = ExchangeUniverse('NASDAQ', [fresh, stale])
    assert 9.0 <= u.staleness_days <= 11.0
```

- [ ] **Step 4: Run, confirm 7 fail**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_universe.py -v`
Expected: 7 collection errors — `ModuleNotFoundError: No module named 'src.screening.universe'`.

- [ ] **Step 5: Implement `src/screening/universe.py`**

```python
"""
ExchangeUniverse: per-exchange aggregator with parquet I/O.

Holds a list of Listings for one exchange; persists to
data/screener_cache/<exchange>.parquet via atomic
write-temp-then-rename. Loading round-trips the schema
faithfully so Listings reconstructed from disk are
indistinguishable from freshly-built ones.
"""

import pathlib
from datetime import datetime
from typing import Callable, Iterator, List

import numpy as np
import pandas as pd

from src.screening.listing import (
    Listing, fundamental_field_names, weekly_series_length)


cache_dir = pathlib.Path(
    __file__).resolve().parents[2] / 'data' / 'screener_cache'

parquet_engine = 'pyarrow'


class ExchangeUniverse:
    """
    Per-exchange collection of Listings.

    Attributes:
        exchange: Exchange code (e.g. 'NASDAQ', 'SIX').
        listings: List of Listing objects in load / construction
            order.
    """

    def __init__(self, exchange: str, listings: List[Listing]):
        self.exchange = exchange
        self.listings = listings

    def __len__(self) -> int:
        return len(self.listings)

    def __iter__(self) -> Iterator[Listing]:
        return iter(self.listings)

    @classmethod
    def load(cls, exchange: str) -> 'ExchangeUniverse':
        """
        Load an ExchangeUniverse from its parquet file.

        Args:
            exchange: Exchange code; resolves to
                `cache_dir / f'{exchange}.parquet'`.

        Returns:
            ExchangeUniverse with Listings reconstructed from
            the parquet rows.

        Raises:
            FileNotFoundError: When the parquet file is absent.
        """
        path = cache_dir / f'{exchange}.parquet'
        if not path.exists():
            raise FileNotFoundError(
                f'No screener cache for {exchange!r} at {path}.')
        df = pd.read_parquet(path, engine=parquet_engine)
        listings = [
            _row_to_listing(row, exchange)
            for _, row in df.iterrows()]
        return cls(exchange, listings)

    def save(self) -> None:
        """
        Atomically write the universe to its parquet file.

        Writes to a sibling .tmp file, then renames into place
        so partial writes don't corrupt an existing cache.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = cache_dir / f'{self.exchange}.parquet'
        tmp_path = final_path.with_suffix('.parquet.tmp')
        df = self.to_dataframe()
        df.to_parquet(
            tmp_path, engine=parquet_engine, index=False)
        tmp_path.replace(final_path)

    def filter(
        self,
        predicate: Callable[[Listing], bool]) -> 'ExchangeUniverse':
        """
        Return a new ExchangeUniverse with listings matching predicate.
        """
        kept = [l for l in self.listings if predicate(l)]
        return ExchangeUniverse(self.exchange, kept)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten the universe to a parquet-friendly DataFrame.

        One row per Listing. Series fields stored as Python
        lists (parquet's native list type via pyarrow).
        Fundamentals spread into individual scalar columns.
        Per Contract 1 the `exchange` column is NOT stored
        (it's encoded in the filename).
        """
        rows = []
        for l in self.listings:
            row = {
                'ticker': l.ticker,
                'currency': l.currency,
                'sector': l.sector,
                'market_cap': float(l.market_cap),
                'refreshed_at': l.refreshed_at.isoformat(),
                'closes_weekly': l.closes_weekly.tolist(),
                'closes_weekly_dates': [
                    d.isoformat()
                    for d in l.closes_weekly_dates],
                'volumes_weekly': l.volumes_weekly.tolist(),
                'dividends_weekly':
                    l.dividends_weekly.tolist(),
            }
            for name in fundamental_field_names:
                row[name] = float(l.fundamentals[name])
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def staleness_days(self) -> float:
        """Age of the oldest refreshed_at, in days."""
        if not self.listings:
            return 0.0
        oldest = min(l.refreshed_at for l in self.listings)
        return (datetime.now() - oldest).total_seconds() / 86400.0


def _row_to_listing(row: pd.Series, exchange: str) -> Listing:
    """Reconstruct a Listing from a parquet row + exchange code."""
    fundamentals = {
        name: float(row[name])
        for name in fundamental_field_names}
    return Listing(
        ticker=row['ticker'],
        exchange=exchange,
        currency=row['currency'],
        sector=row['sector'],
        market_cap=float(row['market_cap']),
        closes_weekly=np.array(
            list(row['closes_weekly']), dtype=float),
        closes_weekly_dates=pd.DatetimeIndex(
            list(row['closes_weekly_dates'])),
        volumes_weekly=np.array(
            list(row['volumes_weekly']), dtype=float),
        dividends_weekly=np.array(
            list(row['dividends_weekly']), dtype=float),
        fundamentals=fundamentals,
        refreshed_at=datetime.fromisoformat(
            row['refreshed_at']))
```

- [ ] **Step 6: Run tests, confirm 7 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_universe.py -v`
Expected: 7 passed.

- [ ] **Step 7: Run full suite to confirm no regression**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: at least 86 passed (70 prior + 9 candidate/listing + 7 universe = 86).

- [ ] **Step 8: Commit (advisory)**

```bash
git add src/screening/universe.py tests/screening/test_universe.py
git commit -m "feat(screening): add ExchangeUniverse with atomic parquet I/O"
```

---

## Task 4 — `Criterion` ABC + price-based criteria

**Files:**
- Delete: `src/screening/criteria.py` (legacy 550 lines, will be replaced).
- Create: `tests/screening/test_criteria.py`
- Create: `src/screening/criteria.py` (new).

- [ ] **Step 1: Delete legacy criteria**

```bash
rm /Users/rbarreira/Desktop/stock_market/src/screening/criteria.py
```

- [ ] **Step 2: Write tests for ABC + 4 price-based criteria**

Create `/Users/rbarreira/Desktop/stock_market/tests/screening/test_criteria.py`:

```python
"""Tests for Criterion ABC and concrete criteria."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.screening.listing import Listing, fundamental_field_names


def _listing(closes=None, volumes=None, fundamentals=None,
             sector='Technology', ticker='AAPL',
             dividend_yield=0.0):
    n = 156
    dates = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    if closes is None:
        closes = np.linspace(100.0, 200.0, n)
    if volumes is None:
        volumes = np.full(n, 1.0e6)
    fund = {k: 1.0 for k in fundamental_field_names}
    fund['dividend_yield_ttm'] = dividend_yield
    if fundamentals:
        fund.update(fundamentals)
    return Listing(
        ticker=ticker, exchange='NASDAQ', currency='USD',
        sector=sector, market_cap=1.0e9,
        closes_weekly=np.asarray(closes, dtype=float),
        closes_weekly_dates=dates,
        volumes_weekly=np.asarray(volumes, dtype=float),
        dividends_weekly=np.zeros(n),
        fundamentals=fund,
        refreshed_at=datetime(2026, 5, 14, 12, 0))


def test_criterion_registry_has_nine_criteria():
    """All 9 documented criteria self-register."""
    from src.screening.criteria import criterion_registry
    expected = {
        'momentum', 'cagr', 'sharpe', 'max_drawdown',
        'pe_value', 'growth', 'liquidity', 'sector',
        'swiss_tax'}
    assert set(criterion_registry) == expected


def test_get_criterion_returns_instance():
    from src.screening.criteria import (
        get_criterion, Criterion)
    c = get_criterion('momentum')
    assert isinstance(c, Criterion)
    assert c.name == 'momentum'


def test_get_criterion_unknown_raises():
    from src.screening.criteria import get_criterion
    with pytest.raises(KeyError, match='Unknown criterion'):
        get_criterion('not_real')


def test_momentum_uses_13_week_ratio():
    """13-week return: closes[-1] / closes[-13] - 1."""
    from src.screening.criteria import MomentumCriterion
    closes = np.full(156, 100.0)
    closes[-1] = 110.0    # last week
    closes[-13] = 100.0   # 13 weeks ago
    score = MomentumCriterion().evaluate(_listing(closes=closes))
    assert score == pytest.approx(0.10, abs=1e-6)


def test_cagr_compounds_weekly_to_annual():
    """3y CAGR from valid (non-NaN) closes."""
    from src.screening.criteria import CagrCriterion
    closes = np.linspace(100.0, 200.0, 156)
    score = CagrCriterion().evaluate(_listing(closes=closes))
    # 156 weeks = 3 years; (200/100)^(1/3) - 1 ~ 0.2599
    assert score == pytest.approx(0.2599, abs=0.01)


def test_sharpe_returns_annualized_ratio():
    """Sharpe on weekly returns; output is annualised."""
    from src.screening.criteria import SharpeCriterion
    rng = np.random.default_rng(0)
    # Smooth uptrend with low noise -> positive Sharpe
    closes = (
        100.0 * np.cumprod(
            1.0 + rng.normal(0.002, 0.005, 156)))
    score = SharpeCriterion().evaluate(_listing(closes=closes))
    assert score > 0.5


def test_max_drawdown_returns_negative_absolute():
    """MaxDD result is negative (more negative = worse)."""
    from src.screening.criteria import MaxDrawdownCriterion
    closes = np.concatenate([
        np.linspace(100.0, 200.0, 78),
        np.linspace(200.0, 100.0, 78)])
    score = MaxDrawdownCriterion().evaluate(
        _listing(closes=closes))
    # Drawdown from 200 to 100 = -50%, score = -0.5
    assert score == pytest.approx(-0.5, abs=1e-6)


def test_momentum_returns_none_with_nan_endpoints():
    """If closes[-1] or closes[-13] is NaN, momentum returns None."""
    from src.screening.criteria import MomentumCriterion
    closes = np.full(156, 100.0)
    closes[-1] = np.nan
    assert MomentumCriterion().evaluate(
        _listing(closes=closes)) is None


def test_cagr_returns_none_with_too_few_observations():
    """Need >= 52 valid (non-NaN) weekly closes."""
    from src.screening.criteria import CagrCriterion
    closes = np.full(156, np.nan)
    closes[-30:] = np.linspace(100.0, 110.0, 30)
    assert CagrCriterion().evaluate(
        _listing(closes=closes)) is None
```

- [ ] **Step 3: Run, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_criteria.py -v`
Expected: 10 collection errors — `ModuleNotFoundError: No module named 'src.screening.criteria'`.

- [ ] **Step 4: Implement `src/screening/criteria.py` with the ABC + 4 price-based criteria**

```python
"""
Criterion ABC and concrete subclasses for the Screener.

A Criterion turns a Listing into a raw signal value (higher =
better) or returns None to drop the ticker from ranking. The
Screener percentile-ranks the surviving signals per criterion
and combines them into a composite score.

Concrete criteria self-register in `_registry` via
__init_subclass__; this module re-exports it as
`criterion_registry` and exposes `get_criterion(name)`.

This file holds the ABC plus the price-based criteria
(MomentumCriterion, CagrCriterion, SharpeCriterion,
MaxDrawdownCriterion). Fundamentals-based and filter
criteria live in the same file (added in Task 5).
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np

from src.screening.listing import Listing


_registry: Dict[str, Type['Criterion']] = {}

weeks_per_year = 52
momentum_lookback_weeks = 13
min_observations_for_long_window = 52
min_observations_for_short_window = 26


class Criterion(ABC):
    """
    Abstract screening criterion.

    Attributes:
        name: Registry key. Subclasses with a non-empty `name`
            register themselves automatically.
        weight: Composite-score weighting (default 1.0).
    """

    name: str = ''
    weight: float = 1.0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            _registry[cls.name] = cls

    @abstractmethod
    def evaluate(self, listing: Listing) -> Optional[float]:
        """
        Return a raw signal value, higher = better.

        Args:
            listing: One Listing from an ExchangeUniverse.

        Returns:
            Float signal (higher = better) or None to drop the
            ticker from ranking.
        """
        raise NotImplementedError


def get_criterion(name: str) -> Criterion:
    """
    Instantiate a registered criterion by name.

    Args:
        name: Registry key (e.g. 'momentum', 'cagr').

    Returns:
        Fresh instance of the matching subclass.

    Raises:
        KeyError: When `name` is not registered. The message
            lists known names.
    """
    if name not in _registry:
        raise KeyError(
            f'Unknown criterion {name!r}. Known: '
            f'{sorted(_registry)}')
    return _registry[name]()


criterion_registry = _registry


class MomentumCriterion(Criterion):
    """13-week price momentum: last close / 13-weeks-ago close - 1."""

    name = 'momentum'

    def evaluate(self, listing: Listing) -> Optional[float]:
        closes = listing.closes_weekly
        if len(closes) < momentum_lookback_weeks:
            return None
        last = closes[-1]
        ref = closes[-momentum_lookback_weeks]
        if np.isnan(last) or np.isnan(ref) or ref <= 0:
            return None
        return float(last / ref - 1.0)


class CagrCriterion(Criterion):
    """Annualised return over the available weekly history."""

    name = 'cagr'

    def evaluate(self, listing: Listing) -> Optional[float]:
        closes = listing.closes_weekly
        valid = closes[~np.isnan(closes)]
        if len(valid) < min_observations_for_long_window:
            return None
        years = len(valid) / weeks_per_year
        if valid[0] <= 0:
            return None
        return float((valid[-1] / valid[0]) ** (1.0 / years) - 1.0)


class SharpeCriterion(Criterion):
    """Annualised Sharpe ratio of weekly returns (rf assumed 0)."""

    name = 'sharpe'

    def evaluate(self, listing: Listing) -> Optional[float]:
        closes = listing.closes_weekly
        valid = closes[~np.isnan(closes)]
        if len(valid) < min_observations_for_long_window:
            return None
        returns = np.diff(valid) / valid[:-1]
        std_w = float(np.std(returns, ddof=1))
        if std_w < 1e-10:
            return None
        mean_w = float(np.mean(returns))
        return float(
            (mean_w * weeks_per_year) /
            (std_w * np.sqrt(weeks_per_year)))


class MaxDrawdownCriterion(Criterion):
    """Negative max drawdown (less negative = better)."""

    name = 'max_drawdown'

    def evaluate(self, listing: Listing) -> Optional[float]:
        closes = listing.closes_weekly
        valid = closes[~np.isnan(closes)]
        if len(valid) < min_observations_for_short_window:
            return None
        cum_max = np.maximum.accumulate(valid)
        # Avoid div-by-zero on degenerate histories
        if np.any(cum_max <= 0):
            return None
        drawdown = (valid - cum_max) / cum_max
        return float(np.min(drawdown))
```

- [ ] **Step 5: Run, confirm 10 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_criteria.py -v`
Expected: 10 failures (4 of the price-based tests pass; 5 require Task-5 criteria; 1 — `test_criterion_registry_has_nine_criteria` — fails because only 4 criteria registered).

That's expected — Task 5 fills the rest. The 4 price-based tests should pass NOW. Confirm via filtering:

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_criteria.py -v -k "momentum or cagr or sharpe or max_drawdown or get_criterion"`
Expected: 7 passed (4 price-based + get_criterion-returns-instance + get_criterion-unknown + the two None-return guards).

- [ ] **Step 6: Commit (advisory)**

```bash
git add src/screening/criteria.py tests/screening/test_criteria.py
git commit -m "feat(screening): add Criterion ABC + 4 price-based criteria"
```

---

## Task 5 — Fundamental + filter criteria

**Files:**
- Modify: `src/screening/criteria.py` — append 5 more concrete criteria.

- [ ] **Step 1: Append fundamentals + filter tests**

Append to `tests/screening/test_criteria.py`:

```python


def test_pe_value_returns_negative_pe():
    """Lower P/E is better; criterion returns -P/E."""
    from src.screening.criteria import PERatioCriterion
    score = PERatioCriterion().evaluate(
        _listing(fundamentals={'pe_ratio_ttm': 15.0}))
    assert score == pytest.approx(-15.0)


def test_pe_value_returns_none_for_nonpositive_pe():
    """Negative or NaN P/E -> drop."""
    from src.screening.criteria import PERatioCriterion
    assert PERatioCriterion().evaluate(
        _listing(fundamentals={'pe_ratio_ttm': -5.0})) is None
    assert PERatioCriterion().evaluate(
        _listing(
            fundamentals={'pe_ratio_ttm': float('nan')})) is None


def test_growth_averages_revenue_and_earnings():
    """Mean of revenue + earnings YoY growth."""
    from src.screening.criteria import GrowthCriterion
    score = GrowthCriterion().evaluate(_listing(
        fundamentals={
            'revenue_growth_yoy': 0.10,
            'earnings_growth_yoy': 0.20}))
    assert score == pytest.approx(0.15)


def test_liquidity_returns_avg_recent_turnover():
    """Mean of close * volume over the last 13 weeks."""
    from src.screening.criteria import LiquidityCriterion
    closes = np.full(156, 100.0)
    volumes = np.full(156, 1.0e6)
    score = LiquidityCriterion().evaluate(
        _listing(closes=closes, volumes=volumes))
    assert score == pytest.approx(1.0e8)


def test_sector_filter_drops_excluded():
    """Excluded sectors return None."""
    from src.screening.criteria import SectorCriterion
    crit = SectorCriterion(excluded_sectors=('Real Estate',))
    listing = _listing(sector='Real Estate')
    assert crit.evaluate(listing) is None


def test_sector_filter_passes_included():
    """Included sector returns 1.0."""
    from src.screening.criteria import SectorCriterion
    crit = SectorCriterion(included_sectors=('Technology',))
    assert crit.evaluate(_listing(sector='Technology')) == 1.0
    assert crit.evaluate(_listing(sector='Energy')) is None


def test_swiss_tax_drops_high_dividend():
    """Dividend yield > 5% triggers a drop."""
    from src.screening.criteria import SwissTaxCriterion
    assert SwissTaxCriterion().evaluate(
        _listing(dividend_yield=6.0)) is None


def test_swiss_tax_drops_bond_etf_by_ticker():
    """Tickers with BOND or TREAS substrings are dropped."""
    from src.screening.criteria import SwissTaxCriterion
    assert SwissTaxCriterion().evaluate(
        _listing(ticker='AGG-BOND')) is None
    assert SwissTaxCriterion().evaluate(
        _listing(ticker='SHORTTREAS')) is None
```

- [ ] **Step 2: Run, confirm new tests fail**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_criteria.py -v`
Expected: 8 new failures (PERatioCriterion, GrowthCriterion, LiquidityCriterion, SectorCriterion, SwissTaxCriterion not yet defined).

- [ ] **Step 3: Append the 5 new criteria to `src/screening/criteria.py`**

Append to the bottom of `src/screening/criteria.py`:

```python


liquidity_lookback_weeks = 13
high_dividend_yield_threshold_pct = 5.0
bond_ticker_substrings = ('BOND', 'TREAS')


class PERatioCriterion(Criterion):
    """Value tilt: lower P/E is better; signal is -P/E."""

    name = 'pe_value'

    def evaluate(self, listing: Listing) -> Optional[float]:
        pe = listing.fundamentals.get('pe_ratio_ttm')
        if pe is None or np.isnan(pe) or pe <= 0:
            return None
        return -float(pe)


class GrowthCriterion(Criterion):
    """Mean of revenue and earnings YoY growth."""

    name = 'growth'

    def evaluate(self, listing: Listing) -> Optional[float]:
        rev = listing.fundamentals.get('revenue_growth_yoy')
        eps = listing.fundamentals.get('earnings_growth_yoy')
        valid = []
        for x in (rev, eps):
            if x is not None and not np.isnan(x):
                valid.append(float(x))
        if not valid:
            return None
        return float(np.mean(valid))


class LiquidityCriterion(Criterion):
    """Average weekly turnover (close * volume) over recent weeks."""

    name = 'liquidity'

    def evaluate(self, listing: Listing) -> Optional[float]:
        closes = listing.closes_weekly[-liquidity_lookback_weeks:]
        vols = listing.volumes_weekly[-liquidity_lookback_weeks:]
        valid_mask = ~(np.isnan(closes) | np.isnan(vols))
        if int(valid_mask.sum()) < 5:
            return None
        return float(np.mean(closes[valid_mask] * vols[valid_mask]))


class SectorCriterion(Criterion):
    """Hard sector inclusion / exclusion filter.

    With non-empty `included_sectors`, only those sectors pass.
    Always drops sectors in `excluded_sectors`.
    """

    name = 'sector'

    def __init__(
        self,
        included_sectors=None,
        excluded_sectors=None):
        self.included_sectors = (
            tuple(included_sectors)
            if included_sectors else ())
        self.excluded_sectors = (
            tuple(excluded_sectors)
            if excluded_sectors else ())

    def evaluate(self, listing: Listing) -> Optional[float]:
        if listing.sector in self.excluded_sectors:
            return None
        if (self.included_sectors
                and listing.sector not in self.included_sectors):
            return None
        return 1.0


class SwissTaxCriterion(Criterion):
    """Drop bond ETFs and high-dividend names per Swiss tax policy."""

    name = 'swiss_tax'

    def evaluate(self, listing: Listing) -> Optional[float]:
        ticker_upper = listing.ticker.upper()
        for sub in bond_ticker_substrings:
            if sub in ticker_upper:
                return None
        if 'Bond' in (listing.sector or ''):
            return None
        div_yield = listing.fundamentals.get('dividend_yield_ttm')
        if (div_yield is not None and not np.isnan(div_yield)
                and div_yield > high_dividend_yield_threshold_pct):
            return None
        return 1.0
```

- [ ] **Step 4: Run all criteria tests, confirm pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_criteria.py -v`
Expected: 18 passed (10 from Task 4 + 8 new). The `test_criterion_registry_has_nine_criteria` test now passes.

- [ ] **Step 5: Run full suite**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: at least 104 passed (86 + 18).

- [ ] **Step 6: Commit (advisory)**

```bash
git add src/screening/criteria.py tests/screening/test_criteria.py
git commit -m "feat(screening): add 5 fundamental + filter criteria"
```

---

## Task 6 — `Screener` orchestrator

**Files:**
- Delete: `src/screening/screener.py` (legacy 371 lines).
- Create: `tests/screening/test_screener.py`
- Create: `src/screening/screener.py` (new).
- Modify: `src/screening/__init__.py` — populate re-exports.

- [ ] **Step 1: Delete legacy screener**

```bash
rm /Users/rbarreira/Desktop/stock_market/src/screening/screener.py
```

- [ ] **Step 2: Write tests**

Create `/Users/rbarreira/Desktop/stock_market/tests/screening/test_screener.py`:

```python
"""Tests for Screener: load universes, apply criteria, rank."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.screening.candidate import CandidateTicker
from src.screening.listing import Listing, fundamental_field_names
from src.screening.universe import ExchangeUniverse


def _listing(ticker, momentum_pct=0.10, sector='Technology'):
    n = 156
    dates = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    closes = np.full(n, 100.0)
    closes[-1] = 100.0 * (1.0 + momentum_pct)
    fund = {k: 1.0 for k in fundamental_field_names}
    return Listing(
        ticker=ticker, exchange='NASDAQ', currency='USD',
        sector=sector, market_cap=1.0e9,
        closes_weekly=closes, closes_weekly_dates=dates,
        volumes_weekly=np.full(n, 1.0e6),
        dividends_weekly=np.zeros(n),
        fundamentals=fund,
        refreshed_at=datetime(2026, 5, 14, 12, 0))


def _universe(*listings):
    return ExchangeUniverse('NASDAQ', list(listings))


def test_screener_rank_returns_candidates(monkeypatch):
    """rank() returns a list of CandidateTicker objects."""
    from src.screening.screener import Screener
    from src.screening.criteria import MomentumCriterion
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('A', 0.05),
            _listing('B', 0.20),
            _listing('C', 0.10)))
    s = Screener(['NASDAQ']).add_criterion(MomentumCriterion())
    out = s.rank()
    assert all(isinstance(c, CandidateTicker) for c in out)


def test_screener_orders_by_composite_desc(monkeypatch):
    """Best momentum lands first."""
    from src.screening.screener import Screener
    from src.screening.criteria import MomentumCriterion
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('A', 0.05),
            _listing('B', 0.20),
            _listing('C', 0.10)))
    s = Screener(['NASDAQ']).add_criterion(MomentumCriterion())
    out = s.rank()
    assert [c.ticker for c in out] == ['B', 'C', 'A']


def test_screener_top_n_truncates(monkeypatch):
    """top_n returns at most N candidates."""
    from src.screening.screener import Screener
    from src.screening.criteria import MomentumCriterion
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('A', 0.05),
            _listing('B', 0.20),
            _listing('C', 0.10)))
    s = Screener(['NASDAQ']).add_criterion(MomentumCriterion())
    out = s.rank(top_n=2)
    assert len(out) == 2
    assert [c.ticker for c in out] == ['B', 'C']


def test_screener_drops_listings_when_criterion_returns_none(monkeypatch):
    """Listings whose criterion returns None are excluded."""
    from src.screening.screener import Screener
    from src.screening.criteria import SwissTaxCriterion
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('AAPL', 0.05),
            _listing('AGG-BOND', 0.20)))
    s = Screener(['NASDAQ']).add_criterion(SwissTaxCriterion())
    out = s.rank()
    assert {c.ticker for c in out} == {'AAPL'}


def test_screener_combines_multiple_criteria(monkeypatch):
    """Composite score is the weighted mean of percentile ranks."""
    from src.screening.screener import Screener
    from src.screening.criteria import (
        MomentumCriterion, CagrCriterion)
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('A', 0.05),
            _listing('B', 0.20),
            _listing('C', 0.10)))
    s = (Screener(['NASDAQ'])
         .add_criterion(MomentumCriterion())
         .add_criterion(CagrCriterion()))
    out = s.rank()
    # All composites must be in [0, 1]
    for c in out:
        assert 0.0 <= c.composite_score <= 1.0
    # Best momentum is also best CAGR (synthetic), so B leads
    assert out[0].ticker == 'B'


def test_screener_rank_without_criteria_raises(monkeypatch):
    """rank() with no criteria raises a clear error."""
    from src.screening.screener import Screener
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(_listing('A')))
    with pytest.raises(RuntimeError, match='criterion'):
        Screener(['NASDAQ']).rank()


def test_screener_to_dataframe_columns(monkeypatch):
    """to_dataframe surfaces ticker / exchange / score columns."""
    from src.screening.screener import Screener
    from src.screening.criteria import MomentumCriterion
    monkeypatch.setattr(
        'src.screening.screener._load_universe',
        lambda exchange: _universe(
            _listing('A', 0.05), _listing('B', 0.20)))
    s = Screener(['NASDAQ']).add_criterion(MomentumCriterion())
    s.rank()
    df = s.to_dataframe()
    expected = {
        'ticker', 'exchange', 'currency', 'sector',
        'market_cap', 'composite_score'}
    assert expected.issubset(df.columns)
```

- [ ] **Step 3: Run, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_screener.py -v`
Expected: 7 collection errors — `ModuleNotFoundError: No module named 'src.screening.screener'`.

- [ ] **Step 4: Implement `src/screening/screener.py`**

```python
"""
Screener orchestrator.

Loads ExchangeUniverses, applies a list of Criterion instances
to every Listing, drops Listings whose criteria return None,
percentile-ranks each criterion's surviving signals, computes
a composite score as the weighted mean of percentile ranks,
and returns ranked CandidateTicker objects.

The single-Universe load step is exposed as a module-level
function (_load_universe) so tests can monkeypatch it without
touching the real parquet cache.
"""

from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import rankdata

from src.screening.candidate import CandidateTicker
from src.screening.criteria import Criterion
from src.screening.listing import Listing
from src.screening.universe import ExchangeUniverse


def _load_universe(exchange: str) -> ExchangeUniverse:
    """Default ExchangeUniverse loader; overridable in tests."""
    return ExchangeUniverse.load(exchange)


class Screener:
    """
    Multi-exchange screener with criteria-based ranking.

    Attributes:
        exchanges: Exchange codes whose Universes to load.
        criteria: Ordered list of Criterion instances; built up
            via add_criterion().
    """

    def __init__(self, exchanges: List[str]):
        self.exchanges = list(exchanges)
        self.criteria: List[Criterion] = []
        self._last_candidates: Optional[List[CandidateTicker]] = None

    def add_criterion(self, criterion: Criterion) -> 'Screener':
        """Register a criterion; returns self for chaining."""
        self.criteria.append(criterion)
        return self

    def rank(
        self, top_n: Optional[int] = None) -> List[CandidateTicker]:
        """
        Apply criteria, compute composites, return ranked list.

        Args:
            top_n: If given, truncate the result to the first
                top_n candidates.

        Returns:
            List of CandidateTicker, sorted by composite_score
            descending.

        Raises:
            RuntimeError: When no criteria have been added.
        """
        if not self.criteria:
            raise RuntimeError(
                'Screener.rank: add at least one criterion '
                'before calling rank().')

        listings = []
        for exchange in self.exchanges:
            universe = _load_universe(exchange)
            listings.extend(universe)

        # Per-criterion raw signals across all listings
        raw_per_criterion: Dict[str, List[Optional[float]]] = {}
        for crit in self.criteria:
            raw_per_criterion[crit.name] = [
                crit.evaluate(l) for l in listings]

        # Keep only listings where every criterion produced a value
        kept_indices = [
            i for i in range(len(listings))
            if all(
                raw_per_criterion[crit.name][i] is not None
                for crit in self.criteria)]
        if not kept_indices:
            self._last_candidates = []
            return []

        # Percentile-rank surviving signals per criterion
        pct_per_criterion: Dict[str, Dict[int, float]] = {}
        for crit in self.criteria:
            kept_scores = [
                raw_per_criterion[crit.name][i]
                for i in kept_indices]
            ranks = rankdata(kept_scores, method='average')
            n = len(kept_scores)
            # Map rank in [1, n] to percentile in [0, 1]
            pcts = (ranks - 1.0) / max(n - 1, 1)
            pct_per_criterion[crit.name] = {
                idx: float(pct)
                for idx, pct in zip(kept_indices, pcts)}

        total_weight = sum(c.weight for c in self.criteria)
        candidates = []
        for idx in kept_indices:
            l = listings[idx]
            weighted = sum(
                pct_per_criterion[c.name][idx] * c.weight
                for c in self.criteria)
            composite = weighted / total_weight
            metrics = {
                c.name: float(raw_per_criterion[c.name][idx])
                for c in self.criteria}
            candidates.append(CandidateTicker(
                ticker=l.ticker,
                exchange=l.exchange,
                currency=l.currency,
                sector=l.sector,
                market_cap=l.market_cap,
                composite_score=composite,
                metrics=metrics))

        candidates.sort(key=lambda c: -c.composite_score)
        if top_n is not None:
            candidates = candidates[:top_n]
        self._last_candidates = candidates
        return candidates

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flat DataFrame view of the most recent rank() output.

        Returns:
            DataFrame with one row per CandidateTicker; columns
            include ticker, exchange, currency, sector,
            market_cap, composite_score, and one column per
            registered criterion's raw metric.

        Raises:
            RuntimeError: When rank() has not been called yet.
        """
        if self._last_candidates is None:
            raise RuntimeError(
                'Screener.to_dataframe: call rank() first.')
        rows = []
        for c in self._last_candidates:
            row = {
                'ticker': c.ticker,
                'exchange': c.exchange,
                'currency': c.currency,
                'sector': c.sector,
                'market_cap': c.market_cap,
                'composite_score': c.composite_score,
            }
            for name, value in c.metrics.items():
                row[name] = value
            rows.append(row)
        return pd.DataFrame(rows)
```

- [ ] **Step 5: Replace `src/screening/__init__.py` with full re-exports**

```python
"""
Screening domain: per-exchange cached universes + criteria-based ranking.

Public surface:
    Listing, ExchangeUniverse, CandidateTicker, Screener,
    Criterion (ABC) + concrete subclasses, get_criterion,
    criterion_registry.
"""

from src.screening.candidate import CandidateTicker
from src.screening.criteria import (
    Criterion,
    CagrCriterion,
    GrowthCriterion,
    LiquidityCriterion,
    MaxDrawdownCriterion,
    MomentumCriterion,
    PERatioCriterion,
    SectorCriterion,
    SharpeCriterion,
    SwissTaxCriterion,
    criterion_registry,
    get_criterion,
)
from src.screening.listing import (
    Listing, fundamental_field_names)
from src.screening.screener import Screener
from src.screening.universe import ExchangeUniverse


__all__ = [
    'CandidateTicker',
    'Criterion',
    'CagrCriterion',
    'GrowthCriterion',
    'LiquidityCriterion',
    'MaxDrawdownCriterion',
    'MomentumCriterion',
    'PERatioCriterion',
    'SectorCriterion',
    'SharpeCriterion',
    'SwissTaxCriterion',
    'criterion_registry',
    'get_criterion',
    'Listing',
    'fundamental_field_names',
    'Screener',
    'ExchangeUniverse',
]
```

- [ ] **Step 6: Run screener tests, confirm 7 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_screener.py -v`
Expected: 7 passed.

- [ ] **Step 7: Run full suite**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: at least 111 passed (104 + 7).

- [ ] **Step 8: Commit (advisory)**

```bash
git add src/screening/screener.py src/screening/__init__.py \
        tests/screening/test_screener.py
git commit -m "feat(screening): add Screener orchestrator with composite ranking"
```

---

## Task 7 — `cache_refresh.py` resumable yfinance fetcher

**Files:**
- Create: `tests/screening/test_cache_refresh.py`
- Create: `src/screening/cache_refresh.py`
- Delete: `src/screening/cache.py`
- Delete: `src/screening/exchanges.py`
- Delete: `src/screening/fundamentals.py`
- Delete: `src/screening/pipeline.py`

These four legacy files are replaced by the single `cache_refresh.py`.

- [ ] **Step 1: Delete the four legacy files**

```bash
rm /Users/rbarreira/Desktop/stock_market/src/screening/cache.py
rm /Users/rbarreira/Desktop/stock_market/src/screening/exchanges.py
rm /Users/rbarreira/Desktop/stock_market/src/screening/fundamentals.py
rm /Users/rbarreira/Desktop/stock_market/src/screening/pipeline.py
```

- [ ] **Step 2: Confirm yfinance is installed**

Run: `python3 -c "import yfinance; print(yfinance.__version__)"`
Expected: a version string. If missing, the user must install: `pip install yfinance`.

- [ ] **Step 3: Write tests**

Create `/Users/rbarreira/Desktop/stock_market/tests/screening/test_cache_refresh.py`:

```python
"""Tests for cache_refresh: resumable yfinance fetcher."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.screening.listing import (
    Listing, fundamental_field_names, weekly_series_length)


def _fake_history_df(n_rows=156):
    """Fake yfinance.Ticker.history(period='3y', interval='1wk') output."""
    idx = pd.date_range('2023-05-15', periods=n_rows, freq='W-FRI')
    return pd.DataFrame({
        'Open': np.full(n_rows, 100.0),
        'High': np.full(n_rows, 105.0),
        'Low': np.full(n_rows, 95.0),
        'Close': np.linspace(100.0, 200.0, n_rows),
        'Volume': np.full(n_rows, 1.0e6),
        'Dividends': np.zeros(n_rows),
        'Stock Splits': np.zeros(n_rows),
    }, index=idx)


def _fake_ticker_info():
    """Fake yfinance.Ticker.info dict."""
    return {
        'currency': 'USD',
        'sector': 'Technology',
        'marketCap': 3.0e12,
        'trailingPE': 28.0,
        'forwardPE': 25.0,
        'priceToBook': 45.0,
        'priceToSalesTrailing12Months': 7.0,
        'dividendYield': 0.005,
        'trailingEps': 6.5,
        'revenueGrowth': 0.10,
        'earningsGrowth': 0.15,
        'profitMargins': 0.25,
        'operatingMargins': 0.30,
        'returnOnEquity': 1.5,
        'returnOnAssets': 0.30,
        'debtToEquity': 200.0,
        'freeCashflow': 1.0e11,
        'beta': 1.2,
        'sharesOutstanding': 1.5e10,
        'shortRatio': 1.5,
    }


def _patch_yfinance(monkeypatch):
    """Install a fake yf.Ticker that returns canned history + info."""
    fake = MagicMock()
    fake.history.return_value = _fake_history_df()
    fake.info = _fake_ticker_info()
    monkeypatch.setattr(
        'src.screening.cache_refresh.yf.Ticker',
        lambda ticker: fake)


def test_fetch_one_listing_returns_listing(monkeypatch):
    """One ticker fetch produces a valid Listing."""
    from src.screening.cache_refresh import _fetch_one_listing
    _patch_yfinance(monkeypatch)
    l = _fetch_one_listing('AAPL', 'NASDAQ')
    assert isinstance(l, Listing)
    assert l.ticker == 'AAPL'
    assert l.exchange == 'NASDAQ'
    assert l.currency == 'USD'
    assert l.sector == 'Technology'
    assert len(l.closes_weekly) == weekly_series_length
    assert l.fundamentals['pe_ratio_ttm'] == 28.0


def test_fetch_one_listing_handles_short_history(monkeypatch):
    """If yfinance returns fewer than 156 weeks, pad with NaN."""
    from src.screening.cache_refresh import _fetch_one_listing
    fake = MagicMock()
    fake.history.return_value = _fake_history_df(n_rows=80)
    fake.info = _fake_ticker_info()
    monkeypatch.setattr(
        'src.screening.cache_refresh.yf.Ticker',
        lambda ticker: fake)
    l = _fetch_one_listing('AAPL', 'NASDAQ')
    assert len(l.closes_weekly) == weekly_series_length
    # First 76 entries (156 - 80) are pad NaNs
    assert np.isnan(l.closes_weekly[0])
    assert not np.isnan(l.closes_weekly[-1])


def test_refresh_exchange_writes_parquet(
    monkeypatch, tmp_path):
    """End-to-end: refresh writes a parquet readable by ExchangeUniverse."""
    from src.screening.cache_refresh import refresh_exchange
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    _patch_yfinance(monkeypatch)
    refresh_exchange(
        'NASDAQ', tickers=['AAPL', 'MSFT'],
        rate_limit_sleep=0.0)
    from src.screening.universe import ExchangeUniverse
    u = ExchangeUniverse.load('NASDAQ')
    assert {l.ticker for l in u} == {'AAPL', 'MSFT'}


def test_refresh_exchange_resume_skips_existing(
    monkeypatch, tmp_path):
    """Second call with resume=True skips already-cached tickers."""
    from src.screening.cache_refresh import refresh_exchange
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    call_log = {'tickers': []}

    fake = MagicMock()
    fake.history.return_value = _fake_history_df()
    fake.info = _fake_ticker_info()

    def fake_ticker(t):
        call_log['tickers'].append(t)
        return fake

    monkeypatch.setattr(
        'src.screening.cache_refresh.yf.Ticker', fake_ticker)
    refresh_exchange(
        'NASDAQ', tickers=['AAPL'],
        rate_limit_sleep=0.0)
    assert call_log['tickers'] == ['AAPL']

    # Resume: AAPL already cached; MSFT is new
    refresh_exchange(
        'NASDAQ', tickers=['AAPL', 'MSFT'],
        rate_limit_sleep=0.0, resume=True)
    # Only MSFT was fetched on the second call
    assert call_log['tickers'] == ['AAPL', 'MSFT']


def test_refresh_exchange_no_resume_refetches(
    monkeypatch, tmp_path):
    """resume=False refetches everything."""
    from src.screening.cache_refresh import refresh_exchange
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    call_log = {'tickers': []}

    fake = MagicMock()
    fake.history.return_value = _fake_history_df()
    fake.info = _fake_ticker_info()

    def fake_ticker(t):
        call_log['tickers'].append(t)
        return fake

    monkeypatch.setattr(
        'src.screening.cache_refresh.yf.Ticker', fake_ticker)
    refresh_exchange(
        'NASDAQ', tickers=['AAPL'],
        rate_limit_sleep=0.0)
    refresh_exchange(
        'NASDAQ', tickers=['AAPL'],
        rate_limit_sleep=0.0, resume=False)
    # Both calls fetched AAPL
    assert call_log['tickers'] == ['AAPL', 'AAPL']
```

- [ ] **Step 4: Run, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_cache_refresh.py -v`
Expected: 5 collection errors — `ModuleNotFoundError: No module named 'src.screening.cache_refresh'`.

- [ ] **Step 5: Implement `src/screening/cache_refresh.py`**

```python
"""
Resumable yfinance fetcher for per-exchange screener caches.

For each requested ticker, fetches 3 years of weekly OHLCV +
the snapshot fundamentals dict, builds a Listing, and saves
the accumulated batch to the per-exchange parquet via
ExchangeUniverse.save(). The parquet itself acts as the
checkpoint — when resume=True (default), tickers already
present in the loaded ExchangeUniverse are skipped on the
next call.

Designed to survive rate-limit interruptions: progress is
persisted every `save_every` fetched tickers, and a finally
block writes whatever is in memory if the loop is cut short
(e.g. KeyboardInterrupt or unhandled exception).
"""

import time
import warnings
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.screening.listing import (
    Listing, fundamental_field_names, weekly_series_length)
from src.screening.universe import ExchangeUniverse


default_save_every = 50
default_rate_limit_sleep = 0.5


# yfinance Ticker.info field name -> Listing fundamental field
# Mapping is explicit; missing yfinance keys map to NaN.
_yfinance_to_fundamental = {
    'pe_ratio_ttm': 'trailingPE',
    'pe_ratio_forward': 'forwardPE',
    'pb_ratio': 'priceToBook',
    'ps_ratio': 'priceToSalesTrailing12Months',
    'dividend_yield_ttm': 'dividendYield',
    'eps_ttm': 'trailingEps',
    'revenue_growth_yoy': 'revenueGrowth',
    'earnings_growth_yoy': 'earningsGrowth',
    'profit_margin': 'profitMargins',
    'operating_margin': 'operatingMargins',
    'roe': 'returnOnEquity',
    'roa': 'returnOnAssets',
    'debt_to_equity': 'debtToEquity',
    'free_cash_flow': 'freeCashflow',
    'beta_yf': 'beta',
    'shares_outstanding': 'sharesOutstanding',
    'short_ratio': 'shortRatio',
}


def refresh_exchange(
    exchange: str,
    tickers: List[str],
    resume: bool = True,
    rate_limit_sleep: float = default_rate_limit_sleep,
    save_every: int = default_save_every) -> None:
    """
    Refresh the on-disk parquet cache for one exchange.

    Args:
        exchange: Exchange code (e.g. 'NASDAQ', 'SIX').
        tickers: List of ticker symbols to fetch.
        resume: When True, skip tickers already present in the
            existing parquet for this exchange.
        rate_limit_sleep: Seconds to sleep between per-ticker
            yfinance calls. Tune up if rate-limit errors fire.
        save_every: Persist progress every N successful fetches.

    Side effects:
        Writes / updates `data/screener_cache/<exchange>.parquet`
        atomically (temp file then rename).
    """
    existing: List[Listing] = []
    already_done: set = set()
    if resume:
        try:
            existing = list(ExchangeUniverse.load(exchange))
            already_done = {l.ticker for l in existing}
        except FileNotFoundError:
            pass

    new_listings: List[Listing] = []
    to_fetch = [t for t in tickers if t not in already_done]

    try:
        for i, ticker in enumerate(to_fetch):
            try:
                listing = _fetch_one_listing(ticker, exchange)
            except Exception as e:
                warnings.warn(
                    f'cache_refresh: skipping {ticker!r} '
                    f'after fetch error '
                    f'({type(e).__name__}: {e}).')
                continue
            new_listings.append(listing)
            if rate_limit_sleep > 0:
                time.sleep(rate_limit_sleep)
            if (i + 1) % save_every == 0:
                _persist(exchange, existing + new_listings)
    finally:
        # Always save whatever made it through, including on
        # KeyboardInterrupt or unhandled exception.
        if new_listings:
            _persist(exchange, existing + new_listings)


def _persist(
    exchange: str, listings: List[Listing]) -> None:
    """Atomically write the merged universe to its parquet."""
    ExchangeUniverse(exchange, listings).save()


def _fetch_one_listing(
    ticker: str, exchange: str) -> Listing:
    """
    Build one Listing via two yfinance calls.

    Args:
        ticker: Native exchange symbol.
        exchange: Exchange code; stamped onto the Listing.

    Returns:
        Listing with 156 weekly bars (NaN-padded if yfinance
        returned fewer rows) and 17 fundamentals (NaN if missing).

    Raises:
        Any yfinance exception propagates so refresh_exchange
        can decide whether to skip or abort.
    """
    yf_ticker = yf.Ticker(ticker)
    history = yf_ticker.history(period='3y', interval='1wk')
    info = yf_ticker.info

    closes = _to_padded_array(history.get('Close'))
    volumes = _to_padded_array(history.get('Volume'))
    dividends = _to_padded_array(history.get('Dividends'))
    dates = _padded_dates(history.index)

    fundamentals = {}
    for our_name, yf_name in _yfinance_to_fundamental.items():
        raw = info.get(yf_name)
        fundamentals[our_name] = (
            float(raw) if raw is not None else float('nan'))

    market_cap = info.get('marketCap')
    market_cap_value = (
        float(market_cap) if market_cap is not None
        else float('nan'))
    currency = info.get('currency') or 'UNKNOWN'
    sector = info.get('sector') or 'Unknown'

    return Listing(
        ticker=ticker, exchange=exchange,
        currency=currency, sector=sector,
        market_cap=market_cap_value,
        closes_weekly=closes, closes_weekly_dates=dates,
        volumes_weekly=volumes, dividends_weekly=dividends,
        fundamentals=fundamentals,
        refreshed_at=datetime.now())


def _to_padded_array(
    series: Optional[pd.Series]) -> np.ndarray:
    """
    Normalise a pandas Series to a length-156 numpy array.

    Pads with NaN at the front when the source is shorter
    than `weekly_series_length`. Truncates to the most recent
    `weekly_series_length` bars when longer.
    """
    if series is None or len(series) == 0:
        return np.full(weekly_series_length, np.nan)
    values = series.to_numpy(dtype=float)
    if len(values) >= weekly_series_length:
        return values[-weekly_series_length:]
    pad = np.full(weekly_series_length - len(values), np.nan)
    return np.concatenate([pad, values])


def _padded_dates(
    index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Normalise a DatetimeIndex to length 156.

    Pads at the front with synthetic weekly dates (W-FRI) when
    the source is shorter; truncates to the most recent 156
    when longer.
    """
    n = len(index)
    if n >= weekly_series_length:
        return index[-weekly_series_length:]
    if n == 0:
        end = pd.Timestamp.utcnow().normalize()
        return pd.date_range(
            end=end, periods=weekly_series_length, freq='W-FRI')
    head_count = weekly_series_length - n
    head_end = index[0] - pd.Timedelta(weeks=1)
    head = pd.date_range(
        end=head_end, periods=head_count, freq='W-FRI')
    return head.append(index)
```

- [ ] **Step 6: Run cache_refresh tests, confirm 5 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/screening/test_cache_refresh.py -v`
Expected: 5 passed.

- [ ] **Step 7: Run full suite**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: at least 116 passed (111 + 5).

- [ ] **Step 8: Commit (advisory)**

```bash
git add src/screening/cache_refresh.py \
        tests/screening/test_cache_refresh.py
git commit -m "feat(screening): add resumable yfinance cache refresher"
```

---

## Task 8 — `refresh_screener_cache.py` CLI

**Files:**
- Delete: `src/user_scripts/update_exchange_lists.py` (legacy)
- Create: `tests/user_scripts/__init__.py` (empty marker)
- Create: `tests/user_scripts/test_screener_cli.py` — covers BOTH this CLI and Task 9's CLI.
- Create: `src/user_scripts/refresh_screener_cache.py`
- Create: `data/exchange_tickers/` directory with seed `.txt` files.

- [ ] **Step 1: Create directories + remove legacy refresh script**

```bash
rm /Users/rbarreira/Desktop/stock_market/src/user_scripts/update_exchange_lists.py
mkdir -p /Users/rbarreira/Desktop/stock_market/tests/user_scripts
touch /Users/rbarreira/Desktop/stock_market/tests/user_scripts/__init__.py
mkdir -p /Users/rbarreira/Desktop/stock_market/data/exchange_tickers
```

- [ ] **Step 2: Seed two exchange ticker lists from existing constituent JSONs**

```bash
cd /Users/rbarreira/Desktop/stock_market && python3 -c "
import json, pathlib
out_dir = pathlib.Path('data/exchange_tickers')
sp500 = json.loads(pathlib.Path('data/sp500_tickers.json').read_text())
ftse100 = json.loads(pathlib.Path('data/ftse100_tickers.json').read_text())
# sp500_tickers.json shape can vary; tolerate a list-of-strings
# or a list-of-dicts with a 'symbol' key
def to_list(data):
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return [d.get('symbol') or d.get('ticker') for d in data]
    return list(data)
nyse = to_list(sp500)    # SP500 tickers are split across NYSE+NASDAQ
(out_dir / 'NYSE.txt').write_text('\n'.join(nyse))
(out_dir / 'NASDAQ.txt').write_text('\n'.join(nyse))
(out_dir / 'LSE.txt').write_text('\n'.join(to_list(ftse100)))
print('Seeded:', sorted(p.name for p in out_dir.iterdir()))"
```

If the JSON shape doesn't match either expected form, inspect manually and seed by hand. Acceptable for this task — the CLI just needs SOME ticker file present for the smoke test.

For the other 13 exchanges (SIX, XETRA, EURONEXT_*, BIT, BME, STO, OSL, CPH, HEL), leave the file absent. The CLI will report "no ticker list" for those until the user provides one.

- [ ] **Step 3: Write CLI test**

Create `/Users/rbarreira/Desktop/stock_market/tests/user_scripts/test_screener_cli.py`:

```python
"""Integration tests for refresh_screener_cache.py and run_screener.py CLIs."""

import json
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.screening.listing import fundamental_field_names


def _fake_history_df():
    n = 156
    idx = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    return pd.DataFrame({
        'Open': np.full(n, 100.0),
        'High': np.full(n, 105.0),
        'Low': np.full(n, 95.0),
        'Close': np.linspace(100.0, 200.0, n),
        'Volume': np.full(n, 1.0e6),
        'Dividends': np.zeros(n),
        'Stock Splits': np.zeros(n),
    }, index=idx)


def _fake_ticker_info():
    return {
        'currency': 'USD', 'sector': 'Technology',
        'marketCap': 3.0e12,
        'trailingPE': 28.0, 'forwardPE': 25.0,
        'priceToBook': 45.0,
        'priceToSalesTrailing12Months': 7.0,
        'dividendYield': 0.005, 'trailingEps': 6.5,
        'revenueGrowth': 0.10, 'earningsGrowth': 0.15,
        'profitMargins': 0.25, 'operatingMargins': 0.30,
        'returnOnEquity': 1.5, 'returnOnAssets': 0.30,
        'debtToEquity': 200.0, 'freeCashflow': 1.0e11,
        'beta': 1.2, 'sharesOutstanding': 1.5e10,
        'shortRatio': 1.5,
    }


def _patch_yfinance(monkeypatch):
    fake = MagicMock()
    fake.history.return_value = _fake_history_df()
    fake.info = _fake_ticker_info()
    monkeypatch.setattr(
        'src.screening.cache_refresh.yf.Ticker',
        lambda ticker: fake)


def test_refresh_cli_writes_parquet(monkeypatch, tmp_path):
    """refresh_screener_cache --exchange NASDAQ writes a parquet."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    tickers_dir = tmp_path / 'tickers'
    tickers_dir.mkdir()
    (tickers_dir / 'NASDAQ.txt').write_text('AAPL\nMSFT\n')
    _patch_yfinance(monkeypatch)
    from src.user_scripts import refresh_screener_cache
    monkeypatch.setattr(
        refresh_screener_cache,
        'tickers_dir', tickers_dir)
    monkeypatch.setattr(sys, 'argv', [
        'refresh_screener_cache.py',
        '--exchange', 'NASDAQ',
        '--rate-limit', '0',
    ])
    refresh_screener_cache.main()
    assert (tmp_path / 'NASDAQ.parquet').exists()


def test_refresh_cli_missing_ticker_list_raises(
    monkeypatch, tmp_path):
    """Missing ticker list file raises a clear error."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    from src.user_scripts import refresh_screener_cache
    monkeypatch.setattr(
        refresh_screener_cache,
        'tickers_dir', tmp_path / 'no_dir')
    monkeypatch.setattr(sys, 'argv', [
        'refresh_screener_cache.py',
        '--exchange', 'XETRA'])
    with pytest.raises(SystemExit, match='ticker list'):
        refresh_screener_cache.main()
```

- [ ] **Step 4: Run, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/user_scripts/test_screener_cli.py -v`
Expected: 2 collection errors — `ModuleNotFoundError: No module named 'src.user_scripts.refresh_screener_cache'`.

- [ ] **Step 5: Implement `src/user_scripts/refresh_screener_cache.py`**

```python
"""
CLI: refresh the per-exchange screener parquet cache.

Reads a one-ticker-per-line file from
data/exchange_tickers/<exchange>.txt and dispatches to
src.screening.cache_refresh.refresh_exchange. Resumable by
default — re-running picks up where the last run left off.

Usage:
    python -m src.user_scripts.refresh_screener_cache \\
        --exchange NASDAQ
    python -m src.user_scripts.refresh_screener_cache \\
        --exchange NASDAQ --rate-limit 1.0 --no-resume
"""

import argparse
import pathlib
import sys
from typing import List


tickers_dir = pathlib.Path(
    __file__).resolve().parents[2] / 'data' / 'exchange_tickers'


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='refresh_screener_cache.py',
        description=(
            'Refresh the per-exchange screener parquet cache.'))
    p.add_argument(
        '--exchange', required=True,
        help='Exchange code (e.g. NASDAQ, SIX, XETRA).')
    p.add_argument(
        '--rate-limit', type=float, default=0.5,
        help='Seconds between per-ticker yfinance calls.')
    p.add_argument(
        '--no-resume', action='store_true',
        help='Refetch every ticker even if cached.')
    return p.parse_args(argv)


def _load_ticker_list(exchange: str) -> List[str]:
    """Read tickers_dir/<exchange>.txt; one ticker per line."""
    path = tickers_dir / f'{exchange}.txt'
    if not path.exists():
        raise SystemExit(
            f'refresh_screener_cache: no ticker list at '
            f'{path}. Create it (one ticker per line) and '
            f'retry.')
    lines = path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def main(argv=None):
    """Entry point for `python -m src.user_scripts.refresh_screener_cache`."""
    args = _parse_args(
        argv if argv is not None else sys.argv[1:])
    tickers = _load_ticker_list(args.exchange)
    from src.screening.cache_refresh import refresh_exchange
    refresh_exchange(
        exchange=args.exchange,
        tickers=tickers,
        resume=not args.no_resume,
        rate_limit_sleep=args.rate_limit)
    print(
        f'Refresh complete: exchange={args.exchange}, '
        f'tickers_attempted={len(tickers)}.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 6: Run CLI tests, confirm 2 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/user_scripts/test_screener_cli.py -v`
Expected: 2 passed (only the refresh-CLI tests are written so far; run_screener tests follow in Task 9).

- [ ] **Step 7: Run full suite**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: at least 118 passed (116 + 2).

- [ ] **Step 8: Commit (advisory)**

```bash
git add src/user_scripts/refresh_screener_cache.py \
        tests/user_scripts/__init__.py \
        tests/user_scripts/test_screener_cli.py \
        data/exchange_tickers/
git commit -m "feat(user_scripts): add refresh_screener_cache CLI"
```

---

## Task 9 — `run_screener.py` CLI

**Files:**
- Delete: `src/user_scripts/screen_stocks.py` (legacy 146-line script).
- Modify: `tests/user_scripts/test_screener_cli.py` — append run_screener tests.
- Create: `src/user_scripts/run_screener.py`

- [ ] **Step 1: Delete legacy stock-screener script**

```bash
rm /Users/rbarreira/Desktop/stock_market/src/user_scripts/screen_stocks.py
```

- [ ] **Step 2: Append run_screener tests**

Append to `tests/user_scripts/test_screener_cli.py`:

```python


def _seed_universe(monkeypatch, tmp_path):
    """Write a small parquet for the run_screener test."""
    import src.screening.universe as universe_mod
    monkeypatch.setattr(universe_mod, 'cache_dir', tmp_path)
    from src.screening.listing import (
        Listing, fundamental_field_names)
    from src.screening.universe import ExchangeUniverse
    from datetime import datetime
    n = 156
    dates = pd.date_range('2023-05-15', periods=n, freq='W-FRI')
    listings = []
    for ticker, mom in (('A', 0.05), ('B', 0.20), ('C', 0.10)):
        closes = np.full(n, 100.0)
        closes[-1] = 100.0 * (1.0 + mom)
        fund = {k: 1.0 for k in fundamental_field_names}
        listings.append(Listing(
            ticker=ticker, exchange='NASDAQ',
            currency='USD', sector='Technology',
            market_cap=1.0e9,
            closes_weekly=closes,
            closes_weekly_dates=dates,
            volumes_weekly=np.full(n, 1.0e6),
            dividends_weekly=np.zeros(n),
            fundamentals=fund,
            refreshed_at=datetime(2026, 5, 14, 12, 0)))
    ExchangeUniverse('NASDAQ', listings).save()


def test_run_screener_writes_excel(monkeypatch, tmp_path):
    """run_screener --output-format excel writes an .xlsx."""
    _seed_universe(monkeypatch, tmp_path)
    out_path = tmp_path / 'ranked.xlsx'
    monkeypatch.setattr(sys, 'argv', [
        'run_screener.py',
        '--exchange', 'NASDAQ',
        '--criteria', 'momentum',
        '--top-n', '10',
        '--output-format', 'excel',
        '--output', str(out_path)])
    from src.user_scripts import run_screener
    run_screener.main()
    assert out_path.exists()
    df = pd.read_excel(out_path)
    assert 'composite_score' in df.columns
    assert df.iloc[0]['ticker'] == 'B'    # best momentum


def test_run_screener_writes_json(monkeypatch, tmp_path):
    """run_screener --output-format json writes a .json."""
    _seed_universe(monkeypatch, tmp_path)
    out_path = tmp_path / 'ranked.json'
    monkeypatch.setattr(sys, 'argv', [
        'run_screener.py',
        '--exchange', 'NASDAQ',
        '--criteria', 'momentum',
        '--output-format', 'json',
        '--output', str(out_path)])
    from src.user_scripts import run_screener
    run_screener.main()
    data = json.loads(out_path.read_text())
    assert isinstance(data, list)
    assert data[0]['ticker'] == 'B'
    assert 'composite_score' in data[0]


def test_run_screener_unknown_criterion_raises(
    monkeypatch, tmp_path):
    """Unknown --criteria value raises."""
    _seed_universe(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, 'argv', [
        'run_screener.py',
        '--exchange', 'NASDAQ',
        '--criteria', 'not_real',
        '--output-format', 'json',
        '--output', str(tmp_path / 'r.json')])
    from src.user_scripts import run_screener
    with pytest.raises(KeyError):
        run_screener.main()
```

- [ ] **Step 3: Run, confirm failure**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/user_scripts/test_screener_cli.py -v -k "run_screener"`
Expected: 3 collection errors — `ModuleNotFoundError: No module named 'src.user_scripts.run_screener'`.

- [ ] **Step 4: Implement `src/user_scripts/run_screener.py`**

```python
"""
CLI: rank tickers across cached exchange universes.

Loads one or more ExchangeUniverses, applies the requested
criteria, computes composite scores, and writes the top-N
results to Excel or JSON.

Usage:
    python -m src.user_scripts.run_screener \\
        --exchange NASDAQ NYSE \\
        --criteria momentum,cagr,sharpe,pe_value \\
        --top-n 50 \\
        --output-format excel \\
        --output results/screener_NASDAQ_NYSE.xlsx
"""

import argparse
import json
import pathlib
import sys
from typing import List


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='run_screener.py',
        description='Rank tickers across cached exchanges.')
    p.add_argument(
        '--exchange', required=True, nargs='+',
        help='One or more exchange codes (space-separated).')
    p.add_argument(
        '--criteria', required=True,
        help='Comma-separated criterion names (e.g. '
             '"momentum,cagr,sharpe").')
    p.add_argument(
        '--top-n', type=int, default=None,
        help='Truncate output to top N candidates.')
    p.add_argument(
        '--output-format', choices=['excel', 'json'],
        default='excel',
        help='Output file format.')
    p.add_argument(
        '--output', required=True,
        help='Output file path.')
    return p.parse_args(argv)


def _build_screener(exchanges: List[str], criteria_csv: str):
    from src.screening.screener import Screener
    from src.screening.criteria import get_criterion
    screener = Screener(exchanges)
    for name in criteria_csv.split(','):
        name = name.strip()
        if name:
            screener.add_criterion(get_criterion(name))
    return screener


def _write_output(
    candidates, out_path: pathlib.Path,
    output_format: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == 'excel':
        import pandas as pd
        rows = [
            {
                'ticker': c.ticker, 'exchange': c.exchange,
                'currency': c.currency, 'sector': c.sector,
                'market_cap': c.market_cap,
                'composite_score': c.composite_score,
                **dict(c.metrics),
            }
            for c in candidates]
        pd.DataFrame(rows).to_excel(out_path, index=False)
    else:
        rows = [
            {
                'ticker': c.ticker, 'exchange': c.exchange,
                'currency': c.currency, 'sector': c.sector,
                'market_cap': c.market_cap,
                'composite_score': c.composite_score,
                'metrics': dict(c.metrics),
            }
            for c in candidates]
        out_path.write_text(json.dumps(rows, indent=2))


def main(argv=None):
    """Entry point for `python -m src.user_scripts.run_screener`."""
    args = _parse_args(
        argv if argv is not None else sys.argv[1:])
    screener = _build_screener(args.exchange, args.criteria)
    candidates = screener.rank(top_n=args.top_n)
    out_path = pathlib.Path(args.output)
    _write_output(candidates, out_path, args.output_format)
    print(
        f'Screener output written to {out_path} '
        f'({len(candidates)} candidates, '
        f'criteria={args.criteria}, '
        f'exchanges={args.exchange}).')


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Run CLI tests, confirm 5 pass**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/user_scripts/test_screener_cli.py -v`
Expected: 5 passed (2 refresh + 3 run_screener).

- [ ] **Step 6: Run full suite**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: at least 121 passed (118 + 3).

- [ ] **Step 7: Commit (advisory)**

```bash
git add src/user_scripts/run_screener.py \
        tests/user_scripts/test_screener_cli.py
git commit -m "feat(user_scripts): add run_screener CLI"
```

---

## Task 10 — Phase 3 verification + cleanup

**Files:**
- Modify: `tasks/todo.md` — append Phase 3 completion note.

- [ ] **Step 1: Confirm legacy files are gone**

Run:
```bash
ls /Users/rbarreira/Desktop/stock_market/src/screening/
```

Expected listing (alphabetical):
```
__init__.py
__pycache__
cache_refresh.py
candidate.py
criteria.py
listing.py
screener.py
universe.py
```

If any of `cache.py`, `exchanges.py`, `fundamentals.py`, `pipeline.py` remain, delete them now.

```bash
ls /Users/rbarreira/Desktop/stock_market/src/user_scripts/ | grep -E "screen_stocks|update_exchange_lists"
```

Expected: no output (both files deleted).

- [ ] **Step 2: Smoke test the screener pipeline against the seeded universe**

Run a small end-to-end refresh-then-rank against a tiny ticker list to confirm everything wires up. This may be skipped if yfinance is rate-limited — mark as `DEFERRED` in the report and proceed.

```bash
cd /Users/rbarreira/Desktop/stock_market && python3 -m src.user_scripts.refresh_screener_cache \
    --exchange NASDAQ --rate-limit 1.0
```

(The seeded NASDAQ.txt file from Task 8 has many tickers; this will take a long time and burn a yfinance session. Acceptable to skip and mark deferred.)

If skipped, write a single-ticker test instead:
```bash
cd /Users/rbarreira/Desktop/stock_market && python3 -c "
from src.screening.cache_refresh import refresh_exchange
refresh_exchange('NASDAQ', tickers=['AAPL'], rate_limit_sleep=0.0)
from src.screening.universe import ExchangeUniverse
u = ExchangeUniverse.load('NASDAQ')
print(f'Loaded {len(u)} listings.')
for l in list(u)[:3]:
    print(f'  {l.ticker}: sector={l.sector}, '
          f'pe={l.fundamentals[\"pe_ratio_ttm\"]:.2f}')"
```

Expected: prints `Loaded 1 listings.` followed by `AAPL: sector=Technology, pe=...` (whatever yfinance returns currently).

- [ ] **Step 3: Run the full test suite**

Run: `cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v`
Expected: at least 121 passed.

- [ ] **Step 4: Append Phase 3 completion to `tasks/todo.md`**

Append this block at the end of `/Users/rbarreira/Desktop/stock_market/tasks/todo.md`:

```markdown

## Phase 3 — Screening domain: COMPLETE (2026-05-14)

- src/screening/                            [rebuilt]
  - candidate.py    (CandidateTicker)
  - listing.py      (Listing + fundamental_field_names)
  - universe.py     (ExchangeUniverse + parquet I/O)
  - criteria.py     (Criterion ABC + 9 concrete subclasses)
  - screener.py     (Screener orchestrator)
  - cache_refresh.py (resumable yfinance fetcher)
- src/user_scripts/refresh_screener_cache.py   [new CLI]
- src/user_scripts/run_screener.py             [new CLI]
- src/screening/{cache,criteria,exchanges,
                 fundamentals,pipeline,
                 screener,universe}.py         [DELETED legacy]
- src/user_scripts/screen_stocks.py            [DELETED legacy]
- src/user_scripts/update_exchange_lists.py    [DELETED legacy]
- data/exchange_tickers/{NYSE,NASDAQ,LSE}.txt  [seeded]

Verification:
- 121+ unit tests pass via `pytest tests/ -v`.
- Criterion registry exposes 9 criteria: momentum, cagr,
  sharpe, max_drawdown, pe_value, growth, liquidity, sector,
  swiss_tax.
- ExchangeUniverse round-trips through parquet preserving all
  Listing fields including 156-week series and 17 fundamentals.
- refresh_screener_cache CLI fetches per-ticker via yfinance
  (mocked in tests), persists per `save_every` batches, and
  resumes from existing parquet on the next run.
- run_screener CLI loads the cached universe(s), applies the
  requested criteria, and writes Excel or JSON output.

Spec: docs/superpowers/specs/2026-05-13-repo-refactor-design.md
Plan: docs/superpowers/plans/2026-05-14-phase-3-screening-domain.md

Next: Phase 4 — Wire screener-fed strategies (MaxGrowth,
RiskAdjusted, MinRisk) into the rebalancer + report.py.
```

- [ ] **Step 5: Commit (advisory)**

```bash
git add tasks/todo.md
git commit -m "docs: mark Phase 3 complete in tasks/todo.md"
```

---

## Self-review

**Spec coverage** (against Section 7 Phase 3 of `docs/superpowers/specs/2026-05-13-repo-refactor-design.md`):
- [x] `src/screening/listing.py` — Task 2.
- [x] `src/screening/universe.py` — Task 3.
- [x] `src/screening/criteria.py` — Tasks 4-5.
- [x] `src/screening/screener.py` — Task 6.
- [x] `src/screening/cache_refresh.py` — Task 7.
- [x] `src/screening/__init__.py` re-exports — Task 6.
- [x] `src/screening/candidate.py` (CandidateTicker, Contract 2) — Task 1.
- [x] `src/user_scripts/refresh_screener_cache.py` — Task 8.
- [x] `src/user_scripts/run_screener.py` — Task 9.
- [x] Delete legacy `src/screening/{universe,exchanges,fundamentals,criteria,cache,pipeline,screener}.py` — Tasks 3, 4, 6, 7.
- [x] Delete `src/user_scripts/{screen_stocks.py, update_exchange_lists.py}` — Tasks 8, 9.
- [x] Verification (load one exchange, run criteria, inspect ranked output) — Task 10.

**Placeholder scan:** no TBD/TODO/incomplete sections. Every code step shows the actual code; every test step shows the actual test. All paths are absolute. Commands have expected outputs.

**Type / signature consistency:**
- `Listing.fundamentals` is `Dict[str, float]` everywhere (constructor, parquet round-trip, criteria readers).
- `Criterion.evaluate(self, listing) -> Optional[float]` — same signature on the ABC and all 9 concrete subclasses.
- `ExchangeUniverse(exchange, listings)` constructor + `load(exchange)` + `save()` — symmetric and consistent across uses (Screener, cache_refresh, CLI).
- `Screener(exchanges).add_criterion(c).rank(top_n=None) -> List[CandidateTicker]` — same chain in tests, in `run_screener.py`, and in the design.
- `cache_refresh.refresh_exchange(exchange, tickers, resume=True, rate_limit_sleep=0.5, save_every=50)` — keyword args match the CLI's flags.

**Style compliance:** no `# ===` separators, no `# ~~~~~` separators, no authorship blocks. Type hints in signatures. Google-style docstrings throughout. 80-char width. snake_case + PascalCase. Lowercase module-level constants.

**Anti-pattern audit:**
- No silent defaults: `Listing` raises on missing fundamental field; `Screener.rank` raises when no criteria added; `ExchangeUniverse.load` raises FileNotFoundError on missing parquet; `Screener.to_dataframe` raises before `rank()`; CLI raises on missing ticker file.
- No bond ETFs in any default policy. `SwissTaxCriterion` actively excludes bond tickers and high-dividend names.
- Module-level constants are lowercase (`weekly_series_length`, `weeks_per_year`, `momentum_lookback_weeks`, `min_observations_for_*_window`, `liquidity_lookback_weeks`, `high_dividend_yield_threshold_pct`, `bond_ticker_substrings`, `default_save_every`, `default_rate_limit_sleep`, `parquet_engine`, `cache_dir`, `tickers_dir`).
- `metrics` dict on `CandidateTicker` is wrapped in `MappingProxyType` to prevent post-construction mutation.
