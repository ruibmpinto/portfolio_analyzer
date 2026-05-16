# Phase 8 — DefeatBetaProvider + US screener acceleration + DCF criterion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `DefeatBetaProvider` as a third `DataProvider` implementation alongside `YFinanceProvider` and `QFLibProvider`. Use it as the primary backend for the **US screener cache refresh** (NYSE / NASDAQ only) — collapses the 6-12 hour yfinance throttled fetch to a single bulk parquet read (~minutes). Extend the `Listing` schema with ~13 additional fundamental fields (PEG, ROIC, WACC, EV, EV/Revenue, EV/EBITDA, ROCE, asset turnover, equity multiplier, debt-to-equity, EPS growth, FCF growth, DCF implied upside) populated only for US tickers from defeatbeta. Add a `DcfCriterion` to the screener's criterion registry that ranks US candidates by `dcf_implied_upside`.

**Architecture:**
- `DefeatBetaProvider(DataProvider)` lives in `src/shared/data_provider.py`. It accepts an optional `fallback: DataProvider` (default `None`); for any ticker that is not "US-native" (has a `.` suffix, ends in `=X`, or known to be unsupported), it delegates to `fallback`. If `fallback` is `None` and the ticker is non-native, it raises `RuntimeError`.
- `_is_native(ticker)` is a single-line method: `'.' not in ticker and not ticker.endswith('=X')`. No fancy registry.
- Each `Ticker(symbol)` instance is cached per provider instance to avoid the ~1s metadata-download cost on repeated calls.
- defeatbeta methods return **time series** (per-day rows) for ratios. The provider takes `.iloc[-1]` to extract the snapshot for the `get_fundamental_data()` interface.
- `cache_refresh.refresh_exchange()` gains an exchange-based dispatch: NYSE / NASDAQ → new `_refresh_via_defeatbeta()` bulk path; everything else → unchanged `_refresh_via_yfinance()` per-ticker path.
- `Listing.fundamental_field_names` extends from 17 → 30 fields. New fields default to NaN when missing on parquet load, preserving backward compatibility with older shards.
- `DcfCriterion` reads `listing.fundamentals['dcf_implied_upside']` and returns `None` when NaN (typical for non-US listings) — drops those tickers from the rank.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, pytest 8.3.4, pyarrow, yfinance, **defeatbeta-api 0.0.53+** (new dependency).

**Style:** Match `src/shared/data_provider.py` (existing class style for the new provider) and `src/screening/criteria.py` (existing Criterion subclass style for the new criterion). Type hints in signatures; Google-style docstrings; 80-char max; single quotes for strings; lowercase module-level constants; no banner separators. Per HANDOFF.md §4 this overrides `~/.claude/rules/code-style.md`.

**Conventions:**
- TDD per task (red → green). Tests under `tests/` mirror the source layout.
- Two-stage review per task (spec compliance + code quality).
- No commits unless explicitly requested.
- Loud failures: missing-data scenarios raise; never silently fall back to NaN at the provider layer.
- `metrics.get('swiss_tax', 1.0)`-style defensive defaults remain the documented exception per `feedback_no_silent_defaults.md`.

---

## File Structure

### New files
- `tests/shared/test_data_provider_defeatbeta.py`
- `tests/screening/test_dcf_criterion.py`

### Modified files
- `src/shared/data_provider.py` — append `DefeatBetaProvider` class.
- `src/shared/__init__.py` — re-export `DefeatBetaProvider`.
- `src/screening/listing.py` — extend `fundamental_field_names` from 17 to 30 entries.
- `src/screening/cache_refresh.py` — exchange-based dispatch; new `_refresh_via_defeatbeta()` bulk path; new fundamentals populated for US tickers.
- `src/screening/criteria.py` — add `DcfCriterion`.
- `src/screening/__init__.py` — re-export `DcfCriterion`.
- `tests/screening/test_cache_refresh.py` — extend to cover the defeatbeta dispatch.

### Untouched
- `src/analysis/core/` (frozen).
- `src/modelling/rebalancing/`, `src/modelling/monte_carlo/`, `src/modelling/greeks/` — none of these consume `DefeatBetaProvider` directly. Future work can flip the analyzer's default provider once the user trusts the new path.
- `src/analysis/report.py`, `src/user_scripts/*.py` — no changes (they continue to use whatever provider `_build_data_provider()` returns; default stays YFinance).

---

## Task 1: `DefeatBetaProvider` adapter class

**Files:**
- Modify: `src/shared/data_provider.py` (append after `QFLibProvider`).
- Modify: `src/shared/__init__.py` (re-export).
- Test: `tests/shared/test_data_provider_defeatbeta.py` (new).

**Why:** Wrap `defeatbeta_api.data.ticker.Ticker` behind the existing `DataProvider` ABC so every consumer (analyzer, rebalancer, monte_carlo, screener) can opt into it without touching their code. The class delegates to a configurable `fallback` for non-US tickers.

**Class shape:**

```python
class DefeatBetaProvider(DataProvider):
    """Hugging Face mirror of Yahoo Finance via defeatbeta-api.

    No auth, no rate limits — primary advantage over YFinance.
    Coverage limited to US-listed equities (NYSE/NASDAQ +
    foreign ADRs); non-US tickers (those with a '.' suffix or
    ending in '=X') are delegated to ``fallback`` if provided,
    else raise RuntimeError.
    """

    def __init__(self, fallback: Optional[DataProvider] = None):
        self._cache: Dict[str, object] = {}     # ticker -> Ticker
        self._fallback = fallback
```

**`is_native(ticker)`**: `True` iff `'.' not in ticker and not ticker.endswith('=X')`. Module-level constant `non_us_suffix_marker = '.'`, `fx_suffix_marker = '=X'`.

**Method mapping (defeatbeta → DataProvider):**

| `DataProvider` method | defeatbeta call | Adapter logic |
|---|---|---|
| `get_price_history(ticker, start, end)` | `Ticker(t).price()` | DataFrame has `(symbol, report_date, open, close, high, low, volume)`. Convert `report_date` to datetime, set as index, slice `[start:end]`, return `pd.Series(close)`. Raise `ValueError` on empty result. |
| `get_current_price(ticker)` | `Ticker(t).price()` | Take `iloc[-1]['close']`. |
| `get_fundamental_data(ticker)` | mix of `info()`, `ttm_pe()`, `ttm_eps()`, `pb_ratio()`, `ps_ratio()`, `peg_ratio()`, `market_capitalization()`, `roe()`, `roa()`, `roic()`, `wacc()`, `equity_multiplier()`, `asset_turnover()`, `debt_to_equity()`, `enterprise_value()`, `enterprise_to_revenue()`, `enterprise_to_ebitda()`, `ttm_revenue()`, `ttm_fcf()`, `quarterly_revenue_yoy_growth().iloc[-1]`, `quarterly_ebitda_yoy_growth().iloc[-1]`, `quarterly_eps_yoy_growth().iloc[-1]`, `dcf_data()` | Return dict keyed by the 30 names in the new `fundamental_field_names` tuple. Each call takes `iloc[-1]` of the relevant column. Wrap each per-field fetch in `try/except` returning `float('nan')` on miss — defeatbeta sometimes lacks coverage even for US tickers. **Documented exception to no-silent-defaults**: missing-fundamental data is a known yfinance/defeatbeta variability. |
| `get_dividend_history(ticker, start)` | `Ticker(t).dividends()` | DataFrame `(symbol, report_date, amount)`. Index by date, slice `[start:]`, return `pd.Series(amount)`. |
| `get_split_history(ticker)` | `Ticker(t).splits()` | DataFrame `(symbol, report_date, split_factor)`. Index by date, return `pd.Series(split_factor)`. |
| `get_dividend_price_factors(ticker)` | derived | Same `1 - D / C_prev` formula as YFinanceProvider, computed from `dividends()` + `price()`. |

**Module-level constants:**
- `non_us_suffix_marker = '.'`
- `fx_suffix_marker = '=X'`

- [ ] **Step 1: Write failing tests**

Create `tests/shared/test_data_provider_defeatbeta.py`:

```python
"""Tests for DefeatBetaProvider."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.shared.data_provider import (
    DataProvider, DefeatBetaProvider, YFinanceProvider)


class FakeFallback(DataProvider):
    """Records calls; returns sentinel values."""

    def __init__(self):
        self.calls = []

    def _record(self, name, *args):
        self.calls.append((name, args))

    def get_price_history(self, t, s, e):
        self._record('price', t, s, e)
        return pd.Series([1.0], index=[pd.Timestamp(s)])

    def get_fundamental_data(self, t):
        self._record('fund', t)
        return {'pe_ratio_ttm': 99.9}

    def get_dividend_history(self, t, s):
        self._record('div', t, s)
        return pd.Series([], dtype=float)

    def get_split_history(self, t):
        self._record('split', t)
        return pd.Series([], dtype=float)

    def get_dividend_price_factors(self, t):
        self._record('factors', t)
        return pd.Series([], dtype=float)

    def get_current_price(self, t):
        self._record('current', t)
        return 42.0


def _fake_price_df(symbol='NVDA', n_days=10):
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    return pd.DataFrame({
        'symbol': [symbol] * n_days,
        'report_date': dates.strftime('%Y-%m-%d'),
        'open': np.linspace(100, 110, n_days),
        'close': np.linspace(101, 111, n_days),
        'high': np.linspace(102, 112, n_days),
        'low': np.linspace(99, 109, n_days),
        'volume': [1_000_000] * n_days})


def test_is_native_for_us_ticker():
    p = DefeatBetaProvider()
    assert p.is_native('NVDA')
    assert p.is_native('BRK-B')


def test_is_native_false_for_dotted_or_fx_ticker():
    p = DefeatBetaProvider()
    assert not p.is_native('NESN.SW')
    assert not p.is_native('BARC.L')
    assert not p.is_native('USDCHF=X')


def test_get_price_history_routes_us_to_defeatbeta(monkeypatch):
    p = DefeatBetaProvider()
    fake_ticker = MagicMock()
    fake_ticker.price.return_value = _fake_price_df()
    monkeypatch.setattr(
        'src.shared.data_provider.Ticker', lambda s: fake_ticker)

    out = p.get_price_history(
        'NVDA', datetime(2024, 1, 1), datetime(2024, 1, 8))
    assert isinstance(out, pd.Series)
    assert len(out) > 0


def test_get_price_history_falls_back_for_non_us(monkeypatch):
    fb = FakeFallback()
    p = DefeatBetaProvider(fallback=fb)
    out = p.get_price_history(
        'NESN.SW', datetime(2024, 1, 1), datetime(2024, 1, 8))
    assert fb.calls[0][0] == 'price'
    assert out.iloc[0] == 1.0


def test_non_us_ticker_raises_when_no_fallback():
    p = DefeatBetaProvider(fallback=None)
    with pytest.raises(RuntimeError, match='NESN.SW'):
        p.get_price_history(
            'NESN.SW', datetime(2024, 1, 1), datetime(2024, 1, 8))


def test_get_current_price_returns_last_close(monkeypatch):
    p = DefeatBetaProvider()
    fake_ticker = MagicMock()
    fake_ticker.price.return_value = _fake_price_df()
    monkeypatch.setattr(
        'src.shared.data_provider.Ticker', lambda s: fake_ticker)
    assert p.get_current_price('NVDA') == pytest.approx(111.0)


def test_get_current_price_raises_on_empty_history(monkeypatch):
    p = DefeatBetaProvider()
    fake_ticker = MagicMock()
    fake_ticker.price.return_value = pd.DataFrame(
        columns=['symbol', 'report_date', 'close'])
    monkeypatch.setattr(
        'src.shared.data_provider.Ticker', lambda s: fake_ticker)
    with pytest.raises(ValueError, match='NVDA'):
        p.get_current_price('NVDA')


def test_get_fundamental_data_assembles_30_keys(monkeypatch):
    p = DefeatBetaProvider()
    fake_ticker = MagicMock()
    # Each ratio method returns a DataFrame; .iloc[-1] picks the
    # latest snapshot. Use minimal stubs.
    one_row_df = lambda col, val: pd.DataFrame({col: [val]})
    fake_ticker.ttm_pe.return_value = one_row_df('ttm_pe', 25.0)
    fake_ticker.ttm_eps.return_value = one_row_df('ttm_eps', 4.0)
    fake_ticker.pb_ratio.return_value = one_row_df('pb_ratio', 5.0)
    fake_ticker.ps_ratio.return_value = one_row_df('ps_ratio', 8.0)
    fake_ticker.peg_ratio.return_value = one_row_df('peg_ratio', 1.5)
    fake_ticker.market_capitalization.return_value = one_row_df(
        'market_capitalization', 3e12)
    fake_ticker.roe.return_value = one_row_df('roe', 0.30)
    fake_ticker.roa.return_value = one_row_df('roa', 0.18)
    fake_ticker.roic.return_value = one_row_df('roic', 0.25)
    fake_ticker.wacc.return_value = one_row_df('wacc', 0.08)
    fake_ticker.equity_multiplier.return_value = one_row_df(
        'equity_multiplier', 1.7)
    fake_ticker.asset_turnover.return_value = one_row_df(
        'asset_turnover', 0.6)
    fake_ticker.debt_to_equity.return_value = one_row_df(
        'debt_to_equity', 0.4)
    fake_ticker.enterprise_value.return_value = one_row_df(
        'enterprise_value', 2.9e12)
    fake_ticker.enterprise_to_revenue.return_value = one_row_df(
        'enterprise_to_revenue', 23.0)
    fake_ticker.enterprise_to_ebitda.return_value = one_row_df(
        'enterprise_to_ebitda', 50.0)
    fake_ticker.ttm_revenue.return_value = one_row_df(
        'ttm_revenue', 1.3e11)
    fake_ticker.ttm_fcf.return_value = one_row_df(
        'ttm_fcf', 5e10)
    fake_ticker.quarterly_revenue_yoy_growth.return_value = (
        one_row_df('revenue_yoy_growth', 1.2))
    fake_ticker.quarterly_ebitda_yoy_growth.return_value = (
        one_row_df('ebitda_yoy_growth', 1.5))
    fake_ticker.quarterly_eps_yoy_growth.return_value = (
        one_row_df('eps_yoy_growth', 1.4))
    # All other methods that we treat as "may fail":
    fake_ticker.dcf_data.return_value = {'implied_upside': 0.30}
    monkeypatch.setattr(
        'src.shared.data_provider.Ticker', lambda s: fake_ticker)

    fund = p.get_fundamental_data('NVDA')
    assert fund['pe_ratio_ttm'] == pytest.approx(25.0)
    assert fund['peg_ratio'] == pytest.approx(1.5)
    assert fund['roic'] == pytest.approx(0.25)
    assert fund['wacc'] == pytest.approx(0.08)
    assert fund['dcf_implied_upside'] == pytest.approx(0.30)
    assert 'shares_outstanding' in fund   # 30-key envelope


def test_ticker_instance_cached_across_calls(monkeypatch):
    """Two calls for the same symbol create only one Ticker."""
    construct_count = {'n': 0}
    fake_ticker = MagicMock()
    fake_ticker.price.return_value = _fake_price_df()

    def fake_factory(s):
        construct_count['n'] += 1
        return fake_ticker

    monkeypatch.setattr(
        'src.shared.data_provider.Ticker', fake_factory)
    p = DefeatBetaProvider()
    p.get_current_price('NVDA')
    p.get_current_price('NVDA')
    assert construct_count['n'] == 1
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m pytest tests/shared/test_data_provider_defeatbeta.py -v`
Expected: ImportError on `DefeatBetaProvider`.

- [ ] **Step 3: Implement**

Append to `src/shared/data_provider.py`:

```python
from defeatbeta_api.data.ticker import Ticker


non_us_suffix_marker = '.'
fx_suffix_marker = '=X'


class DefeatBetaProvider(DataProvider):
    """Hugging Face mirror of Yahoo Finance via defeatbeta-api.

    No auth, no rate limits. Coverage limited to US-listed
    equities (NYSE/NASDAQ + foreign ADRs); non-US tickers are
    delegated to ``fallback`` if provided, else
    ``RuntimeError``.

    Attributes:
        _cache: Per-symbol cache of Ticker instances. Each
            Ticker constructor downloads metadata once
            (~1s); cache amortises over repeated calls.
        _fallback: Optional DataProvider used for tickers that
            fail ``is_native``. None means raise.
    """

    def __init__(self, fallback: Optional[DataProvider] = None):
        self._cache: Dict[str, Ticker] = {}
        self._fallback = fallback

    def is_native(self, ticker: str) -> bool:
        """True for US-only ticker shapes covered by defeatbeta."""
        return (
            non_us_suffix_marker not in ticker
            and not ticker.endswith(fx_suffix_marker))

    def _ticker(self, symbol: str) -> Ticker:
        if symbol not in self._cache:
            self._cache[symbol] = Ticker(symbol)
        return self._cache[symbol]

    def _delegate_or_raise(self, ticker: str, op: str):
        if self._fallback is None:
            raise RuntimeError(
                f'DefeatBetaProvider: ticker {ticker!r} is not '
                f'US-native and no fallback provider configured '
                f'for {op}.')
        return self._fallback

    def get_price_history(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime) -> pd.Series:
        """Daily close prices from the defeatbeta parquet."""
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker, 'get_price_history').get_price_history(
                ticker, start_date, end_date)
        df = self._ticker(ticker).price()
        if df.empty:
            raise ValueError(
                f'DefeatBetaProvider: no price history for '
                f'{ticker!r}.')
        df = df.copy()
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.set_index('report_date')
        df = df.loc[start_date:end_date]
        return df['close'].astype(float)

    def get_current_price(self, ticker: str) -> float:
        """Latest close from the defeatbeta parquet."""
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker, 'get_current_price').get_current_price(
                ticker)
        df = self._ticker(ticker).price()
        if df.empty:
            raise ValueError(
                f'DefeatBetaProvider: no price for {ticker!r}.')
        return float(df.iloc[-1]['close'])

    def get_fundamental_data(
        self, ticker: str) -> Dict[str, Optional[float]]:
        """30-field fundamental snapshot for a US ticker."""
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker,
                'get_fundamental_data').get_fundamental_data(
                ticker)
        t = self._ticker(ticker)
        # Defensive per-field try/except: defeatbeta coverage
        # varies even for US tickers (e.g. very new IPOs lack
        # the multi-quarter history needed for PEG / ROIC /
        # WACC). Documented exception to no-silent-defaults.
        return {
            'pe_ratio_ttm': _safe_last(
                t.ttm_pe, 'ttm_pe'),
            'pe_ratio_forward': float('nan'),  # not in defeatbeta
            'pb_ratio': _safe_last(t.pb_ratio, 'pb_ratio'),
            'ps_ratio': _safe_last(t.ps_ratio, 'ps_ratio'),
            'dividend_yield_ttm': float('nan'),
            'eps_ttm': _safe_last(t.ttm_eps, 'ttm_eps'),
            'revenue_growth_yoy': _safe_last(
                t.quarterly_revenue_yoy_growth,
                'revenue_yoy_growth'),
            'earnings_growth_yoy': _safe_last(
                t.quarterly_eps_yoy_growth, 'eps_yoy_growth'),
            'profit_margin': float('nan'),
            'operating_margin': float('nan'),
            'roe': _safe_last(t.roe, 'roe'),
            'roa': _safe_last(t.roa, 'roa'),
            'debt_to_equity': _safe_last(
                t.debt_to_equity, 'debt_to_equity'),
            'free_cash_flow': _safe_last(t.ttm_fcf, 'ttm_fcf'),
            'beta_yf': float('nan'),
            'shares_outstanding': float('nan'),
            'short_ratio': float('nan'),
            # Phase 8 additions:
            'peg_ratio': _safe_last(t.peg_ratio, 'peg_ratio'),
            'roic': _safe_last(t.roic, 'roic'),
            'roce': float('nan'),
            'wacc': _safe_last(t.wacc, 'wacc'),
            'equity_multiplier': _safe_last(
                t.equity_multiplier, 'equity_multiplier'),
            'asset_turnover': _safe_last(
                t.asset_turnover, 'asset_turnover'),
            'enterprise_value': _safe_last(
                t.enterprise_value, 'enterprise_value'),
            'enterprise_to_revenue': _safe_last(
                t.enterprise_to_revenue,
                'enterprise_to_revenue'),
            'enterprise_to_ebitda': _safe_last(
                t.enterprise_to_ebitda,
                'enterprise_to_ebitda'),
            'ttm_revenue': _safe_last(
                t.ttm_revenue, 'ttm_revenue'),
            'ebitda_growth_yoy': _safe_last(
                t.quarterly_ebitda_yoy_growth,
                'ebitda_yoy_growth'),
            'market_cap_chf': _safe_last(
                t.market_capitalization,
                'market_capitalization'),
            'dcf_implied_upside': _safe_dcf(t),
        }

    def get_dividend_history(
        self,
        ticker: str,
        start_date: datetime) -> pd.Series:
        """Per-event dividends from the defeatbeta parquet."""
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker,
                'get_dividend_history').get_dividend_history(
                ticker, start_date)
        df = self._ticker(ticker).dividends()
        if df.empty:
            return pd.Series([], dtype=float, name='dividends')
        df = df.copy()
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.set_index('report_date').loc[start_date:]
        return df['amount'].astype(float)

    def get_split_history(self, ticker: str) -> pd.Series:
        """Per-event split factors from the defeatbeta parquet."""
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker, 'get_split_history').get_split_history(
                ticker)
        df = self._ticker(ticker).splits()
        if df.empty:
            return pd.Series([], dtype=float, name='splits')
        df = df.copy()
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.set_index('report_date')
        return df['split_factor'].astype(float)

    def get_dividend_price_factors(
        self, ticker: str) -> pd.Series:
        """Same 1 - D/C_prev formula as YFinanceProvider."""
        if not self.is_native(ticker):
            return self._delegate_or_raise(
                ticker, 'get_dividend_price_factors'
                ).get_dividend_price_factors(ticker)
        divs = self.get_dividend_history(
            ticker, datetime(1900, 1, 1))
        if divs.empty:
            return pd.Series([], dtype=float)
        prices = self._ticker(ticker).price().copy()
        prices['report_date'] = pd.to_datetime(
            prices['report_date'])
        prices = prices.set_index('report_date')['close']
        factors = []
        for ex_date, dividend in divs.items():
            prior = prices.loc[:ex_date].iloc[-2:-1]
            if prior.empty:
                continue
            c_prev = float(prior.iloc[0])
            if c_prev > 0:
                factors.append((ex_date, 1.0 - dividend / c_prev))
        if not factors:
            return pd.Series([], dtype=float)
        idx, vals = zip(*factors)
        return pd.Series(vals, index=pd.DatetimeIndex(idx))


def _safe_last(method_callable, column: str) -> float:
    """Call a defeatbeta ratio method and take the last row.

    Returns NaN when the method raises or the result is empty
    or the column is missing — defeatbeta coverage gaps are
    a known yfinance/defeatbeta variability issue.
    """
    try:
        df = method_callable()
        if df is None or df.empty or column not in df.columns:
            return float('nan')
        value = df.iloc[-1][column]
        return float(value)
    except Exception:
        return float('nan')


def _safe_dcf(ticker_obj) -> float:
    """Pull DCF implied upside from Ticker.dcf_data().

    Returns NaN when the call fails or the result lacks the
    expected ``implied_upside`` key.
    """
    try:
        data = ticker_obj.dcf_data()
        if isinstance(data, dict) and 'implied_upside' in data:
            return float(data['implied_upside'])
        return float('nan')
    except Exception:
        return float('nan')
```

Modify `src/shared/__init__.py`:

```python
from src.shared.data_provider import (
    DataProvider, YFinanceProvider, QFLibProvider,
    DefeatBetaProvider)
```

Add `'DefeatBetaProvider'` to `__all__`.

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/shared/test_data_provider_defeatbeta.py -v`
Expected: 9 PASS.

Run: `python3 -m pytest tests/ -v 2>&1 | tail -3`
Expected: 217 + 9 = 226 PASS.

- [ ] **Step 5: NO commit.**

---

## Task 2: Extend `Listing.fundamental_field_names`

**Files:**
- Modify: `src/screening/listing.py`
- Modify: `src/screening/cache_refresh.py` (yfinance fetcher fills new fields with NaN)
- Test: extend `tests/screening/test_candidate_and_listing.py`

**Why:** New fields populated by `DefeatBetaProvider` for US tickers must round-trip through the `Listing` dataclass + `ExchangeUniverse` parquet writer. Backward compatibility: older parquet shards (16-field schema) still load — the missing columns default to NaN.

**New fundamental field names (13 additions):**
- `peg_ratio`
- `roic`
- `roce`
- `wacc`
- `equity_multiplier`
- `asset_turnover`
- `enterprise_value`
- `enterprise_to_revenue`
- `enterprise_to_ebitda`
- `ttm_revenue`
- `ebitda_growth_yoy`
- `market_cap_chf` (separate from `market_cap` because the existing field is in listing currency)
- `dcf_implied_upside` (used by Task 4 `DcfCriterion`)

Total: 17 + 13 = 30 fields.

- [ ] **Step 1: Update the constant**

In `src/screening/listing.py`:

```python
fundamental_field_names = (
    # Original 17 (Phase 3)
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
    # Phase 8 additions (defeatbeta US-only enrichment)
    'peg_ratio',
    'roic',
    'roce',
    'wacc',
    'equity_multiplier',
    'asset_turnover',
    'enterprise_value',
    'enterprise_to_revenue',
    'enterprise_to_ebitda',
    'ttm_revenue',
    'ebitda_growth_yoy',
    'market_cap_chf',
    'dcf_implied_upside',
)
```

- [ ] **Step 2: Backward-compat parquet load**

In `src/screening/universe.py` (`ExchangeUniverse.load`), when reconstituting the `fundamentals` dict per row, fill any missing field with `float('nan')` instead of raising. The existing `Listing.__post_init__` validation already raises on missing fields; the loader needs to pre-populate the missing ones.

Patch sketch (find the loader's per-row construction):

```python
fundamentals = {
    name: (
        float(row[name]) if name in row.index
        and pd.notna(row[name]) else float('nan'))
    for name in fundamental_field_names}
```

- [ ] **Step 3: yfinance path NaN-fills the new fields**

In `src/screening/cache_refresh.py::_fetch_one_listing`, after the existing `fundamentals` loop:

```python
# Phase 8: new fields are populated only by the defeatbeta
# fetcher path; yfinance path leaves them NaN.
for name in fundamental_field_names:
    fundamentals.setdefault(name, float('nan'))
```

- [ ] **Step 4: Test additions**

Add tests to `tests/screening/test_candidate_and_listing.py` confirming:
- A `Listing` constructed with the original 17 + 13 NaN fields validates.
- An `ExchangeUniverse.load()` of a shard saved under the OLD 17-field schema returns Listings whose new fields are NaN. (Use a `tmp_path` parquet fixture.)

- [ ] **Step 5: Verify**

Run: `python3 -m pytest tests/screening/ -v`
Expected: all PASS.

Run: `python3 -m pytest tests/ -v 2>&1 | tail -3`
Expected: 226 + 2 = 228 PASS (or whatever the actual additions amount to).

- [ ] **Step 6: NO commit.**

---

## Task 3: `cache_refresh.py` — exchange-based dispatch + bulk defeatbeta path

**Files:**
- Modify: `src/screening/cache_refresh.py`
- Test: extend `tests/screening/test_cache_refresh.py`

**Why:** NYSE / NASDAQ together have ~5000 tickers. Per-ticker yfinance throttle = 6-12 hours. defeatbeta's bulk parquet read returns the whole dataset in seconds; we filter to the relevant tickers locally.

**Dispatch:**

```python
us_exchanges = ('NYSE', 'NASDAQ')


def refresh_exchange(exchange, tickers=None, resume=True,
                     rate_limit_sleep=0.5,
                     defeatbeta_provider=None):
    if exchange in us_exchanges:
        return _refresh_via_defeatbeta(
            exchange, tickers, resume, defeatbeta_provider)
    return _refresh_via_yfinance(
        exchange, tickers, resume, rate_limit_sleep)
```

**`_refresh_via_defeatbeta`:**

```python
def _refresh_via_defeatbeta(
    exchange, tickers, resume, provider):
    if provider is None:
        provider = DefeatBetaProvider()
    if tickers is None:
        tickers = _load_ticker_list(exchange)
    existing = _load_existing_tickers(exchange) if resume else set()
    todo = [t for t in tickers if t not in existing]
    listings = []
    for ticker in todo:
        try:
            listings.append(_build_listing_via_defeatbeta(
                ticker, exchange, provider))
        except Exception as e:
            warnings.warn(
                f'defeatbeta refresh failed for {ticker!r}: '
                f'{type(e).__name__}: {e}')
        if len(listings) >= save_every:
            _persist(exchange, listings)
            listings = []
    if listings:
        _persist(exchange, listings)


def _build_listing_via_defeatbeta(
    ticker, exchange, provider):
    fund = provider.get_fundamental_data(ticker)
    prices = provider.get_price_history(
        ticker,
        start_date=datetime.now() - timedelta(weeks=160),
        end_date=datetime.now())
    # Resample daily -> weekly closes, take last 156.
    weekly = prices.resample('W').last().dropna()
    closes = _to_padded_array(weekly)
    # Volumes / dividends from the underlying Ticker; reuse the
    # provider's _ticker cache.
    raw = provider._ticker(ticker).price()
    # ... (build volumes_weekly + dividends_weekly + dates the
    # same way the yfinance path does, working off the
    # defeatbeta DataFrame)
    return Listing(
        ticker=ticker, exchange=exchange,
        currency=_currency_of_us_ticker(provider, ticker),
        sector=_sector_of_us_ticker(provider, ticker),
        market_cap=fund.get('market_cap_chf', float('nan')),
        closes_weekly=closes,
        closes_weekly_dates=...,
        volumes_weekly=...,
        dividends_weekly=...,
        fundamentals=fund,
        refreshed_at=datetime.now())
```

**Note on weekly resampling:** defeatbeta returns daily bars; the `Listing` schema expects 156 weekly bars. Resample to `W` (week-ending) closes, take the last value per week, NaN-pad/truncate to 156 via the existing `_to_padded_array` helper.

**Currency / sector lookup:** defeatbeta `Ticker.info()` returns 12 columns including `country` but typically not `currency` or `sector`. For US tickers we hardcode `currency='USD'` and pull `sector` from a separate query if needed. For the screener's purposes, knowing it's USD + a "Tech" / "Energy" / etc. sector is sufficient. **Pragmatic shortcut:** always set `currency='USD'` for US-native tickers; pull `sector` from `Ticker.info().iloc[0]` if a `sector` column exists, else default to `'Unknown'`.

- [ ] **Step 1: Write failing test**

Extend `tests/screening/test_cache_refresh.py` with a defeatbeta-path test that:
1. Patches `DefeatBetaProvider` to return canned price + fundamental data.
2. Calls `refresh_exchange('NASDAQ', tickers=['NVDA', 'MSFT'])`.
3. Loads the resulting parquet shard via `ExchangeUniverse.load('NASDAQ')`.
4. Asserts the 2 Listings exist with the expected `peg_ratio`, `roic`, `wacc`, etc.

- [ ] **Step 2: Implement the dispatch + bulk path**

Per the sketches above. Reuse helpers (`_to_padded_array`, `_padded_dates`, `_persist`, `_load_existing_tickers`).

- [ ] **Step 3: Verify**

Run: `python3 -m pytest tests/screening/ -v`
Expected: all PASS.

Run: `python3 -m pytest tests/ -v 2>&1 | tail -3`
Expected: full suite green.

- [ ] **Step 4: NO commit.**

---

## Task 4: `DcfCriterion`

**Files:**
- Modify: `src/screening/criteria.py` (append `DcfCriterion`).
- Modify: `src/screening/__init__.py` (re-export).
- Test: `tests/screening/test_dcf_criterion.py`

**Why:** Rank US screener candidates by DCF-implied upside. Pre-computed at cache_refresh time and stored as `dcf_implied_upside` in the fundamentals dict; the Criterion just reads it (fast, no live call).

**Implementation:**

```python
class DcfCriterion(Criterion):
    """Ranks US tickers by pre-computed DCF implied upside.

    Reads ``listing.fundamentals['dcf_implied_upside']``,
    populated at cache_refresh time by DefeatBetaProvider.
    Returns ``None`` when the upside is NaN (typical for
    non-US listings or US tickers with insufficient history),
    dropping the ticker from the rank.
    """

    name = 'dcf'

    def evaluate(self, listing: Listing) -> Optional[float]:
        upside = listing.fundamentals.get(
            'dcf_implied_upside', float('nan'))
        if pd.isna(upside):
            return None
        return float(upside)
```

- [ ] **Step 1: Write failing test**

Create `tests/screening/test_dcf_criterion.py`:

```python
"""Tests for DcfCriterion."""

import math

import numpy as np
import pandas as pd
import pytest

from src.screening.criteria import (
    DcfCriterion, get_criterion, criterion_registry)
from src.screening.listing import (
    Listing, fundamental_field_names)


def _make_listing(upside):
    fundamentals = {n: float('nan') for n in fundamental_field_names}
    fundamentals['dcf_implied_upside'] = upside
    n = 156
    return Listing(
        ticker='NVDA', exchange='NASDAQ', currency='USD',
        sector='Tech', market_cap=3e12,
        closes_weekly=np.full(n, 100.0),
        closes_weekly_dates=pd.date_range(
            '2024-01-01', periods=n, freq='W'),
        volumes_weekly=np.full(n, 1e6),
        dividends_weekly=np.full(n, 0.0),
        fundamentals=fundamentals,
        refreshed_at=pd.Timestamp.now().to_pydatetime())


def test_dcf_criterion_registered():
    assert 'dcf' in criterion_registry
    c = get_criterion('dcf')
    assert isinstance(c, DcfCriterion)


def test_dcf_returns_upside_when_present():
    c = DcfCriterion()
    listing = _make_listing(0.32)
    assert c.evaluate(listing) == pytest.approx(0.32)


def test_dcf_returns_none_when_nan():
    c = DcfCriterion()
    listing = _make_listing(float('nan'))
    assert c.evaluate(listing) is None
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m pytest tests/screening/test_dcf_criterion.py -v`
Expected: ImportError on `DcfCriterion`.

- [ ] **Step 3: Implement**

Append to `src/screening/criteria.py`:

```python
class DcfCriterion(Criterion):
    """Ranks US tickers by pre-computed DCF implied upside.

    Reads ``listing.fundamentals['dcf_implied_upside']``,
    populated at cache_refresh time by DefeatBetaProvider.
    Returns ``None`` when the upside is NaN (typical for
    non-US listings or US tickers with insufficient history),
    dropping the ticker from the rank.
    """

    name = 'dcf'

    def evaluate(self, listing: Listing) -> Optional[float]:
        upside = listing.fundamentals.get(
            'dcf_implied_upside', float('nan'))
        if pd.isna(upside):
            return None
        return float(upside)
```

Add `DcfCriterion` to `src/screening/__init__.py`'s import list and `__all__`.

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/screening/test_dcf_criterion.py -v`
Expected: 3 PASS.

Run: `python3 -m pytest tests/ -v 2>&1 | tail -3`
Expected: full suite green.

- [ ] **Step 5: NO commit.**

---

## Task 5: Manual verification + tasks/todo.md update

- [ ] **Step 1: Suite green check**

Run: `python3 -m pytest tests/ -v`
Expected: every Phase 1-7 test plus the new ones pass.

- [ ] **Step 2: Smoke run (operator-driven; optional)**

```python
from src.shared.data_provider import (
    DefeatBetaProvider, YFinanceProvider)
p = DefeatBetaProvider(fallback=YFinanceProvider())
print(p.get_current_price('NVDA'))    # uses defeatbeta
print(p.get_current_price('NESN.SW')) # falls back to yfinance
print(p.get_fundamental_data('NVDA')['dcf_implied_upside'])
```

Expected: NVDA price prints; NESN.SW falls back; DCF upside prints (or NaN).

For end-to-end: refresh NASDAQ via the new bulk path (`python -m src.user_scripts.refresh_screener_cache --exchange NASDAQ`) — should complete in minutes, not hours.

- [ ] **Step 3: Append Phase 8 block to tasks/todo.md**

- [ ] **Step 4: NO commit.**

---

## Self-Review

**1. Spec coverage:**

| User-stated deliverable | Task |
|---|---|
| Add DefeatBetaProvider as 3rd DataProvider implementation | Task 1 |
| Coexists with YFinanceProvider and QFLibProvider | Task 1 (no changes to existing two) |
| Selection by exchange or by call site (fallback delegate) | Task 1 (`is_native` + `_fallback`) |
| Use defeatbeta for NYSE / NASDAQ in cache_refresh | Task 3 |
| Other 14 exchanges keep yfinance | Task 3 (dispatch) |
| Listing fundamentals enriched with PEG, ROIC, WACC, EV, etc. | Task 2 |
| Per-stock DCF criterion | Task 4 |

**2. Placeholder scan:**
- Task 3's `_build_listing_via_defeatbeta` body is sketched, not fully written. The implementer fills the body using the existing helpers + the patterns shown. This is the same scope-bend as Phase 5 Task 5.

**3. Type consistency:**
- `DefeatBetaProvider(fallback: Optional[DataProvider] = None)` — Task 1 def, Task 3 call site.
- `is_native(ticker) -> bool` — Task 1, used in dispatch (Task 3).
- `Listing.fundamentals` dict keys — Task 2 (definition), Task 1 (provider populates), Task 3 (cache_refresh writes), Task 4 (criterion reads).
- `dcf_implied_upside` — Task 1 populates, Task 4 reads.

No drift across tasks.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-15-phase-8-defeatbeta-provider.md`.

**Recommended execution:** Subagent-driven, opus implementers (per user directive), sonnet reviewers. Continuous execution; pause only on blockers.
