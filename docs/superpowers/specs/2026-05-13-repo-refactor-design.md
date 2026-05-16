# Repo refactor — design

**Date:** 2026-05-13
**Scope:** Full refactor of `src/`, except the user-approved frozen surface (`src/analysis/core/`).
**Goal:** One rebalancing module exposing multiple strategies; `src/analysis/report.py` consumes them and writes Excel; screening rebuilt around per-exchange offline cache files; modelling (Monte Carlo + greeks) refactored to consume the new contracts; `src/orchestration/` collapsed into entry-point scripts.

---

## 1. Scope

### Frozen (do not touch)
- `src/analysis/core/` — `Transaction`, `Portfolio`, `PortfolioAnalyzer`. Reference architecture (data model → state aggregator → orchestrator). All other domain refactors mirror this shape.

### Light edits only
- `src/analysis/loaders/` — `csv_loader.py`, `data_exporter.py`. Reused as-is.
- `src/analysis/plots/` — `portfolio_plotter.py`. Reused as-is.
- `src/shared/data_provider.py`, `src/shared/symbol_map.py`, `src/shared/metrics/`. Reused as-is.
- `src/modelling/greeks/` — code-style pass only (line width, header separators, lowercase module-level constants); replace any local risk-free-rate or spot-fetch constants with calls to `src/shared/risk_free_rate.py` + the data provider.

### Rewritten or new
- `src/screening/` — fundamentally rebuilt around per-exchange parquet cache.
- `src/modelling/rebalancing/` — new package replacing `src/modelling/rebalancing.py` (1064-line legacy deleted).
- `src/modelling/monte_carlo/` — new package replacing `src/modelling/monte_carlo.py` (907-line legacy deleted).
- `src/shared/risk_free_rate.py`, `src/shared/constraints.py` — new cross-cutting modules.
- `src/analysis/report.py` — rewritten as thin orchestrator and CLI entry point.
- `src/user_scripts/` — entry-point scripts: refresh, screen, monte carlo, dashboard.
- `src/orchestration/` — deleted in full.

---

## 2. Inter-domain data contracts

Six contracts pin down the surface between domains. Once frozen, each domain refactors independently against them.

### Contract 1 — per-exchange parquet cache

Path: `data/screener_cache/<exchange_code>.parquet`

```
ticker              : str
currency            : str
sector              : str
market_cap          : float
refreshed_at        : str
# Series (free from Ticker.history(period='3y', interval='1wk'))
closes_weekly       : list[float]   # 156 weekly closes, oldest-first
closes_weekly_dates : list[str]     # ISO dates aligned with closes_weekly
volumes_weekly      : list[float]
dividends_weekly    : list[float]
# Snapshots (Ticker.info, single extra HTTP call per ticker)
pe_ratio_ttm        : float
pe_ratio_forward    : float
pb_ratio            : float
ps_ratio            : float
dividend_yield_ttm  : float
eps_ttm             : float
revenue_growth_yoy  : float
earnings_growth_yoy : float
profit_margin       : float
operating_margin    : float
roe                 : float
roa                 : float
debt_to_equity      : float
free_cash_flow      : float
beta_yf             : float
shares_outstanding  : float
short_ratio         : float
```

Exchange codes (16):
`NYSE NASDAQ LSE SIX XETRA EURONEXT_PARIS EURONEXT_AMSTERDAM EURONEXT_BRUSSELS EURONEXT_LISBON EURONEXT_DUBLIN BIT BME STO OSL CPH HEL`

`NaN` permitted in `closes_weekly` for pre-listing weeks and in `market_cap` / fundamentals when unavailable.

Estimated cache size: ~50 MB total across 16 files at ~1.5K tickers per exchange average.

### Contract 2 — Screener output → Rebalancer candidate input

```python
@dataclass(frozen=True)
class CandidateTicker:
    ticker: str
    exchange: str
    currency: str
    sector: str
    market_cap: float
    composite_score: float          # 0-1, higher = better
    metrics: dict[str, float]       # cagr, vol, sharpe, max_dd, beta_vs_spy
```

`Screener.rank(top_n=...) -> list[CandidateTicker]` ordered by `composite_score` desc.

### Contract 3 — Analyzer → Rebalancer holdings input

New methods on `PortfolioAnalyzer` (additive only — no change to existing methods):

```python
def get_holdings_snapshot(self) -> pd.DataFrame:
    """Snapshot DataFrame, columns:
       ticker, shares, value_chf, weight_pct, currency,
       sector, price_chf, category."""

def get_price_panel(self) -> pd.DataFrame:
    """Wide DataFrame, one column per held ticker, daily close
       in native currency."""
```

Strategies and rebalancer consume the DataFrame directly. No wrapper class.

### Contract 4 — Rebalancer constraint input

```python
@dataclass(frozen=True)
class RebalanceConstraints:
    max_weight_default: float = 0.15
    category_caps: dict[str, float] = field(
        default_factory=lambda: {'commodity': 0.05})
    excluded_categories: tuple[str, ...] = ('speculative',)
    min_position_pct: float = 0.5            # below this = dust, dropped
    swiss_tax_filter: bool = True            # excludes bonds + high-div tickers
    new_capital_chf: float = 0.0
    rebalance_band_chf: tuple[float, float] = (-200.0, 50.0)
```

Lives in `src/shared/constraints.py` alongside `default_constraints()` factory.

### Contract 5 — Rebalancer output

```python
@dataclass(frozen=True)
class RebalancingAction:
    ticker: str
    action: str                  # 'BUY' | 'REDUCE' | 'HOLD'
    shares: int                  # signed: + buy, - reduce, 0 hold
    est_cost_chf: float          # magnitude
    current_wt_pct: float
    target_wt_pct: float
    note: str

class RebalancePlan:
    actions: list[RebalancingAction]
    target_weights: dict[str, float]    # ticker -> fraction summing to 1
    strategy_name: str
    constraints: RebalanceConstraints
    holdings_df: pd.DataFrame
    generated_at: datetime
    def to_dataframe(self) -> pd.DataFrame: ...
    def summary(self) -> dict[str, float]: ...
    def target_weights_dataframe(self) -> pd.DataFrame: ...
    @property
    def total_value_chf(self) -> float: ...
```

### Contract 6 — MonteCarloEngine → consumer

Input: `RebalancePlan` + price panel + `horizon_days` + `n_paths`. Output:

```python
class PathDistribution:
    paths: np.ndarray            # (n_paths, n_steps), CHF NAV
    horizon_days: int
    initial_nav_chf: float
    plan_name: str
    generated_at: datetime
    def percentiles(self, qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> pd.DataFrame: ...
    def terminal_distribution(self) -> np.ndarray: ...
    def max_drawdown_distribution(self) -> np.ndarray: ...
    def prob_below(self, threshold_chf: float) -> float: ...
    def expected_terminal_return(self) -> float: ...
    def to_summary_dict(self) -> dict[str, float]: ...
```

`to_summary_dict()` includes the cost-drag fields:
- `cumulative_cost_chf_p50`
- `cumulative_cost_chf_p95`
- `annual_cost_drag_bps`

---

## 3. Screening domain

```
src/screening/
  listing.py        # Listing dataclass, mirrors Contract 1 (one row)
  universe.py       # ExchangeUniverse: parquet I/O, per-exchange aggregator
  cache_refresh.py  # Resumable yfinance fetch -> parquet writer
  criteria.py       # Criterion ABC + concrete implementations
  screener.py       # Screener orchestrator
  __init__.py       # Re-exports Listing, ExchangeUniverse, Screener,
                    # CandidateTicker
```

### `Listing` (data model, mirrors `Transaction`)

```python
@dataclass(frozen=True)
class Listing:
    ticker: str
    exchange: str
    currency: str
    sector: str
    market_cap: float
    closes_weekly: np.ndarray            # length 156, NaN allowed
    closes_weekly_dates: pd.DatetimeIndex
    volumes_weekly: np.ndarray
    dividends_weekly: np.ndarray
    fundamentals: dict[str, float]       # 17 Tier-B fields by name
    refreshed_at: datetime
```

### `ExchangeUniverse` (state aggregator, mirrors `Portfolio`)

```python
class ExchangeUniverse:
    def __init__(self, exchange: str, listings: list[Listing]): ...
    @classmethod
    def load(cls, exchange: str) -> 'ExchangeUniverse':
        """Load from data/screener_cache/<exchange>.parquet."""
    def save(self) -> None:
        """Atomic write: temp file then rename."""
    def filter(self, predicate) -> 'ExchangeUniverse': ...
    def to_dataframe(self) -> pd.DataFrame: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    @property
    def staleness_days(self) -> float: ...
```

### `Screener` (orchestrator, mirrors `PortfolioAnalyzer`)

```python
class Screener:
    def __init__(self, exchanges: list[str] = None):
        """exchanges=None loads all 16 cached exchanges."""
    def add_criterion(self, criterion: Criterion) -> 'Screener': ...
    def rank(self, top_n: int = None) -> list[CandidateTicker]:
        """Apply all criteria, compute composite score (mean of
           per-criterion z-scores), return ranked."""
    def to_dataframe(self) -> pd.DataFrame: ...
```

### `Criterion` (filter primitive)

```python
class Criterion(ABC):
    name: str
    weight: float = 1.0
    @abstractmethod
    def evaluate(self, listing: Listing) -> Optional[float]:
        """Return score in [0, 1], or None to drop the ticker."""

# Concrete:
class MomentumCriterion(Criterion): ...
class CagrCriterion(Criterion): ...
class SharpeCriterion(Criterion): ...
class MaxDrawdownCriterion(Criterion): ...
class PERatioCriterion(Criterion): ...
class GrowthCriterion(Criterion): ...
class LiquidityCriterion(Criterion): ...
class SectorCriterion(Criterion): ...
class SwissTaxCriterion(Criterion): ...
```

### `cache_refresh` (resumable fetcher)

```python
def refresh_exchange(
    exchange: str,
    tickers: list[str] = None,    # None = exchange's full ticker list
    resume: bool = True,          # skip already-fetched tickers
    rate_limit_sleep: float = 0.5
) -> None:
    """Fetch 3y weekly history + Ticker.info per ticker; write
       parquet atomically. Resumability via per-exchange checkpoint:
       data/screener_cache/.<exchange>.checkpoint."""
```

Ticker-list source per exchange (handled inside `cache_refresh.py`):
- NYSE / NASDAQ — NASDAQ Trader public lists.
- LSE — official LSE CSV; SIX/XETRA/Euronext/BIT/BME/STO/OSL/CPH/HEL — official exchange downloads.
- Existing `src/screening/exchanges.py` (450 lines) is consolidated into `cache_refresh.py`.

---

## 4. Rebalancing domain

```
src/modelling/rebalancing/
  action.py                    # RebalancingAction dataclass (Contract 5)
  plan.py                      # RebalancePlan class (Contract 5)
  rebalancer.py                # Rebalancer orchestrator
  strategies/
    __init__.py                # Strategy registry: name -> class
    base.py                    # Strategy ABC
    mvo.py                     # MVOStrategy
    equal_weight.py            # EqualWeightStrategy
    inverse_vol.py             # InverseVolStrategy
    min_variance.py            # MinVarianceStrategy
  __init__.py                  # Re-exports public surface
```

Screener-fed strategies (`MaxGrowthStrategy`, `RiskAdjustedStrategy`, `MinRiskStrategy`) added in Phase 4 (see Section 7), slotting into the same `Strategy` ABC and consuming `candidates: list[CandidateTicker]`.

### `Strategy` ABC

```python
class Strategy(ABC):
    name: str                              # registry key, e.g. 'mvo'

    @abstractmethod
    def propose(
        self,
        holdings_df: pd.DataFrame,         # Contract 3
        prices: pd.DataFrame,              # Contract 3 (price panel)
        constraints: RebalanceConstraints, # Contract 4
        candidates: list[CandidateTicker] = None,  # Contract 2
                                          # None for portfolio-only strategies
        risk_free_rate: float = None,     # decimal annual; None -> live fetch
    ) -> dict[str, float]:
        """Return target weights {ticker: fraction}, summing to 1."""
```

The strategy returns target weights only. Action generation (BUY/REDUCE/HOLD with shares + CHF) is shared logic in `Rebalancer`, never duplicated per strategy.

### Strategy registry

```python
# src/modelling/rebalancing/strategies/__init__.py
strategy_registry: dict[str, type[Strategy]] = {
    'mvo': MVOStrategy,
    'equal_weight': EqualWeightStrategy,
    'inverse_vol': InverseVolStrategy,
    'min_variance': MinVarianceStrategy,
}

def get_strategy(name: str) -> Strategy:
    if name not in strategy_registry:
        raise KeyError(
            f'Unknown strategy {name!r}. Known: '
            f'{sorted(strategy_registry)}')
    return strategy_registry[name]()
```

### `Rebalancer` (orchestrator)

```python
class Rebalancer:
    def __init__(
        self,
        analyzer: PortfolioAnalyzer,
        strategy: Strategy,
        constraints: RebalanceConstraints = None,
    ): ...

    def propose(
        self,
        candidates: list[CandidateTicker] = None,
    ) -> RebalancePlan:
        """1. holdings_df = analyzer.get_holdings_snapshot()
           2. prices = analyzer.get_price_panel()
           3. rfr = get_live_risk_free_rate()
           4. target_weights = strategy.propose(...)
           5. actions = self._generate_actions(target_weights, holdings_df)
           6. return RebalancePlan(...)"""

    def _generate_actions(
        self,
        target_weights: dict[str, float],
        holdings_df: pd.DataFrame,
    ) -> list[RebalancingAction]:
        """Shared logic: target_value - current_value, classify by
        rebalance_band_chf (HOLD), allocate new_capital_chf to underweight
        first, REDUCE overweight beyond -200 CHF gap. Lifted from
        generate_rebalancing_actions in legacy report.py."""
```

### MVO concrete

`MVOStrategy` is a lift-and-shift of the SLSQP optimiser currently in `report.py`:
- Objective: maximise Sharpe = `(wᵀμ - rf) / sqrt(wᵀΣw)`.
- Constraints: long-only (`w_i ≥ 0`), per-name cap from `constraints.max_weight_default` and `constraints.category_caps`, weights sum to 1.
- Inputs: annualised mean returns from price panel; annualised covariance with `min_periods=30`.
- Drops dust positions below `constraints.min_position_pct`.
- Drops tickers in `constraints.excluded_categories`.

`EqualWeightStrategy`, `InverseVolStrategy`, `MinVarianceStrategy` follow the same input/output contract with their own objective/weighting rules.

---

## 5. Modelling: Monte Carlo + Greeks

```
src/modelling/
  rebalancing/         [Section 4]
  monte_carlo/
    path_distribution.py   # PathDistribution data + query class (Contract 6)
    copula.py              # Clayton copula sampler
    shrinkage.py           # James-Stein expected-return shrinkage
    cost_model.py          # TransactionCostModel + ibkr_default_cost_model()
    engine.py              # MonteCarloEngine orchestrator
    __init__.py
  greeks/
    delta.py theta.py vega.py rho.py gamma.py
    greeks_calculator.py
    __init__.py
```

### `TransactionCostModel`

IBKR Swiss-resident defaults (broker scope: IBKR only):

```python
@dataclass(frozen=True)
class TransactionCostModel:
    commission_bps: float = 7.0          # IBKR tiered avg across venues
    commission_min_chf: float = 1.5      # IBKR per-trade floor
    spread_bps: float = 5.0              # half-spread, liquid-name default
    stamp_duty_bps_swiss: float = 7.5    # Swiss federal stamp on CH securities
    stamp_duty_bps_foreign: float = 15.0 # Swiss federal stamp on foreign
    fx_cost_bps: float = 2.0             # IBKR interbank, near-zero markup

    def cost_chf(
        self,
        trade_value_chf: float,
        ticker: str,
        currency: str,
    ) -> float:
        commission = max(
            trade_value_chf * self.commission_bps / 1e4,
            self.commission_min_chf)
        spread = trade_value_chf * self.spread_bps / 1e4
        is_swiss = ticker.endswith('.SW')
        stamp_bps = (
            self.stamp_duty_bps_swiss if is_swiss
            else self.stamp_duty_bps_foreign)
        stamp = trade_value_chf * stamp_bps / 1e4
        fx = 0.0
        if currency != 'CHF':
            fx = trade_value_chf * self.fx_cost_bps / 1e4
        return commission + spread + stamp + fx


def ibkr_default_cost_model() -> TransactionCostModel:
    """Sensible IBKR defaults for a Swiss tax-resident retail
    investor at typical trade sizes (CHF 200-2000)."""
    return TransactionCostModel()
```

### `MonteCarloEngine`

```python
class MonteCarloEngine:
    def __init__(
        self,
        n_paths: int = 10_000,
        copula_theta: float = 2.0,
        shrinkage_intensity: float = 0.5,
        cost_model: TransactionCostModel = None,    # None = no costs
        rebalance_frequency: str = 'monthly',       # 'none' | 'monthly' |
                                                    # 'quarterly' | 'threshold'
        monthly_contribution_chf: float = 2000.0,
        drift_threshold_pct: float = 5.0,
        random_seed: int = None,
    ): ...

    def simulate(
        self,
        plan: RebalancePlan,
        prices: pd.DataFrame,
        horizon_days: int,
        initial_nav_chf: float,
    ) -> PathDistribution:
        """Simulate forward NAV under target_weights with stochastic
        joint returns. At t=0: deduct cost of executing plan from
        initial NAV. Each rebalance step: compute drift trades to
        restore target_weights; if monthly_contribution_chf > 0 add
        cash and re-allocate to underweight names; deduct cost.
        Costs only deducted when cost_model is not None."""
```

### Internal helpers (one per concern)

- `copula.py` — `sample_clayton(theta, n_samples, n_assets, rng) -> np.ndarray`. Pure function. Uniform [0,1] samples with lower-tail dependence.
- `shrinkage.py` — `james_stein_shrink(returns: pd.DataFrame, intensity: float) -> pd.Series`. Pure function. Shrinks mean returns toward grand mean.

### Greeks (kept)

`src/modelling/greeks/` — interface unchanged. Code-style pass: line width, header separators, lowercase module-level constants. Replace local risk-free-rate or spot-fetch constants with calls to `src/shared/risk_free_rate.py` + the data provider.

### Out of scope for MC

Regime-switching, FX-shock paths (currency returns assumed flat at refresh-time FX), tax effects beyond Swiss stamp duty captured in `TransactionCostModel`. Add later as new engine kwargs if required.

---

## 6. Cross-cutting modules + entry-point scripts

### `src/shared/`

```
src/shared/
  data_provider.py         [kept]
  symbol_map.py            [kept]
  metrics/                 [kept]
  risk_free_rate.py        [NEW]
  constraints.py           [NEW]
```

`src/shared/risk_free_rate.py`:

```python
"""Live risk-free rate, FRED Swiss 10Y with offline fallback."""

import warnings
import pandas_datareader.data as pdr

fred_series_id_default = 'IRLTLT01CHM156N'
fallback_rate_pct = 0.5

_cache: dict[str, float] = {}


def get_live_risk_free_rate(
    series_id: str = fred_series_id_default,
    use_cache: bool = True,
) -> float:
    """Annual risk-free rate in percent.

    Parameters
    ----------
    series_id : str, default='IRLTLT01CHM156N'
        FRED series ID. Default is OECD long-term interest rate
        for Switzerland (monthly, percent per annum).
    use_cache : bool, default=True
        Cache result for the lifetime of the process.

    Returns
    -------
    rate_pct : float
        Annual rate in percent. Falls back to 0.5 on fetch failure
        with a warning.
    """
    if use_cache and series_id in _cache:
        return _cache[series_id]
    try:
        rate = float(pdr.DataReader(series_id, 'fred').iloc[-1, 0])
    except Exception as e:
        warnings.warn(
            f'FRED fetch for {series_id!r} failed '
            f'({type(e).__name__}: {e}); using fallback '
            f'{fallback_rate_pct}%.')
        rate = fallback_rate_pct
    if use_cache:
        _cache[series_id] = rate
    return rate
```

Consumers: `Rebalancer`, `MVOStrategy`, `MinVarianceStrategy`, `SharpeCriterion`, `MonteCarloEngine`, `src/shared/metrics/risk.py`.

`src/shared/constraints.py`: Contract 4 verbatim plus `default_constraints()` factory.

`data/ticker_categories.json` remains the source of truth for per-ticker category labels (`core` | `commodity` | `speculative`); read by `PortfolioAnalyzer.get_holdings_snapshot` so the `category` column is on `holdings_df`.

### `src/analysis/report.py`

Kept. Rewritten as a thin orchestrator + CLI entry point. Module-and-script via `if __name__ == '__main__': main()`. Invoked as `python -m src.analysis.report`.

CLI args:
- `--strategy {mvo, equal_weight, inverse_vol, min_variance}` (Phase 2 set; screener-fed strategies added in Phase 4)
- `--new-capital FLOAT` (default 2000)
- `--max-weight FLOAT` (default 0.15)
- `--degiro-csv PATH`
- `--ibkr-csv PATH`
- `--candidates-from PATH` (Phase 4 — JSON output of `run_screener.py`; required for screener-fed strategies)
- `--output PATH`

Internal flow:
1. Build `RebalanceConstraints` from CLI args.
2. Instantiate `PortfolioAnalyzer(degiro_csv_file_path=..., ibkr_csv_file_path=...)`.
3. `Rebalancer(analyzer, get_strategy(args.strategy), constraints).propose(candidates=...)` → `RebalancePlan`.
4. Assemble Excel sheets: Combined Holdings, Risk Metrics, Per-Holding Analysis, Sector Exposure, Correlation Matrix, Target Allocation, Rebalancing Actions.
5. `DataExporter().export_to_excel(...)` → `results/portfolio_report_<strategy>_YYYYMMDD.xlsx`.

### `src/user_scripts/`

```
src/user_scripts/
  refresh_screener_cache.py
  run_screener.py
  run_monte_carlo.py
  run_analysis_dashboard.py        [kept]
  examples/                        [kept]
```

`refresh_screener_cache.py` — CLI: `--exchange XETRA` or `--all`; `--resume`; `--rate-limit 0.5`. Calls `src.screening.cache_refresh.refresh_exchange(...)`. Writes `data/screener_cache/<exchange>.parquet`.

`run_screener.py` — CLI: `--exchange XETRA NYSE`; `--top-n 50`; `--criteria momentum,cagr,sharpe,pe_value,growth,liquidity`; `--output-format excel|json`; `--output PATH`. Calls `Screener(...).add_criterion(...).rank(top_n=...)`. Writes `results/screener_output_YYYYMMDD.{xlsx,json}`.

`run_monte_carlo.py` — CLI: `--strategy mvo`; `--horizon-days 252`; `--n-paths 10000`; `--rebalance monthly|quarterly|threshold|none`; `--monthly-contribution 2000`; `--no-costs`. Internally: instantiate analyzer + rebalancer (same as `report.py`); `MonteCarloEngine(...).simulate(plan, prices, ...)`; render `PathDistribution.to_summary_dict()` + percentile bands to Excel. Writes `results/monte_carlo_<strategy>_YYYYMMDD.xlsx`.

### Cross-domain rule

Only `src/user_scripts/*.py` and `src/analysis/report.py` may import from multiple `src/<domain>/` packages. `src/analysis/report.py` is the designated portfolio-side cross-domain entry point. All other domain modules import only from their own domain + `src/shared/` + `src/analysis/core/`.

Convention only — no automated lint, but greppable.

---

## 7. Vertical-slice sequencing

Seven phases. Each ends with a working pipeline + manual verification.

### Phase 1 — Cross-cutting infrastructure
- New: `src/shared/risk_free_rate.py`, `src/shared/constraints.py`.
- Edit: `src/analysis/core/analyzer.py` — add `get_holdings_snapshot()` + `get_price_panel()` (additive only, reuses existing primitives).
- Verify: REPL: `get_live_risk_free_rate()` returns ~0.5-1%; `analyzer.get_holdings_snapshot()` and `analyzer.get_price_panel()` return well-formed DataFrames.

### Phase 2 — Rebalancing domain + rewritten `report.py`
- New: `src/modelling/rebalancing/{action,plan,rebalancer}.py`.
- New: `src/modelling/rebalancing/strategies/{base,mvo,equal_weight,inverse_vol,min_variance}.py`.
- Rewrite: `src/analysis/report.py` per Section 6 spec.
- Delete: `src/modelling/rebalancing.py` (1064-line legacy).
- Verify: `python -m src.analysis.report --strategy mvo --new-capital 2000`. Diff produced Excel's `Rebalancing Actions` sheet against most recent `portfolio_report_*.xlsx` in `results/` — actions should match within rounding (same MVO logic). Smoke-test `--strategy equal_weight` and `--strategy inverse_vol`.

### Phase 3 — Screening domain
- New: `src/screening/{listing,universe,criteria,screener,cache_refresh}.py`.
- Rewrite: existing `src/screening/{universe,exchanges,fundamentals,criteria,cache,pipeline,screener}.py` — consolidated into the new files; `pipeline.py` deleted.
- New: `src/user_scripts/refresh_screener_cache.py`, `src/user_scripts/run_screener.py`.
- Delete: `src/user_scripts/screen_stocks.py`, `src/user_scripts/update_exchange_lists.py`.
- Data: First full refresh of all 16 exchanges expected ~6-12 hours under yfinance throttling; resumable.
- Verify: refresh one small exchange (e.g. SIX, ~250 tickers, ~30 min) end-to-end. Load `ExchangeUniverse.load('SIX')`, confirm `len() ≈ 250`, sample one `Listing`, confirm 156 weekly closes + 17 fundamental fields populated. Run `run_screener.py --exchange SIX --top-n 20 --criteria momentum,cagr,sharpe,pe_value`, inspect ranked output.

### Phase 4 — Screener-fed strategies wired into `report.py`
- New: `MaxGrowthStrategy`, `RiskAdjustedStrategy`, `MinRiskStrategy` in `src/modelling/rebalancing/strategies/`. Each accepts `candidates: list[CandidateTicker]` (no longer optional). Port logic from legacy `rebalancing.py` minus all hardcoded data — specifically: `IBKR_HOLDINGS` and `DEGIRO_HOLDINGS` are replaced by `analyzer.get_holdings_snapshot()`; `TOP_20_PICKS` and `COMPOSITE_SCORES` come from `candidates`; `BOND_ALLOC` is removed entirely (per Swiss-tax policy); the per-scenario hardcoded `sells` lists become outputs of the strategy's optimisation rather than authored constants.
- Edit: `src/analysis/report.py` — accept `--candidates-from PATH`, pass into `Rebalancer.propose(candidates=...)`.
- Delete: `src/orchestration/construct_scenarios.py`, `src/orchestration/__init__.py`, `src/orchestration/`.
- Verify: chain `run_screener.py | report.py --strategy max_growth --candidates-from results/screener_output_*.json`. Inspect combined Excel.

### Phase 5 — Monte Carlo refactor
- New: `src/modelling/monte_carlo/{path_distribution,copula,shrinkage,cost_model,engine}.py`.
- Delete: `src/modelling/monte_carlo.py` (907-line legacy).
- New: `src/user_scripts/run_monte_carlo.py`.
- Verify: `run_monte_carlo.py --strategy mvo --horizon-days 252 --n-paths 5000 --rebalance monthly`. Inspect path distribution percentile bands + cost-drag fields. Spot-check `--no-costs` returns paths with strictly higher terminal NAV than the same seed with costs.

### Phase 6 — Greeks code-style pass
- Edit: `src/modelling/greeks/*.py` — line-width compliance, header separators, lowercase module-level constants; replace local RFR/spot constants with calls to `src/shared/risk_free_rate.py` + data provider.
- Verify: existing example `src/user_scripts/examples/options_greeks_example.py` still runs end-to-end.

### Phase 7 — Cleanup + docs
- Delete `_OLD/`-equivalent stragglers in active source tree.
- Update `README.md` with new entry-point list, exchange codes, refresh cadence note.
- Grep-audit cross-domain imports per the Section 6 rule.
- Update `tasks/lessons.md` with patterns learned during refactor.

### Aggregate phasing summary

| Phase | New files | Deletes | Working pipeline at end                            |
|-------|----------:|--------:|----------------------------------------------------|
| 1     |  3        | 0       | shared infra wired into core                       |
| 2     | ~10       | 1       | `report.py` replacement, portfolio-only strategies |
| 3     |  6        | ~9      | screener cache + ranked candidate output           |
| 4     |  3        | 3       | full report with screener-fed strategies           |
| 5     |  6        | 1       | MC simulation with cost drag                       |
| 6     |  0        | 0       | greeks compliant                                   |
| 7     |  0        | varies  | docs + lint                                        |

Each phase is independently mergeable. Phases 1+2 are the critical path to the stated goal: *"report.py should rebalance the current portfolio and write the excel report"*. Phases 3-4 layer screening back in. Phases 5-7 finish modelling and tidy.

---

## 8. Style and conventions

All Python files comply with `~/.claude/rules/code-style.md`:
- Maximum 80 characters per line.
- Module docstring → import sections (Standard / Third-party / Local) → authorship → code separators.
- snake_case variables and functions; PascalCase classes; lowercase module-level constants and class attributes (per memory `feedback_no_caps_globals.md`).
- Single quotes for strings; double quotes only for docstrings.
- NumPy-style docstrings; no type hints in signatures (types live in docstrings).
- No silent defaults via `.get(key, default)` to mask missing keys (per memory `feedback_no_silent_defaults.md`); raise or let `KeyError` propagate.
- No bond ETFs in any default strategy (per memory `feedback_swiss_tax_no_bonds.md`).
