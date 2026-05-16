# stock_market

Personal portfolio tooling: load broker statements, value the
combined position in CHF, compute risk and ratio metrics,
simulate forward scenarios, screen for new candidates.

## `src/` layout

```
src/
├── shared/         # cross-cutting infrastructure (leaf of the dep graph)
├── analysis/       # STATIC : read existing portfolio, value it, report on it
├── modelling/      # FORWARD : Monte Carlo, rebalancing, options pricing
├── screening/      # DISCOVERY : find candidate tickers from a universe
└── user_scripts/   # operator-facing CLIs (the only layer allowed to import
                    # from analysis + modelling + screening in one module)
```

### `src/shared/`

Anything every domain needs and no single domain owns.

- `data_provider.py` — yfinance / qf-lib abstraction
  (`DataProvider`, `YFinanceProvider`, `QFLibProvider`)
- `symbol_map.py` — IBKR → Yahoo/Degiro symbol normalisation
- `risk_free_rate.py` — `get_live_risk_free_rate()` returns the
  Swiss long-term rate from FRED as a **decimal fraction** (e.g.
  `0.004` for 0.4%); falls back to `fallback_rate_decimal=0.005`
  on fetch failure. Single source of truth for analyzer
  metrics, rebalancer strategies, and the Monte Carlo engine.
- `constraints.py` — `RebalanceConstraints` dataclass +
  `default_constraints()` factory; per-name caps, category caps,
  excluded categories, rebalance bands, `new_capital_chf`,
  `swiss_tax_filter`.
- `metrics/` — pure metric calculators consumed by both
  analysis and screening:
  - `risk.py` — Sharpe, Sortino, beta, alpha, IR
  - `ratios.py` — P/E, P/B, dividend yield, payout, etc.
  - `volatility.py` — annualised vol, rolling vol, VaR, CVaR

`shared` imports nothing from the other domains.

### `src/analysis/` — static portfolio analysis

Answers: *what do I own and how has it performed?* Everything
in this folder is descriptive / backward-looking.

- `core/` — frozen reference architecture (do not modify):
  - `transaction.py` — single-trade data model
  - `portfolio.py` — holdings tracking across the transaction log
  - `analyzer.py` — `PortfolioAnalyzer`. Phase 1 added
    `get_holdings_snapshot()` and `get_price_panel()` for the
    rebalancer + Monte Carlo engine.
- `loaders/` — Degiro + IBKR CSV parsers, generic
  `DataExporter` (xlsx writer used by every CLI).
- `plots/` — matplotlib dashboards (`PortfolioPlotter`)
- `report.py` — zero-arg orchestrator that loads broker CSVs,
  runs the chosen rebalance strategy, and writes a multi-sheet
  Excel report. Edit the locals at the top of `main()` to change
  strategy / capital / file paths.

### `src/modelling/` — forward-looking quant

Answers: *what might happen under different choices?*

- `rebalancing/` — Phase 2/4 deliverable. Data model:
  `RebalancingAction` (with `'CASH'` pseudo-action), `RebalancePlan`
  (with `total_invested_chf` and `summary()`). Orchestrator:
  `Rebalancer.propose(candidates=...)` which recycles REDUCE
  proceeds into the BUY pool and reserves `target_cash`.
  Strategies (auto-registered via `Strategy.__init_subclass__`):
  - `mvo`, `equal_weight`, `inverse_vol`, `min_variance`
    — portfolio-only.
  - `max_growth`, `risk_adjusted`, `min_risk` — screener-fed
    (consume `list[CandidateTicker]`). Defaults port S1/S2/S3
    from the legacy `construct_scenarios.py` with cash weights
    0% / 10% / 40%. No bond ETFs (per Swiss-tax memo).
  Pricing helper: `pricing.fetch_chf_price` FX-converts non-held
  candidate tickers to CHF for action sizing.
- `monte_carlo/` — Phase 5 deliverable. Replaces the 907-line
  legacy monolith.
  - `engine.py` — `MonteCarloEngine.simulate(plan, prices,
    horizon_days, initial_nav_chf) -> PathDistribution`. Clayton
    copula + Student-t marginals + James-Stein-shrunk loc +
    deterministic cash drift; optional IBKR cost model on
    initial execution and every rebalance; supports
    `none/monthly/quarterly/threshold` rebalance frequencies and
    monthly contributions.
  - `path_distribution.py` — `PathDistribution`: paths array
    plus query methods (percentiles, drawdowns, terminal stats,
    `to_summary_dict` with 22 fields including cost-drag bps,
    contribution-adjusted P&L mean/std, VaR/CVaR, gain/dd
    threshold probabilities).
  - `copula.py` — `sample_clayton` Marshall-Olkin sampler
    (pure function).
  - `shrinkage.py` — `james_stein_shrink` (pure function).
  - `cost_model.py` — `TransactionCostModel` +
    `ibkr_default_cost_model()`. Scalar `cost_chf` plus
    vectorised `cost_chf_array` for the hot path.
- `greeks/` — Black-Scholes option Greeks (delta, gamma, theta,
  vega, rho, `GreeksCalculator`). Interface unchanged through
  Phase 6.

`modelling/` may depend on `src.analysis.core` (`Transaction`,
`Portfolio`, `PortfolioAnalyzer`) — never on anything else in
`analysis/`.

### `src/screening/` — discovery

Answers: *what's worth looking at?*

- `listing.py` — `Listing` dataclass mirroring one row of the
  parquet cache (17 fundamental fields + 156 weekly closes).
- `universe.py` — `ExchangeUniverse` append-only shard parquet
  loader/writer (path: `data/screener_cache/<exchange>/<UTC>.parquet`).
- `criteria.py` — `Criterion` ABC + 9 concrete subclasses
  (momentum, cagr, sharpe, max_drawdown, pe_value, growth,
  liquidity, sector, swiss_tax). Self-registering via
  `Criterion.__init_subclass__`.
- `screener.py` — `Screener` orchestrator. Composite score =
  weighted mean of percentile ranks.
- `candidate.py` — `CandidateTicker` (Contract 2) +
  `load_candidates_from_json` for `report.py` to consume the
  CLI's JSON output.
- `cache_refresh.py` — resumable yfinance fetcher writing
  per-exchange parquet shards.

Self-contained: emits ranked `CandidateTicker` lists; the
rebalancer / report consume them.

### `src/user_scripts/` — operator-facing entry points

- `run_analysis_dashboard.py` — interactive analyser dashboard.
- `refresh_screener_cache.py` — populate
  `data/screener_cache/<exchange>/` shards from yfinance.
  CLI: `--exchange EXCHANGE_CODE` or `--all`, `--resume`,
  `--rate-limit 0.5`.
- `run_screener.py` — rank tickers across one or more cached
  exchanges. CLI: `--exchange ...`, `--criteria
  momentum,cagr,sharpe,...`, `--top-n N`, `--output-format
  excel|json`, `--output PATH`.
- `run_monte_carlo.py` — Monte Carlo simulation of a rebalance
  plan. Zero-arg `main()` with locals at the top — edit them to
  change strategy / horizon / n_paths / rebalance frequency /
  cost-on-off / contribution / candidates JSON / output path.
  Writes a 4-sheet Excel (`Plan`, `Summary`, `Percentiles`,
  `TerminalDistribution`).

The only layer permitted to import across domains.

## Boundary rules

1. `shared/` imports nothing from `analysis/`, `modelling/`,
   `screening/`.
2. Any domain may import from `shared/`.
3. `analysis/`, `modelling/`, and `screening/` do not import
   each other, except: `modelling/` may import
   `analysis.core.{transaction,portfolio,analyzer}` and
   `analysis.loaders` from the frozen reference surface.
4. `src/analysis/report.py` and `src/user_scripts/*.py` are
   the only modules allowed to import from two or more
   domains in a single file.

Verify with:

```bash
grep -rn "from src.analysis"   src/modelling src/screening src/shared
grep -rn "from src.modelling"  src/screening src/shared
grep -rn "from src.screening"  src/analysis  src/modelling src/shared
```

## Quick start

### Analyse current portfolio

```python
from src.analysis.core import PortfolioAnalyzer

analyzer = PortfolioAnalyzer(
    degiro_csv_file_path='data/postprocess_data/processed_portfolio.csv',
    ibkr_csv_file_path='reports/ibkr/<YYYYMMDD>_<YYYYMMDD>_<ACCOUNT_ID>.csv',
    base_currency='CHF')
print(analyzer.get_summary('SPY'))
analyzer.plot_dashboard(benchmark='SPY')
```

### Write a rebalance Excel report

```bash
# Edit src/analysis/report.py: change `strategy = 'mvo'` (or
# 'max_growth' / 'risk_adjusted' / 'min_risk') and
# `candidates_from` to a screener JSON path if using a
# screener-fed strategy.
python -m src.analysis.report
```

### Refresh the screener cache

```bash
# Single exchange (~30 min for SIX, ~6-12 h for the full
# 16-exchange sweep under yfinance throttling).
python -m src.user_scripts.refresh_screener_cache \
    --exchange SIX --resume
```

Supported exchange codes (16):
`NYSE NASDAQ LSE SIX XETRA EURONEXT_PARIS EURONEXT_AMSTERDAM
EURONEXT_BRUSSELS EURONEXT_LISBON EURONEXT_DUBLIN BIT BME STO
OSL CPH HEL`.

Refresh cadence: monthly is enough for fundamentals
(FRED-published OECD rates update monthly anyway); weekly close
series benefit from a weekly refresh during active trading.

### Rank candidate tickers

```bash
python -m src.user_scripts.run_screener \
    --exchange NASDAQ NYSE \
    --criteria momentum,cagr,sharpe,max_drawdown,swiss_tax \
    --top-n 50 \
    --output-format json \
    --output results/screener_us.json
```

### Run Monte Carlo

```bash
# Edit src/user_scripts/run_monte_carlo.py: set strategy,
# horizon_days, n_paths, rebalance_frequency, candidates_from,
# etc. — then run.
python -m src.user_scripts.run_monte_carlo
```

### Compute option Greeks

```python
from src.modelling.greeks import GreeksCalculator

calc = GreeksCalculator()
greeks = calc.calculate_all_greeks(
    spot_price=100, strike_price=105, time_to_expiry=0.25,
    volatility=0.20, risk_free_rate=0.05, option_type='call')
print(greeks)
```

## Other top-level folders

- `data/` — broker exports, screener cache parquet shards,
  exchange ticker lists (inputs)
- `reports/` — raw broker statements (IBKR / Degiro CSV, PDF)
- `results/` — analyzer / modeller outputs (.xlsx, .png, .json)
- `docs/` — strategy notes, `docs/theory/` for finance theory,
  `docs/superpowers/{specs,plans}/` for refactor history
- `refs/` — finance papers and reference screenshots
- `tasks/` — refactor progress log (`todo.md`) and lessons
- `dashboard/` — separate Vite/React frontend (decoupled)
- `API/` — IB gateway / TWS installers
- `_OLD/` — deprecated legacy code kept for reference; not
  imported by anything in `src/`
