# Refactor progress

## Phase 1 — Shared infrastructure: COMPLETE (2026-05-14)

- src/shared/risk_free_rate.py     [new]
- src/shared/constraints.py        [new]
- src/shared/__init__.py           [+5 exports: get_live_risk_free_rate,
                                    fred_series_id_default, fallback_rate_pct,
                                    RebalanceConstraints, default_constraints]
- src/analysis/core/analyzer.py    [+get_holdings_snapshot,
                                    +get_price_panel]
- tests/                           [scaffolded; 27 unit tests, all passing]

Verification:
- 27 unit tests pass via `pytest tests/ -v`
- Live RFR fetch returns a real number (Swiss long-term rate from FRED)
- Analyzer get_holdings_snapshot + get_price_panel produce non-empty
  DataFrames against real CSV fixtures from data/ and reports/

Spec: docs/superpowers/specs/2026-05-13-repo-refactor-design.md
Plan: docs/superpowers/plans/2026-05-14-phase-1-shared-infrastructure.md

Next: Phase 2 — Rebalancing domain + rewritten src/analysis/report.py.
Plan to be authored by re-invoking superpowers:writing-plans against
docs/superpowers/specs/2026-05-13-repo-refactor-design.md, scoped to
Phase 2.

## Phase 2 — Rebalancing domain + rewritten report.py: COMPLETE (2026-05-14)

- src/modelling/rebalancing/                    [new package]
  - action.py    (RebalancingAction)
  - plan.py      (RebalancePlan)
  - rebalancer.py
  - strategies/  (base, mvo, equal_weight, inverse_vol, min_variance)
- src/analysis/report.py                        [rewritten as CLI orchestrator]
- src/modelling/rebalancing.py (1064-line legacy) [DELETED]

Verification:
- 64 unit tests pass via `pytest tests/ -v`
- Strategy registry exposes 4 strategies: mvo, equal_weight, inverse_vol,
  min_variance.
- `python -m src.analysis.report --strategy <name>` end-to-end against fake
  data writes a 4-sheet xlsx (Holdings, Target Weights, Rebalancing Actions,
  Summary).

Notable improvements over the spec:
- RebalancePlan validates ticker-set match between target_weights and
  actions, plus per-ticker weight consistency, beyond the basic sum-to-1
  check originally specified.
- filter_eligible (no leading underscore) is a clean cross-module helper
  in equal_weight.py rather than the originally specified _filter_eligible.

Spec: docs/superpowers/specs/2026-05-13-repo-refactor-design.md
Plan: docs/superpowers/plans/2026-05-14-phase-2-rebalancing-and-report.md

Next: Phase 3 — Screening domain (per-exchange parquet caches + screener).
Plan to be authored by re-invoking superpowers:writing-plans against the
same spec, scoped to Phase 3.

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
- 120 unit tests pass via `pytest tests/ -v`.
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

## Phase 4 — Screener-fed strategies + rebalancer extension: COMPLETE (2026-05-14)

- src/screening/candidate.py                [+load_candidates_from_json]
- src/screening/__init__.py                 [+re-export]
- src/modelling/rebalancing/action.py       [+'CASH' action type;
                                             +cash_ticker constant;
                                             extended shares==0
                                             invariant to CASH]
- src/modelling/rebalancing/plan.py         [summary() +total_cash_chf;
                                             hoisted Counter import]
- src/modelling/rebalancing/pricing.py      [new — fetch_chf_price
                                             FX-aware CHF lookup for
                                             non-held tickers]
- src/modelling/rebalancing/rebalancer.py   [non-held target tickers
                                             priced via candidate
                                             lookup; CASH bypasses
                                             BUY/REDUCE/HOLD passes;
                                             BUY pool now
                                             max(0, new_capital
                                             + total_reduce
                                             - target_cash);
                                             +_build_candidate_prices
                                             and _make_cash_action
                                             helpers]
- src/modelling/rebalancing/strategies/
  - max_growth.py     (MaxGrowthStrategy, S1 default cash 0%)
  - risk_adjusted.py  (RiskAdjustedStrategy, S2 default cash 10%,
                       per-name cap 15% with spill-to-holds when
                       cap infeasible)
  - min_risk.py       (MinRiskStrategy, S3 default cash 40% with
                       12/12/36 ETF/lowvol/holds; CSSPX.SW anchor;
                       no bonds per Swiss-tax memo)
  - __init__.py       [+3 imports; auto-registers new strategies]
- src/analysis/report.py                    [main() stays zero-arg;
                                             extracted main_with_inputs
                                             with candidates_from kwarg;
                                             loads via
                                             load_candidates_from_json]
- src/orchestration/                        [DELETED — collapsed
                                             into report.py]
- tests/                                    [+12 task-spec tests +
                                             integration test for
                                             report.main_with_inputs]

Verification:
- 169 unit tests pass via `python3 -m pytest tests/ -v`.
- Strategy registry exposes 7 strategies: mvo, equal_weight,
  inverse_vol, min_variance, max_growth, risk_adjusted, min_risk.
- RebalancingAction.valid_actions = (BUY, REDUCE, HOLD, CASH);
  CASH inherits shares==0 invariant.
- BUY pool invariant test confirms total_buy <= new_capital
  + total_reduce - total_cash post-Phase-4 (the Phase-2
  total_buy <= new_capital invariant was deliberately
  superseded — see test_rebalancer_buy_pool_includes_
  reduce_proceeds_minus_cash).
- Integration test (test_report_main_with_max_growth_candidates)
  exercises the full screener-JSON -> Rebalancer -> Excel pipe
  with a monkeypatched data provider and analyzer.

Known limitation (documented):
- CASH action's est_cost_chf records the target cash CHF.
  Realised post-trade cash matches the target modulo
  whole-share rounding; large discrepancies should not occur
  given the new recycling BUY pool, but tiny residuals from
  the int(alloc/price) rounding remain.

Architectural notes:
- cash_ticker = 'CASH' lives in action.py (not rebalancer.py)
  so strategies can import it without creating a downward
  dependency on the orchestrator.
- The screener-fed strategies (max_growth, risk_adjusted,
  min_risk) each accept an optional cash_weight kwarg defaulting
  to S1/S2/S3 legacy values (0.0 / 0.10 / 0.40).
- The defensive `metrics.get('swiss_tax', 1.0)` lookup in all
  three new strategies is the documented exception to the
  no-silent-defaults rule (per feedback_no_silent_defaults.md):
  the screener's SwissTaxCriterion is upstream; a 0.0 sentinel
  in metrics signals a fail and drops the candidate.

Spec: docs/superpowers/specs/2026-05-13-repo-refactor-design.md
Plan: docs/superpowers/plans/2026-05-14-phase-4-screener-fed-strategies.md

Next: Phase 5 — Monte Carlo refactor. New
src/modelling/monte_carlo/ package replacing the 907-line
legacy src/modelling/monte_carlo.py; PathDistribution,
Clayton copula, James-Stein shrinkage, IBKR
TransactionCostModel, MonteCarloEngine, new
src/user_scripts/run_monte_carlo.py CLI. See spec §7
Phase 5 for the full deliverable list.

## Phase 5 — Monte Carlo refactor: COMPLETE (2026-05-14)

- src/modelling/monte_carlo/                   [new package]
  - __init__.py     (re-exports the 6 public symbols)
  - path_distribution.py  (PathDistribution — Contract 6;
                           percentiles, drawdown, terminal,
                           prob_below, expected_return,
                           summary_dict with cost-drag fields)
  - copula.py       (sample_clayton — Marshall-Olkin algo,
                     pure function, frailty-based sampling)
  - shrinkage.py    (james_stein_shrink — pure function,
                     shrinks daily means toward ERP anchor)
  - cost_model.py   (TransactionCostModel + IBKR Swiss-resident
                     defaults; cost_chf scalar + cost_chf_array
                     vectorised; ibkr_default_cost_model factory)
  - engine.py       (MonteCarloEngine — Clayton-copula +
                     Student-t marginals + JS-shrunk loc +
                     monthly/quarterly/threshold rebalancing +
                     IBKR cost model; consumes RebalancePlan
                     directly including the CASH pseudo-ticker)
- src/user_scripts/run_monte_carlo.py          [new CLI;
                     argparse with --strategy, --horizon-days,
                     --n-paths, --rebalance, --no-costs,
                     --candidates-from, --random-seed, etc.;
                     writes 4-sheet Excel: Plan, Summary,
                     Percentiles, TerminalDistribution]
- src/modelling/monte_carlo.py (907-line legacy)  [DELETED]
- tests/                                       [+40 task-spec
                                                 tests + 2 CLI
                                                 smoke tests]

Verification:
- 209 unit tests pass via `python3 -m pytest tests/ -v`.
- Engine consumes a Phase-4 RebalancePlan directly. CASH entry
  in target_weights becomes a deterministic cash-rate bucket;
  non-CASH tickers go through the Clayton-copula + Student-t
  pipeline.
- BUY pool semantics inherited from Phase 4 (recycle REDUCE
  proceeds, respect target_cash reservation) are not relevant
  at simulation time — the engine starts from
  initial_nav_chf and runs forward, allocating to the plan's
  target weights at t=0.
- Cost drag fields (cumulative_cost_chf_p50/p95,
  annual_cost_drag_bps) are populated when cost_model is
  enabled; --no-costs run produces strictly higher mean
  terminal NAV at the same seed.
- Threshold rebalancing uses uniform-day trigger (any-path
  drift breaches → all paths rebalance that day) per spec
  guidance.

Architectural notes:
- sample_clayton and james_stein_shrink are pure functions
  with no module-level state, making them independently
  testable and reusable.
- TransactionCostModel.cost_chf_array vectorises the cost
  calculation across path arrays — replaces the original
  per-path Python loop in the engine's rebalance hot path.
  Eliminates 100K Python calls per rebalance event at
  n_paths=10K.
- Engine constructor accepts ticker_currencies dict so
  the CLI can populate per-ticker currency (used to compute
  FX cost). Falls back to 'USD' for unmapped tickers
  (documented exception to no-silent-defaults).
- Initial-execution cost is deducted from cash at t=0;
  cumulative_costs_chf is seeded with the (deterministic
  across paths) t0 cost.

Known limitation:
- run_monte_carlo CLI does not extend the price panel with
  candidate-ticker columns. Screener-fed runs require the
  candidate tickers to already be in the held portfolio
  (so they appear in analyzer.get_price_panel()). Extending
  prices for net-new candidate tickers is deferred to a
  future phase (would require adding a method to the
  analyzer, which is frozen reference per HANDOFF.md).

Spec: docs/superpowers/specs/2026-05-13-repo-refactor-design.md
Plan: docs/superpowers/plans/2026-05-14-phase-5-monte-carlo.md

Next: Phase 6 — Greeks code-style pass.
src/modelling/greeks/{delta,gamma,vega,rho,theta,
greeks_calculator}.py get the Phase 4/5 style treatment
(line width, no banner separators, lowercase module
constants); local RISK_FREE_RATE constants get replaced
with calls to src.shared.risk_free_rate.get_live_risk_free_rate.
No interface changes, no new tests. See spec §7 Phase 6.

## RFR units correction (2026-05-14, pre-Phase-6)

User-driven fix: src.shared.risk_free_rate.get_live_risk_free_rate
now returns a DECIMAL fraction (e.g. 0.004 for 0.4%) instead of
the prior "annual rate in percent" contract. Single source of
truth for consumers that need to multiply by NAV or feed Sharpe
formulas directly.

- src/shared/risk_free_rate.py: fallback_rate_pct (0.5)
  renamed and rebased to fallback_rate_decimal (0.005). FRED
  fetch divides raw percent by percent_per_unit=100.0 before
  returning. Docstring updated.
- src/shared/__init__.py: re-export renamed to
  fallback_rate_decimal.
- src/modelling/rebalancing/strategies/mvo.py: dropped the
  rf_decimal = risk_free_rate / 100.0 conversion; uses the
  decimal directly. Docstring updated.
- src/modelling/rebalancing/strategies/base.py: docstring for
  risk_free_rate ABC arg updated ("Annual rate as a decimal
  fraction").
- tests/shared/test_risk_free_rate.py: 5 tests updated to expect
  decimal values (e.g. FRED's 0.61 -> assertion 0.0061; fallback
  asserted against fallback_rate_decimal).
- tests/modelling/rebalancing/test_strategies.py: 4 mvo tests
  updated from risk_free_rate=0.5 to risk_free_rate=0.005
  (preserves the original 0.5% intent under the new contract).
  fake_rfr fixture in test_mvo_uses_live_rfr_when_none now
  returns 0.005.
- No consumer code in src/modelling/monte_carlo/ touched: the
  engine uses cash_rate_annual (already decimal) for the cash
  drift and does not call get_live_risk_free_rate. The
  src.shared.metrics.risk module did not invoke the function
  directly either.

Verification:
- 217 unit tests pass via `python3 -m pytest tests/ -v`.

## Phase 6 — Greeks code-style pass: COMPLETE-NO-OP (2026-05-14)

Audit verdict: the greeks files were already at Phase 4/5
style. No source edits were required; the phase closes by
inspection.

Audit checklist against HANDOFF.md §8 Phase 6 directives:
- Line-width compliance: PASS (no line in any greeks file
  exceeds 80 chars).
- Header separators: PASS (no banner lines present —
  no `# ===`, `# ~~~`, `# ---`).
- Lowercase module-level constants: PASS (no module-level
  constants exist in any greeks file; vacuously satisfied).
- Replace local RFR/spot constants with calls to
  src.shared.risk_free_rate.get_live_risk_free_rate(): PASS
  (no local RISK_FREE_RATE / RFR constants exist in any
  greeks file; the functions take risk_free_rate as a
  required parameter).

Other Phase 4/5 style checks:
- Type hints in signatures: PASS.
- Single quotes for strings, double for docstrings: PASS.
- No __author__ / __credits__ / __status__ blocks: PASS.
- Plain three-group imports: PASS.

Verification:
- 217 unit tests pass via `python3 -m pytest tests/ -v`.
- src/user_scripts/examples/options_greeks_example.py runs
  end-to-end: call/put greeks computed via both
  GreeksCalculator and the standalone functions; output
  matches pre-Phase-6 values (no behaviour changes).

Files inspected but not modified:
- src/modelling/greeks/__init__.py (45 lines)
- src/modelling/greeks/delta.py (109 lines)
- src/modelling/greeks/gamma.py (90 lines)
- src/modelling/greeks/greeks_calculator.py (404 lines)
- src/modelling/greeks/rho.py (130 lines)
- src/modelling/greeks/theta.py (131 lines)
- src/modelling/greeks/vega.py (110 lines)

Deferred (offered as Scope B during phase planning,
declined by user):
- Adding optional `risk_free_rate: Optional[float] = None`
  defaults on the greek functions with lazy-fetch via
  get_live_risk_free_rate(). Pure additive change — would
  match the MVOStrategy pattern. Skipped because the user
  chose Scope A (literal no-op closure).

Spec: docs/superpowers/specs/2026-05-13-repo-refactor-design.md
Plan: (no plan file authored — closed on audit; see
docs/superpowers/specs/2026-05-13-repo-refactor-design.md §7
Phase 6 for the original directive).

Next: Phase 7 — Cleanup + docs. Final phase. Delete any
_OLD/ stragglers from active source tree, update README.md
with the new entry-point list (run_screener, run_monte_carlo,
report.py), grep-audit cross-domain imports per the rule
"only src/analysis/report.py and src/user_scripts/*.py may
import from multiple src/<domain>/ packages", and append the
final tasks/todo.md block confirming refactor closure.
See spec §7 Phase 7.

## Phase 7 — Cleanup + docs: COMPLETE (2026-05-14)

Scope deviation accepted by user: the spec's "Delete _OLD/
stragglers if any remain in the active source tree" directive
is dropped. The _OLD/ subdirectory at the repo root is a
deliberate archive of deprecated code kept for reference; it
must NOT be deleted. The grep audit (below) confirms nothing
in src/ imports from _OLD/, so retention is harmless.

Deliverables:

- README.md rewritten to reflect the post-Phase-1-6 module
  layout:
  - shared/ now includes risk_free_rate.py (decimal contract)
    and constraints.py.
  - analysis/core/ is documented as the frozen reference
    surface; analysis/report.py is the zero-arg orchestrator.
  - modelling/ documents the rebalancing/ (Phase 2/4),
    monte_carlo/ (Phase 5), and greeks/ packages with their
    public surface and Phase-by-Phase deliverables.
  - screening/ documents the rebuilt Listing /
    ExchangeUniverse / Screener / 9-criterion stack.
  - user_scripts/ documents the 4 operator CLIs.
  - Boundary rules table updated: src/analysis/report.py and
    src/user_scripts/*.py are the only modules permitted to
    cross domains (orchestration/ collapsed in Phase 4).
  - Exchange code list (16) documented.
  - Refresh cadence note added (monthly for fundamentals;
    weekly for close series during active trading).
  - Quick-start sections for analyse / report / refresh /
    screen / monte carlo / greeks.

- Cross-domain import audit performed via three grep passes
  against src/{modelling,screening,shared,analysis} (excluding
  user_scripts and report.py per rule 4):
  - from src.analysis -> modelling/screening/shared: zero hits.
  - from src.modelling -> analysis/screening/shared: one hit
    in src/analysis/report.py (allowed).
  - from src.screening -> analysis/modelling/shared: one hit
    in src/analysis/report.py (allowed).
  Audit result: CLEAN. The two report.py hits are sanctioned
  by rule 4 (orchestrator + user_scripts may cross domains);
  every other domain stays in its lane.

- 217 unit tests pass via `python3 -m pytest tests/ -v`.

Spec: docs/superpowers/specs/2026-05-13-repo-refactor-design.md
Plan: (no plan file — closed inline after user override on the
_OLD/ deletion directive).

Refactor closure
================

All 7 phases of the 2026-05-13 design spec land. Test count
trajectory:
- Phase 1 ship:        27
- Phase 2 ship:        64
- Phase 3 ship:       120
- Phase 4 ship:       169
- Phase 5 ship:       209
- RFR-decimal fix:    217 (no count change; 5 tests rewritten)
- Phase 6 close:      217 (no-op)
- Phase 7 close:      217 (no test changes)

Surface change summary, by phase:
- Phase 1: src/shared/risk_free_rate.py +
  src/shared/constraints.py + 2 additive methods on
  PortfolioAnalyzer.
- Phase 2: src/modelling/rebalancing/ package replacing the
  1064-line legacy src/modelling/rebalancing.py monolith.
  4 portfolio-only strategies. Rewritten src/analysis/report.py.
- Phase 3: src/screening/ rebuild around per-exchange parquet
  shards. 9 criteria, 2 CLIs, resumable cache.
- Phase 4: 3 screener-fed strategies (max_growth /
  risk_adjusted / min_risk), CASH pseudo-ticker action type,
  REDUCE-recycling BUY pool. src/orchestration/ removed.
- Phase 5: src/modelling/monte_carlo/ package replacing the
  907-line legacy monolith. PathDistribution, Clayton sampler,
  JS shrinkage, IBKR TransactionCostModel, MonteCarloEngine,
  run_monte_carlo CLI. Contribution-adjusted P&L summary added
  on user request.
- Phase 6: greeks code-style audit (no-op). RFR units
  correction landed as a pre-Phase-6 fix (function returns
  decimal, not percent).
- Phase 7: README.md + boundary audit.

The HANDOFF.md document at docs/superpowers/HANDOFF.md remains
useful as a historical reference for the in-flight design
decisions; future sessions can read it for context on the
"frozen surface" (analysis/core/) and the auto-registration
patterns used in rebalancing/strategies/ and screening/criteria/.

No commits performed across any phase. Every change is in the
working tree; user commits manually.

## Phase 8 — DefeatBetaProvider integration: COMPLETE (2026-05-15)

User-driven: integrate the defeat-beta/defeatbeta-api package
(Hugging Face mirror of Yahoo Finance, no auth, no rate
limits) as a third DataProvider. Use it as the primary backend
for the US screener cache (NYSE / NASDAQ) — collapses the
6-12h yfinance throttled fetch to ~minutes via a single bulk
parquet read. Other 14 exchanges keep yfinance.

- defeatbeta-api 0.0.53 added as a runtime dependency.
  Pandas pinned to <3.0 to keep pandas-datareader (FRED RFR
  fetch) compatible — defeatbeta works on pandas 2.x in
  practice despite its declared `pandas>=3.0.1`.

- src/shared/data_provider.py: appended
  `DefeatBetaProvider(DataProvider)` with US-only routing
  (`is_native(t)` = no `.` suffix and no `=X` suffix) and
  optional `fallback: DataProvider` for non-US tickers.
  All 6 abstract methods implemented. `get_fundamental_data`
  returns a 30-key envelope (17 original + 13 new). Per-field
  `_safe_last(method, column)` and `_safe_dcf(ticker)` helpers
  wrap defeatbeta calls and return NaN on coverage gaps —
  documented exception to the no-silent-defaults rule.
  Aliased qf_lib's `Ticker` import to `QFTicker` to avoid the
  name collision with defeatbeta's `Ticker` (so test
  monkeypatches resolve to the right symbol).

- src/screening/listing.py: extended
  `fundamental_field_names` from 17 to 30 entries. New fields:
  `peg_ratio`, `roic`, `roce`, `wacc`, `equity_multiplier`,
  `asset_turnover`, `enterprise_value`,
  `enterprise_to_revenue`, `enterprise_to_ebitda`,
  `ttm_revenue`, `ebitda_growth_yoy`, `market_cap_chf`,
  `dcf_implied_upside`. Backward-compat: legacy 17-column
  parquet shards still load via the `_row_to_listing`
  NaN-default path in src/screening/universe.py. The yfinance
  cache path `setdefault`-fills the new fields with NaN; only
  the defeatbeta path populates them with real values.

- src/screening/cache_refresh.py: new module constants
  `us_exchanges = ('NYSE', 'NASDAQ')` and
  `defeatbeta_history_weeks = 160`. `refresh_exchange()`
  becomes a 2-line dispatch: US exchanges call
  `_refresh_via_defeatbeta(...)` (new bulk path); other
  exchanges call `_refresh_via_yfinance(...)` (the previous
  body, extracted verbatim, no behaviour change).
  `_build_listing_via_defeatbeta(ticker, exchange, provider)`
  fetches the 30-field envelope + daily prices, resamples to
  156 weekly bars, hardcodes `currency='USD'` (defeatbeta
  coverage), pulls sector from `provider._ticker(t).info()`
  with 'Unknown' fallback, takes `market_cap` from
  `fundamentals['market_cap_chf']` (USD numeric value despite
  the field name's `_chf` suffix — a Phase 8 naming quirk
  documented in the docstring).

- DcfCriterion was added in Task 4 then REVERTED on user
  review. Reason: yfinance does not expose a DCF method, so
  the `dcf_implied_upside` field has no fallback when
  defeatbeta returns NaN (which it does for many US tickers
  whose `Ticker.dcf_data()` doesn't expose `'implied_upside'`).
  Returning None from the criterion would have hard-dropped
  those tickers from the entire Screener rank — wrong
  behaviour, since DCF is one signal among many. Implementing
  a real DCF computation locally (Gordon-growth from FCF +
  growth + WACC) was deemed out of scope. Net: criteria.py
  stays at 9 entries (the original Phase 3 set);
  `dcf_implied_upside` is still populated in the Listing
  cache by the defeatbeta path for ad-hoc downstream use, but
  no auto-registered Criterion consumes it.

Verification:
- 230 unit tests pass via `python3 -m pytest tests/ -v`
  (was 217 baseline; +9 DefeatBetaProvider + +2 Listing
  schema + +2 cache_refresh dispatch; DcfCriterion's 3
  tests deleted with the revert).
- Live smoke: `DefeatBetaProvider().get_fundamental_data('NVDA')`
  returns peg_ratio=0.45, roic=0.2927, wacc=0.2325 (real
  values). dcf_implied_upside=NaN for NVDA — defeatbeta's
  `dcf_data()` doesn't always expose 'implied_upside' for
  every ticker; the criterion gracefully drops such tickers
  from the rank.
- Cross-domain audit unchanged from Phase 7 closure (no new
  cross-domain imports introduced).

Architectural notes:
- `DefeatBetaProvider` is opt-in. The default factory
  `_build_data_provider()` in `report.py` and
  `run_monte_carlo.py` still returns `YFinanceProvider()`.
  Operators flip in `DefeatBetaProvider(fallback=...)`
  manually when they want US-tickers via the bulk parquet.
- The `cache_refresh` dispatch is automatic: any call with
  `exchange in ('NYSE', 'NASDAQ')` goes through defeatbeta
  with no extra config. No CLI flag needed.

Known limitation (acknowledged):
- `market_cap_chf` field carries USD values on the defeatbeta
  path (defeatbeta only covers US listings). Field naming is
  retained for cross-currency uniformity in the Listing
  schema; documented in
  `_build_listing_via_defeatbeta`'s docstring.
- defeatbeta does not cover Swiss / European / UK / Canadian
  native exchanges (NESN.SW, BARC.L, FP.PA, SHOP.TO, etc.).
  Foreign issuers appear only as US ADRs (e.g. BABA, PDD).
  The other 14 exchanges in the screener config keep yfinance
  per-ticker fetching (multi-hour cache refresh).
- defeatbeta FX is `<CCY>=X` (vs USD); cross-rates like
  EURCHF must be derived via USD intermediary. The provider
  delegates `=X` tickers to the fallback (yfinance, which
  exposes `EURCHF=X` directly).

Spec / plan: docs/superpowers/plans/2026-05-15-phase-8-defeatbeta-provider.md
Reference: github.com/defeat-beta/defeatbeta-api +
huggingface.co/datasets/defeatbeta/yahoo-finance-data

Next: future phase TBD (operator-defined). The 7-phase
refactor (Phases 1-7) is fully closed; Phase 8 is an
additive integration. Possible future work:
- Replace `src/shared/risk_free_rate.py` FRED fetch for the
  US Treasury yield path with defeatbeta's
  `daily_treasury_yield.parquet` (Swiss long-term rate stays
  on FRED).
- Extend `cache_refresh` to use defeatbeta's `Tickers([list])`
  bulk constructor for true single-shot parquet reads
  (currently per-ticker calls amortise the parquet download
  via DuckDB cache, but the explicit batch API may be faster
  on cold starts).
- Add a screener-fed strategy that uses `peg_ratio`, `roic`,
  or DCF upside as its picking signal (currently the
  composite_score-based strategies don't directly leverage
  the new fundamentals).
