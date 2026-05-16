# Handoff: stock_market refactor — Phase 4 onwards

**Working directory:** `/Users/rbarreira/Desktop/stock_market/`
**Repo state:** under git, staged but uncommitted; **do not commit unless the user explicitly asks**.
**Date of handoff:** 2026-05-14.
**User email on profile:** `rui_pinto@brown.edu`.

This document gives a new agent everything needed to continue the refactor from Phase 4 through Phase 7. Read it end-to-end before doing anything.

---

## 1. Current state: where we are

Phases 1, 2, 3 of the refactor are **complete**. All approved end-to-end.

| Phase | Scope | Status |
|------:|-------|--------|
| 1 | Shared infrastructure: `risk_free_rate.py`, `constraints.py`, two new methods on `PortfolioAnalyzer` (`get_holdings_snapshot`, `get_price_panel`), test scaffolding. | ✅ |
| 2 | Rebalancing domain: `RebalancingAction`, `RebalancePlan`, `Strategy` ABC + auto-register registry, 4 strategies (mvo, equal_weight, inverse_vol, min_variance), `Rebalancer` orchestrator; rewritten `src/analysis/report.py`; deleted 1064-line legacy `src/modelling/rebalancing.py`. | ✅ |
| 3 | Screening domain: `CandidateTicker`, `Listing`, `ExchangeUniverse` (now append-only shard parquets), 9 criteria (momentum, cagr, sharpe, max_drawdown, pe_value, growth, liquidity, sector, swiss_tax), `Screener` orchestrator, resumable `cache_refresh`; two CLIs (`refresh_screener_cache.py`, `run_screener.py`); deleted 6 legacy modules. | ✅ |

**Test suite: 122 passing.** Verify with:
```
cd /Users/rbarreira/Desktop/stock_market && python3 -m pytest tests/ -v
```

`tasks/todo.md` carries the per-phase completion blocks (additive).

---

## 2. Immediate next action: Phase 4

Re-invoke `superpowers:writing-plans` against the source spec scoped to Phase 4. Then execute via `superpowers:subagent-driven-development`.

**Phase 4 deliverables** (per spec Section 7):

1. **New screener-fed strategies** in `src/modelling/rebalancing/strategies/`:
   - `MaxGrowthStrategy` (consumes `candidates: List[CandidateTicker]`, picks top-N by composite_score; equal-weight or composite-weighted).
   - `RiskAdjustedStrategy` (top-Sharpe picks, inverse-vol weighting, 15% per-name cap).
   - `MinRiskStrategy` (low-vol picks, broad-ETF weighting; preserves the no-bond Swiss-tax rule).
   - Each strategy registers via `Strategy.__init_subclass__` so adding the file is the only change needed in `strategies/__init__.py`.
   - Ported from legacy `src/modelling/rebalancing.py` (already deleted) — recover logic from spec or git diff if needed.

2. **Edit `src/analysis/report.py`**: add a `candidates_from` local variable in `main()` (path to a JSON file output by `run_screener.py`), load and pass to `Rebalancer.propose(candidates=...)` when the chosen strategy needs it. **Do not reintroduce argparse** — `main()` remains zero-arg with local-variable inputs.

3. **Delete `src/orchestration/`** entirely (its purpose collapses into report.py).

4. Tests: TDD per strategy + an integration test confirming `report.main()` runs with a screener-fed strategy against a JSON fixture.

---

## 3. Workflow conventions

### Skill stack (Superpowers plugin)

You should use these skills in order:
1. `superpowers:writing-plans` — write the per-phase implementation plan into `docs/superpowers/plans/YYYY-MM-DD-phase-N-<topic>.md`. Each task is bite-sized with TDD red→green→commit steps.
2. `superpowers:subagent-driven-development` — execute the plan: dispatch a fresh implementer subagent per task, then run **two-stage review** (spec compliance, then code quality) before marking complete. Continuous execution: do not pause between tasks unless something blocks or the user intervenes.

Skip `superpowers:using-git-worktrees` (the project lives at the canonical path; no worktree).

### Per-task discipline

1. Dispatch implementer with the FULL task text + scene-setting context — do not require the subagent to read the plan file.
2. Implementer follows TDD: failing test → minimal impl → green.
3. Dispatch spec reviewer subagent — verify spec compliance independently.
4. Dispatch code quality reviewer — must be APPROVED before moving on.
5. Mark task complete in TodoWrite.
6. **Never commit** unless the user explicitly asks. Plan-level "commit (advisory)" steps are not authorizations.

### Repo is under git

`git init` was run earlier; `src/` and `docs/` are staged but never committed. Continue this convention. If the user asks for a commit, follow `~/.claude/CLAUDE.md`'s git protocol (Co-Authored-By footer included).

---

## 4. Style conventions (MATCH `src/analysis/core/portfolio.py`)

This codebase **does not** follow the global style guide at `~/.claude/rules/code-style.md`. The user explicitly directed all refactored code to match the style of `src/analysis/core/portfolio.py` and `analyzer.py`. Specifically:

- **No** decorative `# =================` major-section separators.
- **No** decorative `# ~~~~~~~~~~~~~~` subsection separators inside method bodies.
- **No** decorative `# -----------------` between methods.
- **No** `__author__` / `__credits__` / `__status__` blocks.
- **No** `# Modules` / `# Standard` / `# Third-party` / `# Local` banner comments above import groups.
- **Imports** plain, grouped (stdlib / third-party / local), separated by blank lines.
- **Type hints** in signatures (`def foo(x: int) -> str:`).
- **Docstrings** Google-style (`Args:`, `Returns:`, `Raises:`, `Example:`), triple-double-quote.
- **Single quotes** for strings; double quotes only for docstrings.
- **80-char max line width.**
- **snake_case** variables and functions; **PascalCase** classes.
- **Module-level constants in lowercase** (per `feedback_no_caps_globals.md` memory note — overrides PEP 8).

If a reviewer subagent complains that type hints / Google docstrings violate the user's global rules, **ignore** — they are looking at the wrong source of truth.

---

## 5. User preferences (read carefully)

### Absolute mode (from `~/.claude/CLAUDE.md`)

The user prefers blunt, directive responses:
- No filler, no hype, no soft asks, no conversational transitions.
- No "Would you like me to...?" prompts.
- Terminate after delivering info — no closures.
- Prioritize cognitive rebuilding, not tone-matching.

### Memory notes (load and respect)

Located at `/Users/rbarreira/.claude/projects/-Users-rbarreira-Desktop-stock-market/memory/MEMORY.md` and linked files:

- **feedback_no_caps_globals.md** — module-level and class attributes use lowercase, NOT UPPER_CASE. Overrides PEP 8.
- **feedback_no_silent_defaults.md** — don't use `.get(key, default)` to silence missing-key errors. Raise or let `KeyError` propagate unless the default is semantically correct (e.g., yfinance dict variability where missing → NaN is documented).
- **feedback_swiss_tax_no_bonds.md** — no bond ETFs in any default policy. Use CHF cash + low-vol equity. `SwissTaxCriterion` enforces this for the screener.
- **user_investor_profile.md** — Swiss tax-resident investor, IBKR + Degiro brokers (IBKR primary). Monthly CHF 2000 deployment. CHF base currency.

### Repo / data layout

- **Frozen reference:** `src/analysis/core/{transaction,portfolio,analyzer}.py` — DO NOT modify (additive method changes in Phase 1 are the only exception, already done).
- **Light-edit only:** `src/analysis/loaders/`, `src/analysis/plots/`, `src/shared/data_provider.py`, `src/shared/symbol_map.py`, `src/shared/metrics/`.
- Data assets at `data/` (portfolio CSVs, ticker_categories.json, screener_cache/ for parquet shards, exchange_tickers/ for ticker lists).
- IBKR statements at `reports/ibkr/`.

---

## 6. Architectural decisions worth knowing

These were established and approved during Phases 1-3; **do not relitigate**:

1. **Domain shape mirrors `src/analysis/core/`:** data model → state aggregator → orchestrator. Every new domain (rebalancing, screening, monte_carlo) follows this triad.
2. **Strategy / Criterion auto-registration** via `__init_subclass__`. Adding a new strategy or criterion is one new module + one import in the package `__init__.py` (the import triggers registration).
3. **`RebalancePlan.__post_init__`** validates 4 invariants: target_weights sum to 1.0 ± `target_weights_sum_tol`; ticker sets of `target_weights` and `actions` match; no duplicate ticker in actions; per-ticker `target_wt_pct` matches `target_weights[t] * 100` within tolerance.
4. **Screener composite score** = weighted mean of percentile ranks (via `scipy.stats.rankdata`), naturally bounded in [0, 1]. NOT z-score.
5. **`ExchangeUniverse` is append-only shard parquets.** Path: `data/screener_cache/<exchange>/<YYYYMMDDTHHMMSS>.parquet`. `save()` writes a new shard with only this instance's listings (existing shards never touched). `load()` merges all shards, deduping by ticker with most-recent `refreshed_at` winning. The parquet itself is the resume checkpoint for `cache_refresh.refresh_exchange`.
6. **`Listing.exchange`** is stamped at LOAD TIME from the parquet directory name, not stored in the parquet rows.
7. **`CandidateTicker.metrics`** wrapped in `MappingProxyType` so downstream consumers cannot mutate.
8. **`RebalanceConstraints.category_caps`** wrapped in `MappingProxyType` so the frozen dataclass is genuinely immutable.
9. **`src/analysis/report.py`** is the user-edited orchestrator: `main()` takes no arguments; run inputs are local variables at the top of the function the user edits directly. `if __name__ == '__main__': main()` makes `python -m src.analysis.report` work. **Do not reintroduce argparse here.** (Other CLIs in `src/user_scripts/` still use argparse — that's fine.)
10. **`_build_data_provider()`** in `report.py` is a deliberate factory hook for test injection. Keep it. Tests monkeypatch it to inject a fake `DataProvider`.
11. **`get_strategy(name)`** and `get_criterion(name)` raise `KeyError` listing known names on miss — no silent defaults.
12. **Loud errors over silent fallbacks:** MVO and MinVariance raise `RuntimeError` on SLSQP non-convergence and on NaN-mask filter dropping all tickers; `get_price_panel` raises if a held ticker is missing from `_market_data`.
13. **Live risk-free rate** via `src/shared/risk_free_rate.get_live_risk_free_rate()` — FRED `IRLTLT01CHM156N` (Swiss long-term), falls back to 0.5% with a warning on fetch failure. **Fallback is NOT cached** (transient failures don't poison the process).
14. **IBKR-only transaction-cost model** (for the future Phase-5 MC engine) — `TransactionCostModel` defaults: commission 7 bps, min CHF 1.5, half-spread 5 bps, stamp duty 7.5 bps (Swiss) / 15 bps (foreign), FX 2 bps. Per the design spec Section 5.

---

## 7. Source documents

- **Spec (frozen):** `docs/superpowers/specs/2026-05-13-repo-refactor-design.md` — all 7 phases described. Section 7 contains per-phase implementation order.
- **Plans (one per phase, write more as you go):**
  - `docs/superpowers/plans/2026-05-14-phase-1-shared-infrastructure.md`
  - `docs/superpowers/plans/2026-05-14-phase-2-rebalancing-and-report.md`
  - `docs/superpowers/plans/2026-05-14-phase-3-screening-domain.md`
- **Completion log:** `tasks/todo.md` — Phase 1/2/3 completion blocks appended; do the same for each future phase.

---

## 8. Remaining phases (per spec Section 7)

### Phase 4 — Screener-fed strategies wired into report.py

**Files:**
- New strategies in `src/modelling/rebalancing/strategies/`: `max_growth.py`, `risk_adjusted.py`, `min_risk.py`. Each `Strategy` subclass with `name = '...'`, evaluates against `candidates: List[CandidateTicker]`.
- Edit `src/analysis/report.py`: add `candidates_from` local variable; load JSON if non-None and pass to `Rebalancer.propose(candidates=...)`.
- Delete: `src/orchestration/` directory entirely.

**Constraint:** screener-fed strategies must honor `RebalanceConstraints.swiss_tax_filter` (the screener already drops bond/high-div names via `SwissTaxCriterion`, but strategies must not reintroduce them).

**Verification:** chain `python -m src.user_scripts.run_screener --exchange NASDAQ --criteria momentum,cagr,sharpe --output-format json --output results/screener_out.json`, then edit `report.py`'s `strategy` to `'max_growth'` and `candidates_from` to that JSON path, run `python -m src.analysis.report`. Inspect the Excel for screener-fed actions.

### Phase 5 — Monte Carlo refactor

**Files:**
- `src/modelling/monte_carlo/path_distribution.py` — `PathDistribution` class (data + query methods: percentiles, max_drawdown_distribution, prob_below, expected_terminal_return, to_summary_dict).
- `src/modelling/monte_carlo/copula.py` — Clayton copula sampler (lift from legacy).
- `src/modelling/monte_carlo/shrinkage.py` — James-Stein expected-return shrinkage.
- `src/modelling/monte_carlo/cost_model.py` — `TransactionCostModel` dataclass + `ibkr_default_cost_model()` factory (per spec Section 5).
- `src/modelling/monte_carlo/engine.py` — `MonteCarloEngine.simulate(plan, prices, horizon_days, initial_nav_chf) -> PathDistribution`. Args: `n_paths`, `copula_theta`, `shrinkage_intensity`, `cost_model`, `rebalance_frequency`, `monthly_contribution_chf`, `drift_threshold_pct`, `random_seed`.
- New CLI: `src/user_scripts/run_monte_carlo.py` (argparse OK here, or follow report.py's no-args pattern — confirm with user).
- Delete: `src/modelling/monte_carlo.py` (907-line legacy).

**Verification:** `python -m src.user_scripts.run_monte_carlo --strategy mvo --horizon-days 252 --n-paths 5000 --rebalance monthly`. Inspect path distribution percentile bands + cost drag fields. Spot-check `--no-costs` returns paths with strictly higher terminal NAV than the same seed with costs.

### Phase 6 — Greeks code-style pass

**Files:**
- Edit `src/modelling/greeks/{delta,gamma,vega,rho,theta,greeks_calculator}.py` — apply the style conventions from Section 4 of this document. Replace any local `RISK_FREE_RATE` constants with calls to `src/shared/risk_free_rate.get_live_risk_free_rate()`.
- No interface changes. No new tests. The existing `src/user_scripts/examples/options_greeks_example.py` is the smoke test.

### Phase 7 — Cleanup + docs

**Files:**
- Delete `_OLD/` stragglers if any remain in the active source tree.
- Update `README.md` with new entry-point list, exchange codes, refresh cadence note.
- Grep-audit cross-domain imports: only `src/analysis/report.py` and `src/user_scripts/*.py` may import from multiple `src/<domain>/` packages.
- Append a final `tasks/todo.md` block confirming refactor closure.

---

## 9. Known data gaps / open items

1. **`data/exchange_tickers/NYSE.txt` and `NASDAQ.txt` are duplicate S&P 500 lists.** The S&P 500 spans both exchanges. User needs to curate proper venue-specific ticker lists (`data/sp500_tickers.json` has flagging info; alternative: pull from NASDAQ Trader's free public files). Other 13 exchanges have no ticker file at all (SIX, XETRA, EURONEXT_PARIS, EURONEXT_AMSTERDAM, EURONEXT_BRUSSELS, EURONEXT_LISBON, EURONEXT_DUBLIN, BIT, BME, STO, OSL, CPH, HEL). User curates per their workflow.

2. **`data/ticker_categories.json`** may be missing currently-held tickers. The Phase 1 smoke test surfaced `PANW` as missing. `PortfolioAnalyzer.get_holdings_snapshot(categories=...)` raises `RuntimeError` on missing per the no-silent-defaults rule. User adds tickers as they arise; not a code bug.

3. **Live FRED fetch:** at 2026-05-14 the Swiss long-term rate returned by `get_live_risk_free_rate()` is ~0.4% (real value, not the 0.5% fallback). Don't confuse 0.4 with the fallback when running smoke tests.

4. **`weight_drift_post_pct` in `RebalancePlan.summary()`** assumes BUY/REDUCE exactly close the gap. Whole-share rounding leaves residual drift. Under-reports post-drift slightly; acceptable for headline reporting.

5. **`yfinance.Ticker.info`** has variable shape. The `_yfinance_to_fundamental` mapping in `cache_refresh.py` uses `.get(yf_name)` deliberately — missing key maps to NaN, matching the Listing contract. This is a documented exception to the no-silent-defaults rule.

---

## 10. How to start (literal kickoff steps)

1. Read this entire document.
2. Read the spec at `docs/superpowers/specs/2026-05-13-repo-refactor-design.md` (Section 7 Phase 4).
3. Read `tasks/todo.md` to see prior phase completion notes.
4. Read `src/analysis/core/portfolio.py` and `analyzer.py` to internalize the style.
5. Read `src/modelling/rebalancing/strategies/mvo.py` to see how an existing portfolio-only strategy is structured (Phase 4's strategies will mirror this shape but consume `candidates` instead of operating purely on `holdings_df`).
6. Read `src/screening/candidate.py` to know the `CandidateTicker` contract that Phase 4 strategies consume.
7. Invoke `superpowers:writing-plans` skill, write `docs/superpowers/plans/2026-05-<DD>-phase-4-screener-fed-strategies.md` (with today's date).
8. Get user approval on the plan if it deviates materially from the spec.
9. Invoke `superpowers:subagent-driven-development` and execute task-by-task with two-stage review.
10. After Phase 4 lands, repeat for Phases 5, 6, 7 in order.

---

## 11. Environment notes

- Python 3.12 via `/Users/rbarreira/mambaforge`.
- Installed: `pandas`, `numpy`, `scipy`, `pytest 8.3.4`, `pandas_datareader 0.10.0`, `pyarrow 24.0.0`, `yfinance 0.2.65`, `openpyxl`.
- Test harness: `pytest tests/`. Suite is 122 tests as of this handoff. Run from repo root.
- Plain `pytest` (no path) collects junk from `API/` and `wundern/` — always scope to `tests/`.
- Pre-existing harmless warnings: pytz deprecation, pandas_datareader distutils, `qf_lib not installed` (in `data_provider.py:39` — an optional dependency).
- `data/screener_cache/` is currently empty (no real refresh has been run — only tests with monkeypatched yfinance have populated it under `tmp_path`).

---

## 12. Glossary of important file paths

| Path | What it is |
|------|------------|
| `docs/superpowers/specs/2026-05-13-repo-refactor-design.md` | Source spec, 7 phases |
| `docs/superpowers/plans/2026-05-14-phase-{1,2,3}-*.md` | Executed plans |
| `tasks/todo.md` | Per-phase completion log |
| `src/analysis/core/` | FROZEN reference architecture |
| `src/analysis/report.py` | User-edited zero-arg `main()` orchestrator |
| `src/shared/{risk_free_rate,constraints}.py` | Phase 1 deliverables |
| `src/modelling/rebalancing/` | Phase 2 domain package |
| `src/screening/` | Phase 3 domain package |
| `src/modelling/monte_carlo.py` | LEGACY — Phase 5 will replace with `monte_carlo/` package |
| `src/modelling/greeks/` | Light-edit target (Phase 6) |
| `src/orchestration/` | DELETE in Phase 4 |
| `tests/{analysis,modelling,screening,shared,user_scripts}/` | Test tree, 122 passing tests |
| `data/screener_cache/<exchange>/<timestamp>.parquet` | Append-only shard cache (Phase 3) |
| `data/exchange_tickers/{NYSE,NASDAQ,LSE}.txt` | Seed ticker lists (Phase 3) |

---

That's everything. The new agent has full context. Start with step 1 of Section 10.
