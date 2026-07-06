"""
Rebalancer orchestrator.

Wires PortfolioAnalyzer + Strategy + RebalanceConstraints into
a RebalancePlan. Holds the shared action-generation logic that
turns a strategy's target weights into BUY/REDUCE/HOLD/CASH
actions.

Phase 4 funding model: the REDUCE pass runs first to size the
recycling pool, the CASH bucket reserves its target CHF, and
the BUY pass then runs against
``new_capital + total_reduce - target_cash`` (floored at 0).
Largest-gap-first BUY allocation; overweight names beyond the
band are REDUCE; gaps inside the band are HOLD. Target tickers
not in the analyzer's holdings_df are priced via
``pricing.fetch_chf_price`` using a CandidateTicker lookup.
Every ticker in target_weights gets exactly one action so the
RebalancePlan's ticker-set cross-check holds.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.modelling.rebalancing.action import (
    RebalancingAction, cash_ticker)
from src.modelling.rebalancing.plan import RebalancePlan
from src.modelling.rebalancing.pricing import fetch_chf_price
from src.modelling.rebalancing.strategies.base import Strategy
from src.shared.constraints import (
    RebalanceConstraints, default_constraints)


cash_action_note = 'Cash reserve'


class Rebalancer:
    """
    Apply a Strategy to current holdings and produce a plan.

    Attributes:
        analyzer: Provides get_holdings_snapshot() and
            get_price_panel().
        strategy: A concrete Strategy implementation.
        constraints: RebalanceConstraints for this run.
        ticker_categories: Optional ticker -> category map
            forwarded to analyzer.get_holdings_snapshot.
            Required when constraints reference
            excluded_categories or category_caps and the
            analyzer's snapshot won't otherwise have a
            'category' column.
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
            candidates: Optional list of CandidateTicker. Only
                screener-fed strategies (Phase 4) require it.
                Used to source CHF prices for target tickers
                not in the analyzer's holdings_df.

        Returns:
            RebalancePlan bundling actions, target weights,
            input holdings snapshot, strategy name, applied
            constraints, and a generation timestamp.
        """
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
                in holdings_df and are not the CASH
                pseudo-ticker.
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
        """
        Turn target weights + snapshot into actions.

        Phase 4 order: emit the CASH action first, run the
        REDUCE pass to size the recycling pool, then run the
        BUY pass against ``new_capital + total_reduce -
        target_cash``, then close out in-band tickers as HOLD.
        Tickers not in holdings_df are treated as
        current_value=0 / current_w_pct=0 and priced from
        candidate_prices_chf. The CASH pseudo-ticker bypasses
        the BUY/REDUCE/HOLD passes; it emits a single CASH
        action recording target_w * (NAV + new_capital).
        The BUY pool is floored at 0 when
        ``new_capital + total_reduce < target_cash`` — i.e. when
        the cash reservation exceeds available funding, BUYs
        collapse to 0 rather than going negative.

        Args:
            target_weights: Strategy output, ticker -> fraction.
            holdings_df: Snapshot DataFrame from
                analyzer.get_holdings_snapshot().
            candidate_prices_chf: CHF prices for target tickers
                not present in holdings_df (and not equal to
                cash_ticker).

        Returns:
            List of RebalancingAction, one per ticker in
            target_weights.

        Raises:
            RuntimeError: When a ticker in target_weights is
                neither in holdings_df, nor in
                candidate_prices_chf, nor equal to
                ``cash_ticker``.
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
            # CASH handled here; excluded from BUY/REDUCE/HOLD passes
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
                'current_value': current_value,
                'current_w_pct': current_w_pct,
                'target_w_pct': target_w * 100.0,
                'delta_chf': target_value - current_value,
                'price_chf': price_chf,
            })

        # Post-trade weight against the post-trade NAV (holdings +
        # new capital). One formula, applied at every action site.
        def achievable_pct(post_value_chf):
            if target_total <= 0:
                return 0.0
            return post_value_chf / target_total * 100.0

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
                    achievable_wt_pct=achievable_pct(d['current_value']),
                    note='Overweight but <1 share to reduce'))
                continue
            proceeds = shares * d['price_chf']
            total_reduce_chf += proceeds
            actions.append(RebalancingAction(
                ticker=d['ticker'], action='REDUCE',
                shares=shares, est_cost_chf=proceeds,
                current_wt_pct=d['current_w_pct'],
                target_wt_pct=d['target_w_pct'],
                achievable_wt_pct=achievable_pct(d['current_value'] - proceeds),
                note='Only if held >6 months'))

        # BUY pool augments new_capital with REDUCE proceeds
        # and reserves the cash target.
        buy_pool = max(
            0.0,
            self.constraints.new_capital_chf
            + total_reduce_chf
            - target_cash_chf)

        # Proportional fill: when the pool cannot cover every
        # gap, each underweight ticker gets the same fraction of
        # its gap, so near-equal gaps get near-equal funding and
        # sort order no longer decides who is starved. The sort
        # is kept only for stable display order.
        underweight = sorted(
            [d for d in deltas if d['delta_chf'] > upper],
            key=lambda d: -d['delta_chf'])
        total_gap_chf = sum(d['delta_chf'] for d in underweight)
        if total_gap_chf > 0:
            fill_ratio = min(1.0, buy_pool / total_gap_chf)
        else:
            fill_ratio = 0.0
        bought_tickers = set()
        for d in underweight:
            alloc = fill_ratio * d['delta_chf']
            shares = 0
            if d['price_chf'] > 0:
                shares = int(alloc / d['price_chf'])
            if shares <= 0:
                continue
            cost = shares * d['price_chf']
            actions.append(RebalancingAction(
                ticker=d['ticker'], action='BUY',
                shares=shares, est_cost_chf=cost,
                current_wt_pct=d['current_w_pct'],
                target_wt_pct=d['target_w_pct'],
                achievable_wt_pct=achievable_pct(d['current_value'] + cost),
                note=''))
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
                achievable_wt_pct=achievable_pct(d['current_value']),
                note='Underweight but no capital available'))

        # In-band HOLDs
        for d in deltas:
            if lower <= d['delta_chf'] <= upper:
                actions.append(RebalancingAction(
                    ticker=d['ticker'], action='HOLD',
                    shares=0, est_cost_chf=0.0,
                    achievable_wt_pct=achievable_pct(d['current_value']),
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
        target_cash_chf = max(0.0, float(target_w) * float(target_total))
        achievable_wt_pct = 0.0
        if target_total > 0:
            achievable_wt_pct = target_cash_chf / target_total * 100.0
        return RebalancingAction(
            ticker=cash_ticker,
            action='CASH',
            shares=0,
            est_cost_chf=target_cash_chf,
            current_wt_pct=0.0,
            target_wt_pct=float(target_w) * 100.0,
            achievable_wt_pct=achievable_wt_pct,
            note=cash_action_note)
