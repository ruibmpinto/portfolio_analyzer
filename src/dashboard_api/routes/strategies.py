"""Strategies endpoint: plan + Monte Carlo distribution per strategy.

For each configured strategy this endpoint produces a payload
containing:

  - The strategy's target weights and BUY/REDUCE/HOLD/CASH
    action list.
  - A human-readable reasoning blurb from ``config.strategy_reasoning``.
  - The Monte Carlo summary stats (mean, sigma, terminal/MDD
    percentiles, P(Loss), P(Gain>X), VaR/CVaR).
  - Pre-computed P&L and drawdown histograms ready to feed
    Recharts ``BarChart``.
  - The NAV percentile fan-chart series (p5/p25/p50/p75/p95).
  - A weighted 3Y CAGR forecast.

Routes
------
GET /api/strategies
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request

from src.dashboard_api import config
from src.dashboard_api.services import strategy_cagr as cagr_mod
from src.modelling.monte_carlo import (
    MonteCarloEngine, ibkr_default_cost_model)
from src.modelling.monte_carlo import engine as mc_engine
from src.modelling.rebalancing import (
    Rebalancer, get_strategy)
from src.shared.constraints import RebalanceConstraints


router = APIRouter()

pnl_histogram_bins = 40
dd_histogram_bins = 30
fan_chart_step = 21  # ~ one month at 21 trading days


@router.get('/api/strategies')
def get_strategies(request: Request):
    """Return per-strategy payload for the Strategies tab."""
    ctx = request.app.state.ctx
    return ctx.cache.get_or_compute(
        'strategies', lambda: _build_payload(ctx))


def _build_payload(ctx) -> Dict:
    analyzer = ctx.analyzer
    prices = analyzer.get_price_panel()
    cagr_by_ticker = _cagr_map_from_holdings(prices)
    constraints = RebalanceConstraints(
        max_weight_default=config.mc_max_weight,
        new_capital_chf=config.mc_new_capital_chf)
    cost_model = ibkr_default_cost_model()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # One block per strategy
    blocks: List[Dict] = []
    errors: List[Dict] = []
    for name in config.strategies:
        try:
            block = _build_strategy_block(
                ctx=ctx,
                analyzer=analyzer,
                prices=prices,
                strategy_name=name,
                constraints=constraints,
                cost_model=cost_model,
                cagr_by_ticker=cagr_by_ticker)
        except Exception as exc:  # noqa: BLE001
            errors.append({'strategy': name, 'error': str(exc)})
            continue
        blocks.append(block)
    return {
        'strategies': blocks,
        'errors': errors,
        'mc': {
            'n_paths': config.mc_n_paths,
            'horizon_days': config.mc_horizon_days,
            'monthly_contribution_chf':
                config.mc_monthly_contribution_chf,
            'methodology': _mc_methodology(),
        },
    }


def _mc_methodology() -> Dict:
    """Static description of the MC model + equations + params."""
    # Single source of truth for the UI's methodology panel.
    return {
        'summary': (
            'Clayton-copula joint sampling of daily returns. '
            'Each ticker has a Student-t marginal whose '
            'location is James-Stein shrunk toward a constant '
            'anchor return. Cash earns a deterministic daily '
            'rate. Transaction costs are applied at t=0 and on '
            'every rebalance step.'),
        'distributions': [
            ('Per-ticker daily return',
             'r_{t,i} ~ Student-t(nu_i, mu_i, sigma_i), '
             'fit by MLE on historical returns'),
            ('Joint dependence',
             'C_theta(u_1,...,u_d): Clayton copula on the '
             'uniform-transformed marginals'),
            ('Cash growth',
             'C_{t+1} = C_t * (1 + r_cash / 252), '
             'deterministic'),
        ],
        'equations': [
            ('NAV update',
             'V_{t+1} = sum_i H_{t,i} * (1 + r_{t,i}) + C_{t+1}'),
            ('James-Stein shrinkage',
             'mu_i_hat = (1 - w) * mean(r_i) + w * mu_anchor'),
            ('Clayton theta from Kendall tau',
             'theta = 2 * tau / (1 - tau), '
             'tau = median pairwise Kendall tau'),
            ('Standard error of mean return',
             'SE(mean) = sigma / sqrt(N_paths)'),
            ('Standard error of probability',
             'SE(p_hat) = sqrt(p * (1 - p) / N_paths)'),
        ],
        'parameters': {
            'student_t_df_clip':
                [mc_engine.df_floor, mc_engine.df_ceil],
            'clayton_theta_clip':
                [mc_engine.theta_floor, mc_engine.theta_ceil],
            'kendall_tau_clip':
                [mc_engine.tau_floor, mc_engine.tau_ceil],
            'min_history_for_fit_days':
                mc_engine.min_history_for_fit,
            'trading_days_per_year':
                mc_engine.trading_days_per_year,
        },
    }


def _build_strategy_block(
        ctx, analyzer, prices, strategy_name,
        constraints, cost_model, cagr_by_ticker) -> Dict:
    rebalancer = Rebalancer(
        analyzer=analyzer,
        strategy=get_strategy(strategy_name),
        constraints=constraints,
        ticker_categories=ctx.categories or None)
    plan = rebalancer.propose()
    ticker_currencies = dict(zip(
        plan.holdings_df['ticker'],
        plan.holdings_df['currency']))
    engine = MonteCarloEngine(
        n_paths=config.mc_n_paths,
        cost_model=cost_model,
        rebalance_frequency='monthly',
        monthly_contribution_chf=(
            config.mc_monthly_contribution_chf),
        random_seed=config.mc_random_seed,
        ticker_currencies=ticker_currencies)
    dist = engine.simulate(
        plan, prices,
        horizon_days=config.mc_horizon_days,
        initial_nav_chf=plan.total_value_chf)
    summary = dist.to_summary_dict()
    pnl_hist = _pnl_histogram(dist)
    dd_hist = _dd_histogram(dist)
    fan = _fan_chart(dist)
    forecast_pct, missing, coverage = (
        cagr_mod.forecast_strategy_cagr(
            plan.target_weights, cagr_by_ticker))
    return {
        'name': strategy_name,
        'reasoning': config.strategy_reasoning.get(
            strategy_name, ''),
        'target_weights': plan.target_weights,
        'actions': [
            {
                'ticker': a.ticker,
                'action': a.action,
                'shares': a.shares,
                'est_cost_chf': a.est_cost_chf,
                'current_wt_pct': a.current_wt_pct,
                'target_wt_pct': a.target_wt_pct,
                'note': a.note,
            }
            for a in plan.actions],
        'plan_summary': plan.summary(),
        'mc_summary': summary,
        'pnl_histogram': pnl_hist,
        'dd_histogram': dd_hist,
        'fan_chart': fan,
        'cagr_forecast_pct': forecast_pct,
        'cagr_forecast_coverage': coverage,
        'cagr_forecast_missing': missing,
    }


def _pnl_histogram(dist) -> List[Dict]:
    """Histogram of terminal P&L percentages ready for Recharts."""
    terminal = dist.terminal_distribution()
    invested = dist.total_invested_chf
    pnl_pct = (terminal - invested) / invested * 100.0
    counts, edges = np.histogram(pnl_pct, bins=pnl_histogram_bins)
    total = counts.sum()
    out = []
    for i in range(len(counts)):
        mid = float((edges[i] + edges[i + 1]) / 2.0)
        freq = float(counts[i]) / total * 100.0 if total else 0.0
        out.append({'pnl': mid, 'freq': freq})
    return out


def _dd_histogram(dist) -> List[Dict]:
    """Histogram of per-path max drawdowns as negative percent."""
    dd = dist.max_drawdown_distribution() * 100.0
    counts, edges = np.histogram(dd, bins=dd_histogram_bins)
    total = counts.sum()
    out = []
    for i in range(len(counts)):
        mid = float((edges[i] + edges[i + 1]) / 2.0)
        freq = float(counts[i]) / total * 100.0 if total else 0.0
        out.append({'dd': mid, 'freq': freq})
    return out


def _fan_chart(dist) -> List[Dict]:
    """Sub-sampled percentile fan-chart series, indexed by month."""
    pct = dist.percentiles()
    out = []
    n_rows = len(pct)
    step = max(1, fan_chart_step)
    indices = list(range(0, n_rows, step))
    if (n_rows - 1) not in indices:
        indices.append(n_rows - 1)
    for trading_day in indices:
        row = pct.iloc[trading_day]
        month = round(trading_day / 21.0)
        out.append({
            'month': month,
            'trading_day': int(trading_day),
            'p5': float(row['p5']),
            'p25': float(row['p25']),
            'p50': float(row['p50']),
            'p75': float(row['p75']),
            'p95': float(row['p95']),
        })
    return out


def _cagr_map_from_holdings(price_panel) -> Dict[str, float]:
    """Compute per-ticker 3Y CAGR from the analyzer's price panel.

    Falls back to a 0% CAGR for tickers with under ~250 trading
    days of history rather than dropping them — so the weighted
    aggregate behaves predictably.
    """
    cagr: Dict[str, float] = {}
    if price_panel is None or price_panel.empty:
        return cagr
    for ticker in price_panel.columns:
        series = price_panel[ticker].dropna()
        if len(series) < 252:
            continue
        first = float(series.iloc[0])
        last = float(series.iloc[-1])
        if first <= 0:
            continue
        years = len(series) / 252.0
        if years <= 0:
            continue
        cagr[ticker] = (last / first) ** (1.0 / years) - 1.0
    return cagr
