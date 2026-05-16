"""Configuration knobs for the dashboard sidecar.

Centralises the values the route handlers need so they can be
overridden via environment variables at sidecar startup (useful
during ``pnpm tauri dev`` when the bundled binary path is not
the same as production).

All paths are resolved relative to the project root unless an
absolute path is supplied.
"""

import os
import pathlib
from typing import Optional, Tuple

from src.analysis.loaders.csv_loader import (
    default_degiro_path,
    default_ibkr_statement_path)


def _env_path(name: str, default: str) -> pathlib.Path:
    raw = os.environ.get(name, default)
    return pathlib.Path(raw).expanduser().resolve()


def _resolved_statement_path(
        env_key: str,
        helper_fn) -> Optional[pathlib.Path]:
    """Prefer the env override; else pick the latest statement.

    Returns None when no env var is set and no statement file
    is present on disk. Callers `.exists()`-check before use.
    """
    raw = os.environ.get(env_key)
    if raw:
        return pathlib.Path(raw).expanduser().resolve()
    path = helper_fn()
    return path.resolve() if path else None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _env_csv(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return tuple(s.strip() for s in raw.split(',') if s.strip())


degiro_csv_path = _resolved_statement_path(
    'DASHBOARD_DEGIRO_CSV',
    lambda: default_degiro_path('transactions'))
ibkr_csv_path = _resolved_statement_path(
    'DASHBOARD_IBKR_CSV', default_ibkr_statement_path)
ticker_categories_path = _env_path(
    'DASHBOARD_CATEGORIES',
    'data/ticker_categories.json')
screener_metadata_path = _env_path(
    'DASHBOARD_SCREENER_META',
    'data/cache/screener_cache/metadata.json')
rebalance_log_path = _env_path(
    'DASHBOARD_REBALANCE_LOG',
    'data/rebalance_log.json')

base_currency = os.environ.get('DASHBOARD_BASE_CCY', 'CHF')
strategies = _env_csv(
    'DASHBOARD_STRATEGIES',
    ('mvo', 'equal_weight', 'min_variance'))
benchmarks = _env_csv(
    'DASHBOARD_BENCHMARKS', ('SPY', 'DIA', 'VT'))

mc_n_paths = _env_int('DASHBOARD_MC_PATHS', 1000)
mc_horizon_days = _env_int('DASHBOARD_MC_HORIZON', 252)
mc_monthly_contribution_chf = _env_float(
    'DASHBOARD_MC_CONTRIB', 2000.0)
mc_max_weight = _env_float('DASHBOARD_MC_MAX_WT', 0.15)
mc_new_capital_chf = _env_float('DASHBOARD_MC_NEW_CAP', 2000.0)
mc_random_seed = _env_int('DASHBOARD_MC_SEED', 42)

cache_ttl_seconds = _env_float('DASHBOARD_CACHE_TTL', 600.0)
host = os.environ.get('DASHBOARD_HOST', '127.0.0.1')
port = _env_int('DASHBOARD_PORT', 0)


strategy_reasoning = {
    'mvo':
        'Mean-Variance Optimisation: maximise expected return '
        'per unit of variance under the configured per-position '
        'and per-category weight caps. Falls back to equal-weight '
        'when the covariance estimate is degenerate.',
    'equal_weight':
        'Equal-weight: divide capital evenly across the eligible '
        'universe. A robust default when forward-return signals '
        'are noisy or the covariance estimate is unreliable.',
    'min_variance':
        'Minimum-variance: pick the weights that minimise '
        'portfolio variance subject to the constraints. Targets '
        'capital preservation rather than expected return.',
    'inverse_vol':
        'Inverse-volatility: weight each name by 1/sigma. Cheaper '
        'than full mean-variance and avoids over-allocating to '
        'high-vol names.',
    'max_growth':
        'Maximum-growth: concentrate in names with the highest '
        'estimated 3Y CAGR. High return potential, higher '
        'concentration risk.',
    'min_risk':
        'Minimum-risk: blend low-volatility equities with the '
        'cash bucket. Reserved for capital-preservation scenarios.',
    'risk_adjusted':
        'Risk-adjusted growth: rank by Sharpe-like score and '
        'cap each position to manage tail risk while keeping '
        'meaningful expected return.',
}
