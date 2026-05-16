"""James-Stein shrinkage for daily mean returns.

Pure function. Shrinks per-ticker historical daily means toward
a common anchor (default = 6% annualised equity risk premium).
Reduces the influence of survivorship-biased outliers on the
Monte Carlo drift.
"""

from typing import Optional

import pandas as pd


trading_days_per_year = 252.0
ss_floor = 1e-15
min_columns_required = 2
js_min_columns = 3


def james_stein_shrink(
    returns: pd.DataFrame,
    anchor_annual: float = 0.06,
    intensity: Optional[float] = None) -> pd.Series:
    """Return JS-shrunk daily mean per ticker.

    Args:
        returns: Daily returns DataFrame, one column per ticker.
            NaNs allowed; per-column means drop NaNs.
        anchor_annual: Annual target return for the anchor
            (e.g. 0.06 for a 6% equity risk premium).
        intensity: Fixed shrinkage factor in [0, 1]. If None,
            compute the JS-optimal value from the cross-section.

    Returns:
        pd.Series indexed by ticker with shrunk daily means.

    Raises:
        ValueError: When ``returns`` has fewer than two
            columns, or when ``intensity`` is outside [0, 1].
    """
    p = returns.shape[1]
    if p < min_columns_required:
        raise ValueError(
            f'james_stein_shrink: need at least 2 columns, '
            f'got {p}.')
    anchor_daily = anchor_annual / trading_days_per_year
    means = returns.mean()
    scales = returns.std()

    if intensity is None:
        ss = float(((means - anchor_daily) ** 2).sum())
        if p < js_min_columns or ss < ss_floor:
            b_js = 1.0
        else:
            n_obs = max(
                int(returns.notna().sum().mean()), 1)
            var_pool = float((scales ** 2).mean()) / n_obs
            b_js = max(
                0.0, 1.0 - (p - 2) * var_pool / ss)
    else:
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(
                f'james_stein_shrink: intensity must be in '
                f'[0, 1], got {intensity}.')
        b_js = float(intensity)

    shrunk = anchor_daily + b_js * (means - anchor_daily)
    return pd.Series(shrunk, index=returns.columns)
