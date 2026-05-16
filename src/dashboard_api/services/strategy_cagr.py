"""Weighted-average 3-year CAGR estimate per rebalancing strategy.

For each ticker in a strategy's ``target_weights`` we look up
the per-ticker 3Y CAGR from the screener cache (or a passed-in
override map) and aggregate as ``sum(weight_i * cagr_i)`` over
the in-cache subset. Tickers without a known CAGR are reported
in a ``missing`` list so the frontend can surface coverage.

Functions
---------
forecast_strategy_cagr
    Weighted CAGR plus the missing-tickers list.
"""

from typing import Dict, Tuple


def forecast_strategy_cagr(
        target_weights: Dict[str, float],
        cagr_by_ticker: Dict[str, float]
) -> Tuple[float, list, float]:
    """Compute weighted-average forecast CAGR.

    Args:
        target_weights: Strategy output ticker -> fraction.
            Should sum to 1.0; unmatched tickers are skipped
            and the weighted average is renormalised against
            the matched-weight total.
        cagr_by_ticker: Ticker -> 3Y CAGR as a decimal fraction
            (e.g. 0.18 for 18% CAGR).

    Returns:
        Tuple of:
            forecast_pct : float
                Weighted-average CAGR expressed as percent.
                0.0 when no targets matched the CAGR map.
            missing : list[str]
                Tickers in target_weights with no CAGR entry,
                sorted alphabetically.
            coverage_weight : float
                Sum of target weights that had a CAGR (1.0 if
                every ticker matched).
    """
    matched_weight = 0.0
    weighted = 0.0
    missing = []
    for ticker, weight in target_weights.items():
        if ticker not in cagr_by_ticker:
            missing.append(ticker)
            continue
        matched_weight += weight
        weighted += weight * cagr_by_ticker[ticker]
    if matched_weight <= 0.0:
        return 0.0, sorted(missing), 0.0
    # Renormalise so partial coverage still produces a sane
    # weighted average over the matched subset.
    forecast = (weighted / matched_weight) * 100.0
    return float(forecast), sorted(missing), float(matched_weight)
