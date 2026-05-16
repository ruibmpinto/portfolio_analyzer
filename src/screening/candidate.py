"""
Candidate ticker output of the Screener.

Implements Contract 2 of the refactor design spec. Returned
in ranked order by Screener.rank(); consumed by the screener-
fed strategies in src.modelling.rebalancing.strategies.
"""

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List


@dataclass(frozen=True)
class CandidateTicker:
    """
    One ranked candidate from the screener.

    Attributes:
        ticker: Native exchange symbol (e.g. 'AAPL', 'NESN.SW').
        exchange: Exchange code (e.g. 'NASDAQ', 'SIX').
        currency: ISO-4217 listing currency.
        sector: yfinance sector label or 'Unknown'.
        market_cap: Market capitalisation in listing currency
            (NaN if unavailable).
        composite_score: Aggregate ranking in [0, 1]; higher is
            better. The Screener computes this as the weighted
            mean of per-criterion percentile ranks.
        metrics: Per-criterion raw scores keyed by criterion
            name. Wrapped in MappingProxyType so the dict is
            read-only.
    """

    ticker: str
    exchange: str
    currency: str
    sector: str
    market_cap: float
    composite_score: float
    metrics: Dict[str, float]

    def __post_init__(self):
        if not 0.0 <= self.composite_score <= 1.0:
            raise ValueError(
                f'composite_score must be in [0, 1], '
                f'got {self.composite_score}.')
        # Wrap the metrics dict so callers can't mutate it
        if not isinstance(self.metrics, MappingProxyType):
            object.__setattr__(
                self, 'metrics',
                MappingProxyType(dict(self.metrics)))


def load_candidates_from_json(path: str) -> List[CandidateTicker]:
    """Load a list of CandidateTicker from a JSON file.

    Args:
        path: Path to a JSON file containing a list of dicts
            with keys ``ticker, exchange, currency, sector,
            market_cap, composite_score, metrics``. The shape
            matches what ``run_screener.py`` writes for
            ``--output-format json``.

    Returns:
        List of CandidateTicker in the order they appear in the
        file. Returns an empty list when the file contains
        ``[]``.

    Raises:
        KeyError: When a required field is missing from any row.
        ValueError: When ``composite_score`` is outside [0, 1].
    """
    with open(path) as f:
        rows = json.load(f)
    required = (
        'ticker', 'exchange', 'currency', 'sector',
        'market_cap', 'composite_score', 'metrics')
    candidates: List[CandidateTicker] = []
    for row in rows:
        for key in required:
            if key not in row:
                err = KeyError(key)
                err.add_note(
                    f'load_candidates_from_json: row missing '
                    f'required field {key!r}.')
                raise err
        candidates.append(CandidateTicker(
            ticker=row['ticker'],
            exchange=row['exchange'],
            currency=row['currency'],
            sector=row['sector'],
            market_cap=float(row['market_cap']),
            composite_score=float(row['composite_score']),
            metrics=dict(row['metrics'])))
    return candidates
