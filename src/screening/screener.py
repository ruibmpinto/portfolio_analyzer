"""
Screener orchestrator.

Loads ExchangeUniverses, applies a list of Criterion instances
to every Listing, drops Listings whose criteria return None,
percentile-ranks each criterion's surviving signals, computes
a composite score as the weighted mean of percentile ranks,
and returns ranked CandidateTicker objects.

The single-Universe load step is exposed as a module-level
function (_load_universe) so tests can monkeypatch it without
touching the real parquet cache.
"""

from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import rankdata

from src.screening.candidate import CandidateTicker
from src.screening.criteria import Criterion
from src.screening.listing import Listing
from src.screening.universe import ExchangeUniverse


def _load_universe(exchange: str) -> ExchangeUniverse:
    """Default ExchangeUniverse loader; overridable in tests."""
    return ExchangeUniverse.load(exchange)


class Screener:
    """
    Multi-exchange screener with criteria-based ranking.

    Attributes:
        exchanges: Exchange codes whose Universes to load.
        criteria: Ordered list of Criterion instances; built up
            via add_criterion().
    """

    def __init__(self, exchanges: List[str]):
        self.exchanges = list(exchanges)
        self.criteria: List[Criterion] = []
        self._last_candidates: Optional[List[CandidateTicker]] = None

    def add_criterion(self, criterion: Criterion) -> 'Screener':
        """Register a criterion; returns self for chaining."""
        self.criteria.append(criterion)
        return self

    def rank(
        self, top_n: Optional[int] = None) -> List[CandidateTicker]:
        """
        Apply criteria, compute composites, return ranked list.

        Args:
            top_n: If given, truncate the result to the first
                top_n candidates.

        Returns:
            List of CandidateTicker, sorted by composite_score
            descending.

        Raises:
            RuntimeError: When no criteria have been added.
        """
        if not self.criteria:
            raise RuntimeError(
                'Screener.rank: add at least one criterion '
                'before calling rank().')

        listings: List[Listing] = []
        for exchange in self.exchanges:
            universe = _load_universe(exchange)
            listings.extend(universe)

        # Per-criterion raw signals across all listings
        raw_per_criterion: Dict[str, List[Optional[float]]] = {}
        for crit in self.criteria:
            raw_per_criterion[crit.name] = [
                crit.evaluate(l) for l in listings]

        # Keep only listings where every criterion produced a value
        kept_indices = [
            i for i in range(len(listings))
            if all(
                raw_per_criterion[crit.name][i] is not None
                for crit in self.criteria)]
        if not kept_indices:
            self._last_candidates = []
            return []

        # Percentile-rank surviving signals per criterion
        pct_per_criterion: Dict[str, Dict[int, float]] = {}
        for crit in self.criteria:
            kept_scores = [
                raw_per_criterion[crit.name][i]
                for i in kept_indices]
            ranks = rankdata(kept_scores, method='average')
            n = len(kept_scores)
            # Map rank in [1, n] to percentile in [0, 1]
            pcts = (ranks - 1.0) / max(n - 1, 1)
            pct_per_criterion[crit.name] = {
                idx: float(pct)
                for idx, pct in zip(kept_indices, pcts)}

        total_weight = sum(c.weight for c in self.criteria)
        candidates = []
        for idx in kept_indices:
            l = listings[idx]
            weighted = sum(
                pct_per_criterion[c.name][idx] * c.weight
                for c in self.criteria)
            composite = weighted / total_weight
            metrics = {
                c.name: float(raw_per_criterion[c.name][idx])
                for c in self.criteria}
            candidates.append(CandidateTicker(
                ticker=l.ticker,
                exchange=l.exchange,
                currency=l.currency,
                sector=l.sector,
                market_cap=l.market_cap,
                composite_score=composite,
                metrics=metrics))

        candidates.sort(key=lambda c: -c.composite_score)
        if top_n is not None:
            candidates = candidates[:top_n]
        self._last_candidates = candidates
        return candidates

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flat DataFrame view of the most recent rank() output.

        Returns:
            DataFrame with one row per CandidateTicker; columns
            include ticker, exchange, currency, sector,
            market_cap, composite_score, and one column per
            registered criterion's raw metric.

        Raises:
            RuntimeError: When rank() has not been called yet.
        """
        if self._last_candidates is None:
            raise RuntimeError(
                'Screener.to_dataframe: call rank() first.')
        rows = []
        for c in self._last_candidates:
            row = {
                'ticker': c.ticker,
                'exchange': c.exchange,
                'currency': c.currency,
                'sector': c.sector,
                'market_cap': c.market_cap,
                'composite_score': c.composite_score,
            }
            for name, value in c.metrics.items():
                row[name] = value
            rows.append(row)
        return pd.DataFrame(rows)
