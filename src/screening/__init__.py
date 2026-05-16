"""
Screening domain: per-exchange cached universes + criteria-based ranking.

Public surface:
    Listing, ExchangeUniverse, CandidateTicker, Screener,
    Criterion (ABC) + concrete subclasses, get_criterion,
    criterion_registry.
"""

from src.screening.candidate import (
    CandidateTicker, load_candidates_from_json)
from src.screening.criteria import (
    Criterion,
    CagrCriterion,
    GrowthCriterion,
    LiquidityCriterion,
    MaxDrawdownCriterion,
    MomentumCriterion,
    PERatioCriterion,
    SectorCriterion,
    SharpeCriterion,
    SwissTaxCriterion,
    criterion_registry,
    get_criterion,
)
from src.screening.listing import (
    Listing, fundamental_field_names)
from src.screening.screener import Screener
from src.screening.universe import ExchangeUniverse


__all__ = [
    'CandidateTicker',
    'load_candidates_from_json',
    'Criterion',
    'CagrCriterion',
    'GrowthCriterion',
    'LiquidityCriterion',
    'MaxDrawdownCriterion',
    'MomentumCriterion',
    'PERatioCriterion',
    'SectorCriterion',
    'SharpeCriterion',
    'SwissTaxCriterion',
    'criterion_registry',
    'get_criterion',
    'Listing',
    'fundamental_field_names',
    'Screener',
    'ExchangeUniverse',
]
