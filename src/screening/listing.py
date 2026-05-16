"""
Per-ticker cache record for the screener.

Implements Contract 1 of the refactor design spec. One Listing
= one row in an ExchangeUniverse's parquet file. Holds 156
weekly bars (closes / volumes / dividends), 30 Tier-B
fundamental snapshots (17 yfinance-derived + 13 defeatbeta
US-only enrichments), plus identification metadata.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd


weekly_series_length = 156

fundamental_field_names = (
    # Original 17 (Phase 3, yfinance-sourced)
    'pe_ratio_ttm',
    'pe_ratio_forward',
    'pb_ratio',
    'ps_ratio',
    'dividend_yield_ttm',
    'eps_ttm',
    'revenue_growth_yoy',
    'earnings_growth_yoy',
    'profit_margin',
    'operating_margin',
    'roe',
    'roa',
    'debt_to_equity',
    'free_cash_flow',
    'beta_yf',
    'shares_outstanding',
    'short_ratio',
    # Phase 8 additions (defeatbeta US-only enrichment)
    'peg_ratio',
    'roic',
    'roce',
    'wacc',
    'equity_multiplier',
    'asset_turnover',
    'enterprise_value',
    'enterprise_to_revenue',
    'enterprise_to_ebitda',
    'ttm_revenue',
    'ebitda_growth_yoy',
    'market_cap_chf',
    'dcf_implied_upside',
)


@dataclass(frozen=True)
class Listing:
    """
    One row of an ExchangeUniverse parquet cache.

    Attributes:
        ticker: Native exchange symbol.
        exchange: Exchange code; set at load time from the
            parquet filename rather than stored in the file.
        currency: ISO-4217 listing currency.
        sector: yfinance sector label or 'Unknown'.
        market_cap: Listing-currency market cap (NaN allowed).
        closes_weekly: Length-156 numpy array of weekly closes,
            oldest first. NaN allowed for pre-listing weeks.
        closes_weekly_dates: Length-156 DatetimeIndex aligned
            with closes_weekly.
        volumes_weekly: Length-156 numpy array of weekly volumes.
        dividends_weekly: Length-156 numpy array of weekly
            dividend cash per share.
        fundamentals: Dict keyed by `fundamental_field_names`,
            30 floats (NaN allowed). Every name must be present;
            constructor raises if any are missing.
        refreshed_at: Timestamp of the underlying yfinance fetch.
    """

    ticker: str
    exchange: str
    currency: str
    sector: str
    market_cap: float
    closes_weekly: np.ndarray
    closes_weekly_dates: pd.DatetimeIndex
    volumes_weekly: np.ndarray
    dividends_weekly: np.ndarray
    fundamentals: Dict[str, float]
    refreshed_at: datetime

    def __post_init__(self):
        for series_name in (
            'closes_weekly', 'closes_weekly_dates',
            'volumes_weekly', 'dividends_weekly'):
            series = getattr(self, series_name)
            if len(series) != weekly_series_length:
                raise ValueError(
                    f'{series_name} must have length '
                    f'{weekly_series_length}; got {len(series)}.')
        missing = [
            name for name in fundamental_field_names
            if name not in self.fundamentals]
        if missing:
            raise ValueError(
                f'fundamentals missing required field(s): '
                f'{missing}.')
