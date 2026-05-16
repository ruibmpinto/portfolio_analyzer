"""
ExchangeUniverse: per-exchange aggregator with append-only parquet shards.

Holds a list of Listings for one exchange. Persists to a
DIRECTORY at `data/screener_cache/<exchange>/`, where each
save writes a NEW timestamped shard parquet rather than
rewriting an existing file. Loading merges every shard in the
directory and deduplicates by ticker (most recent
`refreshed_at` wins), so callers always see one Listing per
ticker.

This append-only model lets cache_refresh persist incremental
batches at O(batch_size) instead of rewriting the full
universe each checkpoint.
"""

import pathlib
from datetime import datetime
from typing import Callable, Iterator, List

import numpy as np
import pandas as pd

from src.screening.listing import (
    Listing, fundamental_field_names, weekly_series_length)


cache_dir = pathlib.Path(
    __file__).resolve().parents[2] / 'data' / 'screener_cache'

parquet_engine = 'pyarrow'

shard_timestamp_format = '%Y%m%dT%H%M%S'


class ExchangeUniverse:
    """
    Per-exchange collection of Listings.

    Attributes:
        exchange: Exchange code (e.g. 'NASDAQ', 'SIX').
        listings: List of Listing objects in load / construction
            order.
    """

    def __init__(self, exchange: str, listings: List[Listing]):
        self.exchange = exchange
        self.listings = listings

    def __len__(self) -> int:
        return len(self.listings)

    def __iter__(self) -> Iterator[Listing]:
        return iter(self.listings)

    @classmethod
    def load(cls, exchange: str) -> 'ExchangeUniverse':
        """
        Load an ExchangeUniverse by merging every shard in its
        directory.

        Args:
            exchange: Exchange code; resolves to
                `cache_dir / exchange / *.parquet`.

        Returns:
            ExchangeUniverse with one Listing per ticker; when
            multiple shards reference the same ticker, the row
            with the latest `refreshed_at` wins.

        Raises:
            FileNotFoundError: When the exchange directory is
                absent or contains no shard files.
        """
        exchange_dir = cache_dir / exchange
        if not exchange_dir.is_dir():
            raise FileNotFoundError(
                f'No screener cache directory for '
                f'{exchange!r} at {exchange_dir}.')
        shard_paths = sorted(exchange_dir.glob('*.parquet'))
        if not shard_paths:
            raise FileNotFoundError(
                f'No screener cache shards for {exchange!r} '
                f'in {exchange_dir}.')
        frames = [
            pd.read_parquet(p, engine=parquet_engine)
            for p in shard_paths]
        df = pd.concat(frames, ignore_index=True)
        # Dedup by ticker; latest refreshed_at wins
        df = df.sort_values('refreshed_at')
        df = df.drop_duplicates(
            subset='ticker', keep='last')
        df = df.reset_index(drop=True)
        listings = [
            _row_to_listing(row, exchange)
            for _, row in df.iterrows()]
        return cls(exchange, listings)

    def save(self) -> None:
        """
        Append the universe's listings as a new shard file.

        Writes ONLY this instance's listings to a new
        timestamped parquet in `cache_dir / exchange /`.
        Existing shards are never touched. Atomic via
        write-temp-then-rename within the exchange directory.
        """
        if not self.listings:
            return
        exchange_dir = cache_dir / self.exchange
        exchange_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime(shard_timestamp_format)
        final_path = exchange_dir / f'{stamp}.parquet'
        # Disambiguate sub-second collisions
        suffix = 0
        while final_path.exists():
            suffix += 1
            final_path = exchange_dir / f'{stamp}_{suffix}.parquet'
        tmp_path = final_path.with_suffix('.parquet.tmp')
        df = self.to_dataframe()
        df.to_parquet(
            tmp_path, engine=parquet_engine, index=False)
        tmp_path.replace(final_path)

    def filter(
        self,
        predicate: Callable[[Listing], bool]) -> 'ExchangeUniverse':
        """
        Return a new ExchangeUniverse with listings matching predicate.
        """
        kept = [l for l in self.listings if predicate(l)]
        return ExchangeUniverse(self.exchange, kept)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten the universe to a parquet-friendly DataFrame.

        One row per Listing. Series fields stored as Python
        lists (parquet's native list type via pyarrow).
        Fundamentals spread into individual scalar columns.
        The `exchange` column is NOT stored (it is encoded by
        the directory name).
        """
        rows = []
        for l in self.listings:
            row = {
                'ticker': l.ticker,
                'currency': l.currency,
                'sector': l.sector,
                'market_cap': float(l.market_cap),
                'refreshed_at': l.refreshed_at.isoformat(),
                'closes_weekly': l.closes_weekly.tolist(),
                'closes_weekly_dates': [
                    d.isoformat()
                    for d in l.closes_weekly_dates],
                'volumes_weekly': l.volumes_weekly.tolist(),
                'dividends_weekly':
                    l.dividends_weekly.tolist(),
            }
            for name in fundamental_field_names:
                row[name] = float(l.fundamentals[name])
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def staleness_days(self) -> float:
        """Age of the oldest refreshed_at, in days."""
        if not self.listings:
            return 0.0
        oldest = min(l.refreshed_at for l in self.listings)
        return (datetime.now() - oldest).total_seconds() / 86400.0


def _row_to_listing(row: pd.Series, exchange: str) -> Listing:
    """Reconstruct a Listing from a parquet row + exchange code.

    Backward-compat: parquet shards saved before Phase 8 lack
    the 13 new defeatbeta-derived columns. Missing columns
    default to NaN rather than raising, so older shards still
    load.
    """
    fundamentals = {}
    for name in fundamental_field_names:
        if name in row.index and pd.notna(row[name]):
            fundamentals[name] = float(row[name])
        else:
            fundamentals[name] = float('nan')
    return Listing(
        ticker=row['ticker'],
        exchange=exchange,
        currency=row['currency'],
        sector=row['sector'],
        market_cap=float(row['market_cap']),
        closes_weekly=np.array(
            list(row['closes_weekly']), dtype=float),
        closes_weekly_dates=pd.DatetimeIndex(
            list(row['closes_weekly_dates'])),
        volumes_weekly=np.array(
            list(row['volumes_weekly']), dtype=float),
        dividends_weekly=np.array(
            list(row['dividends_weekly']), dtype=float),
        fundamentals=fundamentals,
        refreshed_at=datetime.fromisoformat(
            row['refreshed_at']))
