"""
CLI: refresh the per-exchange screener parquet cache.

Reads a one-ticker-per-line file from
data/exchange_tickers/<exchange>.txt and dispatches to
src.screening.cache_refresh.refresh_exchange. Resumable by
default — re-running picks up where the last run left off.

Usage:
    python -m src.user_scripts.refresh_screener_cache \\
        --exchange NASDAQ
    python -m src.user_scripts.refresh_screener_cache \\
        --exchange NASDAQ --rate-limit 1.0 --no-resume
"""

import argparse
import pathlib
import sys
from typing import List


tickers_dir = pathlib.Path(
    __file__).resolve().parents[2] / 'data' / 'exchange_tickers'


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='refresh_screener_cache.py',
        description=(
            'Refresh the per-exchange screener parquet cache.'))
    p.add_argument(
        '--exchange', required=True,
        help='Exchange code (e.g. NASDAQ, SIX, XETRA).')
    p.add_argument(
        '--rate-limit', type=float, default=0.5,
        help='Seconds between per-ticker yfinance calls.')
    p.add_argument(
        '--no-resume', action='store_true',
        help='Refetch every ticker even if cached.')
    return p.parse_args(argv)


def _load_ticker_list(exchange: str) -> List[str]:
    """Read tickers_dir/<exchange>.txt; one ticker per line."""
    path = tickers_dir / f'{exchange}.txt'
    if not path.exists():
        raise SystemExit(
            f'refresh_screener_cache: no ticker list at '
            f'{path}. Create it (one ticker per line) and '
            f'retry.')
    lines = path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def main(argv=None):
    """Entry point for `python -m src.user_scripts.refresh_screener_cache`."""
    args = _parse_args(
        argv if argv is not None else sys.argv[1:])
    tickers = _load_ticker_list(args.exchange)
    from src.screening.cache_refresh import refresh_exchange
    refresh_exchange(
        exchange=args.exchange,
        tickers=tickers,
        resume=not args.no_resume,
        rate_limit_sleep=args.rate_limit)
    print(
        f'Refresh complete: exchange={args.exchange}, '
        f'tickers_attempted={len(tickers)}.')


if __name__ == '__main__':
    main()
