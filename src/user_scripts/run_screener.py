"""
CLI: rank tickers across cached exchange universes.

Loads one or more ExchangeUniverses, applies the requested
criteria, computes composite scores, and writes the top-N
results to Excel or JSON.

Usage:
    python -m src.user_scripts.run_screener \\
        --exchange NASDAQ NYSE \\
        --criteria momentum,cagr,sharpe,pe_value \\
        --top-n 50 \\
        --output-format excel \\
        --output results/screener_NASDAQ_NYSE.xlsx
"""

import argparse
import json
import pathlib
import sys
from typing import List


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='run_screener.py',
        description='Rank tickers across cached exchanges.')
    p.add_argument(
        '--exchange', required=True, nargs='+',
        help='One or more exchange codes (space-separated).')
    p.add_argument(
        '--criteria', required=True,
        help='Comma-separated criterion names (e.g. '
             '"momentum,cagr,sharpe").')
    p.add_argument(
        '--top-n', type=int, default=None,
        help='Truncate output to top N candidates.')
    p.add_argument(
        '--output-format', choices=['excel', 'json'],
        default='excel',
        help='Output file format.')
    p.add_argument(
        '--output', required=True,
        help='Output file path.')
    return p.parse_args(argv)


def _build_screener(exchanges: List[str], criteria_csv: str):
    from src.screening.screener import Screener
    from src.screening.criteria import get_criterion
    screener = Screener(exchanges)
    for name in criteria_csv.split(','):
        name = name.strip()
        if name:
            screener.add_criterion(get_criterion(name))
    return screener


def _write_output(
    candidates, out_path: pathlib.Path,
    output_format: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == 'excel':
        import pandas as pd
        rows = [
            {
                'ticker': c.ticker, 'exchange': c.exchange,
                'currency': c.currency, 'sector': c.sector,
                'market_cap': c.market_cap,
                'composite_score': c.composite_score,
                **dict(c.metrics),
            }
            for c in candidates]
        pd.DataFrame(rows).to_excel(out_path, index=False)
    else:
        rows = [
            {
                'ticker': c.ticker, 'exchange': c.exchange,
                'currency': c.currency, 'sector': c.sector,
                'market_cap': c.market_cap,
                'composite_score': c.composite_score,
                'metrics': dict(c.metrics),
            }
            for c in candidates]
        out_path.write_text(json.dumps(rows, indent=2))


def main(argv=None):
    """Entry point for `python -m src.user_scripts.run_screener`."""
    args = _parse_args(
        argv if argv is not None else sys.argv[1:])
    screener = _build_screener(args.exchange, args.criteria)
    candidates = screener.rank(top_n=args.top_n)
    out_path = pathlib.Path(args.output)
    _write_output(candidates, out_path, args.output_format)
    print(
        f'Screener output written to {out_path} '
        f'({len(candidates)} candidates, '
        f'criteria={args.criteria}, '
        f'exchanges={args.exchange}).')


if __name__ == '__main__':
    main()
