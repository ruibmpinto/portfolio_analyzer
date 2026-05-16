#!/usr/bin/env python3
"""
Fundamental Fetcher Demonstration.

Shows how to fetch fundamental data in bulk with caching.
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.screening import FundamentalFetcher


def main():
    """Demonstrate fundamental data fetching."""
    print("=" * 70)
    print("FUNDAMENTAL DATA FETCHER DEMONSTRATION")
    print("=" * 70)

    # Sample tickers
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'BRK.B', 'V', 'JNJ']

    fetcher = FundamentalFetcher(
        batch_size=5,
        delay_between_batches=0.5)

    # Show cache stats before
    print("\nCache stats (before):")
    stats = fetcher.get_cache_stats()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Recent entries: {stats['recent_entries']}")

    # Fetch fundamentals
    print(f"\nFetching fundamentals for {len(tickers)} stocks...")
    print("-" * 70)

    data = fetcher.fetch_fundamentals(
        tickers,
        metrics=[
            'price', 'market_cap', 'pe_ratio',
            'dividend_yield', 'profit_margin'
        ],
        use_cache=True,
        show_progress=True)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\n{'Ticker':<8} {'Price':>8} {'Mkt Cap':>12} "
        f"{'P/E':>6} {'Div %':>6} {'Margin %':>8}")
    print("-" * 70)

    for ticker in tickers:
        if ticker in data:
            d = data[ticker]
            price = d.get('price')
            mcap = d.get('market_cap')
            pe = d.get('pe_ratio')
            div_yield = d.get('dividend_yield')
            margin = d.get('profit_margin')

            price_str = (f"${price:.2f}" if price else "N/A")

            mcap_str = (f"${mcap/1e9:.1f}B" if mcap else "N/A")

            pe_str = f"{pe:.1f}" if pe else "N/A"

            div_str = (f"{div_yield:.1f}%" if div_yield else "N/A")

            margin_str = (f"{margin:.1f}%" if margin else "N/A")

            print(
                f"{ticker:<8} {price_str:>8} "
                f"{mcap_str:>12} {pe_str:>6} "
                f"{div_str:>6} {margin_str:>8}")
        else:
            print(f"{ticker:<8} (fetch failed)")

    # Show cache stats after
    print("\n" + "=" * 70)
    print("Cache stats (after):")
    stats = fetcher.get_cache_stats()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Recent entries: {stats['recent_entries']}")
    print(f"  Database: {stats['database_path']}")

    # Demonstrate cache hit
    print("\n" + "=" * 70)
    print("Testing cache (re-fetching same tickers)...")
    print("-" * 70)

    data2 = fetcher.fetch_fundamentals(
        tickers[:3],  # Just first 3
        use_cache=True,
        show_progress=True)

    print(f"\nFetched {len(data2)} tickers (should be instant - from cache)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
