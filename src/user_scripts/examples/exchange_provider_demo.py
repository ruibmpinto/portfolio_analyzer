#!/usr/bin/env python3
"""
Exchange Provider Demonstration.

Shows how to use the ExchangeProvider to get ticker lists
for different exchanges.
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..')))

from src.screening import ExchangeProvider


def main():
    """Demonstrate Exchange Provider usage."""
    print("=" * 60)
    print("EXCHANGE PROVIDER DEMONSTRATION")
    print("=" * 60)

    provider = ExchangeProvider()

    # Show supported exchanges
    print("\nSupported exchanges:")
    for exchange in provider.get_all_exchanges():
        info = provider.get_exchange_info(exchange)
        print(f"  {info['code']:8s} - {info['name']}")

    # Get NASDAQ tickers
    print("\n" + "=" * 60)
    print("NASDAQ Example")
    print("=" * 60)

    nasdaq_tickers = provider.get_tickers('NASDAQ')
    print(f"\nTotal NASDAQ tickers: {len(nasdaq_tickers)}")
    print(f"First 20 tickers:")
    for i, ticker in enumerate(nasdaq_tickers[:20], 1):
        print(f"  {i:2d}. {ticker}")

    # Get NYSE tickers
    print("\n" + "=" * 60)
    print("NYSE Example")
    print("=" * 60)

    nyse_tickers = provider.get_tickers('NYSE')
    print(f"\nTotal NYSE tickers: {len(nyse_tickers)}")
    print(f"First 20 tickers:")
    for i, ticker in enumerate(nyse_tickers[:20], 1):
        print(f"  {i:2d}. {ticker}")

    # Show exchange info
    print("\n" + "=" * 60)
    print("Exchange Information")
    print("=" * 60)

    for exchange in ['NASDAQ', 'NYSE', 'AMEX']:
        info = provider.get_exchange_info(exchange)
        print(f"\n{exchange}:")
        print(f"  Name: {info['name']}")
        print(f"  Ticker Count: {info['ticker_count']}")
        print(f"  Last Updated: {info['last_updated']}")
        print(f"  Source: {info['source']}")

    print("\n" + "=" * 60)
    print("Note: Run update_exchange_lists.py to update")
    print("ticker lists from official sources.")
    print("=" * 60)


if __name__ == '__main__':
    main()
