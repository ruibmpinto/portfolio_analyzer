"""Tests for CHF pricing helper for non-held tickers."""

import pytest

from src.modelling.rebalancing.pricing import fetch_chf_price


class FakeProvider:
    """Minimal data_provider stub for pricing tests."""

    def __init__(self, prices):
        self._prices = prices

    def get_current_price(self, ticker):
        if ticker not in self._prices:
            raise ValueError(
                f'FakeProvider: no price for {ticker!r}.')
        return self._prices[ticker]


def test_fetch_chf_price_for_chf_ticker():
    """No FX call for a CHF-denominated ticker."""
    fp = FakeProvider({'NESN.SW': 95.0})
    assert fetch_chf_price(fp, 'NESN.SW', 'CHF') == 95.0


def test_fetch_chf_price_for_usd_ticker():
    """USD price multiplied by USDCHF=X."""
    fp = FakeProvider({'AAPL': 200.0, 'USDCHF=X': 0.9})
    price = fetch_chf_price(fp, 'AAPL', 'USD')
    assert price == pytest.approx(180.0)


def test_fetch_chf_price_for_gbp_ticker():
    """GBP price multiplied by GBPCHF=X."""
    fp = FakeProvider({'BARC.L': 1.50, 'GBPCHF=X': 1.1})
    price = fetch_chf_price(fp, 'BARC.L', 'GBP')
    assert price == pytest.approx(1.65)


def test_fetch_chf_price_for_gbx_pence_ticker():
    """GBp pence price divided by 100, then GBPCHF=X."""
    fp = FakeProvider({'AAF.L': 150.0, 'GBPCHF=X': 1.1})
    price = fetch_chf_price(fp, 'AAF.L', 'GBp')
    assert price == pytest.approx(1.65)


def test_fetch_chf_price_raises_when_native_missing():
    """No native quote raises a RuntimeError."""
    fp = FakeProvider({})
    with pytest.raises(RuntimeError, match='AAPL'):
        fetch_chf_price(fp, 'AAPL', 'USD')


def test_fetch_chf_price_raises_when_fx_missing():
    """Missing FX pair raises a RuntimeError naming the pair."""
    fp = FakeProvider({'AAPL': 200.0})
    with pytest.raises(RuntimeError, match='USDCHF=X'):
        fetch_chf_price(fp, 'AAPL', 'USD')
