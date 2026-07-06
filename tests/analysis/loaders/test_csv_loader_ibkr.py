"""Tests for the IBKR Activity Statement loader.

IBKR uses different symbol forms in different sections: the
Trades section names Holcim ``HOLNz`` while the Dividends
description names it ``HOLN``. Both must normalise to the single
canonical ``HOLN.SW`` ticker via ``ibkr_symbol_map`` so the
position and its dividend are attributed to the same holding.
"""

from src.analysis.loaders.csv_loader import CSVLoader


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Schema constants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_trades_header = (
    'Trades,Header,DataDiscriminator,Asset Category,Currency,'
    'Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,'
    'Comm/Fee,Basis,Realized P/L,Realized P/L %,MTM P/L,Code')

_dividends_header = 'Dividends,Header,Currency,Date,Description,Amount'


def _write_statement(tmp_path, *rows):
    """Write a minimal IBKR Activity Statement CSV fixture."""
    body = '\n'.join(rows)
    path = tmp_path / 'stmt_ibkr_statement.csv'
    path.write_text(body + '\n')
    return str(path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cross-section symbol reconciliation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_holcim_dividend_resolves_to_trade_ticker(tmp_path):
    """HOLNz trade and HOLN dividend both normalise to HOLN.SW."""
    path = _write_statement(
        tmp_path,
        _trades_header,
        ('Trades,Data,Order,Stocks,CHF,HOLNz,'
         '"2025-11-13, 05:48:17",54,73.92,73.14,-3991.68,'
         '-2.83584,3994.51584,0,0,-42.12,O'),
        _dividends_header,
        ('Dividends,Data,CHF,2026-05-21,'
         'HOLN(CH0012214059) Cash Dividend CHF 1.70 per Share '
         '(Return of Capital),91.8'))
    txns = CSVLoader().load_csv_ibkr(path)
    assert {t.ticker for t in txns} == {'HOLN.SW'}
    dividends = [t for t in txns if t.operation == 'dividend']
    assert len(dividends) == 1
    assert dividends[0].ticker == 'HOLN.SW'
# -------------------------------------------------------------------------
def test_dividend_symbol_normalises_regardless_of_parenthetical(tmp_path):
    """The (ISIN|conid) parenthetical does not affect mapping.

    Two SMICHA dividend rows carry different parentheticals
    (an ISIN and a numeric conid); both resolve to SMICHA.SW
    purely from the base symbol via ``ibkr_symbol_map``.
    """
    path = _write_statement(
        tmp_path,
        _dividends_header,
        ('Dividends,Data,CHF,2026-03-12,'
         'SMICHA(CH0017142719) Cash Dividend CHF 0.71 per Share '
         '(Ordinary Dividend),38.34'),
        ('Dividends,Data,CHF,2026-03-16,'
         'SMICHA(26965707) Cash Dividend CHF 0.68 per Share '
         '(Ordinary Dividend),36.72'))
    txns = CSVLoader().load_csv_ibkr(path)
    assert [t.ticker for t in txns] == ['SMICHA.SW', 'SMICHA.SW']
