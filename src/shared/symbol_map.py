"""
IBKR symbol -> canonical project symbol mapping.

Interactive Brokers uses non-standard tickers for some SIX
Swiss Exchange listings (e.g. 'HOLNz' instead of 'HOLN.SW',
'SMICHA' instead of 'SMICHA.SW'). The rest of the project,
including the Degiro pipeline and the Yahoo Finance data
provider, uses the '.SW' suffix form. This module holds the
translation table.

Extend `ibkr_symbol_map` whenever a new SIX-listed symbol
appears in IBKR Activity Statements. Entries for non-Swiss
exchanges are not needed: IBKR symbols match the canonical
form for NASDAQ, NYSE, ARCA, BATS, LSE-ETF, etc.

Constants
---------
ibkr_symbol_map
    Dict mapping raw IBKR symbol -> canonical project symbol.
"""


ibkr_symbol_map = {
    'HOLNz': 'HOLN.SW',     # Holcim Ltd (Trades section form)
    'HOLN': 'HOLN.SW',      # Holcim Ltd (Dividends description form)
    'SMICHA': 'SMICHA.SW',  # UBS ETF SMI, ISIN CH0017142719
    'VUAA': 'VUAA.L',       # Vanguard S&P 500 UCITS ETF, LSE
}
