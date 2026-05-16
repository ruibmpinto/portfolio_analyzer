"""Tests for the Degiro loader.

Buys and sells come from the transactions CSV (one row per
fill, signed Quantity, native Price, fees in their own
columns). Dividends and withholding tax come from the account
statement, filtered to those two row types.
"""

import pathlib

import pandas as pd
import pytest

from src.analysis.loaders.csv_loader import (
    CSVLoader,
    default_degiro_path,
    load_isin_to_ticker)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Schema constants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_transactions_header = (
    'Date,Time,Product,ISIN,Reference exchange,Venue,'
    'Quantity,Price,,Local value,,Value CHF,Exchange rate,'
    'AutoFX Fee,Transaction and/or third party fees CHF,'
    'Total CHF,Order ID,')

_statement_header = (
    'Date,Time,Value date,Product,ISIN,Description,FX,'
    'Change,,Balance,,Order Id')


def _write_transactions(tmp_path: pathlib.Path, *rows: str) -> pathlib.Path:
    """Write a Degiro transactions CSV fixture."""
    body = '\n'.join((_transactions_header, *rows))
    path = tmp_path / 'transactions.csv'
    path.write_text(body + '\n')
    return path


def _write_statement(tmp_path: pathlib.Path, *rows: str) -> pathlib.Path:
    """Write a Degiro account-statement CSV fixture."""
    body = '\n'.join((_statement_header, *rows))
    path = tmp_path / 'statement.csv'
    path.write_text(body + '\n')
    return path


def _empty_stmt(tmp_path: pathlib.Path) -> str:
    """Header-only statement for trade-only test cases."""
    return str(_write_statement(tmp_path))


def _map():
    """ISIN map covering the fixtures."""
    return {
        'US67066G1040': 'NVDA',
        'US36467W1099': 'GME',
        'CH0244767585': 'UBSG.SW',
        'DE0007030009': 'RHM.DE',
        'US7672921050': 'RIOT',
        'LU0953791844': '0P000101KR.F',
        'NLFLATEXACNT': None}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Trade parsing (transactions CSV)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_usd_buy_emits_one_transaction(tmp_path):
    # Positive Quantity = buy. Fees and AutoFX are in CHF in
    # the source file; the loader converts them back to native.
    rows = [
        '15-03-2024,10:00,NVIDIA CORP,US67066G1040,NDQ,SOHO,'
        '2,100.00,USD,-200.00,USD,-178.00,1.1236,-1.50,-1.85,'
        '-181.35,ord-1,']
    path = _write_transactions(tmp_path, *rows)
    trans = CSVLoader().load_csv_degiro(
        str(path), statement_path=_empty_stmt(tmp_path),
        isin_to_ticker=_map())
    assert len(trans) == 1
    t = trans[0]
    assert t.ticker == 'NVDA'
    assert t.operation == 'buy'
    assert t.amount == 2.0
    assert t.price == 100.00
    assert t.currency == 'USD'
    # Native fee = |CHF fee| / FX rate
    assert t.fee == pytest.approx(1.85 / 1.1236, rel=1e-9)
    assert t.auto_fx_fee == pytest.approx(1.50 / 1.1236, rel=1e-9)


def test_chf_native_buy_keeps_fee_in_chf(tmp_path):
    # CHF trade has Exchange rate = 1.0 and AutoFX Fee = 0.
    rows = [
        '30-01-2024,09:00,UBS GROUP AG,CH0244767585,SWX,SWX,'
        '1,25.85,CHF,-25.85,CHF,-25.85,1.0000,0.00,-6.00,'
        '-31.85,ord-2,']
    path = _write_transactions(tmp_path, *rows)
    trans = CSVLoader().load_csv_degiro(
        str(path), statement_path=_empty_stmt(tmp_path),
        isin_to_ticker=_map())
    assert len(trans) == 1
    t = trans[0]
    assert t.ticker == 'UBSG.SW'
    assert t.operation == 'buy'
    assert t.amount == 1.0
    assert t.currency == 'CHF'
    assert t.fee == pytest.approx(6.00)
    assert t.auto_fx_fee == 0.0


def test_negative_quantity_emits_sell(tmp_path):
    rows = [
        '01-05-2025,18:51,GAMESTOP CORP CLASS A,US36467W1099,'
        'NSY,BATS,-2,27.745,USD,55.49,USD,46.02,1.2058,-0.50,'
        '-1.88,43.64,ord-3,']
    path = _write_transactions(tmp_path, *rows)
    trans = CSVLoader().load_csv_degiro(
        str(path), statement_path=_empty_stmt(tmp_path),
        isin_to_ticker=_map())
    assert len(trans) == 1
    t = trans[0]
    assert t.operation == 'sell'
    assert t.ticker == 'GME'
    assert t.amount == 2.0
    assert t.price == pytest.approx(27.745)


def test_split_fill_under_one_order_id_aggregates(tmp_path):
    # Two fills, same Order ID: aggregate qty, VWAP price.
    rows = [
        '06-05-2025,16:21,RIOT PLATFORMS INC,US7672921050,'
        'NDQ,BATS,41,9.24,USD,-378.84,USD,-309.88,1.2225,'
        '-0.77,,-310.66,ord-rf,',
        '06-05-2025,16:21,RIOT PLATFORMS INC,US7672921050,'
        'NDQ,BATS,25,9.24,USD,-231.00,USD,-188.95,1.2225,'
        '-0.47,-1.88,-191.31,ord-rf,']
    path = _write_transactions(tmp_path, *rows)
    trans = CSVLoader().load_csv_degiro(
        str(path), statement_path=_empty_stmt(tmp_path),
        isin_to_ticker=_map())
    assert len(trans) == 1
    t = trans[0]
    assert t.ticker == 'RIOT'
    assert t.amount == 66.0
    assert t.price == pytest.approx(9.24)
    assert t.fee == pytest.approx(1.88 / 1.2225, rel=1e-9)
    assert t.auto_fx_fee == pytest.approx(
        (0.77 + 0.47) / 1.2225, rel=1e-9)


def test_curly_apostrophe_price_does_not_break_loader(tmp_path):
    # The old statement parser dropped these rows. The new
    # transactions parser reads Price as a plain float.
    rows = [
        '13-08-2025,09:02,RHEINMETALL AG,DE0007030009,XET,'
        'XETA,-1,1594.00,EUR,1594.00,EUR,1501.88,1.0613,-3.75,'
        '-4.63,1493.49,ord-rhm,']
    path = _write_transactions(tmp_path, *rows)
    trans = CSVLoader().load_csv_degiro(
        str(path), statement_path=_empty_stmt(tmp_path),
        isin_to_ticker=_map())
    assert len(trans) == 1
    t = trans[0]
    assert t.ticker == 'RHM.DE'
    assert t.operation == 'sell'
    assert t.amount == 1.0
    assert t.price == 1594.00


def test_blank_quantity_is_recovered_from_local_value(tmp_path):
    # Mutual-fund subscription: Degiro leaves Quantity blank.
    # The loader recovers |Local value| / Price and signs it
    # from Local value (negative = buy → positive qty).
    rows = [
        '04-07-2025,16:00,GOLDMAN SACHS EUROPE HIGH YIELD,'
        'LU0953791844,KFS,,,521.71,EUR,-1999.71,EUR,-1870.54,'
        '1.0691,-4.68,-4.59,-1879.81,ord-fund,']
    path = _write_transactions(tmp_path, *rows)
    trans = CSVLoader().load_csv_degiro(
        str(path), statement_path=_empty_stmt(tmp_path),
        isin_to_ticker=_map())
    assert len(trans) == 1
    t = trans[0]
    assert t.operation == 'buy'
    # 1999.71 / 521.71 = 3.83298...
    assert t.amount == pytest.approx(3.833, rel=1e-3)
    assert t.price == pytest.approx(521.71)
    assert t.currency == 'EUR'


def test_blank_quantity_with_no_local_value_raises(tmp_path):
    # No way to recover; raise rather than guess.
    rows = [
        '01-01-2024,10:00,Random,LU0953791844,KFS,,,521.71,'
        'EUR,,EUR,,1.0,0,0,0,ord-bad,']
    path = _write_transactions(tmp_path, *rows)
    with pytest.raises(RuntimeError, match='Local value'):
        CSVLoader().load_csv_degiro(
            str(path), statement_path=_empty_stmt(tmp_path),
            isin_to_ticker=_map())


def test_blank_order_id_rows_get_synthetic_groups(tmp_path):
    # Corporate-action / split rows arrive with no Order ID.
    # The loader keeps them under a synthetic group so the
    # analyzer's split-row filter can decide what to drop.
    rows = [
        '10-06-2024,10:53,NVIDIA CORP,US67066G1040,NDQ,,'
        '20,120.888,USD,-2417.76,USD,-2165.52,1.1165,0,0,'
        '-2165.52,,']
    path = _write_transactions(tmp_path, *rows)
    trans = CSVLoader().load_csv_degiro(
        str(path), statement_path=_empty_stmt(tmp_path),
        isin_to_ticker=_map())
    assert len(trans) == 1
    assert trans[0].ticker == 'NVDA'
    assert trans[0].amount == 20.0
    assert trans[0].operation == 'buy'


def test_schema_drift_pulls_order_id_from_trailing_column(tmp_path):
    # Simulate the post-2026-03-26 Degiro format: the UUID
    # slips out of the 'Order ID' column into the trailing
    # 'Unnamed: 17' column. The loader pulls it back.
    rows = [
        '08-04-2026,16:00,ADVANCED MICRO DEVICES INC,'
        'US0079031078,NDQ,CDED,-1,228.50,USD,228.50,USD,180.34,'
        '1.2671,-0.45,-1.85,178.04,,5a9b6294-6c68-4266-bae2-8e84b5a8cbb8']
    path = _write_transactions(tmp_path, *rows)
    trans = CSVLoader().load_csv_degiro(
        str(path), statement_path=_empty_stmt(tmp_path),
        isin_to_ticker={'US0079031078': 'AMD'})
    assert len(trans) == 1
    assert trans[0].ticker == 'AMD'
    assert trans[0].operation == 'sell'
    assert trans[0].amount == 1.0


def test_tombstone_row_is_dropped(tmp_path):
    # All-zero placeholder row with no Order ID (fund delisting
    # marker). Should be silently dropped.
    rows = [
        '13-10-2025,15:10,GOLDMAN SACHS,LU0953791844,KFS,,,'
        '0.0000,EUR,0.00,EUR,0.00,1.0,0.00,,0.00,']
    path = _write_transactions(tmp_path, *rows)
    trans = CSVLoader().load_csv_degiro(
        str(path), statement_path=_empty_stmt(tmp_path),
        isin_to_ticker=_map())
    assert trans == []


def test_flatex_cash_account_isin_is_silently_skipped(tmp_path):
    rows = [
        '01-01-2024,00:00,FLATEX CHF BANKACCOUNT,NLFLATEXACNT,'
        'KFS,,1,1.00,CHF,-1.00,CHF,-1.00,1.0,0,0,-1.00,ord-cf,']
    path = _write_transactions(tmp_path, *rows)
    trans = CSVLoader().load_csv_degiro(
        str(path), statement_path=_empty_stmt(tmp_path),
        isin_to_ticker=_map())
    assert trans == []


def test_unknown_isin_raises_with_path_hint(tmp_path):
    rows = [
        '15-03-2024,10:00,Random Co,XX9999999999,NDQ,SOHO,'
        '1,10.00,USD,-10.00,USD,-9.00,1.1,-0.05,-0.05,'
        '-9.10,ord-9,']
    path = _write_transactions(tmp_path, *rows)
    with pytest.raises(RuntimeError, match=r'XX9999999999'):
        CSVLoader().load_csv_degiro(
            str(path), statement_path=_empty_stmt(tmp_path),
            isin_to_ticker=_map())


def test_missing_transactions_columns_raises_value_error(tmp_path):
    bad_path = tmp_path / 'bad.csv'
    bad_path.write_text('foo,bar\n1,2\n')
    with pytest.raises(ValueError, match=r'Missing required columns'):
        CSVLoader().load_csv_degiro(
            str(bad_path), statement_path=_empty_stmt(tmp_path))


def test_file_not_found_raises_typed_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVLoader().load_csv_degiro(
            str(tmp_path / 'nope.csv'), statement_path=_empty_stmt(tmp_path))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Income parsing (statement CSV)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_dividend_row_emits_positive_price(tmp_path):
    trades_path = _write_transactions(tmp_path)  # empty body
    stmt_rows = [
        '05-04-2024,10:00,05-04-2024,NVIDIA CORP,US67066G1040,'
        'Dividend,,USD,0.20,USD,0.20,']
    stmt_path = _write_statement(tmp_path, *stmt_rows)
    trans = CSVLoader().load_csv_degiro(
        str(trades_path), statement_path=str(stmt_path),
        isin_to_ticker=_map())
    assert len(trans) == 1
    t = trans[0]
    assert t.operation == 'dividend'
    assert t.ticker == 'NVDA'
    assert t.price == pytest.approx(0.20)
    assert t.amount == 1
    assert t.currency == 'USD'


def test_dividend_tax_row_emits_negative_price(tmp_path):
    trades_path = _write_transactions(tmp_path)
    stmt_rows = [
        '05-04-2024,10:01,05-04-2024,NVIDIA CORP,US67066G1040,'
        'Dividend Tax,,USD,-0.05,USD,0.15,']
    stmt_path = _write_statement(tmp_path, *stmt_rows)
    trans = CSVLoader().load_csv_degiro(
        str(trades_path), statement_path=str(stmt_path),
        isin_to_ticker=_map())
    assert len(trans) == 1
    t = trans[0]
    assert t.operation == 'tax'
    assert t.price == pytest.approx(-0.05)


def test_delisting_row_emits_a_sell(tmp_path):
    # DELISTING rows live only in the statement (the
    # transactions CSV doesn't carry them). Loader must
    # emit a canonical sell so the position closes out.
    trades_path = _write_transactions(tmp_path)
    stmt_rows = [
        '24-10-2025,15:10,13-10-2025,GOLDMAN SACHS,'
        'LU0953791844,DELISTING: Sell 3.833 Goldman Sachs '
        'Europe High Yield (Former NN) - N Cap EUR@0 EUR '
        '(LU0953791844),,EUR,0.00,EUR,0.00,']
    stmt_path = _write_statement(tmp_path, *stmt_rows)
    trans = CSVLoader().load_csv_degiro(
        str(trades_path), statement_path=str(stmt_path),
        isin_to_ticker=_map())
    assert len(trans) == 1
    t = trans[0]
    assert t.ticker == '0P000101KR.F'
    assert t.operation == 'sell'
    assert t.amount == pytest.approx(3.833)
    assert t.price == 0.0
    assert t.currency == 'EUR'


def test_statement_bookkeeping_rows_are_ignored(tmp_path):
    trades_path = _write_transactions(tmp_path)
    stmt_rows = [
        '11-05-2026,10:30,11-05-2026,,,'
        'Degiro Cash Sweep Transfer,,CHF,891.25,CHF,6.33,',
        '25-01-2024,13:40,24-01-2024,,,'
        'Deposit,,CHF,1.00,CHF,1.00,',
        '23-02-2026,10:43,31-01-2026,,,'
        'DEGIRO Exchange Connection Fee 2026 (Nasdaq - NDQ),,'
        'EUR,-2.50,EUR,-12.50,']
    stmt_path = _write_statement(tmp_path, *stmt_rows)
    trans = CSVLoader().load_csv_degiro(
        str(trades_path), statement_path=str(stmt_path),
        isin_to_ticker=_map())
    assert trans == []


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_load_isin_to_ticker_strips_comment_key(tmp_path):
    path = tmp_path / 'isin.json'
    path.write_text(
        '{"_comment": "ignore me", "US1234567890": "AAPL"}')
    mapping = load_isin_to_ticker(path)
    assert mapping == {'US1234567890': 'AAPL'}


def test_default_transactions_path_returns_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        'src.analysis.loaders.csv_loader.'
        'default_degiro_transactions_dir', tmp_path)
    (tmp_path / '20230101_20231231_degiro_transactions.csv').write_text('x')
    (tmp_path / '20240101_20241231_degiro_transactions.csv').write_text('x')
    (tmp_path / '20240101_20260514_degiro_transactions.csv').write_text('x')
    latest = default_degiro_path('transactions')
    assert latest.name == '20240101_20260514_degiro_transactions.csv'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# End-to-end smoke test on the real bundled files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_real_statement_parses_without_raising():
    """End-to-end contract check on the bundled CSVs.

    Asserts structural invariants only — never any per-ticker
    quantity. The bundled CSVs evolve as new trades land; this
    test should keep passing as long as the loader honours its
    contract.
    """
    trans_path = default_degiro_path('transactions')
    stmt_path = default_degiro_path('statement')
    if trans_path is None or stmt_path is None:
        pytest.skip('Degiro fixtures not bundled')

    trans = CSVLoader().load_csv_degiro(
        str(trans_path), statement_path=str(stmt_path))

    # Per-Transaction structural invariants
    valid_ops = {'buy', 'sell', 'dividend', 'tax'}
    from datetime import datetime as _dt
    for t in trans:
        assert t.ticker
        assert t.operation in valid_ops
        assert t.amount >= 0
        assert isinstance(t.date, _dt)
        assert t.currency
        assert t.fee >= 0
        assert t.auto_fx_fee >= 0

    # Build a cheap "expected count" view from the raw files
    # so the schema-drift regression catches future row drops
    isin_map = load_isin_to_ticker()
    def _maps_to_ticker(isin):
        return isin_map.get(isin) is not None

    raw_tx = pd.read_csv(str(trans_path)).dropna(subset=['Date'])
    # Coalesce Order ID with the trailing column the way the
    # loader does — this is what we are regression-testing
    raw_tx['Order ID'] = raw_tx['Order ID'].fillna(
        raw_tx.iloc[:, -1])
    # Drop tombstones (no quantity, no value, no id)
    keep = ~(raw_tx['Quantity'].isna()
             & (raw_tx['Local value'].fillna(0) == 0)
             & raw_tx['Order ID'].isna())
    raw_tx = raw_tx[keep]
    # Each Order ID is one transaction; empty-ID rows count
    # as their own one-fill groups
    mapped = raw_tx[raw_tx['ISIN'].apply(_maps_to_ticker)]
    expected_trades = (mapped['Order ID'].dropna().nunique()
                       + int(mapped['Order ID'].isna().sum()))

    raw_stmt = pd.read_csv(str(stmt_path)).dropna(subset=['Date'])
    mapped_stmt = raw_stmt[
        raw_stmt['ISIN'].apply(_maps_to_ticker)]
    expected_delisting = int(
        mapped_stmt['Description'].fillna('')
        .str.startswith('DELISTING:').sum())
    expected_div = int(
        (mapped_stmt['Description'] == 'Dividend').sum())
    expected_tax = int(
        (mapped_stmt['Description'] == 'Dividend Tax').sum())

    actual_trades = sum(
        1 for t in trans if t.operation in ('buy', 'sell'))
    actual_div = sum(
        1 for t in trans if t.operation == 'dividend')
    actual_tax = sum(
        1 for t in trans if t.operation == 'tax')

    # Invariants: the loader must not silently drop rows
    assert actual_trades == expected_trades + expected_delisting
    assert actual_div == expected_div
    assert actual_tax == expected_tax
