"""
CSV data loading and validation.

Handles loading portfolio transaction data from CSV files
with proper validation and error handling. The Degiro loader
parses the broker's raw Account Statement (12-column schema)
directly; no upstream postprocessing step.
"""

import csv
import json
import pathlib
import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from src.analysis.core.transaction import Transaction
from src.shared.symbol_map import ibkr_symbol_map


# Default ISIN -> ticker map and statement directories. The
# latest-file helpers pick the lexicographically-newest match,
# which works because filenames embed the period as
# <YYYYMMDD_YYYYMMDD>_<broker>_statement.csv.
default_isin_map_path = pathlib.Path('data/isin_to_ticker.json')

default_degiro_statements_dir = pathlib.Path('reports/degiro/statements')

default_degiro_transactions_dir = pathlib.Path('reports/degiro/transactions')

default_ibkr_statements_dir = pathlib.Path('reports/ibkr/statements')


def default_degiro_path(
        kind: str = 'transactions') -> Optional[pathlib.Path]:
    """Newest Degiro export of the requested kind.

    Args:
        kind: ``'transactions'`` returns the trades-only export
            (signed Quantity, native Price, fees split out).
            ``'statement'`` returns the full account statement
            (dividends + tax + cash sweeps + ...).

    Returns:
        Path to the lexicographically-newest matching CSV, or
        None when the directory is missing / empty.

    Raises:
        ValueError: When `kind` is neither 'transactions' nor
            'statement'.
    """
    # Map each kind to (directory, filename glob)
    if kind == 'transactions':
        return _latest_csv(
            default_degiro_transactions_dir,
            '*_degiro_transactions.csv')
    if kind == 'statement':
        return _latest_csv(
            default_degiro_statements_dir,
            '*_degiro_statement.csv')
    raise ValueError(
        f"default_degiro_path: kind must be 'transactions' "
        f"or 'statement', got {kind!r}.")


def default_ibkr_statement_path() -> Optional[pathlib.Path]:
    """Return the lexicographically-newest IBKR statement CSV."""
    return _latest_csv(
        default_ibkr_statements_dir, '*_ibkr_statement.csv')


def _latest_csv(
        directory: pathlib.Path,
        pattern: str) -> Optional[pathlib.Path]:
    """Newest file in `directory` matching `pattern`."""
    
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern))

    return matches[-1] if matches else None


def _fee_cell(value, column: str, order_id) -> float:
    """Parse a Degiro fee column.

    Genuinely-blank or NaN cells legitimately mean "no fee on
    this fill" (the broker stamps fees on one of the fills when
    an order is partially filled across rows; the others are
    blank). Any other unparseable value is a data-format
    surprise and raises so we never silently swallow it.
    """
    if value is None:
        return 0.0
    
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        pass
    
    if isinstance(value, str) and value.strip() == '':
        return 0.0
    
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'Unparseable {column!r} value {value!r} on '
            f'Order ID {order_id!r}') from exc


def load_isin_to_ticker(
        path: Optional[pathlib.Path] = None) -> Dict[str, Optional[str]]:
    """Load ISIN -> ticker map from JSON.

    The reserved key ``_comment`` is stripped on load. Tickers
    may be ``None`` for known-but-ignorable ISINs (e.g. the
    Flatex cash account).

    Args:
        path: Optional override path. Defaults to
            ``data/isin_to_ticker.json``.

    Returns:
        Dict ISIN -> ticker (or None).

    Raises:
        FileNotFoundError: When the JSON file is missing.
    """
    # Fall back to the project-wide default path
    p = path or default_isin_map_path
    if not p.exists():
        raise FileNotFoundError(
            f'ISIN->ticker map not found at {p}. '
            f'Create one or pass an explicit dict.')
    with open(p) as f:
        raw = json.load(f)
    # Strip the reserved comment key from the returned dict
    out = {}
    for k, v in raw.items():
        if k == '_comment':
            continue
        out[k] = v
    return out


class CSVLoader:
    """
    Loads and validates transaction data from CSV files.

    Degiro: two source files. Buys/sells come from the
    trades-only export (one row per fill, signed Quantity,
    native Price, fees in their own columns). Dividends and
    withholding tax come from the account statement, which is
    the only place those rows appear.

    IBKR: parses an Activity Statement (multi-section CSV).
    """

    # Required column headers in the trades-only Degiro export
    degiro_transactions_required_columns = [
        'Date', 'Time', 'Product', 'ISIN', 'Reference exchange',
        'Venue', 'Quantity', 'Price', 'Local value',
        'Value CHF', 'Exchange rate', 'AutoFX Fee',
        'Transaction and/or third party fees CHF',
        'Total CHF', 'Order ID']

    # Required column headers in the raw Degiro Account
    # Statement export. Used only for the dividend / tax pass.
    degiro_statement_required_columns = [
        'Date', 'Time', 'Value date', 'Product', 'ISIN',
        'Description', 'FX', 'Change', 'Balance', 'Order Id']

    # Back-compat alias (some old tests import this name)
    degiro_required_columns = degiro_statement_required_columns

    # IBKR Activity Statements are multi-section CSVs. Each
    # entry maps a section name to (kind, operation):
    #   kind='trade'  -> dispatch to _parse_ibkr_trade_row
    #   kind='income' -> dispatch to _parse_ibkr_income_row
    #                    using the given operation tag.
    # Extend this dict to surface new sections; the loop in
    # load_csv_ibkr stays unchanged.
    ibkr_sections = {
        'Trades': ('trade', None),
        'Dividends': ('income', 'dividend'),
        'Withholding Tax': ('income', 'tax')}

    def load_csv_degiro(
            self,
            transaction_path: Optional[str] = None,
            statement_path: Optional[str] = None,
            isin_to_ticker: Optional[Dict[str, Optional[str]]] = None
            ) -> List[Transaction]:
        """
        Load Degiro transactions: trades + dividends + taxes.

        Args:
            transaction_path: Path to the Degiro trades-only CSV.
                Source of buys and sells. Each row is one fill;
                partial fills under a single Order ID are
                aggregated.
            statement_path: Path to the Degiro account statement.
                Source of dividends and withholding tax (those
                rows do not appear in the trades export).
            isin_to_ticker: Optional ISIN -> ticker override.
                When None, loads from
                ``data/isin_to_ticker.json``. ISINs mapped to
                None are silently skipped (cash placeholders).

        Returns:
            List of Transaction objects sorted by date.

        Raises:
            ValueError: When required columns are missing.
            FileNotFoundError: When either CSV is missing.
            RuntimeError: When a trade references an ISIN not in
                the ticker map.

        Example:
            >>> loader = CSVLoader()
            >>> trans = loader.load_csv_degiro()  # auto-discover
        """
        # Resolve the trades file; None means "use newest export"
        if transaction_path is None:
            # Auto-pick the lexicographically-newest export
            auto = default_degiro_path('transactions')
            if auto is None:
                raise FileNotFoundError(
                    "No Degiro transactions CSV found at "
                    f"{default_degiro_transactions_dir}.")
            transaction_path = str(auto)
        # Resolve the statement file the same way
        if statement_path is None:
            # Auto-pick the matching statement
            auto = default_degiro_path('statement')
            if auto is None:
                raise FileNotFoundError(
                    "No Degiro statement CSV found at "
                    f"{default_degiro_statements_dir}.")
            statement_path = str(auto)
        # Load the ISIN map when the caller did not pass one
        if isin_to_ticker is None:
            isin_to_ticker = load_isin_to_ticker()
        try:
            # Pass 1: buys / sells from the trades export
            transactions = self._parse_degiro_trades(
                transaction_path, isin_to_ticker)
            # Pass 2: dividends + withholding tax from statement
            transactions.extend(
                self._parse_degiro_income(
                    statement_path, isin_to_ticker))
            # Sort chronologically and run shared validation
            transactions.sort(key=lambda t: t.date)
            self._validate_transactions(transactions)
            return transactions
        except FileNotFoundError:
            raise FileNotFoundError(
                f"CSV file not found: {transaction_path}")
        except (ValueError, RuntimeError):
            # Surface typed errors unchanged so callers can match
            raise
        except Exception as e:
            raise Exception(
                f"Error loading transactions from "
                f"{transaction_path}: {str(e)}")

    def _parse_degiro_trades(
            self,
            transaction_path: str,
            isin_to_ticker: Dict[str, Optional[str]]
            ) -> List[Transaction]:
        """Parse buys and sells from the Degiro transactions CSV.

        One row = one fill. Partial fills that share an Order ID
        are aggregated into a single Transaction (sum shares,
        volume-weighted price, sum fees). Fees in the export
        are CHF-denominated; we convert each one back to the
        trade's native currency using the per-row Exchange rate
        so ``Transaction.fee`` and ``Transaction.auto_fx_fee``
        stay in ``Transaction.currency``, matching the rest of
        the codebase.
        """
        # Read the raw CSV and validate the header up-front
        df = pd.read_csv(transaction_path)
        self._validate_columns(
            df,
            required=self.degiro_transactions_required_columns)
        # Drop trailing blank rows and parse dates
        df = df.dropna(subset=['Date'])
        df['Date'] = pd.to_datetime(
            df['Date'], format='%d-%m-%Y')
        # If Order ID is empty, pull it from the trailing
        # column (absorbs Degiro's mid-export schema drift)
        df['Order ID'] = df['Order ID'].fillna(
            df[df.columns[-1]])
        # Drop tombstone rows (no quantity, no value, no id)
        tombstone = (df['Quantity'].isna()
                     & (df['Local value'].fillna(0) == 0)
                     & df['Order ID'].isna())
        df = df[~tombstone]
        # Remaining empty Order IDs are corporate-action rows;
        # assign synthetic IDs so each becomes its own group
        df['Order ID'] = df['Order ID'].astype(object)
        empty = df['Order ID'].isna()
        df.loc[empty, 'Order ID'] = (
            'synth_' + df.index[empty].astype(str))
        # Emit one Transaction per Order ID group
        out: List[Transaction] = []
        for oid, group in df.groupby('Order ID', sort=False):
            # Aggregate partial fills inside each Order ID
            trans = self._build_degiro_trade_group(
                group, isin_to_ticker)
            # Skip rows whose ISIN maps to a null ticker
            if trans is not None:
                out.append(trans)
        return out

    def _build_degiro_trade_group(
            self,
            group: pd.DataFrame,
            isin_to_ticker: Dict[str, Optional[str]]
            ) -> Optional[Transaction]:
        """Collapse one Order ID's fills into a single Transaction.

        The signed ``Quantity`` column carries the side: positive
        for buys, negative for sells. Every fill in the group
        must agree on side, ticker, and currency.
        """
        # The first fill carries metadata shared by the group
        anchor = group.iloc[0]
        # Resolve ticker via the ISIN map (no silent defaults)
        isin = str(anchor.get('ISIN') or '').strip()
        if not isin:
            raise RuntimeError(
                f"Degiro trade row missing ISIN: "
                f"product={anchor.get('Product')!r}")
        # Unknown ISIN is a hard error so the user fills the map
        if isin not in isin_to_ticker:
            raise RuntimeError(
                f"ISIN {isin!r} (product "
                f"{anchor['Product']!r}) is not in the ISIN -> "
                f"ticker map. Add it to "
                f"{default_isin_map_path}.")
        ticker = isin_to_ticker[isin]
        # Explicitly-null mapping signals "skip" (e.g. cash)
        if ticker is None:
            return None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # The trade currency lives in the unnamed column right
        # after Price. pandas auto-renames blank headers to
        # 'Unnamed: <n>' but the position is stable.
        cols = list(group.columns)
        price_pos = cols.index('Price')
        local_value_pos = cols.index('Local value')
        ccy_col = cols[price_pos + 1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Accumulators for the group aggregation pass
        currencies = set()
        signs = set()
        total_qty = 0.0
        gross_native = 0.0
        fee_chf = 0.0
        autofx_chf = 0.0
        exchange_rate = 1.0
        oid = anchor['Order ID']
        local_value_col = cols[local_value_pos]
        local_ccy_col = cols[local_value_pos + 1]
        # Iterate every fill row inside this Order ID
        for _, row in group.iterrows():
            raw_qty = row['Quantity']
            price = float(row['Price'])
            ccy = str(row[ccy_col]).strip().upper()
            # Degiro labels UK pence as 'GBX'; normalise to
            # GBP and divide the price by 100 so the trade is
            # stored in the actual ISO currency
            if ccy == 'GBX':
                ccy = 'GBP'
                price = price / 100.0
            # Degiro leaves Quantity blank on mutual-fund
            # subscriptions (the broker settles units at NAV
            # later). Recover from |Local value| / Price and
            # take the sign from Local value: negative means
            # money out (buy → positive qty), positive means
            # money in (sell → negative qty).
            try:
                qty_is_nan = pd.isna(raw_qty)
            except TypeError:
                qty_is_nan = False
            if qty_is_nan:
                lv = row[local_value_col]
                if pd.isna(lv) or price == 0.0 or pd.isna(price):
                    raise RuntimeError(
                        f"Cannot recover Quantity for Order ID "
                        f"{oid!r}: Local value or Price is "
                        f"missing.")
                lv_f = float(lv)
                qty = abs(lv_f) / price
                if lv_f > 0:
                    qty = -qty
            else:
                qty = float(raw_qty)
            if pd.isna(qty):
                raise RuntimeError(
                    f"Computed NaN Quantity for Order ID "
                    f"{oid!r}.")
            currencies.add(ccy)
            signs.add(1 if qty > 0 else -1)
            total_qty += qty
            gross_native += abs(qty) * price
            fee_chf += _fee_cell(
                row.get(
                    'Transaction and/or third party fees CHF'),
                'Transaction and/or third party fees CHF', oid)
            autofx_chf += _fee_cell(
                row.get('AutoFX Fee'), 'AutoFX Fee', oid)
            xr = _fee_cell(
                row.get('Exchange rate'), 'Exchange rate', oid)
            if xr > 0:
                # All fills in an order share one trade-time rate
                exchange_rate = xr
        # Fills under one Order ID must agree on side / currency
        if len(currencies) > 1 or len(signs) > 1:
            raise RuntimeError(
                f"Inconsistent fills for Order ID "
                f"{anchor['Order ID']!r}: "
                f"ccys={currencies}, signs={signs}.")
        # Sign of the summed Quantity drives buy vs sell
        currency = currencies.pop()
        operation = 'buy' if total_qty > 0 else 'sell'
        # Float amount so mutual-fund fractional units (e.g.
        # 3.833 of the GS High Yield fund) survive the round-trip.
        amount = abs(float(total_qty))
        price = gross_native / abs(total_qty) if total_qty else 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Convert CHF fees back to native currency so the
        # Transaction's fee fields are in `currency`.
        if currency == 'CHF' or exchange_rate <= 0:
            fee_native = abs(fee_chf)
            autofx_native = abs(autofx_chf)
        else:
            fee_native = abs(fee_chf) / exchange_rate
            autofx_native = abs(autofx_chf) / exchange_rate
        return Transaction(
            ticker=ticker,
            operation=operation,
            date=anchor['Date'].to_pydatetime(),
            price=price,
            amount=amount,
            fee=fee_native,
            auto_fx_fee=autofx_native,
            currency=currency)

    def _parse_degiro_income(
            self,
            statement_path: str,
            isin_to_ticker: Dict[str, Optional[str]]
            ) -> List[Transaction]:
        """Pull dividend + withholding-tax rows from the statement."""
        # Read the statement and validate its 12-column schema
        df = pd.read_csv(statement_path)
        # The statement is one of two valid Degiro inputs; if a
        # caller hands us the wrong schema, fail with the
        # statement-specific column list.
        self._validate_columns(
            df,
            required=self.degiro_statement_required_columns)
        # Drop trailing blank rows and parse dates
        df = df.dropna(subset=['Date'])
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        # The statement's two unnamed columns carry the change
        # amount and balance amount. Rename for clarity.
        rename = {}
        for i, col in enumerate(df.columns):
            if col.startswith('Unnamed:'):
                if i == 8:
                    rename[col] = 'Change Amount'
                elif i == 10:
                    rename[col] = 'Balance Amount'
        if rename:
            df = df.rename(columns=rename)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Emit one Transaction per income / corporate-action row
        out: List[Transaction] = []
        for _, row in df.iterrows():
            desc = str(row.get('Description') or '').strip()
            if desc == 'Dividend':
                trans = self._build_degiro_income(
                    row, 'dividend', isin_to_ticker)
            elif desc == 'Dividend Tax':
                trans = self._build_degiro_income(
                    row, 'tax', isin_to_ticker)
            elif desc.startswith('DELISTING:'):
                # Force-closeout: parse the embedded quantity
                trans = self._build_degiro_delisting(
                    row, isin_to_ticker)
            else:
                continue
            if trans is not None:
                out.append(trans)
        return out

    

    def _build_degiro_delisting(
            self,
            row: pd.Series,
            isin_to_ticker: Dict[str, Optional[str]]) -> Optional[Transaction]:
        """Synthesize a buy/sell from a DELISTING statement row.

        The transactions CSV does not carry these events; only
        the statement records them. The recorded price may be
        zero (cash-settled delisting); we keep that as-is.
        """
        # Description shape: "DELISTING: Sell N <name>@<price> <CCY>"
        degiro_delisting_re = re.compile(
            r'^DELISTING:\s+(?P<op>Buy|Sell)\s+'
            r'(?P<qty>\d+(?:\.\d+)?)\s+.*?'
            r'@(?P<price>[\d.]+)\s+(?P<ccy>[A-Z]{3})')
    
        isin = str(row.get('ISIN') or '').strip()
        if not isin or isin not in isin_to_ticker:
            return None
        
        ticker = isin_to_ticker[isin]
        if ticker is None:
            return None
        
        match = degiro_delisting_re.match(str(row['Description']))
        
        if match is None:
            return None
        
        return Transaction(
            ticker=ticker,
            operation=match.group('op').lower(),
            date=row['Date'].to_pydatetime(),
            price=float(match.group('price')),
            amount=float(match.group('qty')),
            fee=0.0,
            auto_fx_fee=0.0,
            currency=match.group('ccy').upper())

    def _build_degiro_income(
            self,
            row: pd.Series,
            operation: str,
            isin_to_ticker: Dict[str, Optional[str]]) -> Optional[Transaction]:
        """Synthesize a dividend/tax Transaction from one row.

        Args:
            row: Single statement row whose description is
                ``Dividend`` or ``Dividend Tax``.
            operation: 'dividend' or 'tax'.
            isin_to_ticker: Same map as the trade pass.
        """
        isin = str(row.get('ISIN') or '').strip()
        if not isin or isin not in isin_to_ticker:
            # Cash-account interest etc. lacks an ISIN; skip
            return None
        
        ticker = isin_to_ticker[isin]
        if ticker is None:
            return None
        
        change_amt = row.get('Change Amount')
        if pd.isna(change_amt):
            return None
        
        amount_value = float(change_amt)
        currency = str(row.get('Change') or '').strip().upper() or 'USD'

        # Tax magnitudes ship as negative; preserve the sign so
        # `total_cost` semantics match the IBKR loader (price
        # negative for tax, positive for dividend).
        if operation == 'dividend':
            price = abs(amount_value)
        else:
            price = -abs(amount_value)
        # Return
        return Transaction(
            ticker=ticker,
            operation=operation,
            date=row['Date'].to_pydatetime(),
            price=price,
            amount=1,
            fee=0.0,
            auto_fx_fee=0.0,
            currency=currency)

    def _validate_columns(
            self,
            df: pd.DataFrame,
            required: Optional[List[str]] = None) -> None:
        """
        Validate that DataFrame has required columns.

        Args:
            df: DataFrame to validate
            required: Optional explicit column list. Defaults to
                the statement schema for back-compat.

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = (
            required if required is not None
            else self.degiro_required_columns)
        
        missing = []
        for col in required_cols:
            if col not in df.columns:
                missing.append(col)

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Required: {required_cols}")

    def _validate_transactions(self, transactions: List[Transaction]) -> None:
        """
        Validate transaction data.

        Empty lists are accepted: a statement containing only
        bookkeeping rows (cash sweeps, deposits, connection
        fees, etc.) legitimately produces zero canonical
        transactions.

        Args:
            transactions: List of transactions to validate

        Raises:
            ValueError: When any transaction has an unknown
                operation tag.
        """
        # Check for valid operations
        valid_ops = {'buy', 'sell', 'dividend', 'tax'}
        for trans in transactions:
            if trans.operation not in valid_ops:
                raise ValueError(
                    f"Invalid operation '{trans.operation}' "
                    f"for {trans.ticker}. "
                    f"Must be one of: {valid_ops}")

    def load_csv_ibkr(self, file_path: str) -> List[Transaction]:
        """
        Load transaction data from an IBKR Activity Statement.

        IBKR statements are multi-section CSVs: each row begins
        with `<Section>,<Header|Data|SubTotal|Total|Notes>,...`
        and rows from different sections have different column
        counts. This loader walks the file row by row and
        extracts Transactions from three sections:

        - Trades         -> 'buy' / 'sell'
                            (filtered to Stocks + Mutual Funds;
                            Forex rows are skipped)
        - Dividends      -> 'dividend'
        - Withholding Tax -> 'tax'

        Deposits & Withdrawals rows have no ticker and are not
        emitted. The Codes section is ignored at runtime; row
        filtering relies on Asset Category and
        DataDiscriminator instead.

        IBKR symbols for SIX Swiss Exchange listings are
        normalised via the `ibkr_symbol_map` table imported
        from `src/ioput/ibkr_symbol_map.py`.

        Args:
            file_path: Path to IBKR Activity Statement CSV

        Returns:
            List of Transaction objects sorted by date

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If no transactions are loaded
            Exception: For other loading errors

        Example:
            >>> loader = CSVLoader()
            >>> trans = loader.load_csv_ibkr(
            ...     'reports/ibkr/<YYYYMMDD>_<YYYYMMDD>_<ACCOUNT_ID>.csv'
            ... )
            >>> print(f"Loaded {len(trans)} transactions")
        """
        try:
            transactions = []
            headers = {}
            # Walk the file row by row using the stdlib csv
            # reader; pandas cannot parse a heterogeneous
            # multi-section schema cleanly.
            with open(file_path, 'r', newline='', encoding='utf-8') as fh:
                reader = csv.reader(fh)
                for row in reader:
                    # Skip blank or one-cell rows
                    if len(row) < 2:
                        continue
                    section = row[0]
                    marker = row[1]
                    # Only sections we know how to parse
                    if section not in self.ibkr_sections:
                        continue
                    # Cache header so later Data rows in
                    # this section can be zipped to dicts.
                    if marker == 'Header':
                        headers[section] = row
                        continue
                    # Skip SubTotal, Total, Notes, blanks, and
                    # any non-data row.
                    if marker != 'Data':
                        continue
                    # Must have seen the section's header first
                    if section not in headers:
                        continue
                    # Dispatch by section kind
                    kind, operation = self.ibkr_sections[section]
                    if kind == 'trade':
                        # Buys + sells in the Trades section
                        trans = self._parse_ibkr_trade_row(
                            row, headers[section])
                    else:
                        # Dividends + withholding-tax sections
                        trans = self._parse_ibkr_income_row(
                            row, headers[section], operation)
                    # Skip rows the parser refused (e.g. Forex)
                    if trans is not None:
                        transactions.append(trans)
                        
            # Validate and sort.
            self._validate_transactions(transactions)
            transactions.sort(key=lambda t: t.date)
            # Return
            return transactions
        # Error handling
        except FileNotFoundError:
            raise FileNotFoundError(
                f"CSV file not found: {file_path}")
        except Exception as e:
            raise Exception(
                f"Error loading IBKR transactions from {file_path}: {str(e)}")

    def _parse_ibkr_trade_row(self, row, header):
        """
        Map one IBKR Trades Data row to a Transaction.

        Returns None when the row is a SubTotal/Total in
        disguise, a Forex conversion, or any asset category
        outside {Stocks, Mutual Funds}.

        Parameters
        ----------
        row : list[str]
            Raw CSV row.
        header : list[str]
            Active 'Trades,Header,...' row used to name fields.

        Returns
        -------
        Transaction or None
        """
        fields = dict(zip(header, row))
        if fields.get('DataDiscriminator') != 'Order':
            return None
        asset_category = fields.get('Asset Category')
        if asset_category not in ('Stocks', 'Mutual Funds'):
            return None
        
        # Date/Time is "YYYY-MM-DD, HH:MM:SS" (the comma is
        # inside the quoted field, so csv.reader gives us a
        # single string with a literal comma).
        raw_dt = fields['Date/Time']
        date_part = raw_dt.split(',')[0].strip()
        date = datetime.strptime(date_part, '%Y-%m-%d')
        
        # Quantity may include thousands separators.
        # Stocks are whole numbers; 
        # mutual fund units can be fractional (e.g. 20.49 BlackRock units). 
        # Preserve fractional precision so total_cost stays accurate.
        quantity_str = fields['Quantity'].replace(',', '')
        quantity_float = float(quantity_str)
        if quantity_float == 0:
            return None
        if quantity_float.is_integer():
            quantity = int(quantity_float)
        else:
            quantity = quantity_float
        # Select operation based on quantity sign:
        # positive = buy, negative = sell.
        if quantity > 0:
            operation = 'buy'
        else:
            operation = 'sell'
        amount = abs(quantity)
        
        # Price and fee.
        price = float(fields['T. Price'])
        fee_raw = fields['Comm/Fee']
        fee = abs(float(fee_raw))
        
        # Symbol normalisation.
        ticker = self._normalize_ibkr_symbol(
            fields['Symbol'].strip())
        # Trade currency from the IBKR row (USD / CHF / EUR ...)
        currency = fields['Currency'].strip().upper()
        # Return
        return Transaction(
            ticker=ticker,
            operation=operation,
            date=date,
            price=price,
            amount=amount,
            fee=fee,
            auto_fx_fee=0.0,
            currency=currency)

    def _parse_ibkr_income_row(self, row, header, operation):
        """
        Map one IBKR Dividends or Withholding Tax Data row to a
        Transaction.

        Returns None when the ticker cannot be extracted (e.g.
        bond interest with a description that does not match
        the SYMBOL(ISIN) pattern).

        Parameters
        ----------
        row : list[str]
            Raw CSV row.
        header : list[str]
            Active section Header row used to name fields.
        operation : {'dividend', 'tax'}
            Target Transaction.operation value.

        Returns
        -------
        Transaction or None
        """
        fields = dict(zip(header, row))
        description = fields['Description']
        # Skip when the description does not start with a
        # SYMBOL(ID) pattern. This drops two kinds of rows:
        # bond/interest payouts that don't fit the shape, and
        # IBKR per-currency Total rows that share the 'Data'
        # marker but have an empty Description.
        match = re.match(r'^([A-Za-z][A-Za-z0-9]*)\(', description)
        if match is None:
            return None
        ticker = self._normalize_ibkr_symbol(match.group(1))
        # Get date
        date = datetime.strptime(fields['Date'], '%Y-%m-%d')
        # `price` carries the cash amount with its sign:
        # positive for dividends, negative for tax rows.
        price = float(fields['Amount'])
        # Trade currency from the IBKR row (USD / CHF / EUR ...)
        currency = fields['Currency'].strip().upper()
        # Return
        return Transaction(
            ticker=ticker,
            operation=operation,
            date=date,
            price=price,
            amount=1,
            fee=0.0,
            auto_fx_fee=0.0,
            currency=currency)

    @staticmethod
    def _normalize_ibkr_symbol(symbol):
        """
        Translate IBKR's SIX symbols to the `.SW` form used by
        the rest of the project; pass through unchanged for
        every other exchange. Mapping table lives in
        `src/ioput/ibkr_symbol_map.py`.

        Parameters
        ----------
        symbol : str
            Raw symbol from the IBKR file.

        Returns
        -------
        str
            Canonical symbol.
        """
        return ibkr_symbol_map.get(symbol, symbol)
