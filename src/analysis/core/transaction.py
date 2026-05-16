"""
Transaction data model for portfolio management.

This module defines the Transaction dataclass used throughout
the package to represent buy, sell, dividend, and tax
(dividend withholding) transactions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class Transaction:
    """
    Represents a single portfolio transaction.

    Attributes:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        operation: Transaction type ('buy', 'sell', 'dividend',
            'tax'). 'tax' represents dividend withholding tax
            (cash outflow tied to a prior dividend payment).
        date: Transaction date
        price: Price per share, in `currency` (can be negative
            for 'dividend' corrections and 'tax' rows).
        amount: Number of shares (positive for buy/sell;
            set to 1 for dividend/tax cash-flow rows).
            Fractional values are allowed (mutual funds settle
            at fractional NAV units).
        fee: Transaction fee, in `currency`.
        auto_fx_fee: Automatic foreign-exchange fee, in
            `currency`.
        currency: ISO-4217 code of the trade currency (e.g.,
            'USD', 'CHF', 'EUR'). Defaults to 'USD' for
            backward compatibility with older fixtures.

    Example:
        >>> trans = Transaction(
        ...     ticker='AAPL',
        ...     operation='buy',
        ...     date=datetime(2023, 1, 15),
        ...     price=150.00,
        ...     amount=100,
        ...     fee=9.95,
        ...     auto_fx_fee=0.00,
        ...     currency='USD'
        ... )
    """

    ticker: str
    operation: Literal['buy', 'sell', 'dividend', 'tax']
    date: datetime
    price: float
    amount: float
    fee: float
    auto_fx_fee: float
    currency: str = 'USD'

    def __post_init__(self):
        """Validate transaction data after initialization."""
        if self.amount < 0:
            raise ValueError(
                f"Amount must be positive, got {self.amount}"
            )
        # Allow negative prices for dividend corrections and
        # withholding tax rows
        if self.price < 0 and self.operation not in ('dividend', 'tax'):
            raise ValueError(
                f"Price must be positive for {self.operation}, "
                f"got {self.price}")
        if self.fee < 0:
            raise ValueError(
                f"Fee must be non-negative, got {self.fee}")
        if self.auto_fx_fee < 0:
            raise ValueError(
                f"FX fee must be non-negative, got {self.auto_fx_fee}")

    @property
    def total_cost(self) -> float:
        """
        Calculate total transaction cost including fees.

        For buy transactions, returns negative (cash outflow).
        For sell transactions, returns positive (cash inflow).
        For dividend transactions, returns positive (income).
        For tax transactions, returns negative (withholding
        outflow); the negative sign is carried by `price`.

        Returns:
            Total cost/proceeds including fees
        """
        if self.operation == 'buy':
            return -(self.amount * self.price +
                     self.fee + self.auto_fx_fee)
        elif self.operation == 'sell':
            return (self.amount * self.price -
                    self.fee - self.auto_fx_fee)
        elif self.operation == 'dividend':
            return self.amount * self.price
        elif self.operation == 'tax':
            return self.amount * self.price
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

    @property
    def total_fee(self) -> float:
        """Calculate total fees (regular + FX)."""
        return self.fee + self.auto_fx_fee

    def __str__(self) -> str:
        """String representation of transaction."""
        return (
            f"{self.operation.upper()} {self.amount} "
            f"{self.ticker} @ ${self.price:.2f} "
            f"on {self.date.strftime('%Y-%m-%d')}")

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"Transaction(ticker='{self.ticker}', "
            f"operation='{self.operation}', "
            f"date={self.date}, price={self.price}, "
            f"amount={self.amount}, fee={self.fee}, "
            f"auto_fx_fee={self.auto_fx_fee}, "
            f"currency='{self.currency}')")
