#!/usr/bin/env python3
"""
Options Greeks Analysis Example.

Demonstrates calculation of option Greeks using
Black-Scholes model.
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.modelling.greeks import (
    GreeksCalculator,
    calculate_delta,
    calculate_gamma,
    calculate_theta,
    calculate_vega,
    calculate_rho)


def main():
    """Demonstrate Greeks calculations."""
    print("=" * 60)
    print("OPTIONS GREEKS ANALYSIS")
    print("=" * 60)

    # Option parameters
    spot_price = 100.0
    strike_price = 105.0
    time_to_expiry = 0.25  # 3 months
    volatility = 0.20  # 20%
    risk_free_rate = 0.05  # 5%
    dividend_yield = 0.02  # 2%

    print("\nOption Parameters:")
    print(f"  Spot Price:        ${spot_price:.2f}")
    print(f"  Strike Price:      ${strike_price:.2f}")
    print(f"  Time to Expiry:    {time_to_expiry*12:.1f} months")
    print(f"  Volatility:        {volatility*100:.1f}%")
    print(f"  Risk-Free Rate:    {risk_free_rate*100:.1f}%")
    print(f"  Dividend Yield:    {dividend_yield*100:.1f}%")

    # Calculate using GreeksCalculator
    print("\n" + "=" * 60)
    print("USING GREEKS CALCULATOR")
    print("=" * 60)

    calc = GreeksCalculator()

    # Call option Greeks
    print("\nCall Option Greeks:")
    call_greeks = calc.calculate_all_greeks(
        spot_price=spot_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        option_type='call',
        dividend_yield=dividend_yield)
    print(f"  Delta:  {call_greeks.delta:>8.4f}")
    print(f"  Gamma:  {call_greeks.gamma:>8.6f}")
    print(f"  Theta:  {call_greeks.theta:>8.4f} (per day)")
    print(f"  Vega:   {call_greeks.vega:>8.4f} (per 1%)")
    print(f"  Rho:    {call_greeks.rho:>8.4f} (per 1%)")

    # Put option Greeks
    print("\nPut Option Greeks:")
    put_greeks = calc.calculate_all_greeks(
        spot_price=spot_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        option_type='put',
        dividend_yield=dividend_yield)
    print(f"  Delta:  {put_greeks.delta:>8.4f}")
    print(f"  Gamma:  {put_greeks.gamma:>8.6f}")
    print(f"  Theta:  {put_greeks.theta:>8.4f} (per day)")
    print(f"  Vega:   {put_greeks.vega:>8.4f} (per 1%)")
    print(f"  Rho:    {put_greeks.rho:>8.4f} (per 1%)")

    # Using individual functions
    print("\n" + "=" * 60)
    print("USING INDIVIDUAL FUNCTIONS")
    print("=" * 60)

    delta = calculate_delta(
        spot_price, strike_price, time_to_expiry,
        volatility, risk_free_rate, 'call',
        dividend_yield)
    
    gamma = calculate_gamma(
        spot_price, strike_price, time_to_expiry,
        volatility, risk_free_rate, dividend_yield)
    
    theta = calculate_theta(
        spot_price, strike_price, time_to_expiry,
        volatility, risk_free_rate, 'call',
        dividend_yield)
    
    vega = calculate_vega(
        spot_price, strike_price, time_to_expiry,
        volatility, risk_free_rate, dividend_yield)
    
    rho = calculate_rho(
        spot_price, strike_price, time_to_expiry,
        volatility, risk_free_rate, 'call',
        dividend_yield)

    print("\nCall Option Greeks (Individual Functions):")
    print(f"  Delta:  {delta:.4f}")
    print(f"  Gamma:  {gamma:.6f}")
    print(f"  Theta:  {theta:.4f} per day")
    print(f"  Vega:   {vega:.4f} per 1% vol")
    print(f"  Rho:    {rho:.4f} per 1% rate")

    # Practical interpretation
    print("\n" + "=" * 60)
    print("PRACTICAL INTERPRETATION")
    print("=" * 60)

    print("\nCall Option Sensitivities:")
    print(f"  If stock rises $1: "
        f"option gains ${delta:.2f}")
    
    print(
        f"  If volatility rises 1%: "
        f"option gains ${vega:.2f}")
    
    print(
        f"  Time decay per day: "
        f"option loses ${abs(theta):.2f}")
    
    print(
        f"  If rates rise 1%: "
        f"option gains ${rho:.2f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
