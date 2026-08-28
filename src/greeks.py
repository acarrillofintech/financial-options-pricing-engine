"""Option Greeks calculated using the Black-Scholes model."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

from src.black_scholes import (
    OptionParameters,
    calculate_d1_d2,
)


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionGreeks:
    """Sensitivity measures for a European option."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def calculate_gamma(parameters: OptionParameters) -> float:
    """Calculate gamma for a European call or put."""

    d1, _ = calculate_d1_d2(parameters)

    gamma = (
        np.exp(
            -parameters.dividend_yield
            * parameters.time_to_maturity
        )
        * norm.pdf(d1)
        / (
            parameters.spot_price
            * parameters.volatility
            * np.sqrt(parameters.time_to_maturity)
        )
    )

    return float(gamma)


def calculate_vega(parameters: OptionParameters) -> float:
    """
    Calculate vega for a one-percentage-point volatility change.

    The result represents the option price change produced by
    a 1% change in volatility.
    """

    d1, _ = calculate_d1_d2(parameters)

    vega = (
        parameters.spot_price
        * np.exp(
            -parameters.dividend_yield
            * parameters.time_to_maturity
        )
        * norm.pdf(d1)
        * np.sqrt(parameters.time_to_maturity)
        / 100.0
    )

    return float(vega)


def calculate_call_greeks(
    parameters: OptionParameters,
) -> OptionGreeks:
    """Calculate Black-Scholes Greeks for a European call."""

    d1, d2 = calculate_d1_d2(parameters)

    discount_dividend = np.exp(
        -parameters.dividend_yield
        * parameters.time_to_maturity
    )

    discount_rate = np.exp(
        -parameters.risk_free_rate
        * parameters.time_to_maturity
    )

    delta = discount_dividend * norm.cdf(d1)

    gamma = calculate_gamma(parameters)
    vega = calculate_vega(parameters)

    annual_theta = (
        -(
            parameters.spot_price
            * discount_dividend
            * norm.pdf(d1)
            * parameters.volatility
        )
        / (
            2.0
            * np.sqrt(parameters.time_to_maturity)
        )
        - (
            parameters.risk_free_rate
            * parameters.strike_price
            * discount_rate
            * norm.cdf(d2)
        )
        + (
            parameters.dividend_yield
            * parameters.spot_price
            * discount_dividend
            * norm.cdf(d1)
        )
    )

    theta = annual_theta / 365.0

    rho = (
        parameters.strike_price
        * parameters.time_to_maturity
        * discount_rate
        * norm.cdf(d2)
        / 100.0
    )

    return OptionGreeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def calculate_put_greeks(
    parameters: OptionParameters,
) -> OptionGreeks:
    """Calculate Black-Scholes Greeks for a European put."""

    d1, d2 = calculate_d1_d2(parameters)

    discount_dividend = np.exp(
        -parameters.dividend_yield
        * parameters.time_to_maturity
    )

    discount_rate = np.exp(
        -parameters.risk_free_rate
        * parameters.time_to_maturity
    )

    delta = discount_dividend * (norm.cdf(d1) - 1.0)

    gamma = calculate_gamma(parameters)
    vega = calculate_vega(parameters)

    annual_theta = (
        -(
            parameters.spot_price
            * discount_dividend
            * norm.pdf(d1)
            * parameters.volatility
        )
        / (
            2.0
            * np.sqrt(parameters.time_to_maturity)
        )
        + (
            parameters.risk_free_rate
            * parameters.strike_price
            * discount_rate
            * norm.cdf(-d2)
        )
        - (
            parameters.dividend_yield
            * parameters.spot_price
            * discount_dividend
            * norm.cdf(-d1)
        )
    )

    theta = annual_theta / 365.0

    rho = (
        -parameters.strike_price
        * parameters.time_to_maturity
        * discount_rate
        * norm.cdf(-d2)
        / 100.0
    )

    return OptionGreeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def calculate_option_greeks(
    parameters: OptionParameters,
    option_type: OptionType,
) -> OptionGreeks:
    """Calculate Greeks for a European call or put."""

    normalized_option_type = option_type.lower()

    if normalized_option_type == "call":
        return calculate_call_greeks(parameters)

    if normalized_option_type == "put":
        return calculate_put_greeks(parameters)

    raise ValueError("Option type must be 'call' or 'put'.")


if __name__ == "__main__":
    baseline_parameters = OptionParameters(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )

    call_greeks = calculate_call_greeks(
        baseline_parameters
    )

    put_greeks = calculate_put_greeks(
        baseline_parameters
    )

    print("\nBlack-Scholes option Greeks")

    print("\nEuropean call")
    print(f"Delta: {call_greeks.delta:.6f}")
    print(f"Gamma: {call_greeks.gamma:.6f}")
    print(f"Vega (per 1%): {call_greeks.vega:.6f}")
    print(f"Theta (per day): {call_greeks.theta:.6f}")
    print(f"Rho (per 1%): {call_greeks.rho:.6f}")

    print("\nEuropean put")
    print(f"Delta: {put_greeks.delta:.6f}")
    print(f"Gamma: {put_greeks.gamma:.6f}")
    print(f"Vega (per 1%): {put_greeks.vega:.6f}")
    print(f"Theta (per day): {put_greeks.theta:.6f}")
    print(f"Rho (per 1%): {put_greeks.rho:.6f}")