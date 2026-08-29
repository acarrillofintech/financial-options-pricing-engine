"""European option pricing using the Black-Scholes model."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionParameters:
    """Parameters required to value a European option."""

    spot_price: float
    strike_price: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0

    def validate(self) -> None:
        """Validate the option parameters."""

        if self.spot_price <= 0:
            raise ValueError("Spot price must be greater than zero.")

        if self.strike_price <= 0:
            raise ValueError("Strike price must be greater than zero.")

        if self.time_to_maturity <= 0:
            raise ValueError(
                "Time to maturity must be greater than zero."
            )

        if self.volatility <= 0:
            raise ValueError("Volatility must be greater than zero.")


def calculate_d1_d2(
    parameters: OptionParameters,
) -> tuple[float, float]:
    """Calculate the d1 and d2 terms of the Black-Scholes model."""

    parameters.validate()

    numerator = (
        np.log(parameters.spot_price / parameters.strike_price)
        + (
            parameters.risk_free_rate
            - parameters.dividend_yield
            + 0.5 * parameters.volatility**2
        )
        * parameters.time_to_maturity
    )

    denominator = (
        parameters.volatility
        * np.sqrt(parameters.time_to_maturity)
    )

    d1 = numerator / denominator

    d2 = (
        d1
        - parameters.volatility
        * np.sqrt(parameters.time_to_maturity)
    )

    return float(d1), float(d2)


def black_scholes_call(parameters: OptionParameters) -> float:
    """Calculate the price of a European call option."""

    d1, d2 = calculate_d1_d2(parameters)

    discounted_spot = (
        parameters.spot_price
        * np.exp(
            -parameters.dividend_yield
            * parameters.time_to_maturity
        )
    )

    discounted_strike = (
        parameters.strike_price
        * np.exp(
            -parameters.risk_free_rate
            * parameters.time_to_maturity
        )
    )

    call_price = (
        discounted_spot * norm.cdf(d1)
        - discounted_strike * norm.cdf(d2)
    )

    return float(call_price)


def black_scholes_put(parameters: OptionParameters) -> float:
    """Calculate the price of a European put option."""

    d1, d2 = calculate_d1_d2(parameters)

    discounted_spot = (
        parameters.spot_price
        * np.exp(
            -parameters.dividend_yield
            * parameters.time_to_maturity
        )
    )

    discounted_strike = (
        parameters.strike_price
        * np.exp(
            -parameters.risk_free_rate
            * parameters.time_to_maturity
        )
    )

    put_price = (
        discounted_strike * norm.cdf(-d2)
        - discounted_spot * norm.cdf(-d1)
    )

    return float(put_price)


def black_scholes_price(
    parameters: OptionParameters,
    option_type: OptionType,
) -> float:
    """Calculate a European call or put price."""

    normalized_option_type = option_type.lower()

    if normalized_option_type == "call":
        return black_scholes_call(parameters)

    if normalized_option_type == "put":
        return black_scholes_put(parameters)

    raise ValueError("Option type must be 'call' or 'put'.")


def calculate_put_call_parity_difference(
    parameters: OptionParameters,
) -> float:
    """
    Calculate the difference between both sides of put-call parity.

    For European options with continuous dividends:

        C - P = S * exp(-qT) - K * exp(-rT)

    A result close to zero indicates that parity is satisfied.
    """

    call_price = black_scholes_call(parameters)
    put_price = black_scholes_put(parameters)

    left_side = call_price - put_price

    right_side = (
        parameters.spot_price
        * np.exp(
            -parameters.dividend_yield
            * parameters.time_to_maturity
        )
        - parameters.strike_price
        * np.exp(
            -parameters.risk_free_rate
            * parameters.time_to_maturity
        )
    )

    return float(left_side - right_side)


if __name__ == "__main__":
    baseline_parameters = OptionParameters(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )

    call_value = black_scholes_call(baseline_parameters)
    put_value = black_scholes_put(baseline_parameters)

    parity_difference = calculate_put_call_parity_difference(
        baseline_parameters
    )

    print("\nBlack-Scholes European option pricing")
    print(f"Spot price: ${baseline_parameters.spot_price:,.2f}")
    print(f"Strike price: ${baseline_parameters.strike_price:,.2f}")
    print(
        f"Time to maturity: "
        f"{baseline_parameters.time_to_maturity:.2f} years"
    )
    print(
        f"Risk-free rate: "
        f"{baseline_parameters.risk_free_rate:.2%}"
    )
    print(f"Volatility: {baseline_parameters.volatility:.2%}")
    print(
        f"Dividend yield: "
        f"{baseline_parameters.dividend_yield:.2%}"
    )

    print("\nOption values")
    print(f"European call: ${call_value:,.4f}")
    print(f"European put: ${put_value:,.4f}")

    print("\nPut-call parity")
    print(f"Parity difference: {parity_difference:.12f}")