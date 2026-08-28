"""Option pricing using the Cox-Ross-Rubinstein binomial tree."""

from typing import Literal

import numpy as np

from src.black_scholes import OptionParameters


OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


def validate_binomial_inputs(
    steps: int,
    option_type: str,
    exercise_style: str,
) -> None:
    """Validate the binomial model inputs."""

    if not isinstance(steps, int):
        raise TypeError("Steps must be an integer.")

    if steps <= 0:
        raise ValueError("Steps must be greater than zero.")

    if option_type not in {"call", "put"}:
        raise ValueError("Option type must be 'call' or 'put'.")

    if exercise_style not in {"european", "american"}:
        raise ValueError(
            "Exercise style must be 'european' or 'american'."
        )


def calculate_intrinsic_value(
    asset_prices: np.ndarray,
    strike_price: float,
    option_type: OptionType,
) -> np.ndarray:
    """Calculate call or put intrinsic values."""

    if option_type == "call":
        return np.maximum(
            asset_prices - strike_price,
            0.0,
        )

    return np.maximum(
        strike_price - asset_prices,
        0.0,
    )


def binomial_option_price(
    parameters: OptionParameters,
    option_type: OptionType,
    steps: int = 500,
    exercise_style: ExerciseStyle = "european",
) -> float:
    """
    Calculate an option price with the CRR binomial model.

    The implementation supports European and American call and put
    options. Memory usage is reduced by storing only one tree level
    at a time.
    """

    parameters.validate()

    normalized_option_type = option_type.lower()
    normalized_exercise_style = exercise_style.lower()

    validate_binomial_inputs(
        steps=steps,
        option_type=normalized_option_type,
        exercise_style=normalized_exercise_style,
    )

    time_step = parameters.time_to_maturity / steps

    up_factor = np.exp(
        parameters.volatility
        * np.sqrt(time_step)
    )

    down_factor = 1.0 / up_factor

    risk_neutral_growth = np.exp(
        (
            parameters.risk_free_rate
            - parameters.dividend_yield
        )
        * time_step
    )

    risk_neutral_probability = (
        risk_neutral_growth - down_factor
    ) / (up_factor - down_factor)

    if not 0.0 <= risk_neutral_probability <= 1.0:
        raise ValueError(
            "Risk-neutral probability must be between zero and one. "
            "Increase the number of steps or review the parameters."
        )

    discount_factor = np.exp(
        -parameters.risk_free_rate
        * time_step
    )

    terminal_up_moves = np.arange(steps + 1)
    terminal_down_moves = steps - terminal_up_moves

    terminal_asset_prices = (
        parameters.spot_price
        * up_factor**terminal_up_moves
        * down_factor**terminal_down_moves
    )

    option_values = calculate_intrinsic_value(
        asset_prices=terminal_asset_prices,
        strike_price=parameters.strike_price,
        option_type=normalized_option_type,
    )

    for current_step in range(steps - 1, -1, -1):
        option_values = discount_factor * (
            risk_neutral_probability
            * option_values[1:]
            + (
                1.0 - risk_neutral_probability
            )
            * option_values[:-1]
        )

        if normalized_exercise_style == "american":
            up_moves = np.arange(
                current_step + 1
            )

            down_moves = (
                current_step - up_moves
            )

            asset_prices = (
                parameters.spot_price
                * up_factor**up_moves
                * down_factor**down_moves
            )

            intrinsic_values = calculate_intrinsic_value(
                asset_prices=asset_prices,
                strike_price=parameters.strike_price,
                option_type=normalized_option_type,
            )

            option_values = np.maximum(
                option_values,
                intrinsic_values,
            )

    return float(option_values[0])


def binomial_call_price(
    parameters: OptionParameters,
    steps: int = 500,
    exercise_style: ExerciseStyle = "european",
) -> float:
    """Calculate a call option price using the binomial model."""

    return binomial_option_price(
        parameters=parameters,
        option_type="call",
        steps=steps,
        exercise_style=exercise_style,
    )


def binomial_put_price(
    parameters: OptionParameters,
    steps: int = 500,
    exercise_style: ExerciseStyle = "european",
) -> float:
    """Calculate a put option price using the binomial model."""

    return binomial_option_price(
        parameters=parameters,
        option_type="put",
        steps=steps,
        exercise_style=exercise_style,
    )


if __name__ == "__main__":
    baseline_parameters = OptionParameters(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )

    number_of_steps = 500

    european_call = binomial_call_price(
        parameters=baseline_parameters,
        steps=number_of_steps,
        exercise_style="european",
    )

    european_put = binomial_put_price(
        parameters=baseline_parameters,
        steps=number_of_steps,
        exercise_style="european",
    )

    american_call = binomial_call_price(
        parameters=baseline_parameters,
        steps=number_of_steps,
        exercise_style="american",
    )

    american_put = binomial_put_price(
        parameters=baseline_parameters,
        steps=number_of_steps,
        exercise_style="american",
    )

    print("\nCox-Ross-Rubinstein binomial pricing")
    print(f"Number of steps: {number_of_steps:,}")

    print("\nEuropean options")
    print(f"European call: ${european_call:,.4f}")
    print(f"European put: ${european_put:,.4f}")

    print("\nAmerican options")
    print(f"American call: ${american_call:,.4f}")
    print(f"American put: ${american_put:,.4f}")