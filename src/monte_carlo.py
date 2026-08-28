"""European option pricing using risk-neutral Monte Carlo simulation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

from src.black_scholes import OptionParameters


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of a Monte Carlo option pricing simulation."""

    price: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float
    simulations: int


def validate_monte_carlo_inputs(
    simulations: int,
    option_type: str,
) -> None:
    """Validate Monte Carlo simulation inputs."""

    if not isinstance(simulations, int):
        raise TypeError("Simulations must be an integer.")

    if simulations <= 1:
        raise ValueError(
            "Simulations must be greater than one."
        )

    if option_type not in {"call", "put"}:
        raise ValueError(
            "Option type must be 'call' or 'put'."
        )


def generate_standard_normal_shocks(
    simulations: int,
    random_generator: np.random.Generator,
    antithetic: bool,
) -> np.ndarray:
    """Generate standard normal shocks."""

    if not antithetic:
        return random_generator.standard_normal(
            simulations
        )

    half_size = (simulations + 1) // 2

    first_half = random_generator.standard_normal(
        half_size
    )

    antithetic_shocks = np.concatenate(
        (first_half, -first_half)
    )

    return antithetic_shocks[:simulations]


def calculate_terminal_prices(
    parameters: OptionParameters,
    shocks: np.ndarray,
) -> np.ndarray:
    """Calculate terminal asset prices under risk neutrality."""

    drift = (
        parameters.risk_free_rate
        - parameters.dividend_yield
        - 0.5 * parameters.volatility**2
    ) * parameters.time_to_maturity

    diffusion = (
        parameters.volatility
        * np.sqrt(parameters.time_to_maturity)
        * shocks
    )

    terminal_prices = (
        parameters.spot_price
        * np.exp(drift + diffusion)
    )

    return terminal_prices


def calculate_option_payoffs(
    terminal_prices: np.ndarray,
    strike_price: float,
    option_type: OptionType,
) -> np.ndarray:
    """Calculate terminal call or put payoffs."""

    if option_type == "call":
        return np.maximum(
            terminal_prices - strike_price,
            0.0,
        )

    return np.maximum(
        strike_price - terminal_prices,
        0.0,
    )


def monte_carlo_option_price(
    parameters: OptionParameters,
    option_type: OptionType,
    simulations: int = 500_000,
    seed: int | None = 42,
    antithetic: bool = True,
    confidence_level: float = 0.95,
) -> MonteCarloResult:
    """
    Calculate a European option price using Monte Carlo simulation.

    Terminal asset prices are simulated under the risk-neutral
    probability measure. Antithetic variates are enabled by default
    to improve numerical stability.
    """

    parameters.validate()

    normalized_option_type = option_type.lower()

    validate_monte_carlo_inputs(
        simulations=simulations,
        option_type=normalized_option_type,
    )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "Confidence level must be between zero and one."
        )

    random_generator = np.random.default_rng(seed)

    shocks = generate_standard_normal_shocks(
        simulations=simulations,
        random_generator=random_generator,
        antithetic=antithetic,
    )

    terminal_prices = calculate_terminal_prices(
        parameters=parameters,
        shocks=shocks,
    )

    payoffs = calculate_option_payoffs(
        terminal_prices=terminal_prices,
        strike_price=parameters.strike_price,
        option_type=normalized_option_type,
    )

    discount_factor = np.exp(
        -parameters.risk_free_rate
        * parameters.time_to_maturity
    )

    discounted_payoffs = (
        discount_factor * payoffs
    )

    estimated_price = float(
        np.mean(discounted_payoffs)
    )

    standard_error = float(
        np.std(
            discounted_payoffs,
            ddof=1,
        )
        / np.sqrt(simulations)
    )

    critical_probability = (
        0.5 + confidence_level / 2.0
    )

    critical_value = float(
        norm.ppf(critical_probability)
    )

    margin_of_error = (
        critical_value * standard_error
    )

    confidence_interval_low = (
        estimated_price - margin_of_error
    )

    confidence_interval_high = (
        estimated_price + margin_of_error
    )

    return MonteCarloResult(
        price=estimated_price,
        standard_error=standard_error,
        confidence_interval_low=float(
            confidence_interval_low
        ),
        confidence_interval_high=float(
            confidence_interval_high
        ),
        simulations=simulations,
    )


def monte_carlo_call_price(
    parameters: OptionParameters,
    simulations: int = 500_000,
    seed: int | None = 42,
    antithetic: bool = True,
) -> MonteCarloResult:
    """Calculate a European call price with Monte Carlo."""

    return monte_carlo_option_price(
        parameters=parameters,
        option_type="call",
        simulations=simulations,
        seed=seed,
        antithetic=antithetic,
    )


def monte_carlo_put_price(
    parameters: OptionParameters,
    simulations: int = 500_000,
    seed: int | None = 42,
    antithetic: bool = True,
) -> MonteCarloResult:
    """Calculate a European put price with Monte Carlo."""

    return monte_carlo_option_price(
        parameters=parameters,
        option_type="put",
        simulations=simulations,
        seed=seed,
        antithetic=antithetic,
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

    number_of_simulations = 500_000

    call_result = monte_carlo_call_price(
        parameters=baseline_parameters,
        simulations=number_of_simulations,
        seed=42,
        antithetic=True,
    )

    put_result = monte_carlo_put_price(
        parameters=baseline_parameters,
        simulations=number_of_simulations,
        seed=42,
        antithetic=True,
    )

    print("\nRisk-neutral Monte Carlo option pricing")
    print(
        f"Number of simulations: "
        f"{number_of_simulations:,}"
    )

    print("\nEuropean call")
    print(f"Estimated price: ${call_result.price:,.4f}")
    print(
        f"Standard error: "
        f"${call_result.standard_error:,.6f}"
    )
    print(
        "95% confidence interval: "
        f"[${call_result.confidence_interval_low:,.4f}, "
        f"${call_result.confidence_interval_high:,.4f}]"
    )

    print("\nEuropean put")
    print(f"Estimated price: ${put_result.price:,.4f}")
    print(
        f"Standard error: "
        f"${put_result.standard_error:,.6f}"
    )
    print(
        "95% confidence interval: "
        f"[${put_result.confidence_interval_low:,.4f}, "
        f"${put_result.confidence_interval_high:,.4f}]"
    )