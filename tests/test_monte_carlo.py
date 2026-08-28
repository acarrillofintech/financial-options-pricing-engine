"""Tests for risk-neutral Monte Carlo option pricing."""

import numpy as np
import pytest

from src.black_scholes import (
    OptionParameters,
    black_scholes_call,
    black_scholes_put,
)
from src.monte_carlo import (
    MonteCarloResult,
    calculate_option_payoffs,
    calculate_terminal_prices,
    generate_standard_normal_shocks,
    monte_carlo_call_price,
    monte_carlo_option_price,
    monte_carlo_put_price,
)


@pytest.fixture
def baseline_parameters() -> OptionParameters:
    """Return baseline parameters for Monte Carlo tests."""

    return OptionParameters(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )


def test_monte_carlo_call_is_close_to_black_scholes(
    baseline_parameters: OptionParameters,
) -> None:
    """Monte Carlo call price should be statistically close."""

    result = monte_carlo_call_price(
        parameters=baseline_parameters,
        simulations=200_000,
        seed=42,
    )

    analytical_price = black_scholes_call(
        baseline_parameters
    )

    assert result.price == pytest.approx(
        analytical_price,
        abs=4.0 * result.standard_error,
    )


def test_monte_carlo_put_is_close_to_black_scholes(
    baseline_parameters: OptionParameters,
) -> None:
    """Monte Carlo put price should be statistically close."""

    result = monte_carlo_put_price(
        parameters=baseline_parameters,
        simulations=200_000,
        seed=42,
    )

    analytical_price = black_scholes_put(
        baseline_parameters
    )

    assert result.price == pytest.approx(
        analytical_price,
        abs=4.0 * result.standard_error,
    )


def test_call_confidence_interval_contains_black_scholes(
    baseline_parameters: OptionParameters,
) -> None:
    """The call confidence interval should contain the analytical price."""

    result = monte_carlo_call_price(
        parameters=baseline_parameters,
        simulations=200_000,
        seed=42,
    )

    analytical_price = black_scholes_call(
        baseline_parameters
    )

    assert (
        result.confidence_interval_low
        <= analytical_price
        <= result.confidence_interval_high
    )


def test_put_confidence_interval_contains_black_scholes(
    baseline_parameters: OptionParameters,
) -> None:
    """The put confidence interval should contain the analytical price."""

    result = monte_carlo_put_price(
        parameters=baseline_parameters,
        simulations=200_000,
        seed=42,
    )

    analytical_price = black_scholes_put(
        baseline_parameters
    )

    assert (
        result.confidence_interval_low
        <= analytical_price
        <= result.confidence_interval_high
    )


def test_simulation_is_reproducible(
    baseline_parameters: OptionParameters,
) -> None:
    """The same seed should produce the same result."""

    first_result = monte_carlo_call_price(
        parameters=baseline_parameters,
        simulations=10_000,
        seed=42,
    )

    second_result = monte_carlo_call_price(
        parameters=baseline_parameters,
        simulations=10_000,
        seed=42,
    )

    assert first_result == second_result


def test_different_seeds_produce_different_results(
    baseline_parameters: OptionParameters,
) -> None:
    """Different seeds should normally produce different estimates."""

    first_result = monte_carlo_call_price(
        parameters=baseline_parameters,
        simulations=10_000,
        seed=42,
    )

    second_result = monte_carlo_call_price(
        parameters=baseline_parameters,
        simulations=10_000,
        seed=100,
    )

    assert first_result.price != second_result.price


def test_result_is_monte_carlo_dataclass(
    baseline_parameters: OptionParameters,
) -> None:
    """The simulation should return a MonteCarloResult."""

    result = monte_carlo_call_price(
        parameters=baseline_parameters,
        simulations=10_000,
        seed=42,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.simulations == 10_000


def test_confidence_interval_contains_estimate(
    baseline_parameters: OptionParameters,
) -> None:
    """The estimated price should lie inside its interval."""

    result = monte_carlo_call_price(
        parameters=baseline_parameters,
        simulations=10_000,
        seed=42,
    )

    assert (
        result.confidence_interval_low
        < result.price
        < result.confidence_interval_high
    )

    assert result.standard_error > 0.0


def test_terminal_prices_are_positive(
    baseline_parameters: OptionParameters,
) -> None:
    """Geometric Brownian Motion should produce positive prices."""

    shocks = np.array(
        [-3.0, -1.0, 0.0, 1.0, 3.0]
    )

    terminal_prices = calculate_terminal_prices(
        parameters=baseline_parameters,
        shocks=shocks,
    )

    assert np.all(terminal_prices > 0.0)


def test_call_payoffs_are_calculated_correctly() -> None:
    """Call payoffs should equal max(S_T - K, 0)."""

    terminal_prices = np.array(
        [80.0, 100.0, 120.0]
    )

    payoffs = calculate_option_payoffs(
        terminal_prices=terminal_prices,
        strike_price=100.0,
        option_type="call",
    )

    np.testing.assert_array_equal(
        payoffs,
        np.array([0.0, 0.0, 20.0]),
    )


def test_put_payoffs_are_calculated_correctly() -> None:
    """Put payoffs should equal max(K - S_T, 0)."""

    terminal_prices = np.array(
        [80.0, 100.0, 120.0]
    )

    payoffs = calculate_option_payoffs(
        terminal_prices=terminal_prices,
        strike_price=100.0,
        option_type="put",
    )

    np.testing.assert_array_equal(
        payoffs,
        np.array([20.0, 0.0, 0.0]),
    )


def test_antithetic_shocks_have_zero_mean() -> None:
    """An even antithetic sample should have a zero mean."""

    random_generator = np.random.default_rng(42)

    shocks = generate_standard_normal_shocks(
        simulations=1_000,
        random_generator=random_generator,
        antithetic=True,
    )

    assert float(shocks.mean()) == pytest.approx(
        0.0,
        abs=1e-15,
    )


@pytest.mark.parametrize(
    "invalid_simulations",
    [
        -100,
        0,
        1,
    ],
)
def test_invalid_simulation_count_raises_value_error(
    baseline_parameters: OptionParameters,
    invalid_simulations: int,
) -> None:
    """Simulation counts below two should raise ValueError."""

    with pytest.raises(ValueError):
        monte_carlo_call_price(
            parameters=baseline_parameters,
            simulations=invalid_simulations,
        )


def test_non_integer_simulations_raise_type_error(
    baseline_parameters: OptionParameters,
) -> None:
    """A non-integer simulation count should raise TypeError."""

    with pytest.raises(TypeError):
        monte_carlo_call_price(
            parameters=baseline_parameters,
            simulations=1_000.5,  # type: ignore[arg-type]
        )


def test_invalid_option_type_raises_value_error(
    baseline_parameters: OptionParameters,
) -> None:
    """An unsupported option type should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="Option type must be 'call' or 'put'.",
    ):
        monte_carlo_option_price(
            parameters=baseline_parameters,
            option_type="invalid",  # type: ignore[arg-type]
            simulations=1_000,
        )


@pytest.mark.parametrize(
    "invalid_confidence_level",
    [
        -0.10,
        0.0,
        1.0,
        1.10,
    ],
)
def test_invalid_confidence_level_raises_value_error(
    baseline_parameters: OptionParameters,
    invalid_confidence_level: float,
) -> None:
    """Confidence levels outside (0, 1) should raise ValueError."""

    with pytest.raises(ValueError):
        monte_carlo_option_price(
            parameters=baseline_parameters,
            option_type="call",
            simulations=1_000,
            confidence_level=invalid_confidence_level,
        )