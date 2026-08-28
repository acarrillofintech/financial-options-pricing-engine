"""Tests for the Cox-Ross-Rubinstein binomial tree."""

import numpy as np
import pytest

from src.binomial_tree import (
    binomial_call_price,
    binomial_option_price,
    binomial_put_price,
    calculate_intrinsic_value,
)
from src.black_scholes import (
    OptionParameters,
    black_scholes_call,
    black_scholes_put,
)


@pytest.fixture
def baseline_parameters() -> OptionParameters:
    """Return baseline parameters for binomial tests."""

    return OptionParameters(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )


def test_european_call_matches_known_binomial_value(
    baseline_parameters: OptionParameters,
) -> None:
    """The 500-step European call should match a known value."""

    price = binomial_call_price(
        baseline_parameters,
        steps=500,
        exercise_style="european",
    )

    assert price == pytest.approx(
        10.4465851364,
        abs=1e-8,
    )


def test_european_put_matches_known_binomial_value(
    baseline_parameters: OptionParameters,
) -> None:
    """The 500-step European put should match a known value."""

    price = binomial_put_price(
        baseline_parameters,
        steps=500,
        exercise_style="european",
    )

    assert price == pytest.approx(
        5.5695275865,
        abs=1e-8,
    )


def test_american_call_matches_known_value(
    baseline_parameters: OptionParameters,
) -> None:
    """An American call without dividends should match its known value."""

    price = binomial_call_price(
        baseline_parameters,
        steps=500,
        exercise_style="american",
    )

    assert price == pytest.approx(
        10.4465851364,
        abs=1e-8,
    )


def test_american_put_matches_known_value(
    baseline_parameters: OptionParameters,
) -> None:
    """The American put should match its known binomial value."""

    price = binomial_put_price(
        baseline_parameters,
        steps=500,
        exercise_style="american",
    )

    assert price == pytest.approx(
        6.0888101107,
        abs=1e-8,
    )


def test_european_call_converges_to_black_scholes(
    baseline_parameters: OptionParameters,
) -> None:
    """The binomial call should converge to Black-Scholes."""

    binomial_price = binomial_call_price(
        baseline_parameters,
        steps=1_000,
    )

    analytical_price = black_scholes_call(
        baseline_parameters
    )

    assert binomial_price == pytest.approx(
        analytical_price,
        abs=0.01,
    )


def test_european_put_converges_to_black_scholes(
    baseline_parameters: OptionParameters,
) -> None:
    """The binomial put should converge to Black-Scholes."""

    binomial_price = binomial_put_price(
        baseline_parameters,
        steps=1_000,
    )

    analytical_price = black_scholes_put(
        baseline_parameters
    )

    assert binomial_price == pytest.approx(
        analytical_price,
        abs=0.01,
    )


def test_american_put_is_not_less_than_european_put(
    baseline_parameters: OptionParameters,
) -> None:
    """Early exercise must not reduce the value of an American put."""

    european_price = binomial_put_price(
        baseline_parameters,
        steps=500,
        exercise_style="european",
    )

    american_price = binomial_put_price(
        baseline_parameters,
        steps=500,
        exercise_style="american",
    )

    assert american_price >= european_price


def test_american_call_equals_european_without_dividends(
    baseline_parameters: OptionParameters,
) -> None:
    """Without dividends, early exercise adds no call value."""

    european_price = binomial_call_price(
        baseline_parameters,
        steps=500,
        exercise_style="european",
    )

    american_price = binomial_call_price(
        baseline_parameters,
        steps=500,
        exercise_style="american",
    )

    assert american_price == pytest.approx(
        european_price,
        abs=1e-10,
    )


def test_call_intrinsic_values() -> None:
    """Call intrinsic values should be calculated correctly."""

    asset_prices = np.array([80.0, 100.0, 120.0])

    intrinsic_values = calculate_intrinsic_value(
        asset_prices=asset_prices,
        strike_price=100.0,
        option_type="call",
    )

    np.testing.assert_array_equal(
        intrinsic_values,
        np.array([0.0, 0.0, 20.0]),
    )


def test_put_intrinsic_values() -> None:
    """Put intrinsic values should be calculated correctly."""

    asset_prices = np.array([80.0, 100.0, 120.0])

    intrinsic_values = calculate_intrinsic_value(
        asset_prices=asset_prices,
        strike_price=100.0,
        option_type="put",
    )

    np.testing.assert_array_equal(
        intrinsic_values,
        np.array([20.0, 0.0, 0.0]),
    )


@pytest.mark.parametrize(
    "invalid_steps",
    [
        0,
        -1,
    ],
)
def test_invalid_step_count_raises_value_error(
    baseline_parameters: OptionParameters,
    invalid_steps: int,
) -> None:
    """Zero or negative step counts should raise ValueError."""

    with pytest.raises(ValueError):
        binomial_call_price(
            baseline_parameters,
            steps=invalid_steps,
        )


def test_non_integer_steps_raise_type_error(
    baseline_parameters: OptionParameters,
) -> None:
    """A non-integer step count should raise TypeError."""

    with pytest.raises(TypeError):
        binomial_call_price(
            baseline_parameters,
            steps=10.5,  # type: ignore[arg-type]
        )


def test_invalid_option_type_raises_value_error(
    baseline_parameters: OptionParameters,
) -> None:
    """An unsupported option type should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="Option type must be 'call' or 'put'.",
    ):
        binomial_option_price(
            parameters=baseline_parameters,
            option_type="invalid",  # type: ignore[arg-type]
        )


def test_invalid_exercise_style_raises_value_error(
    baseline_parameters: OptionParameters,
) -> None:
    """An unsupported exercise style should raise ValueError."""

    with pytest.raises(
        ValueError,
        match=(
            "Exercise style must be "
            "'european' or 'american'."
        ),
    ):
        binomial_call_price(
            parameters=baseline_parameters,
            exercise_style="invalid",  # type: ignore[arg-type]
        )