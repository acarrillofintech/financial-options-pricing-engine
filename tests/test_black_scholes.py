"""Tests for the Black-Scholes option pricing module."""

import pytest

from src.black_scholes import (
    OptionParameters,
    black_scholes_call,
    black_scholes_price,
    black_scholes_put,
    calculate_d1_d2,
    calculate_put_call_parity_difference,
)


@pytest.fixture
def baseline_parameters() -> OptionParameters:
    """Return the baseline option parameters used by the tests."""

    return OptionParameters(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )


def test_calculate_d1_d2(
    baseline_parameters: OptionParameters,
) -> None:
    """The baseline d1 and d2 values should match known results."""

    d1, d2 = calculate_d1_d2(baseline_parameters)

    assert d1 == pytest.approx(0.35)
    assert d2 == pytest.approx(0.15)


def test_baseline_call_price(
    baseline_parameters: OptionParameters,
) -> None:
    """The European call price should match the known result."""

    call_price = black_scholes_call(baseline_parameters)

    assert call_price == pytest.approx(
        10.4505835722,
        rel=1e-10,
    )


def test_baseline_put_price(
    baseline_parameters: OptionParameters,
) -> None:
    """The European put price should match the known result."""

    put_price = black_scholes_put(baseline_parameters)

    assert put_price == pytest.approx(
        5.5735260223,
        rel=1e-10,
    )


def test_generic_price_function_for_call(
    baseline_parameters: OptionParameters,
) -> None:
    """The generic function should return the call price."""

    generic_price = black_scholes_price(
        baseline_parameters,
        "call",
    )

    direct_price = black_scholes_call(baseline_parameters)

    assert generic_price == pytest.approx(direct_price)


def test_generic_price_function_for_put(
    baseline_parameters: OptionParameters,
) -> None:
    """The generic function should return the put price."""

    generic_price = black_scholes_price(
        baseline_parameters,
        "put",
    )

    direct_price = black_scholes_put(baseline_parameters)

    assert generic_price == pytest.approx(direct_price)


def test_option_type_is_case_insensitive(
    baseline_parameters: OptionParameters,
) -> None:
    """Uppercase option types should also be accepted."""

    uppercase_price = black_scholes_price(
        baseline_parameters,
        "CALL",  # type: ignore[arg-type]
    )

    expected_price = black_scholes_call(baseline_parameters)

    assert uppercase_price == pytest.approx(expected_price)


def test_put_call_parity(
    baseline_parameters: OptionParameters,
) -> None:
    """European call and put prices should satisfy put-call parity."""

    parity_difference = calculate_put_call_parity_difference(
        baseline_parameters
    )

    assert parity_difference == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_prices_with_continuous_dividends() -> None:
    """Prices should match known values with a dividend yield."""

    parameters = OptionParameters(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        dividend_yield=0.02,
    )

    call_price = black_scholes_call(parameters)
    put_price = black_scholes_put(parameters)

    assert call_price == pytest.approx(
        9.2270055082,
        rel=1e-10,
    )

    assert put_price == pytest.approx(
        6.3300806275,
        rel=1e-10,
    )


def test_option_prices_are_positive(
    baseline_parameters: OptionParameters,
) -> None:
    """Baseline call and put prices should be positive."""

    assert black_scholes_call(baseline_parameters) > 0
    assert black_scholes_put(baseline_parameters) > 0


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("spot_price", 0.0),
        ("spot_price", -100.0),
        ("strike_price", 0.0),
        ("strike_price", -100.0),
        ("time_to_maturity", 0.0),
        ("time_to_maturity", -1.0),
        ("volatility", 0.0),
        ("volatility", -0.20),
    ],
)
def test_invalid_parameters_raise_value_error(
    field_name: str,
    invalid_value: float,
) -> None:
    """Invalid model parameters should raise ValueError."""

    parameter_values = {
        "spot_price": 100.0,
        "strike_price": 100.0,
        "time_to_maturity": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "dividend_yield": 0.0,
    }

    parameter_values[field_name] = invalid_value
    parameters = OptionParameters(**parameter_values)

    with pytest.raises(ValueError):
        black_scholes_call(parameters)


def test_invalid_option_type_raises_value_error(
    baseline_parameters: OptionParameters,
) -> None:
    """An unsupported option type should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="Option type must be 'call' or 'put'.",
    ):
        black_scholes_price(
            baseline_parameters,
            "invalid",  # type: ignore[arg-type]
        )