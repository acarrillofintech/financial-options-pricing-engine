"""Tests for the Black-Scholes option Greeks."""

import numpy as np
import pytest

from src.black_scholes import OptionParameters
from src.greeks import (
    OptionGreeks,
    calculate_call_greeks,
    calculate_gamma,
    calculate_option_greeks,
    calculate_put_greeks,
    calculate_vega,
)


@pytest.fixture
def baseline_parameters() -> OptionParameters:
    """Return baseline parameters for the Greeks tests."""

    return OptionParameters(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )


def test_call_greeks_match_known_values(
    baseline_parameters: OptionParameters,
) -> None:
    """Call Greeks should match known Black-Scholes results."""

    greeks = calculate_call_greeks(baseline_parameters)

    assert greeks.delta == pytest.approx(0.6368306512)
    assert greeks.gamma == pytest.approx(0.0187620173)
    assert greeks.vega == pytest.approx(0.3752403469)
    assert greeks.theta == pytest.approx(-0.0175726782)
    assert greeks.rho == pytest.approx(0.5323248155)


def test_put_greeks_match_known_values(
    baseline_parameters: OptionParameters,
) -> None:
    """Put Greeks should match known Black-Scholes results."""

    greeks = calculate_put_greeks(baseline_parameters)

    assert greeks.delta == pytest.approx(-0.3631693488)
    assert greeks.gamma == pytest.approx(0.0187620173)
    assert greeks.vega == pytest.approx(0.3752403469)
    assert greeks.theta == pytest.approx(-0.0045421381)
    assert greeks.rho == pytest.approx(-0.4189046090)


def test_call_and_put_have_equal_gamma(
    baseline_parameters: OptionParameters,
) -> None:
    """Call and put options should have the same gamma."""

    call_greeks = calculate_call_greeks(
        baseline_parameters
    )

    put_greeks = calculate_put_greeks(
        baseline_parameters
    )

    assert call_greeks.gamma == pytest.approx(
        put_greeks.gamma
    )


def test_call_and_put_have_equal_vega(
    baseline_parameters: OptionParameters,
) -> None:
    """Call and put options should have the same vega."""

    call_greeks = calculate_call_greeks(
        baseline_parameters
    )

    put_greeks = calculate_put_greeks(
        baseline_parameters
    )

    assert call_greeks.vega == pytest.approx(
        put_greeks.vega
    )


def test_delta_relationship_without_dividends(
    baseline_parameters: OptionParameters,
) -> None:
    """Call delta minus put delta should equal one."""

    call_delta = calculate_call_greeks(
        baseline_parameters
    ).delta

    put_delta = calculate_put_greeks(
        baseline_parameters
    ).delta

    assert call_delta - put_delta == pytest.approx(1.0)


def test_delta_relationship_with_dividends() -> None:
    """Call-put delta difference should equal exp(-qT)."""

    parameters = OptionParameters(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=2.0,
        risk_free_rate=0.05,
        volatility=0.20,
        dividend_yield=0.03,
    )

    call_delta = calculate_call_greeks(
        parameters
    ).delta

    put_delta = calculate_put_greeks(
        parameters
    ).delta

    expected_difference = np.exp(
        -parameters.dividend_yield
        * parameters.time_to_maturity
    )

    assert call_delta - put_delta == pytest.approx(
        expected_difference
    )


def test_shared_gamma_function(
    baseline_parameters: OptionParameters,
) -> None:
    """The shared gamma function should match the call result."""

    gamma = calculate_gamma(baseline_parameters)

    call_gamma = calculate_call_greeks(
        baseline_parameters
    ).gamma

    assert gamma == pytest.approx(call_gamma)


def test_shared_vega_function(
    baseline_parameters: OptionParameters,
) -> None:
    """The shared vega function should match the call result."""

    vega = calculate_vega(baseline_parameters)

    call_vega = calculate_call_greeks(
        baseline_parameters
    ).vega

    assert vega == pytest.approx(call_vega)


def test_generic_function_returns_call_greeks(
    baseline_parameters: OptionParameters,
) -> None:
    """The generic function should dispatch call calculations."""

    generic_greeks = calculate_option_greeks(
        baseline_parameters,
        "call",
    )

    direct_greeks = calculate_call_greeks(
        baseline_parameters
    )

    assert generic_greeks == direct_greeks


def test_generic_function_returns_put_greeks(
    baseline_parameters: OptionParameters,
) -> None:
    """The generic function should dispatch put calculations."""

    generic_greeks = calculate_option_greeks(
        baseline_parameters,
        "put",
    )

    direct_greeks = calculate_put_greeks(
        baseline_parameters
    )

    assert generic_greeks == direct_greeks


def test_greeks_are_returned_as_dataclass(
    baseline_parameters: OptionParameters,
) -> None:
    """The result should be an OptionGreeks instance."""

    greeks = calculate_call_greeks(
        baseline_parameters
    )

    assert isinstance(greeks, OptionGreeks)


def test_invalid_option_type_raises_value_error(
    baseline_parameters: OptionParameters,
) -> None:
    """An unsupported option type should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="Option type must be 'call' or 'put'.",
    ):
        calculate_option_greeks(
            baseline_parameters,
            "invalid",  # type: ignore[arg-type]
        )