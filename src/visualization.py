"""Visualizations for the financial options pricing engine."""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.binomial_tree import (
    binomial_call_price,
    binomial_put_price,
)
from src.black_scholes import (
    OptionParameters,
    black_scholes_call,
    black_scholes_put,
)
from src.monte_carlo import (
    monte_carlo_call_price,
    monte_carlo_put_price,
)


FIGURES_DIRECTORY = Path("results/figures")


def configure_style() -> None:
    """Configure the visual style used by all charts."""

    sns.set_theme(
        style="whitegrid",
        context="talk",
    )


def prepare_output_directory() -> None:
    """Create the output directory when necessary."""

    FIGURES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def plot_option_payoffs(
    parameters: OptionParameters,
) -> Path:
    """Plot European call and put terminal payoffs."""

    configure_style()
    prepare_output_directory()

    asset_prices = np.linspace(
        0.50 * parameters.strike_price,
        1.50 * parameters.strike_price,
        300,
    )

    call_payoffs = np.maximum(
        asset_prices - parameters.strike_price,
        0.0,
    )

    put_payoffs = np.maximum(
        parameters.strike_price - asset_prices,
        0.0,
    )

    figure, axis = plt.subplots(
        figsize=(14, 8)
    )

    axis.plot(
        asset_prices,
        call_payoffs,
        color="#1F77B4",
        linewidth=2.5,
        label="European call payoff",
    )

    axis.plot(
        asset_prices,
        put_payoffs,
        color="#D62728",
        linewidth=2.5,
        label="European put payoff",
    )

    axis.axvline(
        parameters.strike_price,
        color="#333333",
        linestyle="--",
        linewidth=1.8,
        label=(
            f"Strike price: "
            f"${parameters.strike_price:,.0f}"
        ),
    )

    axis.set_title("European Option Payoffs at Maturity")
    axis.set_xlabel("Asset price at maturity ($)")
    axis.set_ylabel("Option payoff ($)")
    axis.legend()

    figure.tight_layout()

    output_path = (
        FIGURES_DIRECTORY
        / "option_payoffs.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_binomial_convergence(
    parameters: OptionParameters,
) -> Path:
    """Plot binomial convergence toward Black-Scholes."""

    configure_style()
    prepare_output_directory()

    step_counts = np.array(
        [5, 10, 20, 50, 100, 200, 500, 1_000]
    )

    binomial_calls = np.array(
        [
            binomial_call_price(
                parameters,
                steps=int(steps),
            )
            for steps in step_counts
        ]
    )

    binomial_puts = np.array(
        [
            binomial_put_price(
                parameters,
                steps=int(steps),
            )
            for steps in step_counts
        ]
    )

    analytical_call = black_scholes_call(
        parameters
    )

    analytical_put = black_scholes_put(
        parameters
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 10),
        sharex=True,
    )

    axes[0].plot(
        step_counts,
        binomial_calls,
        marker="o",
        color="#1F77B4",
        linewidth=2,
        label="Binomial call",
    )

    axes[0].axhline(
        analytical_call,
        color="#2CA02C",
        linestyle="--",
        linewidth=2,
        label="Black-Scholes call",
    )

    axes[0].set_ylabel("Call price ($)")
    axes[0].legend()

    axes[1].plot(
        step_counts,
        binomial_puts,
        marker="o",
        color="#D62728",
        linewidth=2,
        label="Binomial put",
    )

    axes[1].axhline(
        analytical_put,
        color="#9467BD",
        linestyle="--",
        linewidth=2,
        label="Black-Scholes put",
    )

    axes[1].set_xlabel("Number of binomial steps")
    axes[1].set_ylabel("Put price ($)")
    axes[1].legend()

    figure.suptitle(
        "Binomial Convergence Toward Black-Scholes"
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIRECTORY
        / "binomial_convergence.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_spot_price_sensitivity(
    parameters: OptionParameters,
) -> Path:
    """Plot option prices across different spot prices."""

    configure_style()
    prepare_output_directory()

    spot_prices = np.linspace(
        0.50 * parameters.strike_price,
        1.50 * parameters.strike_price,
        200,
    )

    call_prices = np.array(
        [
            black_scholes_call(
                replace(
                    parameters,
                    spot_price=float(spot),
                )
            )
            for spot in spot_prices
        ]
    )

    put_prices = np.array(
        [
            black_scholes_put(
                replace(
                    parameters,
                    spot_price=float(spot),
                )
            )
            for spot in spot_prices
        ]
    )

    figure, axis = plt.subplots(
        figsize=(14, 8)
    )

    axis.plot(
        spot_prices,
        call_prices,
        color="#1F77B4",
        linewidth=2.5,
        label="European call",
    )

    axis.plot(
        spot_prices,
        put_prices,
        color="#D62728",
        linewidth=2.5,
        label="European put",
    )

    axis.axvline(
        parameters.strike_price,
        color="#333333",
        linestyle="--",
        linewidth=1.8,
        label="At the money",
    )

    axis.set_title(
        "Option Price Sensitivity to the Underlying Asset"
    )

    axis.set_xlabel("Spot price ($)")
    axis.set_ylabel("Option price ($)")
    axis.legend()

    figure.tight_layout()

    output_path = (
        FIGURES_DIRECTORY
        / "spot_price_sensitivity.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_volatility_sensitivity(
    parameters: OptionParameters,
) -> Path:
    """Plot option prices across different volatilities."""

    configure_style()
    prepare_output_directory()

    volatilities = np.linspace(
        0.05,
        0.60,
        200,
    )

    call_prices = np.array(
        [
            black_scholes_call(
                replace(
                    parameters,
                    volatility=float(volatility),
                )
            )
            for volatility in volatilities
        ]
    )

    put_prices = np.array(
        [
            black_scholes_put(
                replace(
                    parameters,
                    volatility=float(volatility),
                )
            )
            for volatility in volatilities
        ]
    )

    figure, axis = plt.subplots(
        figsize=(14, 8)
    )

    axis.plot(
        volatilities * 100.0,
        call_prices,
        color="#1F77B4",
        linewidth=2.5,
        label="European call",
    )

    axis.plot(
        volatilities * 100.0,
        put_prices,
        color="#D62728",
        linewidth=2.5,
        label="European put",
    )

    axis.axvline(
        parameters.volatility * 100.0,
        color="#333333",
        linestyle="--",
        linewidth=1.8,
        label=(
            f"Baseline volatility: "
            f"{parameters.volatility:.0%}"
        ),
    )

    axis.set_title(
        "Option Price Sensitivity to Volatility"
    )

    axis.set_xlabel("Annual volatility (%)")
    axis.set_ylabel("Option price ($)")
    axis.legend()

    figure.tight_layout()

    output_path = (
        FIGURES_DIRECTORY
        / "volatility_sensitivity.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_method_comparison(
    parameters: OptionParameters,
) -> Path:
    """Compare Black-Scholes, binomial, and Monte Carlo prices."""

    configure_style()
    prepare_output_directory()

    analytical_call = black_scholes_call(
        parameters
    )

    analytical_put = black_scholes_put(
        parameters
    )

    binomial_call = binomial_call_price(
        parameters,
        steps=500,
    )

    binomial_put = binomial_put_price(
        parameters,
        steps=500,
    )

    monte_carlo_call = monte_carlo_call_price(
        parameters,
        simulations=500_000,
        seed=42,
    ).price

    monte_carlo_put = monte_carlo_put_price(
        parameters,
        simulations=500_000,
        seed=42,
    ).price

    methods = [
        "Black-Scholes",
        "Binomial",
        "Monte Carlo",
    ]

    call_prices = [
        analytical_call,
        binomial_call,
        monte_carlo_call,
    ]

    put_prices = [
        analytical_put,
        binomial_put,
        monte_carlo_put,
    ]

    positions = np.arange(len(methods))
    bar_width = 0.36

    figure, axis = plt.subplots(
        figsize=(14, 8)
    )

    call_bars = axis.bar(
        positions - bar_width / 2,
        call_prices,
        width=bar_width,
        color="#1F77B4",
        label="European call",
    )

    put_bars = axis.bar(
        positions + bar_width / 2,
        put_prices,
        width=bar_width,
        color="#D62728",
        label="European put",
    )

    axis.bar_label(
        call_bars,
        fmt="$%.4f",
        padding=4,
    )

    axis.bar_label(
        put_bars,
        fmt="$%.4f",
        padding=4,
    )

    axis.set_title(
        "European Option Pricing Method Comparison"
    )

    axis.set_xlabel("Pricing method")
    axis.set_ylabel("Option price ($)")
    axis.set_xticks(
        positions,
        methods,
    )

    axis.legend()

    figure.tight_layout()

    output_path = (
        FIGURES_DIRECTORY
        / "method_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_all_visualizations(
    parameters: OptionParameters,
) -> list[Path]:
    """Create and save all project visualizations."""

    return [
        plot_option_payoffs(parameters),
        plot_binomial_convergence(parameters),
        plot_spot_price_sensitivity(parameters),
        plot_volatility_sensitivity(parameters),
        plot_method_comparison(parameters),
    ]


if __name__ == "__main__":
    baseline_parameters = OptionParameters(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
    )

    generated_figures = create_all_visualizations(
        baseline_parameters
    )

    print("\nGenerated figures")

    for figure_path in generated_figures:
        print(f"- {figure_path}")