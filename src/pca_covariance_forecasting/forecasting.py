"""Fit and apply covariance-factor forecasting models."""

from collections.abc import Callable

import numpy as np
import pandas as pd


D_LOWER_BOUND = -0.49
D_UPPER_BOUND = 0.49
D_GRID_SIZE = 981
D_BOUNDARY_WARNING_DISTANCE = 0.01
OPTIMIZATION_TOLERANCE = 1e-12
OPTIMIZATION_MAX_ITERATIONS = 200


def prepare_whittle_inputs(
    values: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "The Whittle input must be one-dimensional."
        )

    if len(values) < 20:
        raise ValueError(
            "At least 20 observations are required for "
            "the Whittle estimate."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "The Whittle input contains non-finite values."
        )

    mean = float(np.mean(values))
    centered = values - mean

    if np.all(centered == 0.0):
        raise ValueError(
            "The fractional parameter is unidentified "
            "for a constant series."
        )

    sample_size = len(centered)
    positive_frequency_count = (sample_size - 1) // 2

    frequencies = (
        2.0
        * np.pi
        * np.arange(
            1,
            positive_frequency_count + 1,
        )
        / sample_size
    )

    fourier_transform = np.fft.rfft(centered)
    periodogram = (
        np.abs(
            fourier_transform[
                1:
                positive_frequency_count + 1
            ]
        )
        ** 2
        / (2.0 * np.pi * sample_size)
    )

    if not np.isfinite(periodogram).all():
        raise ValueError(
            "The periodogram contains non-finite values."
        )

    if not np.any(periodogram > 0.0):
        raise ValueError(
            "The periodogram contains no positive power."
        )

    log_frequency_factor = np.log(
        2.0 * np.sin(frequencies / 2.0)
    )

    return mean, periodogram, log_frequency_factor


def evaluate_profile_whittle_objective(
    fractional_parameter: float,
    periodogram: np.ndarray,
    log_frequency_factor: np.ndarray,
) -> tuple[float, float]:
    log_spectral_shape = (
        -2.0
        * fractional_parameter
        * log_frequency_factor
    )

    inverse_spectral_shape = np.exp(
        -log_spectral_shape
    )

    innovation_variance = float(
        2.0
        * np.pi
        * np.mean(
            periodogram
            * inverse_spectral_shape
        )
    )

    if (
        not np.isfinite(innovation_variance)
        or innovation_variance <= 0.0
    ):
        return np.inf, np.nan

    objective = float(
        len(periodogram)
        * np.log(innovation_variance)
        + np.sum(log_spectral_shape)
    )

    return objective, innovation_variance


def refine_bounded_minimum(
    objective: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, int]:
    if not lower_bound < upper_bound:
        raise ValueError(
            "The optimization interval must have positive width."
        )

    inverse_golden_ratio = (
        np.sqrt(5.0) - 1.0
    ) / 2.0

    left = float(lower_bound)
    right = float(upper_bound)

    point_left = (
        right
        - inverse_golden_ratio
        * (right - left)
    )
    point_right = (
        left
        + inverse_golden_ratio
        * (right - left)
    )

    value_left = objective(point_left)
    value_right = objective(point_right)
    iterations = 0

    while (
        right - left
        > OPTIMIZATION_TOLERANCE
        and iterations < OPTIMIZATION_MAX_ITERATIONS
    ):
        if value_left <= value_right:
            right = point_right
            point_right = point_left
            value_right = value_left

            point_left = (
                right
                - inverse_golden_ratio
                * (right - left)
            )
            value_left = objective(point_left)
        else:
            left = point_left
            point_left = point_right
            value_left = value_right

            point_right = (
                left
                + inverse_golden_ratio
                * (right - left)
            )
            value_right = objective(point_right)

        iterations += 1

    estimate = (left + right) / 2.0

    return estimate, iterations


def fit_arfima_0d0_whittle(
    values: np.ndarray,
) -> dict[str, float | bool | int]:
    (
        mean,
        periodogram,
        log_frequency_factor,
    ) = prepare_whittle_inputs(values)

    def objective_only(
        fractional_parameter: float,
    ) -> float:
        objective, _ = evaluate_profile_whittle_objective(
            fractional_parameter,
            periodogram,
            log_frequency_factor,
        )
        return objective

    grid = np.linspace(
        D_LOWER_BOUND,
        D_UPPER_BOUND,
        D_GRID_SIZE,
    )

    grid_objectives = np.array(
        [
            objective_only(value)
            for value in grid
        ]
    )

    best_grid_position = int(
        np.argmin(grid_objectives)
    )

    refinement_lower = grid[
        max(0, best_grid_position - 1)
    ]
    refinement_upper = grid[
        min(
            len(grid) - 1,
            best_grid_position + 1,
        )
    ]

    fractional_parameter, iterations = (
        refine_bounded_minimum(
            objective_only,
            refinement_lower,
            refinement_upper,
        )
    )

    objective, innovation_variance = (
        evaluate_profile_whittle_objective(
            fractional_parameter,
            periodogram,
            log_frequency_factor,
        )
    )

    boundary_distance = min(
        fractional_parameter - D_LOWER_BOUND,
        D_UPPER_BOUND - fractional_parameter,
    )

    return {
        "mean": mean,
        "d": float(fractional_parameter),
        "innovation_variance": innovation_variance,
        "profile_whittle_objective": objective,
        "positive_frequency_count": len(periodogram),
        "grid_best_d": float(
            grid[best_grid_position]
        ),
        "refinement_iterations": iterations,
        "distance_to_parameter_boundary": float(
            boundary_distance
        ),
        "boundary_warning": bool(
            boundary_distance
            < D_BOUNDARY_WARNING_DISTANCE
        ),
    }


def fractional_differencing_weights(
    fractional_parameter: float,
    count: int,
) -> np.ndarray:
    if not np.isfinite(fractional_parameter):
        raise ValueError(
            "The fractional parameter must be finite."
        )

    if count < 1:
        raise ValueError(
            "At least one fractional weight is required."
        )

    weights = np.empty(count, dtype=float)
    weights[0] = 1.0

    for lag in range(1, count):
        weights[lag] = (
            weights[lag - 1]
            * (
                lag
                - 1
                - fractional_parameter
            )
            / lag
        )

    return weights


def forecast_arfima_0d0_one_step(
    observed_values: np.ndarray,
    forecast_positions: np.ndarray,
    fractional_parameter: float,
    mean: float,
) -> np.ndarray:
    observed_values = np.asarray(
        observed_values,
        dtype=float,
    )
    forecast_positions = np.asarray(
        forecast_positions,
        dtype=int,
    )

    if observed_values.ndim != 1:
        raise ValueError(
            "Observed values must be one-dimensional."
        )

    if forecast_positions.ndim != 1:
        raise ValueError(
            "Forecast positions must be one-dimensional."
        )

    if not np.isfinite(observed_values).all():
        raise ValueError(
            "Observed values contain non-finite values."
        )

    if not np.isfinite(fractional_parameter):
        raise ValueError(
            "The fractional parameter must be finite."
        )

    if not -0.5 < fractional_parameter < 0.5:
        raise ValueError(
            "The stationary and invertible ARFIMA model "
            "requires -0.5 < d < 0.5."
        )

    if not np.isfinite(mean):
        raise ValueError(
            "The model mean must be finite."
        )

    if (
        len(forecast_positions) > 0
        and (
            forecast_positions.min() < 1
            or forecast_positions.max()
            >= len(observed_values)
        )
    ):
        raise ValueError(
            "Forecast positions are outside the observed series."
        )

    weights = fractional_differencing_weights(
        fractional_parameter,
        len(observed_values),
    )

    centered = observed_values - mean
    forecasts = []

    for position in forecast_positions:
        conditional_centered_forecast = -float(
            weights[1:position + 1]
            @ centered[position - 1::-1]
        )

        forecasts.append(
            mean + conditional_centered_forecast
        )

    return np.asarray(
        forecasts,
        dtype=float,
    )


def fit_and_forecast_arfima_0d0_holdout(
    series: pd.Series,
    training_observation_count: int,
) -> tuple[
    pd.Series,
    dict[str, float | bool | int],
]:
    if not series.index.is_unique:
        raise ValueError(
            "The time-series index must be unique."
        )

    if not series.index.is_monotonic_increasing:
        raise ValueError(
            "The time-series index must be chronologically ordered."
        )

    if not isinstance(
        training_observation_count,
        (int, np.integer),
    ):
        raise TypeError(
            "training_observation_count must be an integer."
        )

    training_observation_count = int(
        training_observation_count
    )

    if training_observation_count < 20:
        raise ValueError(
            "At least 20 training observations are required."
        )

    if training_observation_count >= len(series):
        raise ValueError(
            "The training period must leave at least one "
            "holdout observation."
        )

    values = series.to_numpy(dtype=float)

    fitted_model = fit_arfima_0d0_whittle(
        values[:training_observation_count]
    )

    forecast_positions = np.arange(
        training_observation_count,
        len(values),
        dtype=int,
    )

    forecast_values = forecast_arfima_0d0_one_step(
        observed_values=values,
        forecast_positions=forecast_positions,
        fractional_parameter=float(
            fitted_model["d"]
        ),
        mean=float(fitted_model["mean"]),
    )

    forecasts = pd.Series(
        forecast_values,
        index=series.index[forecast_positions],
        name="arfima_0d0_forecast",
        dtype=float,
    )

    return forecasts, fitted_model
