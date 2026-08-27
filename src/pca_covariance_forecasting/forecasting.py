"""Fit and apply covariance-factor forecasting models."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from pca_covariance_forecasting.pca import (
    transform_covariances_from_reference_basis,
)

D_LOWER_BOUND = -0.49
D_UPPER_BOUND = 0.49
D_GRID_SIZE = 981
D_BOUNDARY_WARNING_DISTANCE = 0.01
OPTIMIZATION_TOLERANCE = 1e-12
OPTIMIZATION_MAX_ITERATIONS = 200


def prepare_whittle_inputs(
    values: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Center a training series and construct its Whittle inputs.

    Parameters
    ----------
    values:
        One-dimensional training observations in chronological order.

    Returns
    -------
    tuple
        The training-sample mean, the periodogram evaluated at the positive
        Fourier frequencies, and ``log(2 sin(omega / 2))`` at those
        frequencies. The latter is reused across candidate values of ``d``.
    """
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
    """Evaluate the profiled ARFIMA(0, d, 0) Whittle criterion.

    The innovation variance is profiled out for the supplied candidate ``d``.
    The returned tuple contains the objective value and the corresponding
    innovation-variance estimate. Invalid non-positive scales produce
    ``(inf, nan)`` so they cannot be selected by the optimizer.
    """
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
    """Refine a bracketed one-dimensional minimum by golden-section search.

    ``objective`` is assumed to be unimodal on the closed interval. The search
    stops when the interval is narrower than ``OPTIMIZATION_TOLERANCE`` or the
    iteration limit is reached. The midpoint estimate and iteration count are
    returned.
    """
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
    """Fit the retained ARFIMA(0, d, 0) Profile-Whittle specification.

    The location estimate is the training-sample mean. The objective is first
    evaluated on the fixed grid from -0.49 to 0.49. The best grid point and its
    immediate neighbours define the interval refined by golden-section search.
    Returned diagnostics include ``mean``, ``d``, innovation variance,
    objective value, frequency count, grid estimate, refinement iterations,
    and distance from the admissible search boundary.
    """
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
    """Return ``count`` coefficients of the fractional-difference expansion.

    Coefficients are generated recursively from ``pi_0 = 1`` according to the
    ARFIMA(0, d, 0) convention used by the forecast recursion.
    """
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
    """Compute finite-history one-step ARFIMA(0, d, 0) forecasts.

    Parameters
    ----------
    observed_values:
        Full observed series, including holdout values that become known before
        later forecast origins.
    forecast_positions:
        Integer target positions. Only observations strictly before each target
        are used.
    fractional_parameter, mean:
        Training-estimated parameters held fixed across all forecast origins.

    Returns
    -------
    numpy.ndarray
        Forecasts in the same order as ``forecast_positions``. The theoretical
        infinite recursion is truncated at the beginning of the observed sample;
        no pre-sample values are imputed.
    """
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
    """Fit on the training slice and forecast every holdout origin once.

    Model parameters are estimated only from the first
    ``training_observation_count`` values. The observed history expands after
    each holdout realization, while the fitted parameters remain fixed. The
    forecast series and complete fit-diagnostic dictionary are returned.
    """
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


def construct_dominant_indicator_covariance_forecasts(
    dominant_indicator_forecasts: pd.Series,
    reference_basis: pd.DataFrame,
    reference_eigenvalues: pd.Series,
) -> pd.DataFrame:
    """Map scalar indicator forecasts through the fixed-remainder PCA model.

    The first transformed diagonal element is replaced by each forecast. All
    remaining diagonal elements stay at their training-reference eigenvalues,
    and transformed off-diagonal elements are zero before reconstruction in the
    original asset coordinates.
    """
    if dominant_indicator_forecasts.empty:
        raise ValueError(
            "At least one dominant-indicator forecast is required."
        )

    if not dominant_indicator_forecasts.index.is_unique:
        raise ValueError(
            "The forecast index must be unique."
        )

    if (
        not dominant_indicator_forecasts
        .index
        .is_monotonic_increasing
    ):
        raise ValueError(
            "The forecasts must be chronologically ordered."
        )

    if reference_basis.shape[0] != reference_basis.shape[1]:
        raise ValueError(
            "The reference basis must be square."
        )

    if not reference_basis.columns.equals(
        reference_eigenvalues.index
    ):
        raise ValueError(
            "Reference-basis components and reference "
            "eigenvalues must have identical ordering."
        )

    forecast_values = (
        dominant_indicator_forecasts
        .to_numpy(dtype=float)
    )

    if not np.isfinite(forecast_values).all():
        raise ValueError(
            "Dominant-indicator forecasts contain "
            "non-finite values."
        )

    if not np.isfinite(
        reference_basis.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "The reference basis contains non-finite values."
        )

    if not np.isfinite(
        reference_eigenvalues.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Reference eigenvalues contain non-finite values."
        )

    components = reference_basis.columns
    transformed_forecasts = []
    timestamps = []

    for timestamp, forecast_value in (
        dominant_indicator_forecasts.items()
    ):
        forecast_diagonal = (
            reference_eigenvalues.copy()
        )
        forecast_diagonal.iloc[0] = float(
            forecast_value
        )

        transformed_forecasts.append(
            pd.DataFrame(
                np.diag(
                    forecast_diagonal.to_numpy(
                        dtype=float
                    )
                ),
                index=components,
                columns=components,
            )
        )
        timestamps.append(timestamp)

    transformed_forecasts = pd.concat(
        transformed_forecasts,
        keys=timestamps,
        names=["timestamp", "component"],
    )

    return transform_covariances_from_reference_basis(
        transformed_covariances=transformed_forecasts,
        reference_basis=reference_basis,
    )


def construct_direct_naive_covariance_forecasts(
    covariance_matrices: pd.DataFrame,
    training_observation_count: int,
) -> pd.DataFrame:
    """Use the most recently observed covariance as each holdout forecast."""
    if covariance_matrices.empty:
        raise ValueError(
            "At least one covariance matrix is required."
        )

    if not isinstance(
        covariance_matrices.index,
        pd.MultiIndex,
    ):
        raise ValueError(  # noqa: TRY004
            "Covariance matrices must use a MultiIndex."
        )

    if "timestamp" not in covariance_matrices.index.names:
        raise ValueError(
            "The covariance-matrix index must contain "
            "a timestamp level."
        )

    timestamps = (
        covariance_matrices.index
        .get_level_values("timestamp")
        .unique()
    )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Covariance matrices must be chronologically ordered."
        )

    if not (
        0
        < training_observation_count
        < len(timestamps)
    ):
        raise ValueError(
            "training_observation_count must be strictly "
            "between zero and the total number of "
            "covariance matrices."
        )

    forecasts = []
    forecast_timestamps = []

    for target_position in range(
        training_observation_count,
        len(timestamps),
    ):
        previous_timestamp = timestamps[
            target_position - 1
        ]
        target_timestamp = timestamps[
            target_position
        ]

        previous_covariance = covariance_matrices.xs(
            previous_timestamp,
            level="timestamp",
        ).copy()

        forecasts.append(previous_covariance)
        forecast_timestamps.append(target_timestamp)

    return pd.concat(
        forecasts,
        keys=forecast_timestamps,
        names=["timestamp", "asset"],
    )
