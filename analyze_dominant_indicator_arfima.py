"""Run an earlier broad diagnostic of the dominant PCA reference component.

This development-history script is not the retained thesis analysis. It
contains additional diagnostic benchmarks and stability checks that are not
part of the reported comparison. Reproduce the thesis results with
``scripts/analyze_covariance_forecasts.py`` or
``notebooks/reproduce_thesis_results.ipynb`` instead.

The fixed PCA reference basis is estimated exclusively from the outer training
covariance matrices.  The dominant transformed component

    lambda_t = (B_tilde.T @ Sigma_t @ B_tilde)[0, 0]

is then forecast in the outer test period by:

1. the training mean as a diagnostic benchmark,
2. the naive one-step forecast lambda_(t-1),
3. a stationary ARFIMA(0, d, 0) model,
4. the observed lambda_t as an oracle representation benchmark.

The ARFIMA model is

    (1 - L)^d (lambda_t - mu) = epsilon_t.

The fractional parameter d is estimated once from the complete outer training
series by a parametric Whittle approximation to the Gaussian likelihood.  The
search is restricted to -0.49 <= d <= 0.49.  The training mean, d, and
innovation variance remain fixed throughout the test.  At every one-step test
origin, all dominant-component observations available strictly before the
forecast timestamp are used.

This is an isolated dominant-indicator experiment.  No residual correction is
included yet.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pca_covariance_forecasting.covariance import (
    compute_block_covariances,
)
from pca_covariance_forecasting.data import (
    compute_log_returns,
    load_closing_prices,
)
from pca_covariance_forecasting.evaluation import (
    compute_aggregated_relative_frobenius_error,
)
from pca_covariance_forecasting.pca import (
    compute_reference_covariance,
    compute_reference_eigendecomposition,
    construct_reference_basis_approximations,
    transform_covariances_from_reference_basis,
    transform_covariances_to_reference_basis,
)


DATA_DIRECTORY = Path("data/raw/binance")
D_LOWER_BOUND = -0.49
D_UPPER_BOUND = 0.49
D_GRID_SIZE = 981
D_BOUNDARY_WARNING_DISTANCE = 0.01
OPTIMIZATION_TOLERANCE = 1e-12
OPTIMIZATION_MAX_ITERATIONS = 200
STABILITY_SHARES = (0.50, 0.75, 1.00)


def select_matrix_timestamps(
    matrices: pd.DataFrame,
    timestamps: pd.Index,
) -> pd.DataFrame:
    mask = (
        matrices.index
        .get_level_values("timestamp")
        .isin(timestamps)
    )
    return matrices.loc[mask].copy()


def matrix_frame_to_array(
    matrices: pd.DataFrame,
    assets: pd.Index,
) -> tuple[np.ndarray, pd.Index]:
    timestamps = (
        matrices.index
        .get_level_values("timestamp")
        .unique()
    )

    values = np.stack(
        [
            matrices.xs(
                timestamp,
                level="timestamp",
            ).loc[assets, assets].to_numpy()
            for timestamp in timestamps
        ]
    )

    return values, timestamps


def fit_reference_representation(
    covariances: pd.DataFrame,
    training_covariances: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    reference_covariance = compute_reference_covariance(
        training_covariances
    )
    reference_basis, reference_eigenvalues = (
        compute_reference_eigendecomposition(
            reference_covariance
        )
    )
    transformed_covariances = (
        transform_covariances_to_reference_basis(
            covariances,
            reference_basis,
        )
    )

    first_component = reference_eigenvalues.index[0]
    dominant_indicator = (
        transformed_covariances
        .xs(
            first_component,
            level="component",
        )
        .loc[:, first_component]
        .copy()
    )
    dominant_indicator.index.name = "timestamp"
    dominant_indicator.name = "dominant_indicator"

    oracle_transformed_approximations = (
        construct_reference_basis_approximations(
            transformed_covariances,
            reference_eigenvalues,
        )
    )
    oracle_one_indicator_covariances = (
        transform_covariances_from_reference_basis(
            oracle_transformed_approximations,
            reference_basis,
        )
    )

    return (
        reference_basis,
        reference_eigenvalues,
        dominant_indicator,
        oracle_one_indicator_covariances,
    )


def construct_covariances_from_indicator(
    indicator: pd.Series,
    reference_basis: pd.DataFrame,
    reference_eigenvalues: pd.Series,
) -> pd.DataFrame:
    components = reference_eigenvalues.index
    first_component = components[0]
    transformed_approximations = []

    for value in indicator.to_numpy():
        diagonal = reference_eigenvalues.copy()
        diagonal.loc[first_component] = value

        transformed_approximations.append(
            pd.DataFrame(
                np.diag(diagonal.to_numpy()),
                index=components,
                columns=components,
            )
        )

    transformed_frame = pd.concat(
        transformed_approximations,
        keys=indicator.index,
        names=["timestamp", "component"],
    )

    return transform_covariances_from_reference_basis(
        transformed_frame,
        reference_basis,
    )


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
    objective,
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
        "boundary_warning": (
            boundary_distance
            < D_BOUNDARY_WARNING_DISTANCE
        ),
    }


def fractional_differencing_weights(
    fractional_parameter: float,
    count: int,
) -> np.ndarray:
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

    return np.asarray(forecasts)


def fractionally_difference_observed_series(
    observed_values: np.ndarray,
    fractional_parameter: float,
    mean: float,
) -> np.ndarray:
    centered = (
        np.asarray(observed_values, dtype=float)
        - mean
    )
    weights = fractional_differencing_weights(
        fractional_parameter,
        len(centered),
    )

    return np.convolve(
        centered,
        weights,
        mode="full",
    )[:len(centered)]


def calculate_lag_correlation(
    values: np.ndarray,
    lag: int,
) -> float:
    values = np.asarray(values, dtype=float)

    if lag < 1 or lag >= len(values):
        raise ValueError(
            "The lag must be positive and smaller "
            "than the series length."
        )

    previous = values[:-lag]
    current = values[lag:]

    if (
        np.std(previous, ddof=0) == 0.0
        or np.std(current, ddof=0) == 0.0
    ):
        return np.nan

    return float(
        np.corrcoef(previous, current)[0, 1]
    )


def build_training_d_stability_table(
    training_values: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for share in STABILITY_SHARES:
        observation_count = int(
            np.floor(
                share
                * len(training_values)
            )
        )
        fit = fit_arfima_0d0_whittle(
            training_values[:observation_count]
        )
        rows.append(
            {
                "training_share": share,
                "observation_count": observation_count,
                "d": fit["d"],
                "mean": fit["mean"],
                "innovation_variance": (
                    fit["innovation_variance"]
                ),
                "distance_to_parameter_boundary": (
                    fit[
                        "distance_to_parameter_boundary"
                    ]
                ),
                "boundary_warning": (
                    fit["boundary_warning"]
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("training_share")
    )


def calculate_scalar_forecast_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    errors = forecast - actual

    if (
        np.std(actual, ddof=0) > 0.0
        and np.std(forecast, ddof=0) > 0.0
    ):
        correlation = float(
            np.corrcoef(actual, forecast)[0, 1]
        )
    else:
        correlation = np.nan

    return {
        "rmse": float(
            np.sqrt(
                np.mean(errors**2)
            )
        ),
        "mae": float(
            np.mean(
                np.abs(errors)
            )
        ),
        "bias": float(np.mean(errors)),
        "forecast_actual_correlation": correlation,
        "squared_error": float(
            np.sum(errors**2)
        ),
        "minimum_forecast": float(
            np.min(forecast)
        ),
        "negative_forecast_share": float(
            np.mean(forecast < 0.0)
        ),
    }


def build_indicator_forecast_comparison(
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
) -> pd.DataFrame:
    metrics = {
        method: calculate_scalar_forecast_metrics(
            actual.to_numpy(),
            forecast.to_numpy(),
        )
        for method, forecast in forecasts.items()
    }
    naive_metrics = metrics["naive_indicator"]
    rows = []

    for method, method_metrics in metrics.items():
        rmse_ratio = (
            method_metrics["rmse"]
            / naive_metrics["rmse"]
        )
        squared_error_ratio = (
            method_metrics["squared_error"]
            / naive_metrics["squared_error"]
        )
        rows.append(
            {
                "method": method,
                "rmse": method_metrics["rmse"],
                "mae": method_metrics["mae"],
                "bias": method_metrics["bias"],
                "forecast_actual_correlation": (
                    method_metrics[
                        "forecast_actual_correlation"
                    ]
                ),
                "rmse_ratio_to_naive": rmse_ratio,
                "rmse_change_percent": (
                    100.0
                    * (rmse_ratio - 1.0)
                ),
                "squared_error_ratio_to_naive": (
                    squared_error_ratio
                ),
                "squared_error_share_captured": (
                    1.0 - squared_error_ratio
                ),
                "minimum_forecast": (
                    method_metrics[
                        "minimum_forecast"
                    ]
                ),
                "negative_forecast_share": (
                    method_metrics[
                        "negative_forecast_share"
                    ]
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("method")
    )


def calculate_matrix_component_metrics(
    actual_values: np.ndarray,
    forecast_values: np.ndarray,
    component: str,
) -> dict[str, float]:
    errors = actual_values - forecast_values
    dimension = actual_values.shape[1]

    if component == "overall":
        actual_component = actual_values.reshape(-1)
        error_component = errors.reshape(-1)
    elif component == "diagonal":
        positions = np.arange(dimension)
        actual_component = actual_values[
            :,
            positions,
            positions,
        ].reshape(-1)
        error_component = errors[
            :,
            positions,
            positions,
        ].reshape(-1)
    elif component == "off-diagonal":
        off_diagonal = ~np.eye(
            dimension,
            dtype=bool,
        )
        actual_component = actual_values[
            :,
            off_diagonal,
        ].reshape(-1)
        error_component = errors[
            :,
            off_diagonal,
        ].reshape(-1)
    else:
        raise ValueError(
            f"Unknown matrix component: {component}"
        )

    squared_error = float(
        np.sum(error_component**2)
    )
    actual_squared_norm = float(
        np.sum(actual_component**2)
    )

    return {
        "rmse": float(
            np.sqrt(
                np.mean(error_component**2)
            )
        ),
        "aggregated_relative_error": float(
            np.sqrt(
                squared_error
                / actual_squared_norm
            )
        ),
        "squared_error": squared_error,
    }


def build_covariance_forecast_comparison(
    actual_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    assets = actual_covariances.columns
    actual_values, timestamps = matrix_frame_to_array(
        actual_covariances,
        assets,
    )
    metrics = {}

    for method, forecast in covariance_forecasts.items():
        selected_forecast = select_matrix_timestamps(
            forecast,
            timestamps,
        )
        (
            forecast_values,
            forecast_timestamps,
        ) = matrix_frame_to_array(
            selected_forecast,
            assets,
        )

        if not timestamps.equals(forecast_timestamps):
            raise ValueError(
                "Actual and forecast timestamps do not match."
            )

        for component in (
            "overall",
            "diagonal",
            "off-diagonal",
        ):
            component_metrics = (
                calculate_matrix_component_metrics(
                    actual_values,
                    forecast_values,
                    component,
                )
            )

            if component == "overall":
                forecast_errors = (
                    actual_covariances
                    - selected_forecast
                )
                component_metrics[
                    "aggregated_relative_error"
                ] = (
                    compute_aggregated_relative_frobenius_error(
                        forecast_errors,
                        actual_covariances,
                    )
                )

            metrics[(method, component)] = (
                component_metrics
            )

    rows = []

    for (
        method,
        component,
    ), method_metrics in metrics.items():
        naive_metrics = metrics[
            ("naive_indicator", component)
        ]
        rmse_ratio = (
            method_metrics["rmse"]
            / naive_metrics["rmse"]
        )
        squared_error_ratio = (
            method_metrics["squared_error"]
            / naive_metrics["squared_error"]
        )

        rows.append(
            {
                "method": method,
                "component": component,
                "rmse": method_metrics["rmse"],
                "aggregated_relative_error": (
                    method_metrics[
                        "aggregated_relative_error"
                    ]
                ),
                "rmse_ratio_to_naive": rmse_ratio,
                "rmse_change_percent": (
                    100.0
                    * (rmse_ratio - 1.0)
                ),
                "squared_error_ratio_to_naive": (
                    squared_error_ratio
                ),
                "squared_error_share_captured": (
                    1.0 - squared_error_ratio
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index(["method", "component"])
    )


def build_positive_semidefinite_diagnostics(
    covariance_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for method, forecast in covariance_forecasts.items():
        values, _ = matrix_frame_to_array(
            forecast,
            forecast.columns,
        )
        minimum_eigenvalues = np.linalg.eigvalsh(
            values
        )[:, 0]
        tolerance = (
            1e-12
            * np.maximum(
                1.0,
                np.max(
                    np.abs(values),
                    axis=(1, 2),
                ),
            )
        )

        rows.append(
            {
                "method": method,
                "minimum_eigenvalue": float(
                    np.min(minimum_eigenvalues)
                ),
                "non_psd_count": int(
                    np.sum(
                        minimum_eigenvalues
                        < -tolerance
                    )
                ),
                "non_psd_share": float(
                    np.mean(
                        minimum_eigenvalues
                        < -tolerance
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("method")
    )


def verify_forecast_invariants(
    dominant_indicator: pd.Series,
    training_count: int,
    indicator_forecasts: dict[str, pd.Series],
    reference_basis: pd.DataFrame,
    reference_eigenvalues: pd.Series,
    oracle_one_indicator_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
    arfima_fit: dict[str, float | bool | int],
) -> None:
    test_indicator = dominant_indicator.iloc[
        training_count:
    ]
    test_timestamps = test_indicator.index

    for forecast in indicator_forecasts.values():
        if not forecast.index.equals(test_timestamps):
            raise ValueError(
                "An indicator forecast has mismatched timestamps."
            )

        if not np.isfinite(forecast.to_numpy()).all():
            raise ValueError(
                "An indicator forecast contains non-finite values."
            )

    np.testing.assert_array_equal(
        indicator_forecasts[
            "naive_indicator"
        ].iloc[0],
        dominant_indicator.iloc[
            training_count - 1
        ],
    )
    np.testing.assert_array_equal(
        indicator_forecasts[
            "oracle_indicator"
        ].to_numpy(),
        test_indicator.to_numpy(),
    )

    first_position = np.array([training_count])
    training_only_forecast_container = np.append(
        dominant_indicator.iloc[
            :training_count
        ].to_numpy(),
        0.0,
    )
    first_arfima_forecast = (
        forecast_arfima_0d0_one_step(
            training_only_forecast_container,
            first_position,
            float(arfima_fit["d"]),
            float(arfima_fit["mean"]),
        )[0]
    )
    direct_first_forecast = (
        forecast_arfima_0d0_one_step(
            dominant_indicator.to_numpy(),
            first_position,
            float(arfima_fit["d"]),
            float(arfima_fit["mean"]),
        )[0]
    )
    np.testing.assert_allclose(
        first_arfima_forecast,
        direct_first_forecast,
        rtol=1e-13,
        atol=0.0,
    )

    selected_oracle = select_matrix_timestamps(
        oracle_one_indicator_covariances,
        test_timestamps,
    )
    np.testing.assert_allclose(
        covariance_forecasts[
            "oracle_indicator"
        ].to_numpy(),
        selected_oracle.to_numpy(),
        rtol=1e-13,
        atol=1e-20,
    )

    components = reference_eigenvalues.index
    first_component = components[0]
    transformation_tolerance = (
        1e-12
        * float(
            np.max(
                np.abs(
                    reference_eigenvalues.to_numpy()
                )
            )
        )
    )

    for method, forecast in covariance_forecasts.items():
        transformed = (
            transform_covariances_to_reference_basis(
                forecast,
                reference_basis,
            )
        )
        transformed_values, transformed_timestamps = (
            matrix_frame_to_array(
                transformed,
                components,
            )
        )

        if not test_timestamps.equals(
            transformed_timestamps
        ):
            raise ValueError(
                "A covariance forecast has mismatched timestamps."
            )

        expected_diagonal = np.tile(
            reference_eigenvalues.to_numpy(),
            (len(test_timestamps), 1),
        )
        expected_diagonal[:, 0] = (
            indicator_forecasts[
                method
            ].to_numpy()
        )
        positions = np.arange(len(components))

        np.testing.assert_allclose(
            transformed_values[
                :,
                positions,
                positions,
            ],
            expected_diagonal,
            rtol=1e-12,
            atol=1e-18,
        )

        off_diagonal = ~np.eye(
            len(components),
            dtype=bool,
        )
        np.testing.assert_allclose(
            transformed_values[:, off_diagonal],
            0.0,
            rtol=0.0,
            atol=transformation_tolerance,
        )

        if first_component not in transformed.columns:
            raise ValueError(
                "The first reference component is missing."
            )


def main() -> None:
    prices = load_closing_prices(DATA_DIRECTORY)
    returns = compute_log_returns(prices)
    covariances = compute_block_covariances(returns)

    all_timestamps = (
        covariances.index
        .get_level_values("timestamp")
        .unique()
    )
    training_count = len(all_timestamps) // 2
    training_timestamps = all_timestamps[
        :training_count
    ]
    test_timestamps = all_timestamps[
        training_count:
    ]

    training_covariances = select_matrix_timestamps(
        covariances,
        training_timestamps,
    )
    test_covariances = select_matrix_timestamps(
        covariances,
        test_timestamps,
    )

    (
        reference_basis,
        reference_eigenvalues,
        dominant_indicator,
        oracle_one_indicator_covariances,
    ) = fit_reference_representation(
        covariances,
        training_covariances,
    )

    training_indicator = dominant_indicator.iloc[
        :training_count
    ]
    test_indicator = dominant_indicator.iloc[
        training_count:
    ]
    arfima_fit = fit_arfima_0d0_whittle(
        training_indicator.to_numpy()
    )
    test_positions = np.arange(
        training_count,
        len(dominant_indicator),
    )
    arfima_forecast_values = (
        forecast_arfima_0d0_one_step(
            dominant_indicator.to_numpy(),
            test_positions,
            float(arfima_fit["d"]),
            float(arfima_fit["mean"]),
        )
    )

    indicator_forecasts = {
        "training_mean_indicator": pd.Series(
            float(arfima_fit["mean"]),
            index=test_timestamps,
            name="dominant_indicator",
        ),
        "naive_indicator": (
            dominant_indicator
            .shift(1)
            .iloc[training_count:]
            .copy()
        ),
        "arfima_indicator": pd.Series(
            arfima_forecast_values,
            index=test_timestamps,
            name="dominant_indicator",
        ),
        "oracle_indicator": test_indicator.copy(),
    }

    covariance_forecasts = {
        method: construct_covariances_from_indicator(
            forecast,
            reference_basis,
            reference_eigenvalues,
        )
        for method, forecast in (
            indicator_forecasts.items()
        )
    }

    verify_forecast_invariants(
        dominant_indicator,
        training_count,
        indicator_forecasts,
        reference_basis,
        reference_eigenvalues,
        oracle_one_indicator_covariances,
        covariance_forecasts,
        arfima_fit,
    )

    indicator_comparison = (
        build_indicator_forecast_comparison(
            test_indicator,
            indicator_forecasts,
        )
    )
    covariance_comparison = (
        build_covariance_forecast_comparison(
            test_covariances,
            covariance_forecasts,
        )
    )
    psd_diagnostics = (
        build_positive_semidefinite_diagnostics(
            covariance_forecasts
        )
    )
    d_stability = build_training_d_stability_table(
        training_indicator.to_numpy()
    )

    training_innovations = (
        fractionally_difference_observed_series(
            training_indicator.to_numpy(),
            float(arfima_fit["d"]),
            float(arfima_fit["mean"]),
        )
    )
    diagnostic_start = max(
        50,
        int(np.sqrt(training_count)),
    )
    retained_innovations = training_innovations[
        diagnostic_start:
    ]
    raw_lag_correlations = {
        lag: calculate_lag_correlation(
            training_indicator.to_numpy(),
            lag,
        )
        for lag in (1, 2, 5, 10, 20, 50)
    }
    innovation_lag_correlations = {
        lag: calculate_lag_correlation(
            retained_innovations,
            lag,
        )
        for lag in (1, 2, 5, 10, 20, 50)
    }
    correlation_diagnostics = pd.DataFrame(
        {
            "raw_training_indicator": (
                raw_lag_correlations
            ),
            "fractionally_differenced_training": (
                innovation_lag_correlations
            ),
        }
    )
    correlation_diagnostics.index.name = "lag"

    print(
        "Experiment status: dominant-indicator forecast only.\n"
        "No residual correction is included."
    )
    print(
        "\nSample sizes:"
        f"\nComplete covariance matrices: {len(all_timestamps)}"
        f"\nOuter training: {training_count}"
        f"\nOuter test: {len(test_timestamps)}"
    )
    print(
        "\nARFIMA protocol:\n"
        "- model: ARFIMA(0, d, 0) on indicator levels\n"
        "- mean: sample mean of the outer training indicator\n"
        "- d estimator: parametric profile Whittle estimate\n"
        "- d search interval: [-0.49, 0.49]\n"
        "- no validation or test data used to estimate parameters\n"
        "- parameters fixed throughout the outer test\n"
        "- all strictly past observed indicators used for each "
        "one-step forecast\n"
        "- full finite fractional-AR history used; no arbitrary "
        "lag truncation"
    )

    fit_display = pd.Series(arfima_fit)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        260,
    ):
        print("\nARFIMA fit on all outer training data:")
        print(fit_display.to_string())

        print(
            "\nTraining-only stability of the d estimate:"
        )
        print(d_stability)

        print(
            "\nTraining correlation diagnostics:"
        )
        print(correlation_diagnostics)

        print(
            "\nOuter-test dominant-indicator "
            "forecast comparison:"
        )
        print(indicator_comparison)

        print(
            "\nOuter-test covariance forecast comparison:"
        )
        print(covariance_comparison)

        print(
            "\nPositive-semidefinite diagnostics:"
        )
        print(psd_diagnostics)

    if bool(arfima_fit["boundary_warning"]):
        print(
            "\nWARNING: The Whittle estimate lies within "
            f"{D_BOUNDARY_WARNING_DISTANCE:.2f} of a stationary "
            "parameter boundary. The imposed stationary "
            "ARFIMA(0, d, 0) specification may be binding."
        )

    print(
        "\nAll forecast invariants passed:"
        "\n- the PCA reference basis uses outer training only;"
        "\n- the first naive test forecast uses the final "
        "training indicator;"
        "\n- the first ARFIMA forecast is identical whether "
        "future observations are present or absent;"
        "\n- the oracle forecast reproduces the repository's "
        "observed one-indicator approximation;"
        "\n- every covariance forecast uses the forecasted first "
        "component and fixed remaining reference eigenvalues."
    )


if __name__ == "__main__":
    main()
