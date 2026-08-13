"""Evaluate one-step forecasts of identity-basis residual coefficients.

This is a conditional residual-forecast experiment.  The one-indicator
covariance approximation at time t still uses the observed dominant reference
component at time t.  Only the three diagonal residual coefficients are
forecast.  Consequently, the resulting covariance metrics are not yet metrics
of the final FARIMA-plus-residual forecasting model.

For each asset, an AR order from 1 through 5 is selected on a chronological
validation tail inside the training period.  The selected model is then refit
on the complete training period.  Its parameters remain fixed throughout the
test period, while the observed coefficient history supplies the lags for each
one-step-ahead forecast.
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
    compute_approximation_errors,
    compute_reference_covariance,
    compute_reference_eigendecomposition,
    construct_reference_basis_approximations,
    transform_covariances_from_reference_basis,
    transform_covariances_to_reference_basis,
)


DATA_DIRECTORY = Path("data/raw/binance")
AR_ORDERS = (1, 2, 3, 4, 5)
VALIDATION_SHARE = 0.20


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


def matrix_array_to_frame(
    values: np.ndarray,
    timestamps: pd.Index,
    assets: pd.Index,
) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                matrix,
                index=assets,
                columns=assets,
            )
            for matrix in values
        ],
        keys=timestamps,
        names=["timestamp", "asset"],
    )


def extract_identity_coefficients(
    errors: pd.DataFrame,
) -> pd.DataFrame:
    assets = errors.columns
    error_values, timestamps = matrix_frame_to_array(
        errors,
        assets,
    )
    positions = np.arange(len(assets))

    return pd.DataFrame(
        error_values[:, positions, positions],
        index=pd.Index(
            timestamps,
            name="timestamp",
        ),
        columns=assets,
    )


def construct_one_indicator_model_and_identity_coefficients(
    covariances: pd.DataFrame,
    reference_estimation_covariances: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_covariance = compute_reference_covariance(
        reference_estimation_covariances
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
    reference_basis_approximations = (
        construct_reference_basis_approximations(
            transformed_covariances,
            reference_eigenvalues,
        )
    )
    one_indicator_covariances = (
        transform_covariances_from_reference_basis(
            reference_basis_approximations,
            reference_basis,
        )
    )
    residuals = compute_approximation_errors(
        covariances,
        one_indicator_covariances,
    )
    identity_coefficients = (
        extract_identity_coefficients(
            residuals
        )
    )

    return (
        one_indicator_covariances,
        identity_coefficients,
    )


def diagonal_coefficients_to_matrix_frame(
    coefficients: pd.DataFrame,
    assets: pd.Index,
) -> pd.DataFrame:
    coefficients = coefficients.loc[:, assets]
    dimension = len(assets)
    matrices = np.zeros(
        (
            len(coefficients),
            dimension,
            dimension,
        ),
        dtype=float,
    )
    positions = np.arange(dimension)
    matrices[:, positions, positions] = (
        coefficients.to_numpy()
    )

    return matrix_array_to_frame(
        matrices,
        coefficients.index,
        assets,
    )


def fit_ar_ols(
    values: np.ndarray,
    order: int,
) -> np.ndarray:
    """Fit an AR model with intercept by ordinary least squares."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("AR input must be one-dimensional.")

    if len(values) <= order:
        raise ValueError(
            "The estimation sample must contain more "
            "observations than the AR order."
        )

    lagged_values = np.column_stack(
        [
            values[
                order - lag:
                len(values) - lag
            ]
            for lag in range(1, order + 1)
        ]
    )
    design = np.column_stack(
        [
            np.ones(len(values) - order),
            lagged_values,
        ]
    )
    targets = values[order:]

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        targets,
        rcond=None,
    )
    return coefficients


def forecast_ar_at_positions(
    observed_values: np.ndarray,
    coefficients: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Produce one-step forecasts from observed lagged values."""
    order = len(coefficients) - 1
    forecasts = []

    for position in positions:
        if position < order:
            raise ValueError(
                "Every forecast position must have all required lags."
            )

        lags = np.array(
            [
                observed_values[position - lag]
                for lag in range(1, order + 1)
            ]
        )
        forecasts.append(
            coefficients[0]
            + coefficients[1:] @ lags
        )

    return np.asarray(forecasts)


def maximum_companion_eigenvalue_modulus(
    coefficients: np.ndarray,
) -> float:
    autoregressive_coefficients = coefficients[1:]
    order = len(autoregressive_coefficients)

    companion = np.zeros((order, order))
    companion[0, :] = autoregressive_coefficients

    if order > 1:
        companion[1:, :-1] = np.eye(order - 1)

    return float(
        np.max(
            np.abs(
                np.linalg.eigvals(companion)
            )
        )
    )


def calculate_scalar_forecast_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    errors = forecast - actual
    squared_error = float(np.sum(errors**2))

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
        "squared_error": squared_error,
    }


def build_validation_comparison_and_select_orders(
    training_coefficients: pd.DataFrame,
    validation_start: int,
) -> tuple[pd.DataFrame, pd.Series]:
    validation_positions = np.arange(
        validation_start,
        len(training_coefficients),
    )
    rows = []
    selected_orders = {}

    for asset in training_coefficients.columns:
        values = training_coefficients[
            asset
        ].to_numpy()
        actual = values[validation_positions]

        forecasts = {
            "zero": np.zeros(len(validation_positions)),
            "naive": values[validation_positions - 1],
        }
        ar_fit_information = {}

        for order in AR_ORDERS:
            coefficients = fit_ar_ols(
                values[:validation_start],
                order,
            )
            method = f"ar_{order}"
            forecasts[method] = forecast_ar_at_positions(
                values,
                coefficients,
                validation_positions,
            )
            maximum_modulus = (
                maximum_companion_eigenvalue_modulus(
                    coefficients
                )
            )
            ar_fit_information[method] = {
                "maximum_companion_eigenvalue_modulus": (
                    maximum_modulus
                ),
                "stationary": maximum_modulus < 1.0,
            }

        metrics_by_method = {
            method: calculate_scalar_forecast_metrics(
                actual,
                forecast,
            )
            for method, forecast in forecasts.items()
        }
        zero_rmse = metrics_by_method["zero"]["rmse"]

        selected_method = min(
            (f"ar_{order}" for order in AR_ORDERS),
            key=lambda method: (
                metrics_by_method[method]["rmse"],
                int(method.split("_")[1]),
            ),
        )
        selected_order = int(
            selected_method.split("_")[1]
        )
        selected_orders[asset] = selected_order

        for method, metrics in metrics_by_method.items():
            row = {
                "asset": asset,
                "method": method,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "bias": metrics["bias"],
                "forecast_actual_correlation": (
                    metrics[
                        "forecast_actual_correlation"
                    ]
                ),
                "rmse_ratio_to_zero": (
                    metrics["rmse"] / zero_rmse
                ),
                "rmse_change_percent": (
                    100.0
                    * (
                        metrics["rmse"]
                        / zero_rmse
                        - 1.0
                    )
                ),
                "selected_ar_order": (
                    method == selected_method
                ),
                (
                    "maximum_companion_"
                    "eigenvalue_modulus"
                ): np.nan,
                "stationary": pd.NA,
            }

            if method in ar_fit_information:
                row.update(ar_fit_information[method])

            rows.append(row)

    comparison = (
        pd.DataFrame(rows)
        .set_index(["asset", "method"])
    )
    selected_order_series = pd.Series(
        selected_orders,
        name="selected_ar_order",
        dtype=int,
    )
    selected_order_series.index.name = "asset"

    return comparison, selected_order_series


def fit_selected_models_and_forecast_test(
    coefficients: pd.DataFrame,
    training_count: int,
    selected_orders: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_positions = np.arange(
        training_count,
        len(coefficients),
    )
    test_timestamps = coefficients.index[test_positions]
    ar_forecasts = {}
    model_rows = []

    for asset in coefficients.columns:
        values = coefficients[asset].to_numpy()
        order = int(selected_orders.loc[asset])
        fitted_coefficients = fit_ar_ols(
            values[:training_count],
            order,
        )
        ar_forecasts[asset] = forecast_ar_at_positions(
            values,
            fitted_coefficients,
            test_positions,
        )

        maximum_modulus = (
            maximum_companion_eigenvalue_modulus(
                fitted_coefficients
            )
        )
        row = {
            "asset": asset,
            "selected_ar_order": order,
            "intercept": fitted_coefficients[0],
            "maximum_companion_eigenvalue_modulus": (
                maximum_modulus
            ),
            "stationary": maximum_modulus < 1.0,
        }

        for lag in AR_ORDERS:
            row[f"phi_{lag}"] = (
                fitted_coefficients[lag]
                if lag <= order
                else np.nan
            )

        model_rows.append(row)

    forecast_frame = pd.DataFrame(
        ar_forecasts,
        index=pd.Index(
            test_timestamps,
            name="timestamp",
        ),
    )
    model_information = (
        pd.DataFrame(model_rows)
        .set_index("asset")
    )

    return forecast_frame, model_information


def build_test_coefficient_forecasts(
    coefficients: pd.DataFrame,
    training_count: int,
    ar_forecasts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    test_coefficients = coefficients.iloc[
        training_count:
    ].copy()
    naive_forecasts = coefficients.shift(1).iloc[
        training_count:
    ].copy()

    return {
        "zero_correction": pd.DataFrame(
            0.0,
            index=test_coefficients.index,
            columns=test_coefficients.columns,
        ),
        "naive_correction": naive_forecasts,
        "ar_correction": ar_forecasts,
        "oracle_correction": test_coefficients,
    }


def build_test_coefficient_comparison(
    actual_coefficients: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    components = ["all_assets", *actual_coefficients.columns]
    metrics = {}

    for method, forecast in forecasts.items():
        for component in components:
            if component == "all_assets":
                actual = actual_coefficients.to_numpy().reshape(-1)
                predicted = forecast.to_numpy().reshape(-1)
            else:
                actual = actual_coefficients[
                    component
                ].to_numpy()
                predicted = forecast[
                    component
                ].to_numpy()

            metrics[(method, component)] = (
                calculate_scalar_forecast_metrics(
                    actual,
                    predicted,
                )
            )

    for (method, component), method_metrics in metrics.items():
        zero_metrics = metrics[
            ("zero_correction", component)
        ]
        rmse_ratio = (
            method_metrics["rmse"]
            / zero_metrics["rmse"]
        )

        rows.append(
            {
                "method": method,
                "component": component,
                "rmse": method_metrics["rmse"],
                "mae": method_metrics["mae"],
                "bias": method_metrics["bias"],
                "forecast_actual_correlation": (
                    method_metrics[
                        "forecast_actual_correlation"
                    ]
                ),
                "rmse_ratio_to_zero": rmse_ratio,
                "rmse_change_percent": (
                    100.0 * (rmse_ratio - 1.0)
                ),
                "residual_squared_error_ratio_to_zero": (
                    method_metrics["squared_error"]
                    / zero_metrics["squared_error"]
                ),
                "residual_squared_error_share_captured": (
                    1.0
                    - (
                        method_metrics["squared_error"]
                        / zero_metrics["squared_error"]
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index(["method", "component"])
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


def build_test_covariance_comparison(
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
        forecast_values, forecast_timestamps = (
            matrix_frame_to_array(
                selected_forecast,
                assets,
            )
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

    for (method, component), method_metrics in metrics.items():
        zero_metrics = metrics[
            ("zero_correction", component)
        ]
        rmse_ratio = (
            method_metrics["rmse"]
            / zero_metrics["rmse"]
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
                "rmse_ratio_to_zero": rmse_ratio,
                "rmse_change_percent": (
                    100.0 * (rmse_ratio - 1.0)
                ),
                "squared_error_ratio_to_zero": (
                    method_metrics["squared_error"]
                    / zero_metrics["squared_error"]
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index(["method", "component"])
    )


def verify_forecast_invariants(
    coefficients: pd.DataFrame,
    training_count: int,
    coefficient_forecasts: dict[str, pd.DataFrame],
    one_indicator_covariances: pd.DataFrame,
    actual_test_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> None:
    test_coefficients = coefficients.iloc[
        training_count:
    ]

    np.testing.assert_array_equal(
        coefficient_forecasts[
            "zero_correction"
        ].to_numpy(),
        np.zeros_like(test_coefficients.to_numpy()),
    )
    np.testing.assert_array_equal(
        coefficient_forecasts[
            "naive_correction"
        ].iloc[0].to_numpy(),
        coefficients.iloc[
            training_count - 1
        ].to_numpy(),
    )
    np.testing.assert_array_equal(
        coefficient_forecasts[
            "oracle_correction"
        ].to_numpy(),
        test_coefficients.to_numpy(),
    )

    test_timestamps = test_coefficients.index
    selected_one_indicator = select_matrix_timestamps(
        one_indicator_covariances,
        test_timestamps,
    )
    np.testing.assert_array_equal(
        covariance_forecasts[
            "zero_correction"
        ].to_numpy(),
        selected_one_indicator.to_numpy(),
    )

    assets = actual_test_covariances.columns
    actual_values, timestamps = matrix_frame_to_array(
        actual_test_covariances,
        assets,
    )
    one_indicator_values, base_timestamps = (
        matrix_frame_to_array(
            selected_one_indicator,
            assets,
        )
    )

    if not timestamps.equals(base_timestamps):
        raise ValueError(
            "Test timestamps do not match the base approximation."
        )

    diagonal_positions = np.arange(len(assets))
    oracle_values, oracle_timestamps = (
        matrix_frame_to_array(
            covariance_forecasts[
                "oracle_correction"
            ],
            assets,
        )
    )

    if not timestamps.equals(oracle_timestamps):
        raise ValueError(
            "Test timestamps do not match the oracle approximation."
        )

    np.testing.assert_allclose(
        oracle_values[
            :,
            diagonal_positions,
            diagonal_positions,
        ],
        actual_values[
            :,
            diagonal_positions,
            diagonal_positions,
        ],
        rtol=1e-12,
        atol=1e-18,
    )

    off_diagonal = ~np.eye(
        len(assets),
        dtype=bool,
    )

    for forecast in covariance_forecasts.values():
        forecast_values, forecast_timestamps = (
            matrix_frame_to_array(
                forecast,
                assets,
            )
        )

        if not timestamps.equals(forecast_timestamps):
            raise ValueError(
                "A covariance forecast has mismatched timestamps."
            )

        np.testing.assert_array_equal(
            forecast_values[:, off_diagonal],
            one_indicator_values[:, off_diagonal],
        )

        if not np.isfinite(forecast_values).all():
            raise ValueError(
                "A covariance forecast contains non-finite values."
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

    validation_count = int(
        np.floor(
            VALIDATION_SHARE
            * training_count
        )
    )
    validation_start = (
        training_count
        - validation_count
    )

    if validation_start <= max(AR_ORDERS):
        raise ValueError(
            "The pre-validation training sample is too short."
        )

    prevalidation_timestamps = training_timestamps[
        :validation_start
    ]
    prevalidation_covariances = (
        select_matrix_timestamps(
            training_covariances,
            prevalidation_timestamps,
        )
    )
    (
        _,
        inner_identity_coefficients,
    ) = (
        construct_one_indicator_model_and_identity_coefficients(
            training_covariances,
            prevalidation_covariances,
        )
    )

    (
        validation_comparison,
        selected_orders,
    ) = build_validation_comparison_and_select_orders(
        inner_identity_coefficients,
        validation_start,
    )

    (
        one_indicator_covariances,
        identity_coefficients,
    ) = (
        construct_one_indicator_model_and_identity_coefficients(
            covariances,
            training_covariances,
        )
    )
    (
        ar_test_forecasts,
        selected_model_information,
    ) = fit_selected_models_and_forecast_test(
        identity_coefficients,
        training_count,
        selected_orders,
    )
    coefficient_forecasts = (
        build_test_coefficient_forecasts(
            identity_coefficients,
            training_count,
            ar_test_forecasts,
        )
    )

    test_one_indicator_covariances = (
        select_matrix_timestamps(
            one_indicator_covariances,
            test_timestamps,
        )
    )
    covariance_forecasts = {}

    for method, coefficient_forecast in (
        coefficient_forecasts.items()
    ):
        correction = (
            diagonal_coefficients_to_matrix_frame(
                coefficient_forecast,
                identity_coefficients.columns,
            )
        )
        covariance_forecasts[method] = (
            test_one_indicator_covariances
            + correction
        )

    verify_forecast_invariants(
        identity_coefficients,
        training_count,
        coefficient_forecasts,
        one_indicator_covariances,
        test_covariances,
        covariance_forecasts,
    )

    test_coefficient_comparison = (
        build_test_coefficient_comparison(
            identity_coefficients.iloc[
                training_count:
            ],
            coefficient_forecasts,
        )
    )
    test_covariance_comparison = (
        build_test_covariance_comparison(
            test_covariances,
            covariance_forecasts,
        )
    )

    print(
        "Experiment status: conditional residual forecast only.\n"
        "The dominant reference component at each evaluated "
        "timestamp is still observed, not forecast."
    )
    print(
        "\nSample sizes:"
        f"\nComplete covariance matrices: {len(all_timestamps)}"
        f"\nOuter training: {training_count}"
        f"\nOuter test: {len(test_timestamps)}"
        f"\nPre-validation AR estimation: {validation_start}"
        f"\nInner validation: {validation_count}"
    )
    print(
        "\nAR protocol:\n"
        "- candidate orders: 1, 2, 3, 4, 5\n"
        "- separate order selection for each asset\n"
        "- selection criterion: lowest inner-validation RMSE\n"
        "- intercept included\n"
        "- inner PCA reference estimated before validation only\n"
        "- final PCA reference re-estimated on all outer training data\n"
        "- selected parameters refit on all outer training data\n"
        "- parameters fixed throughout the outer test\n"
        "- observed past residual coefficients used as one-step lags"
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print("\nInner-validation forecast comparison:")
        print(validation_comparison)

        print("\nSelected AR orders:")
        print(selected_orders.to_string())

        print("\nSelected models fitted on all training data:")
        print(selected_model_information)

        print("\nOuter-test coefficient forecast comparison:")
        print(test_coefficient_comparison)

        print("\nOuter-test conditional covariance comparison:")
        print(test_covariance_comparison)

    print(
        "\nAll forecast invariants passed:"
        "\n- the zero correction equals the one-indicator approximation;"
        "\n- the first naive test forecast uses the last training value;"
        "\n- the oracle correction exactly recovers test diagonals;"
        "\n- every correction leaves all off-diagonal entries unchanged."
    )


if __name__ == "__main__":
    main()
