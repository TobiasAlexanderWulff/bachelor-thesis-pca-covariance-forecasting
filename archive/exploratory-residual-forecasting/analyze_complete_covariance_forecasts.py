"""Evaluate the complete one-step covariance-forecast model.

This script combines the two previously isolated forecasting experiments:

1. the dominant reference-basis component is forecast by either a naive
   forecast or an ARFIMA(0, d, 0) model;
2. the three asset-specific diagonal residual coefficients are forecast by
   either zero correction, naive forecasts, or training-selected AR(p) models.

The six feasible combinations form a small factorial comparison.  Two oracle
variants are added as representation benchmarks:

- observed dominant indicator with no residual correction;
- observed dominant indicator with observed diagonal residual correction.

All PCA reference objects and final model parameters are estimated without
outer-test observations.  At each outer-test origin, one-step forecasts may use
observations strictly before that origin.  The test is never used for model or
AR-order selection.

Keep these previously generated analysis modules beside this file:

- analyze_dominant_indicator_arfima.py
- analyze_identity_residual_forecasts.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_dominant_indicator_arfima as indicator_analysis
import analyze_identity_residual_forecasts as residual_analysis
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
)


DATA_DIRECTORY = Path("data/raw/binance")
BASELINE_METHOD = "naive_indicator__zero_correction"


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

    if actual_squared_norm == 0.0:
        raise ValueError(
            f"The {component} actual squared norm is zero."
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


def collect_covariance_metrics(
    actual_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> dict[tuple[str, str], dict[str, float]]:
    assets = actual_covariances.columns
    actual_values, actual_timestamps = (
        matrix_frame_to_array(
            actual_covariances,
            assets,
        )
    )
    metrics = {}

    for method, forecast in covariance_forecasts.items():
        selected_forecast = select_matrix_timestamps(
            forecast,
            actual_timestamps,
        )
        forecast_values, forecast_timestamps = (
            matrix_frame_to_array(
                selected_forecast,
                assets,
            )
        )

        if not actual_timestamps.equals(
            forecast_timestamps
        ):
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
                component_metrics[
                    "aggregated_relative_error"
                ] = (
                    compute_aggregated_relative_frobenius_error(
                        (
                            actual_covariances
                            - selected_forecast
                        ),
                        actual_covariances,
                    )
                )

            metrics[(method, component)] = (
                component_metrics
            )

    return metrics


def build_covariance_forecast_comparison(
    actual_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    metrics = collect_covariance_metrics(
        actual_covariances,
        covariance_forecasts,
    )
    rows = []

    for (
        method,
        component,
    ), method_metrics in metrics.items():
        baseline_metrics = metrics[
            (BASELINE_METHOD, component)
        ]
        rmse_ratio = (
            method_metrics["rmse"]
            / baseline_metrics["rmse"]
        )
        squared_error_ratio = (
            method_metrics["squared_error"]
            / baseline_metrics["squared_error"]
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
                "rmse_ratio_to_baseline": rmse_ratio,
                "rmse_change_percent": (
                    100.0
                    * (rmse_ratio - 1.0)
                ),
                "squared_error_ratio_to_baseline": (
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


def build_incremental_effect_comparison(
    actual_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    metrics = collect_covariance_metrics(
        actual_covariances,
        covariance_forecasts,
    )
    comparisons = (
        (
            "ARFIMA effect | zero correction",
            "arfima_indicator__zero_correction",
            "naive_indicator__zero_correction",
        ),
        (
            "ARFIMA effect | naive correction",
            "arfima_indicator__naive_correction",
            "naive_indicator__naive_correction",
        ),
        (
            "ARFIMA effect | AR correction",
            "arfima_indicator__ar_correction",
            "naive_indicator__ar_correction",
        ),
        (
            "Naive residual effect | naive indicator",
            "naive_indicator__naive_correction",
            "naive_indicator__zero_correction",
        ),
        (
            "AR residual effect | naive indicator",
            "naive_indicator__ar_correction",
            "naive_indicator__zero_correction",
        ),
        (
            "Naive residual effect | ARFIMA indicator",
            "arfima_indicator__naive_correction",
            "arfima_indicator__zero_correction",
        ),
        (
            "AR residual effect | ARFIMA indicator",
            "arfima_indicator__ar_correction",
            "arfima_indicator__zero_correction",
        ),
        (
            "Complete model vs baseline",
            "arfima_indicator__ar_correction",
            "naive_indicator__zero_correction",
        ),
    )
    rows = []

    for (
        comparison,
        candidate_method,
        reference_method,
    ) in comparisons:
        for component in (
            "overall",
            "diagonal",
            "off-diagonal",
        ):
            candidate = metrics[
                (candidate_method, component)
            ]
            reference = metrics[
                (reference_method, component)
            ]
            rmse_ratio = (
                candidate["rmse"]
                / reference["rmse"]
            )
            squared_error_ratio = (
                candidate["squared_error"]
                / reference["squared_error"]
            )

            rows.append(
                {
                    "comparison": comparison,
                    "component": component,
                    "candidate_method": candidate_method,
                    "reference_method": reference_method,
                    "rmse_ratio": rmse_ratio,
                    "rmse_change_percent": (
                        100.0
                        * (rmse_ratio - 1.0)
                    ),
                    "squared_error_ratio": (
                        squared_error_ratio
                    ),
                    "squared_error_share_captured": (
                        1.0
                        - squared_error_ratio
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .set_index(["comparison", "component"])
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
        eigenvalues = np.linalg.eigvalsh(values)
        minimum_eigenvalues = eigenvalues[:, 0]
        spectral_scales = np.max(
            np.abs(eigenvalues),
            axis=1,
        )
        tolerances = (
            1e-12
            * np.maximum(
                spectral_scales,
                np.finfo(float).tiny,
            )
        )
        materially_negative = (
            minimum_eigenvalues < -tolerances
        )

        rows.append(
            {
                "method": method,
                "minimum_eigenvalue": float(
                    np.min(minimum_eigenvalues)
                ),
                "raw_negative_count": int(
                    np.sum(minimum_eigenvalues < 0.0)
                ),
                "non_psd_count": int(
                    np.sum(materially_negative)
                ),
                "non_psd_share": float(
                    np.mean(materially_negative)
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("method")
    )


def build_indicator_forecasts(
    dominant_indicator: pd.Series,
    training_count: int,
    arfima_fit: dict[str, float | bool | int],
) -> dict[str, pd.Series]:
    test_timestamps = dominant_indicator.index[
        training_count:
    ]
    test_positions = np.arange(
        training_count,
        len(dominant_indicator),
    )
    arfima_values = (
        indicator_analysis.forecast_arfima_0d0_one_step(
            dominant_indicator.to_numpy(),
            test_positions,
            float(arfima_fit["d"]),
            float(arfima_fit["mean"]),
        )
    )

    return {
        "naive_indicator": (
            dominant_indicator
            .shift(1)
            .iloc[training_count:]
            .copy()
        ),
        "arfima_indicator": pd.Series(
            arfima_values,
            index=test_timestamps,
            name="dominant_indicator",
        ),
        "oracle_indicator": (
            dominant_indicator
            .iloc[training_count:]
            .copy()
        ),
    }


def build_residual_forecasts(
    identity_coefficients: pd.DataFrame,
    training_count: int,
    ar_forecasts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    test_coefficients = identity_coefficients.iloc[
        training_count:
    ].copy()

    return {
        "zero_correction": pd.DataFrame(
            0.0,
            index=test_coefficients.index,
            columns=test_coefficients.columns,
        ),
        "naive_correction": (
            identity_coefficients
            .shift(1)
            .iloc[training_count:]
            .copy()
        ),
        "ar_correction": ar_forecasts.copy(),
        "oracle_correction": test_coefficients,
    }


def construct_complete_covariance_forecasts(
    indicator_forecasts: dict[str, pd.Series],
    residual_forecasts: dict[str, pd.DataFrame],
    reference_basis: pd.DataFrame,
    reference_eigenvalues: pd.Series,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    indicator_covariances = {
        method: (
            indicator_analysis
            .construct_covariances_from_indicator(
                forecast,
                reference_basis,
                reference_eigenvalues,
            )
        )
        for method, forecast in indicator_forecasts.items()
    }
    correction_matrices = {
        method: (
            residual_analysis
            .diagonal_coefficients_to_matrix_frame(
                forecast,
                reference_basis.index,
            )
        )
        for method, forecast in residual_forecasts.items()
    }
    method_specifications = (
        (
            "naive_indicator__zero_correction",
            "naive_indicator",
            "zero_correction",
        ),
        (
            "naive_indicator__naive_correction",
            "naive_indicator",
            "naive_correction",
        ),
        (
            "naive_indicator__ar_correction",
            "naive_indicator",
            "ar_correction",
        ),
        (
            "arfima_indicator__zero_correction",
            "arfima_indicator",
            "zero_correction",
        ),
        (
            "arfima_indicator__naive_correction",
            "arfima_indicator",
            "naive_correction",
        ),
        (
            "arfima_indicator__ar_correction",
            "arfima_indicator",
            "ar_correction",
        ),
        (
            "oracle_indicator__zero_correction",
            "oracle_indicator",
            "zero_correction",
        ),
        (
            "oracle_indicator__oracle_correction",
            "oracle_indicator",
            "oracle_correction",
        ),
    )
    covariance_forecasts = {
        complete_method: (
            indicator_covariances[indicator_method]
            + correction_matrices[residual_method]
        )
        for (
            complete_method,
            indicator_method,
            residual_method,
        ) in method_specifications
    }

    return (
        covariance_forecasts,
        indicator_covariances,
        correction_matrices,
    )


def verify_forecast_invariants(
    dominant_indicator: pd.Series,
    identity_coefficients: pd.DataFrame,
    training_count: int,
    arfima_fit: dict[str, float | bool | int],
    selected_orders: pd.Series,
    selected_model_information: pd.DataFrame,
    indicator_forecasts: dict[str, pd.Series],
    residual_forecasts: dict[str, pd.DataFrame],
    indicator_covariances: dict[str, pd.DataFrame],
    correction_matrices: dict[str, pd.DataFrame],
    covariance_forecasts: dict[str, pd.DataFrame],
    test_covariances: pd.DataFrame,
    oracle_one_indicator_covariances: pd.DataFrame,
) -> None:
    test_timestamps = dominant_indicator.index[
        training_count:
    ]
    assets = identity_coefficients.columns

    for forecast in indicator_forecasts.values():
        if not forecast.index.equals(test_timestamps):
            raise ValueError(
                "An indicator forecast has mismatched timestamps."
            )
        if not np.isfinite(forecast.to_numpy()).all():
            raise ValueError(
                "An indicator forecast contains non-finite values."
            )

    for forecast in residual_forecasts.values():
        if not forecast.index.equals(test_timestamps):
            raise ValueError(
                "A residual forecast has mismatched timestamps."
            )
        if not np.isfinite(forecast.to_numpy()).all():
            raise ValueError(
                "A residual forecast contains non-finite values."
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
        residual_forecasts[
            "naive_correction"
        ].iloc[0].to_numpy(),
        identity_coefficients.iloc[
            training_count - 1
        ].to_numpy(),
    )
    np.testing.assert_array_equal(
        residual_forecasts[
            "zero_correction"
        ].to_numpy(),
        np.zeros_like(
            residual_forecasts[
                "zero_correction"
            ].to_numpy()
        ),
    )
    np.testing.assert_array_equal(
        residual_forecasts[
            "oracle_correction"
        ].to_numpy(),
        identity_coefficients.iloc[
            training_count:
        ].to_numpy(),
    )

    training_only_indicator_container = np.append(
        dominant_indicator.iloc[
            :training_count
        ].to_numpy(),
        0.0,
    )
    first_position = np.array([training_count])
    first_arfima_training_only = (
        indicator_analysis
        .forecast_arfima_0d0_one_step(
            training_only_indicator_container,
            first_position,
            float(arfima_fit["d"]),
            float(arfima_fit["mean"]),
        )[0]
    )
    first_arfima_full_series = (
        indicator_analysis
        .forecast_arfima_0d0_one_step(
            dominant_indicator.to_numpy(),
            first_position,
            float(arfima_fit["d"]),
            float(arfima_fit["mean"]),
        )[0]
    )
    np.testing.assert_allclose(
        first_arfima_training_only,
        first_arfima_full_series,
        rtol=1e-13,
        atol=0.0,
    )

    for asset in assets:
        order = int(selected_orders.loc[asset])
        coefficients = np.array(
            [
                selected_model_information.loc[
                    asset,
                    "intercept",
                ],
                *[
                    selected_model_information.loc[
                        asset,
                        f"phi_{lag}",
                    ]
                    for lag in range(1, order + 1)
                ],
            ]
        )
        training_only_residual_container = np.append(
            identity_coefficients[
                asset
            ].iloc[:training_count].to_numpy(),
            0.0,
        )
        first_ar_training_only = (
            residual_analysis.forecast_ar_at_positions(
                training_only_residual_container,
                coefficients,
                first_position,
            )[0]
        )
        first_ar_full_series = (
            residual_analysis.forecast_ar_at_positions(
                identity_coefficients[
                    asset
                ].to_numpy(),
                coefficients,
                first_position,
            )[0]
        )
        np.testing.assert_allclose(
            first_ar_training_only,
            first_ar_full_series,
            rtol=1e-13,
            atol=0.0,
        )

    selected_oracle_one_indicator = (
        select_matrix_timestamps(
            oracle_one_indicator_covariances,
            test_timestamps,
        )
    )
    np.testing.assert_allclose(
        indicator_covariances[
            "oracle_indicator"
        ].to_numpy(),
        selected_oracle_one_indicator.to_numpy(),
        rtol=1e-13,
        atol=1e-20,
    )
    np.testing.assert_allclose(
        covariance_forecasts[
            "naive_indicator__zero_correction"
        ].to_numpy(),
        indicator_covariances[
            "naive_indicator"
        ].to_numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        covariance_forecasts[
            "arfima_indicator__zero_correction"
        ].to_numpy(),
        indicator_covariances[
            "arfima_indicator"
        ].to_numpy(),
        rtol=0.0,
        atol=0.0,
    )

    for indicator_method in (
        "naive_indicator",
        "arfima_indicator",
    ):
        base_values, base_timestamps = (
            matrix_frame_to_array(
                indicator_covariances[
                    indicator_method
                ],
                assets,
            )
        )
        off_diagonal = ~np.eye(
            len(assets),
            dtype=bool,
        )

        for residual_method in (
            "zero_correction",
            "naive_correction",
            "ar_correction",
        ):
            complete_method = (
                f"{indicator_method}__{residual_method}"
            )
            complete_values, complete_timestamps = (
                matrix_frame_to_array(
                    covariance_forecasts[
                        complete_method
                    ],
                    assets,
                )
            )

            if not base_timestamps.equals(
                complete_timestamps
            ):
                raise ValueError(
                    "A complete forecast has mismatched timestamps."
                )

            np.testing.assert_array_equal(
                complete_values[:, off_diagonal],
                base_values[:, off_diagonal],
            )

            expected_complete = (
                indicator_covariances[
                    indicator_method
                ]
                + correction_matrices[
                    residual_method
                ]
            )
            np.testing.assert_allclose(
                covariance_forecasts[
                    complete_method
                ].to_numpy(),
                expected_complete.to_numpy(),
                rtol=0.0,
                atol=0.0,
            )

    actual_values, actual_timestamps = (
        matrix_frame_to_array(
            test_covariances,
            assets,
        )
    )
    oracle_values, oracle_timestamps = (
        matrix_frame_to_array(
            covariance_forecasts[
                "oracle_indicator__oracle_correction"
            ],
            assets,
        )
    )

    if not actual_timestamps.equals(
        oracle_timestamps
    ):
        raise ValueError(
            "Oracle and actual timestamps do not match."
        )

    positions = np.arange(len(assets))
    np.testing.assert_allclose(
        oracle_values[
            :,
            positions,
            positions,
        ],
        actual_values[
            :,
            positions,
            positions,
        ],
        rtol=1e-12,
        atol=1e-18,
    )

    for forecast in covariance_forecasts.values():
        if not np.isfinite(
            forecast.to_numpy()
        ).all():
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
            residual_analysis.VALIDATION_SHARE
            * training_count
        )
    )
    validation_start = (
        training_count - validation_count
    )

    if (
        validation_start
        <= max(residual_analysis.AR_ORDERS)
    ):
        raise ValueError(
            "The pre-validation sample is too short."
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
        residual_analysis
        .construct_one_indicator_model_and_identity_coefficients(
            training_covariances,
            prevalidation_covariances,
        )
    )
    (
        validation_comparison,
        selected_orders,
    ) = (
        residual_analysis
        .build_validation_comparison_and_select_orders(
            inner_identity_coefficients,
            validation_start,
        )
    )

    (
        reference_basis,
        reference_eigenvalues,
        dominant_indicator,
        oracle_one_indicator_covariances,
    ) = indicator_analysis.fit_reference_representation(
        covariances,
        training_covariances,
    )
    residuals = compute_approximation_errors(
        covariances,
        oracle_one_indicator_covariances,
    )
    identity_coefficients = (
        residual_analysis.extract_identity_coefficients(
            residuals
        )
    )

    arfima_fit = (
        indicator_analysis.fit_arfima_0d0_whittle(
            dominant_indicator.iloc[
                :training_count
            ].to_numpy()
        )
    )
    (
        ar_test_forecasts,
        selected_model_information,
    ) = (
        residual_analysis
        .fit_selected_models_and_forecast_test(
            identity_coefficients,
            training_count,
            selected_orders,
        )
    )
    indicator_forecasts = build_indicator_forecasts(
        dominant_indicator,
        training_count,
        arfima_fit,
    )
    residual_forecasts = build_residual_forecasts(
        identity_coefficients,
        training_count,
        ar_test_forecasts,
    )
    (
        covariance_forecasts,
        indicator_covariances,
        correction_matrices,
    ) = construct_complete_covariance_forecasts(
        indicator_forecasts,
        residual_forecasts,
        reference_basis,
        reference_eigenvalues,
    )

    verify_forecast_invariants(
        dominant_indicator,
        identity_coefficients,
        training_count,
        arfima_fit,
        selected_orders,
        selected_model_information,
        indicator_forecasts,
        residual_forecasts,
        indicator_covariances,
        correction_matrices,
        covariance_forecasts,
        test_covariances,
        oracle_one_indicator_covariances,
    )

    indicator_comparison = (
        indicator_analysis
        .build_indicator_forecast_comparison(
            dominant_indicator.iloc[
                training_count:
            ],
            indicator_forecasts,
        )
    )
    residual_comparison = (
        residual_analysis
        .build_test_coefficient_comparison(
            identity_coefficients.iloc[
                training_count:
            ],
            residual_forecasts,
        )
    )
    covariance_comparison = (
        build_covariance_forecast_comparison(
            test_covariances,
            covariance_forecasts,
        )
    )
    incremental_comparison = (
        build_incremental_effect_comparison(
            test_covariances,
            covariance_forecasts,
        )
    )
    psd_diagnostics = (
        build_positive_semidefinite_diagnostics(
            covariance_forecasts
        )
    )

    print(
        "Experiment status: complete covariance forecasts.\n"
        "The dominant indicator and residual corrections are "
        "now both forecast in the feasible model variants."
    )
    print(
        "\nSample sizes:"
        f"\nComplete covariance matrices: {len(all_timestamps)}"
        f"\nOuter training: {training_count}"
        f"\nOuter test: {len(test_timestamps)}"
        f"\nPre-validation AR estimation: {validation_start}"
        f"\nInner AR validation: {validation_count}"
    )
    print(
        "\nCombined protocol:\n"
        "- one fixed PCA reference basis estimated on outer training only\n"
        "- ARFIMA(0, d, 0) parameter d estimated on outer training only\n"
        "- separate residual AR orders selected by chronological inner "
        "validation\n"
        "- final residual AR parameters refitted on outer training only\n"
        "- all fitted parameters fixed throughout the outer test\n"
        "- each one-step forecast uses only strictly past observations\n"
        "- baseline: naive indicator with zero residual correction\n"
        "- identity residual corrections change diagonal entries only\n"
        "- oracle variants are representation bounds, not feasible forecasts"
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        300,
    ):
        print("\nARFIMA fit on all outer training data:")
        print(pd.Series(arfima_fit).to_string())

        print("\nInner-validation residual comparison:")
        print(validation_comparison)

        print("\nSelected residual AR orders:")
        print(selected_orders.to_string())

        print(
            "\nSelected residual models fitted on all "
            "outer training data:"
        )
        print(selected_model_information)

        print(
            "\nOuter-test dominant-indicator "
            "forecast comparison:"
        )
        print(indicator_comparison)

        print(
            "\nOuter-test residual-coefficient "
            "forecast comparison:"
        )
        print(residual_comparison)

        print(
            "\nOuter-test complete covariance "
            "forecast comparison:"
        )
        print(covariance_comparison)

        print(
            "\nIncremental effects under matched conditions:"
        )
        print(incremental_comparison)

        print(
            "\nPositive-semidefinite diagnostics:"
        )
        print(psd_diagnostics)

    if bool(arfima_fit["boundary_warning"]):
        print(
            "\nWARNING: The Whittle estimate lies near a "
            "stationary parameter boundary."
        )

    print(
        "\nAll forecast invariants passed:"
        "\n- the final PCA reference basis uses outer training only;"
        "\n- inner AR selection uses a PCA reference fitted before "
        "the validation interval;"
        "\n- the first naive forecasts use the final training values;"
        "\n- first ARFIMA and AR forecasts are independent of future values;"
        "\n- zero residual variants exactly equal their indicator-only bases;"
        "\n- every residual correction changes diagonal entries only;"
        "\n- the complete oracle exactly recovers all covariance diagonals;"
        "\n- every evaluated covariance forecast is finite."
    )


if __name__ == "__main__":
    main()
