"""Evaluate ARFIMA forecasts of identity-basis residual coefficients.

The fixed PCA reference basis is estimated exclusively from the outer
training covariance matrices.  Its one-indicator approximation leaves the
asset-specific diagonal residual coefficients

    d_(t,a) = E_(t,aa).

This script fits a separate stationary ARFIMA(0, d_a, 0) model to each of
those three coefficient series.  Each fractional parameter, mean, and
innovation variance is estimated once from the complete outer training
period by the same parametric profile Whittle method used for the dominant
indicator.  Parameters remain fixed throughout the outer test, while every
one-step forecast may use all coefficient observations strictly before the
forecast timestamp.

The ARFIMA residual forecasts are compared with zero, naive, and previously
selected AR corrections.  They are then combined with both naive and ARFIMA
dominant-indicator forecasts.  Positive-semidefinite diagnostics are reported
for every complete covariance forecast.

Keep these previously generated analysis modules beside this file:

- analyze_dominant_indicator_arfima.py
- analyze_identity_residual_forecasts.py
- analyze_complete_covariance_forecasts.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_complete_covariance_forecasts as complete_analysis
import analyze_dominant_indicator_arfima as indicator_analysis
import analyze_identity_residual_forecasts as residual_analysis
from pca_covariance_forecasting.covariance import (
    compute_block_covariances,
)
from pca_covariance_forecasting.data import (
    compute_log_returns,
    load_closing_prices,
)
from pca_covariance_forecasting.pca import (
    compute_approximation_errors,
)


DATA_DIRECTORY = Path("data/raw/binance")
TRAINING_CORRELATION_LAGS = (1, 2, 5, 10, 20, 50)


def fit_residual_arfima_models_and_forecast_test(
    identity_coefficients: pd.DataFrame,
    training_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one ARFIMA(0, d, 0) model per asset and forecast the test."""
    test_positions = np.arange(
        training_count,
        len(identity_coefficients),
    )
    test_timestamps = identity_coefficients.index[
        test_positions
    ]
    forecasts = {}
    fit_rows = []

    for asset in identity_coefficients.columns:
        values = identity_coefficients[asset].to_numpy()
        fit = indicator_analysis.fit_arfima_0d0_whittle(
            values[:training_count]
        )
        forecasts[asset] = (
            indicator_analysis.forecast_arfima_0d0_one_step(
                values,
                test_positions,
                float(fit["d"]),
                float(fit["mean"]),
            )
        )
        fit_rows.append(
            {
                "asset": asset,
                **fit,
            }
        )

    forecast_frame = pd.DataFrame(
        forecasts,
        index=pd.Index(
            test_timestamps,
            name="timestamp",
        ),
    )
    fit_information = (
        pd.DataFrame(fit_rows)
        .set_index("asset")
    )

    return forecast_frame, fit_information


def build_prefix_refit_d_stability_table(
    training_covariances: pd.DataFrame,
) -> pd.DataFrame:
    """Refit the PCA representation and residual d on training prefixes."""
    training_timestamps = (
        training_covariances.index
        .get_level_values("timestamp")
        .unique()
    )
    rows = []

    for share in indicator_analysis.STABILITY_SHARES:
        observation_count = int(
            np.floor(
                share
                * len(training_timestamps)
            )
        )
        prefix_timestamps = training_timestamps[
            :observation_count
        ]
        prefix_covariances = (
            complete_analysis.select_matrix_timestamps(
                training_covariances,
                prefix_timestamps,
            )
        )
        (
            _,
            prefix_identity_coefficients,
        ) = (
            residual_analysis
            .construct_one_indicator_model_and_identity_coefficients(
                prefix_covariances,
                prefix_covariances,
            )
        )

        for asset in prefix_identity_coefficients.columns:
            fit = indicator_analysis.fit_arfima_0d0_whittle(
                prefix_identity_coefficients[
                    asset
                ].to_numpy()
            )
            rows.append(
                {
                    "asset": asset,
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
        .set_index(["asset", "training_share"])
    )


def build_training_correlation_diagnostics(
    identity_coefficients: pd.DataFrame,
    training_count: int,
    residual_arfima_fits: pd.DataFrame,
) -> pd.DataFrame:
    """Compare raw and fractionally differenced training correlations."""
    rows = []

    for asset in identity_coefficients.columns:
        training_values = (
            identity_coefficients[
                asset
            ].iloc[:training_count].to_numpy()
        )
        differenced = (
            indicator_analysis
            .fractionally_difference_observed_series(
                training_values,
                float(
                    residual_arfima_fits.loc[
                        asset,
                        "d",
                    ]
                ),
                float(
                    residual_arfima_fits.loc[
                        asset,
                        "mean",
                    ]
                ),
            )
        )

        for lag in TRAINING_CORRELATION_LAGS:
            rows.append(
                {
                    "asset": asset,
                    "lag": lag,
                    "raw_training_coefficient": (
                        indicator_analysis
                        .calculate_lag_correlation(
                            training_values,
                            lag,
                        )
                    ),
                    (
                        "fractionally_differenced_training"
                    ): (
                        indicator_analysis
                        .calculate_lag_correlation(
                            differenced,
                            lag,
                        )
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .set_index(["asset", "lag"])
    )


def build_residual_forecasts(
    identity_coefficients: pd.DataFrame,
    training_count: int,
    ar_forecasts: pd.DataFrame,
    arfima_forecasts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build all residual-correction forecasts on common timestamps."""
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
        "arfima_correction": arfima_forecasts.copy(),
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
    """Combine indicator and residual forecasts in a small factorial design."""
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
            "naive_indicator__arfima_correction",
            "naive_indicator",
            "arfima_correction",
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
            "arfima_indicator__arfima_correction",
            "arfima_indicator",
            "arfima_correction",
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


def build_incremental_effect_comparison(
    actual_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare residual methods under matched indicator forecasts."""
    metrics = complete_analysis.collect_covariance_metrics(
        actual_covariances,
        covariance_forecasts,
    )
    comparisons = (
        (
            "ARFIMA indicator effect | zero correction",
            "arfima_indicator__zero_correction",
            "naive_indicator__zero_correction",
        ),
        (
            "AR residual effect | naive indicator",
            "naive_indicator__ar_correction",
            "naive_indicator__zero_correction",
        ),
        (
            "Residual ARFIMA effect | naive indicator",
            "naive_indicator__arfima_correction",
            "naive_indicator__zero_correction",
        ),
        (
            "Residual ARFIMA vs AR | naive indicator",
            "naive_indicator__arfima_correction",
            "naive_indicator__ar_correction",
        ),
        (
            "AR residual effect | ARFIMA indicator",
            "arfima_indicator__ar_correction",
            "arfima_indicator__zero_correction",
        ),
        (
            "Residual ARFIMA effect | ARFIMA indicator",
            "arfima_indicator__arfima_correction",
            "arfima_indicator__zero_correction",
        ),
        (
            "Residual ARFIMA vs AR | ARFIMA indicator",
            "arfima_indicator__arfima_correction",
            "arfima_indicator__ar_correction",
        ),
        (
            "Complete ARFIMA model vs baseline",
            "arfima_indicator__arfima_correction",
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


def verify_residual_arfima_invariants(
    identity_coefficients: pd.DataFrame,
    training_count: int,
    residual_arfima_fits: pd.DataFrame,
    residual_forecasts: dict[str, pd.DataFrame],
    indicator_covariances: dict[str, pd.DataFrame],
    correction_matrices: dict[str, pd.DataFrame],
    covariance_forecasts: dict[str, pd.DataFrame],
) -> None:
    """Verify no-future use and exact diagonal-combination identities."""
    test_timestamps = identity_coefficients.index[
        training_count:
    ]
    assets = identity_coefficients.columns
    first_position = np.array([training_count])

    arfima_residual_forecast = residual_forecasts[
        "arfima_correction"
    ]

    if not arfima_residual_forecast.index.equals(
        test_timestamps
    ):
        raise ValueError(
            "Residual ARFIMA timestamps do not match the test."
        )

    for asset in assets:
        values = identity_coefficients[asset].to_numpy()
        fit = residual_arfima_fits.loc[asset]
        training_values = values[:training_count]

        np.testing.assert_allclose(
            float(fit["mean"]),
            float(np.mean(training_values)),
            rtol=1e-13,
            atol=1e-30,
        )

        expected_frequency_count = (
            training_count - 1
        ) // 2
        if (
            int(fit["positive_frequency_count"])
            != expected_frequency_count
        ):
            raise ValueError(
                "A residual Whittle fit used the wrong "
                "training-sample size."
            )

        if not (
            indicator_analysis.D_LOWER_BOUND
            <= float(fit["d"])
            <= indicator_analysis.D_UPPER_BOUND
        ):
            raise ValueError(
                "A residual fractional parameter is "
                "outside the search interval."
            )

        training_only_container = np.append(
            training_values,
            0.0,
        )
        first_training_only = (
            indicator_analysis
            .forecast_arfima_0d0_one_step(
                training_only_container,
                first_position,
                float(fit["d"]),
                float(fit["mean"]),
            )[0]
        )
        first_full_series = (
            indicator_analysis
            .forecast_arfima_0d0_one_step(
                values,
                first_position,
                float(fit["d"]),
                float(fit["mean"]),
            )[0]
        )
        np.testing.assert_allclose(
            first_training_only,
            first_full_series,
            rtol=1e-13,
            atol=1e-30,
        )
        np.testing.assert_allclose(
            arfima_residual_forecast[
                asset
            ].iloc[0],
            first_full_series,
            rtol=1e-13,
            atol=1e-30,
        )

    if not np.isfinite(
        arfima_residual_forecast.to_numpy()
    ).all():
        raise ValueError(
            "Residual ARFIMA forecasts contain non-finite values."
        )

    off_diagonal = ~np.eye(
        len(assets),
        dtype=bool,
    )

    for indicator_method in (
        "naive_indicator",
        "arfima_indicator",
    ):
        base_values, base_timestamps = (
            complete_analysis.matrix_frame_to_array(
                indicator_covariances[
                    indicator_method
                ],
                assets,
            )
        )
        complete_method = (
            f"{indicator_method}__arfima_correction"
        )
        complete_values, complete_timestamps = (
            complete_analysis.matrix_frame_to_array(
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
                "Residual ARFIMA covariance timestamps "
                "do not match the indicator base."
            )

        np.testing.assert_array_equal(
            complete_values[:, off_diagonal],
            base_values[:, off_diagonal],
        )
        np.testing.assert_allclose(
            covariance_forecasts[
                complete_method
            ].to_numpy(),
            (
                indicator_covariances[
                    indicator_method
                ]
                + correction_matrices[
                    "arfima_correction"
                ]
            ).to_numpy(),
            rtol=0.0,
            atol=0.0,
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
    training_covariances = (
        complete_analysis.select_matrix_timestamps(
            covariances,
            training_timestamps,
        )
    )
    test_covariances = (
        complete_analysis.select_matrix_timestamps(
            covariances,
            test_timestamps,
        )
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
        complete_analysis.select_matrix_timestamps(
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

    indicator_arfima_fit = (
        indicator_analysis.fit_arfima_0d0_whittle(
            dominant_indicator.iloc[
                :training_count
            ].to_numpy()
        )
    )
    (
        ar_test_forecasts,
        selected_ar_model_information,
    ) = (
        residual_analysis
        .fit_selected_models_and_forecast_test(
            identity_coefficients,
            training_count,
            selected_orders,
        )
    )
    (
        residual_arfima_test_forecasts,
        residual_arfima_fits,
    ) = fit_residual_arfima_models_and_forecast_test(
        identity_coefficients,
        training_count,
    )
    residual_d_stability = (
        build_prefix_refit_d_stability_table(
            training_covariances
        )
    )
    training_correlation_diagnostics = (
        build_training_correlation_diagnostics(
            identity_coefficients,
            training_count,
            residual_arfima_fits,
        )
    )

    indicator_forecasts = (
        complete_analysis.build_indicator_forecasts(
            dominant_indicator,
            training_count,
            indicator_arfima_fit,
        )
    )
    residual_forecasts = build_residual_forecasts(
        identity_coefficients,
        training_count,
        ar_test_forecasts,
        residual_arfima_test_forecasts,
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

    complete_analysis.verify_forecast_invariants(
        dominant_indicator,
        identity_coefficients,
        training_count,
        indicator_arfima_fit,
        selected_orders,
        selected_ar_model_information,
        indicator_forecasts,
        residual_forecasts,
        indicator_covariances,
        correction_matrices,
        covariance_forecasts,
        test_covariances,
        oracle_one_indicator_covariances,
    )
    verify_residual_arfima_invariants(
        identity_coefficients,
        training_count,
        residual_arfima_fits,
        residual_forecasts,
        indicator_covariances,
        correction_matrices,
        covariance_forecasts,
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
        complete_analysis
        .build_covariance_forecast_comparison(
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
        complete_analysis
        .build_positive_semidefinite_diagnostics(
            covariance_forecasts
        )
    )

    print(
        "Experiment status: residual ARFIMA forecasts "
        "inside the complete covariance model."
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
        "\nResidual ARFIMA protocol:\n"
        "- one fixed PCA reference basis estimated on outer training only\n"
        "- model: separate ARFIMA(0, d_a, 0) on each residual level series\n"
        "- mean: separate outer-training sample mean for each asset\n"
        "- d estimator: separate parametric profile Whittle estimate\n"
        "- d search interval: [-0.49, 0.49]\n"
        "- no validation or test data used to estimate ARFIMA parameters\n"
        "- all ARFIMA parameters fixed throughout the outer test\n"
        "- all strictly past observed coefficients used per one-step forecast\n"
        "- full finite fractional-AR history used; no lag truncation\n"
        "- AR comparison retains training-only chronological order selection\n"
        "- baseline: naive indicator with zero residual correction\n"
        "- residual corrections change diagonal entries only"
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        360,
    ):
        print(
            "\nDominant-indicator ARFIMA fit on all "
            "outer training data:"
        )
        print(pd.Series(indicator_arfima_fit).to_string())

        print(
            "\nResidual ARFIMA fits on all outer training data:"
        )
        print(residual_arfima_fits)

        print(
            "\nTraining-prefix stability of residual d estimates "
            "(PCA representation refitted within every prefix):"
        )
        print(residual_d_stability)

        print(
            "\nTraining correlation diagnostics for residual series:"
        )
        print(training_correlation_diagnostics)

        print("\nInner-validation residual AR comparison:")
        print(validation_comparison)

        print("\nSelected residual AR orders:")
        print(selected_orders.to_string())

        print(
            "\nSelected residual AR models fitted on all "
            "outer training data:"
        )
        print(selected_ar_model_information)

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

    residual_boundary_assets = (
        residual_arfima_fits.index[
            residual_arfima_fits[
                "boundary_warning"
            ].astype(bool)
        ]
        .tolist()
    )
    if residual_boundary_assets:
        print(
            "\nWARNING: Residual Whittle estimates near a "
            "stationary parameter boundary for: "
            + ", ".join(residual_boundary_assets)
        )

    print(
        "\nAll forecast invariants passed:"
        "\n- the final PCA reference basis uses outer training only;"
        "\n- every residual ARFIMA fit uses exactly the outer training;"
        "\n- the first residual ARFIMA forecasts are independent "
        "of future values;"
        "\n- the AR comparison retains leakage-free inner validation;"
        "\n- every residual correction changes diagonal entries only;"
        "\n- ARFIMA residual corrections leave off-diagonals unchanged;"
        "\n- all evaluated scalar and covariance forecasts are finite."
    )


if __name__ == "__main__":
    main()
