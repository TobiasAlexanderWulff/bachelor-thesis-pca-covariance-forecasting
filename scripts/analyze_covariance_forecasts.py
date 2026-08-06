"""Run the retained reduced-scope covariance-forecast analysis."""

from pathlib import Path

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
    compute_covariance_rmse,
    compute_psd_diagnostics,
    compute_relative_frobenius_errors,
)
from pca_covariance_forecasting.forecasting import (
    construct_direct_naive_covariance_forecasts,
    construct_dominant_indicator_covariance_forecasts,
    fit_and_forecast_arfima_0d0_holdout,
)
from pca_covariance_forecasting.pca import (
    compute_approximation_errors,
    compute_reference_covariance,
    compute_reference_eigendecomposition,
    construct_reference_basis_approximations,
    transform_covariances_to_reference_basis,
    transform_covariances_from_reference_basis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "binance"
)

EXPECTED_COVARIANCE_COUNT = 4_367
TRAINING_OBSERVATION_COUNT = 2_183

DIRECT_NAIVE = "direct_naive_covariance"
NAIVE_INDICATOR = "naive_dominant_indicator"
ARFIMA_INDICATOR = "arfima_dominant_indicator"

METHOD_NAMES = {
    DIRECT_NAIVE: "Direct naive covariance",
    NAIVE_INDICATOR: "Naive dominant indicator",
    ARFIMA_INDICATOR: "ARFIMA dominant indicator",
}

def select_covariances(
    covariance_matrices: pd.DataFrame,
    timestamps: pd.Index,
) -> pd.DataFrame:
    matrix_timestamps = (
        covariance_matrices.index
        .get_level_values("timestamp")
    )

    selected = covariance_matrices.loc[
        matrix_timestamps.isin(timestamps)
    ].copy()

    selected_timestamps = (
        selected.index
        .get_level_values("timestamp")
        .unique()
    )

    if not selected_timestamps.equals(timestamps):
        raise ValueError(
            "The selected covariance timestamps do not match "
            "the requested timestamps."
        )

    return selected


def compute_reference_pca_table(
    reference_eigenvalues: pd.Series,
) -> pd.DataFrame:
    total_eigenvalue = float(reference_eigenvalues.sum())

    if total_eigenvalue <= 0.0:
        raise ValueError(
            "The total reference eigenvalue must be positive."
        )

    summary = reference_eigenvalues.to_frame(
        name="eigenvalue"
    )
    summary["variance_share_percent"] = (
        reference_eigenvalues
        .divide(total_eigenvalue)
        .multiply(100.0)
    )
    summary.index.name = "reference_component"

    return summary


def compute_approximation_summary_table(
    covariances_by_sample: dict[str, pd.DataFrame],
    errors_by_sample: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if covariances_by_sample.keys() != errors_by_sample.keys():
        raise ValueError(
            "Covariance and error samples must have identical names."
        )

    summaries = {}

    for sample, covariances in covariances_by_sample.items():
        errors = errors_by_sample[sample]
        interval_errors = compute_relative_frobenius_errors(
            errors=errors,
            covariance_matrices=covariances,
        )

        summaries[sample] = pd.Series(
            {
                "aggregated_error": (
                    compute_aggregated_relative_frobenius_error(
                        errors=errors,
                        covariance_matrices=covariances,
                    )
                ),
                "mean_interval_error": interval_errors.mean(),
                "median_interval_error": interval_errors.median(),
                "95th_percentile_interval_error": (
                    interval_errors.quantile(0.95)
                ),
                "maximum_interval_error": interval_errors.max(),
            },
            dtype=float,
        )

    summary = pd.DataFrame(summaries)
    summary.index.name = "error_summary"

    return summary


def compute_rmse_table(
    actual_covariances: pd.DataFrame,
    forecasts_by_method: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rmse_by_method = {}

    for method, forecasts in forecasts_by_method.items():
        if not forecasts.index.equals(
            actual_covariances.index
        ):
            raise ValueError(
                f"Forecast alignment failed for {method}."
            )

        forecast_errors = (
            actual_covariances
            - forecasts
        )

        if forecast_errors.isna().any().any():
            raise ValueError(
                f"Forecast errors contain missing values "
                f"for {method}."
            )

        rmse_by_method[method] = compute_covariance_rmse(
            forecast_errors
        )

    rmse = pd.DataFrame(rmse_by_method).T
    rmse.index = rmse.index.map(METHOD_NAMES)
    rmse.index.name = "method"

    direct_naive_rmse = rmse.loc[
        METHOD_NAMES[DIRECT_NAIVE]
    ]

    rmse_change_percent = (
        rmse
        .divide(direct_naive_rmse)
        .subtract(1.0)
        .multiply(100.0)
    )
    rmse_change_percent.columns = [
        f"{component}_change_percent"
        for component in rmse_change_percent.columns
    ]

    return rmse, rmse_change_percent


def compute_psd_table(
    forecasts_by_method: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    diagnostics = pd.DataFrame(
        {
            method: compute_psd_diagnostics(
                forecasts
            )
            for method, forecasts
            in forecasts_by_method.items()
        }
    ).T

    diagnostics.index = diagnostics.index.map(
        METHOD_NAMES
    )
    diagnostics.index.name = "method"

    diagnostics["raw_negative_count"] = (
        diagnostics["raw_negative_count"].astype(int)
    )
    diagnostics["non_psd_count"] = (
        diagnostics["non_psd_count"].astype(int)
    )

    return diagnostics


def main() -> None:
    closing_prices = load_closing_prices(
        DATA_DIRECTORY
    )
    log_returns = compute_log_returns(
        closing_prices
    )
    covariance_matrices = compute_block_covariances(
        log_returns
    )

    covariance_timestamps = (
        covariance_matrices.index
        .get_level_values("timestamp")
        .unique()
    )

    if len(covariance_timestamps) != (
        EXPECTED_COVARIANCE_COUNT
    ):
        raise ValueError(
            "Expected "
            f"{EXPECTED_COVARIANCE_COUNT} covariance matrices, "
            f"received {len(covariance_timestamps)}."
        )

    training_timestamps = covariance_timestamps[
        :TRAINING_OBSERVATION_COUNT
    ]
    test_timestamps = covariance_timestamps[
        TRAINING_OBSERVATION_COUNT:
    ]

    training_covariances = select_covariances(
        covariance_matrices,
        training_timestamps,
    )
    test_covariances = select_covariances(
        covariance_matrices,
        test_timestamps,
    )

    reference_covariance = compute_reference_covariance(
        training_covariances
    )
    (
        reference_basis,
        reference_eigenvalues,
    ) = compute_reference_eigendecomposition(
        reference_covariance
    )

    transformed_covariances = (
        transform_covariances_to_reference_basis(
            covariances=covariance_matrices,
            reference_basis=reference_basis,
        )
    )

    reference_basis_approximations = (
        construct_reference_basis_approximations(
            transformed_covariances=(
                transformed_covariances
            ),
            reference_eigenvalues=reference_eigenvalues,
        )
    )
    covariance_approximations = (
        transform_covariances_from_reference_basis(
            transformed_covariances=(
                reference_basis_approximations
            ),
            reference_basis=reference_basis,
        )
    )
    approximation_errors = compute_approximation_errors(
        covariances=covariance_matrices,
        approximated_covariances=covariance_approximations,
    )

    training_approximation_errors = select_covariances(
        approximation_errors,
        training_timestamps,
    )
    test_approximation_errors = select_covariances(
        approximation_errors,
        test_timestamps,
    )

    reference_pca_summary = compute_reference_pca_table(
        reference_eigenvalues
    )
    approximation_summary = (
        compute_approximation_summary_table(
            covariances_by_sample={
                "Training": training_covariances,
                "Holdout": test_covariances,
            },
            errors_by_sample={
                "Training": training_approximation_errors,
                "Holdout": test_approximation_errors,
            },
        )
    )

    first_component = reference_basis.columns[0]

    dominant_indicator = (
        transformed_covariances
        .xs(
            first_component,
            level="component",
        )
        .loc[:, first_component]
        .copy()
    )
    dominant_indicator.name = "dominant_indicator"

    naive_indicator_forecasts = (
        dominant_indicator
        .shift(1)
        .iloc[TRAINING_OBSERVATION_COUNT:]
        .copy()
    )
    naive_indicator_forecasts.name = (
        "naive_dominant_indicator_forecast"
    )

    (
        arfima_indicator_forecasts,
        fitted_arfima,
    ) = fit_and_forecast_arfima_0d0_holdout(
        series=dominant_indicator,
        training_observation_count=(
            TRAINING_OBSERVATION_COUNT
        ),
    )

    direct_naive_forecasts = (
        construct_direct_naive_covariance_forecasts(
            covariance_matrices=covariance_matrices,
            training_observation_count=(
                TRAINING_OBSERVATION_COUNT
            ),
        )
    )

    naive_indicator_covariance_forecasts = (
        construct_dominant_indicator_covariance_forecasts(
            dominant_indicator_forecasts=(
                naive_indicator_forecasts
            ),
            reference_basis=reference_basis,
            reference_eigenvalues=(
                reference_eigenvalues
            ),
        )
    )

    arfima_indicator_covariance_forecasts = (
        construct_dominant_indicator_covariance_forecasts(
            dominant_indicator_forecasts=(
                arfima_indicator_forecasts
            ),
            reference_basis=reference_basis,
            reference_eigenvalues=(
                reference_eigenvalues
            ),
        )
    )

    forecasts_by_method = {
        DIRECT_NAIVE: direct_naive_forecasts,
        NAIVE_INDICATOR: (
            naive_indicator_covariance_forecasts
        ),
        ARFIMA_INDICATOR: (
            arfima_indicator_covariance_forecasts
        ),
    }

    rmse, rmse_change_percent = (
        compute_rmse_table(
            actual_covariances=test_covariances,
            forecasts_by_method=forecasts_by_method,
        )
    )
    psd_diagnostics = compute_psd_table(
        forecasts_by_method
    )

    print("Holdout split:")
    print(
        f"  Total covariance matrices: "
        f"{len(covariance_timestamps)}"
    )
    print(
        f"  Training matrices: "
        f"{len(training_timestamps)}"
    )
    print(
        f"  Test matrices: "
        f"{len(test_timestamps)}"
    )
    print(
        f"  Final training timestamp: "
        f"{training_timestamps[-1]}"
    )
    print(
        f"  First test timestamp: "
        f"{test_timestamps[0]}"
    )

    print("\nReference PCA:")
    print(
        reference_pca_summary.to_string(
            formatters={
                "eigenvalue": (
                    lambda value: f"{value:.12e}"
                ),
                "variance_share_percent": (
                    lambda value: f"{value:.6f}"
                ),
            }
        )
    )

    print("\nSingle-indicator approximation errors:")
    print(
        approximation_summary.to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )

    print("\nARFIMA(0, d, 0) training estimate:")
    print(
        f"  mean: "
        f"{float(fitted_arfima['mean']):.12e}"
    )
    print(
        f"  d: "
        f"{float(fitted_arfima['d']):.12f}"
    )
    print(
        f"  innovation variance: "
        f"{float(
            fitted_arfima['innovation_variance']
        ):.12e}"
    )
    print(
        f"  boundary warning: "
        f"{fitted_arfima['boundary_warning']}"
    )

    print("\nTest RMSE:")
    print(
        rmse.to_string(
            float_format=lambda value: f"{value:.12e}"
        )
    )

    print(
        "\nRMSE change relative to direct naive "
        "covariance forecast:"
    )
    print(
        rmse_change_percent.to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )

    print("\nForecast PSD diagnostics:")
    print(
        psd_diagnostics.to_string(
            float_format=lambda value: f"{value:.12e}"
        )
    )


if __name__ == "__main__":
    main()
