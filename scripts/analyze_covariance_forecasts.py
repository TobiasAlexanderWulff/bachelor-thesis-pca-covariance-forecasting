"""Run the primary covariance-forecast holdout comparison."""

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
    automatic_circular_block_length,
    automatic_hac_lag,
    circular_block_bootstrap_mean_interval,
    compute_covariance_rmse,
    compute_covariance_squared_frobenius_losses,
    compute_hac_equal_accuracy_test,
    compute_psd_diagnostics,
)
from pca_covariance_forecasting.forecasting import (
    construct_direct_naive_covariance_forecasts,
    construct_dominant_indicator_covariance_forecasts,
    fit_and_forecast_arfima_0d0_holdout,
)
from pca_covariance_forecasting.pca import (
    compute_reference_covariance,
    compute_reference_eigendecomposition,
    transform_covariances_to_reference_basis,
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

BOOTSTRAP_REPLICATION_COUNT = 100_000
BOOTSTRAP_RANDOM_SEED = 20260729
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95


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


def compute_loss_difference_inference(
    actual_covariances: pd.DataFrame,
    benchmark_forecasts: pd.DataFrame,
    candidate_forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for method, forecasts in {
        "benchmark": benchmark_forecasts,
        "candidate": candidate_forecasts,
    }.items():
        if not forecasts.index.equals(
            actual_covariances.index
        ):
            raise ValueError(
                f"Forecast alignment failed for {method}."
            )

    benchmark_errors = (
        actual_covariances
        - benchmark_forecasts
    )
    candidate_errors = (
        actual_covariances
        - candidate_forecasts
    )

    benchmark_losses = (
        compute_covariance_squared_frobenius_losses(
            benchmark_errors
        )
    )
    candidate_losses = (
        compute_covariance_squared_frobenius_losses(
            candidate_errors
        )
    )

    if not benchmark_losses.index.equals(
        candidate_losses.index
    ):
        raise ValueError(
            "Benchmark and candidate losses are not aligned."
        )

    loss_differences = (
        benchmark_losses
        - candidate_losses
    )

    observation_count = len(loss_differences)
    hac_max_lag = automatic_hac_lag(
        observation_count
    )
    block_length = automatic_circular_block_length(
        observation_count
    )

    mean_loss_summary = pd.DataFrame(
        {
            "benchmark_mean_loss": (
                benchmark_losses.mean()
            ),
            "candidate_mean_loss": (
                candidate_losses.mean()
            ),
            "mean_loss_difference": (
                loss_differences.mean()
            ),
            "candidate_loss_reduction_percent": (
                (
                    benchmark_losses.mean()
                    - candidate_losses.mean()
                )
                .divide(benchmark_losses.mean())
                .multiply(100.0)
            ),
        }
    )
    mean_loss_summary.index.name = "loss_component"

    hac_tests = pd.DataFrame(
        {
            component: compute_hac_equal_accuracy_test(
                loss_differences[component],
                max_lag=hac_max_lag,
            )
            for component in loss_differences.columns
        }
    ).T
    hac_tests.index.name = "loss_component"

    bootstrap_intervals = pd.DataFrame(
        {
            component: (
                circular_block_bootstrap_mean_interval(
                    values=loss_differences[component],
                    block_length=block_length,
                    replication_count=(
                        BOOTSTRAP_REPLICATION_COUNT
                    ),
                    random_seed=(
                        BOOTSTRAP_RANDOM_SEED
                    ),
                    confidence_level=(
                        BOOTSTRAP_CONFIDENCE_LEVEL
                    ),
                )
            )
            for component in loss_differences.columns
        }
    ).T
    bootstrap_intervals.index.name = "loss_component"

    return (
        mean_loss_summary,
        hac_tests,
        bootstrap_intervals,
    )


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

    (
        mean_loss_summary,
        hac_tests,
        bootstrap_intervals,
    ) = compute_loss_difference_inference(
        actual_covariances=test_covariances,
        benchmark_forecasts=direct_naive_forecasts,
        candidate_forecasts=(
            arfima_indicator_covariance_forecasts
        ),
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

    print("\nLoss-difference inference:")
    print(
        "  Benchmark: Direct naive covariance"
    )
    print(
        "  Candidate: ARFIMA dominant indicator"
    )
    print(
        "  Convention: benchmark loss - candidate loss"
    )
    print(
        "  Positive differences favor the candidate."
    )
    print(
        "  Primary loss: overall squared "
        "Frobenius loss."
    )
    print(
        "  Diagonal and off-diagonal results "
        "are supplementary."
    )

    print("\nMean loss comparison:")
    print(
        mean_loss_summary.to_string(
            float_format=lambda value: f"{value:.12e}"
        )
    )

    print("\nBartlett-HAC equal-accuracy tests:")
    print(
        hac_tests[
            [
                "observation_count",
                "hac_max_lag",
                "mean_loss_difference",
                "long_run_variance",
                "standard_error_of_mean",
                "test_statistic",
                "p_value_two_sided",
            ]
        ].to_string(
            float_format=lambda value: f"{value:.12e}"
        )
    )

    print("\nCircular block-bootstrap intervals:")
    print(
        bootstrap_intervals[
            [
                "block_length",
                "bootstrap_replications",
                "confidence_level",
                "bootstrap_mean",
                "confidence_interval_lower",
                "confidence_interval_upper",
            ]
        ].to_string(
            float_format=lambda value: f"{value:.12e}"
        )
    )


if __name__ == "__main__":
    main()
