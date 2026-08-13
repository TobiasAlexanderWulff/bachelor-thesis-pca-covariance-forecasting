"""Compare PCA forecasts with a direct naive covariance forecast.

This final comparison adds the natural matrix-level one-step baseline

    Sigma_hat_t = Sigma_(t-1)

to the previously evaluated PCA forecasting models.  The first test forecast
therefore equals the final observed training covariance matrix.

Forecast accuracy is compared using the timestamp-level squared Frobenius
loss.  The pre-specified primary comparison is

    ARFIMA indicator + zero residual correction
    versus
    direct naive covariance forecast.

Inference is based on the loss differential

    d_t = L_t(benchmark) - L_t(candidate),

so a positive mean indicates that the candidate has lower loss.  A
Diebold-Mariano-style equal-predictive-accuracy test is reported with a
Newey-West/Bartlett HAC estimate of the long-run variance.  A circular
moving-block bootstrap confidence interval is added as a dependence-aware
robustness diagnostic.  Secondary comparisons are clearly labelled and must
not be mistaken for independently pre-specified primary hypotheses.

Keep these previously generated analysis modules beside this file:

- analyze_dominant_indicator_arfima.py
- analyze_identity_residual_forecasts.py
- analyze_complete_covariance_forecasts.py
- analyze_identity_residual_arfima_forecasts.py
"""

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_complete_covariance_forecasts as complete_analysis
import analyze_dominant_indicator_arfima as indicator_analysis
import analyze_identity_residual_arfima_forecasts as residual_arfima_analysis
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
DIRECT_NAIVE_METHOD = "direct_naive_covariance"
PRIMARY_CANDIDATE = "arfima_indicator__zero_correction"
PRIMARY_BENCHMARK = DIRECT_NAIVE_METHOD
PRIMARY_COMPONENT = "overall"
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 20260729
HAC_SENSITIVITY_LAGS = (0, 5, 10, 20, 50)
BLOCK_SENSITIVITY_LENGTHS = (5, 10, 20, 50)
COMPONENTS = ("overall", "diagonal", "off-diagonal")


def construct_direct_naive_covariance_forecast(
    covariances: pd.DataFrame,
    training_count: int,
) -> pd.DataFrame:
    """Use the immediately preceding observed covariance at every test origin."""
    assets = covariances.columns
    covariance_values, timestamps = (
        complete_analysis.matrix_frame_to_array(
            covariances,
            assets,
        )
    )

    if not 1 <= training_count < len(timestamps):
        raise ValueError(
            "training_count must leave at least one training and test matrix."
        )

    forecast_values = covariance_values[
        training_count - 1 : -1
    ].copy()
    test_timestamps = timestamps[training_count:]

    if len(forecast_values) != len(test_timestamps):
        raise ValueError(
            "Direct naive forecasts and test timestamps have different lengths."
        )

    return residual_analysis.matrix_array_to_frame(
        forecast_values,
        test_timestamps,
        assets,
    )


def construct_selected_pca_forecasts(
    covariances: pd.DataFrame,
    training_covariances: pd.DataFrame,
    training_count: int,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, float | bool | int],
    pd.DataFrame,
]:
    """Fit the previously selected PCA models without using test parameters."""
    (
        reference_basis,
        reference_eigenvalues,
        dominant_indicator,
        oracle_one_indicator_covariances,
    ) = indicator_analysis.fit_reference_representation(
        covariances,
        training_covariances,
    )
    approximation_errors = compute_approximation_errors(
        covariances,
        oracle_one_indicator_covariances,
    )
    identity_coefficients = (
        residual_analysis.extract_identity_coefficients(
            approximation_errors
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
        residual_arfima_forecasts,
        residual_arfima_fits,
    ) = (
        residual_arfima_analysis
        .fit_residual_arfima_models_and_forecast_test(
            identity_coefficients,
            training_count,
        )
    )

    indicator_forecasts = (
        complete_analysis.build_indicator_forecasts(
            dominant_indicator,
            training_count,
            indicator_arfima_fit,
        )
    )
    test_timestamps = dominant_indicator.index[
        training_count:
    ]
    zero_residual_forecasts = pd.DataFrame(
        0.0,
        index=test_timestamps,
        columns=identity_coefficients.columns,
    )

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
        if method in {
            "naive_indicator",
            "arfima_indicator",
        }
    }
    residual_arfima_corrections = (
        residual_analysis
        .diagonal_coefficients_to_matrix_frame(
            residual_arfima_forecasts,
            reference_basis.index,
        )
    )
    zero_corrections = (
        residual_analysis
        .diagonal_coefficients_to_matrix_frame(
            zero_residual_forecasts,
            reference_basis.index,
        )
    )

    covariance_forecasts = {
        "naive_indicator__zero_correction": (
            indicator_covariances["naive_indicator"]
            + zero_corrections
        ),
        "arfima_indicator__zero_correction": (
            indicator_covariances["arfima_indicator"]
            + zero_corrections
        ),
        "arfima_indicator__arfima_correction": (
            indicator_covariances["arfima_indicator"]
            + residual_arfima_corrections
        ),
    }

    return (
        covariance_forecasts,
        indicator_arfima_fit,
        residual_arfima_fits,
    )


def calculate_timestamp_losses(
    actual_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
    component: str,
) -> pd.DataFrame:
    """Return one squared matrix-component loss per test timestamp."""
    assets = actual_covariances.columns
    actual_values, actual_timestamps = (
        complete_analysis.matrix_frame_to_array(
            actual_covariances,
            assets,
        )
    )
    dimension = len(assets)

    if component not in COMPONENTS:
        raise ValueError(
            f"Unknown matrix component: {component}"
        )

    losses = {}

    for method, forecast in covariance_forecasts.items():
        selected_forecast = (
            complete_analysis.select_matrix_timestamps(
                forecast,
                actual_timestamps,
            )
        )
        forecast_values, forecast_timestamps = (
            complete_analysis.matrix_frame_to_array(
                selected_forecast,
                assets,
            )
        )

        if not actual_timestamps.equals(
            forecast_timestamps
        ):
            raise ValueError(
                f"Timestamps do not match for {method}."
            )

        squared_errors = np.square(
            actual_values - forecast_values
        )

        if component == "overall":
            method_losses = squared_errors.sum(
                axis=(1, 2)
            )
        elif component == "diagonal":
            positions = np.arange(dimension)
            method_losses = squared_errors[
                :,
                positions,
                positions,
            ].sum(axis=1)
        else:
            off_diagonal = ~np.eye(
                dimension,
                dtype=bool,
            )
            method_losses = squared_errors[
                :,
                off_diagonal,
            ].sum(axis=1)

        losses[method] = method_losses

    return pd.DataFrame(
        losses,
        index=pd.Index(
            actual_timestamps,
            name="timestamp",
        ),
    )


def build_covariance_comparison_to_direct_naive(
    actual_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize aggregate errors relative to the direct naive matrix baseline."""
    metrics = complete_analysis.collect_covariance_metrics(
        actual_covariances,
        covariance_forecasts,
    )
    rows = []

    for method in covariance_forecasts:
        for component in COMPONENTS:
            method_metrics = metrics[
                (method, component)
            ]
            baseline_metrics = metrics[
                (DIRECT_NAIVE_METHOD, component)
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
                    "rmse_ratio_to_direct_naive": rmse_ratio,
                    "rmse_change_percent": (
                        100.0
                        * (rmse_ratio - 1.0)
                    ),
                    (
                        "squared_error_ratio_to_direct_naive"
                    ): squared_error_ratio,
                    (
                        "squared_error_share_captured"
                    ): 1.0 - squared_error_ratio,
                }
            )

    return (
        pd.DataFrame(rows)
        .set_index(["method", "component"])
    )


def automatic_hac_lag(observation_count: int) -> int:
    """Return a conventional sample-size-based HAC truncation lag."""
    if observation_count < 2:
        raise ValueError(
            "At least two observations are required."
        )

    lag = int(
        np.floor(
            4.0
            * (observation_count / 100.0) ** (2.0 / 9.0)
        )
    )
    return min(
        max(lag, 0),
        observation_count - 1,
    )


def calculate_lag_correlation(
    values: np.ndarray,
    lag: int,
) -> float:
    """Calculate a finite-sample lag correlation for diagnostics."""
    values = np.asarray(values, dtype=float)

    if not 0 < lag < len(values):
        return np.nan

    earlier = values[:-lag]
    later = values[lag:]

    if (
        np.std(earlier) == 0.0
        or np.std(later) == 0.0
    ):
        return np.nan

    return float(
        np.corrcoef(earlier, later)[0, 1]
    )


def hac_equal_accuracy_test(
    loss_differences: np.ndarray,
    max_lag: int,
) -> dict[str, float | int | bool]:
    """Test whether the mean loss differential equals zero.

    Bartlett weights yield a Newey-West HAC estimate of the long-run
    variance.  The reported p-value uses the asymptotic standard-normal
    distribution and is two-sided.
    """
    values = np.asarray(
        loss_differences,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "Loss differences must be one-dimensional."
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "Loss differences contain non-finite values."
        )

    observation_count = len(values)

    if observation_count < 2:
        raise ValueError(
            "At least two loss differences are required."
        )
    if not 0 <= max_lag < observation_count:
        raise ValueError(
            "max_lag must be between zero and n - 1."
        )

    mean_difference = float(np.mean(values))
    centered = values - mean_difference
    gamma_zero = float(
        np.dot(centered, centered)
        / observation_count
    )
    long_run_variance = gamma_zero

    for lag in range(1, max_lag + 1):
        autocovariance = float(
            np.dot(
                centered[lag:],
                centered[:-lag],
            )
            / observation_count
        )
        bartlett_weight = (
            1.0
            - lag / (max_lag + 1.0)
        )
        long_run_variance += (
            2.0
            * bartlett_weight
            * autocovariance
        )

    numerical_scale = max(
        gamma_zero,
        np.finfo(float).tiny,
    )
    numerical_tolerance = (
        1e-12 * numerical_scale
    )

    if (
        long_run_variance < 0.0
        and abs(long_run_variance)
        <= numerical_tolerance
    ):
        long_run_variance = 0.0
    if long_run_variance < 0.0:
        raise ValueError(
            "The estimated HAC long-run variance is negative."
        )

    if long_run_variance == 0.0:
        identical_losses = bool(
            np.all(values == values[0])
            and values[0] == 0.0
        )
        return {
            "observation_count": observation_count,
            "hac_max_lag": max_lag,
            "mean_loss_difference": mean_difference,
            "long_run_variance": long_run_variance,
            "standard_error_of_mean": 0.0,
            "test_statistic": np.nan,
            "p_value_two_sided": np.nan,
            "identical_losses": identical_losses,
        }

    standard_error = sqrt(
        long_run_variance
        / observation_count
    )
    test_statistic = (
        mean_difference / standard_error
    )
    p_value = erfc(
        abs(test_statistic) / sqrt(2.0)
    )

    return {
        "observation_count": observation_count,
        "hac_max_lag": max_lag,
        "mean_loss_difference": mean_difference,
        "long_run_variance": long_run_variance,
        "standard_error_of_mean": standard_error,
        "test_statistic": test_statistic,
        "p_value_two_sided": p_value,
        "identical_losses": False,
    }


def circular_block_bootstrap_mean_interval(
    values: np.ndarray,
    block_length: int,
    replication_count: int,
    random_seed: int,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap the mean while retaining dependence within circular blocks."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "Bootstrap values must be one-dimensional."
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "Bootstrap values contain non-finite observations."
        )
    if not 1 <= block_length <= len(values):
        raise ValueError(
            "block_length must be between one and n."
        )
    if replication_count < 1:
        raise ValueError(
            "replication_count must be positive."
        )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie between zero and one."
        )

    observation_count = len(values)
    extended = np.concatenate(
        (
            values,
            values[: block_length - 1],
        )
    )
    block_sums = np.convolve(
        extended,
        np.ones(block_length),
        mode="valid",
    )[:observation_count]
    block_means = (
        block_sums / block_length
    )
    blocks_per_replication = int(
        np.ceil(
            observation_count / block_length
        )
    )
    random_generator = np.random.default_rng(
        random_seed
    )
    sampled_starts = random_generator.integers(
        0,
        observation_count,
        size=(
            replication_count,
            blocks_per_replication,
        ),
    )
    bootstrap_means = block_means[
        sampled_starts
    ].mean(axis=1)
    tail_probability = (
        (1.0 - confidence_level) / 2.0
    )
    lower, upper = np.quantile(
        bootstrap_means,
        [
            tail_probability,
            1.0 - tail_probability,
        ],
    )

    return {
        "block_length": block_length,
        "bootstrap_replications": replication_count,
        "confidence_level": confidence_level,
        "bootstrap_mean": float(
            np.mean(bootstrap_means)
        ),
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
    }


def build_loss_difference_comparison(
    actual_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Evaluate the primary and explicitly secondary loss comparisons."""
    comparisons = (
        (
            "PRIMARY | ARFIMA indicator vs direct naive matrix",
            PRIMARY_CANDIDATE,
            PRIMARY_BENCHMARK,
            True,
        ),
        (
            "SECONDARY | naive indicator vs direct naive matrix",
            "naive_indicator__zero_correction",
            DIRECT_NAIVE_METHOD,
            False,
        ),
        (
            "SECONDARY | ARFIMA vs naive indicator",
            "arfima_indicator__zero_correction",
            "naive_indicator__zero_correction",
            False,
        ),
        (
            "SECONDARY | residual ARFIMA incremental effect",
            "arfima_indicator__arfima_correction",
            "arfima_indicator__zero_correction",
            False,
        ),
        (
            "SECONDARY | complete ARFIMA vs direct naive matrix",
            "arfima_indicator__arfima_correction",
            DIRECT_NAIVE_METHOD,
            False,
        ),
    )
    loss_frames = {
        component: calculate_timestamp_losses(
            actual_covariances,
            covariance_forecasts,
            component,
        )
        for component in COMPONENTS
    }
    rows = []

    for comparison_index, (
        comparison,
        candidate_method,
        benchmark_method,
        is_primary,
    ) in enumerate(comparisons):
        for component_index, component in enumerate(
            COMPONENTS
        ):
            losses = loss_frames[component]
            candidate_losses = losses[
                candidate_method
            ].to_numpy()
            benchmark_losses = losses[
                benchmark_method
            ].to_numpy()
            loss_differences = (
                benchmark_losses
                - candidate_losses
            )
            hac_lag = automatic_hac_lag(
                len(loss_differences)
            )
            hac_result = hac_equal_accuracy_test(
                loss_differences,
                hac_lag,
            )
            block_length = int(
                np.ceil(
                    len(loss_differences)
                    ** (1.0 / 3.0)
                )
            )
            bootstrap_result = (
                circular_block_bootstrap_mean_interval(
                    loss_differences,
                    block_length,
                    BOOTSTRAP_REPLICATIONS,
                    (
                        BOOTSTRAP_SEED
                        + comparison_index * 100
                        + component_index
                    ),
                )
            )
            benchmark_mean_loss = float(
                np.mean(benchmark_losses)
            )

            if benchmark_mean_loss == 0.0:
                raise ValueError(
                    "A benchmark has zero mean loss."
                )

            rows.append(
                {
                    "comparison": comparison,
                    "component": component,
                    "is_primary": is_primary,
                    "candidate_method": candidate_method,
                    "benchmark_method": benchmark_method,
                    "candidate_mean_loss": float(
                        np.mean(candidate_losses)
                    ),
                    "benchmark_mean_loss": benchmark_mean_loss,
                    "mean_loss_difference": (
                        hac_result[
                            "mean_loss_difference"
                        ]
                    ),
                    "relative_mean_loss_reduction": (
                        float(
                            hac_result[
                                "mean_loss_difference"
                            ]
                        )
                        / benchmark_mean_loss
                    ),
                    "candidate_win_share": float(
                        np.mean(
                            candidate_losses
                            < benchmark_losses
                        )
                    ),
                    "loss_difference_lag_1_correlation": (
                        calculate_lag_correlation(
                            loss_differences,
                            1,
                        )
                    ),
                    "hac_max_lag": (
                        hac_result["hac_max_lag"]
                    ),
                    "hac_standard_error": (
                        hac_result[
                            "standard_error_of_mean"
                        ]
                    ),
                    "hac_test_statistic": (
                        hac_result["test_statistic"]
                    ),
                    "hac_p_value_two_sided": (
                        hac_result[
                            "p_value_two_sided"
                        ]
                    ),
                    "identical_losses": (
                        hac_result["identical_losses"]
                    ),
                    "bootstrap_block_length": (
                        bootstrap_result[
                            "block_length"
                        ]
                    ),
                    "bootstrap_ci_lower": (
                        bootstrap_result[
                            "confidence_interval_lower"
                        ]
                    ),
                    "bootstrap_ci_upper": (
                        bootstrap_result[
                            "confidence_interval_upper"
                        ]
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .set_index(["comparison", "component"])
    )


def build_primary_hac_sensitivity(
    actual_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Re-estimate the primary test over several HAC truncation lags."""
    losses = calculate_timestamp_losses(
        actual_covariances,
        covariance_forecasts,
        PRIMARY_COMPONENT,
    )
    differences = (
        losses[PRIMARY_BENCHMARK]
        - losses[PRIMARY_CANDIDATE]
    ).to_numpy()
    automatic_lag = automatic_hac_lag(
        len(differences)
    )
    lags = tuple(
        dict.fromkeys(
            (automatic_lag, *HAC_SENSITIVITY_LAGS)
        )
    )
    rows = []

    for lag in lags:
        if lag >= len(differences):
            continue
        result = hac_equal_accuracy_test(
            differences,
            lag,
        )
        rows.append(
            {
                "hac_max_lag": lag,
                "automatic_choice": (
                    lag == automatic_lag
                ),
                "mean_loss_difference": (
                    result["mean_loss_difference"]
                ),
                "standard_error_of_mean": (
                    result["standard_error_of_mean"]
                ),
                "test_statistic": (
                    result["test_statistic"]
                ),
                "p_value_two_sided": (
                    result["p_value_two_sided"]
                ),
            }
        )

    return pd.DataFrame(rows).set_index(
        "hac_max_lag"
    )


def build_primary_block_bootstrap_sensitivity(
    actual_covariances: pd.DataFrame,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Re-estimate the primary confidence interval over several block lengths."""
    losses = calculate_timestamp_losses(
        actual_covariances,
        covariance_forecasts,
        PRIMARY_COMPONENT,
    )
    differences = (
        losses[PRIMARY_BENCHMARK]
        - losses[PRIMARY_CANDIDATE]
    ).to_numpy()
    automatic_length = int(
        np.ceil(
            len(differences) ** (1.0 / 3.0)
        )
    )
    block_lengths = tuple(
        dict.fromkeys(
            (
                automatic_length,
                *BLOCK_SENSITIVITY_LENGTHS,
            )
        )
    )
    rows = []

    for index, block_length in enumerate(
        block_lengths
    ):
        if block_length > len(differences):
            continue
        result = (
            circular_block_bootstrap_mean_interval(
                differences,
                block_length,
                BOOTSTRAP_REPLICATIONS,
                BOOTSTRAP_SEED + 10_000 + index,
            )
        )
        rows.append(
            {
                "block_length": block_length,
                "automatic_choice": (
                    block_length
                    == automatic_length
                ),
                "bootstrap_mean": (
                    result["bootstrap_mean"]
                ),
                "confidence_interval_lower": (
                    result[
                        "confidence_interval_lower"
                    ]
                ),
                "confidence_interval_upper": (
                    result[
                        "confidence_interval_upper"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows).set_index(
        "block_length"
    )


def verify_direct_naive_and_loss_invariants(
    covariances: pd.DataFrame,
    test_covariances: pd.DataFrame,
    training_count: int,
    covariance_forecasts: dict[str, pd.DataFrame],
) -> None:
    """Verify alignment, no-future use, finite losses, and exact loss identities."""
    assets = covariances.columns
    covariance_values, all_timestamps = (
        complete_analysis.matrix_frame_to_array(
            covariances,
            assets,
        )
    )
    test_values, test_timestamps = (
        complete_analysis.matrix_frame_to_array(
            test_covariances,
            assets,
        )
    )
    direct_values, direct_timestamps = (
        complete_analysis.matrix_frame_to_array(
            covariance_forecasts[
                DIRECT_NAIVE_METHOD
            ],
            assets,
        )
    )

    if not test_timestamps.equals(
        direct_timestamps
    ):
        raise ValueError(
            "Direct naive timestamps do not match the test."
        )

    np.testing.assert_array_equal(
        direct_values[0],
        covariance_values[
            training_count - 1
        ],
    )
    np.testing.assert_array_equal(
        direct_values,
        covariance_values[
            training_count - 1 : -1
        ],
    )

    for method, forecast in covariance_forecasts.items():
        values, timestamps = (
            complete_analysis.matrix_frame_to_array(
                forecast,
                assets,
            )
        )

        if not timestamps.equals(test_timestamps):
            raise ValueError(
                f"Forecast timestamps do not match for {method}."
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Forecast contains non-finite values for {method}."
            )

    for component in COMPONENTS:
        losses = calculate_timestamp_losses(
            test_covariances,
            covariance_forecasts,
            component,
        )

        if not losses.index.equals(
            test_timestamps
        ):
            raise ValueError(
                "Loss timestamps do not match the test."
            )
        if not np.isfinite(
            losses.to_numpy()
        ).all():
            raise ValueError(
                "Timestamp losses contain non-finite values."
            )
        if (losses.to_numpy() < 0.0).any():
            raise ValueError(
                "A squared timestamp loss is negative."
            )

        direct_errors = (
            test_values - direct_values
        )
        if component == "overall":
            expected_direct_losses = np.square(
                direct_errors
            ).sum(axis=(1, 2))
        elif component == "diagonal":
            positions = np.arange(len(assets))
            expected_direct_losses = np.square(
                direct_errors[
                    :,
                    positions,
                    positions,
                ]
            ).sum(axis=1)
        else:
            off_diagonal = ~np.eye(
                len(assets),
                dtype=bool,
            )
            expected_direct_losses = np.square(
                direct_errors[
                    :,
                    off_diagonal,
                ]
            ).sum(axis=1)

        np.testing.assert_allclose(
            losses[
                DIRECT_NAIVE_METHOD
            ].to_numpy(),
            expected_direct_losses,
            rtol=1e-14,
            atol=0.0,
        )

    if not all_timestamps[
        training_count:
    ].equals(test_timestamps):
        raise ValueError(
            "Outer split timestamps are inconsistent."
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

    (
        covariance_forecasts,
        indicator_arfima_fit,
        residual_arfima_fits,
    ) = construct_selected_pca_forecasts(
        covariances,
        training_covariances,
        training_count,
    )
    covariance_forecasts[
        DIRECT_NAIVE_METHOD
    ] = construct_direct_naive_covariance_forecast(
        covariances,
        training_count,
    )

    verify_direct_naive_and_loss_invariants(
        covariances,
        test_covariances,
        training_count,
        covariance_forecasts,
    )

    covariance_comparison = (
        build_covariance_comparison_to_direct_naive(
            test_covariances,
            covariance_forecasts,
        )
    )
    loss_difference_comparison = (
        build_loss_difference_comparison(
            test_covariances,
            covariance_forecasts,
        )
    )
    primary_hac_sensitivity = (
        build_primary_hac_sensitivity(
            test_covariances,
            covariance_forecasts,
        )
    )
    primary_block_sensitivity = (
        build_primary_block_bootstrap_sensitivity(
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
        "Experiment status: direct naive covariance baseline "
        "and loss-difference inference."
    )
    print(
        "\nSample sizes:"
        f"\nComplete covariance matrices: {len(all_timestamps)}"
        f"\nOuter training: {training_count}"
        f"\nOuter test: {len(test_timestamps)}"
    )
    print(
        "\nForecast protocol:\n"
        "- direct naive matrix forecast: Sigma_hat_t = Sigma_(t-1)\n"
        "- first direct naive test forecast: final training covariance\n"
        "- PCA reference basis and all ARFIMA parameters: outer training only\n"
        "- ARFIMA parameters fixed throughout the outer test\n"
        "- all forecasts use observations strictly before the target timestamp\n"
        "- primary model: ARFIMA indicator with zero residual correction\n"
        "- primary benchmark: direct naive covariance forecast\n"
        "- primary loss: timestamp-level squared Frobenius loss\n"
        "- positive loss differential means lower candidate loss"
    )
    print(
        "\nInference protocol:\n"
        "- primary null: equal mean squared Frobenius loss\n"
        "- HAC test: Bartlett/Newey-West long-run variance\n"
        "- automatic HAC lag: floor(4 * (n / 100)^(2 / 9))\n"
        "- p-values: asymptotic, normal, two-sided\n"
        "- supplementary circular block bootstrap: 95% percentile interval\n"
        f"- bootstrap replications: {BOOTSTRAP_REPLICATIONS}\n"
        "- secondary comparisons are exploratory and unadjusted for multiplicity"
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        420,
    ):
        print(
            "\nDominant-indicator ARFIMA fit "
            "on outer training:"
        )
        print(
            pd.Series(
                indicator_arfima_fit
            ).to_string()
        )

        print(
            "\nResidual ARFIMA fits "
            "on outer training:"
        )
        print(residual_arfima_fits)

        print(
            "\nOuter-test covariance forecast comparison "
            "relative to direct naive matrix forecast:"
        )
        print(covariance_comparison)

        print(
            "\nPrimary and secondary loss-difference inference:"
        )
        print(loss_difference_comparison)

        print(
            "\nPrimary HAC-lag sensitivity:"
        )
        print(primary_hac_sensitivity)

        print(
            "\nPrimary circular-block-bootstrap "
            "sensitivity:"
        )
        print(primary_block_sensitivity)

        print(
            "\nPositive-semidefinite diagnostics:"
        )
        print(psd_diagnostics)

    print(
        "\nInterpretation guardrails:"
        "\n- statistical significance does not establish economic importance;"
        "\n- the HAC result is asymptotic and depends on weak-stationarity "
        "conditions for the loss differential;"
        "\n- the block-bootstrap interval is a robustness diagnostic, "
        "not an assumption-free guarantee;"
        "\n- secondary p-values are exploratory and not multiplicity-adjusted;"
        "\n- a non-significant result does not prove equal forecast quality."
    )
    print(
        "\nAll forecast and loss invariants passed:"
        "\n- the direct naive forecast uses exactly the preceding covariance;"
        "\n- its first test forecast is exactly the final training covariance;"
        "\n- all method timestamps are identical;"
        "\n- every evaluated loss is finite and non-negative;"
        "\n- timestamp losses reproduce direct squared matrix errors;"
        "\n- no test observation is used as its own forecast."
    )


if __name__ == "__main__":
    main()
