"""Evaluate covariance-matrix approximations and forecasts."""

from math import ceil, erfc, floor, sqrt

import numpy as np
import pandas as pd


def _matrix_frame_to_array(
    matrices: pd.DataFrame,
) -> np.ndarray:
    """Stack chronologically indexed matrix frames for vectorized metrics."""
    assets = matrices.columns
    timestamps = (
        matrices.index
        .get_level_values("timestamp")
        .unique()
    )

    return np.stack(
        [
            matrices.xs(
                timestamp,
                level="timestamp",
            ).loc[assets, assets].to_numpy()
            for timestamp in timestamps
        ]
    )


def compute_frobenius_norms(
    matrices: pd.DataFrame,
) -> pd.Series:
    """Compute ``||A_t||_F = sqrt(sum_ij A_t,ij²)`` by timestamp.

    This is the matrix norm used in the thesis approximation errors and
    timestamp-level forecast losses.
    """
    assets = matrices.columns

    norms = []
    timestamps = []

    for timestamp, matrix in matrices.groupby(
        level="timestamp",
        sort=False,
    ):
        matrix = (
            matrix
            .droplevel("timestamp")
            .loc[assets, assets]
        )

        frobenius_norm = np.linalg.norm(
            matrix.to_numpy(),
            ord="fro",
        )

        timestamps.append(timestamp)
        norms.append(frobenius_norm)

    return pd.Series(
        data=norms,
        index=pd.Index(
            timestamps,
            name="timestamp",
        ),
        name="frobenius_norm",
        dtype=float,
    )


def compute_relative_frobenius_errors(
    errors: pd.DataFrame,
    covariance_matrices: pd.DataFrame,
) -> pd.Series:
    """Compute the thesis interval-level relative approximation errors.

    Implements ``e_t^rel = ||Σ_t-Σ̃_t||_F / ||Σ_t||_F``
    (``eq:relative-frobenius-approximation-error``) for aligned timestamps.
    """
    error_norms = compute_frobenius_norms(errors)
    covariance_norms = compute_frobenius_norms(
        covariance_matrices,
    )

    if not error_norms.index.equals(covariance_norms.index):
        raise ValueError(
            "Errors and covariance matrices must have "
            "identical timestamps."
        )

    if covariance_norms.eq(0.0).any():
        raise ValueError(
            "Relative errors cannot be computed for "
            "covariance matrices with Frobenius norm zero."
        )

    relative_errors = error_norms / covariance_norms
    relative_errors.name = "relative_frobenius_error"

    return relative_errors


def compute_aggregated_relative_frobenius_error(
    errors: pd.DataFrame,
    covariance_matrices: pd.DataFrame,
) -> float:
    """Compute the thesis aggregate relative approximation error.

    Implements ``sqrt(sum_t ||Σ_t-Σ̃_t||_F² / sum_t ||Σ_t||_F²)``
    (``eq:aggregate-relative-frobenius-approximation-error``). It is not the
    arithmetic mean of the interval-level relative errors.
    """
    error_norms = compute_frobenius_norms(errors)
    covariance_norms = compute_frobenius_norms(
        covariance_matrices,
    )

    if not error_norms.index.equals(covariance_norms.index):
        raise ValueError(
            "Errors and covariance matrices must have "
            "identical timestamps."
        )

    squared_error_norm_sum = np.square(
        error_norms.to_numpy(),
    ).sum()

    squared_covariance_norm_sum = np.square(
        covariance_norms.to_numpy(),
    ).sum()

    if squared_covariance_norm_sum == 0.0:
        raise ValueError(
            "The aggregated relative error cannot be computed "
            "when all covariance matrices are zero."
        )

    return float(
        np.sqrt(
            squared_error_norm_sum
            / squared_covariance_norm_sum
        )
    )


def compute_covariance_rmse(
    errors: pd.DataFrame,
) -> pd.Series:
    """Compute the thesis overall, diagonal, and off-diagonal RMSE.

    Overall RMSE is ``sqrt(sum_t ||E_t||_F² / (|H| p²))``
    (``eq:overall-covariance-rmse``). The other two values apply the same
    aggregation separately to ``p`` diagonal and ``p(p-1)`` off-diagonal
    entries per timestamp (``eq:diagonal-rmse`` and ``eq:off-diagonal-rmse``).
    """
    error_values = _matrix_frame_to_array(errors)
    dimension = error_values.shape[1]

    diagonal_positions = np.arange(dimension)
    diagonal_errors = error_values[
        :,
        diagonal_positions,
        diagonal_positions,
    ]

    off_diagonal_mask = ~np.eye(
        dimension,
        dtype=bool,
    )
    off_diagonal_errors = error_values[
        :,
        off_diagonal_mask,
    ]

    return pd.Series(
        {
            "overall": np.sqrt(
                np.mean(np.square(error_values))
            ),
            "diagonal": np.sqrt(
                np.mean(np.square(diagonal_errors))
            ),
            "off-diagonal": np.sqrt(
                np.mean(np.square(off_diagonal_errors))
            ),
        },
        name="rmse",
        dtype=float,
    )


def compute_psd_diagnostics(
    covariance_matrices: pd.DataFrame,
    relative_tolerance: float = 1e-12,
) -> pd.Series:
    """Check the covariance-validity diagnostic used in the thesis.

    A forecast is classified as non-PSD only when its minimum eigenvalue is
    below a scale-relative numerical tolerance. Raw negative counts are kept
    separately to expose the values before that tolerance is applied.
    """
    if relative_tolerance < 0.0:
        raise ValueError(
            "relative_tolerance must be non-negative."
        )

    covariance_values = _matrix_frame_to_array(
        covariance_matrices,
    )
    eigenvalues = np.linalg.eigvalsh(covariance_values)
    minimum_eigenvalues = eigenvalues[:, 0]

    spectral_scales = np.max(
        np.abs(eigenvalues),
        axis=1,
    )
    tolerances = (
        relative_tolerance
        * np.maximum(
            spectral_scales,
            np.finfo(float).tiny,
        )
    )
    non_psd = minimum_eigenvalues < -tolerances

    return pd.Series(
        {
            "minimum_eigenvalue": np.min(
                minimum_eigenvalues
            ),
            "raw_negative_count": np.sum(
                minimum_eigenvalues < 0.0
            ),
            "non_psd_count": np.sum(non_psd),
            "non_psd_share": np.mean(non_psd),
        },
        name="psd_diagnostics",
    )


def compute_covariance_squared_frobenius_losses(
    errors: pd.DataFrame,
) -> pd.DataFrame:
    """Compute timestamp-level squared Frobenius forecast losses.

    For caller-supplied ``E_t = Σ_t-Σ_hat_t`` (``eq:forecast-error-matrix``),
    this implements ``ell_t = ||E_t||_F² = sum_ij E_t,ij²``
    (``eq:timestamp-frobenius-loss``), alongside its diagonal and off-diagonal
    contributions for descriptive holdout comparisons.
    """
    error_values = _matrix_frame_to_array(errors)
    dimension = error_values.shape[1]

    squared_errors = np.square(error_values)

    diagonal_positions = np.arange(dimension)
    diagonal_losses = squared_errors[
        :,
        diagonal_positions,
        diagonal_positions,
    ].sum(axis=1)

    off_diagonal_mask = ~np.eye(
        dimension,
        dtype=bool,
    )
    off_diagonal_losses = squared_errors[
        :,
        off_diagonal_mask,
    ].sum(axis=1)

    timestamps = (
        errors.index
        .get_level_values("timestamp")
        .unique()
    )

    return pd.DataFrame(
        {
            "overall": squared_errors.sum(
                axis=(1, 2)
            ),
            "diagonal": diagonal_losses,
            "off-diagonal": off_diagonal_losses,
        },
        index=pd.Index(
            timestamps,
            name="timestamp",
        ),
        dtype=float,
    )


def automatic_hac_lag(
    observation_count: int,
) -> int:
    """Choose the exploratory HAC lag ``floor(4(n/100)^(2/9))``.

    HAC inference is not part of the retained thesis evaluation.
    """
    if not isinstance(
        observation_count,
        (int, np.integer),
    ):
        raise TypeError(
            "observation_count must be an integer."
        )

    observation_count = int(observation_count)

    if observation_count < 1:
        raise ValueError(
            "At least one observation is required."
        )

    lag = int(
        floor(
            4.0
            * (observation_count / 100.0)
            ** (2.0 / 9.0)
        )
    )

    return min(
        max(lag, 0),
        observation_count - 1,
    )


def automatic_circular_block_length(
    observation_count: int,
) -> int:
    """Choose ``ceil(n^(1/3))`` for the exploratory block bootstrap.

    Bootstrap inference is not part of the retained thesis evaluation.
    """
    if not isinstance(
        observation_count,
        (int, np.integer),
    ):
        raise TypeError(
            "observation_count must be an integer."
        )

    observation_count = int(observation_count)

    if observation_count < 1:
        raise ValueError(
            "At least one observation is required."
        )

    return min(
        observation_count,
        int(
            ceil(
                observation_count
                ** (1.0 / 3.0)
            )
        ),
    )


def compute_hac_equal_accuracy_test(
    loss_differences: pd.Series | np.ndarray,
    max_lag: int,
) -> pd.Series:
    """Run an exploratory equal-accuracy test with Bartlett HAC variance.

    This diagnostic tests the mean loss difference using a normal reference
    distribution. It is retained for experiments but is not reported in the
    thesis, which makes no statistical-significance claim.
    """
    values = np.asarray(
        loss_differences,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "Loss differences must be one-dimensional."
        )

    if len(values) < 2:
        raise ValueError(
            "At least two loss differences are required."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Loss differences contain non-finite values."
        )

    if not isinstance(
        max_lag,
        (int, np.integer),
    ):
        raise TypeError(
            "max_lag must be an integer."
        )

    max_lag = int(max_lag)

    if not 0 <= max_lag < len(values):
        raise ValueError(
            "max_lag must be between zero and the "
            "observation count minus one."
        )

    observation_count = len(values)
    mean_difference = float(np.mean(values))
    centered = values - mean_difference

    gamma_zero = float(
        centered @ centered
        / observation_count
    )
    long_run_variance = gamma_zero

    for lag in range(1, max_lag + 1):
        autocovariance = float(
            centered[lag:]
            @ centered[:-lag]
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

    numerical_tolerance = (
        1e-12
        * max(
            gamma_zero,
            np.finfo(float).tiny,
        )
    )

    if (
        long_run_variance < 0.0
        and abs(long_run_variance)
        <= numerical_tolerance
    ):
        long_run_variance = 0.0

    if long_run_variance < 0.0:
        raise ValueError(
            "The estimated HAC long-run variance "
            "is negative."
        )

    identical_losses = bool(
        np.all(values == values[0])
        and values[0] == 0.0
    )

    if long_run_variance == 0.0:
        standard_error_of_mean = 0.0
        test_statistic = np.nan
        p_value_two_sided = np.nan
    else:
        standard_error_of_mean = sqrt(
            long_run_variance
            / observation_count
        )
        test_statistic = (
            mean_difference
            / standard_error_of_mean
        )
        p_value_two_sided = erfc(
            abs(test_statistic)
            / sqrt(2.0)
        )

    return pd.Series(
        {
            "observation_count": observation_count,
            "hac_max_lag": max_lag,
            "mean_loss_difference": mean_difference,
            "long_run_variance": long_run_variance,
            "standard_error_of_mean": (
                standard_error_of_mean
            ),
            "test_statistic": test_statistic,
            "p_value_two_sided": p_value_two_sided,
            "identical_losses": identical_losses,
        },
        name="hac_equal_accuracy_test",
    )


def circular_block_bootstrap_mean_interval(
    values: pd.Series | np.ndarray,
    block_length: int,
    replication_count: int,
    random_seed: int,
    confidence_level: float = 0.95,
) -> pd.Series:
    """Estimate an exploratory percentile interval by circular block sampling.

    Contiguous blocks preserve local ordering and wrap at the sample boundary.
    This bootstrap diagnostic is not part of the retained thesis evaluation.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "Bootstrap values must be one-dimensional."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Bootstrap values contain non-finite values."
        )

    if not (
        1
        <= block_length
        <= len(values)
    ):
        raise ValueError(
            "block_length must be between one and "
            "the observation count."
        )

    if replication_count < 1:
        raise ValueError(
            "At least one bootstrap replication "
            "is required."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie strictly "
            "between zero and one."
        )

    observation_count = len(values)

    extended = np.concatenate(
        (
            values,
            values[:block_length - 1],
        )
    )

    block_sums = np.convolve(
        extended,
        np.ones(block_length),
        mode="valid",
    )[:observation_count]

    (
        full_block_count,
        remainder_length,
    ) = divmod(
        observation_count,
        block_length,
    )

    random_generator = np.random.default_rng(
        random_seed
    )

    sampled_full_block_starts = (
        random_generator.integers(
            0,
            observation_count,
            size=(
                replication_count,
                full_block_count,
            ),
        )
    )

    bootstrap_sums = block_sums[
        sampled_full_block_starts
    ].sum(axis=1)

    if remainder_length > 0:
        remainder_extended = np.concatenate(
            (
                values,
                values[:remainder_length - 1],
            )
        )
        remainder_block_sums = np.convolve(
            remainder_extended,
            np.ones(remainder_length),
            mode="valid",
        )[:observation_count]

        sampled_remainder_starts = (
            random_generator.integers(
                0,
                observation_count,
                size=replication_count,
            )
        )

        bootstrap_sums += remainder_block_sums[
            sampled_remainder_starts
        ]

    bootstrap_means = (
        bootstrap_sums
        / observation_count
    )

    tail_probability = (
        1.0 - confidence_level
    ) / 2.0

    lower, upper = np.quantile(
        bootstrap_means,
        [
            tail_probability,
            1.0 - tail_probability,
        ],
    )

    return pd.Series(
        {
            "block_length": block_length,
            "bootstrap_replications": (
                replication_count
            ),
            "confidence_level": confidence_level,
            "bootstrap_mean": float(
                np.mean(bootstrap_means)
            ),
            "confidence_interval_lower": float(
                lower
            ),
            "confidence_interval_upper": float(
                upper
            ),
        },
        name="circular_block_bootstrap",
    )
