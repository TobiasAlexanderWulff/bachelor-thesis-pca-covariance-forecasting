"""Evaluate covariance-matrix approximations and forecasts."""

import numpy as np
import pandas as pd


def _matrix_frame_to_array(
    matrices: pd.DataFrame,
) -> np.ndarray:
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
