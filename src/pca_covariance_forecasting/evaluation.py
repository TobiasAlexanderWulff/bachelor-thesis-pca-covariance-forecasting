"""Evaluate covariance-matrix approximations and forecasts."""

import numpy as np
import pandas as pd


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
