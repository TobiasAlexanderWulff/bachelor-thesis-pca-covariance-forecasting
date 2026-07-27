"""Construct the fixed PCA reference basis."""

import numpy as np
import pandas as pd


def compute_reference_covariance(
    training_covariances: pd.DataFrame,
) -> pd.DataFrame:
    return training_covariances.groupby(
        level="asset",
        sort=False,
    ).mean()


def compute_reference_eigendecomposition(
    reference_covariance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    eigenvalues, eigenvectors = np.linalg.eigh(
        reference_covariance.to_numpy()
    )

    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    component_names = [
        f"component_{number}"
        for number in range(1, len(eigenvalues) + 1)
    ]

    reference_basis = pd.DataFrame(
        eigenvectors,
        index=reference_covariance.index,
        columns=component_names,
    )
    reference_eigenvalues = pd.Series(
        eigenvalues,
        index=component_names,
        name="eigenvalue",
    )

    return reference_basis, reference_eigenvalues


def transform_covariances_to_reference_basis(
    covariances: pd.DataFrame,
    reference_basis: pd.DataFrame,
) -> pd.DataFrame:
    basis = reference_basis.to_numpy()

    transformed_matrices = []
    timestamps = []

    for timestamp, covariance in covariances.groupby(
        level="timestamp",
        sort=False,
    ):
        covariance = covariance.droplevel("timestamp")
        covariance = covariance.loc[
            reference_basis.index,
            reference_basis.index,
        ]

        transformed = (
            basis.T
            @ covariance.to_numpy()
            @ basis
        )

        transformed_matrices.append(
            pd.DataFrame(
                transformed,
                index=reference_basis.columns,
                columns=reference_basis.columns,
            )
        )
        timestamps.append(timestamp)

    return pd.concat(
        transformed_matrices,
        keys=timestamps,
        names=["timestamp", "component"],
    )
