"""Construct PCA-based covariance approximations and their errors."""

from itertools import permutations

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


def construct_reference_basis_approximations(
    transformed_covariances: pd.DataFrame,
    reference_eigenvalues: pd.Series,
) -> pd.DataFrame:
    components = reference_eigenvalues.index

    approximations = []
    timestamps = []

    for timestamp, transformed in transformed_covariances.groupby(
        level="timestamp",
        sort=False,
    ):
        transformed = transformed.droplevel("timestamp").loc[
            components,
            components,
        ]

        approximation_diagonal = reference_eigenvalues.copy()
        first_component = components[0]

        approximation_diagonal.loc[first_component] = transformed.loc[
            first_component,
            first_component,
        ]

        approximations.append(
            pd.DataFrame(
                np.diag(approximation_diagonal.to_numpy()),
                index=components,
                columns=components,
            )
        )
        timestamps.append(timestamp)

    return pd.concat(
        approximations,
        keys=timestamps,
        names=["timestamp", "component"],
    )


def transform_covariances_from_reference_basis(
    transformed_covariances: pd.DataFrame,
    reference_basis: pd.DataFrame,
) -> pd.DataFrame:
    basis = reference_basis.to_numpy()
    components = reference_basis.columns

    reconstructed_matrices = []
    timestamps = []

    for timestamp, transformed in transformed_covariances.groupby(
        level="timestamp",
        sort=False,
    ):
        transformed = transformed.droplevel("timestamp").loc[
            components,
            components,
        ]

        reconstructed = (
            basis
            @ transformed.to_numpy()
            @ basis.T
        )

        reconstructed_matrices.append(
            pd.DataFrame(
                reconstructed,
                index=reference_basis.index,
                columns=reference_basis.index,
            )
        )
        timestamps.append(timestamp)

    return pd.concat(
        reconstructed_matrices,
        keys=timestamps,
        names=["timestamp", "asset"],
    )


def compute_approximation_errors(
    covariances: pd.DataFrame,
    approximated_covariances: pd.DataFrame,
) -> pd.DataFrame:
    return covariances - approximated_covariances


def compute_error_eigendecompositions(
    errors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = errors.columns
    component_names = [
        f"error_component_{number}"
        for number in range(1, len(assets) + 1)
    ]

    error_bases = []
    error_eigenvalues = []
    timestamps = []

    for timestamp, error in errors.groupby(
        level="timestamp",
        sort=False,
    ):
        error = error.droplevel("timestamp").loc[
            assets,
            assets,
        ]

        eigenvalues, eigenvectors = np.linalg.eigh(
            error.to_numpy()
        )

        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        error_bases.append(
            pd.DataFrame(
                eigenvectors,
                index=assets,
                columns=component_names,
            )
        )
        error_eigenvalues.append(eigenvalues)
        timestamps.append(timestamp)

    bases = pd.concat(
        error_bases,
        keys=timestamps,
        names=["timestamp", "asset"],
    )
    eigenvalues = pd.DataFrame(
        error_eigenvalues,
        index=pd.Index(
            timestamps,
            name="timestamp",
        ),
        columns=component_names,
    )

    return bases, eigenvalues


def match_error_eigendecomposition(
    previous_basis: pd.DataFrame,
    current_basis: pd.DataFrame,
    current_eigenvalues: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    similarities = np.abs(
        previous_basis.to_numpy().T
        @ current_basis.to_numpy()
    )

    component_count = previous_basis.shape[1]

    best_permutation = max(
        permutations(range(component_count)),
        key=lambda permutation: sum(
            similarities[previous_index, current_index]
            for previous_index, current_index
            in enumerate(permutation)
        ),
    )

    matched_basis = current_basis.iloc[
        :,
        list(best_permutation)
    ].copy()

    matched_eigenvalues = current_eigenvalues.iloc[
        list(best_permutation)
    ].copy()

    matched_basis.columns = previous_basis.columns
    matched_eigenvalues.index = previous_basis.columns

    for component in previous_basis.columns:
        alignment = np.dot(
            previous_basis[component],
            matched_basis[component],
        )

        if alignment < 0:
            matched_basis[component] *= -1

    return matched_basis, matched_eigenvalues


def match_error_eigendecompositions(
    error_bases: pd.DataFrame,
    error_eigenvalues: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = error_eigenvalues.index

    first_timestamp = timestamps[0]
    previous_basis = error_bases.xs(
        first_timestamp,
        level="timestamp",
    ).copy()

    matched_bases = [previous_basis]
    matched_eigenvalue_rows = [
        error_eigenvalues.loc[first_timestamp].copy()
    ]

    for timestamp in timestamps[1:]:
        current_basis = error_bases.xs(
            timestamp,
            level="timestamp",
        )
        current_eigenvalues = error_eigenvalues.loc[
            timestamp
        ]

        matched_basis, matched_eigenvalues = (
            match_error_eigendecomposition(
                previous_basis,
                current_basis,
                current_eigenvalues,
            )
        )

        matched_bases.append(matched_basis)
        matched_eigenvalue_rows.append(
            matched_eigenvalues
        )

        previous_basis = matched_basis

    bases = pd.concat(
        matched_bases,
        keys=timestamps,
        names=["timestamp", "asset"],
    )
    eigenvalues = pd.DataFrame(
        matched_eigenvalue_rows,
        index=timestamps,
    )

    return bases, eigenvalues


def reconstruct_errors_from_eigendecompositions(
    error_bases: pd.DataFrame,
    error_eigenvalues: pd.DataFrame,
) -> pd.DataFrame:
    components = error_eigenvalues.columns

    reconstructed_errors = []
    timestamps = []

    for timestamp in error_eigenvalues.index:
        basis = error_bases.xs(
            timestamp,
            level="timestamp",
        ).loc[:, components]

        eigenvalues = error_eigenvalues.loc[
            timestamp,
            components,
        ]

        reconstructed = (
            basis.to_numpy()
            @ np.diag(eigenvalues.to_numpy())
            @ basis.to_numpy().T
        )

        reconstructed_errors.append(
            pd.DataFrame(
                reconstructed,
                index=basis.index,
                columns=basis.index,
            )
        )
        timestamps.append(timestamp)

    return pd.concat(
        reconstructed_errors,
        keys=timestamps,
        names=["timestamp", "asset"],
    )
