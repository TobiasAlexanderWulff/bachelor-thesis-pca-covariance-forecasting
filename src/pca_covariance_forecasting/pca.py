"""Construct PCA-based covariance approximations and their errors."""

from itertools import permutations

import numpy as np
import pandas as pd


def compute_reference_covariance(
    training_covariances: pd.DataFrame,
) -> pd.DataFrame:
    """Average the training matrices to obtain the fixed PCA reference.

    Implements ``Σ_bar = |T|^-1 sum_{t in T} Σ_t``
    (``eq:training-reference-covariance``). Holdout matrices must therefore
    not be included in ``training_covariances``.
    """
    return training_covariances.groupby(
        level="asset",
        sort=False,
    ).mean()


def compute_reference_eigendecomposition(
    reference_covariance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute the fixed PCA basis and descending reference eigenvalues.

    This supplies ``Σ_bar = V diag(λ_1,...,λ_p) V^T``
    (``eq:pca-eigendecomposition``), with the columns of ``V`` ordered from
    the largest to the smallest eigenvalue.
    """
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
    """Express every covariance matrix in the fixed training PCA basis.

    Implements ``B_t = V^T Σ_t V`` (``eq:fixed-basis-transform``). Because
    the same ``V`` is used at every timestamp, ``(B_t)_{11}`` is the thesis
    dominant indicator rather than a time-varying largest eigenvalue.
    """
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
    """Build the observed one-indicator approximation in PCA coordinates.

    Together with the inverse transform, this implements
    ``Σ̃_t = V diag(z_t, λ_2, ..., λ_p) V^T``
    (``eq:one-indicator-representation``), where
    ``z_t = (B_t)_{11}`` (``eq:dominant-indicator``); transformed off-diagonal
    terms are set to zero. This is the thesis representation diagnostic, not
    an out-of-sample forecast.
    """
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
    """Map matrices from fixed PCA coordinates back to asset coordinates.

    Implements ``Σ_t = V B_t V^T``. When ``B_t`` is the one-indicator
    diagonal matrix, this is the reconstruction used for both approximation
    diagnostics and covariance forecasts.
    """
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
    """Return ``Σ_t - Σ̃_t`` for the thesis approximation diagnostic."""
    return covariances - approximated_covariances


def compute_error_eigendecompositions(
    errors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagonalize approximation errors for exploratory residual analysis.

    This residual-PCA path is not used in the retained thesis comparison.
    """
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
    """Align one exploratory error basis with its predecessor.

    Components are permuted by absolute inner-product similarity and their
    signs are adjusted for continuity. This is outside the retained thesis
    forecast design.
    """
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
    """Apply exploratory error-basis matching through chronological time.

    This residual-PCA path is not used in the retained thesis comparison.
    """
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
    """Reconstruct exploratory errors as ``V_t diag(λ_t) V_t^T``.

    This residual-PCA path is not used in the retained thesis comparison.
    """
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


def compute_consecutive_error_basis_similarities(
    matched_error_bases: pd.DataFrame,
) -> pd.DataFrame:
    """Measure consecutive exploratory basis similarity by inner products.

    This residual-PCA diagnostic is not reported in the retained thesis.
    """
    timestamps = (
        matched_error_bases.index
        .get_level_values("timestamp")
        .unique()
    )
    components = matched_error_bases.columns

    similarities = []

    for previous_timestamp, current_timestamp in zip(
        timestamps[:-1],
        timestamps[1:],
    ):
        previous_basis = matched_error_bases.xs(
            previous_timestamp,
            level="timestamp",
        ).loc[:, components]

        current_basis = matched_error_bases.xs(
            current_timestamp,
            level="timestamp",
        ).loc[
            previous_basis.index,
            components,
        ]

        component_similarities = np.sum(
            previous_basis.to_numpy()
            * current_basis.to_numpy(),
            axis=0,
        )
        similarities.append(component_similarities)

    return pd.DataFrame(
        similarities,
        index=pd.Index(
            timestamps[1:],
            name="timestamp",
        ),
        columns=components,
    )


def construct_previous_basis_error_approximations(
    matched_error_bases: pd.DataFrame,
    matched_error_eigenvalues: pd.DataFrame,
) -> pd.DataFrame:
    """Combine prior error bases with current exploratory eigenvalues.

    This residual approximation is not part of the retained thesis forecast
    design.
    """
    timestamps = matched_error_eigenvalues.index
    components = matched_error_eigenvalues.columns

    approximations = []
    approximation_timestamps = []

    for previous_timestamp, current_timestamp in zip(
        timestamps[:-1],
        timestamps[1:],
    ):
        previous_basis = matched_error_bases.xs(
            previous_timestamp,
            level="timestamp",
        ).loc[:, components]

        current_eigenvalues = matched_error_eigenvalues.loc[
            current_timestamp,
            components,
        ]

        approximation = (
            previous_basis.to_numpy()
            @ np.diag(current_eigenvalues.to_numpy())
            @ previous_basis.to_numpy().T
        )

        approximations.append(
            pd.DataFrame(
                approximation,
                index=previous_basis.index,
                columns=previous_basis.index,
            )
        )
        approximation_timestamps.append(current_timestamp)

    return pd.concat(
        approximations,
        keys=approximation_timestamps,
        names=["timestamp", "asset"],
    )
