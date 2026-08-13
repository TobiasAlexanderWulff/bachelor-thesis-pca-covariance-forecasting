"""Compare fixed residual bases using oracle diagonal coefficients.

This script contrasts the exact identity basis with the training-optimized
historical residual basis and the fixed PCA reference basis. Test residuals
are used only to obtain oracle projection coefficients; no forecasting is
performed here.
"""

from itertools import permutations
from pathlib import Path

import numpy as np
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
    compute_frobenius_norms,
)
from pca_covariance_forecasting.pca import (
    compute_approximation_errors,
    compute_reference_covariance,
    compute_reference_eigendecomposition,
    construct_reference_basis_approximations,
    transform_covariances_from_reference_basis,
    transform_covariances_to_reference_basis,
)


DATA_DIRECTORY = Path("data/raw/binance")
RANDOM_SEED = 0
NUMBER_OF_RANDOM_STARTS = 8
MAXIMUM_JACOBI_SWEEPS = 100
RELATIVE_OBJECTIVE_TOLERANCE = 1e-12


def select_matrix_timestamps(
    matrices: pd.DataFrame,
    timestamps: pd.Index,
) -> pd.DataFrame:
    mask = (
        matrices.index
        .get_level_values("timestamp")
        .isin(timestamps)
    )
    return matrices.loc[mask].copy()


def matrix_frame_to_array(
    matrices: pd.DataFrame,
    assets: pd.Index,
) -> tuple[np.ndarray, pd.Index]:
    timestamps = (
        matrices.index
        .get_level_values("timestamp")
        .unique()
    )

    values = np.stack(
        [
            matrices.xs(
                timestamp,
                level="timestamp",
            ).loc[assets, assets].to_numpy()
            for timestamp in timestamps
        ]
    )

    return values, timestamps


def matrix_array_to_frame(
    values: np.ndarray,
    timestamps: pd.Index,
    assets: pd.Index,
) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                matrix,
                index=assets,
                columns=assets,
            )
            for matrix in values
        ],
        keys=timestamps,
        names=["timestamp", "asset"],
    )


def squared_off_diagonal_norm(
    matrices: np.ndarray,
) -> float:
    diagonal = np.diagonal(
        matrices,
        axis1=1,
        axis2=2,
    )
    return float(
        np.sum(matrices**2)
        - np.sum(diagonal**2)
    )


def create_random_orthogonal_basis(
    dimension: int,
    generator: np.random.Generator,
) -> np.ndarray:
    candidate = generator.normal(
        size=(dimension, dimension)
    )
    basis, upper_triangular = np.linalg.qr(candidate)

    signs = np.sign(np.diag(upper_triangular))
    signs[signs == 0.0] = 1.0
    basis = basis @ np.diag(signs)

    if np.linalg.det(basis) < 0.0:
        basis[:, 0] *= -1.0

    return basis


def run_jacobi_joint_diagonalization(
    matrices: np.ndarray,
    initial_basis: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    dimension = matrices.shape[1]
    basis = initial_basis.copy()
    transformed = (
        basis.T
        @ matrices
        @ basis
    )

    previous_objective = squared_off_diagonal_norm(
        transformed
    )

    for sweep in range(1, MAXIMUM_JACOBI_SWEEPS + 1):
        for first, second in permutations(
            range(dimension),
            2,
        ):
            if first >= second:
                continue

            off_diagonal = transformed[
                :,
                first,
                second,
            ]
            diagonal_difference = 0.5 * (
                transformed[:, second, second]
                - transformed[:, first, first]
            )

            angle_objective = np.array(
                [
                    [
                        np.dot(
                            off_diagonal,
                            off_diagonal,
                        ),
                        np.dot(
                            off_diagonal,
                            diagonal_difference,
                        ),
                    ],
                    [
                        np.dot(
                            off_diagonal,
                            diagonal_difference,
                        ),
                        np.dot(
                            diagonal_difference,
                            diagonal_difference,
                        ),
                    ],
                ]
            )

            _, direction = np.linalg.eigh(
                angle_objective
            )
            cosine_double_angle, sine_double_angle = (
                direction[:, 0]
            )
            angle = 0.5 * np.arctan2(
                sine_double_angle,
                cosine_double_angle,
            )

            angle = (
                (angle + np.pi / 4.0)
                % (np.pi / 2.0)
                - np.pi / 4.0
            )

            cosine = np.cos(angle)
            sine = np.sin(angle)

            rotation = np.eye(dimension)
            rotation[first, first] = cosine
            rotation[second, second] = cosine
            rotation[first, second] = -sine
            rotation[second, first] = sine

            transformed = (
                rotation.T
                @ transformed
                @ rotation
            )
            basis = basis @ rotation

        objective = squared_off_diagonal_norm(
            transformed
        )
        improvement = previous_objective - objective

        if improvement <= (
            RELATIVE_OBJECTIVE_TOLERANCE
            * max(
                previous_objective,
                np.finfo(float).tiny,
            )
        ):
            return basis, objective, sweep

        previous_objective = objective

    return basis, previous_objective, MAXIMUM_JACOBI_SWEEPS


def canonicalize_basis_against_identity(
    basis: np.ndarray,
) -> np.ndarray:
    dimension = basis.shape[0]

    best_column_permutation = max(
        permutations(range(dimension)),
        key=lambda column_permutation: sum(
            abs(basis[row, column_permutation[row]])
            for row in range(dimension)
        ),
    )

    canonical_basis = basis[
        :,
        list(best_column_permutation),
    ].copy()

    for component in range(dimension):
        if canonical_basis[component, component] < 0.0:
            canonical_basis[:, component] *= -1.0

    return canonical_basis


def optimize_historical_residual_basis(
    training_errors: pd.DataFrame,
    reference_basis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    assets = training_errors.columns
    matrices, _ = matrix_frame_to_array(
        training_errors,
        assets,
    )
    dimension = len(assets)

    generator = np.random.default_rng(RANDOM_SEED)
    initial_bases = [
        np.eye(dimension),
        reference_basis.loc[
            assets,
            reference_basis.columns,
        ].to_numpy(),
    ]
    initial_bases.extend(
        create_random_orthogonal_basis(
            dimension,
            generator,
        )
        for _ in range(NUMBER_OF_RANDOM_STARTS)
    )

    best_basis = None
    best_objective = np.inf
    best_sweeps = 0

    for initial_basis in initial_bases:
        candidate_basis, objective, sweeps = (
            run_jacobi_joint_diagonalization(
                matrices,
                initial_basis,
            )
        )

        if objective < best_objective:
            best_basis = candidate_basis
            best_objective = objective
            best_sweeps = sweeps

    best_basis = canonicalize_basis_against_identity(
        best_basis
    )

    component_names = [
        f"{asset}_aligned_component"
        for asset in assets
    ]
    historical_basis = pd.DataFrame(
        best_basis,
        index=assets,
        columns=component_names,
    )
    optimization_information = pd.Series(
        {
            "number_of_starts": len(initial_bases),
            "best_number_of_sweeps": best_sweeps,
            "training_off_diagonal_squared_norm": (
                best_objective
            ),
        },
        name="value",
    )

    return historical_basis, optimization_information


def project_errors_onto_basis_diagonal(
    errors: pd.DataFrame,
    basis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = errors.columns
    basis_values = basis.loc[
        assets,
        basis.columns,
    ].to_numpy()
    error_values, timestamps = matrix_frame_to_array(
        errors,
        assets,
    )

    transformed_errors = (
        basis_values.T
        @ error_values
        @ basis_values
    )
    coefficients = np.diagonal(
        transformed_errors,
        axis1=1,
        axis2=2,
    ).copy()

    projected_values = np.einsum(
        "ak,tk,bk->tab",
        basis_values,
        coefficients,
        basis_values,
    )

    projected_errors = matrix_array_to_frame(
        projected_values,
        timestamps,
        assets,
    )
    coefficient_frame = pd.DataFrame(
        coefficients,
        index=pd.Index(
            timestamps,
            name="timestamp",
        ),
        columns=basis.columns,
    )

    return projected_errors, coefficient_frame


def calculate_component_metrics(
    actual_values: np.ndarray,
    approximated_values: np.ndarray,
    component: str,
) -> dict[str, float]:
    difference = actual_values - approximated_values
    dimension = actual_values.shape[1]

    if component == "overall":
        actual_component = actual_values.reshape(-1)
        difference_component = difference.reshape(-1)
    elif component == "diagonal":
        positions = np.arange(dimension)
        actual_component = actual_values[
            :,
            positions,
            positions,
        ].reshape(-1)
        difference_component = difference[
            :,
            positions,
            positions,
        ].reshape(-1)
    elif component == "off-diagonal":
        off_diagonal = ~np.eye(
            dimension,
            dtype=bool,
        )
        actual_component = actual_values[
            :,
            off_diagonal,
        ].reshape(-1)
        difference_component = difference[
            :,
            off_diagonal,
        ].reshape(-1)
    else:
        raise ValueError(
            f"Unknown matrix component: {component}"
        )

    squared_error = float(
        np.sum(difference_component**2)
    )
    actual_squared_norm = float(
        np.sum(actual_component**2)
    )

    return {
        "rmse": float(
            np.sqrt(
                np.mean(difference_component**2)
            )
        ),
        "aggregated_relative_error": float(
            np.sqrt(
                squared_error
                / actual_squared_norm
            )
        ),
        "squared_error": squared_error,
    }


def build_method_comparison(
    actual_covariances: pd.DataFrame,
    approximations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    assets = actual_covariances.columns
    actual_values, timestamps = matrix_frame_to_array(
        actual_covariances,
        assets,
    )

    method_metrics = {}

    for method, approximation in approximations.items():
        approximation = select_matrix_timestamps(
            approximation,
            timestamps,
        )
        approximation_values, _ = matrix_frame_to_array(
            approximation,
            assets,
        )

        for component in (
            "overall",
            "diagonal",
            "off-diagonal",
        ):
            metrics = calculate_component_metrics(
                actual_values,
                approximation_values,
                component,
            )

            if component == "overall":
                approximation_errors = (
                    actual_covariances
                    - approximation
                )
                metrics["aggregated_relative_error"] = (
                    compute_aggregated_relative_frobenius_error(
                        approximation_errors,
                        actual_covariances,
                    )
                )

            method_metrics[(method, component)] = metrics

    baseline_method = "one_indicator"
    rows = []

    for (method, component), metrics in method_metrics.items():
        baseline = method_metrics[
            (baseline_method, component)
        ]
        rmse_ratio = (
            metrics["rmse"]
            / baseline["rmse"]
        )

        rows.append(
            {
                "method": method,
                "component": component,
                "rmse": metrics["rmse"],
                "aggregated_relative_error": (
                    metrics[
                        "aggregated_relative_error"
                    ]
                ),
                "rmse_ratio_to_one_indicator": (
                    rmse_ratio
                ),
                "rmse_change_percent": (
                    100.0 * (rmse_ratio - 1.0)
                ),
                (
                    "original_residual_squared_norm_"
                    "remaining"
                ): (
                    metrics["squared_error"]
                    / baseline["squared_error"]
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index(["method", "component"])
    )


def residual_squared_norm_share_remaining(
    errors: pd.DataFrame,
    projected_errors: pd.DataFrame,
    timestamps: pd.Index,
) -> float:
    selected_errors = select_matrix_timestamps(
        errors,
        timestamps,
    )
    selected_projection = select_matrix_timestamps(
        projected_errors,
        timestamps,
    )
    remaining_errors = (
        selected_errors
        - selected_projection
    )

    remaining_norms = compute_frobenius_norms(
        remaining_errors
    )
    original_norms = compute_frobenius_norms(
        selected_errors
    )

    return float(
        np.square(remaining_norms.to_numpy()).sum()
        / np.square(original_norms.to_numpy()).sum()
    )


def build_identity_alignment_table(
    historical_basis: pd.DataFrame,
) -> pd.DataFrame:
    values = historical_basis.to_numpy()

    return pd.DataFrame(
        {
            "aligned_asset": historical_basis.index,
            "absolute_axis_loading": np.abs(
                np.diag(values)
            ),
            "angle_to_identity_axis_degrees": (
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.abs(np.diag(values)),
                            0.0,
                            1.0,
                        )
                    )
                )
            ),
        },
        index=pd.Index(
            historical_basis.columns,
            name="historical_component",
        ),
    )


def build_coefficient_split_comparison(
    coefficients: pd.DataFrame,
    training_timestamps: pd.Index,
    test_timestamps: pd.Index,
) -> pd.DataFrame:
    rows = []

    for component in coefficients.columns:
        training = coefficients.loc[
            training_timestamps,
            component,
        ]
        test = coefficients.loc[
            test_timestamps,
            component,
        ]

        training_standard_deviation = float(
            training.std()
        )
        test_standard_deviation = float(test.std())

        rows.append(
            {
                "component": component,
                "training_mean": float(training.mean()),
                "test_mean": float(test.mean()),
                "training_std": (
                    training_standard_deviation
                ),
                "test_std": test_standard_deviation,
                "test_to_training_std_ratio": (
                    test_standard_deviation
                    / training_standard_deviation
                ),
                "mean_shift_in_training_std": (
                    (
                        float(test.mean())
                        - float(training.mean())
                    )
                    / training_standard_deviation
                ),
                "training_lag_1_autocorrelation": (
                    float(training.autocorr(lag=1))
                ),
                "test_lag_1_autocorrelation": (
                    float(test.autocorr(lag=1))
                ),
                "training_negative_share": float(
                    (training < 0.0).mean()
                ),
                "test_negative_share": float(
                    (test < 0.0).mean()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("component")
    )


def build_test_off_diagonal_pair_comparison(
    actual_covariances: pd.DataFrame,
    approximations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    assets = actual_covariances.columns
    actual_values, timestamps = matrix_frame_to_array(
        actual_covariances,
        assets,
    )
    approximation_values = {
        method: matrix_frame_to_array(
            select_matrix_timestamps(
                approximation,
                timestamps,
            ),
            assets,
        )[0]
        for method, approximation in approximations.items()
    }

    rows = []

    for first in range(len(assets)):
        for second in range(first + 1, len(assets)):
            baseline_difference = (
                actual_values[:, first, second]
                - approximation_values[
                    "one_indicator"
                ][:, first, second]
            )
            baseline_rmse = float(
                np.sqrt(
                    np.mean(
                        baseline_difference**2
                    )
                )
            )

            for method, values in (
                approximation_values.items()
            ):
                difference = (
                    actual_values[:, first, second]
                    - values[:, first, second]
                )
                rmse = float(
                    np.sqrt(
                        np.mean(difference**2)
                    )
                )

                rows.append(
                    {
                        "method": method,
                        "asset_pair": (
                            f"{assets[first]}--"
                            f"{assets[second]}"
                        ),
                        "rmse": rmse,
                        "rmse_ratio_to_one_indicator": (
                            rmse / baseline_rmse
                        ),
                        "rmse_change_percent": (
                            100.0
                            * (
                                rmse
                                / baseline_rmse
                                - 1.0
                            )
                        ),
                    }
                )

    return (
        pd.DataFrame(rows)
        .set_index(["method", "asset_pair"])
    )


def verify_identity_basis_invariants(
    actual_covariances: pd.DataFrame,
    one_indicator_covariances: pd.DataFrame,
    identity_basis_covariances: pd.DataFrame,
) -> None:
    assets = actual_covariances.columns
    actual_values, timestamps = matrix_frame_to_array(
        actual_covariances,
        assets,
    )
    one_indicator_values, _ = matrix_frame_to_array(
        select_matrix_timestamps(
            one_indicator_covariances,
            timestamps,
        ),
        assets,
    )
    identity_basis_values, _ = matrix_frame_to_array(
        select_matrix_timestamps(
            identity_basis_covariances,
            timestamps,
        ),
        assets,
    )

    diagonal_positions = np.arange(len(assets))
    np.testing.assert_allclose(
        identity_basis_values[
            :,
            diagonal_positions,
            diagonal_positions,
        ],
        actual_values[
            :,
            diagonal_positions,
            diagonal_positions,
        ],
        rtol=1e-12,
        atol=1e-18,
    )

    off_diagonal_positions = ~np.eye(
        len(assets),
        dtype=bool,
    )
    np.testing.assert_array_equal(
        identity_basis_values[
            :,
            off_diagonal_positions,
        ],
        one_indicator_values[
            :,
            off_diagonal_positions,
        ],
    )


prices = load_closing_prices(DATA_DIRECTORY)
returns = compute_log_returns(prices)
covariances = compute_block_covariances(returns)

all_timestamps = (
    covariances.index
    .get_level_values("timestamp")
    .unique()
)
split_position = len(all_timestamps) // 2
training_timestamps = all_timestamps[:split_position]
test_timestamps = all_timestamps[split_position:]

training_covariances = select_matrix_timestamps(
    covariances,
    training_timestamps,
)
test_covariances = select_matrix_timestamps(
    covariances,
    test_timestamps,
)

reference_covariance = compute_reference_covariance(
    training_covariances
)
reference_basis, reference_eigenvalues = (
    compute_reference_eigendecomposition(
        reference_covariance
    )
)

transformed_covariances = (
    transform_covariances_to_reference_basis(
        covariances,
        reference_basis,
    )
)
reference_basis_approximations = (
    construct_reference_basis_approximations(
        transformed_covariances,
        reference_eigenvalues,
    )
)
one_indicator_covariances = (
    transform_covariances_from_reference_basis(
        reference_basis_approximations,
        reference_basis,
    )
)
errors = compute_approximation_errors(
    covariances,
    one_indicator_covariances,
)
training_errors = select_matrix_timestamps(
    errors,
    training_timestamps,
)

historical_residual_basis, optimization_information = (
    optimize_historical_residual_basis(
        training_errors,
        reference_basis,
    )
)
identity_residual_basis = pd.DataFrame(
    np.eye(len(errors.columns)),
    index=errors.columns,
    columns=[
        f"{asset}_identity_component"
        for asset in errors.columns
    ],
)
reference_residual_basis = reference_basis.copy()
reference_residual_basis.columns = [
    f"reference_error_component_{number}"
    for number in range(
        1,
        len(reference_basis.columns) + 1,
    )
]

(
    reference_projected_errors,
    reference_coefficients,
) = project_errors_onto_basis_diagonal(
    errors,
    reference_residual_basis,
)
(
    historical_projected_errors,
    historical_coefficients,
) = project_errors_onto_basis_diagonal(
    errors,
    historical_residual_basis,
)
(
    identity_projected_errors,
    identity_coefficients,
) = project_errors_onto_basis_diagonal(
    errors,
    identity_residual_basis,
)

all_approximations = {
    "one_indicator": one_indicator_covariances,
    "reference_basis_diagonal": (
        one_indicator_covariances
        + reference_projected_errors
    ),
    "historical_residual_basis_diagonal": (
        one_indicator_covariances
        + historical_projected_errors
    ),
    "identity_basis_diagonal": (
        one_indicator_covariances
        + identity_projected_errors
    ),
}

verify_identity_basis_invariants(
    covariances,
    one_indicator_covariances,
    all_approximations["identity_basis_diagonal"],
)

training_approximations = {
    method: select_matrix_timestamps(
        approximation,
        training_timestamps,
    )
    for method, approximation in all_approximations.items()
}
test_approximations = {
    method: select_matrix_timestamps(
        approximation,
        test_timestamps,
    )
    for method, approximation in all_approximations.items()
}

training_comparison = build_method_comparison(
    training_covariances,
    training_approximations,
)
test_comparison = build_method_comparison(
    test_covariances,
    test_approximations,
)

coefficient_comparison = pd.concat(
    {
        "historical_residual_basis": (
            build_coefficient_split_comparison(
                historical_coefficients,
                training_timestamps,
                test_timestamps,
            )
        ),
        "identity_basis": (
            build_coefficient_split_comparison(
                identity_coefficients,
                training_timestamps,
                test_timestamps,
            )
        ),
    },
    names=["basis", "component"],
)

test_pair_comparison = (
    build_test_off_diagonal_pair_comparison(
        test_covariances,
        test_approximations,
    )
)

mean_training_error = (
    training_errors
    .groupby(level="asset", sort=False)
    .mean()
)

print(
    "Frobenius norm of mean training residual:"
)
print(
    f"{np.linalg.norm(mean_training_error):.6e}"
)

print("\nHistorical-basis optimization information:")
print(optimization_information.to_string())

print("\nCanonicalized historical residual basis:")
print(historical_residual_basis.to_string())

print("\nAlignment with the exact identity basis:")
print(
    build_identity_alignment_table(
        historical_residual_basis
    ).to_string()
)

print(
    "\nResidual squared-norm share remaining "
    "after exact diagonal projection:"
)
for split_name, timestamps in (
    ("Training", training_timestamps),
    ("Test", test_timestamps),
):
    for basis_name, projected_errors in (
        (
            "reference basis",
            reference_projected_errors,
        ),
        (
            "historical residual basis",
            historical_projected_errors,
        ),
        (
            "identity basis",
            identity_projected_errors,
        ),
    ):
        remaining = (
            residual_squared_norm_share_remaining(
                errors,
                projected_errors,
                timestamps,
            )
        )
        print(
            f"{split_name}, {basis_name}: "
            f"{100.0 * remaining:.2f}%"
        )

with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    220,
):
    print("\nTraining comparison:")
    print(training_comparison)

    print("\nTest comparison:")
    print(test_comparison)

    print(
        "\nHistorical-versus-identity coefficient "
        "train/test comparison:"
    )
    print(coefficient_comparison)

    print("\nTest off-diagonal comparison by asset pair:")
    print(test_pair_comparison)
