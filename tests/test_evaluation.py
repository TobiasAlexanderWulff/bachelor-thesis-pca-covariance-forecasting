"""Tests for covariance-matrix evaluation utilities."""

import unittest

import numpy as np
import pandas as pd

from pca_covariance_forecasting.evaluation import (
    compute_covariance_rmse,
    compute_psd_diagnostics,
)


def matrix_array_to_frame(
    values: np.ndarray,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=len(values),
        freq="30min",
        name="timestamp",
    )
    assets = pd.Index(["A", "B"], name="asset")

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


class CovarianceRmseTests(unittest.TestCase):
    def test_computes_overall_and_component_rmse(
        self,
    ) -> None:
        errors = matrix_array_to_frame(
            np.array(
                [
                    [[1.0, 2.0], [2.0, 3.0]],
                    [[4.0, 5.0], [5.0, 6.0]],
                ]
            )
        )

        result = compute_covariance_rmse(errors)

        expected = pd.Series(
            {
                "overall": np.sqrt(120.0 / 8.0),
                "diagonal": np.sqrt(62.0 / 4.0),
                "off-diagonal": np.sqrt(58.0 / 4.0),
            },
            name="rmse",
            dtype=float,
        )

        pd.testing.assert_series_equal(result, expected)


class PsdDiagnosticsTests(unittest.TestCase):
    def test_distinguishes_rounding_noise_from_non_psd_matrices(
        self,
    ) -> None:
        covariance_matrices = matrix_array_to_frame(
            np.array(
                [
                    [[2.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, -1e-14]],
                    [[1.0, 0.0], [0.0, -1e-4]],
                ]
            )
        )

        result = compute_psd_diagnostics(
            covariance_matrices
        )

        self.assertEqual(
            result["minimum_eigenvalue"],
            -1e-4,
        )
        self.assertEqual(result["raw_negative_count"], 2)
        self.assertEqual(result["non_psd_count"], 1)
        self.assertEqual(
            result["non_psd_share"],
            1.0 / 3.0,
        )

    def test_rejects_negative_relative_tolerance(
        self,
    ) -> None:
        covariance_matrices = matrix_array_to_frame(
            np.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                ]
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "relative_tolerance must be non-negative",
        ):
            compute_psd_diagnostics(
                covariance_matrices,
                relative_tolerance=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
