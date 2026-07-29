"""Tests for covariance-matrix evaluation utilities."""

import unittest
from math import erfc, sqrt

import numpy as np
import pandas as pd

from pca_covariance_forecasting.evaluation import (
    automatic_circular_block_length,
    automatic_hac_lag,
    circular_block_bootstrap_mean_interval,
    compute_covariance_rmse,
    compute_covariance_squared_frobenius_losses,
    compute_hac_equal_accuracy_test,
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


class SquaredFrobeniusLossTests(unittest.TestCase):
    def test_computes_timestamp_level_losses(
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

        result = (
            compute_covariance_squared_frobenius_losses(
                errors
            )
        )

        expected = pd.DataFrame(
            {
                "overall": [18.0, 102.0],
                "diagonal": [10.0, 52.0],
                "off-diagonal": [8.0, 50.0],
            },
            index=result.index,
        )

        pd.testing.assert_frame_equal(
            result,
            expected,
        )


class AutomaticDependenceParameterTests(
    unittest.TestCase
):
    def test_reproduces_frozen_holdout_choices(
        self,
    ) -> None:
        self.assertEqual(
            automatic_hac_lag(2184),
            7,
        )
        self.assertEqual(
            automatic_circular_block_length(2184),
            13,
        )


class HacEqualAccuracyTests(unittest.TestCase):
    def test_computes_bartlett_hac_test(
        self,
    ) -> None:
        differences = np.array(
            [1.0, 2.0, 4.0, 8.0]
        )

        result = compute_hac_equal_accuracy_test(
            differences,
            max_lag=1,
        )

        expected_long_run_variance = 8.546875
        expected_standard_error = sqrt(
            expected_long_run_variance / 4.0
        )
        expected_statistic = (
            3.75 / expected_standard_error
        )
        expected_p_value = erfc(
            abs(expected_statistic)
            / sqrt(2.0)
        )

        self.assertEqual(
            result["observation_count"],
            4,
        )
        self.assertEqual(
            result["hac_max_lag"],
            1,
        )
        self.assertAlmostEqual(
            result["mean_loss_difference"],
            3.75,
        )
        self.assertAlmostEqual(
            result["long_run_variance"],
            expected_long_run_variance,
        )
        self.assertAlmostEqual(
            result["standard_error_of_mean"],
            expected_standard_error,
        )
        self.assertAlmostEqual(
            result["test_statistic"],
            expected_statistic,
        )
        self.assertAlmostEqual(
            result["p_value_two_sided"],
            expected_p_value,
        )
        self.assertFalse(
            result["identical_losses"]
        )

    def test_identifies_identical_losses(
        self,
    ) -> None:
        result = compute_hac_equal_accuracy_test(
            np.zeros(10),
            max_lag=2,
        )

        self.assertTrue(
            result["identical_losses"]
        )
        self.assertEqual(
            result["standard_error_of_mean"],
            0.0,
        )
        self.assertTrue(
            np.isnan(result["test_statistic"])
        )
        self.assertTrue(
            np.isnan(result["p_value_two_sided"])
        )


class CircularBlockBootstrapTests(
    unittest.TestCase
):
    def test_full_length_blocks_preserve_mean(
        self,
    ) -> None:
        result = (
            circular_block_bootstrap_mean_interval(
                values=np.array(
                    [1.0, 2.0, 3.0, 4.0]
                ),
                block_length=4,
                replication_count=100,
                random_seed=20260729,
            )
        )

        self.assertEqual(
            result["block_length"],
            4,
        )
        self.assertEqual(
            result["bootstrap_replications"],
            100,
        )
        self.assertAlmostEqual(
            result["bootstrap_mean"],
            2.5,
        )
        self.assertAlmostEqual(
            result["confidence_interval_lower"],
            2.5,
        )
        self.assertAlmostEqual(
            result["confidence_interval_upper"],
            2.5,
        )


    def test_truncates_last_block_to_sample_size(
        self,
    ) -> None:
        result = (
            circular_block_bootstrap_mean_interval(
                values=np.array(
                    [1.0, 2.0, 4.0, 8.0, 16.0]
                ),
                block_length=2,
                replication_count=4,
                random_seed=20260729,
            )
        )

        self.assertAlmostEqual(
            result["bootstrap_mean"],
            6.55,
        )
        self.assertAlmostEqual(
            result["confidence_interval_lower"],
            4.845,
        )
        self.assertAlmostEqual(
            result["confidence_interval_upper"],
            10.04,
        )


if __name__ == "__main__":
    unittest.main()
