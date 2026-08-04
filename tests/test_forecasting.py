"""Tests for covariance-factor forecasting models."""

import unittest

import numpy as np
import pandas as pd

from pca_covariance_forecasting.forecasting import (
    construct_direct_naive_covariance_forecasts,
    construct_dominant_indicator_covariance_forecasts,
    fit_and_forecast_arfima_0d0_holdout,
    fit_arfima_0d0_whittle,
    forecast_arfima_0d0_one_step,
    fractional_differencing_weights,
)


class FractionalDifferencingWeightTests(
    unittest.TestCase
):
    def test_computes_known_fractional_weights(
        self,
    ) -> None:
        result = fractional_differencing_weights(
            fractional_parameter=0.25,
            count=5,
        )

        expected = np.array(
            [
                1.0,
                -0.25,
                -0.09375,
                -0.0546875,
                -0.03759765625,
            ]
        )

        np.testing.assert_allclose(
            result,
            expected,
            rtol=0.0,
            atol=1e-15,
        )


class ArfimaForecastTests(unittest.TestCase):
    def test_d_zero_forecasts_the_fixed_mean(
        self,
    ) -> None:
        observed_values = np.array(
            [10.0, 12.0, 9.0, 14.0, 11.0]
        )

        result = forecast_arfima_0d0_one_step(
            observed_values=observed_values,
            forecast_positions=np.array([1, 3, 4]),
            fractional_parameter=0.0,
            mean=11.0,
        )

        np.testing.assert_array_equal(
            result,
            np.array([11.0, 11.0, 11.0]),
        )

    def test_forecast_uses_only_values_before_target(
        self,
    ) -> None:
        first_series = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        changed_target = first_series.copy()
        changed_target[4] = 999.0

        first_forecast = forecast_arfima_0d0_one_step(
            observed_values=first_series,
            forecast_positions=np.array([4]),
            fractional_parameter=0.2,
            mean=0.0,
        )
        second_forecast = forecast_arfima_0d0_one_step(
            observed_values=changed_target,
            forecast_positions=np.array([4]),
            fractional_parameter=0.2,
            mean=0.0,
        )

        np.testing.assert_array_equal(
            first_forecast,
            second_forecast,
        )

    def test_rejects_forecast_position_without_target(
        self,
    ) -> None:
        observed_values = np.array(
            [1.0, 2.0, 3.0]
        )

        with self.assertRaisesRegex(
            ValueError,
            "outside the observed series",
        ):
            forecast_arfima_0d0_one_step(
                observed_values=observed_values,
                forecast_positions=np.array([3]),
                fractional_parameter=0.2,
                mean=0.0,
            )


class WhittleEstimationTests(unittest.TestCase):
    def test_recovers_simulated_fractional_parameter(
        self,
    ) -> None:
        random_generator = np.random.default_rng(
            20260729
        )

        observation_count = 4096
        true_fractional_parameter = 0.2

        integration_weights = np.empty(
            observation_count,
            dtype=float,
        )
        integration_weights[0] = 1.0

        for lag in range(1, observation_count):
            integration_weights[lag] = (
                integration_weights[lag - 1]
                * (
                    lag
                    - 1
                    + true_fractional_parameter
                )
                / lag
            )

        innovations = random_generator.normal(
            size=observation_count
        )
        simulated_values = np.convolve(
            innovations,
            integration_weights,
            mode="full",
        )[:observation_count]

        result = fit_arfima_0d0_whittle(
            simulated_values
        )

        self.assertLess(
            abs(
                result["d"]
                - true_fractional_parameter
            ),
            0.03,
        )
        self.assertAlmostEqual(
            result["mean"],
            float(np.mean(simulated_values)),
        )
        self.assertGreater(
            result["innovation_variance"],
            0.0,
        )
        self.assertFalse(
            result["boundary_warning"]
        )

    def test_rejects_constant_series(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "unidentified for a constant series",
        ):
            fit_arfima_0d0_whittle(
                np.ones(100)
            )

    def test_reproduces_frozen_deterministic_estimate(
        self,
    ) -> None:
        values = np.random.default_rng(
            20260729
        ).normal(size=64)

        result = fit_arfima_0d0_whittle(values)

        self.assertAlmostEqual(
            result["mean"],
            -0.0592017245561356,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            result["d"],
            0.04383164360895035,
            delta=1e-10,
        )
        self.assertAlmostEqual(
            result["innovation_variance"],
            1.056855276005954,
            delta=1e-10,
        )
        self.assertAlmostEqual(
            result["profile_whittle_objective"],
            1.5623222150713483,
            delta=1e-10,
        )
        self.assertAlmostEqual(
            result["grid_best_d"],
            0.044,
            delta=1e-15,
        )


class ArfimaHoldoutWorkflowTests(unittest.TestCase):
    def test_fits_model_exclusively_on_training_period(
        self,
    ) -> None:
        random_generator = np.random.default_rng(
            20260729
        )

        training_values = random_generator.normal(
            size=64
        )
        first_holdout = np.array([1.0, 2.0, 3.0])
        changed_holdout = np.array(
            [1000.0, -2000.0, 3000.0]
        )

        index = pd.date_range(
            "2024-01-01",
            periods=67,
            freq="30min",
            name="timestamp",
        )

        first_series = pd.Series(
            np.concatenate(
                [training_values, first_holdout]
            ),
            index=index,
        )
        changed_series = pd.Series(
            np.concatenate(
                [training_values, changed_holdout]
            ),
            index=index,
        )

        first_forecasts, first_model = (
            fit_and_forecast_arfima_0d0_holdout(
                first_series,
                training_observation_count=64,
            )
        )
        changed_forecasts, changed_model = (
            fit_and_forecast_arfima_0d0_holdout(
                changed_series,
                training_observation_count=64,
            )
        )

        self.assertEqual(first_model, changed_model)

        pd.testing.assert_index_equal(
            first_forecasts.index,
            index[64:],
        )

        self.assertEqual(
            first_forecasts.iloc[0],
            changed_forecasts.iloc[0],
        )

    def test_matches_low_level_holdout_forecasts(
        self,
    ) -> None:
        index = pd.date_range(
            "2024-01-01",
            periods=36,
            freq="30min",
            name="timestamp",
        )
        series = pd.Series(
            np.sin(np.arange(36) / 3.0),
            index=index,
        )

        forecasts, fitted_model = (
            fit_and_forecast_arfima_0d0_holdout(
                series,
                training_observation_count=32,
            )
        )

        expected = forecast_arfima_0d0_one_step(
            observed_values=series.to_numpy(),
            forecast_positions=np.arange(32, 36),
            fractional_parameter=float(
                fitted_model["d"]
            ),
            mean=float(fitted_model["mean"]),
        )

        np.testing.assert_allclose(
            forecasts.to_numpy(),
            expected,
            rtol=0.0,
            atol=0.0,
        )


class DominantIndicatorCovarianceForecastTests(
    unittest.TestCase
):
    def test_reconstructs_forecasts_in_reference_basis(
        self,
    ) -> None:
        inverse_square_root_two = 1.0 / np.sqrt(2.0)

        assets = pd.Index(
            ["asset_1", "asset_2"],
            name="asset",
        )

        reference_basis = pd.DataFrame(
            [
                [
                    inverse_square_root_two,
                    -inverse_square_root_two,
                ],
                [
                    inverse_square_root_two,
                    inverse_square_root_two,
                ],
            ],
            index=assets,
            columns=["component_1", "component_2"],
        )
        reference_eigenvalues = pd.Series(
            [5.0, 2.0],
            index=reference_basis.columns,
            name="eigenvalue",
        )

        timestamps = pd.date_range(
            "2024-02-01",
            periods=2,
            freq="30min",
            name="timestamp",
        )
        indicator_forecasts = pd.Series(
            [7.0, 11.0],
            index=timestamps,
            name="arfima_0d0_forecast",
        )

        result = (
            construct_dominant_indicator_covariance_forecasts(
                dominant_indicator_forecasts=(
                    indicator_forecasts
                ),
                reference_basis=reference_basis,
                reference_eigenvalues=(
                    reference_eigenvalues
                ),
            )
        )

        first_expected = pd.DataFrame(
            [
                [4.5, 2.5],
                [2.5, 4.5],
            ],
            index=reference_basis.index,
            columns=reference_basis.index,
        )
        second_expected = pd.DataFrame(
            [
                [6.5, 4.5],
                [4.5, 6.5],
            ],
            index=reference_basis.index,
            columns=reference_basis.index,
        )

        pd.testing.assert_frame_equal(
            result.xs(
                timestamps[0],
                level="timestamp",
            ),
            first_expected,
        )
        pd.testing.assert_frame_equal(
            result.xs(
                timestamps[1],
                level="timestamp",
            ),
            second_expected,
        )

    def test_does_not_hide_negative_indicator_forecast(
        self,
    ) -> None:
        reference_basis = pd.DataFrame(
            np.eye(2),
            index=pd.Index(
                ["asset_1", "asset_2"],
                name="asset",
            ),
            columns=["component_1", "component_2"],
        )
        reference_eigenvalues = pd.Series(
            [3.0, 2.0],
            index=reference_basis.columns,
            name="eigenvalue",
        )

        timestamp = pd.Timestamp("2024-02-01")
        indicator_forecasts = pd.Series(
            [-1.0],
            index=pd.Index(
                [timestamp],
                name="timestamp",
            ),
        )

        result = (
            construct_dominant_indicator_covariance_forecasts(
                dominant_indicator_forecasts=(
                    indicator_forecasts
                ),
                reference_basis=reference_basis,
                reference_eigenvalues=(
                    reference_eigenvalues
                ),
            )
        )

        np.testing.assert_array_equal(
            result.xs(
                timestamp,
                level="timestamp",
            ).to_numpy(),
            np.diag([-1.0, 2.0]),
        )


class DirectNaiveCovarianceForecastTests(
    unittest.TestCase
):
    def test_uses_previous_matrix_for_each_holdout_target(
        self,
    ) -> None:
        timestamps = pd.date_range(
            "2024-01-01",
            periods=4,
            freq="30min",
            name="timestamp",
        )
        assets = pd.Index(
            ["asset_1", "asset_2"],
            name="asset",
        )

        matrices = [
            pd.DataFrame(
                [
                    [value, value / 10.0],
                    [value / 10.0, value + 1.0],
                ],
                index=assets,
                columns=assets,
            )
            for value in [1.0, 2.0, 3.0, 4.0]
        ]

        covariance_matrices = pd.concat(
            matrices,
            keys=timestamps,
            names=["timestamp", "asset"],
        )

        result = (
            construct_direct_naive_covariance_forecasts(
                covariance_matrices=covariance_matrices,
                training_observation_count=2,
            )
        )

        expected = pd.concat(
            [matrices[1], matrices[2]],
            keys=timestamps[2:],
            names=["timestamp", "asset"],
        )

        pd.testing.assert_frame_equal(
            result,
            expected,
        )

    def test_does_not_use_current_target_matrix(
        self,
    ) -> None:
        timestamps = pd.date_range(
            "2024-01-01",
            periods=3,
            freq="30min",
            name="timestamp",
        )
        assets = pd.Index(
            ["asset_1", "asset_2"],
            name="asset",
        )

        matrices = [
            pd.DataFrame(
                np.eye(2) * value,
                index=assets,
                columns=assets,
            )
            for value in [1.0, 2.0, 3.0]
        ]

        changed_matrices = [
            matrix.copy()
            for matrix in matrices
        ]
        changed_matrices[2] *= 1000.0

        first_covariances = pd.concat(
            matrices,
            keys=timestamps,
            names=["timestamp", "asset"],
        )
        changed_covariances = pd.concat(
            changed_matrices,
            keys=timestamps,
            names=["timestamp", "asset"],
        )

        first_result = (
            construct_direct_naive_covariance_forecasts(
                covariance_matrices=first_covariances,
                training_observation_count=2,
            )
        )
        changed_result = (
            construct_direct_naive_covariance_forecasts(
                covariance_matrices=changed_covariances,
                training_observation_count=2,
            )
        )

        pd.testing.assert_frame_equal(
            first_result,
            changed_result,
        )

    def test_rejects_invalid_training_size(
        self,
    ) -> None:
        timestamps = pd.date_range(
            "2024-01-01",
            periods=2,
            freq="30min",
            name="timestamp",
        )
        assets = pd.Index(
            ["asset_1", "asset_2"],
            name="asset",
        )
        covariance_matrices = pd.concat(
            [
                pd.DataFrame(
                    np.eye(2),
                    index=assets,
                    columns=assets,
                )
                for _ in timestamps
            ],
            keys=timestamps,
            names=["timestamp", "asset"],
        )

        for training_observation_count in [0, 2]:
            with self.subTest(
                training_observation_count=(
                    training_observation_count
                )
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "strictly between zero",
                ):
                    construct_direct_naive_covariance_forecasts(
                        covariance_matrices=(
                            covariance_matrices
                        ),
                        training_observation_count=(
                            training_observation_count
                        ),
                    )


class ArfimaForecastTests(unittest.TestCase):

    def test_nonzero_d_matches_manual_recursion(
        self,
    ) -> None:
        result = forecast_arfima_0d0_one_step(
            observed_values=np.array(
                [10.0, 12.0, 9.0, 14.0, 11.0]
            ),
            forecast_positions=np.array([4]),
            fractional_parameter=0.25,
            mean=10.0,
        )

        self.assertAlmostEqual(
            result[0],
            11.015625,
        )

    def test_later_forecast_uses_newly_observed_value(
        self,
    ) -> None:
        original_values = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        changed_values = original_values.copy()
        changed_values[4] = 999.0

        original_forecasts = (
            forecast_arfima_0d0_one_step(
                observed_values=original_values,
                forecast_positions=np.array([4, 5]),
                fractional_parameter=0.2,
                mean=0.0,
            )
        )
        changed_forecasts = (
            forecast_arfima_0d0_one_step(
                observed_values=changed_values,
                forecast_positions=np.array([4, 5]),
                fractional_parameter=0.2,
                mean=0.0,
            )
        )

        self.assertEqual(
            original_forecasts[0],
            changed_forecasts[0],
        )
        self.assertNotEqual(
            original_forecasts[1],
            changed_forecasts[1],
        )


if __name__ == "__main__":
    unittest.main()
