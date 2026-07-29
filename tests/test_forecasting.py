"""Tests for covariance-factor forecasting models."""

import unittest

import numpy as np

from pca_covariance_forecasting.forecasting import (
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


if __name__ == "__main__":
    unittest.main()
