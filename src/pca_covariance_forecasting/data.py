"""Load the validated market data used by the forecasting pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

from pca_covariance_forecasting.experiment_config import ExperimentConfig


def load_closing_prices(
    data_directory: Path,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Load configured monthly ZIP archives into one UTC price table."""
    closing_prices_by_symbol = {}

    for symbol in config.symbols:
        monthly_closing_prices = []

        for month in config.months:
            archive_path = (
                data_directory
                / f"{symbol}-{config.interval}-{month}.zip"
            )

            month_data = pd.read_csv(
                archive_path,
                header=None,
                usecols=[0, 4],
            )
            month_data.columns = ["open_time", "close"]

            month_data["open_time"] = pd.to_datetime(
                month_data["open_time"],
                unit="ms",
                utc=True,
            )
            month_data = month_data.set_index("open_time")

            monthly_closing_prices.append(month_data["close"])

        closing_prices_by_symbol[symbol] = pd.concat(
            monthly_closing_prices
        )

    closing_prices = pd.DataFrame(closing_prices_by_symbol)
    closing_prices.index.name = "timestamp"

    return closing_prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute consecutive log returns and discard the initial missing row."""
    return np.log(prices).diff().iloc[1:]
