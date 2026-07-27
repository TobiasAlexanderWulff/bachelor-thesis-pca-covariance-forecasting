"""Load the validated market data used by the forecasting pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd


SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
MONTHS = ("2024-01", "2024-02", "2024-03")
INTERVAL = "1m"


def load_closing_prices(data_directory: Path) -> pd.DataFrame:
    closing_prices_by_symbol = {}

    for symbol in SYMBOLS:
        monthly_closing_prices = []

        for month in MONTHS:
            archive_path = (
                data_directory
                / f"{symbol}-{INTERVAL}-{month}.zip"
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
    return np.log(prices).diff().iloc[1:]
