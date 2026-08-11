"""Validate the downloaded Binance Spot kline archives for one experiment."""

import argparse
import calendar
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from pca_covariance_forecasting.experiment_config import (
    ExperimentConfig,
    load_experiment_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "binance"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "experiments" / "2024_full_year.yaml"
)

KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)
PRICE_COLUMNS = ("open", "high", "low", "close")
ONE_MINUTE_MS = 60_000


def verify_checksum(archive_path: Path) -> None:
    checksum_path = Path(f"{archive_path}.CHECKSUM")
    expected_hash, expected_filename = checksum_path.read_text().split()

    if expected_filename.lstrip("*") != archive_path.name:
        raise ValueError(f"Unexpected checksum filename: {expected_filename}")

    with archive_path.open("rb") as archive_file:
        actual_hash = hashlib.file_digest(archive_file, "sha256").hexdigest()

    if actual_hash != expected_hash:
        raise ValueError(f"Checksum mismatch: {archive_path.name}")


def validate_archive(
    symbol: str,
    month: str,
    config: ExperimentConfig,
) -> None:
    archive_name = f"{symbol}-{config.interval}-{month}.zip"
    csv_name = f"{symbol}-{config.interval}-{month}.csv"
    archive_path = DATA_DIRECTORY / archive_name

    verify_checksum(archive_path)

    with ZipFile(archive_path) as archive:
        if archive.namelist() != [csv_name]:
            raise ValueError(
                f"{archive_name} does not contain exactly {csv_name}"
            )

        with archive.open(csv_name) as csv_file:
            data = pd.read_csv(csv_file, header=None)

    if data.shape[1] != len(KLINE_COLUMNS):
        raise ValueError(
            f"{csv_name} has {data.shape[1]} columns instead of 12"
        )

    data.columns = KLINE_COLUMNS

    year, month_number = map(int, month.split("-"))
    expected_rows = (
        calendar.monthrange(year, month_number)[1] * 24 * 60
    )
    expected_first_time = int(
        datetime(
            year,
            month_number,
            1,
            tzinfo=timezone.utc,
        ).timestamp()
        * 1000
    )
    expected_last_time = (
        expected_first_time + (expected_rows - 1) * ONE_MINUTE_MS
    )

    if len(data) != expected_rows:
        raise ValueError(
            f"{csv_name} has {len(data)} rows instead of {expected_rows}"
        )

    open_times = pd.to_numeric(data["open_time"], errors="raise")

    if open_times.iloc[0] != expected_first_time:
        raise ValueError(f"Unexpected first timestamp in {csv_name}")

    if open_times.iloc[-1] != expected_last_time:
        raise ValueError(f"Unexpected last timestamp in {csv_name}")

    if not open_times.diff().iloc[1:].eq(ONE_MINUTE_MS).all():
        raise ValueError(f"Missing, duplicate, or unordered minute in {csv_name}")

    prices = data.loc[:, PRICE_COLUMNS].apply(
        pd.to_numeric,
        errors="raise",
    )

    if (
        prices.isna().any().any()
        or prices.isin([float("inf"), float("-inf")]).any().any()
        or prices.le(0).any().any()
    ):
        raise ValueError(f"Invalid price in {csv_name}")

    print(f"{archive_name}: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    config = load_experiment_config(arguments.config)

    for symbol in config.symbols:
        for month in config.months:
            validate_archive(symbol, month, config)

    print(
        f"All {len(config.symbols) * len(config.months)} archives "
        "passed validation."
    )


if __name__ == "__main__":
    main()
