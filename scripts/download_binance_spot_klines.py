"""Download the Binance Spot kline archives for one experiment."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from pca_covariance_forecasting.experiment_config import load_experiment_config


BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "binance"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "experiments" / "2024_full_year.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    config = load_experiment_config(arguments.config)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for symbol in config.symbols:
        for month in config.months:
            archive_name = f"{symbol}-{config.interval}-{month}.zip"

            for filename in (archive_name, f"{archive_name}.CHECKSUM"):
                source_url = (
                    f"{BASE_URL}/{symbol}/{config.interval}/{filename}"
                )
                destination = OUTPUT_DIRECTORY / filename

                print(f"Downloading {filename}")
                urlretrieve(source_url, destination)


if __name__ == "__main__":
    main()
