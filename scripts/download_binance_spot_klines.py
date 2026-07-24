"""Download the Binance Spot kline archives used in the empirical study."""

from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
INTERVAL = "1m"
MONTHS = ("2024-01", "2024-02", "2024-03")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "binance"


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOLS:
        for month in MONTHS:
            archive_name = f"{symbol}-{INTERVAL}-{month}.zip"

            for filename in (archive_name, f"{archive_name}.CHECKSUM"):
                source_url = (
                    f"{BASE_URL}/{symbol}/{INTERVAL}/{filename}"
                )
                destination = OUTPUT_DIRECTORY / filename

                print(f"Downloading {filename}")
                urlretrieve(source_url, destination)


if __name__ == "__main__":
    main()
