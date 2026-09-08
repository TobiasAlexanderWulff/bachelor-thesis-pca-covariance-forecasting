"""Independent cross-check of the thesis ARFIMA(0,d,0) estimate.

This validation script deliberately reconstructs the training dominant-indicator
series from the public Binance archives without using the thesis data,
covariance, or PCA helper functions. It then compares the repository's fitted
profile-Whittle estimate against:

1. a separately coded implementation of the same profiled Whittle criterion,
   optimized directly with SciPy rather than the repository grid/refinement;
2. semiparametric GPH estimates over several low-frequency bandwidths; and
3. semiparametric local-Whittle estimates over the same bandwidths.

A GitHub Actions workflow on the validation branch additionally fits the exported
training series with CRAN package ``fracdiff`` (ARFIMA(0,d,0) approximate maximum
likelihood), providing an external implementation and a different likelihood
approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pca_covariance_forecasting.forecasting import fit_arfima_0d0_whittle


SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
MONTHS = tuple(f"2024-{month:02d}" for month in range(1, 8))
INTERVAL = "1m"
BLOCK_SIZE = 30
TRAINING_OBSERVATION_COUNT = 8_783
D_BOUNDS = (-0.49, 0.49)
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"


def archive_url(symbol: str, month: str) -> str:
    filename = f"{symbol}-{INTERVAL}-{month}.zip"
    return f"{BASE_URL}/{symbol}/{INTERVAL}/{filename}"


def download_one(symbol: str, month: str, directory: Path) -> Path:
    filename = f"{symbol}-{INTERVAL}-{month}.zip"
    destination = directory / filename
    checksum_destination = directory / f"{filename}.CHECKSUM"
    url = archive_url(symbol, month)

    if not destination.exists():
        urllib.request.urlretrieve(url, destination)
    if not checksum_destination.exists():
        urllib.request.urlretrieve(f"{url}.CHECKSUM", checksum_destination)

    expected = checksum_destination.read_text(encoding="utf-8").split()[0]
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest.lower() != expected.lower():
        raise RuntimeError(f"Checksum mismatch for {filename}")

    return destination


def download_archives(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(symbol, month) for symbol in SYMBOLS for month in MONTHS]
    with ThreadPoolExecutor(max_workers=6) as executor:
        paths = list(executor.map(lambda args: download_one(*args, directory), jobs))
    return paths


def load_close_series(path: Path) -> pd.Series:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one CSV in {path.name}, found {members}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, header=None, usecols=[0, 4])

    frame.columns = ["open_time", "close"]
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    return frame.set_index("open_time")["close"]


def reconstruct_training_indicator(archive_directory: Path) -> tuple[pd.Series, dict[str, object]]:
    download_archives(archive_directory)

    closes: dict[str, pd.Series] = {}
    for symbol in SYMBOLS:
        monthly = [
            load_close_series(archive_directory / f"{symbol}-{INTERVAL}-{month}.zip")
            for month in MONTHS
        ]
        closes[symbol] = pd.concat(monthly)

    prices = pd.DataFrame(closes)
    prices.index.name = "timestamp"
    if prices.isna().any().any():
        raise RuntimeError("Aligned closing-price panel contains missing values")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise RuntimeError("Closing-price index is not a unique increasing sequence")

    returns = np.log(prices).diff().iloc[1:]
    required_returns = TRAINING_OBSERVATION_COUNT * BLOCK_SIZE
    if len(returns) < required_returns:
        raise RuntimeError(
            f"Need {required_returns} returns for training, found {len(returns)}"
        )

    training_returns = returns.iloc[:required_returns]
    values = training_returns.to_numpy(dtype=float)
    blocks = values.reshape(TRAINING_OBSERVATION_COUNT, BLOCK_SIZE, len(SYMBOLS))
    centered = blocks - blocks.mean(axis=1, keepdims=True)
    covariances = np.einsum("bti,btj->bij", centered, centered) / (BLOCK_SIZE - 1)

    reference_covariance = covariances.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(reference_covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    leading_vector = eigenvectors[:, 0]

    indicator_values = np.einsum(
        "i,bij,j->b", leading_vector, covariances, leading_vector
    )
    block_end_positions = np.arange(BLOCK_SIZE - 1, required_returns, BLOCK_SIZE)
    block_end_timestamps = training_returns.index[block_end_positions]
    indicator = pd.Series(
        indicator_values,
        index=block_end_timestamps,
        name="dominant_indicator",
        dtype=float,
    )

    summary = {
        "symbols": list(SYMBOLS),
        "months_downloaded": list(MONTHS),
        "training_observations": int(len(indicator)),
        "training_first_timestamp": str(indicator.index[0]),
        "training_last_timestamp": str(indicator.index[-1]),
        "reference_eigenvalues": [float(value) for value in eigenvalues],
        "reference_variance_shares": [
            float(value / eigenvalues.sum()) for value in eigenvalues
        ],
        "indicator_mean": float(indicator.mean()),
        "indicator_std_ddof0": float(indicator.to_numpy().std(ddof=0)),
    }
    return indicator, summary


def independent_profile_whittle(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    n = len(centered)
    m = (n - 1) // 2
    frequencies = 2.0 * np.pi * np.arange(1, m + 1) / n
    fft = np.fft.rfft(centered)
    periodogram = np.abs(fft[1 : m + 1]) ** 2 / (2.0 * np.pi * n)
    log_factor = np.log(2.0 * np.sin(frequencies / 2.0))

    def objective(d: float) -> float:
        log_g = -2.0 * d * log_factor
        sigma2 = 2.0 * np.pi * np.mean(periodogram * np.exp(-log_g))
        return float(m * np.log(sigma2) + np.sum(log_g))

    result = minimize_scalar(
        objective,
        bounds=D_BOUNDS,
        method="bounded",
        options={"xatol": 1e-13, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Independent Whittle optimization failed: {result.message}")

    d_hat = float(result.x)
    log_g = -2.0 * d_hat * log_factor
    sigma2 = float(2.0 * np.pi * np.mean(periodogram * np.exp(-log_g)))
    return {
        "d": d_hat,
        "innovation_variance": sigma2,
        "objective": float(result.fun),
        "function_evaluations": int(result.nfev),
    }


def gph_estimate(values: np.ndarray, bandwidth_exp: float) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    n = len(centered)
    bandwidth = int(math.floor(n**bandwidth_exp))
    frequencies = 2.0 * np.pi * np.arange(1, bandwidth + 1) / n
    fft = np.fft.rfft(centered)
    periodogram = np.abs(fft[1 : bandwidth + 1]) ** 2 / (2.0 * np.pi * n)
    x = np.log(4.0 * np.sin(frequencies / 2.0) ** 2)
    y = np.log(periodogram)
    slope = np.polyfit(x, y, 1)[0]
    return float(-slope)


def local_whittle_estimate(values: np.ndarray, bandwidth_exp: float) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    n = len(centered)
    bandwidth = int(math.floor(n**bandwidth_exp))
    frequencies = 2.0 * np.pi * np.arange(1, bandwidth + 1) / n
    fft = np.fft.rfft(centered)
    periodogram = np.abs(fft[1 : bandwidth + 1]) ** 2 / (2.0 * np.pi * n)
    mean_log_frequency = float(np.mean(np.log(frequencies)))

    def objective(d: float) -> float:
        scale = float(np.mean((frequencies ** (2.0 * d)) * periodogram))
        return float(np.log(scale) - 2.0 * d * mean_log_frequency)

    result = minimize_scalar(
        objective,
        bounds=D_BOUNDS,
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Local-Whittle optimization failed: {result.message}")
    return float(result.x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("validation_output"))
    parser.add_argument("--archive-dir", type=Path, default=Path("validation_output/binance"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    indicator, data_summary = reconstruct_training_indicator(args.archive_dir)
    indicator_path = args.output_dir / "training_dominant_indicator.csv"
    indicator.rename_axis("timestamp").to_csv(indicator_path)

    thesis_fit = fit_arfima_0d0_whittle(indicator.to_numpy(dtype=float))
    independent_fit = independent_profile_whittle(indicator.to_numpy(dtype=float))

    bandwidth_exponents = (0.5, 0.6, 0.7)
    gph = {
        str(exp): gph_estimate(indicator.to_numpy(dtype=float), exp)
        for exp in bandwidth_exponents
    }
    local_whittle = {
        str(exp): local_whittle_estimate(indicator.to_numpy(dtype=float), exp)
        for exp in bandwidth_exponents
    }

    result = {
        "data": data_summary,
        "repository_profile_whittle": {
            "d": float(thesis_fit["d"]),
            "mean": float(thesis_fit["mean"]),
            "innovation_variance": float(thesis_fit["innovation_variance"]),
            "objective": float(thesis_fit["profile_whittle_objective"]),
        },
        "independent_same_objective_scipy": independent_fit,
        "absolute_d_difference_same_objective": abs(
            float(thesis_fit["d"]) - float(independent_fit["d"])
        ),
        "gph_bandwidth_exponent": gph,
        "local_whittle_bandwidth_exponent": local_whittle,
    }

    result_path = args.output_dir / "python_crosscheck.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote training series to {indicator_path}")
    print(f"Wrote Python cross-check to {result_path}")


if __name__ == "__main__":
    main()
