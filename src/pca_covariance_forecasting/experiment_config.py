"""Load the experiment settings shared by the data scripts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings that identify one reproducible data experiment."""

    symbols: tuple[str, ...]
    interval: str
    months: tuple[str, ...]
    block_size: int


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")

    return value


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load the tracked empirical-design settings from one YAML file.

    Symbols, months, sampling interval, and covariance-block length correspond
    to the explicit study-design choices described in the thesis methodology.
    """
    with path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    root = _require_mapping(raw_config, "The experiment configuration")
    data = _require_mapping(root.get("data"), "data")
    covariance = _require_mapping(root.get("covariance"), "covariance")

    symbols = tuple(data.get("symbols", ()))
    months = tuple(data.get("months", ()))
    interval = data.get("interval")
    block_size = covariance.get("block_size")

    if not symbols or not all(isinstance(symbol, str) for symbol in symbols):
        raise ValueError("data.symbols must contain one or more strings.")
    if not months or not all(isinstance(month, str) for month in months):
        raise ValueError("data.months must contain one or more strings.")
    if not isinstance(interval, str) or not interval:
        raise ValueError("data.interval must be a non-empty string.")
    if not isinstance(block_size, int) or block_size < 2:
        raise ValueError("covariance.block_size must be an integer of at least 2.")

    return ExperimentConfig(
        symbols=symbols,
        interval=interval,
        months=months,
        block_size=block_size,
    )
