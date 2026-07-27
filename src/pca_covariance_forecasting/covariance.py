"""Estimate covariance matrices from non-overlapping return blocks."""

import pandas as pd


BLOCK_SIZE = 30


def compute_block_covariances(
    returns: pd.DataFrame,
    block_size: int = BLOCK_SIZE,
) -> pd.DataFrame:
    number_of_complete_blocks = len(returns) // block_size

    covariance_matrices = []
    block_end_timestamps = []

    for block_index in range(number_of_complete_blocks):
        start = block_index * block_size
        stop = start + block_size
        block = returns.iloc[start:stop]

        covariance_matrices.append(block.cov(ddof=1))
        block_end_timestamps.append(block.index[-1])

    return pd.concat(
        covariance_matrices,
        keys=block_end_timestamps,
        names=["timestamp", "asset"],
    )
