# Current Status

Last updated: 2026-07-24

## Project phase

The codebase is being rebuilt from the ground up to reflect the current methodological direction and to make every implementation decision explicit and understandable.

The previous repository, `bachelor-thesis-volatility-forecasting`, remains unchanged as a reference for earlier implementations, tests, experiments, and decisions. Components from it are not transferred automatically.

The model-independent data acquisition, validation, loading, and log-return calculation components have been implemented. No covariance estimation, PCA transformation, residual analysis, or forecasting model has been implemented yet.

## Established methodological direction

Given a time series of covariance matrices

$$
\Sigma_1, \Sigma_2, \ldots, \Sigma_n,
$$

a historical reference covariance matrix is decomposed as

$$
\widetilde{\Sigma}
=
\widetilde{B}
\widetilde{\Lambda}
\widetilde{B}^{\top}.
$$

Each covariance matrix is represented in the fixed reference basis:

$$
\widetilde{\Lambda}_i
=
\widetilde{B}^{\top}
\Sigma_i
\widetilde{B}.
$$

The dominant diagonal element

$$
\widetilde{\lambda}_{i,11}
$$

is the central scalar indicator investigated in the thesis.

The main research question is whether this single indicator contains enough information to obtain useful covariance forecasts.

## Approximation and residual

A covariance approximation is reconstructed using the dominant indicator and the historical reference decomposition. The remaining approximation error is

$$
E_i
=
\Sigma_i-\widehat{\Sigma}_i.
$$

The residual matrices must be examined to determine whether their information is small enough to be neglected in forecasting.

The currently discussed residual decomposition is

$$
E_i
=
B_{E,i}\Theta_i B_{E,i}^{\top},
$$

where $B_{E,i}$ contains eigenvectors of $E_i$ and $\Theta_i$ contains its eigenvalues.

This replaces the representation in the initial PCA document that used the reference basis $\widetilde{B}$ for the residual decomposition. That earlier representation is not treated as the current specification.

## Open methodological questions

- The exact construction of the historical reference covariance matrix and reference basis must be specified without look-ahead bias.
- It remains to be decided whether the residuals are negligible or must be forecast explicitly.
- If residuals are forecast, the treatment or estimation of their changing eigenvector bases remains open.
- Because $E_i$ is not necessarily positive semidefinite, its decomposition must be interpreted as an eigendecomposition; its eigenvalues are not automatically variances.
- The exact FARIMA specification remains open.

## Forecasting clarification

The supervisor clarified that the intended forecasting approach is FARIMA with a potentially non-integer differencing parameter $d$, rather than a random-walk forecast.

Clarification questions concerning the exact use of FARIMA are still awaiting a response. Therefore, no concrete FARIMA model order, estimation procedure, or implementation is currently fixed.

Earlier AR and ARIMA specifications are retained only as historical context until the final forecasting specification has been clarified.

## Current implementation status

The repository currently contains:

- a Python 3.12.13 environment managed with `uv`,
- an installable `src` package,
- `pandas` as the first runtime dependency,
- a downloader for the fixed Q1 2024 Binance Spot dataset,
- a central validator for the downloaded raw data,
- a loader that combines the validated Q1 2024 closing-price series,
- a function that calculates one-minute log returns,
- an initial README,
- this knowledge base.

The fixed raw dataset comprises one-minute Spot klines for BTCUSDT, ETHUSDT, and BNBUSDT from January through March 2024.

The validator checks:

- the official SHA-256 checksums,
- the expected ZIP and CSV structure,
- twelve Kline columns per observation,
- complete monthly coverage,
- exact one-minute timestamp continuity,
- numeric, finite, and strictly positive OHLC prices.

After this central validation succeeds, subsequent pipeline stages may treat the raw input data as structurally valid. This validation establishes file integrity and suitability for the pipeline; it does not independently verify the economic accuracy of Binance market data.

The next implementation step is to define and implement the rolling covariance-matrix estimation from the one-minute log returns.
