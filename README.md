# PCA Covariance Forecasting

Research code for the bachelor thesis *PCA-based Forecasting of Portfolio
Volatility*. The retained experiment constructs 30-minute sample covariance
matrices for BTC, ETH, and BNB from one-minute observations during 2024. A fixed
PCA reference basis is estimated on the chronological training period, and the
dominant fixed-basis indicator is forecast with naive persistence and an
ARFIMA(0, d, 0) model. Direct covariance persistence is used as the matrix-level
benchmark.

## Reproducible environment

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required. The complete
dependency resolution is recorded in `uv.lock`.

```bash
uv sync --frozen
```

All commands below are run from the repository root with the locked environment.

## Data

The experiment configuration is stored in
`config/experiments/2024_full_year.yaml`. It identifies the symbols, interval,
months, official Binance archive base URL, and local data directory. Downloaded
archives and checksum sidecars are stored under `data/raw/binance/` and are
excluded from Git.

Download the exact monthly archives and their official checksum sidecars:

```bash
uv run python scripts/download_binance_spot_klines.py
```

Validate checksums, archive structure, schema, monthly calendar coverage,
60,000-ms timestamp continuity, and finite positive OHLC prices:

```bash
uv run python scripts/validate_binance_spot_klines.py
```

The retained configuration comprises 36 archives: three symbols multiplied by
twelve months. Analysis should be run only after all 36 archives pass validation.

## Analysis and thesis figures

Run the retained reduced-scope analysis:

```bash
uv run python scripts/analyze_covariance_forecasts.py
```

The command reports the chronological split, reference-PCA shares,
single-indicator approximation errors, Profile-Whittle estimate, covariance
RMSE comparison, and positive-semidefiniteness diagnostics.

Generate the complete thesis figure set and supporting CSV tables:

```bash
uv run python scripts/generate_thesis_figures.py
```

Figures are exported to `output/figures/` as PNG, PDF, and SVG files. Supporting
tables used to audit the plotted statements are written to
`output/figures/tables/`.

## Executable reproduction notebook

The executed companion notebook
`notebooks/reproduce_thesis_results.ipynb` reproduces the retained empirical
analysis in a reader-facing sequence. It imports and calls the data,
covariance, PCA, forecasting, and evaluation modules directly. Mathematical
implementation is not copied into notebook cells. The established archive
validator, plotting functions, and unit tests remain the technical control
path.

Create the locked notebook environment and open the notebook with:

```bash
uv sync --frozen --group notebook
uv run --group notebook jupyter lab notebooks/reproduce_thesis_results.ipynb
```

Execute and overwrite the stored outputs non-interactively with:

```bash
uv run --group notebook jupyter nbconvert \
  --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/reproduce_thesis_results.ipynb
```

The notebook stops if supplied-data validation, headline-result
reconciliation, figure generation, or the test suite fails. Its stored outputs
allow the verified results and figures to be reviewed without executing it.

## Tests

Run the unit tests with:

```bash
uv run python -m unittest discover -s tests
```

The tests cover the fractional-differencing weights, Profile-Whittle inputs and
optimization, expanding-history one-step forecasts, covariance reconstruction,
direct persistence benchmark, RMSE aggregation, loss calculations, and
positive-semidefiniteness diagnostics.

## Repository structure

```text
config/experiments/   Retained experiment settings
notebooks/             Executed reader-facing reproduction notebook
scripts/              Download, validation, analysis, and figure entry points
src/pca_covariance_forecasting/
                      Data, covariance, PCA, forecasting, and evaluation modules
tests/                Unit tests for the retained pipeline
output/figures/       Reproduced thesis figures and supporting tables
archive/              Exploratory analyses excluded from the retained comparison
uv.lock               Complete locked dependency resolution
```

The scripts and notebook import the package modules instead of duplicating the
analysis logic. The package modules and tests therefore remain the single
implementation and verification basis for both reproduction paths.
