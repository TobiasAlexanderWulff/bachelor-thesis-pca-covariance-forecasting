# PCA Covariance Forecasting

Research code for a bachelor's thesis investigating PCA-based representations and forecasts of covariance matrices.

## Project status

This repository is being rebuilt from the ground up. Methodological decisions and implementation details are developed and documented incrementally.

## Development setup

The project requires Python 3.12 and uses `uv` for dependency and environment management.

```bash
uv sync
```

To run Python commands inside the project environment:

```bash
uv run python
```

## Reproducible analysis and thesis figures

Run the retained reduced-scope analysis with:

```bash
uv run python scripts/analyze_covariance_forecasts.py
```

Generate the thesis figure set and its supporting CSV tables with:

```bash
uv run python scripts/generate_thesis_figures.py
```

The figures are exported to `output/figures/` as PNG, PDF, and SVG files.
