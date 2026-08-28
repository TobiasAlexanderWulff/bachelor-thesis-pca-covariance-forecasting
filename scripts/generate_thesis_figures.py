"""Generate the thesis-ready figures for the reduced analysis scope.

The figure set follows the approved evidence chain: data context, PCA
motivation, forecast target, approximation quality, benchmark performance,
and a descriptive time-path diagnostic. It deliberately excludes additional
time-varying residual models and inferential HAC or bootstrap results.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter, ScalarFormatter

from pca_covariance_forecasting.covariance import (
    compute_block_covariances,
)
from pca_covariance_forecasting.data import (
    compute_log_returns,
    load_closing_prices,
)
from pca_covariance_forecasting.evaluation import (
    compute_aggregated_relative_frobenius_error,
    compute_covariance_rmse,
    compute_covariance_squared_frobenius_losses,
    compute_psd_diagnostics,
    compute_relative_frobenius_errors,
)
from pca_covariance_forecasting.experiment_config import load_experiment_config
from pca_covariance_forecasting.forecasting import (
    construct_direct_naive_covariance_forecasts,
    construct_dominant_indicator_covariance_forecasts,
    fit_and_forecast_arfima_0d0_holdout,
)
from pca_covariance_forecasting.pca import (
    compute_approximation_errors,
    compute_reference_covariance,
    compute_reference_eigendecomposition,
    construct_reference_basis_approximations,
    transform_covariances_from_reference_basis,
    transform_covariances_to_reference_basis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "binance"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output" / "figures"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "experiments" / "2024_full_year.yaml"
)

TRAINING_OBSERVATION_COUNT = 8_783

DIRECT_NAIVE = "direct_naive_covariance"
NAIVE_INDICATOR = "naive_dominant_indicator"
ARFIMA_INDICATOR = "arfima_dominant_indicator"

METHOD_LABELS = {
    DIRECT_NAIVE: "Direct naive covariance",
    NAIVE_INDICATOR: "Naive dominant indicator",
    ARFIMA_INDICATOR: "ARFIMA dominant indicator",
}

INK = "#202833"
MUTED = "#5D6875"
GRID = "#D8DEE6"
BLUE = "#2F628F"
BLUE_MID = "#7FA6C9"
BLUE_LIGHT = "#C8D9E8"
ORANGE = "#C36A2D"
GOLD = "#A77A24"
OLIVE = "#71834A"
NEUTRAL = "#9AA5B1"

ASSET_COLORS = {
    "BTCUSDT": BLUE,
    "ETHUSDT": ORANGE,
    "BNBUSDT": OLIVE,
}

FIGURE_CONTRACTS = {
    "01_normalized_closing_prices": (
        "Data context: compare the three assets over 2024 without "
        "letting their different price levels determine the visual scale."
    ),
    "02_reference_pca_variance_share": (
        "PCA motivation: show how much of the training-reference total "
        "variance is associated with each component."
    ),
    "03_dominant_indicator_time_series": (
        "Forecast target: show the dominant fixed-basis indicator and the "
        "look-ahead-safe training/holdout boundary."
    ),
    "04_approximation_error_distributions": (
        "Representation quality: compare the full interval-level relative "
        "Frobenius-error distributions in training and holdout."
    ),
    "05_covariance_rmse_reduction": (
        "Benchmark performance: compare descriptive RMSE changes against "
        "the direct naive covariance forecast."
    ),
    "06_cumulative_loss_difference": (
        "Temporal diagnostic: show when the aggregate holdout advantage "
        "over the direct naive covariance forecast accumulated."
    ),
    "07_cumulative_indicator_loss_difference": (
        "Temporal diagnostic: show when the aggregate holdout advantage "
        "of ARFIMA over the naive dominant-indicator forecast accumulated."
    ),
}


def configure_plot_style() -> None:
    """Set a restrained, consistent style for all thesis figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.axisbelow": True,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def select_covariances(
    covariance_matrices: pd.DataFrame,
    timestamps: pd.Index,
) -> pd.DataFrame:
    """Select matrices at the requested timestamps and verify alignment."""
    matrix_timestamps = covariance_matrices.index.get_level_values(
        "timestamp"
    )
    selected = covariance_matrices.loc[
        matrix_timestamps.isin(timestamps)
    ].copy()
    selected_timestamps = (
        selected.index.get_level_values("timestamp").unique()
    )

    if not selected_timestamps.equals(timestamps):
        raise ValueError(
            "The selected covariance timestamps do not match the request."
        )

    return selected


def add_subtitle(ax: Axes, text: str) -> None:
    """Add a consistent subtitle directly below the main title."""
    ax.text(
        0.0,
        1.015,
        text,
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.0,
        ha="left",
        va="bottom",
    )


def add_source_note(fig: Figure, text: str) -> None:
    """Add a compact source or interpretation note below a figure."""
    layout_engine = fig.get_layout_engine()

    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.07, 1.0, 1.0))

    fig.text(
        0.01,
        0.012,
        text,
        color=MUTED,
        fontsize=7.5,
        ha="left",
        va="bottom",
    )


def style_time_axis(ax: Axes) -> None:
    """Use concise, readable UTC date labels."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(
    fig: Figure,
    output_directory: Path,
    stem: str,
) -> list[Path]:
    """Save one figure as a high-resolution raster and two vector files."""
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for extension in ("png", "pdf", "svg"):
        output_path = output_directory / f"{stem}.{extension}"
        save_arguments = {"format": extension}

        if extension == "png":
            save_arguments["dpi"] = 300

        fig.savefig(output_path, **save_arguments)

        if extension == "svg":
            svg_text = output_path.read_text(encoding="utf-8")
            normalized_svg = "\n".join(
                line.rstrip() for line in svg_text.splitlines()
            ) + "\n"
            output_path.write_text(normalized_svg, encoding="utf-8")

        saved_paths.append(output_path)

    plt.close(fig)
    return saved_paths


def plot_normalized_closing_prices(
    closing_prices: pd.DataFrame,
    holdout_start: pd.Timestamp,
    period_label: str,
    output_directory: Path,
) -> list[Path]:
    """Plot daily normalized closing-price paths for dataset context."""
    daily_closes = closing_prices.resample("1D").last()
    normalized = daily_closes.divide(daily_closes.iloc[0]).multiply(100.0)

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)

    for asset in normalized.columns:
        ax.plot(
            normalized.index,
            normalized[asset],
            color=ASSET_COLORS[asset],
            linewidth=1.8,
            label=asset.removesuffix("USDT"),
        )

    ax.axvline(
        holdout_start,
        color=INK,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        label="Training/holdout boundary",
    )
    ax.set_title("Normalized closing-price paths", loc="left", pad=28)
    add_subtitle(
        ax,
        "Daily last close; index = 100 at the start of the configured period",
    )
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Normalized price index")
    ax.legend(ncols=4, loc="upper left")
    ax.margins(x=0.01)
    style_time_axis(ax)
    add_source_note(
        fig,
        f"Source: Binance Spot one-minute klines, {period_label}. "
        "Daily sampling is used only for visual context.",
    )

    return save_figure(
        fig,
        output_directory,
        "01_normalized_closing_prices",
    )


def plot_reference_pca_variance_share(
    reference_eigenvalues: pd.Series,
    output_directory: Path,
) -> list[Path]:
    """Plot variance shares of the fixed training-reference components."""
    shares = (
        reference_eigenvalues
        .divide(reference_eigenvalues.sum())
        .multiply(100.0)
    )
    labels = [
        f"PC{position}"
        for position in range(1, len(shares) + 1)
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    bars = ax.bar(
        labels,
        shares.to_numpy(),
        color=[BLUE, BLUE_MID, BLUE_LIGHT],
        edgecolor=INK,
        linewidth=0.7,
        width=0.62,
    )

    ax.bar_label(
        bars,
        labels=[f"{value:.2f}%" for value in shares],
        padding=4,
        fontsize=10.0,
    )
    ax.set_title(
        "Variance shares of the reference PCA components",
        loc="left",
        pad=28,
    )
    add_subtitle(
        ax,
        "Fixed reference covariance estimated from 8,783 training matrices",
    )
    ax.set_xlabel("Reference component")
    ax.set_ylabel("Share of total variance (%)")
    ax.set_ylim(0.0, 100.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    add_source_note(
        fig,
        "Source: training reference covariance; components ordered by "
        "decreasing eigenvalue.",
    )

    return save_figure(
        fig,
        output_directory,
        "02_reference_pca_variance_share",
    )


def plot_dominant_indicator_time_series(
    dominant_indicator: pd.Series,
    holdout_start: pd.Timestamp,
    output_directory: Path,
) -> list[Path]:
    """Plot the central scalar indicator over the full study period."""
    scaled_indicator = dominant_indicator.multiply(1e6)

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.plot(
        scaled_indicator.index,
        scaled_indicator,
        color=BLUE,
        linewidth=0.85,
        label=r"$\widetilde{\lambda}_{t,11}$",
    )
    ax.axvline(
        holdout_start,
        color=ORANGE,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        label="Holdout start",
    )
    ax.set_title("Dominant fixed-basis indicator", loc="left", pad=28)
    add_subtitle(
        ax,
        "One value per non-overlapping 30-minute covariance block",
    )
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(r"Dominant indicator ($\times 10^{-6}$)")
    ax.set_ylim(bottom=0.0)
    ax.legend(ncols=2, loc="upper left")
    ax.margins(x=0.01)
    style_time_axis(ax)
    add_source_note(
        fig,
        "The PCA basis is estimated from the training period only and "
        "remains fixed in the holdout.",
    )

    return save_figure(
        fig,
        output_directory,
        "03_dominant_indicator_time_series",
    )


def empirical_cdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted values and cumulative shares for an empirical CDF."""
    sorted_values = np.sort(values.to_numpy(dtype=float))

    if len(sorted_values) == 0:
        raise ValueError("An empirical CDF requires at least one value.")
    if not np.isfinite(sorted_values).all():
        raise ValueError("The empirical CDF contains non-finite values.")
    if np.any(sorted_values <= 0.0):
        raise ValueError(
            "The log-scaled empirical CDF requires positive values."
        )

    cumulative_share = (
        np.arange(1, len(sorted_values) + 1, dtype=float)
        / len(sorted_values)
        * 100.0
    )
    return sorted_values * 100.0, cumulative_share


def plot_approximation_error_distributions(
    training_errors: pd.Series,
    holdout_errors: pd.Series,
    aggregated_training_error: float,
    aggregated_holdout_error: float,
    output_directory: Path,
) -> list[Path]:
    """Plot full empirical distributions of interval approximation error."""
    training_x, training_y = empirical_cdf(training_errors)
    holdout_x, holdout_y = empirical_cdf(holdout_errors)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax.step(
        training_x,
        training_y,
        where="post",
        color=NEUTRAL,
        linewidth=1.7,
        linestyle=(0, (5, 3)),
        label="Training",
    )
    ax.step(
        holdout_x,
        holdout_y,
        where="post",
        color=BLUE,
        linewidth=1.8,
        label="Holdout",
    )
    ax.axvline(
        100.0,
        color=INK,
        linewidth=0.9,
        linestyle=":",
        label="Error norm = covariance norm",
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_title(
        "Distribution of interval-level approximation errors",
        loc="left",
        pad=28,
    )
    add_subtitle(
        ax,
        "Relative Frobenius error; logarithmic horizontal axis",
    )
    ax.set_xlabel("Relative Frobenius error (%)")
    ax.set_ylabel("Cumulative share of intervals (%)")
    ax.set_ylim(0.0, 100.5)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.grid(color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")

    summary_text = (
        "Median / 95th percentile\n"
        f"Training: {training_errors.median() * 100:.1f}% / "
        f"{training_errors.quantile(0.95) * 100:.1f}%\n"
        f"Holdout: {holdout_errors.median() * 100:.1f}% / "
        f"{holdout_errors.quantile(0.95) * 100:.1f}%"
    )
    ax.text(
        0.02,
        0.96,
        summary_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": GRID,
            "linewidth": 0.8,
        },
    )
    add_source_note(
        fig,
        "Aggregated relative Frobenius error: "
        f"training {aggregated_training_error * 100:.1f}%, "
        f"holdout {aggregated_holdout_error * 100:.1f}%. "
        "The aggregated and interval-level summaries use different weighting.",
    )

    return save_figure(
        fig,
        output_directory,
        "04_approximation_error_distributions",
    )


def plot_covariance_rmse_reduction(
    rmse_by_method: pd.DataFrame,
    holdout_count: int,
    output_directory: Path,
) -> list[Path]:
    """Plot signed RMSE reductions relative to the direct naive benchmark."""
    components = ["overall", "diagonal", "off-diagonal"]
    component_labels = ["Overall", "Diagonal", "Off-diagonal"]
    methods = [NAIVE_INDICATOR, ARFIMA_INDICATOR]

    benchmark = rmse_by_method.loc[DIRECT_NAIVE, components]
    reductions = (
        1.0
        - rmse_by_method.loc[methods, components].divide(benchmark)
    ).multiply(100.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.1), constrained_layout=True)
    y_positions = np.arange(len(components), dtype=float)
    bar_height = 0.34

    for offset, method, color, hatch in (
        (-bar_height / 2, NAIVE_INDICATOR, GOLD, "//"),
        (bar_height / 2, ARFIMA_INDICATOR, BLUE, ".."),
    ):
        values = reductions.loc[method, components].to_numpy()
        bars = ax.barh(
            y_positions + offset,
            values,
            height=bar_height,
            color=color,
            edgecolor=INK,
            linewidth=0.7,
            hatch=hatch,
            label=METHOD_LABELS[method],
        )

        for bar, value in zip(bars, values, strict=True):
            horizontal_offset = 0.35 if value >= 0.0 else -0.35
            ax.text(
                value + horizontal_offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.1f}%",
                ha="left" if value >= 0.0 else "right",
                va="center",
                fontsize=8.5,
            )

    ax.axvline(0.0, color=INK, linewidth=1.0)
    minimum = float(reductions.to_numpy().min())
    maximum = float(reductions.to_numpy().max())
    ax.set_xlim(minimum - 2.5, maximum + 3.0)
    ax.set_yticks(y_positions, component_labels)
    ax.set_ylim(len(components) - 0.5, -0.8)
    ax.set_title(
        "RMSE reduction relative to the direct naive forecast",
        loc="left",
        pad=28,
    )
    add_subtitle(
        ax,
        "Positive values indicate lower RMSE; fixed holdout with "
        f"{holdout_count:,} matrices",
    )
    ax.set_xlabel("RMSE reduction (%)")
    ax.set_ylabel("Error category")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", ncols=2)
    add_source_note(
        fig,
        "Benchmark: direct one-step covariance carry-forward. "
        "Values are descriptive for the investigated asset universe and period.",
    )

    return save_figure(
        fig,
        output_directory,
        "05_covariance_rmse_reduction",
    )


def plot_cumulative_loss_difference(
    benchmark_losses: pd.Series,
    candidate_losses: pd.Series,
    benchmark_label: str,
    candidate_label: str,
    output_stem: str,
    output_directory: Path,
) -> list[Path]:
    """Plot cumulative benchmark-minus-candidate squared loss."""
    if not benchmark_losses.index.equals(candidate_losses.index):
        raise ValueError("Loss series must use identical timestamps.")

    loss_difference = benchmark_losses - candidate_losses
    cumulative_difference = loss_difference.cumsum().divide(1e-8)
    candidate_win_count = int(loss_difference.gt(0.0).sum())
    observation_count = len(loss_difference)
    candidate_win_share = candidate_win_count / observation_count * 100.0

    fig, ax = plt.subplots(figsize=(9.2, 4.9), constrained_layout=True)
    ax.plot(
        cumulative_difference.index,
        cumulative_difference,
        color=BLUE,
        linewidth=1.4,
    )
    ax.axhline(0.0, color=INK, linewidth=0.9)
    final_value = float(cumulative_difference.iloc[-1])
    ax.scatter(
        [cumulative_difference.index[-1]],
        [final_value],
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.6,
        zorder=3,
    )
    ax.annotate(
        f"Final cumulative difference: {final_value:.2f}",
        xy=(cumulative_difference.index[-1], final_value),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.5,
    )
    ax.text(
        0.015,
        0.96,
        f"{candidate_label} lower loss at "
        f"{candidate_win_count:,}/{observation_count:,} timestamps "
        f"({candidate_win_share:.1f}%)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID,
            "linewidth": 0.8,
        },
    )
    ax.set_title(
        "Cumulative difference in squared Frobenius loss",
        loc="left",
        pad=28,
    )
    add_subtitle(
        ax,
        f"{benchmark_label} loss minus {candidate_label} loss; "
        f"positive values favor {candidate_label}",
    )
    ax.set_xlabel("Holdout date (UTC)")
    ax.set_ylabel(r"Cumulative loss difference ($\times 10^{-8}$)")
    ax.margins(x=0.01)
    style_time_axis(ax)
    add_source_note(
        fig,
        "Descriptive holdout diagnostic only; it is not a significance "
        "test and does not establish a causal explanation.",
    )

    return save_figure(
        fig,
        output_directory,
        output_stem,
    )


def save_supporting_tables(
    output_directory: Path,
    reference_eigenvalues: pd.Series,
    approximation_summary: pd.DataFrame,
    rmse_by_method: pd.DataFrame,
    loss_difference: pd.Series,
    psd_summary: pd.DataFrame,
) -> None:
    """Save compact source tables used to audit the plotted statements."""
    table_directory = output_directory / "tables"
    table_directory.mkdir(parents=True, exist_ok=True)

    pca_summary = reference_eigenvalues.to_frame(name="eigenvalue")
    pca_summary["variance_share_percent"] = (
        reference_eigenvalues
        .divide(reference_eigenvalues.sum())
        .multiply(100.0)
    )
    pca_summary.to_csv(table_directory / "reference_pca_summary.csv")
    approximation_summary.to_csv(
        table_directory / "approximation_error_summary.csv"
    )
    rmse_by_method.rename(index=METHOD_LABELS).to_csv(
        table_directory / "covariance_rmse_summary.csv"
    )
    loss_difference.rename("direct_naive_minus_arfima_loss").to_csv(
        table_directory / "timestamp_loss_differences.csv"
    )
    psd_summary.rename(index=METHOD_LABELS).to_csv(
        table_directory / "forecast_psd_summary.csv"
    )


def main() -> None:
    """Reproduce the reduced analysis and export the approved figure set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    config = load_experiment_config(arguments.config)

    configure_plot_style()

    closing_prices = load_closing_prices(DATA_DIRECTORY, config)
    log_returns = compute_log_returns(closing_prices)
    covariance_matrices = compute_block_covariances(
        log_returns,
        block_size=config.block_size,
    )
    covariance_timestamps = (
        covariance_matrices.index.get_level_values("timestamp").unique()
    )

    if len(covariance_timestamps) <= TRAINING_OBSERVATION_COUNT:
        raise ValueError(
            "The configured data period does not contain enough covariance "
            "matrices for the fixed training split."
        )

    training_timestamps = covariance_timestamps[
        :TRAINING_OBSERVATION_COUNT
    ]
    holdout_timestamps = covariance_timestamps[
        TRAINING_OBSERVATION_COUNT:
    ]
    training_covariances = select_covariances(
        covariance_matrices,
        training_timestamps,
    )
    holdout_covariances = select_covariances(
        covariance_matrices,
        holdout_timestamps,
    )

    reference_covariance = compute_reference_covariance(
        training_covariances
    )
    reference_basis, reference_eigenvalues = (
        compute_reference_eigendecomposition(reference_covariance)
    )
    transformed_covariances = transform_covariances_to_reference_basis(
        covariances=covariance_matrices,
        reference_basis=reference_basis,
    )
    approximation_in_reference_basis = (
        construct_reference_basis_approximations(
            transformed_covariances=transformed_covariances,
            reference_eigenvalues=reference_eigenvalues,
        )
    )
    covariance_approximations = transform_covariances_from_reference_basis(
        transformed_covariances=approximation_in_reference_basis,
        reference_basis=reference_basis,
    )
    approximation_errors = compute_approximation_errors(
        covariances=covariance_matrices,
        approximated_covariances=covariance_approximations,
    )
    training_approximation_errors = select_covariances(
        approximation_errors,
        training_timestamps,
    )
    holdout_approximation_errors = select_covariances(
        approximation_errors,
        holdout_timestamps,
    )
    training_relative_errors = compute_relative_frobenius_errors(
        errors=training_approximation_errors,
        covariance_matrices=training_covariances,
    )
    holdout_relative_errors = compute_relative_frobenius_errors(
        errors=holdout_approximation_errors,
        covariance_matrices=holdout_covariances,
    )
    aggregated_training_error = (
        compute_aggregated_relative_frobenius_error(
            errors=training_approximation_errors,
            covariance_matrices=training_covariances,
        )
    )
    aggregated_holdout_error = (
        compute_aggregated_relative_frobenius_error(
            errors=holdout_approximation_errors,
            covariance_matrices=holdout_covariances,
        )
    )

    first_component = reference_basis.columns[0]
    dominant_indicator = (
        transformed_covariances
        .xs(first_component, level="component")
        .loc[:, first_component]
        .copy()
    )
    dominant_indicator.name = "dominant_indicator"
    naive_indicator_forecasts = (
        dominant_indicator.shift(1)
        .iloc[TRAINING_OBSERVATION_COUNT:]
        .copy()
    )
    arfima_indicator_forecasts, _ = fit_and_forecast_arfima_0d0_holdout(
        series=dominant_indicator,
        training_observation_count=TRAINING_OBSERVATION_COUNT,
    )
    direct_naive_forecasts = construct_direct_naive_covariance_forecasts(
        covariance_matrices=covariance_matrices,
        training_observation_count=TRAINING_OBSERVATION_COUNT,
    )
    naive_indicator_covariance_forecasts = (
        construct_dominant_indicator_covariance_forecasts(
            dominant_indicator_forecasts=naive_indicator_forecasts,
            reference_basis=reference_basis,
            reference_eigenvalues=reference_eigenvalues,
        )
    )
    arfima_indicator_covariance_forecasts = (
        construct_dominant_indicator_covariance_forecasts(
            dominant_indicator_forecasts=arfima_indicator_forecasts,
            reference_basis=reference_basis,
            reference_eigenvalues=reference_eigenvalues,
        )
    )

    forecasts_by_method = {
        DIRECT_NAIVE: direct_naive_forecasts,
        NAIVE_INDICATOR: naive_indicator_covariance_forecasts,
        ARFIMA_INDICATOR: arfima_indicator_covariance_forecasts,
    }
    rmse_by_method = pd.DataFrame(
        {
            method: compute_covariance_rmse(
                holdout_covariances - forecasts
            )
            for method, forecasts in forecasts_by_method.items()
        }
    ).T
    psd_summary = pd.DataFrame(
        {
            method: compute_psd_diagnostics(forecasts)
            for method, forecasts in forecasts_by_method.items()
        }
    ).T

    direct_naive_losses = compute_covariance_squared_frobenius_losses(
        holdout_covariances - direct_naive_forecasts
    )["overall"]
    naive_indicator_losses = compute_covariance_squared_frobenius_losses(
        holdout_covariances - naive_indicator_covariance_forecasts
    )["overall"]
    arfima_losses = compute_covariance_squared_frobenius_losses(
        holdout_covariances - arfima_indicator_covariance_forecasts
    )["overall"]
    loss_difference = direct_naive_losses - arfima_losses

    approximation_summary = pd.DataFrame(
        {
            "Training": {
                "aggregated_error": aggregated_training_error,
                "mean_interval_error": training_relative_errors.mean(),
                "median_interval_error": training_relative_errors.median(),
                "95th_percentile_interval_error": (
                    training_relative_errors.quantile(0.95)
                ),
                "maximum_interval_error": training_relative_errors.max(),
            },
            "Holdout": {
                "aggregated_error": aggregated_holdout_error,
                "mean_interval_error": holdout_relative_errors.mean(),
                "median_interval_error": holdout_relative_errors.median(),
                "95th_percentile_interval_error": (
                    holdout_relative_errors.quantile(0.95)
                ),
                "maximum_interval_error": holdout_relative_errors.max(),
            },
        }
    )
    approximation_summary.index.name = "error_summary"

    if not np.allclose(
        reference_eigenvalues.divide(reference_eigenvalues.sum()).sum(),
        1.0,
    ):
        raise ValueError("Reference PCA variance shares do not sum to one.")
    if int(psd_summary["non_psd_count"].sum()) != 0:
        raise ValueError("At least one plotted forecast method is non-PSD.")

    generated_paths = []
    period_label = f"{config.months[0]} to {config.months[-1]}"
    generated_paths.extend(
        plot_normalized_closing_prices(
            closing_prices,
            holdout_timestamps[0],
            period_label,
            OUTPUT_DIRECTORY,
        )
    )
    generated_paths.extend(
        plot_reference_pca_variance_share(
            reference_eigenvalues,
            OUTPUT_DIRECTORY,
        )
    )
    generated_paths.extend(
        plot_dominant_indicator_time_series(
            dominant_indicator,
            holdout_timestamps[0],
            OUTPUT_DIRECTORY,
        )
    )
    generated_paths.extend(
        plot_approximation_error_distributions(
            training_relative_errors,
            holdout_relative_errors,
            aggregated_training_error,
            aggregated_holdout_error,
            OUTPUT_DIRECTORY,
        )
    )
    generated_paths.extend(
        plot_covariance_rmse_reduction(
            rmse_by_method,
            len(holdout_timestamps),
            OUTPUT_DIRECTORY,
        )
    )
    generated_paths.extend(
        plot_cumulative_loss_difference(
            direct_naive_losses,
            arfima_losses,
            "Direct naive",
            "ARFIMA indicator",
            "06_cumulative_loss_difference",
            OUTPUT_DIRECTORY,
        )
    )
    generated_paths.extend(
        plot_cumulative_loss_difference(
            naive_indicator_losses,
            arfima_losses,
            "Naive indicator",
            "ARFIMA indicator",
            "07_cumulative_indicator_loss_difference",
            OUTPUT_DIRECTORY,
        )
    )
    save_supporting_tables(
        output_directory=OUTPUT_DIRECTORY,
        reference_eigenvalues=reference_eigenvalues,
        approximation_summary=approximation_summary,
        rmse_by_method=rmse_by_method,
        loss_difference=loss_difference,
        psd_summary=psd_summary,
    )

    print("Generated thesis figures:")
    for path in generated_paths:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print("Supporting tables:")
    for path in sorted((OUTPUT_DIRECTORY / "tables").glob("*.csv")):
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print("Figure contracts:")
    for stem, contract in FIGURE_CONTRACTS.items():
        print(f"  {stem}: {contract}")


if __name__ == "__main__":
    main()