"""Geometry plot for the misspecified identification experiment.

Shows, in the two-dimensional slice config.SLICE of the parameter space, the
regions of the hypotheses that are nearest in prediction risk, the
target-risk ellipse around the true system, the OLS estimates and how often
the exponential-weights estimator selects each hypothesis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import config
from src.experiment import ExperimentResult
from src.plotting import save_figure

# =============================================================================
# Helpers for the two-dimensional slice
# =============================================================================


def _slice_coordinates(theta: np.ndarray) -> np.ndarray:
    """Coordinates of theta (or a stack of thetas) in the plotted slice."""

    return np.stack([theta[..., i, j] for i, j in config.SLICE], axis=-1)


def _slice_metric(covariance: np.ndarray) -> np.ndarray:
    """Quadratic form of the prediction risk restricted to the slice.

    For entries (i, j), (k, l) of theta the metric is Sigma[j, l] if i == k
    (same output row) and 0 otherwise.
    """

    return np.array(
        [
            [covariance[j, col] if i == row else 0.0 for row, col in config.SLICE]
            for i, j in config.SLICE
        ]
    )


def _slice_label(entry: tuple[int, int]) -> str:
    """LaTeX label of a theta entry, e.g. (0, 1) -> $a_{12}$, (0, 2) -> $b_{11}$."""

    i, j = entry
    if j < config.N_X:
        return rf"$a_{{{i + 1}{j + 1}}}$"
    return rf"$b_{{{i + 1}{j - config.N_X + 1}}}$"


# =============================================================================
# Plot elements
# =============================================================================


def _plot_risk_regions(ax, covariance: np.ndarray) -> None:
    """Shade the regions in which each hypothesis is the risk-nearest one."""

    hypothesis_coordinates = [_slice_coordinates(theta) for theta in config.HYPOTHESES]

    gx, gy = np.meshgrid(
        np.linspace(*config.REGION_GRID_X, config.REGION_GRID_RESOLUTION),
        np.linspace(*config.REGION_GRID_Y, config.REGION_GRID_RESOLUTION),
    )
    grid = np.stack([gx, gy], axis=-1)

    metric = _slice_metric(covariance)

    distances = np.stack(
        [
            np.einsum("...i,ij,...j->...", grid - hypothesis, metric, grid - hypothesis)
            for hypothesis in hypothesis_coordinates
        ],
        axis=-1,
    )

    ax.contourf(
        gx,
        gy,
        distances.argmin(axis=-1),
        levels=np.arange(-0.5, len(config.HYPOTHESES)),
        alpha=0.15,
        colors=config.HYPOTHESIS_COLORS,
    )


def _plot_hypotheses(ax, target_index: int, estimates: dict[str, np.ndarray]) -> None:
    """Plot the hypotheses, annotated with the exp-weights selection share."""

    for index, theta in enumerate(config.HYPOTHESES):
        x, y = _slice_coordinates(theta)

        ax.scatter(
            x,
            y,
            marker="s",
            s=70,
            color=config.HYPOTHESIS_COLORS[index],
            zorder=3,
        )

        # Percentage of repetitions in which exp-weights selected this hypothesis.
        share = 100 * np.mean(
            np.all(np.isclose(estimates["Exp_Weight"], theta), axis=(1, 2))
        )

        ax.annotate(
            f"H{index + 1}: {share:.0f}%",
            (x, y),
            xytext=(-50, 10),
            textcoords="offset points",
            fontsize=12,
        )

        if index == target_index:
            ax.scatter(
                x,
                y,
                s=260,
                facecolor="none",
                edgecolor=config.COLOR_REFERENCE,
                lw=2,
                zorder=4,
                label=config.LABEL_TARGET_HYPOTHESIS,
            )


def _plot_true_system(ax) -> None:
    """Plot the true system."""

    x, y = _slice_coordinates(config.THETA_TRUE)

    ax.scatter(
        x,
        y,
        marker="*",
        s=220,
        color=config.COLOR_TARGET,
        edgecolor=config.COLOR_REFERENCE,
        zorder=6,
        label=r"True system $\theta_*$",
    )


def _plot_estimates(ax, estimates: dict[str, np.ndarray]) -> None:
    """Scatter the Monte-Carlo OLS estimates."""

    coordinates = _slice_coordinates(estimates["OLS"])

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=7,
        alpha=0.35,
        color=config.COLOR_ESTIMATOR["OLS"],
        label="OLS estimates",
    )


def _plot_risk_ellipse(ax, covariance: np.ndarray, target_risk: float) -> None:
    """Plot the level set (x - theta_*)^T G (x - theta_*) = target_risk."""

    metric = _slice_metric(covariance)
    center = _slice_coordinates(config.THETA_TRUE)

    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    semi_axes = np.sqrt(target_risk / eigenvalues)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    ax.add_patch(
        Ellipse(
            xy=center,
            width=2 * semi_axes[0],
            height=2 * semi_axes[1],
            angle=angle,
            fill=False,
            edgecolor=config.COLOR_REFERENCE,
            linestyle="--",
            linewidth=1.2,
        )
    )


def _format_axis(ax) -> None:
    """Apply common axis formatting."""

    ax.set_xlabel(_slice_label(config.SLICE[0]), fontsize=12)
    ax.set_ylabel(_slice_label(config.SLICE[1]), fontsize=12)

    ax.set_xlim(*config.GEOMETRY_XLIM)
    ax.set_ylim(*config.GEOMETRY_YLIM)

    ax.grid(True, alpha=0.2)
    ax.tick_params(axis="both", labelsize=11)


# =============================================================================
# Figure
# =============================================================================


def plot_geometry(results: ExperimentResult, path_stem: Path) -> None:
    """Create and save the geometry figure (left: i.i.d., right: trajectory)."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

    for ax, mode in zip(axes, config.MODES, strict=True):
        risk_result = results.risk[mode]
        geometry_result = results.geometry[mode]

        _plot_risk_regions(ax, risk_result.covariance)
        _plot_hypotheses(ax, risk_result.target_index, geometry_result.estimates)
        _plot_estimates(ax, geometry_result.estimates)
        _plot_true_system(ax)
        _plot_risk_ellipse(ax, risk_result.covariance, risk_result.target_risk)
        _format_axis(ax)

    # Legend: show the target-hypothesis marker smaller than in the plot.
    handles, labels = axes[0].get_legend_handles_labels()
    handles[labels.index(config.LABEL_TARGET_HYPOTHESIS)] = Line2D(
        [],
        [],
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgecolor=config.COLOR_REFERENCE,
        markeredgewidth=1.5,
        linestyle="none",
        label=config.LABEL_TARGET_HYPOTHESIS,
    )
    axes[0].legend(handles=handles, fontsize=12, loc="upper left")

    fig.tight_layout()

    save_figure(
        fig,
        path_stem,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        transparent=False,
    )

    plt.close(fig)
