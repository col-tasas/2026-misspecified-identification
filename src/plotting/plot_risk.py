"""Excess-risk plot for the misspecified identification experiment."""

from pathlib import Path

import matplotlib.pyplot as plt

import config
from src.experiment import ExperimentResult, RiskExperimentResult
from src.plotting import save_figure


def _plot_mode(ax, result: RiskExperimentResult) -> None:
    """Plot mean excess risk vs. sample size for one data-generation mode."""

    for estimator in ("OLS", "Exp_Weight"):
        ax.plot(
            config.SAMPLE_SIZES,
            result.mean_risk[estimator],
            marker="o",
            linewidth=3,
            markersize=5,
            color=config.COLOR_ESTIMATOR[estimator],
            label=estimator,
        )

    ax.axhline(
        result.target_risk,
        color=config.COLOR_REFERENCE,
        linestyle="--",
        linewidth=2,
        label=config.LABEL_TARGET_HYPOTHESIS,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Number of samples", fontsize=18)

    ax.grid(True, which="both", alpha=0.2)
    ax.tick_params(axis="both", labelsize=17)


def plot_excess_risk(results: ExperimentResult, path_stem: Path) -> None:
    """Create and save the excess-risk figure (left: i.i.d., right: trajectory)."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    _plot_mode(axes[0], results.risk["iid"])
    _plot_mode(axes[1], results.risk["trajectory"])

    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles, fontsize=16, loc="upper right")

    axes[0].set_ylabel("Expected Prediction Error", fontsize=18)

    # The top margin keeps the panel proportions of the published figure.
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    save_figure(fig, path_stem, bbox_inches="tight")

    plt.close(fig)
