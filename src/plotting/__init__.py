"""Figures for the misspecified identification experiment."""

from pathlib import Path

import config


def save_figure(fig, path_stem: Path, **savefig_kwargs) -> None:
    """Save `fig` once per format in config.FIGURE_FORMATS (e.g. .pdf and .png)."""

    for fmt in config.FIGURE_FORMATS:
        fig.savefig(
            Path(path_stem).with_suffix(f".{fmt}"),
            format=fmt,
            dpi=config.FIGURE_DPI,
            **savefig_kwargs,
        )
