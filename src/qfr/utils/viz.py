"""Publication-quality plotting helpers (consistent style + figure saving)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never pop a window

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from qfr.utils.config import settings  # noqa: E402

PALETTE = {
    "primary": "#1b3a5b",  # deep navy
    "accent": "#c1121f",  # red
    "muted": "#6c757d",
    "green": "#2a9d8f",
    "gold": "#e9c46a",
}


def set_plot_style() -> None:
    """Apply a clean, consistent house style for all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#444444",
            "grid.alpha": 0.30,
            "legend.frameon": False,
        }
    )


def save_fig(fig, name: str, subdir: str | None = None) -> Path:
    """Save ``fig`` as a PNG under charts/ and close it. Returns the path."""
    d = settings.charts_dir if subdir is None else settings.charts_dir / subdir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
