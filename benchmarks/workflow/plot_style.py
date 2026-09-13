"""Publication plot style: Arial, no top/right spines, 300 DPI, PNG + editable-text PDF.

Copy this file into a project as ``<pkg>/plot_style.py`` and add a project-specific
``COLORS`` dict mapping condition names to hex codes. Call ``setup_plot_style()`` once
per script, after any ``seaborn.set_theme()`` call (seaborn overwrites rcParams).
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

# Role-based palette: ONE saturated color for the condition of interest, muted tints for
# everything else. Identity of the tints is carried by the legend, not by hue, so use
# them only when a legend or direct label is present.
PALETTE = {
    "key": "#0F4D92",  # the method / condition the figure is about
    "key_light": "#3775BA",
    "neutral": "#CFCECE",  # reference or background categories
    "neutral_dark": "#767676",
    "green_1": "#DDF3DE",  # related positives, light -> dark
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",  # contrasts / alternatives, light -> dark
    "red_2": "#E9A6A1",
    "red_3": "#B64342",
    "highlight": "#FFD700",  # a single callout only
}

# Colorblind-safe ordered set (Okabe-Ito) for equal-status categories. Assign in this
# order and never cycle; a 7th category folds into "Other". Validated: every adjacent
# pair has CVD delta-E >= 9.6 on a light surface.
CATEGORICAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"]

# Named sizes in inches. Every size shrinks ~0.6x when placed at journal column width
# (89 mm single, 183 mm double), which turns the 12 pt base font into ~7 pt on the page.
FIGSIZE = {
    "single": (6, 5),  # one panel
    "wide": (10, 5),  # one wide panel (time series, many categories)
    "square": (8, 8),  # scatter, heatmap
    "double": (12, 5),  # two panels side by side
    "quad": (10, 8),  # 2x2 grid
    "row": (20, 5),  # 3-4 comparison panels + legend panel
}

_FONT_DIRS = [
    "/usr/share/fonts/truetype/msttcorefonts",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype",
]


def _register_fonts() -> None:
    """Register system TrueType fonts so Arial resolves inside conda environments."""
    for d in _FONT_DIRS:
        for f in fm.findSystemFonts(fontpaths=[d]):
            with contextlib.suppress(Exception):
                fm.fontManager.addfont(f)


def setup_plot_style(font_size: int = 12) -> None:
    """Configure matplotlib rcParams for publication figures.

    Args:
        font_size: Base font size in points. Other sizes scale from it.
    """
    plt.rcdefaults()
    _register_fonts()
    plt.rcParams.update(
        {
            # Fonts
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.titleweight": "bold",
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.titlesize": font_size + 4,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            # Vector export: TrueType (type 42) keeps text editable in Illustrator/Inkscape
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # Axes: no top/right spines, slightly heavy remaining spines
            "axes.linewidth": 1.5,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            # Ticks
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            # Marks
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "errorbar.capsize": 3,
            # Legend: frameless
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            # Figure
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def save_figure(
    fig: Figure,
    path: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    close: bool = True,
) -> list[Path]:
    """Save a figure in every requested format and return the written paths.

    Args:
        fig: Figure to save.
        path: Output path; the suffix is replaced per format.
        formats: File formats to write. PDF/SVG keep text editable.
        dpi: Raster resolution for PNG/TIFF.
        close: Close the figure after saving to free memory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        out = path.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        written.append(out)
    if close:
        plt.close(fig)
    return written


def alpha_ramp(color: str, n: int, lo: float = 0.3, hi: float = 1.0) -> list[tuple]:
    """Return n RGBA tints of one color, light to full, for ablation or dose series."""
    r, g, b, _ = to_rgba(color)
    step = (hi - lo) / max(n - 1, 1)
    return [(r, g, b, lo + i * step) for i in range(n)]


def legend_panel(ax: Axes, handles, labels, **kwargs) -> None:
    """Turn an axis into a legend-only panel so the legend never covers data."""
    ax.set_axis_off()
    kwargs.setdefault("loc", "center")
    kwargs.setdefault("frameon", False)
    ax.legend(handles, labels, **kwargs)


def label_panels(
    axes: Sequence[Axes],
    labels: str = "ABCDEFGHIJ",
    x: float = -0.12,
    y: float = 1.05,
    fontsize: int | None = None,
) -> None:
    """Add bold panel letters at the top-left of each axis (Nature style uses lowercase)."""
    fontsize = fontsize or plt.rcParams["axes.titlesize"]
    for ax, letter in zip(axes, labels, strict=False):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize, fontweight="bold", va="bottom", ha="right")


def add_value_labels(
    ax: Axes, bars: BarContainer, fmt: str = "{:.2f}", fontsize: int | None = None, offset: int = 4
) -> None:
    """Print each bar's height above it."""
    fontsize = fontsize or plt.rcParams["xtick.labelsize"]
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# Benchmark condition colors: one saturated key color for the carried/winning
# condition, neutral for priors/baselines, ordered categories for deliveries.
COLORS = {
    "affinage": PALETTE["key"],
    "uniprot": PALETTE["neutral_dark"],
    "prior": PALETTE["neutral"],
    "selected": PALETTE["key"],
    "candidate": PALETTE["green_2"],
    "single_call": CATEGORICAL[0],
    "cot": CATEGORICAL[1],
    "stepwise": CATEGORICAL[2],
}
