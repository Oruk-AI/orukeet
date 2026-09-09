"""Portable academic figure style inspired by Chen Liu's figures4papers.

Source: https://github.com/ChenLiu-1996/figures4papers
This local helper implements styling/export; it does not infer statistics.
Python 3.10+, Matplotlib, and NumPy. Run --demo BASENAME for synthetic examples.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
import numpy as np


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#767676",
    "ink": "#272727",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}
DEFAULT_COLORS = tuple(
    PALETTE[key]
    for key in ("blue_main", "red_strong", "teal", "violet", "neutral_dark", "green_3")
)
_FORMATS = frozenset(("pdf", "svg", "png", "tif", "tiff"))


def _available_font():
    for family in ("Arial", "Helvetica", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "sans-serif"


@contextmanager
def publication_style(font_size=9, font_family=None, axes_linewidth=0.8, rc=None):
    """Apply a local paper-sized style; caller controls figure dimensions."""
    if not np.isfinite(font_size) or font_size <= 0:
        raise ValueError("font_size must be finite and positive")
    if not np.isfinite(axes_linewidth) or axes_linewidth <= 0:
        raise ValueError("axes_linewidth must be finite and positive")
    params = {
        "font.family": font_family or _available_font(),
        "font.size": font_size,
        "text.color": PALETTE["ink"],
        "text.usetex": False,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": font_size + 1,
        "axes.titlesize": font_size + 1,
        "axes.titlepad": 8,
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=DEFAULT_COLORS),
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": max(font_size - 0.5, 1),
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "grid.color": "#E4E4E4",
        "grid.linewidth": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "savefig.bbox": None,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    params.update(rc or {})
    with mpl.rc_context(params):
        yield


def plot_interval(ax, x, mean, lower, upper, *, band_alpha=0.18, **line_kwargs):
    """Plot supplied bounds, without inventing uncertainty or bridging gaps."""
    if any(np.ma.isMaskedArray(a) and np.ma.getmaskarray(a).any()
           for a in (x, mean, lower, upper)):
        raise ValueError("Handle masked-data segments explicitly before plotting")
    arrays = [np.asarray(a, dtype=float) for a in (x, mean, lower, upper)]
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("x, mean, lower, and upper must be one-dimensional")
    if not arrays[0].size or len({a.size for a in arrays}) != 1:
        raise ValueError("x, mean, lower, and upper must have equal nonzero length")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Handle missing-data segments explicitly; inputs must be finite")
    x, mean, lower, upper = arrays
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing; resolve ordering explicitly")
    if np.any(lower > upper):
        raise ValueError("lower must not exceed upper")
    if not np.isfinite(band_alpha) or not 0 <= band_alpha <= 1:
        raise ValueError("band_alpha must be between 0 and 1")
    line, = ax.plot(x, mean, **line_kwargs)
    band = ax.fill_between(
        x, lower, upper, color=line.get_color(), alpha=band_alpha,
        linewidth=0, zorder=line.get_zorder() - 0.1, label="_nolegend_",
    )
    return line, band


def save_figure(
    fig, path, *, formats=None, dpi=300, crop=False, pad=0.04,
    close=True, transparent=False,
):
    """Save explicit formats with exact canvas size unless crop=True."""
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be finite and positive")
    if not np.isfinite(pad) or pad < 0:
        raise ValueError("pad must be finite and nonnegative")
    target = Path(path).expanduser()
    suffix = target.suffix.lower().lstrip(".")
    base = target.with_suffix("") if suffix in _FORMATS else target
    if formats is None:
        formats = (suffix,) if suffix in _FORMATS else ("pdf", "png")
    elif isinstance(formats, str):
        formats = (formats,)
    formats = tuple(dict.fromkeys(str(f).lower().lstrip(".") for f in formats))
    if not formats or any(f not in _FORMATS for f in formats):
        raise ValueError("Supported formats: pdf, svg, png, tif, tiff")
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    # Explicitly reset a caller's global tight-cropping default when preserving size.
    with mpl.rc_context({"savefig.bbox": None}):
        fig.canvas.draw()
        for fmt in formats:
            destination = Path(str(base) + "." + fmt)
            fig.savefig(
                destination, format=fmt, dpi=dpi,
                bbox_inches="tight" if crop else None, pad_inches=pad,
                transparent=transparent,
                facecolor="none" if transparent else fig.get_facecolor(),
            )
            paths.append(destination)
    if close:
        from matplotlib import pyplot as plt
        plt.close(fig)
    return paths


def _demo(basename):
    """Generate clearly labeled synthetic plots to inspect this helper."""
    mpl.use("Agg")
    from matplotlib import pyplot as plt

    rng = np.random.default_rng(12)
    steps = np.arange(0, 61, 10)
    methods = ("Method A", "Method B")
    colors = (PALETTE["blue_main"], PALETTE["red_strong"])
    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained")
        for idx, (name, color) in enumerate(zip(methods, colors)):
            runs = (
                0.3 + idx * 0.1 + (0.8 - idx * 0.1) * np.exp(-steps / 23)
                + rng.normal(0, 0.03, size=(5, steps.size))
            )
            mean, sd = runs.mean(axis=0), runs.std(axis=0, ddof=1)
            plot_interval(
                axes[0], steps, mean, mean - sd, mean + sd,
                color=color, label=name, marker=("o", "s")[idx],
                linestyle=("-", "--")[idx],
            )
        axes[0].set(xlabel="Training step", ylabel="Validation loss")
        axes[0].set_title("a  Learning dynamics", loc="left", fontweight="bold")
        axes[0].legend(loc="upper right")

        conditions = ("Full", "No alignment", "No pretraining")
        scores = np.array([
            [87.4, 88.1, 87.7, 88.4, 87.9],
            [84.9, 85.7, 85.2, 85.9, 85.4],
            [81.8, 82.6, 82.0, 82.8, 82.3],
        ])
        for idx, (values, color) in enumerate(zip(
            scores, (PALETTE["blue_main"], PALETTE["teal"], PALETTE["neutral_dark"]),
        )):
            axes[1].errorbar(
                values.mean(), idx, xerr=values.std(ddof=1),
                color=color, fmt=("o", "s", "D")[idx], capsize=3,
            )
        axes[1].set_yticks(np.arange(3), conditions)
        axes[1].set(xlabel="Accuracy (%)", xlim=(80, 90), ylim=(2.5, -0.5))
        axes[1].set_title("b  Component ablation", loc="left", fontweight="bold")
        axes[1].grid(axis="x")
        fig.supxlabel(
            "Illustrative synthetic data · 5 runs · bands and error bars show ±1 sample SD",
            fontsize=8,
        )
        return save_figure(fig, basename, formats=("pdf", "svg", "png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", metavar="BASENAME", required=True,
                        help="Write a labeled synthetic PDF/SVG/PNG example")
    for output in _demo(parser.parse_args().demo):
        print(output)
