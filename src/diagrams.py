from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

from .diagram_style import (
    ARTICLE_COLORS,
    add_round_box,
    clean_axis,
    draw_cfa_grid,
)
from .demosaic import MHC_KERNELS, bilinear_rggb, mhc_rggb
from .experiments import single_hot_pixel


def _center_crop(array: np.ndarray, size: int = 5) -> np.ndarray:
    h, w = array.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    return array[cy - half : cy + half + 1, cx - half : cx + half + 1]


def _annotate_matrix(ax, matrix: np.ndarray, fmt: str = ".2g") -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                format(float(value), fmt),
                ha="center",
                va="center",
                fontsize=8.5,
                color=ARTICLE_COLORS["text"],
            )


def draw_bayer_overview():
    """Introductory Bayer CFA diagram for the first section.

    Kept deliberately compact so it fits the Quarto article column without
    horizontal clipping.
    """
    fig = plt.figure(figsize=(8.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.48],
        width_ratios=[1.0, 1.0],
        hspace=0.08,
        wspace=0.18,
    )

    ax_grid = fig.add_subplot(gs[0, 0])
    draw_cfa_grid(ax_grid, rows=6, cols=6)
    ax_grid.set_title("RGGB Bayer 配列", pad=8)

    ax_hot = fig.add_subplot(gs[0, 1])
    draw_cfa_grid(ax_hot, rows=5, cols=5, hot=(2, 2))
    ax_hot.set_title("例：Rサイト1点だけが異常", pad=8)
    ax_hot.text(
        0.5,
        -0.07,
        "黄色枠 = 今回追跡するホットサンプル",
        transform=ax_hot.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        color=ARTICLE_COLORS["muted"],
    )

    ax_note = fig.add_subplot(gs[1, :])
    clean_axis(ax_note)

    add_round_box(
        ax_note,
        (0.06, 0.28),
        0.40,
        0.48,
        "各受光位置が記録するのは\nR / G / B のうち 1 成分だけ",
        facecolor="#eef4fa",
        edgecolor="#c9d9ea",
        fontsize=10.8,
        fontweight="bold",
    )

    ax_note.add_patch(
        FancyArrowPatch(
            (0.49, 0.52),
            (0.60, 0.52),
            transform=ax_note.transAxes,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            color=ARTICLE_COLORS["accent"],
        )
    )

    ax_note.text(
        0.78,
        0.52,
        "残りの2成分は\nデモザイクで推定する",
        transform=ax_note.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        color=ARTICLE_COLORS["muted"],
        linespacing=1.45,
    )

    return fig


def draw_bilinear_overview():
    """Show how one R hot sample spreads under normalized bilinear interpolation."""
    raw, _ = single_hot_pixel(size=9, channel="R")
    r_plane = bilinear_rggb(raw)[..., 0]
    r_crop = _center_crop(r_plane, 5)

    fig = plt.figure(figsize=(10.5, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.30)

    ax_raw = fig.add_subplot(gs[0, 0])
    draw_cfa_grid(ax_raw, rows=5, cols=5, hot=(2, 2))
    ax_raw.set_title("入力：Rサイト1点が高い", pad=10)

    ax_map = fig.add_subplot(gs[0, 1])
    im = ax_map.imshow(
        r_crop,
        cmap="Reds",
        vmin=0.0,
        vmax=max(1.0, float(r_crop.max())),
        interpolation="nearest",
    )
    ax_map.set_title("補間後の R 成分", pad=10)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    for row in range(r_crop.shape[0]):
        for col in range(r_crop.shape[1]):
            if r_crop[row, col] > 0:
                ax_map.text(
                    col,
                    row,
                    f"{r_crop[row, col]:.2g}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=ARTICLE_COLORS["text"],
                )
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    ax_note = fig.add_subplot(gs[0, 2])
    clean_axis(ax_note)
    add_round_box(
        ax_note,
        (0.08, 0.57),
        0.84,
        0.22,
        "近傍サンプルの\n重み付き平均",
        facecolor="#fff4e6",
        edgecolor="#ead5b7",
        fontsize=11,
        fontweight="bold",
    )
    ax_note.text(
        0.50,
        0.36,
        r"$w_i \geq 0,\quad \sum_i w_i=1$",
        transform=ax_note.transAxes,
        ha="center",
        va="center",
        fontsize=12,
    )
    ax_note.text(
        0.50,
        0.18,
        "正規化された正の重みなので\n入力範囲を越える新しい極値を作らない",
        transform=ax_note.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color=ARTICLE_COLORS["muted"],
        linespacing=1.45,
    )

    return fig


def draw_mhc_overview():
    """Show MHC spread and one representative signed correction kernel."""
    raw, _ = single_hot_pixel(size=9, channel="R")
    r_plane = mhc_rggb(raw)[..., 0]
    r_crop = _center_crop(r_plane, 5)

    kernel = MHC_KERNELS["R/B at G"]
    vmax = max(abs(float(r_crop.min())), abs(float(r_crop.max())), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(10.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.15], wspace=0.30)

    ax_raw = fig.add_subplot(gs[0, 0])
    draw_cfa_grid(ax_raw, rows=5, cols=5, hot=(2, 2))
    ax_raw.set_title("入力：同じ1点異常", pad=10)

    ax_map = fig.add_subplot(gs[0, 1])
    im = ax_map.imshow(
        r_crop,
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )
    ax_map.set_title("MHC後の R 成分", pad=10)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    for row in range(r_crop.shape[0]):
        for col in range(r_crop.shape[1]):
            value = r_crop[row, col]
            if abs(value) > 1e-12:
                ax_map.text(
                    col,
                    row,
                    f"{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color=ARTICLE_COLORS["text"],
                )
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    ax_kernel = fig.add_subplot(gs[0, 2])
    kernel_abs = np.max(np.abs(kernel))
    knorm = TwoSlopeNorm(vmin=-kernel_abs, vcenter=0.0, vmax=kernel_abs)
    ax_kernel.imshow(kernel, cmap="RdBu_r", norm=knorm, interpolation="nearest")
    _annotate_matrix(ax_kernel, kernel)
    ax_kernel.set_title("例：G位置でR/Bを推定する係数", pad=10)
    ax_kernel.set_xticks([])
    ax_kernel.set_yticks([])
    for spine in ax_kernel.spines.values():
        spine.set_visible(False)
    ax_kernel.text(
        0.5,
        -0.10,
        "負の係数を含む → overshoot / undershoot が可能",
        transform=ax_kernel.transAxes,
        ha="center",
        va="top",
        fontsize=9.3,
        color=ARTICLE_COLORS["muted"],
    )

    return fig
