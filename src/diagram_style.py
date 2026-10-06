from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ARTICLE_COLORS = {
    "R": "#d94b45",
    "G": "#3f9d62",
    "B": "#3978b8",
    "hot": "#f2b134",
    "positive": "#d95d50",
    "negative": "#4f7fbf",
    "text": "#30363d",
    "muted": "#68727d",
    "grid": "#cfd6dd",
    "panel": "#f6f8fa",
    "accent": "#4b86c6",
    "white": "#ffffff",
}


def configure_article_matplotlib() -> None:
    """Apply a restrained article-wide Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Yu Gothic",
                "Meiryo",
                "Noto Sans CJK JP",
                "DejaVu Sans",
            ],
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": ARTICLE_COLORS["grid"],
            "axes.labelcolor": ARTICLE_COLORS["text"],
            "axes.titlecolor": ARTICLE_COLORS["text"],
            "xtick.color": ARTICLE_COLORS["muted"],
            "ytick.color": ARTICLE_COLORS["muted"],
            "text.color": ARTICLE_COLORS["text"],
            "figure.facecolor": ARTICLE_COLORS["white"],
            "axes.facecolor": ARTICLE_COLORS["white"],
            "lines.linewidth": 1.8,
            "svg.fonttype": "none",
        }
    )


def cfa_label(row: int, col: int) -> str:
    """Return the RGGB channel label for one CFA site."""
    if row % 2 == 0:
        return "R" if col % 2 == 0 else "G"
    return "G" if col % 2 == 0 else "B"


def draw_cfa_grid(
    ax,
    rows: int = 6,
    cols: int = 6,
    hot: tuple[int, int] | None = None,
    show_letters: bool = True,
    alpha: float = 0.94,
) -> None:
    """Draw an RGGB grid with an optional highlighted hot sample."""
    for row in range(rows):
        for col in range(cols):
            label = cfa_label(row, col)
            rect = Rectangle(
                (col, row),
                1,
                1,
                facecolor=ARTICLE_COLORS[label],
                edgecolor=ARTICLE_COLORS["white"],
                linewidth=1.4,
                alpha=alpha,
            )
            ax.add_patch(rect)
            if show_letters:
                ax.text(
                    col + 0.5,
                    row + 0.53,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                )

    if hot is not None:
        row, col = hot
        ax.add_patch(
            Rectangle(
                (col + 0.04, row + 0.04),
                0.92,
                0.92,
                fill=False,
                edgecolor=ARTICLE_COLORS["hot"],
                linewidth=3.2,
                zorder=10,
            )
        )

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_round_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str | None = None,
    edgecolor: str | None = None,
    fontsize: float = 10.5,
    fontweight: str = "normal",
) -> None:
    """Draw a rounded annotation box in axes coordinates."""
    facecolor = facecolor or ARTICLE_COLORS["panel"]
    edgecolor = edgecolor or ARTICLE_COLORS["grid"]

    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=ARTICLE_COLORS["text"],
        linespacing=1.45,
    )


def clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
