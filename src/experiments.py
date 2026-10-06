from __future__ import annotations

import numpy as np

from .demosaic import rggb_masks, rggb_site_masks


def single_hot_pixel(
    size: int = 21,
    value: float = 1.0,
    channel: str = "R",
) -> tuple[np.ndarray, tuple[int, int]]:
    """Create an otherwise-zero RGGB CFA with one hot sample.

    channel may be R, Gr, Gb, G, or B. Gr and Gb distinguish the two green
    phases in the RGGB lattice.
    """
    if size % 2 == 0:
        raise ValueError("size must be odd")

    r_m, gr_m, gb_m, b_m = rggb_site_masks((size, size))
    masks = {
        "R": r_m,
        "Gr": gr_m,
        "Gb": gb_m,
        "G": gr_m + gb_m,
        "B": b_m,
    }
    if channel not in masks:
        raise ValueError("channel must be R, Gr, Gb, G, or B")

    mask = masks[channel]
    cy = cx = size // 2

    candidates = np.argwhere(mask == 1.0)
    nearest = candidates[
        np.argmin(np.sum((candidates - np.array([cy, cx])) ** 2, axis=1))
    ]
    y, x = map(int, nearest)

    raw = np.zeros((size, size), dtype=float)
    raw[y, x] = value
    return raw, (y, x)


def neutral_step(
    height: int = 256,
    width: int = 256,
    dark: float = 0.05,
    bright: float = 0.98,
) -> np.ndarray:
    """RGB vertical step edge."""
    rgb = np.full((height, width, 3), dark, dtype=float)
    rgb[:, width // 2 :, :] = bright
    return rgb


def diagonal_step(
    height: int = 256,
    width: int = 256,
    dark: float = 0.08,
    bright: float = 0.98,
) -> np.ndarray:
    """RGB diagonal step edge."""
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.full((height, width, 3), dark, dtype=float)
    rgb[(xx + yy) > width] = bright
    return rgb


def step_edge_angle(
    height: int = 256,
    width: int = 256,
    angle_deg: float = 0.0,
    dark: float = 0.05,
    bright: float = 0.98,
) -> np.ndarray:
    """Neutral step edge with controllable normal direction."""
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx - (width - 1) / 2.0
    y = yy - (height - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    signed_distance = x * np.cos(theta) + y * np.sin(theta)

    rgb = np.full((height, width, 3), dark, dtype=float)
    rgb[signed_distance >= 0.0] = bright
    return rgb


def flat_field(
    height: int = 256,
    width: int = 256,
    level: float = 0.18,
) -> np.ndarray:
    """Neutral linear-RGB flat field."""
    return np.full((height, width, 3), level, dtype=float)


def add_shot_noise(
    cfa: np.ndarray,
    full_well_e: float = 4096.0,
    seed: int = 12345,
) -> np.ndarray:
    """Add Poisson shot noise in an electron-count toy model and normalize."""
    rng = np.random.default_rng(seed)
    electrons = np.clip(cfa, 0.0, None) * full_well_e
    noisy = rng.poisson(electrons) / full_well_e
    return noisy


def add_read_noise(
    cfa: np.ndarray,
    sigma: float = 0.004,
    seed: int = 12345,
    clip: bool = False,
) -> np.ndarray:
    """Add zero-mean Gaussian read-noise toy model in normalized RAW units."""
    rng = np.random.default_rng(seed)
    noisy = np.asarray(cfa, dtype=float) + rng.normal(0.0, sigma, cfa.shape)
    return np.clip(noisy, 0.0, 1.0) if clip else noisy


def add_poisson_gaussian_noise(
    cfa: np.ndarray,
    full_well_e: float = 4096.0,
    read_noise_e: float = 3.0,
    seed: int = 12345,
    clip: bool = False,
) -> np.ndarray:
    """Poisson shot noise + Gaussian read noise in a simple electron model."""
    rng = np.random.default_rng(seed)
    signal_e = np.clip(np.asarray(cfa, dtype=float), 0.0, None) * full_well_e
    noisy_e = rng.poisson(signal_e) + rng.normal(0.0, read_noise_e, cfa.shape)
    noisy = noisy_e / full_well_e
    return np.clip(noisy, 0.0, 1.0) if clip else noisy


def replace_hot_pixel_same_color_median(
    cfa: np.ndarray,
    hot_xy: tuple[int, int],
    radius: int = 2,
) -> np.ndarray:
    """Toy pre-demosaic defect correction using same-CFA-phase neighbors.

    The center sample is replaced by the median of samples with the same RGGB
    phase inside a local window. This is intentionally simple and is used only
    to demonstrate why correction order matters.
    """
    y, x = hot_xy
    out = np.array(cfa, dtype=float, copy=True)
    h, w = out.shape

    phase = (y % 2, x % 2)
    values: list[float] = []

    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)

    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if (yy, xx) == (y, x):
                continue
            if (yy % 2, xx % 2) == phase:
                values.append(float(out[yy, xx]))

    if not values:
        raise ValueError("no same-phase neighbors found; increase radius")

    out[y, x] = float(np.median(values))
    return out
