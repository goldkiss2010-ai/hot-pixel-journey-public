from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve


def rggb_site_masks(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return masks for the four RGGB CFA phases: R, Gr, Gb, B."""
    h, w = shape
    r = np.zeros((h, w), dtype=float)
    gr = np.zeros((h, w), dtype=float)
    gb = np.zeros((h, w), dtype=float)
    b = np.zeros((h, w), dtype=float)

    r[0::2, 0::2] = 1.0
    gr[0::2, 1::2] = 1.0
    gb[1::2, 0::2] = 1.0
    b[1::2, 1::2] = 1.0
    return r, gr, gb, b


def rggb_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return combined R, G, B sampling masks for an RGGB Bayer CFA."""
    r, gr, gb, b = rggb_site_masks(shape)
    return r, gr + gb, b


def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """Sample an RGB image with an RGGB Bayer CFA."""
    r, g, b = rggb_masks(rgb.shape[:2])
    return rgb[..., 0] * r + rgb[..., 1] * g + rgb[..., 2] * b


def bilinear_rggb(cfa: np.ndarray) -> np.ndarray:
    """Simple normalized bilinear interpolation for an RGGB CFA."""
    r_m, g_m, b_m = rggb_masks(cfa.shape)
    kernel = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]],
        dtype=float,
    )

    channels = []
    for mask in (r_m, g_m, b_m):
        numerator = convolve(cfa * mask, kernel, mode="mirror")
        denominator = convolve(mask, kernel, mode="mirror")
        interp = numerator / np.maximum(denominator, 1e-12)
        channels.append(np.where(mask == 1.0, cfa, interp))

    return np.stack(channels, axis=-1)


def mhc_rggb(cfa: np.ndarray) -> np.ndarray:
    """Malvar-He-Cutler linear demosaicing for RGGB CFA.

    The kernels intentionally contain negative coefficients. Output is not clipped.
    """
    r_m, g_m, b_m = rggb_masks(cfa.shape)

    k_g = np.array(
        [[0, 0, -1, 0, 0],
         [0, 0,  2, 0, 0],
         [-1, 2, 4, 2, -1],
         [0, 0,  2, 0, 0],
         [0, 0, -1, 0, 0]],
        dtype=float,
    ) / 8.0

    k_rg = np.array(
        [[0, 0, 0.5, 0, 0],
         [0, -1, 0, -1, 0],
         [-1, 4, 5, 4, -1],
         [0, -1, 0, -1, 0],
         [0, 0, 0.5, 0, 0]],
        dtype=float,
    ) / 8.0

    k_bg = k_rg.T

    k_rb = np.array(
        [[0, 0, -1.5, 0, 0],
         [0, 2, 0, 2, 0],
         [-1.5, 0, 6, 0, -1.5],
         [0, 2, 0, 2, 0],
         [0, 0, -1.5, 0, 0]],
        dtype=float,
    ) / 8.0

    r = cfa * r_m
    g = cfa * g_m
    b = cfa * b_m

    g_est = convolve(cfa, k_g, mode="mirror")
    g = np.where((r_m + b_m) > 0, g_est, g)

    a = convolve(cfa, k_rg, mode="mirror")
    b_est = convolve(cfa, k_bg, mode="mirror")
    c = convolve(cfa, k_rb, mode="mirror")

    gr = np.zeros_like(cfa, dtype=bool)
    gb = np.zeros_like(cfa, dtype=bool)
    gr[0::2, 1::2] = True
    gb[1::2, 0::2] = True

    rp = r_m.astype(bool)
    bp = b_m.astype(bool)

    r = np.where(gr, a, r)
    r = np.where(gb, b_est, r)
    b = np.where(gb, a, b)
    b = np.where(gr, b_est, b)
    r = np.where(bp, c, r)
    b = np.where(rp, c, b)

    return np.stack([r, g, b], axis=-1)


def summarize_rgb(img: np.ndarray, eps: float = 1e-12) -> dict[str, float | int]:
    """Summarize support and range violations in an RGB reconstruction."""
    affected = np.any(np.abs(img) > eps, axis=2)
    out_of_range = np.any((img < 0.0) | (img > 1.0), axis=2)
    return {
        "affected_pixels": int(affected.sum()),
        "out_of_range_pixels": int(out_of_range.sum()),
        "minimum": float(img.min()),
        "maximum": float(img.max()),
    }


def kernel_range(kernel: np.ndarray) -> tuple[float, float]:
    """Exact range of a linear kernel for independent inputs constrained to [0,1]."""
    lo = float(kernel[kernel < 0].sum())
    hi = float(kernel[kernel > 0].sum())
    return lo, hi


MHC_KERNELS = {
    "G at R/B": np.array(
        [[0, 0, -1, 0, 0],
         [0, 0,  2, 0, 0],
         [-1, 2, 4, 2, -1],
         [0, 0,  2, 0, 0],
         [0, 0, -1, 0, 0]],
        dtype=float,
    ) / 8.0,
    "R/B at G": np.array(
        [[0, 0, 0.5, 0, 0],
         [0, -1, 0, -1, 0],
         [-1, 4, 5, 4, -1],
         [0, -1, 0, -1, 0],
         [0, 0, 0.5, 0, 0]],
        dtype=float,
    ) / 8.0,
    "R at B / B at R": np.array(
        [[0, 0, -1.5, 0, 0],
         [0, 2, 0, 2, 0],
         [-1.5, 0, 6, 0, -1.5],
         [0, 2, 0, 2, 0],
         [0, 0, -1.5, 0, 0]],
        dtype=float,
    ) / 8.0,
}
