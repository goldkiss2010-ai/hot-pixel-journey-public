from __future__ import annotations

import numpy as np


def apply_white_balance(
    rgb: np.ndarray,
    gains: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Apply per-channel gains without clipping."""
    g = np.asarray(gains, dtype=float).reshape(1, 1, 3)
    return np.asarray(rgb, dtype=float) * g


def apply_ccm(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 color correction matrix without clipping.

    RGB vectors are treated as column vectors mathematically:
        x_out = M x_in
    The array implementation stores RGB on the final axis.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    return np.einsum("...c,dc->...d", np.asarray(rgb, dtype=float), m)


def clip_rgb(rgb: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Component-wise clipping."""
    return np.clip(np.asarray(rgb, dtype=float), low, high)


def range_violation_mask(
    rgb: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    """Return one boolean per pixel: any RGB component lies outside [low, high]."""
    x = np.asarray(rgb, dtype=float)
    return np.any((x < low) | (x > high), axis=-1)


def component_violation_count(
    rgb: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
) -> int:
    """Count individual RGB components outside the display cube."""
    x = np.asarray(rgb, dtype=float)
    return int(np.count_nonzero((x < low) | (x > high)))


def stage_metrics(
    rgb: np.ndarray,
    reference: np.ndarray | None = None,
    low: float = 0.0,
    high: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Summarize range, support, and optional deviation from a reference."""
    x = np.asarray(rgb, dtype=float)
    px_oog = range_violation_mask(x, low=low, high=high)

    metrics: dict[str, float | int] = {
        "affected_pixels": int(np.count_nonzero(np.any(np.abs(x) > eps, axis=-1))),
        "out_of_range_pixels": int(np.count_nonzero(px_oog)),
        "out_of_range_components": component_violation_count(x, low=low, high=high),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }

    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        if ref.shape != x.shape:
            raise ValueError("reference must have the same shape as rgb")
        delta = x - ref
        metrics["delta_l1"] = float(np.sum(np.abs(delta)))
        metrics["delta_l2"] = float(np.sqrt(np.sum(delta * delta)))
        metrics["changed_pixels"] = int(
            np.count_nonzero(np.any(np.abs(delta) > eps, axis=-1))
        )

    return metrics


def clip_metrics(
    rgb: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Quantify what component-wise clipping destroys."""
    x = np.asarray(rgb, dtype=float)
    y = clip_rgb(x, low=low, high=high)
    delta = y - x
    return {
        "pixels_modified_by_clip": int(
            np.count_nonzero(np.any(np.abs(delta) > eps, axis=-1))
        ),
        "components_modified_by_clip": int(np.count_nonzero(np.abs(delta) > eps)),
        "clip_l1": float(np.sum(np.abs(delta))),
        "clip_l2": float(np.sqrt(np.sum(delta * delta))),
        "preclip_minimum": float(np.min(x)),
        "preclip_maximum": float(np.max(x)),
    }


def rgb_noise_statistics(noise_rgb: np.ndarray) -> dict[str, float]:
    """Channel covariance/correlation plus simple luminance/chroma directions.

    The luminance-like axis uses Rec.709 linear-RGB coefficients only as a
    diagnostic direction. Two chroma-like coordinates are R-G and B-G.
    """
    n = np.asarray(noise_rgb, dtype=float).reshape(-1, 3)
    cov = np.cov(n, rowvar=False)
    corr = np.corrcoef(n, rowvar=False)

    luma = n @ np.array([0.2126, 0.7152, 0.0722])
    rg = n[:, 0] - n[:, 1]
    bg = n[:, 2] - n[:, 1]

    return {
        "std_R": float(np.std(n[:, 0])),
        "std_G": float(np.std(n[:, 1])),
        "std_B": float(np.std(n[:, 2])),
        "corr_RG": float(corr[0, 1]),
        "corr_GB": float(corr[1, 2]),
        "corr_RB": float(corr[0, 2]),
        "cov_RG": float(cov[0, 1]),
        "cov_GB": float(cov[1, 2]),
        "cov_RB": float(cov[0, 2]),
        "std_luma709": float(np.std(luma)),
        "std_R_minus_G": float(np.std(rg)),
        "std_B_minus_G": float(np.std(bg)),
    }


# Illustrative only: not tied to any camera.
# Rows sum to 1 so the neutral axis (t,t,t) is preserved before clipping.
ILLUSTRATIVE_NEUTRAL_PRESERVING_CCM = np.array(
    [
        [1.45, -0.30, -0.15],
        [-0.10, 1.25, -0.15],
        [-0.05, -0.35, 1.40],
    ],
    dtype=float,
)
