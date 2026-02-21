"""
Stage B — Deterministic Bridge: Liver Metric Extraction.

Computes clinically meaningful metrics from the SwinUNETR binary mask
and the normalised CT volume.  All numbers are calculated mathematically
— no LLM is involved here — to avoid hallucinated measurements.
"""

import numpy as np


# ─── HU Conversion ──────────────────────────────────────────────────────────

def _to_hu(vol_norm: np.ndarray, a_min=-200.0, a_max=200.0) -> np.ndarray:
    """Inverse of ScaleIntensityRange: normalised [0,1] → approximate HU."""
    return vol_norm * (a_max - a_min) + a_min


# ─── Public API ──────────────────────────────────────────────────────────────

def compute_metrics(
    mask: np.ndarray,
    volume: np.ndarray,
    voxel_spacing: tuple,
    effective_spacing: tuple = None,
    a_min: float = -200.0,
    a_max: float = 200.0,
) -> dict:
    """
    Extract evidence dict for the LLM from mask + CT volume.

    Parameters
    ----------
    mask              : (H, W, D) binary uint8
    volume            : (H, W, D) normalised [0, 1]
    voxel_spacing     : (sx, sy, sz) in mm from the NIfTI header
    effective_spacing : (sx, sy, sz) in mm — actual voxel size after
                        Spacingd + Resized.  If None, falls back to 1mm isotropic.
    a_min, a_max      : HU clipping bounds used during preprocessing

    Returns
    -------
    dict with:
      volume_cc        – liver volume in cubic centimetres
      mean_hu          – mean Hounsfield Unit inside liver
      std_hu           – std dev of HU (texture / heterogeneity)
      median_hu        – median HU
      steatosis_flag   – True if mean_hu < 40
      hepatomegaly_flag– True if volume_cc > 1800
      low_frac_40      – fraction of liver voxels below 40 HU
      low_frac_50      – fraction below 50 HU
      hist_counts      – histogram bin counts (for XAI plot)
      hist_bins        – histogram bin edges
      n_liver_voxels   – total liver voxels
    """
    hu = _to_hu(volume, a_min, a_max)
    liver_vals = hu[mask > 0]
    n_vox = int(liver_vals.size)

    if n_vox == 0:
        return _empty_metrics()

    # Effective voxel volume in mm³
    if effective_spacing is not None:
        sx, sy, sz = effective_spacing
    else:
        sx, sy, sz = (1.0, 1.0, 1.0)
    voxel_vol_mm3 = sx * sy * sz
    volume_cc = n_vox * voxel_vol_mm3 / 1000.0

    mean_hu  = float(liver_vals.mean())
    std_hu   = float(liver_vals.std())
    median_hu = float(np.median(liver_vals))
    p10, p25, p75, p90 = [float(v) for v in np.percentile(liver_vals, [10, 25, 75, 90])]

    low_frac_40 = float(np.mean(liver_vals < 40.0))
    low_frac_50 = float(np.mean(liver_vals < 50.0))

    counts, bins = np.histogram(liver_vals, bins=50, range=(a_min, a_max))

    return {
        "volume_cc":         round(volume_cc, 1),
        "mean_hu":           round(mean_hu, 2),
        "std_hu":            round(std_hu, 2),
        "median_hu":         round(median_hu, 2),
        "p10_hu":            round(p10, 1),
        "p25_hu":            round(p25, 1),
        "p75_hu":            round(p75, 1),
        "p90_hu":            round(p90, 1),
        "steatosis_flag":    mean_hu < 40.0,
        "hepatomegaly_flag": volume_cc > 1800.0,
        "low_frac_40":       round(low_frac_40, 4),
        "low_frac_50":       round(low_frac_50, 4),
        "hist_counts":       counts.tolist(),
        "hist_bins":         bins.tolist(),
        "n_liver_voxels":    n_vox,
    }


def _empty_metrics() -> dict:
    return {
        "volume_cc": 0.0, "mean_hu": float("nan"), "std_hu": float("nan"),
        "median_hu": float("nan"), "p10_hu": float("nan"), "p25_hu": float("nan"),
        "p75_hu": float("nan"), "p90_hu": float("nan"),
        "steatosis_flag": False, "hepatomegaly_flag": False,
        "low_frac_40": 0.0, "low_frac_50": 0.0,
        "hist_counts": [], "hist_bins": [], "n_liver_voxels": 0,
    }
