"""
Test-Time Augmentation (TTA) — "High-Precision Mode".

Rotates the preprocessed volume 90° around the Z-axis, re-segments,
rotates the mask back, and averages with the original mask.
This reduces boundary noise and improves segmentation reliability.
"""

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from liverlens.segmentation import (
    _get_model,
    _build_transforms,
    _build_pre_resize_transforms,
    ROI_SIZE,
    SW_OVERLAP,
    SPATIAL_SIZE,
    A_MIN,
    A_MAX,
)

import nibabel as nib


def segment_with_tta(nifti_path: str) -> dict:
    """
    Run liver segmentation with Test-Time Augmentation.

    Process:
        1. Normal inference → mask₁
        2. Rotate volume 90° (k=1 around Z) → inference → mask₂ → rotate back
        3. Average(mask₁, mask₂) → threshold at 0.5

    Returns
    -------
    Same dict as segmentation.segment(), but the mask is TTA-averaged.
    """
    model, device = _get_model()

    # ── Read header ──
    nii = nib.load(nifti_path)
    voxel_spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])

    # ── Pre-resize shape for effective spacing ──
    pre_resize_tfm = _build_pre_resize_transforms()
    pre_data = pre_resize_tfm({"image": nifti_path})
    resampled_shape = pre_data["image"].shape[1:]

    effective_spacing = tuple(
        float(resampled_shape[i]) / float(SPATIAL_SIZE[i])
        for i in range(3)
    )

    # ── Full transform ──
    transforms = _build_transforms()
    data = transforms({"image": nifti_path})
    tensor = data["image"].unsqueeze(0).to(device)  # [1, 1, H, W, D]
    vol = tensor[0, 0].cpu().numpy()

    # ── Pass 1: Normal inference ──
    print("[LiverLens TTA] Pass 1/2: standard orientation...")
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out1 = sliding_window_inference(
                tensor, roi_size=ROI_SIZE,
                sw_batch_size=2, predictor=model, overlap=SW_OVERLAP,
            )
    # Softmax probabilities for class 1 (liver)
    probs1 = torch.softmax(out1, dim=1)[0, 1].cpu().numpy()  # (H, W, D)

    # ── Pass 2: Rotate 90° around Z-axis ──
    print("[LiverLens TTA] Pass 2/2: rotated 90° orientation...")
    # Rotate the 5D tensor: dims are [B, C, H, W, D], rotate in (H, W) plane
    tensor_rot = torch.rot90(tensor, k=1, dims=[2, 3])

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out2 = sliding_window_inference(
                tensor_rot, roi_size=ROI_SIZE,
                sw_batch_size=2, predictor=model, overlap=SW_OVERLAP,
            )
    probs2_rot = torch.softmax(out2, dim=1)[0, 1].cpu().numpy()

    # Rotate mask back (-90°)
    probs2 = np.rot90(probs2_rot, k=-1, axes=(0, 1))

    # ── Average and threshold ──
    avg_probs = (probs1 + probs2) / 2.0
    mask = (avg_probs > 0.5).astype(np.uint8)

    n_original = int((probs1 > 0.5).sum())
    n_tta = int(mask.sum())
    print(f"[LiverLens TTA] Original: {n_original} voxels → "
          f"TTA: {n_tta} voxels (Δ {n_tta - n_original:+d})")

    return {
        "mask_3d":           mask,
        "volume_3d":         vol,
        "voxel_spacing":     voxel_spacing,
        "effective_spacing":  effective_spacing,
        "n_slices":          mask.shape[2],
    }
