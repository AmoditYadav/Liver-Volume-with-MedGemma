"""
Uncertainty Quantification: SwinUNETR vs MedGemma Disagreement.

Runs the MedGemma segmentation model on each axial slice and compares
its predictions with the SwinUNETR 3D mask to produce a disagreement map.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

# ─── Config ──────────────────────────────────────────────────────────────────

MEDGEMMA_MODEL_PATH = r"D:\Monai\model_medgemma"
DECODER_CHECKPOINT  = r"D:\Monai\results\medgemma_liver\best_model.pth"
IMG_SIZE = 224


# ─── Model singleton ─────────────────────────────────────────────────────────

_seg_model = None
_device = None


def _get_seg_model():
    """Lazy-load the MedGemma segmentation model (encoder + decoder)."""
    global _seg_model, _device
    if _seg_model is not None:
        return _seg_model, _device

    import sys
    sys.path.insert(0, r"D:\Monai")
    from medgemma_model import MedGemmaSegModel

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[LiverLens] Loading MedGemma segmentation model for uncertainty...")

    _seg_model = MedGemmaSegModel(MEDGEMMA_MODEL_PATH, img_size=IMG_SIZE)
    ckpt = torch.load(DECODER_CHECKPOINT, map_location=_device, weights_only=True)
    _seg_model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    _seg_model = _seg_model.to(_device)
    _seg_model.eval()
    print("[LiverLens] MedGemma segmentation model ready")
    return _seg_model, _device


# ─── Public API ──────────────────────────────────────────────────────────────

def compute_disagreement(
    swin_mask: np.ndarray,
    volume_norm: np.ndarray,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Compare SwinUNETR 3D mask with per-slice MedGemma predictions.

    Parameters
    ----------
    swin_mask    : (H, W, D) uint8 binary mask from SwinUNETR
    volume_norm  : (H, W, D) normalised [0, 1] CT volume
    batch_size   : slices processed per batch

    Returns
    -------
    disagreement : (H, W, D) float32 in [0, 1]
                   — abs difference between SwinUNETR and MedGemma masks
    """
    model, device = _get_seg_model()
    H, W, D = swin_mask.shape

    # Prepare MedGemma predictions for each axial slice
    medgemma_mask = np.zeros((H, W, D), dtype=np.float32)

    slices_batch = []
    indices_batch = []

    for z in range(D):
        # Extract slice, convert to 3-channel, resize to 224×224
        sl = volume_norm[:, :, z]                           # (H, W)
        sl_3ch = np.stack([sl, sl, sl], axis=0)             # (3, H, W)
        slices_batch.append(sl_3ch)
        indices_batch.append(z)

        if len(slices_batch) == batch_size or z == D - 1:
            batch_tensor = torch.tensor(
                np.array(slices_batch), dtype=torch.float32
            ).to(device)   # (B, 3, H, W)

            # Resize to 224×224 for MedGemma
            batch_resized = F.interpolate(
                batch_tensor, size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear", align_corners=False,
            )

            # Normalise to [-1, 1] (SigLIP convention)
            batch_resized = batch_resized * 2.0 - 1.0

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    preds = model(batch_resized)
            preds = torch.sigmoid(preds)                    # (B, 1, 224, 224)

            # Resize predictions back to original (H, W)
            preds_orig = F.interpolate(
                preds, size=(H, W), mode="bilinear", align_corners=False,
            )
            preds_np = (preds_orig[:, 0].cpu().numpy() > 0.5).astype(np.float32)

            for i, idx in enumerate(indices_batch):
                medgemma_mask[:, :, idx] = preds_np[i]

            slices_batch.clear()
            indices_batch.clear()

    # Compute disagreement
    disagreement = np.abs(
        swin_mask.astype(np.float32) - medgemma_mask
    )

    n_disagree = int(disagreement.sum())
    n_swin     = int(swin_mask.sum())
    pct = (n_disagree / max(n_swin, 1)) * 100
    print(f"[LiverLens] Disagreement: {n_disagree} voxels "
          f"({pct:.1f}% of liver region)")

    return disagreement
