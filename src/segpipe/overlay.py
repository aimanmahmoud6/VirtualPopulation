# =============================================================================
# Overlay Step (Dilation + Smoothing + Winner-Takes-All)
#
# Summary:
#   Per-label dilation, smoothing, and overlay into a combined label volume
#   using max-probability selection.
#
#   NOTE:
#     Label 52 (Spinal Cord) is intentionally SKIPPED here and applied later
#     (smoothed-only) just before QC in pipeline.py.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

from typing import Dict, List, Tuple
import numpy as np
import scipy.ndimage as ndimage
from .utils import dilation_iters_from_mm, effective_smoothing


def overlay_organs(raw_crop_zyx: np.ndarray,
                  organs: List[Dict],
                  spacing_xyz: Tuple[float, float, float],
                  overlay_cfg: Dict):
    final_vol = np.zeros_like(raw_crop_zyx, dtype=np.uint16)
    max_prob  = np.zeros_like(raw_crop_zyx, dtype=np.float32) + 0.05

    vox_mm = float(spacing_xyz[0])

    default_dil_mm = float(overlay_cfg["default_dilation_mm"])
    dil_over = overlay_cfg.get("dilation_mm_overrides", {}) or {}
    sigma_alpha = float(overlay_cfg["sigma_alpha"])
    median_beta = float(overlay_cfg["median_beta"])

    for o in organs:
        lid = int(o["id"])
        method = str(o["method"])
        base_param = o["param"]

        # Skip spinal cord here; applied later in pipeline (smoothed-only)
        if lid == 52:
            continue

        base_mask = (raw_crop_zyx == lid)
        if not np.any(base_mask):
            continue

        dil_mm = float(dil_over.get(lid, default_dil_mm))
        dil_it = dilation_iters_from_mm(dil_mm, vox_mm)

        mask_bin = ndimage.binary_dilation(base_mask, iterations=dil_it) if dil_it > 0 else base_mask

        # RULE: Spinal canal (71) dilation must NOT invade spinal cord (52)
        # (cord is applied later, but this still reduces conflicts)
        if lid == 71:
            mask_bin[raw_crop_zyx == 52] = False

        mask_f = mask_bin.astype(np.float32)

        eff_param = effective_smoothing(method, base_param, dil_it, sigma_alpha, median_beta)

        if method == "gaussian":
            smoothed = ndimage.gaussian_filter(mask_f, sigma=float(eff_param))
        else:
            smoothed = ndimage.median_filter(mask_f, size=int(eff_param))

        better = smoothed > max_prob
        final_vol[better] = lid
        max_prob[better]  = smoothed[better]

    return final_vol
