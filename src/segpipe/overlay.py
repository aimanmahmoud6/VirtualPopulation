# =============================================================================
# Overlay Step (Dilation + Smoothing + Winner-Takes-All)
#
# Summary:
#   Implements your organ loop: per-label dilation, smoothing, and overlay into
#   a combined label volume using max-probability selection.
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

        base_mask = (raw_crop_zyx == lid)
        if not np.any(base_mask):
            continue
        

        dil_mm = float(dil_over.get(lid, default_dil_mm))
        dil_it = dilation_iters_from_mm(dil_mm, vox_mm)

        mask_bin = ndimage.binary_dilation(base_mask, iterations=dil_it) if dil_it > 0 else base_mask
        mask_f = mask_bin.astype(np.float32)

        eff_param = effective_smoothing(method, base_param, dil_it, sigma_alpha, median_beta)

        if method == "gaussian":
            smoothed = ndimage.gaussian_filter(mask_f, sigma=float(eff_param))
        else:
            smoothed = ndimage.median_filter(mask_f, size=int(eff_param))

        better = smoothed > max_prob
        final_vol[better] = lid
        max_prob[better]  = smoothed[better]


        # --------------------------------------------------
        # HARDCODE TEST: Spinal Cord (52) = smoothed only
        # (no winner-takes-all competition)
        # --------------------------------------------------
        if int(lid) == 52:
            thr = 0.5  # test threshold; tune if needed (0.3–0.7)
            cord = smoothed > thr

            # Write cord directly
            final_vol[cord] = 52

            # Protect it from being overwritten later:
            # set max_prob high where cord is present
            max_prob[cord] = 1.0

            continue


    return final_vol
