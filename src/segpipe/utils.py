# =============================================================================
# Utilities
#
# Summary:
#   Small reusable helpers (logger, smoothing parameter adjustment, organ map).
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

import sys
from typing import Dict
import numpy as np

class Logger:
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, tb):
        sys.stdout = self.terminal
        self.log.close()

    def write(self, msg: str):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def dilation_iters_from_mm(dil_mm: float, spacing_mm: float) -> int:
    if dil_mm <= 0:
        return 0
    return int(np.ceil(float(dil_mm) / float(spacing_mm)))

def effective_smoothing(method: str, base_param, dil_iters: int, sigma_alpha: float, median_beta: float):
    if method == "gaussian":
        base_sigma = float(base_param)
        return float(max(0.1, base_sigma + float(sigma_alpha) * float(dil_iters)))
    base_size = float(base_param)
    size_eff = max(1.0, base_size + float(median_beta) * float(dil_iters))
    return int(max(1, round(size_eff)))

def make_organ_map(organs) -> Dict[int, str]:
    return {int(o["id"]): str(o["name"]) for o in organs}
