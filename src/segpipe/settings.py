# =============================================================================
# Pipeline Settings (Pure Python Config)
#
# Summary:
#   Single source of truth for all pipeline inputs/outputs, organ definitions,
#   QC limits, and auto-repair parameters. Replaces YAML configs entirely.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


# ------------------------------------------
# INPUTS / OUTPUTS
# ------------------------------------------
INPUT_BODY  = r"data\input\AC09-06-2022_final_skin.nii.gz"
INPUT_BONES = r"data\input\AC09-06-2022_bones_separate_labels.nii.gz"

OUTPUT_NIFTI  = r"data\output\processed_anatomy_combined_2mm.nii.gz"
OUTPUT_VTI    = r"data\output\processed_wbseg_combined_2mm.vti"
OUTPUT_LOG    = r"data\output\processing_log.txt"
OUTPUT_QC_CSV = r"data\output\segmentation_qc_report.csv"



# ------------------------------------------
# GLOBAL PIPELINE PARAMS
# ------------------------------------------
TARGET_SPACING_MM: Tuple[float, float, float] = (2.0, 2.0, 2.0)

# Skin
SKIN_LABEL: int = 73
GENERATE_SKIN: bool = True
SKIN_THICK_MM: float = 4.0

# Overlay coupled dilation + smoothing
DEFAULT_DILATION_MM: float = 2.0
DILATION_MM_OVERRIDES: Dict[int, float] = {}  # optional per-label overrides
SIGMA_ALPHA: float = 0.35
MEDIAN_BETA: float = 1.00

# Refine
DUST_MAX_VOX: int = 20
SMALL_FRAC: float = 0.01
MULTIPART_MIN_CC: int = 6
MULTIPART_RATIO: float = 2.0
FINAL_POLISH_DUST_VOX: int = 2


# ------------------------------------------
# ORGAN LIST (your full list)
# Each item: (id, name, method, param)
# ------------------------------------------
ORGAN_LIST: List[Tuple[int, str, str, Any]] = [
    (1, "Spleen", "gaussian", 3.0), (2, "Kidneys", "gaussian", 3.0),
    (4, "Gallbladder", "gaussian", 2.0), (5, "Liver", "gaussian", 3.0),
    (6, "Stomach", "gaussian", 3.0), (7, "Pancreas", "gaussian", 3.0),
    (8, "Adrenal Glands", "median", 2), (10, "Lungs", "gaussian", 3.0),
    (15, "Esophagus", "gaussian", 3.0), (16, "Trachea", "gaussian", 3.0),
    (17, "Thyroid Gland", "gaussian", 2.0), (18, "Intestine", "gaussian", 1.0),
    (19, "Duodenum", "gaussian", 3.0), (21, "Bladder", "gaussian", 3.0),
    (22, "Prostate", "gaussian", 3.0), (24, "Heart", "gaussian", 3.0),
    (26, "Vessels", "median", 1.5), (52, "Spinal Cord", "gaussian", 2.0),
    (64, "Costal Cartilages", "median", 2.0), (65, "Subcutaneous Fat", "gaussian", 2.0),
    (66, "Muscle", "gaussian", 1.5), (67, "Visceral Fat", "gaussian", 2.0),
    (68, "Vertebral Discs", "median", 1), (71, "Spinal Canal", "median", 3),
    (74, "Parotid Glands", "gaussian", 2.0), (75, "Air (Sinuses)", "median", 4),
    (77, "CSF", "median", 2.0), (78, "Gray Matter", "median", 3.0),
    (79, "White Matter", "median", 3.0), (80, "Cerebellum", "gaussian", 1.0),
    (81, "Optic Nerve", "median", 2), (82, "Mucosa", "median", 2),
    (83, "Eyeball", "median", 2), (109, "Genitals", "gaussian", 2),
    (117, "Uterus", "gaussian", 2),

    # Detailed bones
    (23, "Sacrum", "gaussian", 1.0), (42, "Humerus L", "gaussian", 1.0),
    (43, "Humerus R", "gaussian", 1.0), (44, "Scapula L", "gaussian", 1.0),
    (45, "Scapula R", "gaussian", 1.0), (46, "Clavicula L", "gaussian", 1.0),
    (47, "Clavicula R", "gaussian", 1.0), (48, "Femur L", "gaussian", 1.0),
    (49, "Femur R", "gaussian", 1.0), (50, "Hip L", "gaussian", 1.0),
    (51, "Hip R", "gaussian", 1.0), (63, "Sternum", "gaussian", 1.0),
    (72, "Bone Other", "gaussian", 1.0), (91, "Skull", "gaussian", 1.0),
]

VERTEBRAE = set(list(range(212, 218)) + list(range(221, 233)) + list(range(241, 246)))
for v in list(range(212, 218)) + list(range(221, 233)) + list(range(241, 246)):
    ORGAN_LIST.append((v, f"Vertebra_{v}", "gaussian", 1.0))


# ------------------------------------------
# QC CONFIG (from your script)
# ------------------------------------------
QC_IGNORE_LABELS = set()

OK_MULTI = {2, 10, 17, 26, 74, 81, 83, 68, 72, 75}
OK_FRAGMENTED = {18, 64, 65, 66, 67, 77, 78, 79, 91}
OK_THIN = {52, 64, 66, 15, 26, 19, 109, 71}

# Special expected component counts (max allowed)
# Vertebral discs (68) should have up to 22 components
MAX_COMPONENTS: Dict[int, int] = {
    68: 22
}

DEFAULT_LIMITS = {
    "min_vol": 2000.0,
    "min_spher": 0.35,
    "max_ecc": 0.96,
}

LABEL_LIMITS: Dict[int, Dict[str, float]] = {
    15: {"min_vol": 300.0,   "min_spher": 0.05, "max_ecc": 0.999},
    16: {"min_vol": 300.0,   "min_spher": 0.05, "max_ecc": 0.995},
    19: {"min_vol": 300.0,   "min_spher": 0.05, "max_ecc": 0.999},
    26: {"min_vol": 500.0,   "min_spher": 0.03, "max_ecc": 0.999},
    52: {"min_vol": 150.0,   "min_spher": 0.03, "max_ecc": 0.999},
    64: {"min_vol": 500.0,   "min_spher": 0.01, "max_ecc": 0.999},
    66: {"min_vol": 2000.0,  "min_spher": 0.01, "max_ecc": 0.999},
    67: {"min_vol": 2000.0,  "min_spher": 0.01, "max_ecc": 0.98},
    71: {"min_vol": 500.0,   "min_spher": 0.03, "max_ecc": 0.999},
    73: {"min_vol": 10000.0, "min_spher": 0.02, "max_ecc": 0.99},
    74: {"min_vol": 300.0,   "min_spher": 0.20, "max_ecc": 0.995},
    77: {"min_vol": 1000.0,  "min_spher": 0.01, "max_ecc": 0.98},
    78: {"min_vol": 1000.0,  "min_spher": 0.01, "max_ecc": 0.98},
    79: {"min_vol": 1000.0,  "min_spher": 0.01, "max_ecc": 0.98},
    81: {"min_vol": 50.0,    "min_spher": 0.03, "max_ecc": 0.999},
    83: {"min_vol": 300.0,   "min_spher": 0.20, "max_ecc": 0.995},
    109: {"min_vol": 300.0,  "min_spher": 0.05, "max_ecc": 0.999},

    # Relax elongation only
    18: {"max_ecc": 0.999},
    42: {"max_ecc": 0.999}, 43: {"max_ecc": 0.999},
    44: {"max_ecc": 0.999}, 45: {"max_ecc": 0.999},
    46: {"max_ecc": 0.999}, 47: {"max_ecc": 0.999},
    48: {"max_ecc": 0.999}, 49: {"max_ecc": 0.999},
    63: {"max_ecc": 0.999},
    65: {"max_ecc": 0.999},
    68: {"max_ecc": 0.999},
    72: {"max_ecc": 0.999},
}

for v in VERTEBRAE:
    LABEL_LIMITS.setdefault(int(v), {})
    LABEL_LIMITS[int(v)]["max_ecc"] = 0.98


# ------------------------------------------
# AUTO-REPAIR CONFIG
# ------------------------------------------
ALLOW_OVERWRITE_LABELS = (0, 65, 66)
MAX_CLOSE_ITERS_DEFAULT = 4
MAX_CLOSE_ITERS_CARTILAGE = 6
BRIDGE_RADIUS_DEFAULT = 1
BRIDGE_RADIUS_CARTILAGE = 2
BRIDGE_MAX_ITERS = 10


# ------------------------------------------
# Convenience maps
# ------------------------------------------
ORGAN_MAP = {lid: name for (lid, name, *_rest) in ORGAN_LIST}
ORGAN_MAP[SKIN_LABEL] = "Skin"


@dataclass
class Settings:
    input_body: str = INPUT_BODY
    input_bones: str = INPUT_BONES
    out_nifti: str = OUTPUT_NIFTI
    out_vti: str = OUTPUT_VTI
    out_log: str = OUTPUT_LOG
    out_qc_csv: str = OUTPUT_QC_CSV


DEFAULT_SETTINGS = Settings()
