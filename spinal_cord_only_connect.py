#!/usr/bin/env python3
"""
spinal_cord_only_connect.py

- Load segmentation NIfTI (binary or multi-label)
- Extract spinal cord mask
- Visualize (ax/cor/sag + MIP) and save PNG
- Print NCC (original vs connected)
- Connect multiple components by iterative "safe closing" in mm steps (never removes original voxels)
- Save connected mask NIfTI

Dependencies:
  pip install SimpleITK numpy scipy matplotlib
"""

import argparse
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------
def sitk_to_zyx(img: sitk.Image) -> np.ndarray:
    """SimpleITK -> numpy array in Z,Y,X order (sitk.GetArrayFromImage)."""
    return sitk.GetArrayFromImage(img)


def zyx_to_sitk(arr_zyx: np.ndarray, ref: sitk.Image, dtype=sitk.sitkUInt8) -> sitk.Image:
    """numpy Z,Y,X -> SimpleITK image with same geometry as ref."""
    out = sitk.GetImageFromArray(arr_zyx.astype(np.uint8))
    out.CopyInformation(ref)
    return sitk.Cast(out, dtype)


def list_nonzero_labels(seg_zyx: np.ndarray, max_show=50):
    labels = np.unique(seg_zyx)
    labels = labels[labels != 0]
    if labels.size == 0:
        return []
    labels = labels.astype(int).tolist()
    if len(labels) > max_show:
        return labels[:max_show] + ["..."]
    return labels


def make_ball_struct_mm(radius_mm: float, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """
    Create a 3D ellipsoidal structuring element in Z,Y,X array order,
    corresponding to a ball of radius_mm in physical space.

    spacing_xyz = (sx, sy, sz) in mm
    """
    sx, sy, sz = map(float, spacing_xyz)

    # Convert to voxel radii along each axis
    rx = max(1, int(np.ceil(radius_mm / sx)))
    ry = max(1, int(np.ceil(radius_mm / sy)))
    rz = max(1, int(np.ceil(radius_mm / sz)))

    # Grid in voxel units -> physical distance
    zz, yy, xx = np.mgrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    dist2 = (xx * sx) ** 2 + (yy * sy) ** 2 + (zz * sz) ** 2
    se = dist2 <= (radius_mm ** 2 + 1e-9)
    return se


def connected_components_count(mask_zyx: np.ndarray) -> tuple[int, np.ndarray]:
    """Return number of connected components (26-connectivity) and labeled map."""
    if mask_zyx.dtype != np.bool_:
        mask_zyx = mask_zyx.astype(bool)
    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    lab, n = ndi.label(mask_zyx, structure=structure)
    return int(n), lab


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalized cross-correlation between two arrays.
    Uses only bounding box of union to avoid being dominated by zeros.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    union = (a != 0) | (b != 0)
    if not np.any(union):
        return 1.0

    coords = np.argwhere(union)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    aa = a[z0:z1, y0:y1, x0:x1].ravel()
    bb = b[z0:z1, y0:y1, x0:y1].ravel()

    aa_m = aa.mean()
    bb_m = bb.mean()
    aa_c = aa - aa_m
    bb_c = bb - bb_m
    denom = (np.sqrt((aa_c * aa_c).sum()) * np.sqrt((bb_c * bb_c).sum()))
    if denom < 1e-12:
        # If one is constant, NCC isn't meaningful; fall back to overlap-ish score
        # but keep it defined.
        return float(1.0 if np.allclose(aa, bb) else 0.0)
    return float((aa_c * bb_c).sum() / denom)


# ---------------------------
# Core “safe connect” method
# ---------------------------
def connect_mask_safe_closing(
    mask_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    start_mm: float = 0.5,
    step_mm: float = 0.5,
    max_mm: float = 8.0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Iteratively apply closing with increasing radius, but NEVER remove original voxels:
        connected = mask OR closing(mask, se)

    This is good for thin anatomical structures (like spinal cord) where you want
    to bridge small gaps without losing any piece.

    Returns: (new_mask, report)
    """
    mask0 = mask_zyx.astype(bool)
    cur = mask0.copy()

    n0, _ = connected_components_count(cur)
    if verbose:
        print(f"[INFO] Components before: {n0}")

    report = {
        "components_before": n0,
        "components_after": None,
        "radii_mm_tried": [],
        "components_by_radius": [],
        "voxels_added_by_radius": [],
        "radius_mm_used": None,
    }

    if n0 <= 1:
        report["components_after"] = n0
        report["radius_mm_used"] = 0.0
        return cur, report

    r = float(start_mm)
    while r <= float(max_mm) + 1e-9:
        se = make_ball_struct_mm(r, spacing_xyz)

        closed = ndi.binary_closing(cur, structure=se)
        new = cur | closed  # SAFE: never removes original voxels

        n, _ = connected_components_count(new)
        added = int(new.sum() - cur.sum())

        report["radii_mm_tried"].append(float(r))
        report["components_by_radius"].append(int(n))
        report["voxels_added_by_radius"].append(int(added))

        if verbose:
            print(f"[TRY] r={r:.2f} mm -> components={n} | added_voxels={added}")

        cur = new

        if n <= 1:
            report["radius_mm_used"] = float(r)
            break

        r += float(step_mm)

    n_final, _ = connected_components_count(cur)
    report["components_after"] = int(n_final)
    if report["radius_mm_used"] is None:
        report["radius_mm_used"] = float(max_mm)

    if verbose:
        print(f"[INFO] Components after: {n_final} (radius_used={report['radius_mm_used']:.2f} mm)")
    return cur, report


# ---------------------------
# Visualization
# ---------------------------
def save_qc_figure(mask0: np.ndarray, mask1: np.ndarray, out_png: str):
    """
    Save 2x2 panel:
      - axial mid
      - coronal mid
      - sagittal mid
      - MIP (axial projection)
    Shows original in one contour and connected in another.
    """
    m0 = mask0.astype(bool)
    m1 = mask1.astype(bool)

    # Find a decent center based on union
    u = m0 | m1
    if np.any(u):
        coords = np.argwhere(u)
        cz, cy, cx = np.round(coords.mean(axis=0)).astype(int)
    else:
        cz, cy, cx = np.array(m0.shape) // 2

    def _clip(v, lo, hi):
        return int(np.clip(v, lo, hi))

    cz = _clip(cz, 0, m0.shape[0]-1)
    cy = _clip(cy, 0, m0.shape[1]-1)
    cx = _clip(cx, 0, m0.shape[2]-1)

    ax0 = m0[cz, :, :]
    ax1 = m1[cz, :, :]

    cor0 = m0[:, cy, :]
    cor1 = m1[:, cy, :]

    sag0 = m0[:, :, cx]
    sag1 = m1[:, :, cx]

    mip0 = np.max(m0.astype(np.uint8), axis=0)
    mip1 = np.max(m1.astype(np.uint8), axis=0)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)

    panels = [
        ("Axial (mid Z)", ax0, ax1),
        ("Coronal (mid Y)", cor0, cor1),
        ("Sagittal (mid X)", sag0, sag1),
        ("MIP (max over Z)", mip0.astype(bool), mip1.astype(bool)),
    ]

    for i, (title, a, b) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.imshow((a | b).astype(np.uint8), interpolation="nearest")
        # contours for difference
        ax.contour(a.astype(np.uint8), levels=[0.5], linewidths=1)
        ax.contour(b.astype(np.uint8), levels=[0.5], linewidths=1, linestyles="--")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Spinal cord mask QC (solid=original, dashed=connected)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input NIfTI (segmentation)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output NIfTI (spinal cord connected mask)")
    ap.add_argument("--label", type=int, default=None, help="Spinal cord label id (if multi-label).")
    ap.add_argument("--start_mm", type=float, default=0.5, help="Start radius (mm) for connecting.")
    ap.add_argument("--step_mm", type=float, default=0.5, help="Step radius (mm) for connecting.")
    ap.add_argument("--max_mm", type=float, default=8.0, help="Max radius (mm) for connecting.")
    ap.add_argument("--viz", default=None, help="Optional QC PNG output path.")
    args = ap.parse_args()

    img = sitk.ReadImage(args.in_path)
    spacing_xyz = img.GetSpacing()  # (x,y,z)
    seg = sitk_to_zyx(img)

    print(f"[INFO] Loaded: {args.in_path}")
    print(f"[INFO] Shape (Z,Y,X): {seg.shape}")
    print(f"[INFO] Spacing (X,Y,Z) mm: {spacing_xyz}")

    # Extract spinal cord mask
    uniq = np.unique(seg)
    nonzero = uniq[uniq != 0]

    if args.label is not None:
        mask0 = (seg == args.label)
        if not np.any(mask0):
            labels = list_nonzero_labels(seg)
            raise SystemExit(
                f"[ERROR] Label {args.label} not found / empty.\n"
                f"Non-zero labels present (sample): {labels}"
            )
        print(f"[INFO] Using label={args.label} for spinal cord. Voxels={int(mask0.sum())}")

    else:
        # Auto: if already binary or single nonzero label, treat as spinal cord mask
        if seg.dtype != np.bool_ and nonzero.size == 1:
            only_label = int(nonzero[0])
            mask0 = (seg == only_label)
            print(f"[INFO] Auto-detected single label={only_label}. Voxels={int(mask0.sum())}")
        elif seg.dtype == np.bool_:
            mask0 = seg.astype(bool)
            print(f"[INFO] Input is boolean mask. Voxels={int(mask0.sum())}")
        else:
            labels = list_nonzero_labels(seg)
            raise SystemExit(
                "[ERROR] Multi-label segmentation detected and --label not provided.\n"
                f"Non-zero labels present (sample): {labels}\n"
                "Re-run with: --label <SPINAL_CORD_LABEL_ID>"
            )

    # Component stats before
    n_before, _ = connected_components_count(mask0)
    print(f"[METRIC] components_before = {n_before}")

    # Connect
    mask1, rep = connect_mask_safe_closing(
        mask0,
        spacing_xyz=spacing_xyz,
        start_mm=args.start_mm,
        step_mm=args.step_mm,
        max_mm=args.max_mm,
        verbose=True,
    )

    # NCC metric
    val_ncc = ncc(mask0.astype(np.float32), mask1.astype(np.float32))
    print(f"[METRIC] NCC(original, connected) = {val_ncc:.6f}")

    # Component stats after
    n_after, _ = connected_components_count(mask1)
    print(f"[METRIC] components_after = {n_after}")
    print(f"[METRIC] voxels_original = {int(mask0.sum())}")
    print(f"[METRIC] voxels_connected = {int(mask1.sum())}")
    print(f"[METRIC] voxels_added = {int(mask1.sum() - mask0.sum())}")
    print(f"[INFO] radius_used_mm = {rep['radius_mm_used']:.2f}")

    # Save output
    out_img = zyx_to_sitk(mask1.astype(np.uint8), img, dtype=sitk.sitkUInt8)
    sitk.WriteImage(out_img, args.out_path)
    print(f"[INFO] Saved: {args.out_path}")

    # QC figure
    if args.viz:
        save_qc_figure(mask0, mask1, args.viz)
        print(f"[INFO] Saved QC PNG: {args.viz}")


if __name__ == "__main__":
    main()
