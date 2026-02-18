# =============================================================================
# Refinement Step (Orphans + Dust Polish)
#
# Summary:
#   Your fine-refine stage for orphan component removal and reassignment,
#   followed by a final "dust" voxel reassignment polish.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

import numpy as np
import scipy.ndimage as ndimage

def fine_refine_remove_orphans_and_reassign(final_vol: np.ndarray,
                                           organ_map: dict,
                                           dust_max_vox: int,
                                           small_frac: float,
                                           multipart_min_cc: int,
                                           multipart_ratio: float):
    print("Fine refining: removing tiny/small components + reassigning orphans...")
    vol = final_vol.copy()
    orphan_mask = np.zeros(vol.shape, dtype=bool)
    struct = np.ones((3, 3, 3), dtype=bool)

    labels = np.unique(vol)
    labels = labels[labels != 0]

    per_label = []

    for lid in labels:
        lid = int(lid)
        m = (vol == lid)
        tot = int(m.sum())
        if tot == 0:
            continue

        lab_cc, ncc = ndimage.label(m, structure=struct)
        if ncc <= 1:
            per_label.append((lid, tot, int(ncc), [tot], 0, "single"))
            continue

        sizes = ndimage.sum(m, lab_cc, index=np.arange(1, ncc + 1)).astype(np.int64)
        order = np.argsort(-sizes)
        sizes_sorted = sizes[order]
        fracs = sizes / float(tot)

        ref_idx = 1 if ncc > 1 else 0
        ratio = float(sizes_sorted[0] / max(1, sizes_sorted[ref_idx]))
        is_multipart = (ncc >= multipart_min_cc) and (ratio < multipart_ratio)

        if is_multipart:
            orphan_cids = np.where(sizes <= dust_max_vox)[0] + 1
            mode = "multipart(dust-only)"
        else:
            orphan_cids = np.where((sizes <= dust_max_vox) | (fracs < small_frac))[0] + 1
            mode = f"non-multipart(dust or <{small_frac*100:.1f}%)"

        orphan_vox = 0
        if orphan_cids.size > 0:
            om = np.isin(lab_cc, orphan_cids)
            orphan_vox = int(np.sum(om))
            if orphan_vox > 0:
                orphan_mask |= om

        per_label.append((lid, tot, int(ncc), sizes_sorted.tolist(), orphan_vox, mode))

    print("\n[Refine] Per-label components + orphan decision:")
    print(f"{'ID':>4} | {'Name':<18} | {'Vox':>9} | {'nCC':>4} | {'TopSizes':<34} | {'OrphanVox':>9} | Mode")
    print("-" * 120)
    per_label.sort(key=lambda x: x[0])
    for lid, tot, ncc, sizes_sorted, orphan_vox, mode in per_label:
        name = organ_map.get(lid, f"Label_{lid}")
        top = sizes_sorted[:6]
        top_str = ",".join(str(int(x)) for x in top) + (",..." if len(sizes_sorted) > 6 else "")
        print(f"{lid:>4} | {name:<18} | {tot:>9} | {ncc:>4} | {top_str:<34} | {orphan_vox:>9} | {mode}")

    if not np.any(orphan_mask):
        print("\n[Refine] No orphans found. No reassignment performed.")
        return vol

    n_orphans_total = int(orphan_mask.sum())
    vol[orphan_mask] = 0

    sources = (vol != 0)
    if not np.any(sources):
        print("\n[Refine] Warning: all labels removed? Skipping reassignment.")
        return vol

    _, inds = ndimage.distance_transform_edt(~sources, return_indices=True)
    nearest_labels = vol[tuple(inds)]
    vol[orphan_mask] = nearest_labels[orphan_mask]

    print(f"\n[Refine] Reassigned orphan voxels: {n_orphans_total}")
    return vol

def remove_label_dust_inplace(vol: np.ndarray, dust_max_vox: int):
    struct = np.ones((3,3,3), dtype=bool)
    orphan_mask = np.zeros(vol.shape, dtype=bool)

    labels = np.unique(vol)
    labels = labels[labels != 0]

    for lid in labels:
        m = (vol == int(lid))
        if not np.any(m):
            continue
        cc, ncc = ndimage.label(m, structure=struct)
        if ncc <= 1:
            continue
        sizes = ndimage.sum(m, cc, index=np.arange(1, ncc+1)).astype(np.int64)
        orphan_cids = np.where(sizes <= dust_max_vox)[0] + 1
        if orphan_cids.size == 0:
            continue
        orphan_mask |= np.isin(cc, orphan_cids)

    if not np.any(orphan_mask):
        print(f"[Polish] No dust <= {dust_max_vox} vox found.")
        return vol

    n_orph = int(orphan_mask.sum())
    vol[orphan_mask] = 0

    sources = (vol != 0)
    _, inds = ndimage.distance_transform_edt(~sources, return_indices=True)
    nearest_labels = vol[tuple(inds)]
    vol[orphan_mask] = nearest_labels[orphan_mask]

    print(f"[Polish] Reassigned {n_orph} dust voxels (<= {dust_max_vox}).")
    return vol
