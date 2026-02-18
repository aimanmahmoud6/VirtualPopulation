# =============================================================================
# Post-QC Repairs (Connect, else Reassign Extra Components)
#
# Summary:
#   For labels with unexpected extra connected components:
#     1) Try add-only closing (never deletes label voxels)
#     2) Try minimal bridging (add-only)
#     3) If still too many components: keep the largest K components and
#        reassign all remaining voxels to nearest surrounding labels.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

from typing import Sequence, Optional
import numpy as np
import scipy.ndimage as ndimage


def connected_components_26(mask: np.ndarray):
    struct26 = np.ones((3, 3, 3), dtype=bool)
    cc, ncc = ndimage.label(mask.astype(bool), structure=struct26)
    return cc, int(ncc)


def bresenham3d(p0, p1):
    # p0, p1 are (z, y, x)
    z0, y0, x0 = map(int, p0)
    z1, y1, x1 = map(int, p1)

    dz = abs(z1 - z0)
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)

    sz = 1 if z1 >= z0 else -1
    sy = 1 if y1 >= y0 else -1
    sx = 1 if x1 >= x0 else -1

    zz, yy, xx = [], [], []

    if dx >= dy and dx >= dz:
        p1_err = 2 * dy - dx
        p2_err = 2 * dz - dx
        y, z = y0, z0
        for i in range(dx + 1):
            x = x0 + i * sx
            zz.append(z); yy.append(y); xx.append(x)
            if p1_err >= 0:
                y += sy
                p1_err -= 2 * dx
            if p2_err >= 0:
                z += sz
                p2_err -= 2 * dx
            p1_err += 2 * dy
            p2_err += 2 * dz

    elif dy >= dx and dy >= dz:
        p1_err = 2 * dx - dy
        p2_err = 2 * dz - dy
        x, z = x0, z0
        for i in range(dy + 1):
            y = y0 + i * sy
            zz.append(z); yy.append(y); xx.append(x)
            if p1_err >= 0:
                x += sx
                p1_err -= 2 * dy
            if p2_err >= 0:
                z += sz
                p2_err -= 2 * dy
            p1_err += 2 * dx
            p2_err += 2 * dz

    else:
        p1_err = 2 * dy - dz
        p2_err = 2 * dx - dz
        y, x = y0, x0
        for i in range(dz + 1):
            z = z0 + i * sz
            zz.append(z); yy.append(y); xx.append(x)
            if p1_err >= 0:
                y += sy
                p1_err -= 2 * dz
            if p2_err >= 0:
                x += sx
                p2_err -= 2 * dz
            p1_err += 2 * dy
            p2_err += 2 * dx

    return np.array(zz), np.array(yy), np.array(xx)


def add_only_closing(mask: np.ndarray, iters: int, struct26: np.ndarray):
    closed = ndimage.binary_dilation(mask, structure=struct26, iterations=int(iters))
    closed = ndimage.binary_erosion(closed, structure=struct26, iterations=int(iters))
    return mask | closed


def connect_label_components_by_bridging(
    vol: np.ndarray,
    label_id: int,
    allow_overwrite_labels: Sequence[int],
    bridge_radius_vox: int,
    max_iters: int,
    verbose: bool,
    name: str,
):
    """
    Add-only bridging: connect smaller components to the main component.
    Writes bridges only where vol is in allow_overwrite_labels or already label_id.
    """
    struct26 = np.ones((3, 3, 3), dtype=bool)
    lab = int(label_id)

    for it in range(max_iters):
        mask = (vol == lab)
        cc, ncc = ndimage.label(mask, structure=struct26)

        if verbose:
            if ncc > 0:
                sizes = ndimage.sum(mask, cc, index=np.arange(1, ncc + 1)).astype(np.int64)
                sizes_sorted = np.sort(sizes)[::-1]
                print(f"[Repair {lab}: {name}] bridge iter {it+1}: nCC={ncc}, top={sizes_sorted[:8].tolist()}")
            else:
                print(f"[Repair {lab}: {name}] bridge iter {it+1}: nCC=0")

        if ncc <= 1:
            break

        sizes = ndimage.sum(mask, cc, index=np.arange(1, ncc + 1)).astype(np.int64)
        order = np.argsort(-sizes)

        main_cid = int(order[0] + 1)
        main_mask = (cc == main_cid)

        dt, inds = ndimage.distance_transform_edt(~main_mask, return_indices=True)
        bridges = np.zeros_like(mask, dtype=bool)

        for k in range(1, ncc):
            cid = int(order[k] + 1)
            other_mask = (cc == cid)
            if not np.any(other_mask):
                continue

            other_coords = np.argwhere(other_mask)
            dvals = dt[other_mask]
            j = int(np.argmin(dvals))
            p_other = tuple(other_coords[j])
            p_main = (int(inds[0][p_other]), int(inds[1][p_other]), int(inds[2][p_other]))

            zz, yy, xx = bresenham3d(p_other, p_main)
            bridges[zz, yy, xx] = True

        if bridge_radius_vox > 0:
            bridges = ndimage.binary_dilation(bridges, structure=struct26, iterations=int(bridge_radius_vox))

        allow = np.isin(vol, np.array(allow_overwrite_labels, dtype=vol.dtype)) | (vol == lab)
        write_mask = bridges & allow

        if verbose:
            print(f"[Repair {lab}: {name}] bridge add voxels: {int(write_mask.sum())}")

        vol[write_mask] = lab

    return vol


def reassign_extra_components(
    vol: np.ndarray,
    label_id: int,
    keep_components: int,
    name: str,
    verbose: bool = True,
):
    """
    Keep largest 'keep_components' components of label_id.
    Reassign all other voxels of label_id to nearest existing non-zero label.
    """
    lab = int(label_id)
    struct26 = np.ones((3, 3, 3), dtype=bool)

    mask = (vol == lab)
    cc, ncc = ndimage.label(mask, structure=struct26)
    if ncc <= keep_components:
        return vol

    sizes = ndimage.sum(mask, cc, index=np.arange(1, ncc + 1)).astype(np.int64)
    order = np.argsort(-sizes)  # largest first
    keep_cids = set(int(order[i] + 1) for i in range(min(keep_components, len(order))))

    extra_mask = (vol == lab) & (~np.isin(cc, np.array(list(keep_cids), dtype=cc.dtype)))

    n_extra = int(extra_mask.sum())
    if n_extra == 0:
        return vol

    if verbose:
        print(f"[Repair {lab}: {name}] reassigning extra components: voxels={n_extra}, keepCC={keep_components}/{ncc}")

    # remove extra voxels temporarily
    vol[extra_mask] = 0

    sources = (vol != 0)
    if not np.any(sources):
        # shouldn't happen, but guard
        vol[extra_mask] = lab
        if verbose:
            print(f"[Repair {lab}: {name}] WARNING: no sources for reassignment, restored label.")
        return vol

    _, inds = ndimage.distance_transform_edt(~sources, return_indices=True)
    nearest_labels = vol[tuple(inds)]
    vol[extra_mask] = nearest_labels[extra_mask]

    return vol


def repair_label_post_qc(
    vol: np.ndarray,
    label_id: int,
    name: str,
    expected_max_components: int,
    allow_overwrite_labels: Sequence[int] = (0, 65, 66),
    max_close_iters: int = 4,
    bridge_radius_vox: int = 1,
    bridge_max_iters: int = 10,
    verbose: bool = True,
):
    """
    Single label repair strategy:
      - If nCC <= expected_max_components: do nothing
      - else: add-only closing (increasing) then bridging
      - if still too many: reassign extra components to nearest labels
    """
    lab = int(label_id)
    struct26 = np.ones((3, 3, 3), dtype=bool)

    mask0 = (vol == lab)
    _, ncc0 = connected_components_26(mask0)
    if ncc0 <= expected_max_components:
        return vol

    if verbose:
        print(f"\n[Repair {lab}: {name}] start nCC={ncc0} (expected <= {expected_max_components})")

    # 1) add-only closing in mask space, then write back add-only (restricted)
    mask = mask0.copy()
    for it in range(1, int(max_close_iters) + 1):
        mask = add_only_closing(mask, iters=it, struct26=struct26)
        _, ncc = ndimage.label(mask, structure=struct26)
        if verbose:
            print(f"[Repair {lab}: {name}] add-only closing it={it}: nCC={ncc}")
        if ncc <= expected_max_components:
            break

    add_vox = mask & (~mask0)
    if np.any(add_vox):
        allow = np.isin(vol, np.array(allow_overwrite_labels, dtype=vol.dtype)) | (vol == lab)
        write_mask = add_vox & allow
        vol[write_mask] = lab
        if verbose:
            print(f"[Repair {lab}: {name}] added voxels from closing: {int(write_mask.sum())}")

    # 2) bridging
    mask1 = (vol == lab)
    _, ncc1 = connected_components_26(mask1)
    if verbose:
        print(f"[Repair {lab}: {name}] after closing nCC={ncc1}")

    if ncc1 > expected_max_components:
        vol = connect_label_components_by_bridging(
            vol,
            label_id=lab,
            allow_overwrite_labels=allow_overwrite_labels,
            bridge_radius_vox=int(bridge_radius_vox),
            max_iters=int(bridge_max_iters),
            verbose=verbose,
            name=name,
        )

    mask2 = (vol == lab)
    _, ncc2 = connected_components_26(mask2)
    if verbose:
        print(f"[Repair {lab}: {name}] after bridging nCC={ncc2}")

    # 3) if still too many: reassign extra components
    if ncc2 > expected_max_components:
        vol = reassign_extra_components(vol, lab, keep_components=expected_max_components, name=name, verbose=verbose)
        mask3 = (vol == lab)
        _, ncc3 = connected_components_26(mask3)
        if verbose:
            print(f"[Repair {lab}: {name}] after reassignment nCC={ncc3}")

    return vol
