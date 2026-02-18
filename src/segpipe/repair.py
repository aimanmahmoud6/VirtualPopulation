# =============================================================================
# Auto-Repair (Add-only Closing + Minimal Bridging)
#
# Summary:
#   Repairs EXTRA COMPONENTS failures by (1) add-only morphological closing
#   (never deletes voxels) and (2) minimal bridging between components using a
#   3D line + optional dilation, restricted to allowed overwrite labels.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

from typing import Sequence
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
            zz.append(z)
            yy.append(y)
            xx.append(x)
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
            zz.append(z)
            yy.append(y)
            xx.append(x)
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
            zz.append(z)
            yy.append(y)
            xx.append(x)
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
    # closing = dilate then erode; but return UNION with original to avoid losing anything
    closed = ndimage.binary_dilation(mask, structure=struct26, iterations=int(iters))
    closed = ndimage.binary_erosion(closed, structure=struct26, iterations=int(iters))
    return mask | closed


def connect_label_components_by_bridging(
    vol: np.ndarray,
    label_id: int,
    allow_overwrite_labels: Sequence[int] = (0, 65, 66),
    bridge_radius_vox: int = 1,
    max_iters: int = 10,
    verbose: bool = True,
):
    """
    Make label_id a single connected component by ADDING minimal bridges only.
    Never deletes label_id.
    Writes bridges only where vol is in allow_overwrite_labels (or already label_id).
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
                print(f"[Bridge {lab}] iter {it+1}: nCC={ncc}, top={sizes_sorted[:8].tolist()}")
            else:
                print(f"[Bridge {lab}] iter {it+1}: nCC=0")

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
            print(f"[Bridge {lab}] adding voxels: {int(write_mask.sum())}")

        vol[write_mask] = lab

    return vol


def repair_extra_components_add_only(
    vol: np.ndarray,
    lid: int,
    organ_name: str,
    max_close_iters: int = 4,
    bridge_after: bool = True,
    allow_overwrite_labels=(0, 65, 66),
    bridge_radius_vox: int = 1,
    bridge_max_iters: int = 10,
    verbose: bool = True,
):
    """
    For NON-OK_MULTI labels with nCC>1:
      1) Try add-only closing with increasing strength
      2) If still multi, try bridging
    """
    struct26 = np.ones((3, 3, 3), dtype=bool)
    lab = int(lid)

    mask0 = (vol == lab)
    _, ncc0 = connected_components_26(mask0)
    if ncc0 <= 1:
        return vol

    if verbose:
        print(f"\n[Repair {lab}] start nCC={ncc0} ({organ_name})")

    # 1) Add-only closing (increasing)
    mask = mask0.copy()
    for it in range(1, int(max_close_iters) + 1):
        mask = add_only_closing(mask, iters=it, struct26=struct26)
        _, ncc = ndimage.label(mask, structure=struct26)
        if verbose:
            print(f"[Repair {lab}] add-only closing it={it}: nCC={ncc}")
        if ncc <= 1:
            break

    # write back (only add)
    add_vox = mask & (~mask0)
    if np.any(add_vox):
        allow = np.isin(vol, np.array(allow_overwrite_labels, dtype=vol.dtype)) | (vol == lab)
        write_mask = add_vox & allow
        vol[write_mask] = lab
        if verbose:
            print(f"[Repair {lab}] added voxels from closing: {int(write_mask.sum())}")

    # check
    _, ncc1 = connected_components_26(vol == lab)
    if verbose:
        print(f"[Repair {lab}] after closing nCC={ncc1}")

    # 2) Bridge if still multi
    if bridge_after and ncc1 > 1:
        vol = connect_label_components_by_bridging(
            vol,
            label_id=lab,
            allow_overwrite_labels=allow_overwrite_labels,
            bridge_radius_vox=int(bridge_radius_vox),
            max_iters=int(bridge_max_iters),
            verbose=verbose,
        )
        _, ncc2 = connected_components_26(vol == lab)
        if verbose:
            print(f"[Repair {lab}] after bridging nCC={ncc2}")

    return vol
