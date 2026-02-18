# =============================================================================
# QC Metrics and QC Runner
#
# Summary:
#   Computes label QC metrics (sphericity, eccentricity, solidity, components)
#   and produces a CSV report + failure dictionary used by the post-QC repair.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

from typing import Dict
import numpy as np
import pandas as pd
import SimpleITK as sitk


def calc_sphericity(phys_vol, perimeter):
    if perimeter <= 0 or phys_vol <= 0:
        return 0.0
    return float((np.pi ** (1 / 3) * (6.0 * phys_vol) ** (2 / 3)) / perimeter)


def calc_eccentricity(principal_moments):
    if principal_moments is None or len(principal_moments) != 3:
        return 0.0
    m = np.array(principal_moments, dtype=float)
    if m[2] <= 0:
        return 0.0
    m = np.sort(m)
    ratio = np.clip(m[0] / m[2], 0.0, 1.0)
    return float(np.sqrt(1.0 - ratio))


def sitk_component_stats(img, lid):
    # 26-connectivity via filter API (works on older SimpleITK)
    mask = sitk.Equal(img, int(lid))
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    cc = cc_filter.Execute(mask)
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)

    st = sitk.LabelShapeStatisticsImageFilter()
    st.Execute(rel)
    labs = list(st.GetLabels())

    if len(labs) == 0:
        return 0, 0.0, rel

    sizes = np.array([st.GetNumberOfPixels(c) for c in labs], dtype=float)
    ncc = int(len(labs))
    lfrac = float(sizes[0] / sizes.sum()) if sizes.sum() > 0 else 0.0
    return ncc, lfrac, rel


def qc_run_on_image_path(
    file_path: str,
    qc_cfg: Dict,
    organ_map: Dict[int, str],
    save_csv_path: str,
    save_csv: bool = True,
    prefix: str = "[QC]",
):
    """
    Run QC on a labelmap NIfTI at file_path.

    Returns:
      df    : pandas DataFrame with per-label QC metrics and status
      fails : dict[label_id] -> list of issue strings
    """
    img = sitk.ReadImage(file_path)
    spacing = img.GetSpacing()

    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(img)

    ignore = set(int(x) for x in (qc_cfg.get("ignore_labels", []) or []))
    ok_multi = set(int(x) for x in qc_cfg.get("ok_multi", []))
    ok_fragmented = set(int(x) for x in qc_cfg.get("ok_fragmented", []))
    ok_thin = set(int(x) for x in qc_cfg.get("ok_thin", []))

    default_limits = qc_cfg.get("default_limits", {}) or {}
    label_limits = qc_cfg.get("label_limits", {}) or {}

    # optional per-label max components (e.g. discs 68 -> 22)
    max_components = qc_cfg.get("max_components", {}) or {}

    # vertebra special rule
    vertebrae_max_ecc = float(qc_cfg.get("vertebrae_max_ecc", 0.98))

    labels = [int(l) for l in shape.GetLabels() if int(l) != 0 and int(l) not in ignore]

    header = (
        f"{'ID':<5} | {'Organ Name':<22} | {'Spher':>6} | {'Ecc':>6} | "
        f"{'Sol':>6} | {'nCC':>4} | {'LCC%':>6} | {'Multi':>8} | Status"
    )
    print(f"\n{prefix} Running QC on: {file_path}")
    print(prefix + " " + header)
    print(prefix + " " + "-" * len(header))

    qc_rows = []
    fails = {}

    def get_limits(lid: int):
        lim = dict(default_limits)
        lim.update(label_limits.get(str(int(lid)), {}) or {})
        # heuristic: vertebrae are >=200 in your label scheme
        if int(lid) >= 200:
            lim["max_ecc"] = min(float(lim.get("max_ecc", 0.999)), vertebrae_max_ecc)
        return lim

    for lid in sorted(labels):
        name = organ_map.get(lid, f"Unknown({lid})")
        lim = get_limits(lid)

        ncc, lfrac, rel_cc = sitk_component_stats(img, lid)

        # multi flag logic with optional max component override
        max_cc = max_components.get(str(lid), None)
        if max_cc is not None:
            max_cc = int(max_cc)
        elif lid in max_components:
            max_cc = int(max_components[lid])

        if ncc <= 1:
            multi_flag = "-"
        else:
            if max_cc is not None:
                multi_flag = "OK" if ncc <= max_cc else "MULTI"
            else:
                multi_flag = "OK" if lid in ok_multi else "MULTI"

        # Metrics mode:
        # - OK_MULTI + multi: compute on largest CC only
        # - otherwise union label
        if (lid in ok_multi) and (ncc > 1):
            st = sitk.LabelShapeStatisticsImageFilter()
            st.Execute(rel_cc)
            c = 1  # largest after relabel
            phys_vol = float(st.GetPhysicalSize(c))
            perimeter = float(st.GetPerimeter(c))
            sphericity = calc_sphericity(phys_vol, perimeter)
            ecc = calc_eccentricity(st.GetPrincipalMoments(c))
            bbox = st.GetBoundingBox(c)
            bbox_phys_vol = (bbox[3] * spacing[0]) * (bbox[4] * spacing[1]) * (bbox[5] * spacing[2])
            solidity = (phys_vol / bbox_phys_vol) if bbox_phys_vol > 0 else 0.0
            metrics_mode = "largestCC"
        else:
            phys_vol = float(shape.GetPhysicalSize(lid))
            perimeter = float(shape.GetPerimeter(lid))
            sphericity = calc_sphericity(phys_vol, perimeter)
            ecc = calc_eccentricity(shape.GetPrincipalMoments(lid))
            bbox = shape.GetBoundingBox(lid)
            bbox_phys_vol = (bbox[3] * spacing[0]) * (bbox[4] * spacing[1]) * (bbox[5] * spacing[2])
            solidity = (phys_vol / bbox_phys_vol) if bbox_phys_vol > 0 else 0.0
            metrics_mode = "union"

        issues = []

        # EXTRA COMPONENTS:
        # - if lid not in OK_MULTI and ncc>1 -> fail
        # - if lid has explicit max_components and ncc > max -> fail
        if max_cc is not None:
            if ncc > max_cc:
                issues.append(f"EXTRA COMPONENTS DETECTED (>{max_cc})")
        else:
            if (lid not in ok_multi) and (ncc > 1):
                issues.append("EXTRA COMPONENTS DETECTED")

        # VOLUME
        if phys_vol < float(lim.get("min_vol", 0.0)):
            issues.append("VOLUME LOWER LIMIT EXCEEDED")

        # ELONGATION (skip if OK_THIN)
        if (lid not in ok_thin) and (ecc > float(lim.get("max_ecc", 0.999))):
            issues.append("ELONGATION LIMIT EXCEEDED")

        # FRAGMENTATION (skip if OK_FRAGMENTED)
        if (lid not in ok_fragmented) and (sphericity < float(lim.get("min_spher", 0.0))):
            issues.append("FRAGMENTATION DETECTED")

        status = "PASS" if not issues else f"FAIL({'; '.join(issues)})"

        line = (
            f"{lid:<5} | {name:<22} | {sphericity:6.3f} | {ecc:6.3f} | "
            f"{solidity:6.3f} | {ncc:4d} | {100 * lfrac:6.1f} | {multi_flag:>8} | {status}"
        )
        print(prefix + " " + line)

        qc_rows.append(
            {
                "Label_ID": lid,
                "Organ_Name": name,
                "MetricsMode": metrics_mode,
                "PhysicalVolume_mm3": round(float(phys_vol), 3),
                "Sphericity": round(float(sphericity), 4),
                "Eccentricity": round(float(ecc), 4),
                "Solidity": round(float(solidity), 4),
                "NumComponents": int(ncc),
                "LCC_percent": round(100.0 * float(lfrac), 2),
                "MultiFlag": multi_flag,
                "Status": status,
            }
        )

        if issues:
            fails[lid] = issues

    print(prefix + " " + "-" * len(header))

    df = pd.DataFrame(qc_rows).sort_values(["Label_ID"]).reset_index(drop=True)
    if save_csv:
        df.to_csv(save_csv_path, index=False)
        print(f"{prefix} QC Report saved to: {save_csv_path}")

    return df, fails
