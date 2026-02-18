# =============================================================================
# Main Pipeline Orchestrator
#
# Summary:
#   Runs full pipeline using pure Python configuration from segpipe.settings:
#   overlay -> skin -> refine -> polish -> save -> QC -> (single post-QC repair)
#   -> QC re-run -> VTI export + reordering table.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

import time
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk

from .utils import Logger
from .io import resample_image, crop_nonzero, save_nifti_from_array, export_vti_from_nifti
from .overlay import overlay_organs
from .refine import fine_refine_remove_orphans_and_reassign, remove_label_dust_inplace
from .qc import qc_run_on_image_path
from .repair import repair_label_post_qc

from . import settings as S


def print_step(step: int, msg: str):
    print(f"[step {step}] {msg}")


def run_pipeline(run_settings: S.Settings):
    body_path = run_settings.input_body
    bones_path = run_settings.input_bones

    out_nifti = run_settings.out_nifti
    out_vti = run_settings.out_vti
    log_path = run_settings.out_log
    qc_csv = run_settings.out_qc_csv

    spacing = tuple(float(x) for x in S.TARGET_SPACING_MM)

    with Logger(log_path):
        start_time = time.time()
        print(f"STARTING: {time.ctime()}")

        step = 1
        print_step(step, "Loading + resampling NIfTIs to target spacing...")
        step += 1

        # 1) Load & Align
        img_body = resample_image(sitk.ReadImage(body_path), spacing)
        img_bones = resample_image(sitk.ReadImage(bones_path), spacing)
        img_bones = sitk.Resample(img_bones, img_body, sitk.Transform(), sitk.sitkNearestNeighbor, 0)

        data_body = sitk.GetArrayFromImage(img_body)  # z,y,x
        data_bones = sitk.GetArrayFromImage(img_bones)

        print_step(step, "Merging detailed bone labels into body segmentation...")
        step += 1
        data_body[data_bones > 0] = data_bones[data_bones > 0]

        print_step(step, "Cropping around non-zero region...")
        step += 1
        raw_crop = crop_nonzero(data_body, pad=3)

        print_step(step, "Overlay: dilation -> smoothing -> winner-takes-all merge...")
        step += 1
        overlay_cfg = dict(
            default_dilation_mm=S.DEFAULT_DILATION_MM,
            dilation_mm_overrides=S.DILATION_MM_OVERRIDES,
            sigma_alpha=S.SIGMA_ALPHA,
            median_beta=S.MEDIAN_BETA,
        )
        final_vol = overlay_organs(raw_crop, _organs_as_dicts(), spacing, overlay_cfg)

        # Skin generation + add-only watertight close
        if bool(S.GENERATE_SKIN):
            print_step(step, f"Generating + sealing skin ({S.SKIN_THICK_MM:.1f}mm, label {S.SKIN_LABEL})...")
            step += 1

            dist = ndimage.distance_transform_edt(final_vol == 0, sampling=spacing)
            skin0 = (dist > 0) & (dist <= float(S.SKIN_THICK_MM)) & (final_vol == 0)
            final_vol[skin0] = int(S.SKIN_LABEL)

            skin_mask = (final_vol == int(S.SKIN_LABEL))
            skin_closed = ndimage.binary_closing(
                skin_mask,
                structure=np.ones((3, 3, 3), dtype=bool),
                iterations=2
            )
            for z in range(skin_closed.shape[0]):
                skin_closed[z] = ndimage.binary_fill_holes(skin_closed[z])

            add_skin = skin_closed & (~skin_mask) & (final_vol == 0)
            final_vol[add_skin] = int(S.SKIN_LABEL)
            print(f"[Skin] Added voxels from closing: {int(add_skin.sum())}")

        # Fine refining
        print_step(step, "Refine: remove tiny components + reassign orphans (table below)...")
        step += 1
        final_vol = fine_refine_remove_orphans_and_reassign(
            final_vol,
            organ_map=S.ORGAN_MAP,
            dust_max_vox=int(S.DUST_MAX_VOX),
            small_frac=float(S.SMALL_FRAC),
            multipart_min_cc=int(S.MULTIPART_MIN_CC),
            multipart_ratio=float(S.MULTIPART_RATIO),
        )

        # Final polish dust
        print_step(step, f"Polish: reassign dust voxels (<= {S.FINAL_POLISH_DUST_VOX})...")
        step += 1
        final_vol = remove_label_dust_inplace(final_vol, dust_max_vox=int(S.FINAL_POLISH_DUST_VOX))

        # Save NIfTI (pre-QC)
        print_step(step, f"Saving NIfTI: {out_nifti}")
        step += 1
        save_nifti_from_array(final_vol.astype(np.uint16), spacing, out_nifti)

        # QC iteration 1
        qc_cfg = _qc_cfg_dict()
        print(f"[QC] Starting QC (iteration 1)...")
        _, fails = qc_run_on_image_path(out_nifti, qc_cfg, S.ORGAN_MAP, save_csv_path=qc_csv, save_csv=True, prefix="[QC]")

        _print_qc_fail_summary(fails)

        # Single post-QC repair step:
        # - For labels NOT in OK_MULTI: expected max components = 1
        # - For labels with explicit MAX_COMPONENTS (e.g. discs 68): expected max that value
        to_fix = _labels_triggering_component_repair(fails)

        if len(to_fix) == 0:
            print("[Repair] No component-count repairs triggered.")
        else:
            print(f"[Repair] Running post-QC component repairs for labels: {to_fix}")

            img_q = sitk.ReadImage(out_nifti)
            vol = sitk.GetArrayFromImage(img_q).astype(np.uint16)

            for lid in to_fix:
                lid = int(lid)
                name = S.ORGAN_MAP.get(lid, f"Label_{lid}")

                expected_max = int(S.MAX_COMPONENTS.get(lid, 1))
                if lid in S.MAX_COMPONENTS:
                    expected_max = int(S.MAX_COMPONENTS[lid])

                # your cartilage gets slightly stronger connect settings
                if lid == 64:
                    max_close = int(S.MAX_CLOSE_ITERS_CARTILAGE)
                    bridge_radius = int(S.BRIDGE_RADIUS_CARTILAGE)
                else:
                    max_close = int(S.MAX_CLOSE_ITERS_DEFAULT)
                    bridge_radius = int(S.BRIDGE_RADIUS_DEFAULT)

                vol = repair_label_post_qc(
                    vol,
                    label_id=lid,
                    name=name,
                    expected_max_components=expected_max,
                    allow_overwrite_labels=S.ALLOW_OVERWRITE_LABELS,
                    max_close_iters=max_close,
                    bridge_radius_vox=bridge_radius,
                    bridge_max_iters=int(S.BRIDGE_MAX_ITERS),
                    verbose=True,
                )

            # Save repaired volume
            print_step(step, "Saving repaired NIfTI (post-QC repairs applied)...")
            step += 1
            save_nifti_from_array(vol.astype(np.uint16), spacing, out_nifti)

            # QC iteration 2
            print(f"[QC] Starting QC (iteration 2)...")
            _, fails2 = qc_run_on_image_path(out_nifti, qc_cfg, S.ORGAN_MAP, save_csv_path=qc_csv, save_csv=True, prefix="[QC]")
            _print_qc_fail_summary(fails2)

        # Export VTI + mapping table
        print_step(step, "Converting to VTI...")
        step += 1
        mapping = export_vti_from_nifti(out_nifti, spacing, out_vti)

        print_step(step, "Reordering labels to 1..N (mapping table):")
        step += 1
        print(f"{'OLD_ID':<8} | {'NEW_ID':<8} | {'NAME'}")
        print("-" * 40)
        for old_id in sorted(mapping.keys()):
            print(f"{old_id:<8} | {mapping[old_id]:<8} | {S.ORGAN_MAP.get(old_id, 'Unknown')}")

        print(f"\nCOMPLETED in {time.time() - start_time:.2f}s")


def _organs_as_dicts():
    return [{"id": lid, "name": name, "method": method, "param": param} for (lid, name, method, param) in S.ORGAN_LIST]


def _qc_cfg_dict():
    # qc.py expects label_limits to be string keys
    return {
        "ignore_labels": list(S.QC_IGNORE_LABELS),
        "ok_multi": list(S.OK_MULTI),
        "ok_fragmented": list(S.OK_FRAGMENTED),
        "ok_thin": list(S.OK_THIN),
        "default_limits": dict(S.DEFAULT_LIMITS),
        "label_limits": {str(int(k)): dict(v) for k, v in S.LABEL_LIMITS.items()},
        "vertebrae_max_ecc": 0.98,
        "max_components": {str(int(k)): int(v) for k, v in (S.MAX_COMPONENTS or {}).items()},
    }


def _print_qc_fail_summary(fails: dict):
    if not fails:
        print("[QC] All labels PASS.")
        return
    print("[QC] FAIL summary:")
    for lid in sorted(fails.keys()):
        name = S.ORGAN_MAP.get(int(lid), f"Label_{lid}")
        print(f"  - {lid}: {name} -> {', '.join(fails[lid])}")


def _labels_triggering_component_repair(fails: dict):
    """
    Decide which labels should undergo the post-QC component repair step.
    Trigger rules:
      - Any label with EXTRA COMPONENTS detected (including discs > 22)
    """
    out = []
    for lid, issues in fails.items():
        if any("EXTRA COMPONENTS" in s for s in issues):
            out.append(int(lid))
    return sorted(set(out))
