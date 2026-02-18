# =============================================================================
# Main Pipeline Orchestrator
#
# Summary:
#   Runs the full pipeline: resample, merge, crop, overlay, targeted repairs,
#   skin generation, refine + polish, save NIfTI, QC + auto-repair loop,
#   then export VTI + mapping report.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

import time
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk

from .utils import Logger, make_organ_map
from .io import resample_image, crop_nonzero, save_nifti_from_array, export_vti_from_nifti
from .overlay import overlay_organs
from .refine import fine_refine_remove_orphans_and_reassign, remove_label_dust_inplace
from .qc import qc_run_on_image_path
from .repair import repair_extra_components_add_only


def run_pipeline(cfg):
    p = cfg.params
    organs = cfg.organs
    qc_cfg = cfg.qc

    body_path = p["inputs"]["body"]
    bones_path = p["inputs"]["bones"]

    out_nifti = p["outputs"]["nifti"]
    out_vti = p["outputs"]["vti"]
    log_path = p["outputs"]["log"]
    qc_csv = p["outputs"]["qc_csv"]

    spacing = tuple(float(x) for x in p["target_spacing_mm"])

    skin_cfg = p["skin"]
    overlay_cfg = p["overlay"]
    refine_cfg = p["refine"]
    ar_cfg = p["autorepair"]

    skin_label = int(skin_cfg["label"])

    organ_map = make_organ_map(organs)
    organ_map[skin_label] = "Skin"

    with Logger(log_path):
        start_time = time.time()
        print(f"STARTING: {time.ctime()}")

        # 1) Load & Align
        img_body = resample_image(sitk.ReadImage(body_path), spacing)
        img_bones = resample_image(sitk.ReadImage(bones_path), spacing)
        img_bones = sitk.Resample(img_bones, img_body, sitk.Transform(), sitk.sitkNearestNeighbor, 0)

        data_body = sitk.GetArrayFromImage(img_body)  # z,y,x
        data_bones = sitk.GetArrayFromImage(img_bones)

        # 2) Merge detailed bones into body
        print("Merging specific bone labels...")
        data_body[data_bones > 0] = data_bones[data_bones > 0]

        # 3) Crop around non-zero region
        raw_crop = crop_nonzero(data_body, pad=3)

        # 4) Dilation -> Smoothing -> Overlay loop
        print("Processing organs and bones (dilate everywhere -> smooth)...")
        final_vol = overlay_organs(raw_crop, organs, spacing, overlay_cfg)

        # 5) Targeted repairs (your existing ones)
        print("Repairing Spinal Canal and Discs...")

        # Spinal Canal (71)
        canal_mask = (final_vol == 71)
        cord_mask = (final_vol == 52)

        if np.any(canal_mask):
            combined_starting_mask = canal_mask | cord_mask
            for z in range(combined_starting_mask.shape[0]):
                combined_starting_mask[z] = ndimage.binary_fill_holes(combined_starting_mask[z])

            refined = ndimage.binary_dilation(combined_starting_mask, iterations=8)
            for z in range(refined.shape[0]):
                refined[z] = ndimage.binary_fill_holes(refined[z])

            canal_final = ndimage.binary_erosion(refined, iterations=7)
            final_vol[canal_final & (final_vol != 52)] = 71

        # Discs (68)
        disc_mask = (final_vol == 68)
        if np.any(disc_mask):
            refined_d = ndimage.binary_closing(disc_mask, structure=np.ones((3, 3, 3)), iterations=2)
            final_vol[refined_d] = 68

        # 6) Skin generation
        if bool(skin_cfg.get("generate", True)):
            thickness = float(skin_cfg["thickness_mm"])
            print(f"Generating {thickness:.1f}mm skin...")
            dist = ndimage.distance_transform_edt(final_vol == 0, sampling=spacing)
            final_vol[(dist > 0) & (dist <= thickness) & (final_vol == 0)] = skin_label

        # 7) Fine refining
        final_vol = fine_refine_remove_orphans_and_reassign(
            final_vol,
            organ_map=organ_map,
            dust_max_vox=int(refine_cfg["dust_max_vox"]),
            small_frac=float(refine_cfg["small_frac"]),
            multipart_min_cc=int(refine_cfg["multipart_min_cc"]),
            multipart_ratio=float(refine_cfg["multipart_ratio"]),
        )

        # 8) Final polish dust
        final_vol = remove_label_dust_inplace(final_vol, dust_max_vox=int(refine_cfg["final_polish_dust_vox"]))

        # 9) Save NIfTI (pre-QC)
        save_nifti_from_array(final_vol.astype(np.uint16), spacing, out_nifti)
        print(f"NIfTI Saved: {out_nifti}")

        # 10) QC + AUTO-REPAIR LOOP
        _, fails = qc_run_on_image_path(out_nifti, qc_cfg, organ_map, save_csv_path=qc_csv, save_csv=True)

        ok_multi = set(int(x) for x in qc_cfg.get("ok_multi", []))
        to_fix = [
            lid
            for lid, issues in fails.items()
            if ("EXTRA COMPONENTS DETECTED" in issues) and (int(lid) not in ok_multi)
        ]

        if len(to_fix) == 0:
            print("\n[AutoRepair] No non-OK_MULTI extra-component failures. Skipping repairs.")
        else:
            print("\n[AutoRepair] Will attempt to fix EXTRA COMPONENTS for labels:", to_fix)

            img_q = sitk.ReadImage(out_nifti)
            vol = sitk.GetArrayFromImage(img_q).astype(np.uint16)

            allow_overwrite = tuple(int(x) for x in ar_cfg["allow_overwrite_labels"])

            for lid in to_fix:
                lid = int(lid)
                name = organ_map.get(lid, f"Label_{lid}")

                if lid == 64:
                    max_close = int(ar_cfg["max_close_iters_cartilage"])
                    bridge_radius = int(ar_cfg["bridge_radius_cartilage"])
                else:
                    max_close = int(ar_cfg["max_close_iters_default"])
                    bridge_radius = int(ar_cfg["bridge_radius_default"])

                print(f"\n[AutoRepair] Fixing {lid} ({name}) ...")
                vol = repair_extra_components_add_only(
                    vol,
                    lid=lid,
                    organ_name=name,
                    max_close_iters=max_close,
                    bridge_after=True,
                    allow_overwrite_labels=allow_overwrite,
                    bridge_radius_vox=bridge_radius,
                    bridge_max_iters=int(ar_cfg["bridge_max_iters"]),
                    verbose=True,
                )

            save_nifti_from_array(vol.astype(np.uint16), spacing, out_nifti)
            print(f"\n[AutoRepair] Repaired NIfTI Saved: {out_nifti}")

            _, fails2 = qc_run_on_image_path(out_nifti, qc_cfg, organ_map, save_csv_path=qc_csv, save_csv=True)
            print("\n[AutoRepair] Remaining FAIL labels:", sorted(list(fails2.keys())))

        # 11) VTI Conversion & mapping report
        print("\nConverting to VTI...")
        mapping = export_vti_from_nifti(out_nifti, spacing, out_vti)

        print(f"\n{'OLD_ID':<8} | {'NEW_ID':<8} | {'NAME'}")
        print("-" * 40)
        for old_id in sorted(mapping.keys()):
            print(f"{old_id:<8} | {mapping[old_id]:<8} | {organ_map.get(old_id, 'Unknown')}")

        print(f"\nCOMPLETED in {time.time() - start_time:.2f}s")
