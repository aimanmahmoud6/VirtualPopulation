# =============================================================================
# I/O Utilities (NIfTI/VTI)
#
# Summary:
#   Resampling, cropping, saving NIfTI, and exporting VTI with 1..N remapping.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

from typing import Tuple, Dict
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import pyvista as pv

def resample_image(itk_image: sitk.Image, new_spacing: Tuple[float, float, float]) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(new_spacing))
    new_size = [
        int(round(itk_image.GetSize()[i] * (itk_image.GetSpacing()[i] / new_spacing[i])))
        for i in range(3)
    ]
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(itk_image)

def crop_nonzero(data: np.ndarray, pad: int = 3) -> np.ndarray:
    z_idx, y_idx, x_idx = np.where(data > 0)
    s_z = slice(max(0, int(np.min(z_idx)) - pad), min(data.shape[0], int(np.max(z_idx)) + pad + 1))
    s_y = slice(max(0, int(np.min(y_idx)) - pad), min(data.shape[1], int(np.max(y_idx)) + pad + 1))
    s_x = slice(max(0, int(np.min(x_idx)) - pad), min(data.shape[2], int(np.max(x_idx)) + pad + 1))
    return data[s_z, s_y, s_x]

def relabel_to_1_to_N(vol: np.ndarray):
    vol_i = np.rint(vol).astype(np.int32)
    labels = np.unique(vol_i)
    labels = labels[labels != 0]
    new_vol = np.zeros_like(vol_i, dtype=np.int32)
    mapping: Dict[int, int] = {}
    for new_id, old_id in enumerate(labels, start=1):
        new_vol[vol_i == old_id] = new_id
        mapping[int(old_id)] = int(new_id)
    return new_vol, mapping

def save_nifti_from_array(vol_zyx: np.ndarray, spacing_xyz, out_path: str):
    img = sitk.GetImageFromArray(vol_zyx)
    img.SetSpacing(tuple(spacing_xyz))
    sitk.WriteImage(img, out_path)

def export_vti_from_nifti(nifti_path: str, spacing_xyz, vti_path: str):
    nii = nib.load(nifti_path)
    v_data, mapping = relabel_to_1_to_N(nii.get_fdata())

    grid = pv.ImageData(
        dimensions=v_data.shape,
        spacing=tuple(spacing_xyz),
        origin=nii.affine[:3, 3]
    )
    grid.point_data["labels"] = v_data.ravel(order="F")
    grid.save(vti_path)

    return mapping
