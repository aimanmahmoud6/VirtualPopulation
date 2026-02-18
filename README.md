<!--
=============================================================================
Segmentation Pipeline (Modular Repo)

Summary:
  Refactored pipeline from a single script into modules + YAML configs.
  Produces a processed labelmap NIfTI, runs QC, applies add-only repairs,
  and exports a VTI with 1..N remapped labels.

Created by Aiman Mahmoud, King's College London, February 2026.
Email: aiman.2.mahmoud@kcl.ac.uk
=============================================================================
-->

# Segmentation Pipeline (NIfTI -> QC -> Auto-repair -> VTI)

## Install
```bash
pip install -r requirements.txt

## What it does
1) Load body + bones NIfTI  
2) Resample to target spacing  
3) Merge bones into body  
4) Crop around non-zero ROI  
5) Dilation + smoothing + overlay organ loop  
6) Targeted repairs (spinal canal + discs)  
7) Skin generation  
8) Fine refine (orphans reassignment)  
9) Final dust polish  
10) QC + auto-repair loop (add-only closing + bridging; cartilage uses bridging)  
11) Save NIfTI + export VTI + mapping report

## Install
```bash
pip install -r requirements.txt
