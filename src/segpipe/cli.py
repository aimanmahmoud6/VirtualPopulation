# =============================================================================
# Command Line Interface (Pure Python Config)
#
# Summary:
#   Runs the pipeline using settings from segpipe.settings. Supports
#   optional overrides for input/output paths.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

import argparse
from .settings import DEFAULT_SETTINGS
from .pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="Segmentation pipeline (pure Python settings)")
    ap.add_argument("--input-body", default=None, help="Override input body NIfTI path")
    ap.add_argument("--input-bones", default=None, help="Override input bones NIfTI path")
    ap.add_argument("--output-nifti", default=None, help="Override output NIfTI path")
    ap.add_argument("--output-vti", default=None, help="Override output VTI path")
    ap.add_argument("--output-log", default=None, help="Override log file path")
    ap.add_argument("--output-qc", default=None, help="Override QC CSV path")
    args = ap.parse_args()

    s = DEFAULT_SETTINGS
    if args.input_body: s.input_body = args.input_body
    if args.input_bones: s.input_bones = args.input_bones
    if args.output_nifti: s.out_nifti = args.output_nifti
    if args.output_vti: s.out_vti = args.output_vti
    if args.output_log: s.out_log = args.output_log
    if args.output_qc: s.out_qc_csv = args.output_qc

    run_pipeline(s)


if __name__ == "__main__":
    main()
