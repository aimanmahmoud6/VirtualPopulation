# =============================================================================
# Command Line Interface (CLI)
#
# Summary:
#   Entry point to run the pipeline with YAML configs and optional path
#   overrides for input/output. Run with: python -m segpipe.cli
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

import argparse
from .config import load_config
from .pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="Segmentation pipeline: NIfTI -> QC -> AutoRepair -> VTI")
    ap.add_argument("--config", default="config/params.yaml", help="Path to params.yaml")
    ap.add_argument("--organs", default="config/organs.yaml", help="Path to organs.yaml")
    ap.add_argument("--qc", default="config/qc_limits.yaml", help="Path to qc_limits.yaml")

    # overrides
    ap.add_argument("--input-body", default=None, help="Override input body NIfTI path")
    ap.add_argument("--input-bones", default=None, help="Override input bones NIfTI path")
    ap.add_argument("--output", default=None, help="Override output NIfTI path (processed labelmap)")

    args = ap.parse_args()

    cfg = load_config(args.config, args.organs, args.qc)

    if args.input_body:
        cfg.params["inputs"]["body"] = args.input_body
    if args.input_bones:
        cfg.params["inputs"]["bones"] = args.input_bones
    if args.output:
        cfg.params["outputs"]["nifti"] = args.output

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
