#!/usr/bin/env python3
"""
Example: run xTB optimization from an input XYZ.

Equivalent to:
  {xtb_binary} {input_struct_filename} {' '.join(run_flags)} > xtb.log

Default flags correspond to:
  xtb_flags: ["--gfn","2","--opt"]

Usage:
  python examples/conformation_generation/02_optimize_xtb_from_xyz.py \
    --input rdkit.xyz \
    --workdir xtb_opt \
    --xtb-binary xtb \
    --charge 0 \
    --uhf 0 \
    --flags --gfn 2 --opt

Example: +1 charge with 2 unpaired electrons:
  python examples/conformation_generation/02_optimize_xtb_from_xyz.py \
    --input rdkit.xyz \
    --workdir xtb_opt_c1_uhf2 \
    --charge 1 \
    --uhf 2 \
    --flags --gfn 2 --opt
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from typing import Optional


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run xTB optimization starting from an XYZ file.")
    ap.add_argument("--input", required=True, help="Input XYZ file.")
    ap.add_argument("--workdir", default="xtb_opt", help="Work directory (default: xtb_opt)")
    ap.add_argument("--xtb-binary", default="xtb", help="Path to xtb binary (default: xtb)")
    ap.add_argument("--charge", type=int, default=0, help="Molecular charge, passed as xtb --chrg (default: 0)")
    ap.add_argument(
        "--uhf",
        type=int,
        default=0,
        help="Number of unpaired electrons, passed as xtb --uhf (default: 0)",
    )
    ap.add_argument(
        "--flags",
        nargs=argparse.REMAINDER,
        default=["--gfn", "2", "--opt"],
        help="Flags passed to xtb (default: --gfn 2 --opt). Everything after --flags is forwarded.",
    )
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()

    input_xyz = os.path.abspath(args.input)
    if not os.path.isfile(input_xyz):
        raise FileNotFoundError(f"Input XYZ not found: {input_xyz}")

    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)

    input_name = os.path.basename(input_xyz)
    local_input_xyz = os.path.join(workdir, input_name)
    shutil.copy2(input_xyz, local_input_xyz)

    xtb_log = os.path.join(workdir, "xtb.log")
    cmd = [
        args.xtb_binary,
        input_name,
        "--chrg",
        str(args.charge),
        "--uhf",
        str(args.uhf),
        *args.flags,
    ]

    print(f"Running in {workdir}: {' '.join(cmd)}")
    with open(xtb_log, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, cwd=workdir, stdout=log_f, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"xTB failed (exit={proc.returncode}). See log: {xtb_log}")

    # xTB commonly writes optimized geometry to xtbopt.xyz in the working directory.
    opt_xyz = os.path.join(workdir, "xtbopt.xyz")
    if os.path.isfile(opt_xyz):
        print(f"Optimized XYZ: {opt_xyz}")
    else:
        print(f"Done. xTB finished, but {opt_xyz} was not found. See log: {xtb_log}")


if __name__ == "__main__":
    main()


