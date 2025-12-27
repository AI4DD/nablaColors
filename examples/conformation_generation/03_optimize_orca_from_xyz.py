#!/usr/bin/env python3
"""
Example: run ORCA geometry optimization from an input XYZ.

This script writes an ORCA input file like:

  ! r2SCAN-3c TightSCF OPT
  # ! CPCM(water)   # uncomment and set solvent if desired
  %maxcore 7000
  %pal
    nprocs 8
  end
  * xyzfile  0 1 input.xyz

Usage:
  python examples/conformation_generation/03_optimize_orca_from_xyz.py \
    --input xtb.xyz \
    --workdir orca_opt \
    --orca-binary orca \
    --threads 8 \
    --charge 0 \
    --mult 1

With solvent:
  python examples/conformation_generation/03_optimize_orca_from_xyz.py \
    --input xtb.xyz --workdir orca_opt_solv --threads 8 --charge 0 --mult 1 \
    --solvent water
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from typing import Optional


def render_orca_input(
    xyz_filename: str,
    charge: int,
    mult: int,
    threads: int,
    maxcore_mb: int,
    solvent: Optional[str],
) -> str:
    header = "! r2SCAN-3c TightSCF OPT"
    solvent_line = f"! CPCM({solvent})" if solvent else "# ! CPCM(water)   # uncomment and set solvent if desired"
    return "\n".join(
        [
            header,
            solvent_line,
            f"%maxcore {int(maxcore_mb)}",
            "%pal",
            f"  nprocs {int(threads)}",
            "end",
            f"* xyzfile  {int(charge)} {int(mult)} {xyz_filename}",
            "",
        ]
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run ORCA optimization from an XYZ file.")
    ap.add_argument("--input", required=True, help="Input XYZ file.")
    ap.add_argument("--workdir", default="orca_opt", help="Work directory (default: orca_opt)")
    ap.add_argument("--orca-binary", default="orca", help="Path to orca binary (default: orca)")
    ap.add_argument("--threads", type=int, default=8, help="ORCA nprocs (default: 8)")
    ap.add_argument("--charge", type=int, default=0, help="Molecular charge (default: 0)")
    ap.add_argument("--mult", type=int, default=1, help="Spin multiplicity (default: 1)")
    # NOTE: argparse uses '%' for interpolation in help strings -> escape as '%%' if needed.
    ap.add_argument("--maxcore-mb", type=int, default=7000, help="ORCA %%maxcore in MB per core (default: 7000)")
    ap.add_argument(
        "--solvent",
        default=None,
        help="If set, enables CPCM(solvent). If omitted, a commented CPCM line is written.",
    )
    ap.add_argument("--inp", default="orca_opt.inp", help="ORCA input filename (default: orca_opt.inp)")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()

    input_xyz = os.path.abspath(args.input)
    if not os.path.isfile(input_xyz):
        raise FileNotFoundError(f"Input XYZ not found: {input_xyz}")

    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)

    xyz_name = os.path.basename(input_xyz)
    local_xyz = os.path.join(workdir, xyz_name)
    shutil.copy2(input_xyz, local_xyz)

    inp_path = os.path.join(workdir, args.inp)
    inp_text = render_orca_input(
        xyz_filename=xyz_name,
        charge=args.charge,
        mult=args.mult,
        threads=args.threads,
        maxcore_mb=args.maxcore_mb,
        solvent=args.solvent,
    )
    with open(inp_path, "w", encoding="utf-8") as f:
        f.write(inp_text)

    orca_log = os.path.join(workdir, "orca.log")
    cmd = [args.orca_binary, os.path.basename(inp_path)]
    print(f"Running in {workdir}: {' '.join(cmd)}")
    with open(orca_log, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, cwd=workdir, stdout=log_f, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"ORCA failed (exit={proc.returncode}). See log: {orca_log}")

    print(f"Done. ORCA log: {orca_log}")
    print(f"ORCA input: {inp_path}")


if __name__ == "__main__":
    main()


