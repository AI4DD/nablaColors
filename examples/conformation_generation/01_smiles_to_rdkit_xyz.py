#!/usr/bin/env python3
"""
Example: SMILES -> RDKit 3D conformation -> XYZ file.

Requires:
  - RDKit (e.g. `conda install -c conda-forge rdkit`)

Usage:
  python examples/conformation_generation/01_smiles_to_rdkit_xyz.py \
    --smiles "CCO" \
    --out rdkit.xyz
"""

from __future__ import annotations

import argparse
import os
from typing import Optional


def build_xyz_from_smiles(
    smiles: str,
    out_xyz: str,
    seed: int,
    threads: int,
    max_embed_iterations: int,
    max_uff_iterations: int,
) -> None:
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom
    from rdkit.Chem import rdForceFieldHelpers

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.maxIterations = int(max_embed_iterations)
    params.numThreads = int(threads)
    params.trackFailures = True
    params.randomSeed = int(seed)

    conf_id = rdDistGeom.EmbedMolecule(mol, params=params)
    if conf_id < 0:
        raise RuntimeError("RDKit embedding failed (EmbedMolecule returned -1).")

    # Optimize 3D geometry (UFF). Returns list of (status, energy) for each conformer.
    # status == 0 means OK.
    results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol, maxIters=int(max_uff_iterations), numThreads=int(threads))
    if not results:
        raise RuntimeError("UFFOptimizeMoleculeConfs returned empty results.")
    status, _energy = results[0]
    if status != 0:
        raise RuntimeError(f"UFF optimization did not converge (status={status}). Try increasing --max-uff-iterations.")

    os.makedirs(os.path.dirname(os.path.abspath(out_xyz)) or ".", exist_ok=True)
    Chem.MolToXYZFile(mol, out_xyz, confId=int(conf_id))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate an RDKit 3D conformation from SMILES and write XYZ.")
    ap.add_argument("--smiles", required=True, help="Input SMILES string.")
    ap.add_argument("--out", default="rdkit.xyz", help="Output XYZ filename (default: rdkit.xyz)")
    ap.add_argument("--seed", type=int, default=0, help="RDKit random seed (default: 0)")
    ap.add_argument("--threads", type=int, default=6, help="Number of threads for RDKit (default: 6)")
    ap.add_argument("--max-embed-iterations", type=int, default=100, help="RDKit embedding iterations (default: 100)")
    ap.add_argument("--max-uff-iterations", type=int, default=10000, help="UFF optimization iterations (default: 10000)")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    build_xyz_from_smiles(
        smiles=args.smiles,
        out_xyz=args.out,
        seed=args.seed,
        threads=args.threads,
        max_embed_iterations=args.max_embed_iterations,
        max_uff_iterations=args.max_uff_iterations,
    )
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()


