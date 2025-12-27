#!/usr/bin/env python3
"""
Example: CSV (chromophore SMILES + solvent SMILES) -> RDKit conformer -> LMDB for UniProp inference.

The UniProp `pcq` task expects each LMDB entry (gzip-compressed pickle) to contain:
  - atoms: array/list of atom symbols (WITHOUT hydrogens)
  - input_pos: list of conformer coordinate arrays, shape (n_atoms, 3)
  - label_pos: target coordinates array, shape (n_atoms, 3)  (required by task wrapper)
  - smi: chromophore SMILES string
  - solvent_smi: solvent SMILES string
  - node_attr: int32 array, shape (n_atoms, n_atom_features)
  - edge_index: int32 array, shape (2, n_edges)
  - edge_attr: int32 array, shape (n_edges, n_bond_features)
  - target: float (required by current `unimol_plus/inference.py`; use dummy 0.0 for screening)

This script mirrors feature encoding from `unimol_plus/scripts/get_3d_lmdb.py`.

Usage:
  python examples/conformation_generation/04_csv_to_lmdb_rdkit.py \
    --csv screening.csv \
    --out-dir /path/to/lmdb_dir \
    --split test \
    --smiles-col smiles \
    --solvent-col solvent_smi
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import lmdb
import numpy as np


def encode_int_key(x: int, nbytes: int = 8) -> bytes:
    if x < 0:
        raise ValueError("Negative key not allowed")
    return int(x).to_bytes(nbytes, byteorder="big", signed=False)


# --- Graph feature encoding (copied to match unimol_plus/scripts/get_3d_lmdb.py) ---

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


def safe_index(l: Sequence, e) -> int:
    """Return index of e in list l; if missing, return last index."""
    try:
        return list(l).index(e)
    except Exception:
        return len(l) - 1


def atom_to_feature_vector(atom) -> List[int]:
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(allowable_features["possible_hybridization_list"], str(atom.GetHybridization())),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond) -> List[int]:
    bond_feature = [
        safe_index(allowable_features["possible_bond_type_list"], str(bond.GetBondType())),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def get_graph(mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (node_attr, edge_index, edge_attr) using the encoding above."""
    atom_features_list: List[List[int]] = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.asarray(atom_features_list, dtype=np.int32)

    num_bond_features = 3
    if mol.GetNumBonds() > 0:
        edges_list: List[Tuple[int, int]] = []
        edge_features_list: List[List[int]] = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_index = np.asarray(edges_list, dtype=np.int32).T
        edge_attr = np.asarray(edge_features_list, dtype=np.int32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int32)

    return x, edge_index, edge_attr


def guess_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def rdkit_conformers_heavy_only(
    smiles: str,
    num_confs: int,
    seed: int,
    threads: int,
    max_embed_iterations: int,
    max_uff_iterations: int,
) -> Tuple[List[str], List[np.ndarray]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdDistGeom

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles!r}")

    # Heavy-atom-only mol for graph + atoms
    mol_heavy = Chem.RemoveHs(mol)
    atoms = [a.GetSymbol() for a in mol_heavy.GetAtoms()]

    mol_h = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.maxIterations = int(max_embed_iterations)
    params.numThreads = int(threads)
    params.trackFailures = True
    params.randomSeed = int(seed)

    conf_ids = list(
        rdDistGeom.EmbedMultipleConfs(
            mol_h,
            numConfs=int(num_confs),
            params=params,
        )
    )
    if not conf_ids:
        raise RuntimeError("RDKit embedding failed (no conformers generated).")

    # Optimize all conformers (UFF)
    _results = AllChem.UFFOptimizeMoleculeConfs(mol_h, maxIters=int(max_uff_iterations), numThreads=int(threads))

    # Drop hydrogens, keeping conformers
    mol_h_noh = Chem.RemoveHs(mol_h)
    heavy_n = mol_heavy.GetNumAtoms()

    coords_list: List[np.ndarray] = []
    for cid in conf_ids:
        pos = mol_h_noh.GetConformer(int(cid)).GetPositions()
        if pos.shape[0] != heavy_n:
            raise RuntimeError(f"Unexpected atom count after H removal: got {pos.shape[0]} expected {heavy_n}")
        coords_list.append(np.asarray(pos, dtype=np.float32))

    return atoms, coords_list


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build an LMDB (pcq task format) from a CSV of SMILES+solvent.")
    ap.add_argument("--csv", required=True, help="Input CSV file.")
    ap.add_argument("--out-dir", required=True, help="Output directory. Will write <split>.lmdb inside.")
    ap.add_argument("--split", default="test", help="Output split name (default: test)")
    ap.add_argument("--map-size-gb", type=float, default=5.0, help="LMDB map size in GB (default: 5.0)")

    ap.add_argument("--smiles-col", default=None, help="CSV column name for chromophore SMILES.")
    ap.add_argument("--solvent-col", default=None, help="CSV column name for solvent SMILES.")
    ap.add_argument("--target-col", default=None, help="Optional CSV column name for target value.")
    ap.add_argument("--id-col", default=None, help="Optional CSV column name for integer id (LMDB key).")
    ap.add_argument("--start-id", type=int, default=0, help="Starting id if --id-col not provided (default: 0)")

    ap.add_argument("--num-confs", type=int, default=1, help="Number of RDKit conformers per molecule (default: 1)")
    ap.add_argument("--seed", type=int, default=0, help="RDKit random seed (default: 0)")
    ap.add_argument("--threads", type=int, default=6, help="RDKit threads (default: 6)")
    ap.add_argument("--max-embed-iterations", type=int, default=100, help="RDKit embedding iterations (default: 100)")
    ap.add_argument("--max-uff-iterations", type=int, default=10000, help="UFF optimization iterations (default: 10000)")

    ap.add_argument("--default-target", type=float, default=0.0, help="Default/dummy target if target not provided (default: 0.0)")
    ap.add_argument("--skip-invalid", action="store_true", help="Skip rows that fail parsing/embedding.")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_lmdb = os.path.join(out_dir, f"{args.split}.lmdb")

    map_size = int(float(args.map_size_gb) * (1024**3))
    env = lmdb.open(
        out_lmdb,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=map_size,
    )

    n_ok = 0
    n_bad = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header / fieldnames.")

        smiles_col = args.smiles_col or guess_column(reader.fieldnames, ["smiles", "chromophore_smiles", "solute_smiles", "smi"])
        solvent_col = args.solvent_col or guess_column(reader.fieldnames, ["solvent_smi", "solvent_smiles", "solvent"])
        target_col = args.target_col or guess_column(reader.fieldnames, ["target", "y"])
        id_col = args.id_col or guess_column(reader.fieldnames, ["id", "idx", "index"])

        if smiles_col is None:
            raise ValueError(f"Could not infer SMILES column. Provide --smiles-col. Fieldnames={reader.fieldnames}")
        if solvent_col is None:
            raise ValueError(f"Could not infer solvent column. Provide --solvent-col. Fieldnames={reader.fieldnames}")

        with env.begin(write=True) as txn:
            for row_i, row in enumerate(reader):
                try:
                    smi = (row.get(smiles_col) or "").strip()
                    solvent_smi = (row.get(solvent_col) or "").strip()
                    if not smi:
                        raise ValueError("Empty SMILES")

                    if id_col is not None and (row.get(id_col) is not None) and str(row.get(id_col)).strip() != "":
                        rec_id = int(str(row[id_col]).strip())
                    else:
                        rec_id = int(args.start_id) + int(row_i)

                    if target_col is not None and (row.get(target_col) is not None) and str(row.get(target_col)).strip() != "":
                        target = float(str(row[target_col]).strip())
                    else:
                        target = float(args.default_target)

                    atoms, coords_list = rdkit_conformers_heavy_only(
                        smiles=smi,
                        num_confs=args.num_confs,
                        seed=args.seed + row_i,
                        threads=args.threads,
                        max_embed_iterations=args.max_embed_iterations,
                        max_uff_iterations=args.max_uff_iterations,
                    )

                    # Graph features computed on heavy-atom RDKit mol (no explicit Hs)
                    from rdkit import Chem

                    mol_heavy = Chem.MolFromSmiles(smi)
                    if mol_heavy is None:
                        raise ValueError(f"Failed to parse SMILES for graph: {smi!r}")
                    mol_heavy = Chem.RemoveHs(mol_heavy)
                    node_attr, edge_index, edge_attr = get_graph(mol_heavy)

                    if node_attr.shape[0] != len(atoms):
                        raise RuntimeError(f"node_attr atoms mismatch: node_attr={node_attr.shape[0]} atoms={len(atoms)}")
                    for c in coords_list:
                        if c.shape[0] != len(atoms) or c.shape[1] != 3:
                            raise RuntimeError(f"Bad coordinate shape: {c.shape} expected ({len(atoms)}, 3)")

                    # Required by pcq task wrappers:
                    input_pos = coords_list
                    label_pos = coords_list[-1]

                    record: Dict = {
                        "atoms": np.asarray(atoms),
                        "input_pos": input_pos,
                        "label_pos": label_pos,
                        "smi": smi,
                        "solvent_smi": solvent_smi,
                        "node_attr": node_attr.astype(np.int32, copy=False),
                        "edge_index": edge_index.astype(np.int32, copy=False),
                        "edge_attr": edge_attr.astype(np.int32, copy=False),
                        "target": target,
                    }

                    key = encode_int_key(rec_id)
                    value = gzip.compress(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                    ok = txn.put(key, value, overwrite=True)
                    if not ok:
                        raise RuntimeError(f"Failed to write key={rec_id}")
                    n_ok += 1
                except Exception as e:
                    n_bad += 1
                    if not args.skip_invalid:
                        raise RuntimeError(f"Row {row_i} failed: {e}") from e

    env.sync()
    env.close()

    print(f"Wrote LMDB: {out_lmdb}")
    print(f"Records OK: {n_ok}, failed: {n_bad}")


if __name__ == "__main__":
    main()


