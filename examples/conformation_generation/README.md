## Conformation generation examples

This folder contains minimal, runnable examples for generating and optimizing conformations:

- `01_smiles_to_rdkit_xyz.py`: SMILES → RDKit 3D conformation → `rdkit.xyz`
- `02_optimize_xtb_from_xyz.py`: `input.xyz` → xTB optimization (writes `xtb.log`, typically `xtbopt.xyz`)
- `03_optimize_orca_from_xyz.py`: `input.xyz` → ORCA optimization (writes `orca_opt.inp`, `orca.log`)

### Quickstart

RDKit conformer:

```bash
python examples/conformation_generation/01_smiles_to_rdkit_xyz.py \
  --smiles "CCO" \
  --out rdkit.xyz
```

xTB optimization:

```bash
python examples/conformation_generation/02_optimize_xtb_from_xyz.py \
  --input rdkit.xyz \
  --workdir xtb_opt \
  --xtb-binary xtb \
  --flags --gfn 2 --opt
```

ORCA optimization (vacuum):

```bash
python examples/conformation_generation/03_optimize_orca_from_xyz.py \
  --input xtb_opt/xtbopt.xyz \
  --workdir orca_opt \
  --orca-binary orca \
  --threads 8 \
  --charge 0 \
  --mult 1
```

ORCA optimization (with solvent):

```bash
python examples/conformation_generation/03_optimize_orca_from_xyz.py \
  --input xtb_opt/xtbopt.xyz \
  --workdir orca_opt_solv \
  --threads 8 \
  --charge 0 \
  --mult 1 \
  --solvent water
```


