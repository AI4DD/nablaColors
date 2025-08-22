# nablaColors
This repository accompanies the article “nablaColors: A 3D Benchmark for Optical Property Prediction with Solvent-aware Graph Neural Networks.”

## Installation

Follow these steps to set up the environment and install dependencies.

1) Install Uni-Core and required packages

```bash
# From the repo root
bash install_unicore.sh
```

2) Install Uni-Mol+ components in editable mode

```bash
cd unimol_plus
pip install -e .
```

Notes:
- The script `install_unicore.sh` should install Uni-Core and its dependencies (PyTorch, CUDA toolkit compatibility, and Unicore Python packages). If you prefer manual install, install the local `Uni-Core` first (e.g., `pip install -e Uni-Core/`) and ensure `unicore-train` is available in your environment.
- Recommended Python: 3.10.

## Dataset and LMDB reading

- **Download**: the dataset archive is available on Zenodo: [Zenodo record](https://zenodo.org/records/16886724).

### Dataset contents

This dataset provides curated molecular conformations and predefined splits for benchmarking machine learning models in optical absorption prediction. The core file, `absorption_conformations.zip`, contains an LMDB database of molecular geometries optimized at multiple levels of theory (xTB, DFT in vacuum, and DFT with implicit solvent). The accompanying CSV files (`absorption_pairs_all.csv`, `absorption_train.csv`, `absorption_val.csv`, `absorption_test.csv`) define the train/validation/test splits for supervised absorption prediction. To support robust evaluation, scaffold-based cross-validation splits are also provided for both single-property absorption (`absorption_crossval.zip`) and multitarget learning (`multitarget_crossval.zip`). The files `smiles_to_replace.csv` and `smiles_to_remove.csv` document corrections and exclusions applied during curation to ensure dataset quality. Together, these resources enable reproducible training and evaluation of 2D and 3D models for molecular optical property prediction.

Examples of how to read from the LMDB databases are available at: https://github.com/AI4DD/nablaColors

This dataset compiles experimental data on absorption and emission maxima, as well as photoluminescence quantum yield, from the following sources: Joung et al. (2020), Ju et al. (2021), Venkatraman et al. (2018), and Venkatraman & Chellappan (2020).

### Example: reading records from LMDB (Python)

The values are gzip-compressed, pickled dictionaries; keys are byte strings from which an integer identifier can be derived.

```python
import os
import lmdb
import gzip
import pickle

def read_first_record(db_path: str):
    assert os.path.isfile(db_path), f"LMDB not found: {db_path}"

    env = lmdb.open(
        db_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        if not cursor.first():
            return None
        key, value = cursor.item()

        # Values are gzip-compressed and pickled
        value = gzip.decompress(value)
        record = pickle.loads(value)

        return record

if __name__ == "__main__":
    # Example path after extracting the archive
    db_path = "/path/to/absorption_conformations.lmdb"  # replace with your actual path
    sample = read_first_record(db_path)
    if sample is None:
        print("LMDB is empty")
    else:
        # Inspect available fields in the record
        print("keys:", list(sample.keys()))
```


## Pretrained checkpoints

- Uni-Mol+ small checkpoint is taken from the official repository (see `unimol_plus`): https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_plus
- Chemprop model for solvent embedding is expected at: `models/chemprop/fold_0/model_1/model.pt`


## Reproducing experiments (training scripts)

We provide shell scripts under `unimol_plus/` to run the experiments. All scripts accept `--data-path` and `--pretrained-model` CLI flags and have sensible defaults.

### Multitarget: head pretrain

```bash
bash unimol_plus/head_pretrain_uniprop_multitarget.sh \
  --data-path /path/to/dataset \
  --pretrained-model /path/to/unimol_plus_pcq_small.pt
```

Notes:
- Uses the Uni-Mol+ small checkpoint from https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_plus as initialization.
- Trains prediction head first while freezing the backbone.

### Multitarget: finetune with unfrozen backbone

```bash
bash unimol_plus/finetune_unfreeze_backbone_multitarget.sh \
  --data-path /path/to/dataset \
  --pretrained-model /path/to/head_pretrained_checkpoint.pt
```

Notes:
- Continues from the head-pretrained checkpoint and unfreezes the backbone.

### Single-target (absorption): head pretrain

```bash
bash unimol_plus/head_pretrain_uniprop_singletarget.sh \
  --data-path /path/to/dataset \
  --pretrained-model /path/to/unimol_plus_pcq_small.pt
```

### Single-target (absorption): finetune with unfrozen backbone

```bash
bash unimol_plus/finetune_unfreeze_backbone_singletarget.sh \
  --data-path /path/to/dataset \
  --pretrained-model /path/to/head_pretrained_checkpoint.pt
```


## Validation and metrics

After training, run validation to generate metrics and predictions. Metrics are saved as JSON alongside per-rank intermediate outputs and a merged pickle.

### Multitarget validation

```bash
bash unimol_plus/validate_multitarget.sh \
  --data-path /path/to/dataset \
  --weight-path /path/to/checkpoint.pt \
  --subset valid \
  --results-path /path/to/save/eval
```

### Single-target (absorption) validation

```bash
bash unimol_plus/validate_singletarget_absorption.sh \
  --data-path /path/to/dataset \
  --weight-path /path/to/checkpoint.pt \
  --subset valid \
  --results-path /path/to/save/eval
```

Outputs (per subset):
- `${results_path}/${subset}.metrics.json` — aggregated metrics
- `${results_path}/${subset}.preds.pkl` — merged predictions and labels

