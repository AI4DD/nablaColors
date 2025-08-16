# nablaColors
This repository accompanies the article “nablaColors: A 3D Benchmark for Optical Property Prediction with Solvent-aware Graph Neural Networks.”

## Dataset and LMDB reading

- **Download**: the dataset archive is available on Zenodo: [Zenodo record](https://zenodo.org/records/16886724).
  - Download `absorption_conformations.zip` and extract it.

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


