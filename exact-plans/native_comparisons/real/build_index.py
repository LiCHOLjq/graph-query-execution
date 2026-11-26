"""Rebuild the optional FB15k-237 HNSW index without timing queries."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

import hnswlib
import numpy as np


def main():
    directory = Path(__file__).resolve().parent
    output = directory / "hnsw_index.bin"
    if output.exists():
        raise SystemExit(f"Index already exists: {output.name}")
    with np.load(directory / "fb15k237_text.npz") as prepared:
        vectors = prepared["vectors"].astype(np.float32)
    size, dimension = vectors.shape
    index = hnswlib.Index(space="l2", dim=dimension)
    index.init_index(max_elements=size, ef_construction=200, M=16, random_seed=20261023)
    index.set_num_threads(1)
    index.add_items(vectors, np.arange(size), num_threads=1)
    # A relative ASCII filename also works in non-ASCII Windows directories.
    os.chdir(directory)
    with tempfile.TemporaryDirectory(dir=".") as tmp:
        path = Path(tmp) / "index.bin"
        index.save_index(str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.replace(output)
    frozen = json.loads((directory / "hnsw_summary.json").read_text())["hashes"][output.name]
    print("Built HNSW index", output.name, digest)
    print("Matches original serialized index:", digest == frozen)


if __name__ == "__main__":
    main()
