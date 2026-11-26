"""Verify a retained measured source or its declared staging adaptation."""
from pathlib import Path
import hashlib
def verify_benchmark_source(path, expected):
    path=Path(path)
    digest=lambda b: hashlib.sha256(b).hexdigest()
    current=path.read_bytes()
    if digest(current)==expected:
        return True
    root=path.parents[2]
    original=(root/'UPSTREAM_SOURCE_SNAPSHOTS'/'sensitivity_controls'/'real'/'benchmark_filtered.py').read_bytes()
    adapted=original
    adapted=adapted.replace(b'psutil.Process().cpu_affinity([4])', b"if __name__ == '__main__':\n    psutil.Process().cpu_affinity([4])")
    return digest(original)==expected and current==adapted
