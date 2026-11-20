"""Content fingerprint of every prepared study data file, to prove that two machines hold IDENTICAL data. Read-only.

Hashes what the code reads, not the bytes on disk (compression can differ between machines without the data
differing): decoded arrays (+ dtype, shape, affine) for .nii.gz / .b2nd, parsed objects for .json / .pkl.
The machine's project root inside JSON strings is replaced by <ROOT>. Status markers (DONE) and logs are skipped.

    python tools/data_fingerprint.py <out.json> [--workers 8]
    python tools/data_fingerprint.py --compare gpu-a240.json blackwell.json
"""
import argparse
import hashlib
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbts import config as C  # noqa: E402

SKIP_NAMES = {"DONE", "ACCESS_LOG.txt"}
TREES = ["data/raw", "data/head", "data/locked_test", f"nnunet/raw/{C.DATASET_NAME}", f"nnunet/preprocessed/{C.DATASET_NAME}"]


def _h(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:24]


def _canon(o):
    """JSON-able, order-independent form; numpy arrays become (dtype, shape, hash)."""
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    if isinstance(o, np.ndarray):
        return ["ndarray", str(o.dtype), list(o.shape), _h(np.ascontiguousarray(o).tobytes())]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, str):
        return o.replace(str(C.ROOT), "<ROOT>")
    return o


def fingerprint(path: str):
    p = Path(path)
    name = p.name
    if name.endswith(".nii.gz") or name.endswith(".nii"):
        import nibabel as nib
        img = nib.load(str(p))
        a = np.asanyarray(img.dataobj)
        return _h(str(a.dtype), a.shape, np.ascontiguousarray(a).tobytes(), np.round(img.affine, 6).tobytes())
    if name.endswith(".b2nd"):
        import blosc2
        a = blosc2.open(str(p), mode="r")[:]
        return _h(str(a.dtype), a.shape, np.ascontiguousarray(a).tobytes())
    if name.endswith(".pkl"):
        return _h(json.dumps(_canon(pickle.load(open(p, "rb"))), sort_keys=True))
    if name.endswith(".json"):
        return _h(json.dumps(_canon(json.load(open(p))), sort_keys=True))
    return _h(p.read_bytes().replace(str(C.ROOT).encode(), b"<ROOT>"))


def build(out, workers):
    files = sorted(str(f.relative_to(C.ROOT)) for t in TREES for f in (C.ROOT / t).rglob("*")
                   if f.is_file() and f.name not in SKIP_NAMES)
    with ProcessPoolExecutor(workers) as ex:
        hashes = list(ex.map(fingerprint, [str(C.ROOT / f) for f in files], chunksize=8))
    json.dump({"root": str(C.ROOT), "n_files": len(files), "files": dict(zip(files, hashes))}, open(out, "w"), indent=0)
    print(f"{len(files)} files fingerprinted -> {out}")


def compare(a, b):
    A, B = json.load(open(a))["files"], json.load(open(b))["files"]
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    diff = sorted(k for k in set(A) & set(B) if A[k] != B[k])
    print(f"{len(A)} vs {len(B)} files | identical: {len(set(A) & set(B)) - len(diff)} | differ: {len(diff)} | "
          f"only in {Path(a).name}: {len(only_a)} | only in {Path(b).name}: {len(only_b)}")
    for title, lst in (("DIFFER", diff), ("ONLY IN " + a, only_a), ("ONLY IN " + b, only_b)):
        for k in lst[:30]:
            print(f"  {title}: {k}")
    return not (diff or only_a or only_b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    build(a.out, a.workers)
