"""Step 01: the study split = outer fold 0 of the friend's nested 5x5 file.

    python -m pbts.data.splits

  locked test : the 60 scans of outer fold 0. Scored once, at the very end.
  development : the other 234 scans, cross-validated over that fold's 5 inner folds.
Every check below is fatal. Outputs:
  data/locked_test/cases.json   the locked list (read by the guard in pbts.data.guard)
  data/head/study_split.json    dev cases, 5 inner folds, fingerprints
"""
import hashlib
import json
import sys

from pbts import config as C


def patient(case: str) -> str:
    return case.rsplit("-", 1)[0]          # BraTS-PED-00025-103 -> BraTS-PED-00025


def _fp(cases) -> str:
    return hashlib.sha256("\n".join(sorted(cases)).encode()).hexdigest()[:16]


def build():
    nested = json.load(open(C.NESTED_SPLITS))
    outer = nested["nested_5x5"][C.OUTER_FOLD]
    locked = sorted(outer["test"])
    inner = [{"train": sorted(f["train"]), "val": sorted(f["val"])} for f in outer["inner_folds"]]
    from pbts.data import io
    all_cases = set(io.list_kaggle_cases())
    dev = sorted(all_cases - set(locked))

    errs = []
    if len(all_cases) != 294 or len(locked) != 60 or len(dev) != 234:
        errs.append(f"sizes: all={len(all_cases)} locked={len(locked)} dev={len(dev)}")
    if not set(locked) <= all_cases:
        errs.append("locked cases missing from the data")
    if {patient(c) for c in locked} & {patient(c) for c in dev}:
        errs.append("a patient has scans in both locked test and development")
    if len(inner) != C.N_INNER:
        errs.append(f"{len(inner)} inner folds")
    vals = []
    for k, f in enumerate(inner):
        tr, va = set(f["train"]), set(f["val"])
        if tr & va:
            errs.append(f"inner {k}: train/val overlap")
        if tr | va != set(dev):
            errs.append(f"inner {k}: train+val != development set")
        if {patient(c) for c in tr} & {patient(c) for c in va}:
            errs.append(f"inner {k}: patient leakage between train and val")
        vals += f["val"]
    if sorted(vals) != dev:
        errs.append("inner validation folds do not partition the development set exactly once")
    if errs:
        sys.exit("SPLIT CHECK FAILED:\n  " + "\n  ".join(errs))

    C.LOCKED_DIR.mkdir(parents=True, exist_ok=True)
    C.HEAD_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"locked": locked, "fingerprint": _fp(locked)}, open(C.LOCKED_DIR / "cases.json", "w"), indent=1)
    split = {"source": str(C.NESTED_SPLITS), "outer_fold": C.OUTER_FOLD,
             "development": dev, "inner_folds": inner,
             "fingerprints": {"locked": _fp(locked), "development": _fp(dev),
                              **{f"inner{k}_val": _fp(f["val"]) for k, f in enumerate(inner)}}}
    json.dump(split, open(C.HEAD_DIR / "study_split.json", "w"), indent=1)
    sizes = [(len(f["train"]), len(f["val"])) for f in inner]
    print(f"locked {len(locked)} | development {len(dev)} | inner folds (train, val): {sizes}")
    return split


if __name__ == "__main__":
    build()
