"""Step 00: Kaggle download -> clean float32 .nii.gz copy, verified case by case against the manifest.

    python -m pbts.data.convert [--workers 12]

Fatal checks (any failure stops the step and nothing is marked done):
  shape 240x240x155, identical affine across the 4 channels and the label, spacing 1 mm, LPS,
  finite intensities, label values within {0..4}, and per-label voxel counts equal to the manifest.
Recorded but not fatal: source dtype vs manifest, float64->float32 rounding error.
Outputs: data/raw/{imagesTr,labelsTr}, data/raw/conversion_report.csv, data/raw/fingerprints.json
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd

from pbts import config as C
from pbts.data import io

LABEL_COLS = {"ET": [1], "NET": [2], "CC": [3], "ED": [4], "TC": [1, 2, 3], "WT": [1, 2, 3, 4]}


def convert_case(case: str, row: dict) -> dict:
    fatal, notes = [], []
    imgs = [io.load_nifti(io.kaggle_image_path(case, c)) for c in range(len(C.CHANNELS))]
    seg_img = io.load_nifti(io.kaggle_label_path(case))
    ref = imgs[0]

    for name, im in list(zip(C.CHANNELS, imgs)) + [("label", seg_img)]:
        if tuple(im.shape) != C.IMAGE_SHAPE:
            fatal.append(f"{name} shape {im.shape}")
        if not np.allclose(im.affine, ref.affine, atol=1e-4):
            fatal.append(f"{name} affine differs from FLAIR")
    if not np.allclose(ref.header.get_zooms()[:3], C.SPACING, atol=1e-4):
        fatal.append(f"spacing {ref.header.get_zooms()[:3]}")
    orient = "".join(nib.aff2axcodes(ref.affine))
    if orient != row["orientation"]:
        fatal.append(f"orientation {orient} != manifest {row['orientation']}")

    out = {"case": case, "orientation": orient}
    fps = {}
    for c, (name, im) in enumerate(zip(C.CHANNELS, imgs)):
        src_dtype = str(im.get_data_dtype())
        if src_dtype != row[f"{name}_dtype"]:
            notes.append(f"{name} dtype {src_dtype} != manifest {row[f'{name}_dtype']}")
        a = im.get_fdata(dtype=np.float64)
        if not np.isfinite(a).all():
            fatal.append(f"{name} has non-finite values")
        a32 = a.astype(np.float32)
        out[f"{name}_src_dtype"] = src_dtype
        out[f"{name}_max_rel_err"] = float(np.abs(a32.astype(np.float64) - a).max() / max(np.abs(a).max(), 1e-12))
        out[f"{name}_min"], out[f"{name}_max"] = float(a32.min()), float(a32.max())
        io.save_like(a32, ref, io.raw_image_path(case, c), np.float32)
        fps[name] = io.fingerprint(a32)

    s = np.asanyarray(seg_img.dataobj)
    if not np.array_equal(s, np.round(s)):
        fatal.append("label has non-integer values")
    s = s.astype(np.int64)
    vals = set(np.unique(s).tolist())
    if not vals <= set(C.LABELS.values()):
        fatal.append(f"unexpected label values {sorted(vals - set(C.LABELS.values()))}")
    for region, labs in LABEL_COLS.items():
        n = int(np.isin(s, labs).sum())
        out[f"vox_{region}"] = n
        if n != int(row[f"vox_{region}"]):
            fatal.append(f"vox_{region} {n} != manifest {row[f'vox_{region}']}")
    s8 = s.astype(np.uint8)
    io.save_like(s8, ref, io.raw_label_path(case), np.uint8)
    fps["label"] = io.fingerprint(s8)

    out["fatal"] = "; ".join(fatal)
    out["notes"] = "; ".join(notes)
    out["fingerprints"] = fps
    return out


def _job(args):
    case, row = args
    try:
        return convert_case(case, row)
    except Exception as e:  # report every failure instead of dying on the first
        return {"case": case, "fatal": f"exception: {e!r}", "notes": "", "fingerprints": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    manifest = pd.read_csv(C.MANIFEST).set_index("subject_id")
    cases = io.list_kaggle_cases()
    missing = sorted(set(manifest.index) - set(cases))
    extra = sorted(set(cases) - set(manifest.index))
    if missing or extra or len(cases) != 294:
        sys.exit(f"case list mismatch: {len(cases)} found, missing={missing}, extra={extra}")

    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(_job, [(c, manifest.loc[c].to_dict()) for c in cases]))

    fps = {r["case"]: r.pop("fingerprints") for r in results}
    rep = pd.DataFrame(results)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    rep.to_csv(C.RAW_DIR / "conversion_report.csv", index=False)
    with open(C.RAW_DIR / "fingerprints.json", "w") as f:
        json.dump(fps, f, indent=1, sort_keys=True)

    bad = rep[rep["fatal"].astype(bool)]
    print(f"converted {len(rep)} cases; fatal problems in {len(bad)}; notes in {int(rep['notes'].astype(bool).sum())}")
    print("max float32 rounding error:", float(rep[[f"{n}_max_rel_err" for n in C.CHANNELS]].max().max()))
    if len(bad):
        print(bad[["case", "fatal"]].to_string(index=False))
        sys.exit(1)
    (C.RAW_DIR / "DONE").write_text("conversion verified against manifest\n")


if __name__ == "__main__":
    main()
