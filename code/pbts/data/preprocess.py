"""Step 02: head crop + per-channel normalisation. Identical for every model.

    python -m pbts.data.preprocess [--workers 14]

Per case:
  head mask (images only) -> crop all 4 channels and the label to the mask's bounding box
  -> z-score each channel with mean/std of HEAD voxels -> nothing else (no blanking, no clipping).
Development cases (234) are written as an nnU-Net raw dataset; the 60 locked cases go to
data/locked_test/images and are never registered with nnU-Net.
Every case gets a record (data/head/records/<case>.json) holding what is needed to map predictions
back exactly: box in original space, normalisation stats, the normalised value of raw 0 (the padding
value), and flags. Scoring always happens back in the original 240x240x155 space.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from pbts import config as C
from pbts.data import io
from pbts.data.headmask import head_mask


def crop_affine(affine: np.ndarray, lo) -> np.ndarray:
    a = affine.copy()
    a[:3, 3] = affine[:3, :3] @ np.asarray(lo, float) + affine[:3, 3]
    return a


def preprocess_case(case: str, locked: bool) -> dict:
    imgs, seg, ref = io.load_raw_case(case)
    mask, info = head_mask(imgs)
    lo, hi = np.array(info["box_lo"]), np.array(info["box_hi"])
    sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))
    crop_img, crop_seg, crop_mask = imgs[(slice(None),) + sl], seg[sl], mask[sl]

    means, stds, bgs = [], [], []
    out = np.empty_like(crop_img)
    for c in range(crop_img.shape[0]):
        v = crop_img[c][crop_mask].astype(np.float64)
        mu, sd = float(v.mean()), float(max(v.std(), 1e-8))
        out[c] = ((crop_img[c] - mu) / sd).astype(np.float32)
        means.append(mu); stds.append(sd); bgs.append(float((0.0 - mu) / sd))

    # write images with the affine shifted to the crop origin, so geometry stays consistent
    import nibabel as nib
    ref_c = nib.Nifti1Image(np.zeros(crop_seg.shape, np.uint8), crop_affine(ref.affine, lo), header=ref.header.copy())
    if locked:
        img_dir, lab_dir = C.LOCKED_DIR / "images", None
    else:
        base = C.NNUNET_RAW / C.DATASET_NAME
        img_dir, lab_dir = base / "imagesTr", base / "labelsTr"
    for c in range(out.shape[0]):
        io.save_like(out[c], ref_c, img_dir / f"{case}_{c:04d}.nii.gz", np.float32)
    if lab_dir is not None:
        io.save_like(crop_seg, ref_c, lab_dir / f"{case}.nii.gz", np.uint8)
    io.save_like(crop_mask.astype(np.uint8), ref_c, C.HEAD_DIR / "masks" / f"{case}.nii.gz", np.uint8)

    ext = (hi - lo + 1).tolist()
    rec = {
        "case": case, "locked": locked,
        "original_shape": list(seg.shape), "original_affine": ref.affine.tolist(),
        "box_lo": lo.tolist(), "box_hi": hi.tolist(), "box_shape": ext,
        "oversized_vs_frame": bool(any(e > f for e, f in zip(ext, C.FRAME_XYZ))),
        "channels": list(C.CHANNELS), "norm_mean": means, "norm_std": stds, "pad_value": bgs,
        "otsu_threshold": info["otsu_threshold"], "skull_stripped": info["skull_stripped"],
        "head_vox": info["head_vox"], "nonzero_vox": info["nonzero_vox"],
        "label_vox": int((seg > 0).sum()), "label_vox_outside_box": int((seg > 0).sum() - (crop_seg > 0).sum()),
        "label_vox_outside_mask": int(((seg[sl] > 0) & ~crop_mask).sum()) + int((seg > 0).sum() - (crop_seg > 0).sum()),
        "fingerprint_normalised": io.fingerprint(out),
    }
    (C.HEAD_DIR / "records").mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(C.HEAD_DIR / "records" / f"{case}.json", "w"), indent=1)
    return rec


def _job(a):
    return preprocess_case(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    locked = set(json.load(open(C.LOCKED_DIR / "cases.json"))["locked"])
    cases = io.list_kaggle_cases()
    with ProcessPoolExecutor(args.workers) as ex:
        recs = list(ex.map(_job, [(c, c in locked) for c in cases]))
    df = pd.DataFrame([{k: (v if not isinstance(v, list) else json.dumps(v)) for k, v in r.items()
                        if k != "original_affine"} for r in recs])
    df.to_csv(C.HEAD_DIR / "records.csv", index=False)
    print(f"{len(df)} cases | locked {int(df.locked.sum())} | skull-stripped {df.skull_stripped.sum()} "
          f"| oversized {df.oversized_vs_frame.sum()} | label voxels outside box: {df.label_vox_outside_box.sum()}")
    (C.HEAD_DIR / "DONE").write_text("preprocessed\n")


if __name__ == "__main__":
    main()
