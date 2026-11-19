"""Predict scans WITHOUT labels (e.g. the 91 official BraTS-PEDs validation cases), processed exactly like the
development cases, and write challenge-ready label maps.

    python -m pbts.final.external --in_dir <dir> --out_dir <dir> --runs runs/<model>/<tag> [more run groups ...]

Input layout (BraTS 2023+ naming): <in_dir>/<CaseID>/<CaseID>-t2f.nii.gz, -t1n, -t1c, -t2w
  -> our channel order FLAIR (t2f), T1 (t1n), T1CE (t1c), T2 (t2w).
Each --runs group contributes its 5 fold models; all probability maps are averaged (mean of softmax), argmax,
pasted back into the original 240x240x155 space with the input header: <out_dir>/<CaseID>.nii.gz (uint8).
Preprocessing reuses pbts.data.headmask and the same crop/z-score arithmetic as pbts.data.preprocess; a test
checks it reproduces the development arrays bit-for-bit.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from pbts import config as C
from pbts.data.headmask import head_mask

SUFFIX = {"FLAIR": "t2f", "T1": "t1n", "T1CE": "t1c", "T2": "t2w"}


def load_external_case(case_dir: Path):
    case = case_dir.name
    imgs, ref = [], None
    for ch in C.CHANNELS:
        p = case_dir / f"{case}-{SUFFIX[ch]}.nii.gz"
        im = nib.load(str(p))
        if ref is None:
            ref = im
        if tuple(im.shape) != C.IMAGE_SHAPE or not np.allclose(im.header.get_zooms()[:3], C.SPACING, atol=1e-4):
            raise ValueError(f"{p}: shape {im.shape} / spacing {im.header.get_zooms()} differ from training data")
        if not np.allclose(im.affine, ref.affine, atol=1e-4):
            raise ValueError(f"{p}: affine differs from {SUFFIX['FLAIR']}")
        imgs.append(im.get_fdata(dtype=np.float64).astype(np.float32))   # same cast as pbts.data.convert
    return case, np.stack(imgs), ref


def crop_normalize(imgs: np.ndarray):
    """Identical arithmetic to pbts.data.preprocess.preprocess_case (images only)."""
    mask, info = head_mask(imgs)
    lo, hi = np.array(info["box_lo"]), np.array(info["box_hi"])
    sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))
    crop_img, crop_mask = imgs[(slice(None),) + sl], mask[sl]
    out = np.empty_like(crop_img)
    means, stds, pads = [], [], []
    for c in range(crop_img.shape[0]):
        v = crop_img[c][crop_mask].astype(np.float64)
        mu, sd = float(v.mean()), float(max(v.std(), 1e-8))
        out[c] = ((crop_img[c] - mu) / sd).astype(np.float32)
        means.append(mu); stds.append(sd); pads.append(float((0.0 - mu) / sd))
    rec = {"box_lo": lo.tolist(), "box_hi": hi.tolist(), "original_shape": list(imgs.shape[1:]),
           "norm_mean": means, "norm_std": stds, "pad_value": pads, **{k: info[k] for k in ("skull_stripped", "head_vox")}}
    return out, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run groups: runs/<model>/<tag> (folds 0-4 inside)")
    ap.add_argument("--no_tta", action="store_true")
    a = ap.parse_args()
    import torch
    from pbts.final.netload import load_network
    from pbts.infer.predict import Predictor
    out = Path(a.out_dir); (out / "probabilities").mkdir(parents=True, exist_ok=True)
    cases = sorted(p for p in Path(a.in_dir).iterdir() if p.is_dir())
    prepared = {}
    for cd in cases:
        case, imgs, ref = load_external_case(cd)
        x, rec = crop_normalize(imgs)
        prepared[case] = (x, rec, ref)
    acc = {c: None for c in prepared}
    n_models = 0
    for group in a.runs:
        for k in range(C.N_INNER):
            net, frame, _ = load_network(Path(group) / f"fold_{k}")
            pred = Predictor(net, frame_xyz=frame)
            for case, (x, rec, _) in prepared.items():
                logits = pred.predict_logits(x, rec["pad_value"], tta=not a.no_tta)
                e = np.exp(logits - logits.max(0, keepdims=True)); p = e / e.sum(0, keepdims=True)
                acc[case] = p if acc[case] is None else acc[case] + p
            n_models += 1
            del net, pred
            torch.cuda.empty_cache()
    for case, (x, rec, ref) in prepared.items():
        probs = acc[case] / n_models
        full = np.zeros(rec["original_shape"], np.uint8)
        full[tuple(slice(a_, b_ + 1) for a_, b_ in zip(rec["box_lo"], rec["box_hi"]))] = probs.argmax(0).astype(np.uint8)
        hdr = ref.header.copy(); hdr.set_data_dtype(np.uint8); hdr["scl_slope"], hdr["scl_inter"] = 1.0, 0.0
        nib.save(nib.Nifti1Image(full, ref.affine, header=hdr), str(out / f"{case}.nii.gz"))
        np.savez_compressed(out / "probabilities" / f"{case}.npz", probs=probs.astype(np.float16),
                            box_lo=np.array(rec["box_lo"]), box_hi=np.array(rec["box_hi"]))
        json.dump(rec, open(out / "probabilities" / f"{case}.json", "w"), indent=1)
    json.dump({"runs": a.runs, "n_models_averaged": n_models, "tta": not a.no_tta, "cases": list(prepared)},
              open(out / "external_info.json", "w"), indent=1)
    print(f"{len(prepared)} cases -> {out} ({n_models} models averaged)")


if __name__ == "__main__":
    main()
