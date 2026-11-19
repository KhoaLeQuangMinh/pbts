"""Evaluate a folder of predictions (<case>.nii.gz label maps in the original 240x240x155 space).

    python -m pbts.eval.evaluate --pred_dir <dir> --out_dir <dir> [--cases fold0|<file>] [--workers 12]

Writes to out_dir:
  per_case_region.csv   one row per case x region: official 2026 scores + 2023 lesion-wise + diagnostics
  per_lesion.csv        every GT lesion and every false-positive predicted component
  confusion.csv         summed 5x5 voxel label confusion (rows GT, cols prediction) inside the tumor union
  confusion_per_case.json
  official_panoptica.json, official_scores.csv, official_versions.json
  EVAL_DONE.json        counts and checks; written only if everything above is complete
Raw predictions only: no post-processing is applied anywhere (study decision).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from pbts import config as C
from pbts.data.guard import check_not_locked
from pbts.eval.diagnostics import REGION_NAMES, evaluate_case
from pbts.eval.official import score_official

OFFICIAL_KEYS = {"ET": "et", "NET": "netc", "CC": "cc", "ED": "ed", "TC": "tc", "WT": "wt"}
OFFICIAL_METRICS = ["global_dsc", "global_nsd", "global_hd95", "all_instance_tp", "all_instance_fp",
                    "all_instance_fn", "all_instance_f1", "all_instance_prec", "all_instance_rec",
                    "lesionwise_dsc_mean", "lesionwise_nsd_mean", "lesionwise_hd95_mean"]


def _load_label(path):
    img = nib.load(str(path))
    a = np.asanyarray(img.dataobj)
    if not np.array_equal(a, np.round(a)):
        raise ValueError(f"{path}: non-integer labels")
    a = a.astype(np.uint8)
    if not set(np.unique(a)) <= set(C.LABELS.values()):
        raise ValueError(f"{path}: labels {sorted(set(np.unique(a)))}")
    return a, tuple(float(z) for z in img.header.get_zooms()[:3])


def _one(args):
    case, pred_dir, ref_dir = args
    pred, sp = _load_label(Path(pred_dir) / f"{case}.nii.gz")
    gt, sp_gt = _load_label(Path(ref_dir) / f"{case}.nii.gz")
    if pred.shape != gt.shape or not np.allclose(sp, sp_gt):
        raise ValueError(f"{case}: prediction geometry {pred.shape}/{sp} != GT {gt.shape}/{sp_gt}")
    return evaluate_case(case, pred, gt, sp_gt)


def evaluate(cases, pred_dir, out_dir, ref_dir=None, workers=12, allow_locked=False, purpose=""):
    check_not_locked(cases, allow_locked=allow_locked, purpose=purpose)
    ref_dir = Path(ref_dir or C.RAW_DIR / "labelsTr")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in cases if not (Path(pred_dir) / f"{c}.nii.gz").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} predictions missing, e.g. {missing[:3]}")

    official, _ = score_official(cases, pred_dir, ref_dir, out_dir, workers=max(1, workers // 2))
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_one, [(c, str(pred_dir), str(ref_dir)) for c in cases]))

    region_rows = [r for res in results for r in res[0]]
    lesion_rows = [r for res in results for r in res[1]]
    df = pd.DataFrame(region_rows)
    off = official.set_index("case")
    for m in OFFICIAL_METRICS:
        df[f"off_{m}"] = [off.at[c, f"{m}_{OFFICIAL_KEYS[r]}"] if f"{m}_{OFFICIAL_KEYS[r]}" in off.columns
                          else np.nan for c, r in zip(df.case, df.region)]
    df.to_csv(out_dir / "per_case_region.csv", index=False)
    pd.DataFrame(lesion_rows).to_csv(out_dir / "per_lesion.csv", index=False)
    conf = sum(res[2] for res in results)
    names = list(C.LABELS)
    pd.DataFrame(conf, index=[f"gt_{n}" for n in names], columns=[f"pred_{n}" for n in names]).to_csv(out_dir / "confusion.csv")
    json.dump({c: res[2].tolist() for c, res in zip(cases, results)}, open(out_dir / "confusion_per_case.json", "w"))

    done = {"n_cases": len(cases), "n_rows": len(df), "n_lesion_rows": len(lesion_rows),
            "expected_rows": len(cases) * len(REGION_NAMES),
            "official_cases": int(off.index.isin(cases).sum())}
    assert done["n_rows"] == done["expected_rows"] and done["official_cases"] == len(cases), done
    json.dump(done, open(out_dir / "EVAL_DONE.json", "w"), indent=1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cases", default=None, help="'foldK' for inner fold K's validation cases, or a text file")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.cases and a.cases.startswith("fold"):
        cases = json.load(open(C.HEAD_DIR / "study_split.json"))["inner_folds"][int(a.cases[4:])]["val"]
    elif a.cases:
        cases = [l.strip() for l in open(a.cases) if l.strip()]
    else:
        cases = sorted(p.name.replace(".nii.gz", "") for p in Path(a.pred_dir).glob("*.nii.gz"))
    df = evaluate(cases, a.pred_dir, a.out_dir, workers=a.workers)
    print(df.groupby("region")[["off_global_dsc", "off_all_instance_f1", "lw2023_dice"]].mean().round(4))


if __name__ == "__main__":
    main()
