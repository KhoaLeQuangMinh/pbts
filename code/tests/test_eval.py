"""Evaluator tests. Run on the server (takes a few minutes):
    ../env/bin/python -m pytest -q -p no:cacheprovider tests/test_eval.py
Predictions are synthetic perturbations of real development-set GT, written to analysis/eval_test/."""
import json
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from pbts import config as C
from pbts.data import io
from pbts.eval.diagnostics import REGION_NAMES, region_mask
from pbts.eval.evaluate import evaluate
from pbts.eval.lesionwise2023 import region_scores
from pbts.eval.official import score_official

OUT = C.ROOT / "analysis/eval_test"
SPLIT = json.load(open(C.HEAD_DIR / "study_split.json"))
# development cases only (the evaluator refuses locked cases); include CC/ED-rich and small-lesion cases
REC = pd.read_csv(C.MANIFEST).set_index("subject_id")
DEV = [c for c in SPLIT["inner_folds"][0]["val"]]
RICH = [c for c in DEV if REC.at[c, "vox_CC"] > 0 and REC.at[c, "vox_ED"] > 0][:3]
CASES = sorted(set(RICH + DEV[:3]))
VARIANTS = ["perfect", "empty", "erode", "shift", "drop_small_add_specks", "cc_as_net"]


def perturb(gt, variant):
    if variant == "perfect":
        return gt.copy()
    if variant == "empty":
        return np.zeros_like(gt)
    if variant == "erode":
        p = gt.copy()
        for l in (1, 2, 3, 4):
            m = gt == l
            p[m & ~ndi.binary_erosion(m)] = 0
        return p
    if variant == "shift":
        return np.roll(gt, 2, axis=0)
    if variant == "drop_small_add_specks":
        p = gt.copy()
        for l in (1, 2, 3, 4):
            lab, n = ndi.label(gt == l, structure=np.ones((3, 3, 3)))
            sizes = np.bincount(lab.ravel())
            for k in range(1, n + 1):
                if sizes[k] < 100:
                    p[lab == k] = 0
        for (x, y, z), l in [((70, 120, 60), 4), ((170, 120, 60), 3), ((120, 70, 110), 4)]:
            if not gt[x - 2:x + 2, y - 2:y + 2, z - 2:z + 2].any():
                p[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = l
        return p
    if variant == "cc_as_net":
        p = gt.copy(); p[gt == 3] = 2
        return p
    raise ValueError(variant)


@pytest.fixture(scope="module")
def preds():
    ref = nib.load(str(io.raw_label_path(CASES[0])))
    for v in VARIANTS:
        d = OUT / "preds" / v
        d.mkdir(parents=True, exist_ok=True)
        for c in CASES:
            gt = np.asanyarray(nib.load(str(io.raw_label_path(c))).dataobj).astype(np.uint8)
            io.save_like(perturb(gt, v), ref, d / f"{c}.nii.gz", np.uint8)
    return OUT / "preds"


def test_lesionwise_matches_original_2023_code(preds, tmp_path):
    jobs, ours = [], {}
    for v in VARIANTS:
        for c in CASES:
            gt = np.asanyarray(nib.load(str(io.raw_label_path(c))).dataobj)
            pr = np.asanyarray(nib.load(str(preds / v / f"{c}.nii.gz")).dataobj)
            for r in REGION_NAMES:
                g, p = region_mask(gt, r), region_mask(pr, r)
                key = f"{v}|{c}|{r}"
                gp, pp = tmp_path / f"{key.replace('|', '_')}_gt.nii.gz", tmp_path / f"{key.replace('|', '_')}_pr.nii.gz"
                ref = nib.load(str(io.raw_label_path(c)))
                io.save_like(g.astype(np.uint8) * 3, ref, gp, np.uint8)
                io.save_like(p.astype(np.uint8) * 3, ref, pp, np.uint8)
                jobs.append({"pred": str(pp), "gt": str(gp), "key": key})
                ours[key] = region_scores(p, g)[0]
    jf, of = tmp_path / "jobs.json", tmp_path / "ref.json"
    json.dump(jobs, open(jf, "w"))
    subprocess.check_call([str(C.ROOT / "env_ref2023/bin/python"), "-I", str(C.ROOT / "code/tests/ref2023_runner.py"),
                           str(jf), str(of)], stdout=subprocess.DEVNULL)
    bad, ref_errors = [], []
    for r in json.load(open(of)):
        if "error" in r:
            ref_errors.append(r)
            continue
        o = ours[r["key"]]
        for k in ["lw2023_tp", "lw2023_fn", "lw2023_fp"]:
            if o[k] != r[k]:
                bad.append((r["key"], k, o[k], r[k]))
        for k in ["lw2023_dice", "lw2023_hd95", "legacy_dice", "legacy_hd95"]:
            if not (np.isclose(o[k], r[k], atol=1e-6) or (np.isinf(o[k]) and np.isinf(r[k]))):
                bad.append((r["key"], k, o[k], r[k]))
    assert not ref_errors, f"original code failed on {len(ref_errors)} jobs: {[(e['key'], e['error']) for e in ref_errors[:4]]}\n{ref_errors[0]['trace']}"
    assert not bad, f"{len(bad)} mismatches vs original code, e.g. {bad[:5]}"


def test_official_parallel_equals_single(preds, tmp_path):
    a, _ = score_official(CASES, preds / "drop_small_add_specks", out_dir=tmp_path / "w1", workers=1)
    b, _ = score_official(CASES, preds / "drop_small_add_specks", out_dir=tmp_path / "w4", workers=4)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_full_evaluation_sanity(preds):
    per = {}
    for v in ["perfect", "empty", "drop_small_add_specks"]:
        per[v] = evaluate(CASES, preds / v, OUT / "eval" / v, workers=8)
        assert json.load(open(OUT / "eval" / v / "EVAL_DONE.json"))["n_cases"] == len(CASES)

    p = per["perfect"]
    present = p[p.gt_present]
    assert np.allclose(present.off_global_dsc, 1) and np.allclose(present.lw2023_dice, 1)
    assert p[~p.gt_present].off_global_dsc.isna().all()          # official: absent in both -> NaN
    assert np.allclose(p.lw2023_dice, 1)                           # 2023: absent in both -> 1
    assert set(p.presence) <= {"TP", "TN"}
    conf = pd.read_csv(OUT / "eval" / "perfect" / "confusion.csv", index_col=0).values
    assert (conf - np.diag(np.diag(conf))).sum() == 0

    e = per["empty"]
    assert np.allclose(e[e.gt_present].off_global_dsc, 0) and set(e.presence) <= {"FN", "TN"}

    s = per["drop_small_add_specks"]
    les = pd.read_csv(OUT / "eval" / "drop_small_add_specks" / "per_lesion.csv")
    n_fp = les[les.kind == "pred_fp"].groupby(["case", "region"]).size()
    lw_fp = s.set_index(["case", "region"]).lw2023_fp
    assert (lw_fp.reindex(n_fp.index) == n_fp).all() and lw_fp.sum() == n_fp.sum()


def test_locked_cases_refused(tmp_path):
    locked = json.load(open(C.LOCKED_DIR / "cases.json"))["locked"][:1]
    with pytest.raises(PermissionError):
        evaluate(locked, tmp_path, tmp_path / "out")
